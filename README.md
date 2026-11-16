# 搜索排序 · 有限候选集离线策略评估系统

在**本地合成日志**上比较排序策略的离线价值估计。系统刻意把每一种估计方法的
**适用前提、覆盖不足、权重不稳定与方差**摆在台面上，并在每个结论旁标注：
**离线估计 ≠ 上线收益**。

仅依赖 Python 3.11 标准库（sqlite3 / http.server / 原生 JS + SVG），无需安装。

## 1. 它解决什么问题

搜索评估页需要把**候选结果、真实曝光、点击、后续成交（含迟到退款）**关联起来。
两个根本约束：

1. **候选集是有限的，且曝光有偏**：某些商品在日志中从未曝光。未曝光只说明
   日志策略没给机会，**不能把"没点击"当成差评**——把未曝光商品奖励记 0 是
   本系统明确保留并展示其偏差的错误对照（`naive_zero`）。
2. **离线方法各有前提**：前提不成立时，数字可能"自信但错误"。合成数据带
   已知真实曝光概率与 ground truth，可以直接核对每个方法何时可信、何时失效。

## 2. 方法与适用前提

| 方法 | 估计量（每个查询） | 适用前提 | 主要风险 |
|---|---|---|---|
| `ips` 倾向加权（Horvitz–Thompson） | Σ rᵢ/pᵢ | 正性（目标商品 p>0）、记录倾向=真实倾向、无混杂 | 小倾向导致权重爆炸、高方差 |
| `cips` 截断 IPS | Σ rᵢ·min(1/pᵢ, clip) | 同 IPS，截断阈值**预先固定** | 引入偏差换方差 |
| `snips` 自归一化 IPS | k·Σwᵢrᵢ/Σwᵢ | 同 IPS | 比率估计，倾向系统偏置时仍有偏 |
| `replay_all` top-k 集合匹配重放 | 匹配查询的真实奖励均值 | 匹配指示与奖励独立（独立 Bernoulli 日志满足） | 覆盖率=倾向连乘，候选稍大即趋近 0 |
| `replay_any` 主动作匹配重放 | top-1 曝光查询的奖励均值 | 同上 | 只衡量主动作，不能代表整页 |
| `naive_zero` 错误对照 | 未曝光记 0 | **无——故意错误的口径** | 系统性向下偏差，禁止用于决策 |

方差由查询级解析方差与 **bootstrap（查询重采样，SNIPS 用 cluster bootstrap
重算全局比率）95% 置信区间**共同给出。

## 3. 五个合成场景（均含已知 p 与 ground truth）

| 场景 | 日志倾向特征 | 教学结论 |
|---|---|---|
| `base` | 随排名温和变化 | 一般情形；IPS 在 ~1 SE 内命中 GT，重放覆盖率个位数 |
| `uniform` | 全部 0.5 等概率 | 重放前提成立的对照：覆盖率回升、估计贴近重放口径 GT |
| `positivity` | 排名 10 名外 **p 严格为 0** | 正性违反：IPS 只能估**可识别值**，与部署值的缺口页面标红 |
| `rare` | p≈1%–3%，1/p 最高约 100 | 权重不稳定：IPS SE 巨大，CIPS/SNIPS 偏差-方差权衡可见 |
| `misspec` | 记录倾向相对真实倾向整体偏移 | 倾向模型错误设定：IPS 显著偏低且 CI 很窄——自信但错误 |

合成数据还注入**迟到退款**（约 1/3 在 v1 cutoff 后到达），用于演示归因窗口
与退款换版（arrival → occurrence）导致的口径差异。

## 4. 防止"挑结果"的版本与口径治理

- **数据版本**：`data_versions` 是 cutoff 不可变快照，`data_hash` 覆盖快照内
  全部事件行。迟到事件只进入新版本；任务创建时校验哈希，篡改即报错。
- **配置冻结**：任务创建时 `config_json` 与 `config_hash` 固定，执行期不接受
  改参。换排序策略、归因窗口、退款规则、截断阈值、replay_k 都产生新哈希。
- **同口径组**：`comparability_key` 只含方法口径（不含策略名）。对比页只允许
  同组任务并排；不同窗口/退款口径的试验无法混入同一对比。
- **报告只增不改**：`reports` 表 append-only，保存时同时固化 data_hash 与
  config_hash，事后无法让旧报告指向新样本。

## 5. 中断恢复

- 每 50 个查询提交一次 `task_progress` checkpoint（WAL 模式）。
- SIGINT/SIGTERM：当前查询后优雅停止，状态保持 `running`。
- SIGKILL/断电：未提交批次丢失，恢复时自动重做；最终全量重算汇总，
  **恢复后的估计与一次跑完逐位一致**（测试 `test_resume_after_hard_crash` 锁定）。
- Web 工作线程不能注册信号，依赖 checkpoint + 重入；CLI 主线程注册信号。

## 6. 目录结构

```
offpolicy_eval/
  db.py          SQLite schema（日志/策略/GT/任务/报告，append-only 设计）
  hashing.py     config_hash / data_hash / comparability_key
  synthetic.py   五场景合成日志、真实倾向、ground truth、数据版本快照
  rewards.py     归因窗口 + 三种退款口径（arrival/occurrence/ignore）
  estimators.py  六个估计器、权重诊断、bootstrap、无支持区域标记
  tasks.py       任务创建/冻结/分批/checkpoint/信号与崩溃恢复
  cli.py         命令行入口
  webapp/        http.server JSON API + 静态页（六个页签）
tests/
  test_offpolicy.py   20 个 unittest（无偏性/正性/错误设定/恢复/版本/退款）
  _crash_runner.py    SIGKILL 硬崩溃子进程辅助
data/              演示数据库（*.db 已 gitignore，需本地生成）
```

## 7. 使用

```bash
# 生成合成数据（五场景之一）
python3 -I -m offpolicy_eval.cli generate --db data/app.db --scenario base --n 600

# 一条龙：同口径任务组（6 个策略版本）+ 执行 + 对照表
python3 -I -m offpolicy_eval.cli demo --db data/app.db --scenario base

# 单步：建组 → 执行（可 --crash-after N 演示硬崩溃）→ 查看
python3 -I -m offpolicy_eval.cli make-group --db data/app.db --scenario base --data-version 1
python3 -I -m offpolicy_eval.cli run --db data/app.db --task task_xxx
python3 -I -m offpolicy_eval.cli show --db data/app.db --task task_xxx

# 网页（六页签：总览/任务/对比/逐查询贡献/支持区域与权重/报告）
python3 -I -m offpolicy_eval.cli web --db data/app.db --port 8765

# 测试
python3 -I -m unittest discover -s tests -v
```

> `-I` 为隔离模式，忽略环境变量与用户站点包；脚本内已把仓库根插入 `sys.path`。

## 8. 页面能做什么

1. **数据总览**：事件计数、两个 cutoff 版本及哈希、策略版本清单。
2. **评估任务**：表单创建即冻结口径（指标/窗口/退款/截断/replay_k/方法），
   显示进度与心跳，可执行/恢复。
3. **策略对比**：同口径组柱状图（含 GT 参照线）、SE、前提告警（无支持区域、
   权重 p99、重放覆盖率、错误对照警示），顶部固定"非上线收益"声明。
4. **逐查询贡献**：按贡献绝对值排序下钻、贡献分布直方图、无支持查询标红。
5. **无支持区域/权重诊断**：无支持查询数与占比、权重 p99/max/ESS/截断比例、
   权重分布直方图、极端权重商品 Top 50。
6. **保存报告**：append-only，备注结论与前提，哈希随报告固化。

## 9. 边界与免责

- 本系统所有数字都是**历史日志上的离线估计**。上线效果只能由随机化在线实验
  （A/B）回答；任何把离线估计表述为"上线收益"的用法都被页面与报告明确拒绝。
- 合成数据的 ground truth 仅用于检验估计器，真实业务中不存在这一列。
- Bernoulli 独立曝光是重放无偏性的前提；真实榜单式日志若不满足该前提，
  重放结果应视为有偏，页面告警仍会提示覆盖率但无法自动识别该前提失效。
