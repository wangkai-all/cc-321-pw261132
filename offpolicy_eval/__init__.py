"""离线策略评估系统（Off-policy evaluation for search ranking logs）.

仅依赖 Python 标准库。模块划分：
  db          -- SQLite 连接与 schema
  hashing     -- 配置/数据版本哈希
  synthetic   -- 合成日志生成（含已知曝光概率与 ground truth）
  rewards     -- 归因窗口 / 退款换版下的奖励口径
  estimators  -- Replay / IPS / CIPS / SNIPS / 错误对照估计器与诊断
  tasks       -- 评估任务执行、checkpoint、中断恢复
  webapp      -- 本地网页与 JSON API
"""

__version__ = "1.0.0"
