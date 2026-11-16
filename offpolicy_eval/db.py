"""SQLite 连接管理与全部表结构。

设计要点：
- 日志原始事件（曝光/点击/成交/退款）只追加，不更新删除；评估任务通过
  data_version 的 cutoff 快照读取固定样本，保证后续迟到事件不改变旧任务。
- 报告与逐查询贡献只增不改（append-only），任务配置以 JSON 冻结并算哈希。
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
PRAGMA journal_mode=WAL;

-- 一次搜索请求（查询）
CREATE TABLE IF NOT EXISTS queries (
    query_id        TEXT PRIMARY KEY,
    scenario        TEXT NOT NULL,
    query_text      TEXT NOT NULL,
    q_intent        REAL NOT NULL,
    ts_log          TEXT NOT NULL,
    n_candidates    INTEGER NOT NULL
);

-- 候选商品（有限候选集，每个 query 一行/商品，含真实奖励潜变量）
CREATE TABLE IF NOT EXISTS candidates (
    query_id        TEXT NOT NULL REFERENCES queries(query_id),
    item_id         TEXT NOT NULL,
    rank            INTEGER NOT NULL,
    rel             REAL NOT NULL,   -- 相关性潜变量
    quality         REAL NOT NULL,   -- 商品质量潜变量
    m_click         REAL NOT NULL,   -- 真实点击倾向（ground truth）
    m_revenue       REAL NOT NULL,   -- 真实成交期望净收入（ground truth）
    m_conv          REAL NOT NULL,   -- 真实成交概率（ground truth）
    PRIMARY KEY (query_id, item_id)
);

-- 曝光日志（Bernoulli 日志策略）。p_logging 为真实曝光概率，
-- p_recorded 为系统里"记录到的"倾向（misspec 场景下二者不同）。
CREATE TABLE IF NOT EXISTS exposures (
    query_id        TEXT NOT NULL,
    item_id         TEXT NOT NULL,
    position        INTEGER,                 -- 实际曝光位置，未曝光为 NULL
    exposed         INTEGER NOT NULL,        -- 0/1
    p_logging       REAL NOT NULL,           -- 生成数据的真实倾向
    p_recorded      REAL NOT NULL,           -- 记录/估计倾向（评估实际使用）
    PRIMARY KEY (query_id, item_id)
);

CREATE TABLE IF NOT EXISTS clicks (
    query_id        TEXT NOT NULL,
    item_id         TEXT NOT NULL,
    ts_event        TEXT NOT NULL,
    PRIMARY KEY (query_id, item_id)
);

CREATE TABLE IF NOT EXISTS orders (
    order_id        TEXT PRIMARY KEY,
    query_id        TEXT NOT NULL,
    item_id         TEXT NOT NULL,
    gross_amount    REAL NOT NULL,
    ts_order        TEXT NOT NULL
);

-- 迟到退款，带到达时间；cutoff / 退款换版规则据此决定是否计入
CREATE TABLE IF NOT EXISTS refunds (
    refund_id       TEXT PRIMARY KEY,
    order_id        TEXT NOT NULL REFERENCES orders(order_id),
    amount          REAL NOT NULL,
    ts_refund       TEXT NOT NULL,           -- 退款实际发生时间
    ts_arrival      TEXT NOT NULL            -- 日志到达时间（可能迟到）
);

-- 排序策略版本：策略分数在生成时固定，策略表只增不改
CREATE TABLE IF NOT EXISTS strategy_versions (
    strategy_id     TEXT NOT NULL,
    version         TEXT NOT NULL,
    label           TEXT NOT NULL,
    description     TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    PRIMARY KEY (strategy_id, version)
);

CREATE TABLE IF NOT EXISTS strategy_scores (
    strategy_id     TEXT NOT NULL,
    version         TEXT NOT NULL,
    query_id        TEXT NOT NULL,
    item_id         TEXT NOT NULL,
    score           REAL NOT NULL,
    rank            INTEGER NOT NULL,
    PRIMARY KEY (strategy_id, version, query_id, item_id)
);

-- 数据版本：不可变 cutoff 快照。counts 为该快照内的事件计数摘要，
-- data_hash 覆盖事件内容，事件追加后重新生成的新版本哈希必然不同。
CREATE TABLE IF NOT EXISTS data_versions (
    data_version    INTEGER PRIMARY KEY,
    cutoff          TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    n_queries       INTEGER NOT NULL,
    n_exposures     INTEGER NOT NULL,
    n_clicks        INTEGER NOT NULL,
    n_orders        INTEGER NOT NULL,
    n_refunds       INTEGER NOT NULL,
    data_hash       TEXT NOT NULL
);

-- 每个 query / 策略版本的 ground truth（合成数据已知，用于校验估计器）
CREATE TABLE IF NOT EXISTS ground_truth (
    query_id        TEXT NOT NULL,
    strategy_id     TEXT NOT NULL,
    version         TEXT NOT NULL,
    metric          TEXT NOT NULL,           -- revenue / conv / click
    deploy_value    REAL NOT NULL,           -- 部署值：top-k 全部真实期望之和
    identifiable_value REAL NOT NULL,        -- 可识别值：p>0 部分的期望和
    top1_value      REAL NOT NULL,           -- 主动作 top-1 的真实期望
    PRIMARY KEY (query_id, strategy_id, version, metric)
);

-- 评估任务（配置冻结，状态机 running/completed/failed）
CREATE TABLE IF NOT EXISTS eval_tasks (
    task_id         TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    status          TEXT NOT NULL,           -- pending/running/completed/failed
    config_json     TEXT NOT NULL,
    config_hash     TEXT NOT NULL,
    data_version    INTEGER NOT NULL REFERENCES data_versions(data_version),
    scenario        TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    started_at      TEXT,
    finished_at     TEXT,
    error_msg       TEXT,
    last_heartbeat  TEXT,
    pid             INTEGER
);

-- 已完成 query 的 checkpoint，中断后据此跳过
CREATE TABLE IF NOT EXISTS task_progress (
    task_id         TEXT NOT NULL REFERENCES eval_tasks(task_id),
    query_id        TEXT NOT NULL,
    done_at         TEXT NOT NULL,
    PRIMARY KEY (task_id, query_id)
);

-- 保存的报告（只增不改）
CREATE TABLE IF NOT EXISTS reports (
    report_id       TEXT PRIMARY KEY,
    task_id         TEXT NOT NULL REFERENCES eval_tasks(task_id),
    created_at      TEXT NOT NULL,
    note            TEXT,
    config_json     TEXT NOT NULL,
    data_hash       TEXT NOT NULL,
    config_hash     TEXT NOT NULL
);

-- 每个方法的总体结果与诊断
CREATE TABLE IF NOT EXISTS method_results (
    task_id         TEXT NOT NULL,
    method          TEXT NOT NULL,
    estimate        REAL NOT NULL,
    variance        REAL,
    ci_low          REAL,
    ci_high         REAL,
    n_effective     REAL NOT NULL,
    diagnostics_json TEXT NOT NULL,
    PRIMARY KEY (task_id, method)
);

-- 逐 query × 方法 的贡献（页面可下钻）
CREATE TABLE IF NOT EXISTS query_contribs (
    task_id         TEXT NOT NULL,
    query_id        TEXT NOT NULL,
    method          TEXT NOT NULL,
    contribution    REAL NOT NULL,
    supported       INTEGER NOT NULL,
    PRIMARY KEY (task_id, query_id, method)
);

-- 逐 query × 商品 的权重诊断（权重不稳定/极端权重检查）
CREATE TABLE IF NOT EXISTS item_weights (
    task_id         TEXT NOT NULL,
    query_id        TEXT NOT NULL,
    item_id         TEXT NOT NULL,
    p_recorded      REAL NOT NULL,
    ips_weight      REAL,                    -- target 选中时 1/p，否则 NULL
    target_selected INTEGER NOT NULL,
    PRIMARY KEY (task_id, query_id, item_id)
);

CREATE INDEX IF NOT EXISTS idx_scores ON strategy_scores(strategy_id, version, query_id);
CREATE INDEX IF NOT EXISTS idx_contrib_task ON query_contribs(task_id, method);
CREATE INDEX IF NOT EXISTS idx_weights_task ON item_weights(task_id);
CREATE INDEX IF NOT EXISTS idx_expo_q ON exposures(query_id);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()
