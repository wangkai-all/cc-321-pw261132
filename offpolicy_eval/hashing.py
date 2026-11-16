"""配置哈希与数据版本哈希。

两个作用：
1. config_hash：冻结评估方法口径（方法集合、截断阈值、bootstrap、归因窗口、
   退款换版规则、奖励指标、top_k）。口径不同的试验不得当作同组对比挑结果。
2. data_hash：覆盖 cutoff 内全部事件行，事件追加后旧版本哈希保持不变，
   证明旧任务样本未被悄悄修改。
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any

# 参与"同组可比"判定的口径字段（任务名/说明不参与）
COMPARABILITY_KEYS = (
    "methods", "clip_threshold", "n_bootstrap", "bootstrap_seed",
    "attribution_window_hours", "refund_policy", "reward_metric", "top_k",
    "replay_k",
)


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def config_hash(config: dict) -> str:
    payload = canonical_json(config)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def comparability_key(config: dict) -> str:
    """同口径判定：只哈希方法口径，不含策略本身。

    不同策略、但方法参数/归因窗口/退款规则/指标完全一致的任务才属于
    同一对比组；换任何一个口径字段都会得到新键，禁止跨组并排挑结果。
    """
    sub = {k: config.get(k) for k in COMPARABILITY_KEYS}
    return hashlib.sha256(canonical_json(sub).encode("utf-8")).hexdigest()[:16]


def compute_data_hash(conn: sqlite3.Connection, cutoff_iso: str) -> str:
    """对 cutoff 内所有事件行做有序哈希。

    只追加的表在此 cutoff 下内容恒定，故同一 data_version 的哈希可复现。
    """
    h = hashlib.sha256()

    def feed(table: str, where: str, params: tuple) -> None:
        rows = conn.execute(
            f"SELECT * FROM {table} WHERE {where} ORDER BY rowid", params
        ).fetchall()
        for row in rows:
            h.update(canonical_json(dict(row)).encode("utf-8"))

    feed("queries", "ts_log <= ?", (cutoff_iso,))
    feed(
        "exposures",
        "query_id IN (SELECT query_id FROM queries WHERE ts_log <= ?)",
        (cutoff_iso,),
    )
    feed("clicks", "ts_event <= ?", (cutoff_iso,))
    feed("orders", "ts_order <= ?", (cutoff_iso,))
    feed(
        "refunds",
        "ts_arrival <= ? AND ts_refund <= ?",
        (cutoff_iso, cutoff_iso),
    )
    return h.hexdigest()[:16]


def verify_data_hash(conn: sqlite3.Connection, data_version: int) -> bool:
    row = conn.execute(
        "SELECT cutoff, data_hash FROM data_versions WHERE data_version=?",
        (data_version,),
    ).fetchone()
    if row is None:
        return False
    return compute_data_hash(conn, row["cutoff"]) == row["data_hash"]
