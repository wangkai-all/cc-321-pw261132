"""奖励口径：归因窗口 + 退款换版规则。

口径属于评估任务配置的一部分，创建后冻结（见 hashing.config_hash）：
- attribution_window_hours：以 query 发生时间为起点，窗口外的点击/成交不计入；
- refund_policy：
    arrival    退款日志在 cutoff 前到达才扣减（旧线上口径，迟到退款漏算）；
    occurrence 按退款实际发生时间判定（换版后修正口径，不依赖日志到达）；
    ignore     完全不考虑退款（毛收入口径，仅用于敏感性分析）。
- reward_metric：revenue（净收入）/ conv（成交数）/ click（点击数）。
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta

REFUND_POLICIES = ("arrival", "occurrence", "ignore")
REWARD_METRICS = ("revenue", "conv", "click")


@dataclass
class QueryRewards:
    query_id: str
    cutoff: str
    window_hours: float
    refund_policy: str
    metric: str
    rewards: dict[str, float] = field(default_factory=dict)
    n_orders: int = 0
    n_refunds_used: int = 0
    n_late_arrival: int = 0


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def load_query_rewards(conn: sqlite3.Connection, query_id: str,
                       cutoff_iso: str, window_hours: float,
                       refund_policy: str, metric: str) -> QueryRewards:
    if refund_policy not in REFUND_POLICIES:
        raise ValueError(f"refund_policy 必须是 {REFUND_POLICIES}")
    if metric not in REWARD_METRICS:
        raise ValueError(f"reward_metric 必须是 {REWARD_METRICS}")

    q = conn.execute(
        "SELECT ts_log FROM queries WHERE query_id=?", (query_id,)
    ).fetchone()
    t0 = _parse(q["ts_log"])
    window_end = t0 + timedelta(hours=window_hours)
    cutoff = _parse(cutoff_iso)
    upper = min(window_end, cutoff)

    out = QueryRewards(query_id, cutoff_iso, window_hours, refund_policy, metric)
    candidates = conn.execute(
        "SELECT item_id FROM candidates WHERE query_id=?", (query_id,)
    ).fetchall()
    rewards = {row["item_id"]: 0.0 for row in candidates}

    if metric == "click":
        clicks = conn.execute(
            "SELECT item_id, ts_event FROM clicks WHERE query_id=?", (query_id,)
        ).fetchall()
        for c in clicks:
            if t0 <= _parse(c["ts_event"]) <= upper:
                rewards[c["item_id"]] = rewards.get(c["item_id"], 0.0) + 1.0
        out.rewards = rewards
        return out

    # conv / revenue：窗口内的成交
    orders = conn.execute(
        "SELECT order_id, item_id, gross_amount, ts_order "
        "FROM orders WHERE query_id=? ORDER BY ts_order", (query_id,)
    ).fetchall()
    valid_orders = []
    for o in orders:
        if t0 <= _parse(o["ts_order"]) <= upper:
            valid_orders.append(o)
    out.n_orders = len(valid_orders)

    for o in valid_orders:
        rewards[o["item_id"]] = rewards.get(o["item_id"], 0.0) + (
            1.0 if metric == "conv" else o["gross_amount"]
        )

    if metric == "revenue" and refund_policy != "ignore":
        for o in valid_orders:
            refunds = conn.execute(
                "SELECT refund_id, amount, ts_refund, ts_arrival "
                "FROM refunds WHERE order_id=?", (o["order_id"],)
            ).fetchall()
            for r in refunds:
                t_ref, t_arr = _parse(r["ts_refund"]), _parse(r["ts_arrival"])
                if t_ref > cutoff:
                    continue  # 退款在 cutoff 之后才发生，任何口径都还不知道
                if refund_policy == "arrival":
                    counted = t_arr <= cutoff
                    if not counted:
                        out.n_late_arrival += 1
                else:  # occurrence：只看实际发生时间
                    counted = True
                if counted:
                    rewards[o["item_id"]] -= r["amount"]
                    out.n_refunds_used += 1

    out.rewards = rewards
    return out
