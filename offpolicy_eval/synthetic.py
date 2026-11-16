"""合成搜索日志生成。

关键事实（合成数据的"上帝视角"）：
- 每个候选商品都有真实点击概率 m_click、成交概率 m_conv、期望净收入 m_revenue；
- 每次曝光的真实倾向 p_logging 已知并落库；misspec 场景下系统记录的
  p_recorded 与真实值不同，用于演示倾向模型错误设定；
- ground_truth 表保存每个策略 top-k 选择的真实部署值与可识别值，
  可直接核对各估计器是否有偏。

五个场景对应不同的方法论前提：
  base       日志倾向随排名温和变化（一般情形）
  uniform    所有候选等概率 0.5 曝光（匹配重放前提成立的对照）
  positivity 排名 10 名以外倾向严格为 0（正性违反 / 无支持区域）
  rare       倾向极小（1.5%~3.5%），IPS 权重巨大（高方差反例）
  misspec    记录倾向相对真实倾向整体偏移（倾向模型错误设定反例）
"""
from __future__ import annotations

import hashlib
import math
import random
import sqlite3
from datetime import datetime, timedelta

from .db import init_db
from .hashing import compute_data_hash

SCENARIOS = ("base", "uniform", "positivity", "rare", "misspec")

STRATEGIES = (
    ("logging_rank", "v1", "日志排名策略", "完全沿用线上日志排名"),
    ("quality", "v1", "质量优先策略", "0.7*质量 + 0.3*相关性"),
    ("relevance", "v1", "相关性优先策略", "0.8*相关性 + 0.2*质量"),
    ("fixed_random", "v1", "固定随机策略", "基于哈希的确定性随机排序（探索对照）"),
    ("explore_v2", "v1", "探索策略 v1", "相关性+质量加权（线上旧版）"),
    ("explore_v2", "v2", "探索策略 v2", "引入相关性×质量交互项（换版后新版）"),
)

DAY0 = datetime(2026, 10, 1, 0, 0, 0)
# v1 快照：迟到退款尚未到达；v2 快照：全部事件到齐
CUTOFF_V1 = datetime(2026, 10, 13, 0, 0, 0)
CUTOFF_V2 = datetime(2026, 10, 17, 0, 0, 0)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _stable_rand(tag: str, *parts: object) -> float:
    raw = "|".join(map(str, parts)).encode("utf-8")
    digest = hashlib.md5(tag.encode("utf-8") + b"#" + raw).hexdigest()
    return int(digest[:12], 16) / float(16 ** 12)


def propensity(scenario: str, rank: int) -> float:
    """真实曝光概率（Bernoulli 日志策略）。"""
    if scenario == "base":
        return _sigmoid(1.2 - 0.25 * rank)
    if scenario == "uniform":
        return 0.5
    if scenario == "positivity":
        return 0.0 if rank >= 10 else _sigmoid(1.0 - 0.2 * rank)
    if scenario == "rare":
        # 极小倾向：1/p 最大约 100，天然演示权重不稳定与 IPS 高方差
        return 0.01 + 0.02 * math.exp(-0.15 * rank)
    if scenario == "misspec":
        return _sigmoid(0.8 - 0.25 * rank)
    raise ValueError(f"unknown scenario: {scenario}")


def recorded_propensity(scenario: str, rank: int, p_true: float) -> float:
    """系统记录/估计到的倾向。misspec 下故意偏离真实值。"""
    if scenario == "misspec":
        logit = math.log(p_true / (1.0 - p_true))
        return _sigmoid(logit + 0.8)  # 系统性高估倾向 → 权重整体偏小
    return p_true


def _strategy_score(strategy_id: str, version: str, rank: int,
                    rel: float, quality: float, query_id: str,
                    item_id: str) -> float:
    if strategy_id == "logging_rank":
        return -float(rank)
    if strategy_id == "quality":
        return 0.7 * quality + 0.3 * rel - rank * 1e-9
    if strategy_id == "relevance":
        return 0.8 * rel + 0.2 * quality - rank * 1e-9
    if strategy_id == "fixed_random":
        return _stable_rand("rand-strategy", query_id, item_id) - rank * 1e-12
    if strategy_id == "explore_v2":
        noise = _stable_rand("explore", version, query_id, item_id)
        if version == "v1":
            return rel + 0.5 * quality + 0.1 * noise
        return rel + 0.5 * quality + 1.2 * rel * quality + 0.1 * noise
    raise ValueError(strategy_id)


def generate_dataset(conn: sqlite3.Connection, scenario: str,
                     n_queries: int = 600, seed: int = 20261116) -> None:
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario 必须是 {SCENARIOS} 之一")
    init_db(conn)
    if conn.execute("SELECT 1 FROM queries LIMIT 1").fetchone():
        raise RuntimeError("数据库中已有数据；如需重新生成请使用新文件或 --force")

    rng = random.Random(seed)
    order_seq = 0
    refund_seq = 0

    for qi in range(n_queries):
        qrng = random.Random(seed * 1_000_003 + qi)
        query_id = f"q{qi:06d}"
        q_intent = qrng.betavariate(2, 2)
        query_text = f"合成查询-{scenario}-{qi % 50:02d}"
        ts_log = DAY0 + timedelta(
            days=qrng.randint(0, 9),
            hours=qrng.randint(0, 23),
            minutes=qrng.randint(0, 59),
        )
        n_items = 12 + qrng.randint(0, 8)
        items = []
        for rank in range(n_items):
            item_id = f"{query_id}-i{rank:02d}"
            rel = _clip01(qrng.betavariate(2, 3) + 0.3 * q_intent - 0.15)
            quality = qrng.betavariate(3, 2)
            m_click = _clip01(0.05 + 0.6 * rel + 0.2 * quality)
            cond_conv = _clip01(0.1 + 0.5 * quality)
            m_conv = m_click * cond_conv
            amount = 20.0 + 180.0 * quality
            gross_exp = m_conv * amount
            refund_prob = 0.35 * (1.0 - quality)  # 低质量商品更可能退款
            m_revenue = gross_exp * (1.0 - refund_prob)
            items.append(dict(
                item_id=item_id, rank=rank, rel=rel, quality=quality,
                m_click=m_click, m_conv=m_conv, m_revenue=m_revenue,
                refund_prob=refund_prob, amount=amount,
            ))

        conn.execute(
            "INSERT INTO queries VALUES (?,?,?,?,?,?)",
            (query_id, scenario, query_text, q_intent,
             ts_log.isoformat(), n_items),
        )
        conn.executemany(
            "INSERT INTO candidates VALUES (?,?,?,?,?,?,?,?)",
            [(query_id, it["item_id"], it["rank"], it["rel"], it["quality"],
              it["m_click"], it["m_revenue"], it["m_conv"]) for it in items],
        )

        # 曝光：Bernoulli 日志策略，位置按曝光先后（日志排名顺序）分配
        position = 0
        for it in items:
            p_true = propensity(scenario, it["rank"])
            p_rec = recorded_propensity(scenario, it["rank"], p_true)
            exposed = qrng.random() < p_true
            pos = position if exposed else None
            if exposed:
                position += 1
            conn.execute(
                "INSERT INTO exposures VALUES (?,?,?,?,?,?)",
                (query_id, it["item_id"], pos, int(exposed), p_true, p_rec),
            )

            # 点击/成交只能发生在曝光之后；未曝光商品不可能有点击
            if exposed and qrng.random() < it["m_click"]:
                ts_click = ts_log + timedelta(minutes=qrng.randint(1, 30))
                conn.execute(
                    "INSERT INTO clicks VALUES (?,?,?)",
                    (query_id, it["item_id"], ts_click.isoformat()),
                )
                if qrng.random() < (it["m_conv"] / it["m_click"]):
                    order_seq += 1
                    order_id = f"ord{order_seq:07d}"
                    gross = it["amount"] * qrng.uniform(0.85, 1.15)
                    ts_order = ts_click + timedelta(minutes=qrng.randint(2, 600))
                    conn.execute(
                        "INSERT INTO orders VALUES (?,?,?,?,?)",
                        (order_id, query_id, it["item_id"],
                         round(gross, 2), ts_order.isoformat()),
                    )
                    if qrng.random() < it["refund_prob"]:
                        refund_seq += 1
                        refund_id = f"ref{refund_seq:07d}"
                        ts_refund = ts_order + timedelta(
                            days=qrng.randint(1, 4),
                            hours=qrng.randint(0, 23),
                        )
                        # 约 1/3 退款日志迟到：v1 cutoff 之后才到达
                        if qrng.random() < 0.34:
                            ts_arrival = CUTOFF_V1 + timedelta(
                                hours=rng.randint(2, 70)
                            )
                        else:
                            ts_arrival = ts_refund + timedelta(
                                hours=qrng.randint(1, 20)
                            )
                        conn.execute(
                            "INSERT INTO refunds VALUES (?,?,?,?,?)",
                            (refund_id, order_id, round(gross, 2),
                             ts_refund.isoformat(), ts_arrival.isoformat()),
                        )

        # 策略分数与排名（一次性固定，策略版本只增不改）
        for sid, ver, _label, _desc in STRATEGIES:
            scored = sorted(
                items,
                key=lambda it: _strategy_score(
                    sid, ver, it["rank"], it["rel"], it["quality"],
                    query_id, it["item_id"],
                ),
                reverse=True,
            )
            conn.executemany(
                "INSERT INTO strategy_scores VALUES (?,?,?,?,?,?)",
                [(sid, ver, query_id, it["item_id"],
                  _strategy_score(sid, ver, it["rank"], it["rel"],
                                  it["quality"], query_id, it["item_id"]),
                  r)
                 for r, it in enumerate(scored)],
            )

        # ground truth：每个策略 top-k 的真实部署值 / 可识别值
        for sid, ver, _l, _d in STRATEGIES:
            rows = conn.execute(
                "SELECT item_id, rank FROM strategy_scores "
                "WHERE strategy_id=? AND version=? AND query_id=? ORDER BY rank",
                (sid, ver, query_id),
            ).fetchall()
            for metric in ("revenue", "conv", "click"):
                deploy, identifiable, top1 = 0.0, 0.0, 0.0
                for row in rows[:10]:
                    cand = conn.execute(
                        "SELECT m_revenue, m_conv, m_click FROM candidates "
                        "WHERE query_id=? AND item_id=?",
                        (query_id, row["item_id"]),
                    ).fetchone()
                    expo = conn.execute(
                        "SELECT p_logging FROM exposures "
                        "WHERE query_id=? AND item_id=?",
                        (query_id, row["item_id"]),
                    ).fetchone()
                    m = cand[f"m_{metric}"]
                    deploy += m
                    if expo["p_logging"] > 0.0:
                        identifiable += m
                    if row["rank"] == 0:
                        top1 = m
                conn.execute(
                    "INSERT INTO ground_truth VALUES (?,?,?,?,?,?,?)",
                    (query_id, sid, ver, metric, deploy, identifiable, top1),
                )

    for sid, ver, label, desc in STRATEGIES:
        conn.execute(
            "INSERT OR IGNORE INTO strategy_versions VALUES (?,?,?,?,?)",
            (sid, ver, label, desc, datetime.now().isoformat(timespec="seconds")),
        )
    snapshot_data_version(conn, CUTOFF_V1)
    snapshot_data_version(conn, CUTOFF_V2)
    conn.commit()


def snapshot_data_version(conn: sqlite3.Connection, cutoff: datetime) -> int:
    """在 cutoff 处创建不可变数据版本快照。"""
    cutoff_iso = cutoff.isoformat()
    existing = conn.execute(
        "SELECT data_version FROM data_versions WHERE cutoff=?", (cutoff_iso,)
    ).fetchone()
    if existing:
        return int(existing["data_version"])

    q = conn.execute(
        "SELECT COUNT(*) c FROM queries WHERE ts_log <= ?", (cutoff_iso,)
    ).fetchone()["c"]
    n_expo = conn.execute(
        "SELECT COUNT(*) c FROM exposures e JOIN queries qu ON qu.query_id=e.query_id "
        "WHERE qu.ts_log <= ?", (cutoff_iso,)
    ).fetchone()["c"]
    n_click = conn.execute(
        "SELECT COUNT(*) c FROM clicks WHERE ts_event <= ?", (cutoff_iso,)
    ).fetchone()["c"]
    n_ord = conn.execute(
        "SELECT COUNT(*) c FROM orders WHERE ts_order <= ?", (cutoff_iso,)
    ).fetchone()["c"]
    n_ref = conn.execute(
        "SELECT COUNT(*) c FROM refunds "
        "WHERE ts_arrival <= ? AND ts_refund <= ?", (cutoff_iso, cutoff_iso)
    ).fetchone()["c"]
    dh = compute_data_hash(conn, cutoff_iso)
    cur = conn.execute(
        "INSERT INTO data_versions "
        "(cutoff, created_at, n_queries, n_exposures, n_clicks, n_orders, "
        " n_refunds, data_hash) VALUES (?,?,?,?,?,?,?,?)",
        (cutoff_iso, datetime.now().isoformat(timespec="seconds"),
         q, n_expo, n_click, n_ord, n_ref, dh),
    )
    conn.commit()
    return int(cur.lastrowid)
