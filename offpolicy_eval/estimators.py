"""离线策略估计器与诊断。

估计目标（每个查询的期望价值 V(π)）：
    V(π) = E_q [ Σ_{i ∈ S_π(q)} m_i ]，S_π(q) 为目标策略 top-k 选择集。

实现的方法及其适用前提：

1. ips（逆倾向加权, Horvitz–Thompson）
   估计量：(1/Q) Σ_q Σ_{i∈S_q, p_i>0} r_i / p_i
   前提：(a) 正性——目标会选的商品日志里都有正曝光概率；
         (b) 记录倾向等于真实倾向；(c) 无未观测混杂（随机化日志）。
   正性违反时只估计"可识别部分"，页面必须提示无支持区域。

2. cips（截断 IPS）：w_i = min(1/p_i, clip_threshold)。
   用偏差换方差；截断阈值越大越接近 IPS，诊断里同时给出被截断权重比例。

3. snips（自归一化 IPS，按 IPS 尺度标定）：
   k · [Σ w_i r_i] / [Σ w_i]，比率估计，倾向整体偏置时仍有偏，
   但方差通常更小；尺度与 IPS 相同（乘回每查询选择数 k）。

4. replay_all（top-k 集合匹配重放）：仅保留"目标前 replay_k 个商品全部实际
   曝光"的查询，取这些商品的真实奖励均值（replay_k 个商品的价值）。
   前提：匹配指示与奖励独立——独立 Bernoulli 日志满足；确定性榜单式日志
   不满足。匹配概率为倾向连乘，replay_k 越大覆盖率越低，这是重放法的核心
   折中；uniform 场景下覆盖率恢复，构成前提成立的对照。

5. replay_any（主动作匹配重放）：仅保留目标 top-1 实际曝光的查询。
   覆盖率高于集合匹配，但只衡量主动作，不能代表整页价值。

6. naive_zero（错误对照）：把目标选中但未曝光的商品奖励直接记 0。
   这是需求中明确禁止的错误口径——"从未曝光 ≠ 用户不点击"，
   保留它仅用于在合成数据上展示其向下偏差。
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .hashing import comparability_key

METHODS = ("ips", "cips", "snips", "replay_all", "replay_any", "naive_zero")
# 页面上展示的方法标签与前提说明
METHOD_META = {
    "ips": ("倾向加权 IPS", "正性 + 倾向记录正确 + 无混杂；权重可能极不稳定"),
    "cips": ("截断 IPS", "同 IPS；截断引入偏差但压低方差，阈值需预先固定"),
    "snips": ("自归一化 SNIPS", "同 IPS；比率估计，尺度已标定到 IPS 口径"),
    "replay_all": ("集合匹配重放", "匹配指示与奖励独立；覆盖率低时方差大"),
    "replay_any": ("主动作匹配重放", "仅评估 top-1 主动作；不代表整页价值"),
    "naive_zero": ("错误对照：未曝光记零", "错误方法，仅用于展示向下偏差"),
}


@dataclass
class ItemObs:
    item_id: str
    p_recorded: float
    exposed: bool
    reward: float
    m_true: float = 0.0      # 该商品的真实期望奖励（合成数据 GT，按任务指标）


@dataclass
class QueryRecord:
    query_id: str
    items: list[ItemObs]
    gt_deploy: float          # 真实部署值（合成数据 ground truth）
    gt_identifiable: float    # 仅 p>0 部分的真实值
    unsupported: bool         # 目标选中项中存在 p==0
    gt_top1: float = 0.0      # 主动作 top-1 的真实值（replay_any 对照）


@dataclass
class MethodEstimate:
    method: str
    estimate: float
    variance: float
    ci_low: float
    ci_high: float
    n_effective: float
    diagnostics: dict
    contributions: dict[str, float] = field(default_factory=dict)
    supported_flags: dict[str, int] = field(default_factory=dict)


def _sample_var(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return sum((v - mean) ** 2 for v in values) / (n - 1)


def _bootstrap(values_by_query: list[tuple[str, float, int]],
               n_bootstrap: int, seed: int, kind: str) -> tuple[float, float]:
    """对查询做有放回重采样。values_by_query: (qid, contribution, supported)。

    kind=mean  ：全部查询参与均值（IPS 类）；
    kind=match ：仅 supported 查询的匹配均值（Replay 类）。
    """
    if n_bootstrap <= 0 or not values_by_query:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(values_by_query)
    stats = []
    if kind == "mean":
        vals = [v for _q, v, _s in values_by_query]
        for _ in range(n_bootstrap):
            stats.append(sum(vals[rng.randrange(n)] for _ in range(n)) / n)
    else:
        matched = [v for _q, v, s in values_by_query if s]
        m = len(matched)
        if m < 2:
            return float("nan"), float("nan")
        for _ in range(n_bootstrap):
            stats.append(sum(matched[rng.randrange(m)] for _ in range(m)) / m)
    stats.sort()
    # 等尾 95% 百分位（边界钳制，避免小 bootstrap 次数下索引越界）
    lo_idx = max(0, min(n_bootstrap - 1, int(0.025 * n_bootstrap)))
    hi_idx = max(0, min(n_bootstrap - 1, math.ceil(0.975 * n_bootstrap) - 1))
    lo, hi = stats[lo_idx], stats[hi_idx]
    return lo, hi


def _bootstrap_ratio(nums: list[float], dens: list[float], k: int,
                     n_bootstrap: int, seed: int) -> tuple[float, float]:
    """SNIPS 专用：查询级 cluster bootstrap，每次重算全局比率再乘 k。"""
    n = len(nums)
    if n_bootstrap <= 0 or n < 2:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    stats = []
    for _ in range(n_bootstrap):
        sn = sd = 0.0
        for _ in range(n):
            i = rng.randrange(n)
            sn += nums[i]
            sd += dens[i]
        stats.append(k * (sn / sd) if sd else 0.0)
    stats.sort()
    # 等尾 95% 百分位（边界钳制，避免小 bootstrap 次数下索引越界）
    lo_idx = max(0, min(n_bootstrap - 1, int(0.025 * n_bootstrap)))
    hi_idx = max(0, min(n_bootstrap - 1, math.ceil(0.975 * n_bootstrap) - 1))
    lo, hi = stats[lo_idx], stats[hi_idx]
    return lo, hi


def run_all(records: list[QueryRecord], config: dict,
            gt_metric: str) -> dict[str, MethodEstimate]:
    clip = float(config.get("clip_threshold", 20.0))
    n_boot = int(config.get("n_bootstrap", 300))
    seed = int(config.get("bootstrap_seed", 7))
    k = int(config.get("top_k", 10))
    chosen = [m for m in config.get("methods", METHODS) if m in METHODS]
    q_total = len(records)
    out: dict[str, MethodEstimate] = {}

    # ---------- 通用权重表（IPS 族共享） ----------
    # raw_weights 保留完整 1/p（诊断极端权重用）；clipped_weights 用于
    # ESS 等截断口径指标——不能用截断后的值报告"权重是否不稳定"。
    raw_weights, clipped_weights = [], []
    clipped_count = 0
    selected_count = 0
    for rec in records:
        for it in rec.items:
            selected_count += 1
            if it.p_recorded > 0:
                w = 1.0 / it.p_recorded
                raw_weights.append(w)
                clipped_weights.append(min(w, clip))
                if w > clip:
                    clipped_count += 1
    if raw_weights:
        sw = sum(clipped_weights)
        ess = sw * sw / sum(w * w for w in clipped_weights)
        raw_sorted = sorted(raw_weights)
        w_p99 = raw_sorted[min(len(raw_sorted) - 1,
                               int(0.99 * len(raw_sorted)))]
        w_max = raw_sorted[-1]
        w_mean = sum(raw_weights) / len(raw_weights)
    else:
        ess = w_p99 = w_max = w_mean = 0.0
    n_unsupported = sum(1 for r in records if r.unsupported)

    # ---------- IPS / CIPS / SNIPS / naive_zero ----------
    ips_contrib, cips_contrib, snips_num, snips_den = {}, {}, [], []
    naive_contrib = {}
    for rec in records:
        v_ips = v_cips = v_naive = 0.0
        num_q, den_q = 0.0, 0.0
        for it in rec.items:
            v_naive += it.reward if it.exposed else 0.0
            if it.p_recorded <= 0:
                continue
            w = 1.0 / it.p_recorded
            v_ips += it.reward * w
            v_cips += it.reward * min(w, clip)
            # 经典 SNIPS：完整权重（不截断）的比率
            num_q += w * it.reward
            den_q += w
        ips_contrib[rec.query_id] = v_ips
        cips_contrib[rec.query_id] = v_cips
        naive_contrib[rec.query_id] = v_naive
        snips_num.append(num_q)
        snips_den.append(den_q)

    def finish_weighted(name: str, contrib: dict[str, float],
                        extra_diag: dict) -> MethodEstimate:
        vals = list(contrib.values())
        est = sum(vals) / q_total if q_total else 0.0
        var = _sample_var(vals) / q_total if q_total else 0.0
        lo, hi = _bootstrap(
            [(q, v, 1) for q, v in contrib.items()], n_boot, seed, "mean"
        )
        gt_dep = sum(r.gt_deploy for r in records) / q_total if records else 0.0
        gt_id = (sum(r.gt_identifiable for r in records) / q_total
                 if records else 0.0)
        diag = dict(
            coverage=1.0,
            n_queries=q_total,
            n_unsupported_queries=n_unsupported,
            unsupported_fraction=(n_unsupported / q_total if q_total else 0.0),
            weight_max=w_max, weight_p99=w_p99,
            weight_mean=w_mean,
            ess=ess,
            selected_item_count=selected_count,
            clipped_fraction=(clipped_count / selected_count
                              if selected_count else 0.0),
            gt_deploy=gt_dep, gt_identifiable=gt_id,
            rel_error_vs_deploy=((est - gt_dep) / gt_dep if gt_dep else None),
            rel_error_vs_identifiable=((est - gt_id) / gt_id if gt_id else None),
            comparability_key=comparability_key(config),
            metric=gt_metric,
        )
        diag.update(extra_diag)
        return MethodEstimate(name, est, var, lo, hi, ess, diag,
                              dict(contrib),
                              {r.query_id: 0 if r.unsupported else 1
                               for r in records})

    if "ips" in chosen:
        out["ips"] = finish_weighted("ips", ips_contrib, {"bias_note": "无偏（前提成立时）"})
    if "cips" in chosen:
        out["cips"] = finish_weighted(
            "cips", cips_contrib,
            {"clip_threshold": clip,
             "bias_note": "截断使估计有偏；与 IPS 的差值即偏差-方差权衡"},
        )
    if "snips" in chosen:
        # 经典 SNIPS：全样本比率 k · Σ_q Σ_i w_i r_i / Σ_q Σ_i w_i
        # （w=1/p 完整权重；比率的期望是"目标选择的平均单商品奖励"，
        # 乘回每查询选择数 k 后与 IPS/GT 同尺度）
        num, den = sum(snips_num), sum(snips_den)
        ratio = (num / den) if den else 0.0
        est = k * ratio
        # 逐查询贡献（页面下钻用）：每查询自身比率乘 k，全样本点估计
        # 仍以全局比率为准（二者仅在等分母时相等，这是比率估计的固有特性）
        per_q_ratio = [k * (a / b) if b else 0.0
                       for a, b in zip(snips_num, snips_den)]
        var = _sample_var(per_q_ratio) / q_total if q_total else 0.0
        lo, hi = _bootstrap_ratio(
            snips_num, snips_den, k, n_boot, seed,
        )
        gt_dep = sum(r.gt_deploy for r in records) / q_total
        gt_id = sum(r.gt_identifiable for r in records) / q_total
        out["snips"] = MethodEstimate(
            "snips", est, var, lo, hi, ess,
            dict(coverage=1.0, n_queries=q_total,
                 n_unsupported_queries=n_unsupported,
                 unsupported_fraction=n_unsupported / q_total if q_total else 0,
                 weight_max=w_max, weight_p99=w_p99, ess=ess,
                 snips_raw_ratio=ratio,
                 gt_deploy=gt_dep, gt_identifiable=gt_id,
                 rel_error_vs_deploy=((est - gt_dep) / gt_dep if gt_dep else None),
                 rel_error_vs_identifiable=((est - gt_id) / gt_id if gt_id else None),
                 comparability_key=comparability_key(config), metric=gt_metric,
                 bias_note="比率估计；倾向系统性偏置时依然有偏"),
            {r.query_id: v for r, v in zip(records, per_q_ratio)},
            {r.query_id: 0 if r.unsupported else 1 for r in records},
        )
    if "naive_zero" in chosen:
        out["naive_zero"] = finish_weighted(
            "naive_zero", naive_contrib,
            {"bias_note": "错误口径：未曝光商品被当成零奖励，必然系统性偏低"},
        )

    # ---------- Replay 族 ----------
    replay_k = int(config.get("replay_k", 3))
    for method, full in (("replay_all", True), ("replay_any", False)):
        if method not in chosen:
            continue
        contrib, flags = {}, {}
        for rec in records:
            if full:
                # 目标前 replay_k 个商品全部实际曝光才匹配（独立 Bernoulli
                # 下匹配指示与奖励独立；p=0 的商品永远无法匹配）
                head = rec.items[:replay_k]
                matched = all(it.exposed for it in head)
                matched_reward = sum(it.reward for it in head)
            else:
                matched = rec.items[0].exposed  # 目标 top-1
                matched_reward = rec.items[0].reward
            flags[rec.query_id] = int(matched)
            if matched:
                contrib[rec.query_id] = matched_reward
        m = len(contrib)
        coverage = m / q_total if q_total else 0.0
        vals = list(contrib.values())
        est = sum(vals) / m if m else 0.0
        var = _sample_var(vals) / m if m > 1 else 0.0
        # bootstrap 输入必须是真实匹配贡献（未匹配查询不参与重放重采样）
        lo, hi = _bootstrap(
            [(q, contrib.get(q, 0.0), flags[q]) for q in flags],
            n_boot, seed, "match",
        )
        if full:
            # 重放口径 GT（精确）：每个查询前 replay_k 个商品真实期望之和
            gt_replay = sum(sum(it.m_true for it in rec.items[:replay_k])
                            for rec in records) / q_total if q_total else 0.0
            gt_matched = (
                sum(sum(it.m_true for it in rec.items[:replay_k])
                    for rec in records if flags[rec.query_id]) / m
                if m else 0.0)
            diag = dict(coverage=coverage, n_matched=m, n_queries=q_total,
                        replay_k=replay_k,
                        gt_matched=gt_matched, gt_deploy=gt_replay,
                        rel_error_vs_deploy=((est - gt_replay) / gt_replay
                                             if gt_replay else None),
                        match_independence_required=True,
                        comparability_key=comparability_key(config),
                        metric=gt_metric,
                        bias_note="独立 Bernoulli 日志下无偏；确定性榜单日志"
                                  "不满足前提。覆盖率为前 replay_k 倾向连乘，"
                                  "k 越大覆盖率越低")
        else:
            gt_top1_all = sum(r.gt_top1 for r in records) / q_total
            gt_top1_matched = (sum(r.gt_top1 for r in records
                                   if flags[r.query_id]) / m if m else 0.0)
            diag = dict(coverage=coverage, n_matched=m, n_queries=q_total,
                        gt_matched=gt_top1_matched, gt_deploy=gt_top1_all,
                        rel_error_vs_deploy=((est - gt_top1_all) / gt_top1_all
                                             if gt_top1_all else None),
                        match_independence_required=True,
                        comparability_key=comparability_key(config),
                        metric=gt_metric,
                        gt_top1=gt_top1_all, gt_top1_matched=gt_top1_matched,
                        rel_error_vs_top1=((est - gt_top1_all) / gt_top1_all
                                           if gt_top1_all else None),
                        bias_note="主动作指标，不能外推为整页 top-k 价值")
        out[method] = MethodEstimate(
            method, est, var, lo, hi, float(m), diag,
            contrib, flags,
        )

    return out

