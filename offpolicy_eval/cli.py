"""命令行入口。

用法：
  python -m offpolicy_eval.cli generate --db data.db --scenario base [--n 600] [--force]
  python -m offpolicy_eval.cli list
  python -m offpolicy_eval.cli make-group --db data.db --scenario base \
      --data-version 1 [--strategy logging_rank@v1 ...]
  python -m offpolicy_eval.cli run --db data.db --task task_xxx [--crash-after N]
  python -m offpolicy_eval.cli show --db data.db --task task_xxx
  python -m offpolicy_eval.cli demo --db data.db --scenario base   # 一条龙演示
  python -m offpolicy_eval.cli web --db data.db [--port 8765]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from .db import connect, init_db
from .estimators import METHODS
from .hashing import comparability_key
from .rewards import REFUND_POLICIES, REWARD_METRICS
from .synthetic import SCENARIOS, STRATEGIES, generate_dataset
from .tasks import create_task, run_task
from .webapp import serve

DEFAULT_CONFIG = {
    "methods": list(METHODS),
    "clip_threshold": 20.0,
    "n_bootstrap": 300,
    "bootstrap_seed": 7,
    "attribution_window_hours": 720.0,
    "refund_policy": "arrival",
    "reward_metric": "revenue",
    "top_k": 10,
    "replay_k": 3,
}


def _parse_strategy(token: str) -> dict:
    sid, _, ver = token.partition("@")
    if not ver:
        raise argparse.ArgumentTypeError("策略格式应为 strategy_id@version")
    return {"strategy_id": sid, "version": ver}


def _cmd_generate(args) -> None:
    if args.force and os.path.exists(args.db):
        os.remove(args.db)
        for suffix in ("-wal", "-shm"):
            p = args.db + suffix
            if os.path.exists(p):
                os.remove(p)
    with connect(args.db) as conn:
        init_db(conn)
        generate_dataset(conn, args.scenario, n_queries=args.n)
    print(f"已生成场景 {args.scenario}：{args.n} 个查询 → {args.db}")


def _cmd_list(args) -> None:
    with connect(args.db) as conn:
        print("== 数据版本 ==")
        for r in conn.execute("SELECT * FROM data_versions ORDER BY data_version"):
            print(f"  v{r['data_version']} cutoff={r['cutoff']} "
                  f"queries={r['n_queries']} hash={r['data_hash']}")
        print("== 策略版本 ==")
        for r in conn.execute("SELECT * FROM strategy_versions"):
            print(f"  {r['strategy_id']}@{r['version']}  {r['label']}")
        print("== 任务 ==")
        for r in conn.execute("SELECT * FROM eval_tasks ORDER BY created_at"):
            print(f"  {r['task_id']} [{r['status']}] {r['name']} "
                  f"hash={r['config_hash']} dv={r['data_version']}")


def _cmd_make_group(args) -> None:
    strategies = args.strategy or [
        {"strategy_id": s, "version": v} for s, v, *_ in STRATEGIES
    ]
    with connect(args.db) as conn:
        ids = []
        for st in strategies:
            cfg = dict(DEFAULT_CONFIG)
            cfg["strategies"] = [st]
            name = f"{args.scenario}|{st['strategy_id']}@{st['version']}"
            tid = create_task(conn, name, cfg, args.data_version)
            ids.append(tid)
        print("同口径任务组 comparability_key="
              f"{comparability_key(dict(DEFAULT_CONFIG, strategies=strategies[:1]))}")
        for tid, st in zip(ids, strategies):
            print(f"  {tid}  {st['strategy_id']}@{st['version']}")


def _cmd_run(args) -> None:
    with connect(args.db) as conn:
        status = run_task(conn, args.task, crash_after=args.crash_after)
    print(f"任务 {args.task} 状态：{status}")


def _cmd_show(args) -> None:
    with connect(args.db) as conn:
        t = conn.execute("SELECT * FROM eval_tasks WHERE task_id=?",
                         (args.task,)).fetchone()
        if t is None:
            sys.exit("任务不存在")
        cfg = json.loads(t["config_json"])
        print(f"任务 {t['task_id']} [{t['status']}] {t['name']}")
        print(f"  策略: {cfg['strategies']}  数据版本: v{t['data_version']}")
        print(f"  口径哈希: {t['config_hash']}  归因窗口: "
              f"{cfg['attribution_window_hours']}h  退款: {cfg['refund_policy']}")
        print(f"  {'方法':<12}{'估计值':>12}{'标准误':>10}{'CI95%':>22}{'诊断'}")
        for r in conn.execute(
                "SELECT * FROM method_results WHERE task_id=?", (args.task,)):
            diag = json.loads(r["diagnostics_json"])
            se = (r["variance"] ** 0.5) if r["variance"] is not None else None
            se_txt = f"{se:10.3f}" if se is not None else f"{'—':>10}"
            if r["ci_low"] is None or r["ci_high"] is None:
                ci_txt = f"{'[—, —]':>22}"
            else:
                ci_txt = f"[{r['ci_low']:.2f}, {r['ci_high']:.2f}]".rjust(22)
            # 不同方法的诊断字段不同（Replay 无 ESS/权重，IPS 无 replay_k）
            diag_parts = [f"覆盖={diag.get('coverage', 0):.2%}"]
            if "n_unsupported_queries" in diag:
                diag_parts.append(f"无支持={diag['n_unsupported_queries']}")
            if isinstance(diag.get("ess"), (int, float)):
                diag_parts.append(f"ESS={diag['ess']:.1f}")
            if isinstance(diag.get("weight_p99"), (int, float)):
                diag_parts.append(f"p99权重={diag['weight_p99']:.1f}")
            if "n_matched" in diag:
                diag_parts.append(f"匹配数={diag['n_matched']}")
            print(f"  {r['method']:<12}{r['estimate']:12.3f}{se_txt}"
                  f"  {ci_txt}  {' '.join(diag_parts)}")
        if "gt_deploy" in str(conn.execute(
                "SELECT diagnostics_json FROM method_results LIMIT 1").fetchone()):
            row = conn.execute(
                "SELECT diagnostics_json FROM method_results WHERE method='ips' "
                "AND task_id=?", (args.task,)).fetchone()
            if row:
                d = json.loads(row["diagnostics_json"])
                print(f"  [ground truth] 部署值={d['gt_deploy']:.3f} "
                      f"可识别值={d['gt_identifiable']:.3f}")


def _cmd_demo(args) -> None:
    """一条龙：生成数据 → 建同口径任务组 → 跑完 → 打印对照表。"""
    if not os.path.exists(args.db):
        with connect(args.db) as conn:
            init_db(conn)
            generate_dataset(conn, args.scenario, n_queries=args.n)
    with connect(args.db) as conn:
        dv = conn.execute(
            "SELECT MIN(data_version) d FROM data_versions").fetchone()["d"]
        ids = []
        for sid, ver, *_ in STRATEGIES:
            cfg = dict(DEFAULT_CONFIG)
            cfg["strategies"] = [{"strategy_id": sid, "version": ver}]
            ids.append((f"{sid}@{ver}",
                        create_task(conn, f"demo|{sid}@{ver}", cfg, dv)))
        for _label, tid in ids:
            run_task(conn, tid)
    print(f"\n场景={args.scenario}  数据版本=v{dv}（口径全部相同）")
    print(f"{'策略':<22}{'IPS':>10}{'CIPS':>10}{'SNIPS':>10}"
          f"{'Replay':>10}{'错误对照':>10}{'GT部署值':>10}")
    with connect(args.db) as conn:
        for label, tid in ids:
            vals = {}
            gt = None
            for r in conn.execute(
                    "SELECT * FROM method_results WHERE task_id=?", (tid,)):
                vals[r["method"]] = r["estimate"]
                if r["method"] == "ips":
                    gt = json.loads(r["diagnostics_json"])["gt_deploy"]
            print(f"{label:<22}{vals.get('ips', 0):>10.2f}"
                  f"{vals.get('cips', 0):>10.2f}{vals.get('snips', 0):>10.2f}"
                  f"{vals.get('replay_all', 0):>10.2f}"
                  f"{vals.get('naive_zero', 0):>10.2f}{gt:>10.2f}")
    print("\n注意：以上均为离线日志估计，不代表上线后的真实收益。")


def _cmd_web(args) -> None:
    serve(args.db, host=args.host, port=args.port)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="offpolicy_eval",
                                description="有限候选集离线策略评估系统")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="生成合成日志")
    g.add_argument("--db", required=True)
    g.add_argument("--scenario", choices=SCENARIOS, required=True)
    g.add_argument("--n", type=int, default=600)
    g.add_argument("--force", action="store_true")
    g.set_defaults(func=_cmd_generate)

    ls = sub.add_parser("list", help="查看数据版本/策略/任务")
    ls.add_argument("--db", required=True)
    ls.set_defaults(func=_cmd_list)

    mg = sub.add_parser("make-group", help="创建同口径评估任务组")
    mg.add_argument("--db", required=True)
    mg.add_argument("--scenario", required=True)
    mg.add_argument("--data-version", type=int, required=True)
    mg.add_argument("--strategy", type=_parse_strategy, action="append")
    mg.set_defaults(func=_cmd_make_group)

    r = sub.add_parser("run", help="执行或恢复任务")
    r.add_argument("--db", required=True)
    r.add_argument("--task", required=True)
    r.add_argument("--crash-after", type=int, default=None,
                   help="测试用：处理 N 个查询后模拟硬崩溃")
    r.set_defaults(func=_cmd_run)

    sh = sub.add_parser("show", help="查看任务结果")
    sh.add_argument("--db", required=True)
    sh.add_argument("--task", required=True)
    sh.set_defaults(func=_cmd_show)

    d = sub.add_parser("demo", help="一条龙演示")
    d.add_argument("--db", required=True)
    d.add_argument("--scenario", choices=SCENARIOS, default="base")
    d.add_argument("--n", type=int, default=600)
    d.set_defaults(func=_cmd_demo)

    w = sub.add_parser("web", help="启动本地网页")
    w.add_argument("--db", required=True)
    w.add_argument("--host", default="127.0.0.1")
    w.add_argument("--port", type=int, default=8765)
    w.set_defaults(func=_cmd_web)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
