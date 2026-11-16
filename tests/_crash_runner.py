"""测试辅助：创建任务后在处理 N 个查询时模拟 SIGKILL 硬崩溃。

由 test_offpolicy.py 以子进程方式调用；崩溃后数据库中只保留已提交的
checkpoint 批次，用于验证主进程的断点续跑能力。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from offpolicy_eval.db import connect
from offpolicy_eval.tasks import create_task, run_task

# 与测试中一致的配置
CONFIG = dict(
    methods=["ips", "cips", "snips", "replay_all", "replay_any", "naive_zero"],
    clip_threshold=20.0, n_bootstrap=100, bootstrap_seed=7,
    attribution_window_hours=720.0, refund_policy="arrival",
    reward_metric="revenue", top_k=10, replay_k=3,
    strategies=[{"strategy_id": "quality", "version": "v1"}],
)


def main() -> None:
    db_path, crash_after = sys.argv[1], int(sys.argv[2])
    with connect(db_path) as conn:
        tid = create_task(conn, "crash-task", CONFIG, 1)
        run_task(conn, tid, crash_after=crash_after)
    # 正常走到这里说明崩溃注入没生效，测试应当失败
    print("ERROR: crash did not happen", file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    main()
