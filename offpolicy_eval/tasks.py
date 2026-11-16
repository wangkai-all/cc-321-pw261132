"""评估任务：配置冻结、分批执行、checkpoint、中断恢复。

建模约定：一个评估任务 = 固定数据版本 + 固定方法口径 + 一个目标策略版本。
比较多个策略时创建一组同口径任务（comparability_key 相同），对比页只允许
同口径任务并排展示，杜绝"挑完结果再改样本/口径"。

恢复语义：
- 每处理完一个批次（BATCH_SIZE 个查询）提交一次 task_progress；
- 进程收到 SIGINT/SIGTERM 时在当前查询后优雅停止，状态保持 running；
- 再次启动同一 task_id 时跳过已完成查询，从断点继续；
  最终汇总对全量查询重算，估计结果与一次跑完完全一致；
- crash_after 用于测试：处理到指定查询数后以 SIGKILL 退出（模拟断电），
  未提交批次的进度丢失，恢复后自动重做。

口径锁定：任务创建时 config_json 与 config_hash 即固定，执行期不接受任何
参数变更；数据通过 data_version 的 cutoff 快照读取，迟到事件不影响旧任务。
"""
from __future__ import annotations

import json
import os
import signal
import sqlite3
import threading
import uuid
from datetime import datetime
from typing import Callable

from .estimators import ItemObs, QueryRecord, METHODS, run_all
from .hashing import config_hash, verify_data_hash
from .rewards import load_query_rewards

BATCH_SIZE = 50
# 当前主线程任务的停止事件：信号处理器只能访问进程级对象，而具体停止动作
# 通过闭包传递给任务，避免多个并发任务共享一个全局停止标志。
_active_stop: threading.Event | None = None


def _handle_signal(signum, frame):  # noqa: ANN001
    if _active_stop is not None:
        _active_stop.set()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def create_task(conn: sqlite3.Connection, name: str, config: dict,
                data_version: int) -> str:
    _validate_config(config)
    dv = conn.execute(
        "SELECT data_version, data_hash FROM data_versions "
        "WHERE data_version=?", (data_version,)
    ).fetchone()
    if dv is None:
        raise ValueError(f"data_version={data_version} 不存在")
    if not verify_data_hash(conn, data_version):
        raise RuntimeError("数据版本哈希校验失败：cutoff 内事件已被篡改")
    if len(config.get("strategies", [])) != 1:
        raise ValueError("一个任务只能评估一个策略版本；策略比较请创建同口径任务组")

    task_id = "task_" + uuid.uuid4().hex[:12]
    scenario = conn.execute(
        "SELECT DISTINCT scenario FROM queries LIMIT 1"
    ).fetchone()["scenario"]
    frozen = json.dumps(config, sort_keys=True, ensure_ascii=False)
    conn.execute(
        "INSERT INTO eval_tasks "
        "(task_id, name, status, config_json, config_hash, data_version, "
        " scenario, created_at, pid) VALUES (?,?,?,?,?,?,?,?,?)",
        (task_id, name, "pending", frozen, config_hash(config), data_version,
         scenario, _now(), os.getpid()),
    )
    conn.commit()
    return task_id


def _validate_config(config: dict) -> None:
    for s in config.get("strategies", []):
        if not {"strategy_id", "version"} <= set(s):
            raise ValueError("strategies 每项需包含 strategy_id 与 version")
    unknown = set(config.get("methods", [])) - set(METHODS)
    if unknown:
        raise ValueError(f"未知方法: {sorted(unknown)}")
    if float(config.get("clip_threshold", 20)) <= 0:
        raise ValueError("clip_threshold 必须为正数")
    if float(config.get("attribution_window_hours", 720)) <= 0:
        raise ValueError("attribution_window_hours 必须为正数")


def _load_query_record(conn: sqlite3.Connection, query_id: str,
                       strategy_id: str, version: str, config: dict,
                       cutoff_iso: str) -> QueryRecord:
    top_k = int(config.get("top_k", 10))
    rw = load_query_rewards(
        conn, query_id, cutoff_iso,
        float(config["attribution_window_hours"]),
        config["refund_policy"], config["reward_metric"],
    )
    rows = conn.execute(
        "SELECT s.item_id, ca.m_revenue, ca.m_conv, ca.m_click "
        "FROM strategy_scores s JOIN candidates ca "
        "ON ca.query_id=s.query_id AND ca.item_id=s.item_id "
        "WHERE s.strategy_id=? AND s.version=? AND s.query_id=? "
        "ORDER BY s.rank LIMIT ?",
        (strategy_id, version, query_id, top_k),
    ).fetchall()
    items, unsupported = [], False
    metric_col = {"revenue": "m_revenue", "conv": "m_conv",
                  "click": "m_click"}[config["reward_metric"]]
    for row in rows:
        expo = conn.execute(
            "SELECT p_logging, p_recorded, exposed FROM exposures "
            "WHERE query_id=? AND item_id=?", (query_id, row["item_id"])
        ).fetchone()
        if expo["p_logging"] <= 0.0:
            unsupported = True
        items.append(ItemObs(
            item_id=row["item_id"],
            p_recorded=expo["p_recorded"],
            exposed=bool(expo["exposed"]),
            reward=rw.rewards.get(row["item_id"], 0.0),
            m_true=row[metric_col],
        ))
    gt = conn.execute(
        "SELECT deploy_value, identifiable_value, top1_value FROM ground_truth "
        "WHERE query_id=? AND strategy_id=? AND version=? AND metric=?",
        (query_id, strategy_id, version, config["reward_metric"]),
    ).fetchone()
    return QueryRecord(
        query_id, items, gt["deploy_value"], gt["identifiable_value"],
        unsupported, gt["top1_value"],
    )


def _task_queries(conn: sqlite3.Connection, cutoff_iso: str) -> list[str]:
    return [r["query_id"] for r in conn.execute(
        "SELECT query_id FROM queries WHERE ts_log <= ? ORDER BY query_id",
        (cutoff_iso,),
    )]


def run_task(conn: sqlite3.Connection, task_id: str,
             crash_after: int | None = None,
             progress_cb: Callable[[int, int], None] | None = None,
             install_signals: bool = True,
             stop_event: threading.Event | None = None) -> str:
    """执行（或恢复）任务。返回最终状态。

    install_signals：CLI 主线程运行时可注册 SIGINT/SIGTERM；Web 工作线程
    中 Python 不允许注册信号，传 False——线程内任务仍按批次提交 checkpoint，
    进程被停止后再次调用本函数即可从断点恢复。
    stop_event：外部注入的停止事件（主要用于测试确定性模拟信号）；
    不传则创建任务私有的新事件。
    """
    if install_signals and threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

    # 任务级停止事件：每个 run_task 独立持有，互不影响；信号处理器通过
    # _active_stop 间接 set 它。局部事件保证并发任务不会被彼此的信号波及。
    stop_event = stop_event or threading.Event()
    global _active_stop
    is_main = threading.current_thread() is threading.main_thread()
    if is_main:
        _active_stop = stop_event

    def wrapped_cb(done: int, total: int) -> None:
        if progress_cb:
            progress_cb(done, total)

    task = conn.execute(
        "SELECT * FROM eval_tasks WHERE task_id=?", (task_id,)
    ).fetchone()
    if task is None:
        raise ValueError(f"任务 {task_id} 不存在")
    if task["status"] == "completed":
        return "completed"
    config = json.loads(task["config_json"])
    sid, ver = config["strategies"][0]["strategy_id"], config["strategies"][0]["version"]
    cutoff = conn.execute(
        "SELECT cutoff FROM data_versions WHERE data_version=?",
        (task["data_version"],),
    ).fetchone()["cutoff"]

    conn.execute(
        "UPDATE eval_tasks SET status='running', started_at=COALESCE(started_at,?), "
        "pid=?, last_heartbeat=? WHERE task_id=?",
        (_now(), os.getpid(), _now(), task_id),
    )
    conn.commit()

    done = {r["query_id"] for r in conn.execute(
        "SELECT query_id FROM task_progress WHERE task_id=?", (task_id,)
    )}
    queries = _task_queries(conn, cutoff)
    total = len(queries)
    processed = 0
    buffer: list[QueryRecord] = []

    def flush(records: list[QueryRecord]) -> None:
        if not records:
            return
        estimates = run_all(records, config, config["reward_metric"])
        for method, est in estimates.items():
            conn.executemany(
                "INSERT OR REPLACE INTO query_contribs VALUES (?,?,?,?,?)",
                [(task_id, rec.query_id, method,
                  est.contributions.get(rec.query_id, 0.0),
                  est.supported_flags.get(rec.query_id, 1))
                 for rec in records],
            )
        conn.executemany(
            "INSERT OR REPLACE INTO item_weights VALUES (?,?,?,?,?,?)",
            [(task_id, rec.query_id, it.item_id, it.p_recorded,
              (1.0 / it.p_recorded if it.p_recorded > 0 else None), 1)
             for rec in records for it in rec.items],
        )
        conn.executemany(
            "INSERT OR IGNORE INTO task_progress VALUES (?,?,?)",
            [(task_id, rec.query_id, _now()) for rec in records],
        )
        conn.execute(
            "UPDATE eval_tasks SET last_heartbeat=? WHERE task_id=?",
            (_now(), task_id),
        )
        conn.commit()

    for query_id in queries:
        if query_id in done:
            processed += 1
            continue
        if crash_after is not None and processed >= crash_after:
            # 模拟硬崩溃：缓冲不提交，进程立即退出；恢复时本批自动重做
            os.kill(os.getpid(), signal.SIGKILL)
        buffer.append(_load_query_record(conn, query_id, sid, ver,
                                         config, cutoff))
        processed += 1
        wrapped_cb(processed, total)
        if len(buffer) >= BATCH_SIZE:
            flush(buffer)
            buffer = []
        if stop_event.is_set():
            flush(buffer)
            if is_main:
                _active_stop = None
            return "running"

    flush(buffer)

    # 全量重算汇总：读取同一 cutoff 的固定样本，结果确定且与断点续跑无关
    all_records = [
        _load_query_record(conn, qid, sid, ver, config, cutoff)
        for qid in queries
    ]
    estimates = run_all(all_records, config, config["reward_metric"])
    for method, est in estimates.items():
        conn.execute(
            "INSERT OR REPLACE INTO method_results VALUES (?,?,?,?,?,?,?,?)",
            (task_id, method, est.estimate, est.variance,
             est.ci_low, est.ci_high, est.n_effective,
             json.dumps(est.diagnostics, ensure_ascii=False, sort_keys=True)),
        )
    conn.execute(
        "UPDATE eval_tasks SET status='completed', finished_at=?, "
        "last_heartbeat=? WHERE task_id=?", (_now(), _now(), task_id),
    )
    conn.commit()
    return "completed"
