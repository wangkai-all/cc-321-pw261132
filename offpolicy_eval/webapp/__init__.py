"""本地评估网页服务（仅标准库 http.server）。

API 一览：
  GET  /api/overview              数据版本、场景、策略、任务计数总览
  GET  /api/strategies            策略版本列表
  GET  /api/data_versions         数据版本快照及哈希
  GET  /api/tasks                 任务列表（含状态/口径/同组键）
  POST /api/tasks                 创建任务（配置即冻结）
  POST /api/tasks/<id>/run        同步执行/恢复任务（支持 ?crash_after=）
  GET  /api/tasks/<id>            任务详情、方法结果与诊断
  GET  /api/tasks/<id>/contribs   逐查询贡献（可按方法过滤、分页）
  GET  /api/tasks/<id>/weights    商品级权重诊断
  GET  /api/compare?group=key     同口径任务组对比（不同口径拒绝并排）
  GET  /api/reports / POST /api/reports   保存/列出报告（只增不改）

所有评估结果均为离线估计，API 不提供、前端也不展示任何"上线收益"字样。
"""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import closing
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ..db import connect, init_db
from ..hashing import comparability_key
from ..tasks import create_task, run_task

STATIC_DIR = Path(__file__).parent / "static"


def _json_response(handler: BaseHTTPRequestHandler, obj, status: int = 200) -> None:
    payload = json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()
    handler.wfile.write(payload)


def make_handler(db_path: str):
    lock = threading.Lock()  # SQLite 写串行化；读请求可并发但 WALL 模式下安全

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # 安静日志
            pass

        def _conn(self) -> sqlite3.Connection:
            return connect(db_path)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path, qs = parsed.path, parse_qs(parsed.query)
            try:
                if path == "/" or path.startswith("/static/"):
                    return self._serve_static(path)
                if path == "/api/overview":
                    return self._overview()
                if path == "/api/strategies":
                    return self._strategies()
                if path == "/api/data_versions":
                    return self._data_versions()
                if path == "/api/tasks":
                    return self._list_tasks()
                if path.startswith("/api/tasks/"):
                    return self._task_get(path, qs)
                if path == "/api/compare":
                    return self._compare(qs)
                if path == "/api/reports":
                    return self._list_reports()
                _json_response(self, {"error": "not found"}, 404)
            except Exception as exc:  # noqa: BLE001
                _json_response(self, {"error": str(exc)}, 500)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            path, qs = parsed.path, parse_qs(parsed.query)
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                if path == "/api/tasks":
                    return self._create_task(body)
                if path.startswith("/api/tasks/") and path.endswith("/run"):
                    task_id = path.split("/")[3]
                    crash_after = qs.get("crash_after", [None])[0]
                    return self._run_task(task_id,
                                          int(crash_after) if crash_after else None)
                if path == "/api/reports":
                    return self._save_report(body)
                _json_response(self, {"error": "not found"}, 404)
            except Exception as exc:  # noqa: BLE001
                _json_response(self, {"error": str(exc)}, 400)

        # ---------- 静态资源 ----------
        def _serve_static(self, path: str) -> None:
            name = "index.html" if path == "/" else path.removeprefix("/static/")
            fpath = (STATIC_DIR / name).resolve()
            if not str(fpath).startswith(str(STATIC_DIR.resolve())) or not fpath.is_file():
                _json_response(self, {"error": "not found"}, 404)
                return
            ctype = {
                ".html": "text/html; charset=utf-8",
                ".js": "application/javascript; charset=utf-8",
                ".css": "text/css; charset=utf-8",
            }.get(fpath.suffix, "application/octet-stream")
            data = fpath.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        # ---------- API 实现 ----------
        def _overview(self) -> None:
            with closing(self._conn()) as conn:
                dv = [dict(r) for r in conn.execute(
                    "SELECT * FROM data_versions ORDER BY data_version")]
                scenario = conn.execute(
                    "SELECT DISTINCT scenario FROM queries").fetchone()
                counts = {
                    "queries": conn.execute("SELECT COUNT(*) c FROM queries").fetchone()["c"],
                    "exposures": conn.execute("SELECT COUNT(*) c FROM exposures").fetchone()["c"],
                    "exposed": conn.execute(
                        "SELECT COUNT(*) c FROM exposures WHERE exposed=1").fetchone()["c"],
                    "clicks": conn.execute("SELECT COUNT(*) c FROM clicks").fetchone()["c"],
                    "orders": conn.execute("SELECT COUNT(*) c FROM orders").fetchone()["c"],
                    "refunds": conn.execute("SELECT COUNT(*) c FROM refunds").fetchone()["c"],
                }
                tasks = conn.execute(
                    "SELECT status, COUNT(*) c FROM eval_tasks GROUP BY status"
                ).fetchall()
            _json_response(self, {
                "scenario": scenario["scenario"] if scenario else None,
                "counts": counts,
                "data_versions": dv,
                "task_status": {r["status"]: r["c"] for r in tasks},
            })

        def _strategies(self) -> None:
            with closing(self._conn()) as conn:
                rows = conn.execute(
                    "SELECT * FROM strategy_versions ORDER BY strategy_id, version"
                ).fetchall()
            _json_response(self, [dict(r) for r in rows])

        def _data_versions(self) -> None:
            with closing(self._conn()) as conn:
                rows = conn.execute(
                    "SELECT * FROM data_versions ORDER BY data_version").fetchall()
            _json_response(self, [dict(r) for r in rows])

        def _list_tasks(self) -> None:
            with closing(self._conn()) as conn:
                rows = conn.execute(
                    "SELECT t.*, "
                    " (SELECT COUNT(*) FROM task_progress p WHERE p.task_id=t.task_id) AS n_done "
                    "FROM eval_tasks t ORDER BY t.created_at DESC").fetchall()
                out = []
                for r in rows:
                    d = dict(r)
                    cfg = json.loads(d.pop("config_json"))
                    d["config"] = cfg
                    d["comparability_key"] = comparability_key(cfg)
                    out.append(d)
            _json_response(self, out)

        def _create_task(self, body: dict) -> None:
            with lock, self._conn() as conn:
                tid = create_task(conn, body["name"], body["config"],
                                  int(body["data_version"]))
            _json_response(self, {"task_id": tid}, 201)

        def _run_task(self, task_id: str, crash_after: int | None) -> None:
            with lock, self._conn() as conn:
                status = run_task(conn, task_id, crash_after=crash_after,
                                  install_signals=False)
            _json_response(self, {"task_id": task_id, "status": status})

        def _task_get(self, path: str, qs: parse_qs) -> None:
            parts = [p for p in path.split("/") if p]
            task_id = parts[2]
            with closing(self._conn()) as conn:
                row = conn.execute(
                    "SELECT * FROM eval_tasks WHERE task_id=?", (task_id,)
                ).fetchone()
                if row is None:
                    return _json_response(self, {"error": "任务不存在"}, 404)
                task = dict(row)
                cfg = json.loads(task.pop("config_json"))
                task["config"] = cfg
                task["comparability_key"] = comparability_key(cfg)
                task["n_done"] = conn.execute(
                    "SELECT COUNT(*) c FROM task_progress WHERE task_id=?",
                    (task_id,)).fetchone()["c"]
                task["total_queries"] = conn.execute(
                    "SELECT COUNT(*) c FROM queries q "
                    "JOIN data_versions d ON d.data_version=? "
                    "WHERE q.ts_log <= d.cutoff",
                    (task["data_version"],)).fetchone()["c"]

                if len(parts) == 3:
                    results = [dict(r) for r in conn.execute(
                        "SELECT * FROM method_results WHERE task_id=?",
                        (task_id,)).fetchall()]
                    for r in results:
                        r["diagnostics"] = json.loads(r.pop("diagnostics_json"))
                    task["results"] = results
                    return _json_response(self, task)

                if parts[3] == "contribs":
                    method = qs.get("method", ["ips"])[0]
                    limit = int(qs.get("limit", [200])[0])
                    offset = int(qs.get("offset", [0])[0])
                    rows = conn.execute(
                        "SELECT c.query_id, c.contribution, c.supported, "
                        "q.query_text, q.ts_log FROM query_contribs c "
                        "JOIN queries q ON q.query_id=c.query_id "
                        "WHERE c.task_id=? AND c.method=? "
                        "ORDER BY ABS(c.contribution) DESC LIMIT ? OFFSET ?",
                        (task_id, method, limit, offset)).fetchall()
                    return _json_response(self, [dict(r) for r in rows])

                if parts[3] == "weights":
                    limit = int(qs.get("limit", [500])[0])
                    rows = conn.execute(
                        "SELECT * FROM item_weights WHERE task_id=? "
                        "ORDER BY ips_weight IS NULL, ips_weight DESC LIMIT ?",
                        (task_id, limit)).fetchall()
                    return _json_response(self, [dict(r) for r in rows])
            _json_response(self, {"error": "not found"}, 404)

        def _compare(self, qs: parse_qs) -> None:
            """同口径任务组对比。group 即 comparability_key，不匹配则拒绝。"""
            group = qs.get("group", [""])[0]
            with closing(self._conn()) as conn:
                tasks = conn.execute(
                    "SELECT * FROM eval_tasks WHERE status='completed'").fetchall()
                picked = []
                for t in tasks:
                    cfg = json.loads(t["config_json"])
                    if comparability_key(cfg) != group:
                        continue
                    d = dict(t)
                    d["config"] = cfg
                    d["results"] = [dict(r, diagnostics=json.loads(r["diagnostics_json"]))
                                    for r in conn.execute(
                                        "SELECT * FROM method_results WHERE task_id=?",
                                        (t["task_id"],))]
                    picked.append(d)
                if not picked:
                    return _json_response(self, {"error": "该口径组没有已完成任务"}, 404)
                # 二次校验：所有任务口径必须完全一致
                keys = {comparability_key(t["config"]) for t in picked}
                if len(keys) != 1:
                    return _json_response(self, {"error": "口径不一致，禁止并排比较"}, 400)
            _json_response(self, {"group": group, "tasks": picked})

        def _save_report(self, body: dict) -> None:
            import uuid
            report_id = "rpt_" + uuid.uuid4().hex[:10]
            with lock, self._conn() as conn:
                t = conn.execute(
                    "SELECT * FROM eval_tasks WHERE task_id=?",
                    (body["task_id"],)).fetchone()
                if t is None:
                    return _json_response(self, {"error": "任务不存在"}, 404)
                cfg = json.loads(t["config_json"])
                dv = conn.execute(
                    "SELECT data_hash FROM data_versions WHERE data_version=?",
                    (t["data_version"],)).fetchone()
                conn.execute(
                    "INSERT INTO reports VALUES (?,?,?,?,?,?,?)",
                    (report_id, body["task_id"],
                     datetime.now().isoformat(timespec="seconds"),
                     body.get("note", ""),
                     json.dumps(cfg, ensure_ascii=False, sort_keys=True),
                     dv["data_hash"], t["config_hash"]))
                conn.commit()
            _json_response(self, {"report_id": report_id}, 201)

        def _list_reports(self) -> None:
            with closing(self._conn()) as conn:
                rows = conn.execute(
                    "SELECT * FROM reports ORDER BY created_at DESC").fetchall()
            _json_response(self, [dict(r) for r in rows])

    return Handler


def serve(db_path: str, host: str = "127.0.0.1", port: int = 8765) -> None:
    with closing(connect(db_path)) as conn:
        init_db(conn)
    httpd = ThreadingHTTPServer((host, port), make_handler(db_path))
    httpd.daemon_threads = True  # 任务线程不阻塞进程退出
    print(f"离线策略评估页面: http://{host}:{port}/  (db={db_path})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()
    finally:
        httpd.server_close()
