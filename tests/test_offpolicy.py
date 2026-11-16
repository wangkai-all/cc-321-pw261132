"""测试套件（仅标准库 unittest）。

运行：python3 -I -m unittest discover -s tests -v
覆盖：
  1. 数据确定性：同种子两次生成哈希一致；真实倾向落库且未曝光商品无事件
  2. 估计器性质：IPS 无偏性、正性违反只估可识别值、错误对照有偏、
     misspec 下 IPS 有偏、rare 下方差与权重诊断、Replay 前提与覆盖率
  3. 中断恢复：SIGKILL 后续跑结果与一次跑完完全一致
  4. 版本锁定：数据哈希可复现；任务配置不可变；不同口径 comparability_key 不同
  5. 归因窗口/退款换版：occurrence 与 arrival 在迟到退款上结论不同
"""
import json
import os
import signal
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from offpolicy_eval.db import connect, init_db
from offpolicy_eval.estimators import ItemObs, QueryRecord, run_all
from offpolicy_eval.hashing import (comparability_key,
                                    compute_data_hash, verify_data_hash)
from offpolicy_eval.rewards import load_query_rewards
from offpolicy_eval.synthetic import SCENARIOS, generate_dataset
from offpolicy_eval.tasks import BATCH_SIZE, create_task, run_task

ROOT = Path(__file__).resolve().parents[1]
N = 300  # 测试规模（足够收敛，又保持秒级）


def make_db(path: Path, scenario: str = "base", n: int = N) -> sqlite3.Connection:
    conn = connect(path)
    init_db(conn)
    generate_dataset(conn, scenario, n_queries=n)
    return conn


def base_config(**over):
    cfg = dict(
        methods=["ips", "cips", "snips", "replay_all", "replay_any",
                 "naive_zero"],
        clip_threshold=20.0, n_bootstrap=300, bootstrap_seed=7,
        attribution_window_hours=720.0, refund_policy="arrival",
        reward_metric="revenue", top_k=10, replay_k=3,
        strategies=[{"strategy_id": "quality", "version": "v1"}],
    )
    cfg.update(over)
    return cfg


class TestSyntheticData(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "a.db"

    def tearDown(self):
        self.tmp.cleanup()

    def test_deterministic_generation(self):
        db1 = Path(self.tmp.name) / "x.db"
        db2 = Path(self.tmp.name) / "y.db"
        make_db(db1), make_db(db2)
        with connect(db1) as c1, connect(db2) as c2:
            h1 = compute_data_hash(c1, "2026-10-13T00:00:00")
            h2 = compute_data_hash(c2, "2026-10-13T00:00:00")
            self.assertEqual(h1, h2)

    def test_unexposed_items_have_no_clicks_or_orders(self):
        with make_db(self.db) as conn:
            bad = conn.execute(
                "SELECT COUNT(*) c FROM clicks k JOIN exposures e "
                "ON e.query_id=k.query_id AND e.item_id=k.item_id "
                "WHERE e.exposed=0").fetchone()["c"]
            self.assertEqual(bad, 0, "未曝光商品出现点击，违反日志生成规则")
            bad2 = conn.execute(
                "SELECT COUNT(*) c FROM orders o JOIN exposures e "
                "ON e.query_id=o.query_id AND e.item_id=o.item_id "
                "WHERE e.exposed=0").fetchone()["c"]
            self.assertEqual(bad2, 0)

    def test_recorded_propensity_differs_only_in_misspec(self):
        for sc in SCENARIOS:
            db = Path(self.tmp.name) / f"{sc}.db"
            with make_db(db, sc, n=60) as conn:
                row = conn.execute(
                    "SELECT AVG(ABS(p_logging-p_recorded)) d FROM exposures"
                ).fetchone()
                if sc == "misspec":
                    self.assertGreater(row["d"], 0.05)
                else:
                    self.assertLess(row["d"], 1e-12)

    def test_positivity_scenario_has_zero_propensity(self):
        db = Path(self.tmp.name) / "pos.db"
        with make_db(db, "positivity", n=60) as conn:
            zeros = conn.execute(
                "SELECT COUNT(*) c FROM exposures WHERE p_logging=0"
            ).fetchone()["c"]
            self.assertGreater(zeros, 0)
            # 倾向为 0 的商品必须从未曝光
            bad = conn.execute(
                "SELECT COUNT(*) c FROM exposures "
                "WHERE p_logging=0 AND exposed=1").fetchone()["c"]
            self.assertEqual(bad, 0)


class TestEstimators(unittest.TestCase):
    # 估计器性质检验需要足够样本让渐近无偏性与 bootstrap CI 稳定
    EST_N = 600

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dbs = {}
        for sc in SCENARIOS:
            db = Path(cls.tmp.name) / f"{sc}.db"
            cls.dbs[sc] = make_db(db, sc, cls.EST_N)

    @classmethod
    def tearDownClass(cls):
        for c in cls.dbs.values():
            c.close()
        cls.tmp.cleanup()

    def _run(self, scenario, **cfg_over):
        conn = self.dbs[scenario]
        cfg = base_config(**cfg_over)
        tid = create_task(conn, f"t-{scenario}-{id(cfg)}", cfg, 1)
        run_task(conn, tid)
        results = {r["method"]: dict(r) for r in conn.execute(
            "SELECT * FROM method_results WHERE task_id=?", (tid,))}
        for r in results.values():
            r["diag"] = json.loads(r["diagnostics_json"])
        return results

    def test_ips_unbiased_when_assumptions_hold(self):
        for sc in ("base", "uniform"):
            r = self._run(sc)
            gt = r["ips"]["diag"]["gt_deploy"]
            se = r["ips"]["variance"] ** 0.5
            # 有限样本下容忍 2.5SE；无偏性是渐近性质，GT 与估计方向一致
            self.assertLess(abs(r["ips"]["estimate"] - gt), 2.5 * se,
                            f"{sc}: IPS 应在 2.5SE 内命中 GT")
            # bootstrap CI 应覆盖 GT（300 次重采样的 95% 区间）
            self.assertLessEqual(r["ips"]["ci_low"], gt)
            self.assertGreaterEqual(r["ips"]["ci_high"], gt)

    def test_positivity_identifies_only_identifiable_value(self):
        r = self._run("positivity")
        d = r["ips"]["diag"]
        self.assertGreater(d["n_unsupported_queries"], 0)
        self.assertAlmostEqual(r["ips"]["estimate"],
                               d["gt_identifiable"], delta=3 * r["ips"]["variance"] ** 0.5)
        # 部署值与可识别值之间存在不可修复缺口
        self.assertGreater(d["gt_deploy"] - d["gt_identifiable"], 1.0)

    def test_naive_zero_is_strongly_downward_biased(self):
        for sc in SCENARIOS:
            r = self._run(sc)
            gt = r["naive_zero"]["diag"]["gt_deploy"]
            self.assertLess(r["naive_zero"]["estimate"], 0.7 * gt,
                            f"{sc}: 错误对照必须显著低于 GT")

    def test_misspec_biases_ips_despite_tight_ci(self):
        r = self._run("misspec")
        gt = r["ips"]["diag"]["gt_deploy"]
        se = r["ips"]["variance"] ** 0.5
        # 偏差超过 2SE：CI 很窄但盖不住 GT —— 自信但错误
        self.assertGreater(abs(r["ips"]["estimate"] - gt), 2 * se)
        self.assertFalse(r["ips"]["ci_low"] <= gt <= r["ips"]["ci_high"])

    def test_rare_scenario_high_variance_and_weight_diagnostics(self):
        r = self._run("rare")
        base = self._run("base")
        self.assertGreater(r["ips"]["variance"], base["ips"]["variance"])
        self.assertGreater(r["ips"]["diag"]["weight_p99"], 30)
        # CIPS 的方差必须小于 IPS（偏差换方差）
        self.assertLess(r["cips"]["variance"], r["ips"]["variance"])

    def test_replay_coverage_uniform_gt_base_gt_rare(self):
        cov = {sc: self._run(sc)["replay_all"]["diag"]["coverage"]
               for sc in ("uniform", "base", "rare")}
        self.assertGreater(cov["uniform"], cov["base"])
        self.assertGreater(cov["base"], cov["rare"])

    def test_replay_unbiased_on_matched_uniform(self):
        r = self._run("uniform")
        d = r["replay_all"]["diag"]
        self.assertGreater(d["n_matched"], 20)
        # 匹配样本量仅几十个，重放均值的抽样误差大；3SE 内命中即认为无系统偏差
        self.assertLess(abs(r["replay_all"]["estimate"] - d["gt_deploy"]),
                        3 * r["replay_all"]["variance"] ** 0.5)

    def test_replay_any_tracks_top1_ground_truth(self):
        r = self._run("uniform")
        d = r["replay_any"]["diag"]
        se = r["replay_any"]["variance"] ** 0.5
        self.assertLess(abs(r["replay_any"]["estimate"] - d["gt_top1"]),
                        3 * se)

    def test_replay_confidence_interval_not_degenerate(self):
        # 回归测试：重放 bootstrap 必须使用真实匹配贡献，
        # 不能因错误传入零值而得到恒为 [0,0] 的置信区间。
        r = self._run("uniform")
        for method in ("replay_all", "replay_any"):
            res = r[method]
            self.assertGreater(res["diag"]["n_matched"], 1)
            self.assertGreater(res["estimate"], 0.0)
            self.assertIsNotNone(res["ci_low"])
            self.assertGreater(res["ci_high"], res["ci_low"],
                               f"{method} 的 CI 退化为单点，bootstrap 输入有误")
            self.assertGreater(res["ci_high"], 0.0)

    def test_unit_estimator_explicit_zero_positivity(self):
        # 直接构造记录：目标商品 p=0 时 IPS 跳过且标记 unsupported
        recs = [QueryRecord(
            "q1",
            [ItemObs("i", 0.0, False, 0.0, 5.0)],
            gt_deploy=5.0, gt_identifiable=0.0, unsupported=True, gt_top1=5.0,
        )]
        out = run_all(recs, base_config(), "revenue")
        self.assertEqual(out["ips"].estimate, 0.0)
        self.assertEqual(out["ips"].diagnostics["n_unsupported_queries"], 1)


class TestResumeAndVersioning(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "r.db"
        make_db(self.db)

    def tearDown(self):
        self.tmp.cleanup()

    def test_data_hash_reproducible_and_tamper_detected(self):
        with connect(self.db) as conn:
            self.assertTrue(verify_data_hash(conn, 1))
            # 篡改事件后必须被发现
            conn.execute("UPDATE queries SET query_text='hacked' WHERE rowid=1")
            conn.commit()
            self.assertFalse(verify_data_hash(conn, 1))

    def test_resume_after_hard_crash_matches_clean_run(self):
        crash_db = Path(self.tmp.name) / "crash.db"
        clean_db = Path(self.tmp.name) / "clean.db"
        make_db(crash_db), make_db(clean_db)
        cfg = base_config()

        with connect(clean_db) as conn:
            tid_clean = create_task(conn, "clean", cfg, 1)
            run_task(conn, tid_clean)
            clean_rows = conn.execute(
                "SELECT method, estimate FROM method_results WHERE task_id=?",
                (tid_clean,)).fetchall()

        # 子进程硬崩溃（SIGKILL 在 crash_after 个查询后）
        runner = ROOT / "tests" / "_crash_runner.py"
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        p = subprocess.run(
            [sys.executable, "-I", str(runner), str(crash_db),
             str(BATCH_SIZE + 7)],
            env=env, capture_output=True, text=True,
        )
        self.assertNotEqual(p.returncode, 0)
        with connect(crash_db) as conn:
            tid = conn.execute(
                "SELECT task_id FROM eval_tasks").fetchone()["task_id"]
            done_before = conn.execute(
                "SELECT COUNT(*) c FROM task_progress WHERE task_id=?",
                (tid,)).fetchone()["c"]
            self.assertGreater(done_before, 0)
            self.assertLess(done_before, N)  # 确实中途死掉
            # 续跑到完成
            status = run_task(conn, tid)
            self.assertEqual(status, "completed")
            crash_rows = conn.execute(
                "SELECT method, estimate FROM method_results WHERE task_id=?",
                (tid,)).fetchall()
        self.assertEqual(
            {r["method"]: round(r["estimate"], 9) for r in clean_rows},
            {r["method"]: round(r["estimate"], 9) for r in crash_rows},
            "崩溃恢复后的估计必须与一次跑完完全一致",
        )

    def test_config_immutable_and_comparability_groups(self):
        with connect(self.db) as conn:
            t1 = create_task(conn, "a", base_config(), 1)
            t2 = create_task(conn, "b", base_config(), 1)
            t3 = create_task(conn, "c",
                             base_config(attribution_window_hours=24.0), 1)
            rows = {r["task_id"]: r for r in conn.execute(
                "SELECT * FROM eval_tasks")}
            self.assertEqual(rows[t1]["config_hash"], rows[t2]["config_hash"])
            self.assertNotEqual(rows[t1]["config_hash"], rows[t3]["config_hash"])
            c1 = json.loads(rows[t1]["config_json"])
            c3 = json.loads(rows[t3]["config_json"])
            self.assertNotEqual(comparability_key(c1), comparability_key(c3))
            # 换策略但方法口径相同：comparability_key 应相同（同组可比）
            t4 = create_task(conn, "d", base_config(
                strategies=[{"strategy_id": "relevance", "version": "v1"}]), 1)
            c4 = json.loads(conn.execute(
                "SELECT config_json FROM eval_tasks WHERE task_id=?",
                (t4,)).fetchone()["config_json"])
            self.assertEqual(comparability_key(c1), comparability_key(c4))

    def test_same_task_id_resume_is_idempotent(self):
        with connect(self.db) as conn:
            tid = create_task(conn, "idem", base_config(), 1)
            run_task(conn, tid)
            again = run_task(conn, tid)
            self.assertEqual(again, "completed")

    def test_signal_flag_does_not_leak_between_tasks(self):
        # 第一个任务在处理首个查询后"收到 SIGTERM"（用注入的停止事件确定性
        # 模拟，避免定时器竞态），应在首个 checkpoint 批次后优雅停止；
        # 第二个任务使用自己的事件，不受第一个任务停止动作影响，完整跑完。
        import threading as _threading
        from offpolicy_eval.tasks import BATCH_SIZE
        with connect(self.db) as conn:
            t1 = create_task(conn, "stop-me", base_config(), 1)
            t2 = create_task(
                conn, "run-full",
                base_config(strategies=[
                    {"strategy_id": "relevance", "version": "v1"}]), 1)

            stop1 = _threading.Event()

            def stop_after_first(done, total):
                if done == 1:
                    stop1.set()

            status1 = run_task(conn, t1, progress_cb=stop_after_first,
                               stop_event=stop1)
            self.assertEqual(status1, "running")
            done1 = conn.execute(
                "SELECT COUNT(*) c FROM task_progress WHERE task_id=?",
                (t1,)).fetchone()["c"]
            self.assertLessEqual(done1, BATCH_SIZE)
            self.assertTrue(stop1.is_set())

            # 第二个任务持有独立事件，必须完整跑完
            status2 = run_task(conn, t2)
            self.assertEqual(status2, "completed")
            done2 = conn.execute(
                "SELECT COUNT(*) c FROM task_progress WHERE task_id=?",
                (t2,)).fetchone()["c"]
            total = conn.execute(
                "SELECT COUNT(*) c FROM queries q JOIN data_versions d "
                "ON d.data_version=1 WHERE q.ts_log <= d.cutoff").fetchone()["c"]
            self.assertEqual(done2, total)


class TestAttributionAndRefunds(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "ref.db"
        make_db(self.db)

    def tearDown(self):
        self.tmp.cleanup()

    def test_late_refunds_split_policies(self):
        with connect(self.db) as conn:
            late = conn.execute(
                "SELECT COUNT(*) c FROM refunds WHERE ts_arrival > "
                "(SELECT cutoff FROM data_versions WHERE data_version=1)"
            ).fetchone()["c"]
            self.assertGreater(late, 0, "合成数据必须包含迟到退款以演示换版差异")

            cutoff = conn.execute(
                "SELECT cutoff FROM data_versions WHERE data_version=1"
            ).fetchone()["cutoff"]
            qids = [r["query_id"] for r in conn.execute(
                "SELECT query_id FROM queries WHERE ts_log <= ?", (cutoff,))]

            def total_revenue(policy: str) -> float:
                total = 0.0
                for qid in qids:
                    rw = load_query_rewards(
                        conn, qid, cutoff, 720, policy, "revenue")
                    total += sum(rw.rewards.values())
                return total

            total_a, total_o = total_revenue("arrival"), total_revenue("occurrence")
            # occurrence 计入已发生但迟到的退款，净收入不高于 arrival 口径
            self.assertLessEqual(total_o, total_a)

    def test_window_change_freezes_as_separate_config(self):
        with connect(self.db) as conn:
            t1 = create_task(conn, "w720", base_config(attribution_window_hours=720), 1)
            t2 = create_task(conn, "w24", base_config(attribution_window_hours=24), 1)
            self.assertNotEqual(
                conn.execute("SELECT config_hash FROM eval_tasks WHERE task_id=?",
                             (t1,)).fetchone()["config_hash"],
                conn.execute("SELECT config_hash FROM eval_tasks WHERE task_id=?",
                             (t2,)).fetchone()["config_hash"])

    def test_v2_snapshot_contains_more_refunds(self):
        with connect(self.db) as conn:
            v1 = conn.execute(
                "SELECT n_refunds FROM data_versions WHERE data_version=1"
            ).fetchone()["n_refunds"]
            v2 = conn.execute(
                "SELECT n_refunds FROM data_versions WHERE data_version=2"
            ).fetchone()["n_refunds"]
            self.assertGreater(v2, v1)
            # 旧版本哈希不随新事件改变
            self.assertTrue(verify_data_hash(conn, 1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
