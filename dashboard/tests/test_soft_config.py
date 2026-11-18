"""软约束（soft）配置的测试：校验、plan 默认值、透传 argv、API 回显。

软约束是**运行参数**：它只拼进 bot-runtime 的命令行（``--soft``），
不进 ``policy.npz``/``model.json``。这里把这条约束固化成测试。

全部离线：不启 Paper、不启运行时。
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

DASHBOARD_DIR = Path(__file__).resolve().parents[1]
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

from config import DashboardConfig  # noqa: E402
from errors import BadRequest  # noqa: E402
from match_runner import MatchPlan, MatchRunner  # noqa: E402
from validation import parse_soft_spec  # noqa: E402


def make_cfg(root: Path) -> DashboardConfig:
    return DashboardConfig(
        host="127.0.0.1",
        port=8080,
        artifacts_dir=root / "artifacts",
        var_dir=root / "var",
        server_dir=root / "server",
        bot_runtime_dir=root / "bot-runtime",
        training_dir=root / "training",
        python_exe=sys.executable,
        java_exe="java",
        node_exe="node",
        allow_process_control=False,
        scan_external=False,
        server_jar="missing.jar",
        mc_port=25998,
        rcon_port=25575,
        runtime_port_min=8770,
        runtime_port_max=8789,
    )


class SoftSpecParseTest(unittest.TestCase):
    def test_empty_means_all_allowed(self) -> None:
        for value in (None, "", "   "):
            self.assertEqual(parse_soft_spec(value), "")

    def test_accepts_supported_forms(self) -> None:
        for spec in ("noRod", "noRod,noGapple", "rod=off,gapple=off",
                     "cps=12", "noRod,cps=12", "reachOffset=0.5",
                     "ROD=OFF", "no-block,jump=on", "rod,,gapple=on"):
            self.assertEqual(parse_soft_spec(spec), spec.strip(),
                             f"{spec} 应该被接受并原样返回")

    def test_rejects_unknown_key(self) -> None:
        with self.assertRaises(BadRequest) as ctx:
            parse_soft_spec("noRod,fly=on")
        self.assertIn("未知软约束键: fly", str(ctx.exception))
        self.assertIn("rod, gapple, block, jump, cps, reachOffset", str(ctx.exception))

    def test_rejects_bad_values(self) -> None:
        cases = {
            "cps=99": "1..40",
            "cps=0": "1..40",
            "cps=abc": "不是数字",
            "cps": "需要值",
            "rod=maybe": "不是布尔",
            "noRod=1": "不能同时带值",
            "noCps": "没有意义",
            "reachOffset=9": "0..3",
        }
        for spec, needle in cases.items():
            with self.assertRaises(BadRequest, msg=f"{spec} 应该被拒绝") as ctx:
                parse_soft_spec(spec)
            self.assertIn(needle, str(ctx.exception), f"{spec} 的报错信息不符")

    def test_rejects_non_string_and_too_long(self) -> None:
        with self.assertRaises(BadRequest):
            parse_soft_spec(["noRod"])
        with self.assertRaises(BadRequest) as ctx:
            parse_soft_spec(",".join(["noRod"] * 80))
        self.assertIn("太长", str(ctx.exception))


class MatchPlanSoftTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.cfg = make_cfg(Path(self._tmp.name))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_default_is_empty(self) -> None:
        plan = MatchPlan.from_payload({"player": "Steve"}, self.cfg)
        self.assertEqual(plan.soft, "")
        self.assertEqual(plan.to_dict()["soft"], "")

    def test_roundtrip_through_payload(self) -> None:
        plan = MatchPlan.from_payload(
            {"player": "Steve", "soft": "cps=12,noRod"}, self.cfg)
        self.assertEqual(plan.soft, "cps=12,noRod")
        # GET /api/match 的快照直接把 plan.to_dict() 摊平，前端据此回显。
        self.assertEqual(plan.to_dict()["soft"], "cps=12,noRod")

    def test_invalid_spec_rejected_before_start(self) -> None:
        with self.assertRaises(BadRequest):
            MatchPlan.from_payload({"player": "Steve", "soft": "rod=maybe"}, self.cfg)

    def test_soft_is_not_written_into_model_fields(self) -> None:
        """软约束只出现在 plan 的 soft 字段里，不该混进 brain / 模型路径。"""
        plan = MatchPlan.from_payload(
            {"player": "Steve", "soft": "noRod", "brain": "dagger-1118b"}, self.cfg)
        data = plan.to_dict()
        self.assertEqual(data["soft"], "noRod")
        self.assertNotIn("noRod", str(data["brain"]))
        self.assertIsNone(data["brainPath"] if "brainPath" in data else None)


class BrainArgvTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.cfg = make_cfg(Path(self._tmp.name))
        # 只用到 _brain_argv，不需要真的构造 server/runtimes。
        self.runner = MatchRunner.__new__(MatchRunner)
        self.runner.cfg = self.cfg

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _plan(self, **kwargs) -> MatchPlan:
        plan = MatchPlan.from_payload({"player": "Steve", **kwargs}, self.cfg)
        plan.brain_path = str(self.cfg.var_dir / "models" / "m" / "policy.npz")
        return plan

    def test_no_brain_no_args(self) -> None:
        plan = MatchPlan.from_payload({"player": "Steve", "soft": "noRod"}, self.cfg)
        plan.brain_path = None
        self.assertEqual(self.runner._brain_argv(plan), [])

    def test_soft_appended_after_existing_flags(self) -> None:
        argv = self.runner._brain_argv(self._plan(brain="m", soft="cps=12,noRod"))
        self.assertIn("--brain", argv)
        self.assertEqual(argv[-2:], ["--soft", "cps=12,noRod"])
        # 关键顺序：--soft 必须带值，不能吞掉后面的参数。
        self.assertEqual(argv[argv.index("--soft") + 1], "cps=12,noRod")

    def test_empty_soft_does_not_add_flag(self) -> None:
        argv = self.runner._brain_argv(self._plan(brain="m"))
        self.assertNotIn("--soft", argv)
        self.assertEqual(argv[-2:], ["--target-player", "Steve"])

    def test_models_dir_points_at_var_models(self) -> None:
        argv = self.runner._brain_argv(self._plan(brain="m", soft="noGapple"))
        self.assertEqual(argv[argv.index("--models-dir") + 1],
                         str((self.cfg.var_dir / "models").resolve()))


class SoftEndpointShapeTest(unittest.TestCase):
    """不起 HTTP 服务器，只验证快照字段形状（GET /api/match 用的就是这份 dict）。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.cfg = make_cfg(Path(self._tmp.name))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_snapshot_contains_soft(self) -> None:
        plan = MatchPlan.from_payload(
            {"player": "Steve", "soft": "reachOffset=0.5"}, self.cfg)
        payload = plan.to_dict()
        self.assertIn("soft", payload)
        self.assertEqual(payload["soft"], "reachOffset=0.5")

    def test_idle_snapshot_has_empty_soft(self) -> None:
        """没有对局时也返回 soft=""，前端不必区分「缺键」与「空串」。"""
        runner = MatchRunner.__new__(MatchRunner)
        runner.cfg = self.cfg
        snapshot = runner._base_snapshot()
        self.assertEqual(snapshot["soft"], "")


if __name__ == "__main__":
    unittest.main()
