"""训练计划「实验参数（高级）」的测试（纯标准库 unittest）。

覆盖（对照实验从网页发起的完整链路）：
- 默认值兼容：缺省请求生成的 argv 与改造前一致，且与后端 ``defaults`` 同源；
- argv 透传：算法 / 动作空间 / BC 局数 / 学习率 / 目标熵 / 熵系数 / 胜利权重；
- 布尔开关只在为 True 时出现（False 完全不加参数）；
- ``initPolicy``：模型 id → ``var/models/<id>/policy.npz``、文件路径、覆盖 modelId 起点；
- 校验：端口 1..65535 / 重复 / 数量、用户数量与端口一对一、枚举、数值区间、
  ``actionSpace=v2`` 与 ``bcEpisodes>0`` 冲突；
- 多场地：train 专属参数明确 400，照抄默认值放行。

运行：
    cd D:\\cbot-gladiator\\dashboard
    ..\\training\\.venv\\Scripts\\python.exe -m unittest discover -s tests -q
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

DASHBOARD_DIR = Path(__file__).resolve().parents[1]
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

from app import DashboardApp  # noqa: E402
from config import DashboardConfig  # noqa: E402
from errors import BadRequest  # noqa: E402
from validation import EXPERIMENT_DEFAULTS  # noqa: E402


class ExperimentFieldTests(unittest.TestCase):
    """``_build_train`` / ``_build_multi`` 对实验参数的处理。"""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.root = root
        self.cfg = DashboardConfig(
            host="127.0.0.1",
            port=8093,
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
            mc_port=25993,
        )
        self.app = DashboardApp(self.cfg)
        self.runner = self.app.training

    def tearDown(self) -> None:
        self.app.close()
        self.tmp.cleanup()

    # --- 夹具 -----------------------------------------------------------
    def payload(self, **extra: object) -> dict:
        base = {
            "users": ["Bot_T1", "Bot_T2"],
            "ports": [8770, 8771],
            "plan": [{"opponentMode": "idle", "episodes": 1}],
        }
        base.update(extra)
        return base

    @staticmethod
    def arg(argv: list[str], flag: str) -> str:
        return argv[argv.index(flag) + 1]

    def make_model(self, ident: str = "exp-a", *, policy: bool = True) -> Path:
        """造一个注册表模型（可选带权重），模拟「已有模型」下拉里的条目。"""
        folder = self.cfg.var_dir / "models" / ident
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "model.json").write_text(
            json.dumps({"format": "gladiator.model/v1", "id": ident, "name": ident,
                        "obsDim": 23, "numActions": 11}),
            encoding="utf-8",
        )
        path = folder / "policy.npz"
        if policy:
            path.write_bytes(b"npz")
        return path

    # --- 默认值 ---------------------------------------------------------
    def test_defaults_use_cli_equivalent_values(self) -> None:
        """缺省请求：新增开关都取与 CLI 默认行为等价的取值，且不重复既有参数。"""
        argv, params, _total, _counts = self.runner._build_train(self.payload())
        self.assertEqual(self.arg(argv, "--algo"), "ppo")
        self.assertEqual(self.arg(argv, "--action-space"), "v1")
        self.assertEqual(self.arg(argv, "--bc-episodes"), "0")
        self.assertEqual(self.arg(argv, "--lr"), str(EXPERIMENT_DEFAULTS["lr"]))
        self.assertEqual(self.arg(argv, "--target-entropy"), str(EXPERIMENT_DEFAULTS["targetEntropy"]))
        self.assertEqual(self.arg(argv, "--entropy-coef"), str(EXPERIMENT_DEFAULTS["entropyCoef"]))
        self.assertEqual(self.arg(argv, "--victory"), str(EXPERIMENT_DEFAULTS["victory"]))
        self.assertNotIn("--eval-before-train", argv)
        self.assertNotIn("--reset-policy-head", argv)
        # 既有参数不能被实验参数重复追加（argparse 后写覆盖前写，重复即隐患）。
        for flag in ("--seed", "--users", "--ports", "--stages"):
            self.assertEqual(argv.count(flag), 1, f"{flag} 只应出现一次")
        self.assertEqual(params["experiment"]["seed"], EXPERIMENT_DEFAULTS["seed"])

    def test_defaults_match_backend_options(self) -> None:
        """/api/training/options 的 defaults 是前端的唯一默认值来源。"""
        defaults = self.runner.options()["defaults"]
        for key in ("algo", "actionSpace", "initPolicy", "bcEpisodes", "evalBeforeTrain",
                    "victory", "lr", "targetEntropy", "entropyCoef", "resetPolicyHead",
                    "seed", "users", "ports"):
            self.assertIn(key, defaults, f"defaults 缺少 {key}")
        self.assertEqual(defaults["algo"], "ppo")
        self.assertEqual(defaults["actionSpace"], "v1")
        self.assertEqual(defaults["bcEpisodes"], 0)
        self.assertEqual(defaults["evalBeforeTrain"], False)
        self.assertEqual(defaults["resetPolicyHead"], False)
        self.assertEqual(defaults["users"], ["Bot_T1", "Bot_T2"])
        self.assertEqual(defaults["ports"], [8770, 8771])

    def test_experiment_argv_passthrough(self) -> None:
        argv, params, _total, _counts = self.runner._build_train(self.payload(
            algo="reinforce",
            actionSpace="v1",
            bcEpisodes=5,
            evalBeforeTrain=True,
            resetPolicyHead=True,
            seed=11,
            victory=80,
            lr=0.0005,
            targetEntropy=0.8,
            entropyCoef=0.03,
        ))
        self.assertEqual(self.arg(argv, "--algo"), "reinforce")
        self.assertEqual(self.arg(argv, "--action-space"), "v1")
        self.assertEqual(self.arg(argv, "--bc-episodes"), "5")
        self.assertEqual(self.arg(argv, "--seed"), "11")
        self.assertEqual(self.arg(argv, "--victory"), "80.0")
        self.assertAlmostEqual(float(self.arg(argv, "--lr")), 0.0005)
        self.assertAlmostEqual(float(self.arg(argv, "--target-entropy")), 0.8)
        self.assertAlmostEqual(float(self.arg(argv, "--entropy-coef")), 0.03)
        self.assertIn("--eval-before-train", argv)
        self.assertIn("--reset-policy-head", argv)

    def test_boolean_flags_only_when_true(self) -> None:
        """False 不追加任何参数（否则命令行里会出现“关”的歧义写法）。"""
        argv, _params, _total, _counts = self.runner._build_train(self.payload(
            evalBeforeTrain=False, resetPolicyHead=False))
        self.assertNotIn("--eval-before-train", argv)
        self.assertNotIn("--reset-policy-head", argv)
        argv_on, _params, _total, _counts = self.runner._build_train(self.payload(
            evalBeforeTrain=True, resetPolicyHead=True))
        self.assertIn("--eval-before-train", argv_on)
        self.assertIn("--reset-policy-head", argv_on)

    def test_params_and_run_json_record_experiment(self) -> None:
        """响应与 run.json 都要留下这次对照实验的口径。"""
        _argv, params, _total, _counts = self.runner._build_train(self.payload(
            algo="reinforce", bcEpisodes=2, victory=60))
        exp = params["experiment"]
        self.assertEqual(exp["algo"], "reinforce")
        self.assertEqual(exp["bcEpisodes"], 2)
        self.assertEqual(exp["victory"], 60.0)
        self.assertEqual(exp["initPolicyField"], "")
        self.assertIsNone(exp["initPolicy"])
        meta = json.loads((Path(params["runDir"]) / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["experiment"]["algo"], "reinforce")
        self.assertEqual(meta["experiment"]["bcEpisodes"], 2)

    # --- 初始模型 -------------------------------------------------------
    def test_init_policy_model_id_resolved(self) -> None:
        policy = self.make_model("exp-a")
        argv, params, _total, _counts = self.runner._build_train(self.payload(initPolicy="exp-a"))
        self.assertEqual(Path(self.arg(argv, "--init-policy")), policy.resolve())
        self.assertEqual(Path(params["initPolicy"]), policy.resolve())
        self.assertEqual(params["experiment"]["initPolicyField"], "exp-a")

    def test_init_policy_path_resolved(self) -> None:
        path = self.root / "weights.npz"
        path.write_bytes(b"npz")
        argv, params, _total, _counts = self.runner._build_train(self.payload(initPolicy=str(path)))
        self.assertEqual(Path(self.arg(argv, "--init-policy")), path.resolve())
        self.assertEqual(Path(params["initPolicy"]), path.resolve())

    def test_init_policy_missing_path_rejected(self) -> None:
        with self.assertRaises(BadRequest) as ctx:
            self.runner._build_train(self.payload(initPolicy=str(self.root / "nope.npz")))
        self.assertIn("initPolicy", str(ctx.exception))

    def test_init_policy_model_without_policy_rejected(self) -> None:
        """模型 id 存在但没有 policy.npz：400，而不是等训练进程启动才崩。"""
        self.make_model("exp-empty", policy=False)
        with self.assertRaises(BadRequest) as ctx:
            self.runner._build_train(self.payload(initPolicy="exp-empty"))
        self.assertIn("policy.npz", str(ctx.exception))

    def test_init_policy_overrides_registry_start(self) -> None:
        """显式初始模型优先于 modelId 解析出的起点（对照实验的常见用法）。"""
        override = self.root / "override.npz"
        override.write_bytes(b"npz")
        argv, _params, _total, _counts = self.runner._build_train(
            self.payload(modelId="default", initPolicy=str(override))
        )
        self.assertEqual(argv.count("--init-policy"), 1)
        self.assertEqual(Path(self.arg(argv, "--init-policy")), override.resolve())

    def test_multi_accepts_init_policy_path(self) -> None:
        """multi_train 也有 --init-policy：两边对「初始模型」的支持保持一致。"""
        path = self.root / "multi-start.npz"
        path.write_bytes(b"npz")
        argv, params, _total, _counts = self.runner._build_multi({
            "pairs": 1, "usersPrefix": "Bot_P", "portsBase": 8790,
            "plan": [{"opponentMode": "idle", "episodes": 1}],
            "initPolicy": str(path),
        })
        self.assertEqual(Path(self.arg(argv, "--init-policy")), path.resolve())
        self.assertEqual(Path(params["initPolicy"]), path.resolve())

    # --- 端口 / 用户 ----------------------------------------------------
    def test_ports_range_rejected(self) -> None:
        for port in (0, 65536, -1):
            with self.subTest(port=port):
                with self.assertRaises(BadRequest):
                    self.runner._build_train(self.payload(ports=[port, 8771]))

    def test_ports_duplicate_rejected(self) -> None:
        with self.assertRaises(BadRequest) as ctx:
            self.runner._build_train(self.payload(ports=[8770, 8770]))
        self.assertIn("重复", str(ctx.exception))

    def test_ports_count_mismatch_rejected(self) -> None:
        with self.assertRaises(BadRequest) as ctx:
            self.runner._build_train(self.payload(ports=[8770, 8771, 8772]))
        self.assertIn("ports", str(ctx.exception))
        with self.assertRaises(BadRequest):
            self.runner._build_train(self.payload(ports=[8770], users=["Bot_T1", "Bot_T2"]))

    def test_users_require_two_and_non_empty(self) -> None:
        with self.assertRaises(BadRequest):
            self.runner._build_train(self.payload(users=["Bot_T1"]))
        with self.assertRaises(BadRequest):
            self.runner._build_train(self.payload(users=["Bot_T1", ""]))

    # --- 枚举 / 数值 / 交叉规则 -----------------------------------------
    def test_action_space_v2_with_bc_rejected(self) -> None:
        with self.assertRaises(BadRequest) as ctx:
            self.runner._build_train(self.payload(actionSpace="v2", bcEpisodes=4))
        message = str(ctx.exception)
        self.assertIn("v2", message)
        self.assertIn("BC", message)
        # v2 + 0 局 BC 是合法的
        argv, _params, _total, _counts = self.runner._build_train(
            self.payload(actionSpace="v2", bcEpisodes=0)
        )
        self.assertEqual(self.arg(argv, "--action-space"), "v2")

    def test_enum_fields_rejected(self) -> None:
        for extra, field in (({"algo": "dqn"}, "algo"),
                             ({"actionSpace": "v3"}, "actionSpace")):
            with self.subTest(field=field):
                with self.assertRaises(BadRequest) as ctx:
                    self.runner._build_train(self.payload(**extra))
                self.assertIn(field, str(ctx.exception))

    def test_numeric_ranges_rejected(self) -> None:
        """越界一律 400，错误信息要同时带字段名与合法范围。"""
        cases = (
            ({"lr": 0.5}, "lr", "1e-06"),
            ({"lr": 1e-9}, "lr", "1e-06"),
            ({"victory": -1}, "victory", "1000"),
            ({"victory": 1001}, "victory", "1000"),
            ({"targetEntropy": 6}, "targetEntropy", "5"),
            ({"entropyCoef": 11}, "entropyCoef", "10"),
            ({"seed": -1}, "seed", "2147483647"),
            ({"bcEpisodes": 101}, "bcEpisodes", "100"),
            ({"bcEpisodes": -1}, "bcEpisodes", "0"),
        )
        for extra, field, bound in cases:
            with self.subTest(field=field, extra=extra):
                with self.assertRaises(BadRequest) as ctx:
                    self.runner._build_train(self.payload(**extra))
                message = str(ctx.exception)
                self.assertIn(field, message)
                self.assertIn(bound, message)

    # --- 多场地 ---------------------------------------------------------
    def test_multi_rejects_train_only_fields(self) -> None:
        with self.assertRaises(BadRequest) as ctx:
            self.runner._build_multi({
                "pairs": 1, "usersPrefix": "Bot_P", "portsBase": 8790,
                "plan": [{"opponentMode": "idle", "episodes": 1}],
                "algo": "reinforce",
            })
        self.assertIn("多场地", str(ctx.exception))

    def test_multi_accepts_default_values(self) -> None:
        """前端会把默认值一起发上来：照抄默认值不应被误拒。"""
        argv, _params, _total, _counts = self.runner._build_multi({
            "pairs": 1, "usersPrefix": "Bot_P", "portsBase": 8790,
            "plan": [{"opponentMode": "idle", "episodes": 1}],
            "algo": "ppo", "actionSpace": "v1", "bcEpisodes": 0,
            "evalBeforeTrain": False, "resetPolicyHead": False,
        })
        self.assertNotIn("--algo", argv)


if __name__ == "__main__":
    unittest.main()
