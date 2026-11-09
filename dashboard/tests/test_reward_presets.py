"""Reward 预设（``gladiator.reward-preset/v1``）与多场地评估回合限制的测试。

覆盖：
- 校验：未知 base/条件键、非法 unit、重复 term id、非有限数字、上限；
- 与训练侧 schema 的常量等价性（防止 Dashboard 与 training 规则漂移）；
- ``RewardPresetStore``：首次写 default、覆盖前备份、软删除、id 校验；
- HTTP：``/api/reward-presets`` 的 GET / GET <id> / POST / DELETE；
- 训练接线：``--reward-preset`` 绝对路径、params 回显、run.json 记录；
- 修复 A：``_build_multi`` 拒绝 evalEpochs > 0；
- 修复 B：训练预设 fields 的类型/范围校验与覆盖备份。

运行：
    cd D:\\cbot-gladiator\\dashboard
    ..\\training\\.venv\\Scripts\\python.exe -m unittest discover -s tests -q
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any
from urllib import error, request

DASHBOARD_DIR = Path(__file__).resolve().parents[1]
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

import reward_presets  # noqa: E402
from app import DashboardApp, make_server  # noqa: E402
from config import DashboardConfig  # noqa: E402
from errors import BadRequest, NotFound  # noqa: E402
from presets import PresetStore  # noqa: E402
from reward_presets import RewardPresetStore, normalize_preset  # noqa: E402

SCHEMA = "gladiator.reward-preset/v1"


def _valid_preset(**overrides: Any) -> dict:
    preset = {
        "format": SCHEMA,
        "name": "测试预设",
        "description": "单测用",
        "base": {"kill": 12.0, "still": -0.2},
        "terms": [
            {"id": "threat", "weight": 0.3, "unit": "second",
             "when": {"distanceMin": 2.8, "distanceMax": 4.2}},
            {"id": "punishBlock", "weight": -0.2, "unit": "step",
             "when": {"actionIn": ["block"], "distanceMin": 4.2}},
        ],
    }
    preset.update(overrides)
    return preset


class SchemaParityTests(unittest.TestCase):
    """Dashboard 的常量必须来自训练侧 schema，不能各写一份。"""

    def test_reference_file_is_used(self) -> None:
        self.assertTrue(
            reward_presets.SCHEMA_FILE.is_file(),
            f"找不到训练侧 schema: {reward_presets.SCHEMA_FILE}",
        )
        self.assertIsNotNone(reward_presets._REF, "未能加载训练侧 schema")

    def test_constants_match_training_source(self) -> None:
        spec = importlib.util.spec_from_file_location("_ref_for_test", reward_presets.SCHEMA_FILE)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        # dataclass 装饰期要回查 sys.modules[cls.__module__]
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)  # type: ignore[union-attr]
        finally:
            sys.modules.pop(spec.name, None)

        self.assertEqual(reward_presets.SCHEMA, module.SCHEMA)
        self.assertEqual(list(reward_presets.BASE_ORDER), list(module.BASE_FIELDS))
        self.assertEqual(dict(reward_presets.DEFAULT_BASE), dict(module.DEFAULT_BASE))
        self.assertEqual(set(reward_presets.CONDITION_KEYS), set(module.CONDITION_KEYS))
        self.assertEqual(set(reward_presets.UNITS), set(module.UNITS))


class NormalizeTests(unittest.TestCase):
    def test_valid_preset_normalized(self) -> None:
        preset = normalize_preset(_valid_preset())
        self.assertEqual(preset["format"], SCHEMA)
        self.assertEqual(preset["base"]["kill"], 12.0)
        # 未提供的基础权重补齐默认值
        self.assertEqual(preset["base"]["death"], reward_presets.DEFAULT_BASE["death"])
        self.assertEqual(preset["terms"][0]["unit"], "second")
        self.assertEqual(preset["terms"][1]["when"]["actionIn"], ["block"])

    def test_unit_defaults_to_second(self) -> None:
        preset = normalize_preset(_valid_preset(terms=[{"id": "t", "weight": 1, "when": {}}]))
        self.assertEqual(preset["terms"][0]["unit"], "second")

    def test_unknown_base_key_rejected(self) -> None:
        with self.assertRaises(BadRequest) as ctx:
            normalize_preset(_valid_preset(base={"walking": -1.0}))
        self.assertIn("未知的基础权重", str(ctx.exception))

    def test_unknown_condition_rejected(self) -> None:
        with self.assertRaises(BadRequest) as ctx:
            normalize_preset(_valid_preset(terms=[{"id": "t", "weight": 1, "when": {"onGround": True}}]))
        self.assertIn("不支持的条件", str(ctx.exception))

    def test_bad_unit_rejected(self) -> None:
        with self.assertRaises(BadRequest) as ctx:
            normalize_preset(_valid_preset(terms=[{"id": "t", "weight": 1, "unit": "minute", "when": {}}]))
        self.assertIn("unit", str(ctx.exception))

    def test_duplicate_term_id_rejected(self) -> None:
        terms = [{"id": "same", "weight": 1, "when": {}}, {"id": "same", "weight": 2, "when": {}}]
        with self.assertRaises(BadRequest) as ctx:
            normalize_preset(_valid_preset(terms=terms))
        self.assertIn("重复", str(ctx.exception))

    def test_non_finite_numbers_rejected(self) -> None:
        with self.assertRaises(BadRequest):
            normalize_preset(_valid_preset(base={"kill": float("nan")}))
        with self.assertRaises(BadRequest):
            normalize_preset(_valid_preset(base={"kill": float("inf")}))
        with self.assertRaises(BadRequest):
            normalize_preset(_valid_preset(terms=[{"id": "t", "weight": float("inf"), "when": {}}]))

    def test_bool_weight_rejected(self) -> None:
        with self.assertRaises(BadRequest):
            normalize_preset(_valid_preset(base={"kill": True}))

    def test_missing_name_rejected(self) -> None:
        with self.assertRaises(BadRequest):
            normalize_preset({"base": {}, "terms": []})

    def test_too_many_terms_rejected(self) -> None:
        terms = [{"id": f"t{i}", "weight": 1, "when": {}} for i in range(reward_presets.MAX_TERMS + 1)]
        with self.assertRaises(BadRequest) as ctx:
            normalize_preset(_valid_preset(terms=terms))
        self.assertIn("最多", str(ctx.exception))

    def test_action_in_requires_non_empty_list(self) -> None:
        with self.assertRaises(BadRequest):
            normalize_preset(_valid_preset(terms=[{"id": "t", "weight": 1, "when": {"actionIn": []}}]))

    def test_attacked_requires_bool(self) -> None:
        with self.assertRaises(BadRequest):
            normalize_preset(_valid_preset(terms=[{"id": "t", "weight": 1, "when": {"attacked": "yes"}}]))

    def test_wrong_format_rejected(self) -> None:
        with self.assertRaises(BadRequest):
            normalize_preset(_valid_preset(format="gladiator.training-preset/v1"))


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "reward-presets"
        self.store = RewardPresetStore(self.root)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_ensure_writes_default(self) -> None:
        self.store.ensure()
        self.assertTrue((self.root / "default.json").is_file())
        items = self.store.list()
        self.assertEqual([item["id"] for item in items], ["default"])
        self.assertTrue(items[0]["builtin"])

    def test_summary_fields(self) -> None:
        self.store.save("p1", _valid_preset())
        item = self.store.list()[0]
        self.assertEqual(item["id"], "p1")
        self.assertEqual(item["termsCount"], 2)
        self.assertEqual(item["baseSummary"]["kill"], 12.0)
        self.assertIn("damageTaken", item["baseSummary"])

    def test_overwrite_backs_up_previous_version(self) -> None:
        self.store.save("p1", _valid_preset())
        self.store.save("p1", _valid_preset(name="改过"))
        backups = list((self.root / "trash").glob("p1-overwrite-*.json"))
        self.assertEqual(len(backups), 1, "覆盖保存前应备份旧版本")
        old = json.loads(backups[0].read_text(encoding="utf-8"))
        self.assertEqual(old["name"], "测试预设")
        self.assertEqual(self.store.get("p1")["name"], "改过")

    def test_delete_moves_to_trash(self) -> None:
        self.store.save("p2", _valid_preset())
        self.assertEqual(self.store.delete("p2"), "p2")
        self.assertFalse((self.root / "p2.json").exists())
        backups = list((self.root / "trash").glob("p2-deleted-*.json"))
        self.assertEqual(len(backups), 1)
        with self.assertRaises(NotFound):
            self.store.get("p2")

    def test_path_for_returns_absolute_file(self) -> None:
        self.store.save("p3", _valid_preset())
        path = Path(self.store.path_for("p3"))
        self.assertTrue(path.is_absolute())
        self.assertTrue(path.is_file())
        with self.assertRaises(NotFound):
            self.store.path_for("missing")

    def test_invalid_id_rejected(self) -> None:
        with self.assertRaises(BadRequest):
            self.store.save("../evil", _valid_preset())
        with self.assertRaises(BadRequest):
            self.store.delete("a/b")

    def test_broken_file_skipped_in_list(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "broken.json").write_text("{", encoding="utf-8")
        self.store.save("ok", _valid_preset())
        self.assertEqual([item["id"] for item in self.store.list()], ["ok"])


class _AppFixture(unittest.TestCase):
    allow_process_control = False
    mc_port = 25991

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.root = root
        cls.cfg = DashboardConfig(
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
            allow_process_control=cls.allow_process_control,
            scan_external=False,
            server_jar="missing.jar",
            mc_port=cls.mc_port,
        )
        cls.app = DashboardApp(cls.cfg)
        cls.httpd = make_server(cls.app, "127.0.0.1", 0)
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.app.close()
        cls.thread.join(timeout=5)
        cls.tmp.cleanup()

    def _request(self, path: str, method: str = "GET", payload: Any = None) -> tuple[int, bytes]:
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = request.Request(self.base + path, data=data, method=method, headers=headers)
        try:
            with request.urlopen(req, timeout=15) as resp:
                return resp.status, resp.read()
        except error.HTTPError as exc:
            return exc.code, exc.read()

    def _json(self, path: str, method: str = "GET", payload: Any = None) -> dict:
        status, body = self._request(path, method, payload)
        self.assertEqual(status, 200, body.decode("utf-8", "replace"))
        return json.loads(body.decode("utf-8"))


class RewardPresetHttpTests(_AppFixture):
    def test_list_ships_default_and_schema(self) -> None:
        data = self._json("/api/reward-presets")
        # 本类共用一个临时 var 目录，其他用例可能已建过预设；这里只要求 default 在内。
        self.assertIn("default", [item["id"] for item in data["presets"]])
        keys = [item["key"] for item in data["schema"]["baseKeys"]]
        self.assertIn("still", keys)
        self.assertIn("inactive", keys)
        self.assertEqual(set(data["schema"]["units"]), {"second", "step"})

    def test_crud_round_trip(self) -> None:
        created = self._json("/api/reward-presets", "POST", {"id": "http-1", "preset": _valid_preset()})
        self.assertEqual(created["id"], "http-1")
        self.assertEqual(created["preset"]["terms"][0]["id"], "threat")

        detail = self._json("/api/reward-presets/http-1")
        self.assertEqual(detail["preset"]["name"], "测试预设")
        self.assertTrue(Path(detail["path"]).is_file())

        overwritten = self._json("/api/reward-presets", "POST", {
            "id": "http-1", "preset": _valid_preset(name="覆盖后", terms=[]),
        })
        self.assertEqual(overwritten["preset"]["terms"], [])
        backups = list((self.root / "var" / "reward-presets" / "trash").glob("http-1-overwrite-*.json"))
        self.assertEqual(len(backups), 1)

        status, body = self._request("/api/reward-presets/http-1", "DELETE")
        self.assertEqual(status, 200, body.decode("utf-8", "replace"))
        self.assertEqual(json.loads(body)["deleted"], "http-1")
        status, _body = self._request("/api/reward-presets/http-1")
        self.assertEqual(status, 404)

    def test_flat_body_accepted(self) -> None:
        payload = dict(_valid_preset(name="扁平"))
        payload["id"] = "http-flat"
        saved = self._json("/api/reward-presets", "POST", payload)
        self.assertEqual(saved["preset"]["name"], "扁平")

    def test_invalid_preset_returns_400(self) -> None:
        status, body = self._request("/api/reward-presets", "POST", {
            "id": "bad", "preset": _valid_preset(base={"walking": -1}),
        })
        self.assertEqual(status, 400)
        self.assertIn("未知的基础权重", json.loads(body.decode("utf-8"))["error"])

    def test_missing_preset_returns_404(self) -> None:
        status, _body = self._request("/api/reward-presets/nope")
        self.assertEqual(status, 404)

    def test_options_expose_reward_presets(self) -> None:
        options = self._json("/api/training/options")["options"]
        self.assertIn("default", [item["id"] for item in options["rewardPresets"]])
        self.assertEqual(options["defaults"]["rewardPreset"], "default")
        self.assertTrue(options["rewardPresetSchema"]["baseKeys"])


class TrainingWiringTests(_AppFixture):
    mc_port = 25992

    def _arg(self, argv: list[str], flag: str) -> str:
        return argv[argv.index(flag) + 1]

    def _save_preset(self, ident: str = "t1") -> Path:
        self.app.reward_presets.save(ident, _valid_preset())
        return Path(self.app.reward_presets.path_for(ident))

    def test_train_passes_reward_preset_path(self) -> None:
        path = self._save_preset()
        argv, params, total, _counts = self.app.training._build_train({
            "users": ["Bot_T1", "Bot_T2"],
            "ports": [8770, 8771],
            "plan": [{"opponentMode": "scripted", "episodes": 3}],
            "epochs": 2,
            "evalEpochs": 1,
            "rewardPreset": "t1",
        })
        self.assertEqual(self._arg(argv, "--reward-preset"), str(path))
        self.assertEqual(params["rewardPreset"]["id"], "t1")
        self.assertEqual(params["rewardPreset"]["name"], "测试预设")
        # evalRounds = 每回合对局数；evalEpisodes = 评估总局数
        self.assertEqual(params["evalRounds"], 3)
        self.assertEqual(params["evalEpisodes"], 3)
        self.assertEqual(total, 6)

    def test_train_without_reward_preset_omits_flag(self) -> None:
        argv, params, _total, _counts = self.app.training._build_train({
            "users": ["Bot_T1", "Bot_T2"],
            "ports": [8772, 8773],
            "plan": [{"opponentMode": "idle", "episodes": 1}],
        })
        self.assertNotIn("--reward-preset", argv)
        self.assertIsNone(params["rewardPreset"])

    def test_unknown_reward_preset_rejected(self) -> None:
        with self.assertRaises(BadRequest):
            self.app.training._build_train({
                "users": ["Bot_T1", "Bot_T2"],
                "ports": [8774, 8775],
                "plan": [{"opponentMode": "idle", "episodes": 1}],
                "rewardPreset": "does-not-exist",
            })

    def test_run_json_records_reward_preset(self) -> None:
        self._save_preset("t2")
        _argv, params, _total, _counts = self.app.training._build_train({
            "users": ["Bot_T1", "Bot_T2"],
            "ports": [8776, 8777],
            "plan": [{"opponentMode": "idle", "episodes": 1}],
            "rewardPreset": "t2",
        })
        meta = json.loads((Path(params["runDir"]) / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["rewardPreset"]["id"], "t2")
        self.assertEqual(meta["evalRounds"], 1)

    def test_multi_rejects_eval_epochs(self) -> None:
        with self.assertRaises(BadRequest) as ctx:
            self.app.training._build_multi({
                "pairs": 1,
                "usersPrefix": "Bot_P",
                "portsBase": 8790,
                "plan": [{"opponentMode": "idle", "episodes": 1}],
                "evalEpochs": 1,
            })
        self.assertIn("多场地训练不支持评估回合", str(ctx.exception))

    def test_multi_accepts_zero_eval_epochs(self) -> None:
        argv, params, _total, _counts = self.app.training._build_multi({
            "pairs": 1,
            "usersPrefix": "Bot_P",
            "portsBase": 8792,
            "plan": [{"opponentMode": "idle", "episodes": 2}],
            "evalEpochs": 0,
        })
        self.assertNotIn("--eval-epochs", argv)
        self.assertEqual(params["evalEpochs"], 0)
        self.assertEqual(params["evalRounds"], 0)
        self.assertFalse(params["evalSupported"])

    def test_multi_passes_reward_preset_path(self) -> None:
        """multi_train 已注册 --reward-preset（G-620 修复）：与 train 一样传绝对路径。"""
        path = self._save_preset("t3")
        argv, params, _total, _counts = self.app.training._build_multi({
            "pairs": 2,
            "usersPrefix": "Bot_P",
            "portsBase": 8794,
            "plan": [{"opponentMode": "rusher", "episodes": 2}],
            "rewardPreset": "t3",
        })
        self.assertEqual(self._arg(argv, "--reward-preset"), str(path))
        self.assertEqual(params["rewardPreset"]["id"], "t3")
        self.assertTrue(params["rewardPreset"]["applied"])
        meta = json.loads((Path(params["runDir"]) / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["rewardPreset"]["id"], "t3")


class TrainingPresetFieldTests(unittest.TestCase):
    """修复 B：训练预设 fields 的类型/范围校验 + 覆盖备份 + 命名统一。"""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = PresetStore(Path(self.tmp.name) / "presets")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_field_ranges_enforced(self) -> None:
        cases = [
            {"epochs": 0},
            {"epochs": 201},
            {"evalEpochs": 51},
            {"evalEpochs": -1},
            {"maxSteps": 0},
            {"maxSteps": 5001},
            {"seed": -1},
            {"spacing": 0.5},
            {"spacing": 21},
            {"prepSeconds": 31},
            {"center": "1,2"},
            {"autoSpawn": "maybe"},
            {"epochs": "abc"},
        ]
        for fields in cases:
            with self.subTest(fields=fields), self.assertRaises(BadRequest):
                self.store.save({"name": "bad", "plan": [{"opponentMode": "idle", "episodes": 1}], "fields": fields})

    def test_valid_fields_normalized(self) -> None:
        preset = self.store.save({
            "name": "ok",
            "plan": [{"opponentMode": "idle", "episodes": 1}],
            "fields": {"epochs": 3, "evalEpochs": 2, "center": "1, 4.5 ,2", "autoSpawn": "false",
                       "maxSteps": 200, "seed": "9", "spacing": "6.5", "prepSeconds": 2},
        })
        self.assertEqual(preset["fields"]["epochs"], 3)
        self.assertEqual(preset["fields"]["evalEpochs"], 2)
        self.assertEqual(preset["fields"]["center"], "1,4.5,2")
        self.assertIs(preset["fields"]["autoSpawn"], False)
        self.assertEqual(preset["fields"]["seed"], 9)
        self.assertEqual(preset["fields"]["spacing"], 6.5)

    def test_legacy_eval_episodes_maps_to_eval_epochs(self) -> None:
        preset = self.store.save({
            "name": "legacy",
            "plan": [{"opponentMode": "idle", "episodes": 1}],
            "fields": {"evalEpisodes": 4},
        })
        self.assertEqual(preset["fields"], {"evalEpochs": 4})
        path = Path(self.tmp.name) / "presets" / f"{preset['id']}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn("evalEpochs", data["fields"])
        self.assertNotIn("evalEpisodes", data["fields"])

    def test_overwrite_backs_up_previous_file(self) -> None:
        first = self.store.save({"name": "P", "plan": [{"opponentMode": "idle", "episodes": 1}]})
        self.store.save({"name": "P", "plan": [{"opponentMode": "idle", "episodes": 2}]})
        backups = list((Path(self.tmp.name) / "presets-trash").glob(f"{first['id']}-overwrite-*.json"))
        self.assertEqual(len(backups), 1, "同名覆盖前应备份旧版本")
        old = json.loads(backups[0].read_text(encoding="utf-8"))
        self.assertEqual(old["plan"][0]["episodes"], 1)


class RewardIdCharsetTests(unittest.TestCase):
    """P-4：reward 预设 id 统一为 ^[a-z0-9][a-z0-9_-]{0,31}$。"""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "reward-presets"
        self.store = RewardPresetStore(self.root)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_valid_ids(self) -> None:
        for ident in ("default", "rush-v1", "a", "0abc", "x" * 32):
            with self.subTest(ident=ident):
                self.store.save(ident, _valid_preset())
                self.assertEqual(self.store.get(ident)["name"], "测试预设")

    def test_invalid_ids_rejected(self) -> None:
        for ident in ("Rush", "-lead", "_under", "a" * 33, "with space", "dot.name", "", "../x", "中文"):
            with self.subTest(ident=ident), self.assertRaises(BadRequest) as ctx:
                self.store.save(ident, _valid_preset())
            self.assertIn("id", str(ctx.exception))

    def test_get_invalid_id_gives_chinese_error(self) -> None:
        with self.assertRaises(BadRequest) as ctx:
            self.store.get("Bad-Id")
        self.assertIn("32 位", str(ctx.exception))

    def test_list_skips_illegal_file_names(self) -> None:
        self.store.save("good-one", _valid_preset())
        (self.root / "Bad Name.json").write_text(
            json.dumps(_valid_preset(), ensure_ascii=False), encoding="utf-8")
        ids = [item["id"] for item in self.store.list()]
        self.assertEqual(ids, ["good-one"])


if __name__ == "__main__":
    unittest.main()
