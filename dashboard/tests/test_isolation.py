"""测试隔离：任何测试都不得写入真实的 ``dashboard/var``。

偶发失败的历史怀疑是「测试与正在运行的 8080 共享 var 目录」。这里做两件事：
1. 断言所有测试用的 ``DashboardConfig.var_dir`` 都在临时目录里；
2. 在临时实例上跑一遍会落盘的写操作（脚本表、reward 预设、训练预设、失败 run.log），
   前后对比真实 ``dashboard/var`` 下**本工具会写的那些文件**，要求零变化。

（不比较 `out.log` / `err.log` / `logs/*`：正在运行的 8080 会持续追加它们。）
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

DASHBOARD_DIR = Path(__file__).resolve().parents[1]
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

from app import DashboardApp  # noqa: E402
from config import DashboardConfig  # noqa: E402
from training_runner import TrainingJob  # noqa: E402

REAL_VAR = DASHBOARD_DIR / "var"

#: 本工具会写、且不属于「实时日志」的相对路径（前缀匹配）。
WATCHED = (
    "scripts.json",
    "scripts-trash",
    "reward-presets",
    "presets",
    "presets-trash",
    "managed.json",
    "launcher.pid",
)


def _snapshot(root: Path) -> dict[str, tuple[int, int]]:
    """记录关心文件的 (mtime_ns, size)；目录本身也记一条，便于发现新增子项。"""
    result: dict[str, tuple[int, int]] = {}
    if not root.is_dir():
        return result
    for prefix in WATCHED:
        target = root / prefix
        if target.is_file():
            stat = target.stat()
            result[prefix] = (stat.st_mtime_ns, stat.st_size)
        elif target.is_dir():
            for path in sorted(target.rglob("*")):
                if not path.is_file():
                    continue
                stat = path.stat()
                result[str(path.relative_to(root))] = (stat.st_mtime_ns, stat.st_size)
    return result


class VarIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # 真实目录的快照在导入阶段就取一次，作为基线
        cls.baseline = _snapshot(REAL_VAR)

    @classmethod
    def tearDownClass(cls) -> None:
        pass

    def test_real_var_untouched_by_temp_app(self) -> None:
        before = _snapshot(REAL_VAR)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = DashboardConfig(
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
                mc_port=25987,
            )
            self.assertNotEqual(cfg.var_dir, REAL_VAR, "测试必须使用临时 var_dir")
            app = DashboardApp(cfg)
            try:
                app.scripts.ensure()
                app.scripts.save({"id": "iso-check", "name": "Iso", "color": "&a"})
                app.reward_presets.save("iso-check", {
                    "format": "gladiator.reward-preset/v1", "name": "Iso",
                    "base": {"kill": 1.0}, "terms": [],
                })
                app.presets.save({"name": "Iso", "plan": [{"opponentMode": "idle", "episodes": 1}]})
                job = TrainingJob(id="training", kind="train", params={}, total_episodes=1,
                                  run_dir=str(root / "artifacts" / "runs" / "iso-run"))
                app.training._fail_before_spawn(job, "隔离测试")
                # 临时目录里应真的产生了这些文件（证明写操作被真正执行过）
                self.assertTrue((cfg.var_dir / "scripts.json").is_file())
                self.assertTrue((cfg.var_dir / "reward-presets" / "iso-check.json").is_file())
                self.assertTrue((root / "artifacts" / "runs" / "iso-run" / "run.log").is_file())
            finally:
                app.close()
        after = _snapshot(REAL_VAR)
        self.assertEqual(before, after, "测试不得改动真实 dashboard/var")

    def test_all_test_configs_use_temp_var(self) -> None:
        """静态检查：测试代码里构造 DashboardConfig 时都应显式给 var_dir。"""
        offenders: list[str] = []
        for path in sorted((DASHBOARD_DIR / "tests").glob("test_*.py")):
            text = path.read_text(encoding="utf-8")
            for index, line in enumerate(text.splitlines(), start=1):
                if "DashboardConfig(" not in line:
                    continue
                # 单行构造（如 DashboardConfig(java_exe=...)）不落盘，跳过
                if line.rstrip().endswith(")"):
                    continue
                block: list[str] = []
                for follow in text.splitlines()[index:index + 20]:
                    block.append(follow)
                    if follow.strip().startswith(")") and "var_dir" not in "\n".join(block):
                        offenders.append(f"{path.name}:{index}")
                        break
        self.assertEqual(offenders, [], f"这些构造没有临时 var_dir: {offenders}")


if __name__ == "__main__":
    unittest.main()
