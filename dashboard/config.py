"""Dashboard 配置。

所有路径/端口都可通过环境变量覆盖，方便测试与换机器：

    DASHBOARD_PORT         监听端口（默认 8080，只允许回环地址）
    DASHBOARD_ARTIFACTS    训练产物目录（默认 training/artifacts）
    DASHBOARD_VAR          Dashboard 自己的日志/PID 审计目录（默认 dashboard/var）
    DASHBOARD_PYTHON       训练用的 Python（默认 training/.venv 里的解释器）
    GLADIATOR_JAVA / DASHBOARD_JAVA   启动 Paper 用的 java.exe
    DASHBOARD_NODE         node 可执行文件
    DASHBOARD_ALLOW_PROCESS_CONTROL  0 时只读（禁止启动/停止任何子进程）

Java 探测顺序（不做任何本机路径硬编码，见 :func:`discover_java`）：

    dashboard/config.json 的 ``javaPath`` → ``GLADIATOR_JAVA`` / ``DASHBOARD_JAVA``
    → ``JAVA_HOME\\bin\\java.exe`` → ``PATH`` 里的 ``java``
    → 常见安装根（Eclipse Adoptium / Java / Zulu / Microsoft）下的 ``jdk*8*``
    → 都没有则返回 ``None``（启动服务器时给中文提示）
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(os.environ.get("GLADIATOR_ROOT", Path(__file__).resolve().parents[1])).resolve()

#: 可选的本机配置文件（不入库，见 .gitignore）；目前只用于 ``javaPath`` 等本机路径。
CONFIG_FILE = Path(__file__).resolve().parent / "config.json"

#: 常见 JDK 安装根；按顺序扫描其中的 ``jdk*8*`` 目录。不含任何个人目录。
COMMON_JAVA_ROOTS: tuple[str, ...] = (
    r"C:\Program Files\Eclipse Adoptium",
    r"C:\Program Files\Java",
    r"C:\Program Files\Zulu",
    r"C:\Program Files\Microsoft",
    r"C:\Program Files (x86)\Eclipse Adoptium",
    r"C:\Program Files (x86)\Java",
    r"C:\Program Files (x86)\Zulu",
    r"C:\Program Files (x86)\Microsoft",
)

#: ``jdk*8*`` 的目录名匹配（Adoptium ``jdk8u402-b06`` / ``jdk-8u402-b06`` /
#: Zulu ``zulu8.78.0.19-ca-jdk8.0.412-win_x64``）。
_JAVA8_DIR_MARKERS = ("jdk8", "jdk-8", "jdk1.8", "zulu8", "jre8", "jre-8")


def load_config_file(path: Path | None = None) -> dict:
    """读取可选的本机配置；不存在或损坏时返回空 dict（不抛异常）。"""
    target = Path(path) if path is not None else CONFIG_FILE
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _looks_like_java8_dir(name: str) -> bool:
    lowered = name.strip().lower()
    return any(marker in lowered for marker in _JAVA8_DIR_MARKERS)


#: Java 版本探测结果缓存：key = (规范化路径, mtime_ns, size) → major。
_VERSION_CACHE: dict[tuple[str, int, int], int | None] = {}

#: ``java -version`` 的输出解析：``java version "1.8.0_492"`` / ``openjdk version "17.0.19"``。
_VERSION_RE = re.compile(r'version\s+"(\d+)(?:\.(\d+))?')


class JavaDiscoveryError(RuntimeError):
    """显式配置的 Java 不可用（不是 Java 8 或无法执行）时抛出。"""


@dataclass(frozen=True)
class JavaChoice:
    """一次探测的结论：可用路径 + 版本 + 给人工看的提示。"""

    path: str | None
    version: int | None
    hint: str = ""
    #: 扫描过程中被拒的候选（路径, 检测到的 major），用于排障。
    rejected: tuple[tuple[str, int | None], ...] = ()


def clear_java_version_cache() -> None:
    _VERSION_CACHE.clear()


def _run_java_version(path: str) -> str:
    """执行 ``java -version`` 并返回 stderr+stdout（版本信息在 stderr）。"""
    try:
        completed = subprocess.run(
            [path, "-version"], capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return ""
    return (completed.stderr or "") + (completed.stdout or "")


def parse_java_version(output: str) -> int | None:
    """从版本输出里取 major：``1.8.0_492`` → 8，``17.0.19`` → 17。"""
    match = _VERSION_RE.search(output or "")
    if match is None:
        return None
    first = int(match.group(1))
    if first == 1 and match.group(2):
        return int(match.group(2))
    return first


def java_major_version(path: str | None, *, runner=None, cache: dict | None = None,
                       use_cache: bool = True) -> int | None:
    """返回该 java 可执行文件的 major 版本；不可执行/无法解析时返回 ``None``。

    - ``runner`` 注入：``runner(path) -> str``（默认执行 ``java -version``），测试不依赖本机安装；
    - 缓存键为 ``(路径, mtime_ns, size)``，文件被替换后自动失效。
    """
    if not path:
        return None
    target = str(Path(path))
    store = _VERSION_CACHE if cache is None else cache
    key: tuple[str, int, int] | None = None
    if use_cache:
        try:
            stat = os.stat(target)
            key = (os.path.normcase(target), int(stat.st_mtime_ns), int(stat.st_size))
        except OSError:
            key = None
        if key is not None and key in store:
            return store[key]
    probe = runner or _run_java_version
    try:
        output = probe(target) or ""
    except Exception:  # noqa: BLE001 - 探测失败按“不可用”处理
        output = ""
    version = parse_java_version(output)
    if key is not None:
        store[key] = version
    return version


def _hint_for(rejected: tuple[tuple[str, int | None], ...], explicit: str | None) -> str:
    """拼一句可操作的提示：优先点名显式配置，其次点名义最“新”的那个。"""
    versioned = [(path, ver) for path, ver in rejected if ver is not None]
    if versioned:
        path, version = versioned[0] if explicit is None else next(
            ((p, v) for p, v in versioned if str(p) == str(explicit)), versioned[0])
        return (
            f"检测到 Java {version}（{path}），Paper 1.8.8 需要 Java 8，"
            "请配置 javaPath（dashboard/config.json）或环境变量 GLADIATOR_JAVA 指向 Java 8"
        )
    return (
        "未找到 Java 8，请配置 javaPath：在 dashboard/config.json 写入 "
        '{"javaPath": "…/jdk8/bin/java.exe"}，或设置环境变量 GLADIATOR_JAVA 后重启 Dashboard'
    )


def discover_java(
    *,
    explicit: str | None = None,
    java_home: str | None = None,
    path_lookup: str | None = None,
    roots: tuple[str, ...] | list[str] | None = None,
    is_file=None,
    version_probe=None,
    required_major: int = 8,
) -> JavaChoice:
    """按优先级探测 **Java 8**，返回 :class:`JavaChoice`（path 为 None 表示没找到）。

    顺序：显式配置 → ``JAVA_HOME`` → ``PATH`` → 常见安装根下的 ``jdk*8*``。
    每个候选都必须实测 major 版本等于 8；不是 8 就继续找下一个（绝不静默使用 17）。
    """
    check = is_file or os.path.isfile
    probe = version_probe or (lambda exe: java_major_version(exe))
    rejected: list[tuple[str, int | None]] = []

    candidates: list[str] = []
    if explicit:
        candidates.append(str(Path(explicit)))
    if java_home:
        candidates.append(str(Path(java_home) / "bin" / "java.exe"))
        candidates.append(str(Path(java_home) / "bin" / "java"))
    if path_lookup:
        candidates.append(str(Path(path_lookup)))

    for root in (COMMON_JAVA_ROOTS if roots is None else roots):
        base = Path(root)
        try:
            if not base.is_dir():
                continue
            entries = sorted(base.iterdir())
        except OSError:
            continue
        for entry in entries:
            if not _looks_like_java8_dir(entry.name):
                continue
            for exe_name in ("java.exe", "java"):
                exe = entry / "bin" / exe_name
                if check(str(exe)):
                    candidates.append(str(exe))

    for candidate in candidates:
        if not check(candidate):
            continue
        version = probe(candidate)
        if version == required_major:
            return JavaChoice(path=str(Path(candidate)), version=version, rejected=tuple(rejected))
        rejected.append((candidate, version))

    return JavaChoice(path=None, version=None,
                      hint=_hint_for(tuple(rejected), explicit), rejected=tuple(rejected))


def _resolve_java() -> JavaChoice:
    """真实环境下的探测：配置 → 环境变量 → JAVA_HOME → PATH → 常见安装根（只认 Java 8）。"""
    file_cfg = load_config_file()
    explicit = (
        str(file_cfg.get("javaPath") or "").strip()
        or os.environ.get("GLADIATOR_JAVA")
        or os.environ.get("DASHBOARD_JAVA")
        or None
    )
    return discover_java(
        explicit=explicit,
        java_home=os.environ.get("JAVA_HOME") or None,
        path_lookup=shutil.which("java"),
    )


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _resolve_python() -> str:
    override = os.environ.get("DASHBOARD_PYTHON")
    if override:
        return override
    venv = ROOT / "training" / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if venv.is_file():
        return str(venv)
    return sys.executable


def _resolve_node() -> str:
    return os.environ.get("DASHBOARD_NODE") or shutil.which("node") or "node"


def _resolve_gmod_dir() -> Path:
    """.gmod 扫描目录：环境变量 > dashboard/config.json 的 gmodDir > 仓库根 models/。

    相对路径按仓库根解析；测试可注入临时目录，不依赖本机真实安装。
    """
    raw = (os.environ.get("DASHBOARD_GMOD_DIR")
           or load_config_file().get("gmodDir")
           or ROOT / "models")
    path = Path(str(raw))
    return path if path.is_absolute() else (ROOT / path).resolve()


@dataclass(frozen=True)
class DashboardConfig:
    host: str = "127.0.0.1"
    port: int = 8080
    artifacts_dir: Path = ROOT / "training" / "artifacts"
    var_dir: Path = ROOT / "dashboard" / "var"
    server_dir: Path = ROOT / "server"
    bot_runtime_dir: Path = ROOT / "bot-runtime"
    training_dir: Path = ROOT / "training"
    #: .gmod 模型包扫描目录（放文件即出现）；默认仓库根 models/。
    gmod_dir: Path = ROOT / "models"
    python_exe: str = ""
    #: 未探测到 Java 8 时为 None（启动服务器会给出中文提示，不崩溃）。
    java_exe: str | None = ""
    #: 实测 major 版本（8 / 17 / None=探测不到）。
    java_version: int | None = None
    #: 没有可用 Java 8 时的中文提示（含“检测到 Java 17”这类信息）。
    java_hint: str = ""
    node_exe: str = ""
    allow_process_control: bool = True
    rcon_host: str = "127.0.0.1"
    rcon_port: int = 25575
    rcon_password: str = "devpass"
    mc_host: str = "127.0.0.1"
    mc_port: int = 25566
    bridge_port: int = 8765
    runtime_port_min: int = 8770
    runtime_port_max: int = 8789
    scan_external: bool = True
    server_jar: str = "paper-1.8.8-445.jar"
    server_memory_mb: int = 1024

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifacts_dir", Path(self.artifacts_dir))
        object.__setattr__(self, "var_dir", Path(self.var_dir))
        object.__setattr__(self, "server_dir", Path(self.server_dir))
        object.__setattr__(self, "bot_runtime_dir", Path(self.bot_runtime_dir))
        object.__setattr__(self, "training_dir", Path(self.training_dir))
        object.__setattr__(self, "gmod_dir", Path(self.gmod_dir))
        if self.host not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("Dashboard 只允许监听回环地址（127.0.0.1 / localhost / ::1）")
        if not (1 <= int(self.port) <= 65535):
            raise ValueError(f"端口非法: {self.port}")
        if not (1024 <= int(self.runtime_port_min) <= int(self.runtime_port_max) <= 65535):
            raise ValueError(
                f"运行时端口范围非法: {self.runtime_port_min}-{self.runtime_port_max}"
            )

    # --- 派生路径 -----------------------------------------------------
    @property
    def static_dir(self) -> Path:
        return Path(__file__).resolve().parent / "static"

    @property
    def logs_dir(self) -> Path:
        return self.var_dir / "logs"

    @property
    def state_file(self) -> Path:
        return self.var_dir / "managed.json"

    @property
    def java_path(self) -> Path | None:
        return Path(self.java_exe) if self.java_exe else None

    @property
    def java_available(self) -> bool:
        path = self.java_path
        return bool(path and path.is_file())

    @property
    def java_ready(self) -> bool:
        """Paper 1.8.8 能不能用：文件存在 **且** 实测 major == 8。"""
        return self.java_available and self.java_version == 8

    @property
    def node_path(self) -> Path:
        return Path(self.node_exe)

    @property
    def server_jar_path(self) -> Path:
        return self.server_dir / self.server_jar

    def ensure_dirs(self) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def public(self) -> dict:
        """可安全暴露给前端的配置（不含密码）。"""
        return {
            "host": self.host,
            "port": self.port,
            "artifactsDir": str(self.artifacts_dir),
            "varDir": str(self.var_dir),
            "serverDir": str(self.server_dir),
            "botRuntimeDir": str(self.bot_runtime_dir),
            "trainingDir": str(self.training_dir),
            "gmodDir": str(self.gmod_dir),
            "python": self.python_exe,
            # 没有可用 Java 8 时 java 为 null，javaHint 说明原因（例如 PATH 里命中的是 17）。
            "java": self.java_exe if self.java_ready else None,
            "node": self.node_exe,
            "javaExists": self.java_available,
            "javaVersion": self.java_version,
            "javaHint": "" if self.java_ready else (
                self.java_hint
                or "未找到 Java 8，请配置 javaPath（dashboard/config.json）或环境变量 GLADIATOR_JAVA"
            ),
            "nodeExists": self.node_path.is_file() or shutil.which(self.node_exe) is not None,
            "jarExists": self.server_jar_path.is_file(),
            "serverJar": self.server_jar,
            "allowProcessControl": self.allow_process_control,
            "mcHost": self.mc_host,
            "mcPort": self.mc_port,
            "bridgePort": self.bridge_port,
            "rconPort": self.rcon_port,
            "runtimePortMin": self.runtime_port_min,
            "runtimePortMax": self.runtime_port_max,
            "scanExternal": self.scan_external,
        }

    @classmethod
    def from_env(cls) -> "DashboardConfig":
        java = _resolve_java()
        return cls(
            host=os.environ.get("DASHBOARD_HOST", "127.0.0.1"),
            port=_env_int("DASHBOARD_PORT", 8080),
            artifacts_dir=Path(os.environ.get("DASHBOARD_ARTIFACTS", ROOT / "training" / "artifacts")),
            var_dir=Path(os.environ.get("DASHBOARD_VAR", ROOT / "dashboard" / "var")),
            server_dir=Path(os.environ.get("DASHBOARD_SERVER_DIR", ROOT / "server")),
            bot_runtime_dir=Path(os.environ.get("DASHBOARD_BOT_RUNTIME_DIR", ROOT / "bot-runtime")),
            training_dir=Path(os.environ.get("DASHBOARD_TRAINING_DIR", ROOT / "training")),
            gmod_dir=_resolve_gmod_dir(),
            python_exe=_resolve_python(),
            java_exe=java.path,
            java_version=java.version,
            java_hint=java.hint,
            node_exe=_resolve_node(),
            allow_process_control=_env_bool("DASHBOARD_ALLOW_PROCESS_CONTROL", True),
            rcon_host=os.environ.get("DASHBOARD_RCON_HOST", "127.0.0.1"),
            rcon_port=_env_int("DASHBOARD_RCON_PORT", 25575),
            rcon_password=os.environ.get("DASHBOARD_RCON_PASSWORD", "devpass"),
            mc_host=os.environ.get("DASHBOARD_MC_HOST", "127.0.0.1"),
            mc_port=_env_int("DASHBOARD_MC_PORT", 25566),
            bridge_port=_env_int("DASHBOARD_BRIDGE_PORT", 8765),
            runtime_port_min=_env_int("DASHBOARD_RUNTIME_PORT_MIN", 8770),
            runtime_port_max=_env_int("DASHBOARD_RUNTIME_PORT_MAX", 8789),
            scan_external=_env_bool("DASHBOARD_SCAN_EXTERNAL", True),
            server_jar=os.environ.get("DASHBOARD_SERVER_JAR", "paper-1.8.8-445.jar"),
            server_memory_mb=_env_int("DASHBOARD_SERVER_MEMORY_MB", 1024),
        )


def is_loopback(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host == "localhost"
