"""Dashboard 的 HTTP 层：标准库 ThreadingHTTPServer + JSON API + 静态单页。

只监听回环地址；所有写操作在只读模式（DASHBOARD_ALLOW_PROCESS_CONTROL=0）下返回 403。
"""

from __future__ import annotations

import json
import mimetypes
import os
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from artifacts import RUN_ID_RE, ArtifactStore
from config import DashboardConfig
from errors import ApiError, BadRequest, Forbidden, NotFound, PayloadTooLarge
from gmod import GmodScanner
from match_runner import MatchRunner
from models import ModelStore
from presets import PresetStore
from processes import ProcessManager
from reward_presets import RewardPresetStore
from scripts import ScriptStore
from runtime_manager import RuntimeManager
from server_manager import ServerManager
from terminal import TerminalService
from training_runner import TrainingRunner
from validation import parse_int
from viz import build_viz

VERSION = "1.0.0"
MAX_BODY_BYTES = 1024 * 1024


class DashboardApp:
    """把各管理器组装在一起，HTTP handler 只做路由和序列化。"""

    def __init__(self, cfg: DashboardConfig) -> None:
        self.cfg = cfg
        cfg.ensure_dirs()
        self.started_at = time.time()
        self.procs = ProcessManager(cfg.logs_dir, cfg.state_file)
        self.artifacts = ArtifactStore(cfg.artifacts_dir)
        self.server = ServerManager(cfg, self.procs)
        self.runtimes = RuntimeManager(cfg, self.procs)
        self.reward_presets = RewardPresetStore(cfg.var_dir / "reward-presets")
        try:
            # 首次运行写入内置 default，前端下拉才有默认项。
            self.reward_presets.ensure()
        except (OSError, ApiError):
            pass
        self.gmod = GmodScanner(cfg.gmod_dir, cfg.var_dir / "gmod-cache")
        self.models = ModelStore(cfg.var_dir / "models", cfg.artifacts_dir, self.gmod)
        try:
            self.models.ensure()
        except (OSError, ApiError):
            pass
        #: 打开本机文件夹（Windows 才有）；测试可替换这两个属性。
        self.open_folder_supported = os.name == "nt"
        self.folder_opener = getattr(os, "startfile", None)
        self.scripts = ScriptStore(cfg.var_dir / "scripts.json")
        try:
            self.scripts.ensure()
        except (OSError, ApiError):
            pass
        self.training = TrainingRunner(cfg, self.procs, self.artifacts, self.runtimes,
                                       self.reward_presets, self.scripts, self.models)
        self.match = MatchRunner(cfg, self.server, self.runtimes)
        self.presets = PresetStore(cfg.var_dir / "presets")
        self.terminal = TerminalService(cfg, self.server, self.runtimes, self.training,
                                        self.artifacts, self.procs)
        self._lock = threading.Lock()

    # --- 本机文件夹 -----------------------------------------------------
    def folder_path(self, kind: str, run_id: str | None = None) -> Path:
        """把 ``kind`` 解析成一个绝对目录，并强制落在允许根内（防穿越）。"""
        var = self.cfg.var_dir
        artifacts = self.cfg.artifacts_dir
        name = str(kind or "").strip()
        direct = {
            "scripts": var,
            "presets": var / "presets",
            "reward-presets": var / "reward-presets",
            "models": var / "models",
            "gmod": self.cfg.gmod_dir,
            "artifacts": artifacts,
            "logs": var / "logs",
        }
        if name in ("run", "run-reviews"):
            ident = str(run_id or "").strip()
            if not ident or not RUN_ID_RE.match(ident) or ".." in ident:
                raise BadRequest("run 需要合法的 runId")
            run_dir = artifacts / "runs" / ident
            if not run_dir.is_dir():
                raise NotFound(f"运行不存在: {ident}")
            target = run_dir if name == "run" else run_dir / "reviews"
        elif name in direct:
            target = direct[name]
        else:
            raise BadRequest(
                "kind 必须是 scripts / presets / reward-presets / models / gmod / run / "
                "run-reviews / artifacts / logs"
            )
        resolved = target.resolve()
        roots = [var.resolve(), artifacts.resolve(), self.cfg.gmod_dir.resolve()]
        if not any(resolved == root or root in resolved.parents for root in roots):
            raise BadRequest(f"目录越界（不在 Dashboard 数据根内）: {resolved}")
        if not resolved.is_dir():
            raise NotFound(f"目录不存在: {resolved}")
        return resolved

    def open_folder(self, kind: str, run_id: str | None = None) -> dict:
        if not self.cfg.allow_process_control:
            raise Forbidden("Dashboard 以只读模式运行（DASHBOARD_ALLOW_PROCESS_CONTROL=0），不能打开本机文件夹")
        if not self.open_folder_supported or self.folder_opener is None:
            raise ApiError(501, "打开文件夹仅支持 Windows（os.startfile 不可用）")
        path = self.folder_path(kind, run_id)
        try:
            self.folder_opener(str(path))
        except OSError as exc:
            raise ApiError(500, f"打开文件夹失败: {exc}") from exc
        return {"ok": True, "path": str(path)}

    def episode_detail(self, run_id: str, episode: str | int) -> dict:
        """单局详情：先走 report.json；还没写出来时回退内存里正在跑的 job。"""
        try:
            return self.artifacts.episode_detail(run_id, episode)
        except (BadRequest, NotFound) as exc:
            partial = self._partial_episode(run_id, episode)
            if partial is None:
                raise exc
            return partial

    def _partial_episode(self, run_id: str, episode: str | int) -> dict | None:
        """报告尚未落盘时，用 ``TrainingJob.episodes`` 给一局不完整的结果。"""
        job = (self.training.current() or {}).get("job") or {}
        if not isinstance(job, dict) or job.get("runId") != run_id:
            return None
        try:
            index = int(episode)
        except (TypeError, ValueError):
            return None
        history = job.get("episodes") or job.get("history") or []
        if not isinstance(history, list) or index < 1 or index > len(history):
            return None
        item = history[index - 1]
        if not isinstance(item, dict):
            return None
        params = job.get("params") or {}
        model = None
        if params.get("modelId"):
            model = {
                "id": str(params.get("modelId")),
                "name": str(params.get("modelName") or params.get("modelId")),
            }
        return {
            "runId": run_id,
            "runKind": job.get("kind"),
            "episode": index,
            "episodeCount": len(history),
            "startedAt": item.get("startedAt"),
            "durationSeconds": item.get("durationSeconds"),
            "steps": item.get("steps"),
            "totalReward": item.get("totalReward", item.get("reward")),
            "winner": item.get("winner"),
            "terminated": item.get("terminated"),
            "truncated": item.get("truncated"),
            "opponentMode": item.get("opponentMode"),
            "pairKey": item.get("pairKey"),
            "model": model,
            "review": None,
            "partial": True,
        }

    # --- 聚合 -----------------------------------------------------------
    def health(self) -> dict:
        return {
            "ok": True,
            "name": "gladiator-dashboard",
            "version": VERSION,
            "pid": os.getpid(),
            "uptimeSeconds": round(time.time() - self.started_at, 1),
            "host": self.cfg.host,
            "port": self.cfg.port,
            "readOnly": not self.cfg.allow_process_control,
            "config": self.cfg.public(),
        }

    def annotate_runtimes(self, runtimes: list) -> list:
        """给运行时补上“角色 / 策略”标注，供前端显示彩色徽章与 hover 说明。

        - 训练中：users[0] 是被训练方（红），其余按 opponentMode 标成代码驱动策略；
        - 人机对战：人类玩家 vs 指定策略；
        - 其余运行时：未知来源。
        """
        roles = {}
        training = self.training.current()
        job = training.get("job") or {}
        config = job.get("params") or job.get("config") or {}
        users = [str(item).lower() for item in (config.get("users") or [])]
        mode = str(config.get("opponentMode") or job.get("opponentMode") or "").strip()
        plan = config.get("plan") or []
        if not mode and plan and isinstance(plan[0], dict):
            mode = str(plan[0].get("opponentMode") or "").strip()
        trained = {users[0]} if users else set()
        if config.get("pairs"):
            # 多场地并行：每对里 A 是被训练方，B 是对手
            trained = {name for name in users if name.endswith("a")}
        for index, name in enumerate(users):
            if name in trained:
                roles[name] = ("trained", "被训练方（红色）", "策略模型输出，正在通过强化学习更新")
            else:
                roles[name] = ("policy", f"代码驱动 · {mode or '启发式'}",
                               f"脚本策略：{mode or '启发式'}，不学习，用于对抗训练")
        match = self.match.current(refresh=False)
        for bot_name in (match.get("bots") or []):
            roles[str(bot_name).lower()] = ("match", "人机对战 · 代码驱动",
                                            f"由 {match.get('strategy') or '脚本策略'} 驱动，与人类玩家对战")
        annotated = []
        for item in runtimes:
            entry = dict(item)
            name = (entry.get("username") or "").lower()
            role, label, hint = roles.get(name, ("unknown", "未知来源", "不是本次训练/对战拉起的 Bot"))
            entry["role"] = role
            entry["roleLabel"] = label
            entry["roleHint"] = hint
            annotated.append(entry)
        return annotated

    def overview(self) -> dict:
        server = self.server.status(tail_lines=8)
        training = self.training.current()
        match = self.match.current()
        runtimes = self.runtimes.list()
        return {
            "checkedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "server": {
                "running": server["running"],
                "managed": server["managed"],
                "external": server["external"],
                "pid": server["pid"],
                "tps": server["tps"],
                "ports": server["ports"],
                "gladiator": server["gladiator"],
                "rconError": server["rconError"],
            },
            "runtimes": {
                "total": len(runtimes),
                "alive": sum(1 for item in runtimes if item.get("alive")),
                "external": sum(1 for item in runtimes if item.get("external")),
            },
            "training": {
                "running": training["running"],
                "status": (training.get("job") or {}).get("status"),
                "runId": (training.get("job") or {}).get("runId"),
                "startedAt": (training.get("job") or {}).get("startedAt"),
                "episodesCompleted": (training.get("job") or {}).get("episodesCompleted"),
                "totalEpisodes": (training.get("job") or {}).get("totalEpisodes"),
            },
            "match": {
                "running": match["running"],
                "status": match["status"],
                "address": match["address"],
            },
            "artifacts": {
                "trainReport": self.artifacts.train_report().get("exists"),
                "multiReport": self.artifacts.multi_report().get("exists"),
                "stressReport": self.artifacts.stress().get("exists"),
                "reviews": len(self.artifacts.reviews()),
                "runs": len(self.artifacts.runs()),
            },
        }

    def close(self) -> None:
        """Dashboard 退出时只清理自己托管的进程，避免留下半启动的 java/node。"""
        for proc_id in ("training",):
            proc = self.procs.get(proc_id)
            if proc is not None and proc.alive:
                try:
                    self.procs.stop(proc_id, timeout=5.0)
                except Exception:  # noqa: BLE001
                    pass
        try:
            self.match.stop()
        except Exception:  # noqa: BLE001 - 没有对战时会 409
            pass
        try:
            self.runtimes.stop_all()
        except Exception:  # noqa: BLE001
            pass
        try:
            proc = self.procs.get("paper")
            if proc is not None and proc.alive:
                self.procs.stop("paper", graceful_line="stop\n", timeout=8.0)
        except Exception:  # noqa: BLE001
            pass


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = f"GladiatorDashboard/{VERSION}"
    protocol_version = "HTTP/1.1"
    app: DashboardApp  # 由 create_handler 注入

    # --- 基础响应 -------------------------------------------------------
    def handle_one_request(self) -> None:
        """每个请求开始时清空 body 缓存。

        ``BaseHTTPRequestHandler`` 对同一 keep-alive 连接复用同一个 handler 实例；
        若不重置，第二个请求会命中上一次的 ``_body_cache``，自己的 body 留在 socket 里，
        被当成下一行请求行解析（历史上表现为 ``501 Unsupported method ('{}GET')``）。
        """
        self._body_cache = None
        super().handle_one_request()

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {self.address_string()} {fmt % args}\n")

    def _common_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'",
        )

    def _send_json(self, status: int, payload: dict | list) -> None:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self._common_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, status: int, body: bytes, content_type: str, cache: str = "no-cache") -> None:
        self.send_response(status)
        self._common_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", cache)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> bytes:
        """一次性读走请求体并缓存。

        keep-alive 下如果 POST 处理器不读 body，剩下的字节会被当成下一个请求行，
        出现 ``Unsupported method ('{}GET')`` 这类错位，所以在分发前必须先读干净。
        """
        cached = getattr(self, "_body_cache", None)
        if cached is not None:
            return cached
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError as exc:
            self.close_connection = True
            raise BadRequest("Content-Length 非法") from exc
        if length > MAX_BODY_BYTES:
            self.close_connection = True
            raise PayloadTooLarge("请求体超过 1MB")
        if length:
            self._body_cache = self.rfile.read(length)
            return self._body_cache
        # 没有 Content-Length 时可能是 chunked（curl -T / node fetch 的流式 body）；
        # 必须读完，否则分块数据会残留到下一个请求。
        encoding = (self.headers.get("Transfer-Encoding") or "").lower()
        if "chunked" in encoding:
            self._body_cache = self._read_chunked_body()
            return self._body_cache
        self._body_cache = b""
        return self._body_cache

    def _read_chunked_body(self) -> bytes:
        """读取 ``Transfer-Encoding: chunked`` 的请求体并返回解码后的字节。"""
        chunks: list[bytes] = []
        total = 0
        while True:
            line = self.rfile.readline(65536)
            if not line:
                self.close_connection = True
                break
            try:
                size = int(line.split(b";", 1)[0].strip() or b"0", 16)
            except ValueError:
                self.close_connection = True
                raise BadRequest("chunked 请求体格式非法")
            if size == 0:
                # 末尾可能有 trailer，读到空行为止
                while True:
                    trailer = self.rfile.readline(65536)
                    if trailer in (b"\r\n", b"\n", b""):
                        break
                break
            total += size
            if total > MAX_BODY_BYTES:
                self.close_connection = True
                raise PayloadTooLarge("请求体超过 1MB")
            chunks.append(self.rfile.read(size))
            self.rfile.read(2)  # 每个 chunk 结尾的 CRLF
        return b"".join(chunks)

    def _read_json(self) -> dict:
        raw = self._read_body()
        if not raw.strip():
            return {}
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BadRequest(f"请求体不是合法 JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise BadRequest("请求体需要 JSON 对象")
        return payload

    # --- 路由 -----------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 约定
        try:
            self._handle_get()
        except ApiError as exc:
            self._send_json(exc.status, {"ok": False, "error": exc.message})
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            self._send_json(500, {"ok": False, "error": "服务器内部错误，请查看 Dashboard 控制台"})

    def do_POST(self) -> None:  # noqa: N802
        try:
            self._read_body()
            self._handle_post()
        except ApiError as exc:
            self._send_json(exc.status, {"ok": False, "error": exc.message})
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            self._send_json(500, {"ok": False, "error": "服务器内部错误，请查看 Dashboard 控制台"})

    def do_DELETE(self) -> None:  # noqa: N802
        try:
            self._read_body()
            self._handle_delete()
        except ApiError as exc:
            self._send_json(exc.status, {"ok": False, "error": exc.message})
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            self._send_json(500, {"ok": False, "error": "服务器内部错误，请查看 Dashboard 控制台"})

    def _handle_delete(self) -> None:
        parsed = urlsplit(self.path)
        path = unquote(parsed.path)
        query = parse_qs(parsed.query)
        app = self.app
        if path == "/api/models":
            target = (query.get("id") or [""])[0]
            deleted = app.models.delete(target)
            return self._send_json(200, {"ok": True, "deleted": deleted,
                                         "models": app.models.list()})
        if path == "/api/scripts":
            target = (query.get("id") or [""])[0]
            deleted = app.scripts.delete(target)
            return self._send_json(200, {
                "ok": True, "deleted": deleted, "scripts": app.scripts.items(),
            })
        prefix = "/api/reward-presets/"
        if path.startswith(prefix):
            ident = path[len(prefix):].strip("/")
            return self._send_json(200, {"ok": True, "deleted": app.reward_presets.delete(ident)})
        raise NotFound(f"未知接口: {path}")

    def _handle_get(self) -> None:
        parsed = urlsplit(self.path)
        path = unquote(parsed.path)
        query = parse_qs(parsed.query)
        app = self.app

        if path in ("/", "/index.html"):
            return self._serve_static("index.html")
        if path == "/terminal":
            return self._serve_static("index.html")
        if path.startswith("/static/"):
            return self._serve_static(path[len("/static/"):])
        if path == "/api/terminal/stream":
            source = (query.get("source") or [""])[0]
            name = (query.get("name") or [""])[0]
            return self._serve_sse(source, name)
        if path == "/favicon.ico":
            return self._send_bytes(204, b"", "image/x-icon")

        run_id = ((query.get("run") or query.get("id") or [""])[0] or None)
        routes = {
            "/api/health": lambda: app.health(),
            "/api/config": lambda: {"ok": True, "config": app.cfg.public()},
            "/api/overview": lambda: {"ok": True, **app.overview()},
            "/api/server": lambda: {"ok": True, "server": app.server.status()},
            "/api/server/status": lambda: {"ok": True, "server": app.server.status()},
            "/api/runtimes": lambda: {"ok": True, "runtimes": app.annotate_runtimes(app.runtimes.list()),
                                      "defaults": app.runtimes.defaults()},
            "/api/training": lambda: {"ok": True, **app.training.current()},
            "/api/training/options": lambda: {"ok": True, "options": app.training.options()},
            "/api/match": lambda: {"ok": True, "match": app.match.current(refresh=True)},
            "/api/artifacts": lambda: {"ok": True, **app.artifacts.summary(run_id)},
            "/api/artifacts/summary": lambda: {"ok": True, **app.artifacts.summary(run_id)},
            "/api/artifacts/runs": lambda: {"ok": True, "runs": app.artifacts.runs()},
            "/api/artifacts/train": lambda: {"ok": True, "train": app.artifacts.train_report()},
            "/api/artifacts/multi": lambda: {"ok": True, "multi": app.artifacts.multi_report()},
            "/api/artifacts/reviews": lambda: {"ok": True, "reviews": app.artifacts.reviews(run_id)},
            "/api/artifacts/stress": lambda: {"ok": True, "stress": app.artifacts.stress()},
            "/api/presets": lambda: {"ok": True, "presets": app.presets.list()},
            "/api/models": lambda: (
                {"ok": True, "model": app.models.detail(ident), "models": app.models.list()}
                if (ident := (query.get("id") or [""])[0])
                else app.models.payload()
            ),
            "/api/scripts": lambda: {
                "ok": True,
                "scripts": app.scripts.items(),
                "builtin": [item["id"] for item in app.scripts.items() if item["builtin"]],
                "colors": [f"§{code}" for code in "0123456789abcdef"],
            },
            "/api/reward-presets": lambda: {
                "ok": True,
                "presets": app.reward_presets.list(),
                "schema": app.reward_presets.schema(),
            },
        }
        if path in ("/api/artifacts/merge", "/api/artifacts/merge/summary"):
            run_ids: list[str] = []
            for value in query.get("runs") or []:
                run_ids.extend(part for part in value.split(",") if part.strip())
            return self._send_json(200, {"ok": True, **app.artifacts.merge(run_ids)})
        if path == "/api/artifacts/run":
            run_id = (query.get("id") or [""])[0]
            return self._send_json(200, {"ok": True, "run": app.artifacts.run_detail(run_id)})
        if path == "/api/models/viz":
            ident = (query.get("id") or [""])[0]
            if not ident:
                raise BadRequest("缺少 id 参数")
            return self._send_json(200, build_viz(app.models, app.models.artifacts_dir, ident))
        if path == "/api/artifacts/review":
            name = (query.get("path") or [""])[0]
            markdown = app.artifacts.review_markdown(name)
            return self._send_json(200, {"ok": True, "path": name, "markdown": markdown})
        if path == "/api/training/episodes":
            run_id = (query.get("run") or [""])[0]
            episode = (query.get("episode") or [""])[0]
            if not episode:
                raise BadRequest("episode 不能为空")
            return self._send_json(200, {"ok": True, **app.episode_detail(run_id, episode)})
        if path.startswith("/api/reward-presets/"):
            preset_id = path[len("/api/reward-presets/"):].strip("/")
            return self._send_json(200, {"ok": True, **app.reward_presets.detail(preset_id)})
        handler = routes.get(path)
        if handler is None:
            raise NotFound(f"未知接口: {path}")
        self._send_json(200, handler())

    def _handle_post(self) -> None:
        parsed = urlsplit(self.path)
        path = unquote(parsed.path)
        query = parse_qs(parsed.query)
        app = self.app
        if path == "/api/server/start":
            return self._send_json(200, {"ok": True, "server": app.server.start()})
        if path == "/api/server/stop":
            return self._send_json(200, {"ok": True, "server": app.server.stop()})
        if path == "/api/server/console":
            payload = self._read_json()
            line = payload.get("line")
            if not isinstance(line, str) or not line.strip():
                raise BadRequest("line 不能为空")
            return self._send_json(200, {"ok": True, **app.server.write_console(line)})
        if path == "/api/runtimes/start":
            payload = self._read_json()
            username = payload.get("username")
            env_port = payload.get("envPort")
            if not isinstance(username, str) or not username.strip():
                raise BadRequest("username 不能为空")
            if env_port in (None, ""):
                raise BadRequest("envPort 不能为空")
            runtime = app.runtimes.start(username, parse_int(env_port, "envPort", minimum=1024, maximum=65535))
            return self._send_json(200, {"ok": True, "runtime": runtime})
        if path == "/api/runtimes/stop":
            payload = self._read_json()
            runtime_id = payload.get("id")
            if not isinstance(runtime_id, str) or not runtime_id.strip():
                raise BadRequest("id 不能为空")
            return self._send_json(200, {"ok": True, "runtime": app.runtimes.stop(runtime_id)})
        if path == "/api/runtimes/stop-all":
            return self._send_json(200, {"ok": True, "runtimes": app.runtimes.stop_all()})
        if path == "/api/training/start":
            payload = self._read_json()
            return self._send_json(200, {"ok": True, **app.training.start(payload)})
        if path == "/api/training/stop":
            return self._send_json(200, {"ok": True, **app.training.stop()})
        if path == "/api/presets/save":
            payload = self._read_json()
            return self._send_json(200, {"ok": True, "preset": app.presets.save(payload)})
        if path == "/api/presets/load":
            payload = self._read_json()
            return self._send_json(200, {"ok": True, "preset": app.presets.load(payload.get("id"))})
        if path == "/api/presets/delete":
            payload = self._read_json()
            return self._send_json(200, {"ok": True, "deleted": app.presets.delete(payload.get("id"))})
        if path == "/api/open-folder":
            payload = self._read_json()
            return self._send_json(200, app.open_folder(payload.get("kind"), payload.get("id")))
        if path == "/api/models/reload":
            return self._send_json(200, app.models.reload_payload())
        if path == "/api/models/export":
            # body `{"id": "...", "overwrite": false}`，也接受 query `?id=`。
            payload = self._read_json()
            ident = payload.get("id") or (query.get("id") or [""])[0]
            if not isinstance(ident, str) or not ident.strip():
                raise BadRequest("导出需要模型 id（body.id 或 query id）")
            overwrite = payload.get("overwrite")
            if overwrite is None:
                overwrite = (query.get("overwrite") or ["0"])[0]
            if isinstance(overwrite, str):
                overwrite = overwrite.strip().lower() in ("1", "true", "yes", "on")
            return self._send_json(200, {
                "ok": True,
                **app.models.export_gmod(ident.strip(), overwrite=bool(overwrite)),
            })
        if path == "/api/models":
            payload = self._read_json()
            return self._send_json(200, {"ok": True, "model": app.models.create(payload),
                                         "models": app.models.list()})
        if path == "/api/models/update":
            payload = self._read_json()
            return self._send_json(200, {"ok": True, "model": app.models.update(payload),
                                         "models": app.models.list()})
        if path == "/api/scripts":
            payload = self._read_json()
            saved = app.scripts.save(payload)
            return self._send_json(200, {"ok": True, "script": saved, "scripts": app.scripts.items()})
        if path == "/api/reward-presets":
            payload = self._read_json()
            preset_id = payload.get("id")
            preset = payload.get("preset")
            if preset is None:
                # 兼容直接把预设字段放在顶层（前端编辑弹窗就是这么发的）。
                preset = {key: value for key, value in payload.items() if key != "id"}
            saved = app.reward_presets.save(preset_id, preset)
            return self._send_json(200, {
                "ok": True,
                "id": str(preset_id or "").strip(),
                "preset": saved,
            })
        if path == "/api/match/start":
            payload = self._read_json()
            return self._send_json(200, {"ok": True, "match": app.match.start(payload)})
        if path == "/api/match/stop":
            return self._send_json(200, {"ok": True, "match": app.match.stop()})
        raise NotFound(f"未知接口: {path}")

    def _serve_sse(self, source: str, name: str) -> None:
        """只读 SSE：先解析来源（错误仍返回 JSON 4xx），再发送事件流。"""
        spec = self.app.terminal.resolve(source, name)
        self.send_response(200)
        self._common_headers()
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        self.close_connection = True
        try:
            for chunk in self.app.terminal.stream(spec):
                self.wfile.write(chunk.encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return

    def _serve_static(self, relative: str) -> None:
        relative = relative.replace("\\", "/").lstrip("/")
        if not relative:
            raise NotFound("缺少静态文件名")
        root = self.app.cfg.static_dir.resolve()
        target = (root / relative).resolve()
        if target != root and root not in target.parents:
            raise BadRequest("静态路径越界")
        if not target.is_file():
            raise NotFound(f"静态文件不存在: {relative}")
        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type in (
            "application/javascript", "application/json", "image/svg+xml"
        ):
            content_type += "; charset=utf-8"
        try:
            body = target.read_bytes()
        except OSError as exc:
            raise NotFound(f"读取静态文件失败: {exc}") from exc
        # 本地工具：前端改动应立刻可见，避免“代码已更新但浏览器仍用旧 JS/CSS”。
        cache = "no-cache"
        self._send_bytes(200, body, content_type, cache=cache)


def create_handler(app: DashboardApp):
    class _Handler(DashboardHandler):
        pass

    _Handler.app = app
    return _Handler


class _QuietServer(ThreadingHTTPServer):
    """浏览器断开 SSE / 轮询时不要打印堆栈。

    长连接页面（终端、日志）在刷新或关标签页时会让 ``wfile.write`` 抛
    ConnectionResetError；这是客户端行为，不是服务端错误。
    """

    daemon_threads = True

    def handle_error(self, request, client_address) -> None:  # noqa: D102 - 覆盖标准库钩子
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)):
            return
        super().handle_error(request, client_address)


def make_server(app: DashboardApp, host: str | None = None, port: int | None = None) -> ThreadingHTTPServer:
    server = _QuietServer(
        (host or app.cfg.host, app.cfg.port if port is None else port),
        create_handler(app),
    )
    server.daemon_threads = True
    return server


def serve(cfg: DashboardConfig | None = None, port: int | None = None) -> None:
    cfg = cfg or DashboardConfig.from_env()
    app = DashboardApp(cfg)
    httpd = make_server(app, port=port)
    url = f"http://{cfg.host}:{httpd.server_address[1]}/"
    print(f"[dashboard] Gladiator Dashboard 已启动: {url}", flush=True)
    print(f"[dashboard] 产物目录: {cfg.artifacts_dir}", flush=True)
    print("[dashboard] Ctrl+C 退出（会停止本 Dashboard 托管的服务器/运行时/训练进程）", flush=True)
    try:
        httpd.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("\n[dashboard] 收到 Ctrl+C，正在停止托管进程…", flush=True)
    finally:
        httpd.shutdown()
        httpd.server_close()
        app.close()


def _parse_cli(argv: list[str]) -> tuple[int | None, Path | None]:
    """解析 ``--port`` / ``--log-file``。

    无窗口启动（pythonw）时 stdout 会被丢弃，所以启动器会传 ``--log-file``，
    让 Dashboard 自己把日志写到文件里，进程树与终端彻底解耦。
    """
    port: int | None = None
    log_file: Path | None = None
    index = 0
    while index < len(argv):
        raw = argv[index]
        if raw.startswith("--port="):
            port = int(raw.split("=", 1)[1])
        elif raw == "--port" and index + 1 < len(argv):
            index += 1
            port = int(argv[index])
        elif raw.startswith("--log-file="):
            log_file = Path(raw.split("=", 1)[1])
        elif raw == "--log-file" and index + 1 < len(argv):
            index += 1
            log_file = Path(argv[index])
        elif raw in ("-h", "--help"):
            print("用法: python dashboard/app.py [--port 8080] [--log-file <path>]")
            raise SystemExit(0)
        index += 1
    return port, log_file


def _redirect_output(path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a", encoding="utf-8", buffering=1)
    sys.stdout = stream
    sys.stderr = stream


def main() -> None:
    try:
        port, log_file = _parse_cli(sys.argv[1:])
        _redirect_output(log_file)
        serve(port=port)
    except ValueError as exc:
        print(f"[dashboard] 配置错误: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
