"use strict";

/* ============================================================
 * Gladiator Dashboard 前端：原生 JS，无外部依赖。
 * ============================================================ */

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

const state = {
  tab: "overview",
  health: null,
  server: null,
  runtimes: [],
  runtimeDefaults: null,
  training: null,
  options: null,
  plan: [],
  planTouched: false,
  planInitialized: false,
  scripts: [],
  models: [],
  modelSupport: false,
  presets: [],
  presetsAvailable: true,
  rewardPresets: [],
  rewardSchema: null,
  scriptItems: [],
  scriptEditing: null,
  models: [],
  modelDetail: null,
  modelEditingId: null,
  gmodDir: null,
  gmodScanned: 0,
  gmodInvalid: [],
  persistTouched: false,
  persistToastKey: null,
  importGmodId: null,
  episodeRunId: null,
  rewardEditor: null,
  rewardEditing: null,
  dragPlanIndex: null,
  match: null,
  artifacts: null,
  runs: [],
  selectedRun: null,
  selectedRunData: null,
  selectedRuns: new Set(),
  mergeMode: false,
  pairFilter: null,
  runtimeLogId: null,
  terminal: { source: "server", name: "", es: null, paused: false, lines: [] },
  inflight: new Set(),
  charts: 0,
};

const STATUS_TEXT = {
  idle: "未运行",
  starting: "启动中",
  running: "运行中",
  finished: "已完成",
  failed: "失败",
  stopped: "已停止",
};

/* ---------------- 基础工具 ---------------- */

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[ch]));
}

/* Minecraft 颜色码（§0–§f）→ CSS 颜色；未知码回退 inherit。 */
const MC_COLORS = {
  "0": "#000000", "1": "#0000aa", "2": "#00aa00", "3": "#00aaaa",
  "4": "#aa0000", "5": "#aa00aa", "6": "#ffaa00", "7": "#aaaaaa",
  "8": "#555555", "9": "#5555ff", "a": "#55ff55", "b": "#55ffff",
  "c": "#ff5555", "d": "#ff55ff", "e": "#ffff55", "f": "#ffffff",
};

function mcColor(code) {
  const key = String(code ?? "").replace(/^§/, "").toLowerCase();
  return MC_COLORS[key] || "inherit";
}

/* 仅当后端未返回 scripts 时使用（例如仍在运行的旧版 Dashboard 进程）。
   与后端 OPPONENT_MODES 保持一致，避免页面出现空的脚本下拉。 */
const FALLBACK_SCRIPTS = [
  { id: "random", name: "Random", builtin: true, color: "§7" },
  { id: "wander", name: "Wander", builtin: true, color: "§a" },
  { id: "idle", name: "Idle", builtin: true, color: "§8" },
  { id: "scripted", name: "Scripted", builtin: true, color: "§e" },
  { id: "rusher", name: "Rusher", builtin: true, color: "§c" },
  { id: "aimbot", name: "Aimbot", builtin: true, color: "§d" },
];

function num(value, digits = 0) {
  if (value === null || value === undefined || value === "" || Number.isNaN(Number(value))) return "—";
  const n = Number(value);
  return n.toLocaleString("zh-CN", { maximumFractionDigits: digits, minimumFractionDigits: 0 });
}

function fmtDuration(seconds) {
  if (seconds === null || seconds === undefined || Number.isNaN(Number(seconds))) return "—";
  const total = Math.max(0, Math.floor(Number(seconds)));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h) return `${h}h ${m}m`;
  if (m) return `${m}m ${s}s`;
  return `${s}s`;
}

function fmtTime(value) {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "number") {
    const date = new Date(value < 1e12 ? value * 1000 : value);
    return Number.isNaN(date.getTime()) ? "—" : date.toLocaleString("zh-CN", { hour12: false });
  }
  const text = String(value);
  const date = new Date(text);
  if (!Number.isNaN(date.getTime())) return date.toLocaleString("zh-CN", { hour12: false });
  return text.replace("T", " ");
}

function statusPill(status, text) {
  const cls = ["running", "starting", "finished"].includes(status) ? "ok"
    : status === "failed" ? "bad"
      : status === "stopped" ? "warn" : "neutral";
  return `<span class="pill ${cls}">${escapeHtml(text || STATUS_TEXT[status] || status || "未知")}</span>`;
}

function boolPill(value, on = "在线", off = "离线") {
  return `<span class="pill ${value ? "ok" : "bad"}">${value ? on : off}</span>`;
}

function statCard(label, value, sub = "", extra = "") {
  return `<div class="card stat-card">
    <div class="stat-label">${escapeHtml(label)}</div>
    <div class="stat-value">${value}</div>
    ${sub ? `<div class="stat-sub">${sub}</div>` : ""}
    ${extra}
  </div>`;
}

function kvItem(key, value) {
  return `<div class="kv-item"><div class="k">${escapeHtml(key)}</div><div class="v">${value}</div></div>`;
}

async function api(path, options = {}) {
  const init = { method: options.method || "GET", headers: {} };
  if (options.body !== undefined) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(options.body);
  }
  const response = await fetch(path, init);
  let payload = null;
  try { payload = await response.json(); } catch (err) { /* 非 JSON 响应 */ }
  if (!response.ok) {
    const message = (payload && payload.error) || `HTTP ${response.status}`;
    // 旧进程没有新路由时会回 404「未知接口」：给可操作的提示，别只显示原始错误。
    if (response.status === 404 && /未知接口|not found|无法找到该路由/i.test(message)) {
      throw new Error(`后端可能是旧进程，请重启 Dashboard（${message}）`);
    }
    throw new Error(message);
  }
  return payload;
}

function toast(message, kind = "ok") {
  const wrap = $("#toast-wrap");
  const node = document.createElement("div");
  node.className = `toast ${kind}`;
  node.textContent = message;
  wrap.appendChild(node);
  setTimeout(() => {
    node.style.opacity = "0";
    node.style.transition = "opacity .25s";
    setTimeout(() => node.remove(), 260);
  }, kind === "err" ? 6000 : 3200);
}

async function withBusy(button, fn) {
  if (button) button.disabled = true;
  try {
    await fn();
  } catch (err) {
    toast(err.message || String(err), "err");
  } finally {
    if (button) button.disabled = false;
  }
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    toast("已复制", "ok");
  } catch (err) {
    const area = document.createElement("textarea");
    area.value = text;
    document.body.appendChild(area);
    area.select();
    try { document.execCommand("copy"); toast("已复制", "ok"); }
    catch (e) { toast("复制失败，请手动选择文本", "err"); }
    area.remove();
  }
}

/* ---------------- 标签页 ---------------- */

function showTab(name) {
  state.tab = name;
  $$(".tab").forEach((tab) => tab.classList.toggle("active", tab.dataset.tab === name));
  $$(".panel").forEach((panel) => panel.classList.toggle("active", panel.id === `panel-${name}`));
  if (name === "data") loadArtifacts(true);
  if (name === "training" && state.options) {
    fillTrainingForm(state.options);
    loadPresets();
  }
  if (name === "models") {
    loadModels();
    loadArtifacts(false);
  }
  if (name === "runtime") loadRuntimes();
  if (name === "match") loadMatch();
  if (name === "server" || name === "overview") loadServer();
  if (name === "terminal") initTerminal();
  if (name !== "terminal") stopTerminalStream();
}

/* ---------------- 顶栏 ---------------- */

function renderTop() {
  const server = state.server;
  const alive = state.runtimes.filter((item) => item.alive).length;
  const training = state.training || {};
  const job = training.job;
  const parts = [
    `<span class="pill ${server && server.running ? "ok" : "bad"}">服务器 ${server ? (server.running ? "在线" : "离线") : "未知"}</span>`,
    `<span class="pill ${alive ? "ok" : "neutral"}">Bot 运行时 ${alive}</span>`,
  ];
  if (job) {
    parts.push(statusPill(job.status, `训练 ${STATUS_TEXT[job.status] || job.status}`));
  } else {
    parts.push(`<span class="pill neutral">训练 未运行</span>`);
  }
  if (state.match && state.match.running) {
    parts.push(`<span class="pill info">对战进行中</span>`);
  }
  $("#top-status").innerHTML = parts.join("");
}

/* ---------------- 总览 ---------------- */

function renderOverview() {
  const server = state.server || {};
  const gla = server.gladiator || {};
  const job = (state.training || {}).job;
  const match = state.match || {};
  const alive = state.runtimes.filter((item) => item.alive).length;
  const cards = [
    statCard("Paper 服务器", server.running ? "在线" : "离线",
      server.pid ? `PID ${server.pid} · ${fmtDuration(server.uptimeSeconds)}` : "未检测到进程",
      `<div style="margin-top:6px">${boolPill(server.running)} ${server.external ? '<span class="pill warn">外部进程（只读）</span>' : ""}</div>`),
    statCard("Bot 运行时", `${alive}`,
      `通道 A 会话 ${gla.sessions ?? "—"} · bots ${gla.bots ?? "—"}`),
    statCard("训练任务", job ? (STATUS_TEXT[job.status] || job.status) : (gla.training ? "外部训练中" : "未运行"),
      job ? `${job.episodesCompleted}/${job.totalEpisodes} 回合 · ${num(job.steps)} 步` : (gla.training ? `插件报告 ${gla.training} 个训练会话（只读）` : "可在「训练计划」启动")),
    statCard("人机对战", match.running ? "进行中" : "空闲",
      match.running ? `${match.bots.filter((bot) => bot.ready).length}/${match.count} Bot ready` : "一键拉起 Bot 陪练"),
  ];
  $("#overview-cards").innerHTML = cards.join("");

  const cfg = state.health && state.health.config;
  if (cfg) {
    const address = `${cfg.mcHost}:${cfg.mcPort}`;
    $("#overview-address").textContent = address;
    const spectate = "/gladiator spectate <bot>";
    const code = $("#overview-spectate");
    code.textContent = spectate;
    code.dataset.copy = spectate;
  }
  if (server.logTail) {
    $("#overview-log").textContent = server.logTail.join("\n") || "日志为空。";
    $("#overview-log-path").textContent = server.logPath || "";
  } else {
    $("#overview-log").textContent = "未连接服务器；启动后这里会显示日志尾部。";
  }
  const startBtn = $("#btn-quick-server");
  if (startBtn) {
    startBtn.textContent = server.running ? "服务器已在线" : "启动 Paper 服务器";
    startBtn.disabled = Boolean(server.running) || (state.health && state.health.readOnly);
  }
}

/* ---------------- 服务管理 ---------------- */

function renderServer() {
  const server = state.server;
  if (!server) return;
  const gla = server.gladiator;
  const ports = server.ports || {};
  const cards = [
    statCard("进程状态", server.running ? (server.managed ? "托管运行" : "外部运行") : "已停止",
      server.pid ? `PID ${server.pid} · 运行 ${fmtDuration(server.uptimeSeconds)}` : "—",
      `<div style="margin-top:6px">${boolPill(server.running)}</div>`),
    statCard("TPS", server.tps === null || server.tps === undefined ? "—" : num(server.tps, 2),
      server.tps === null ? "RCON 未返回" : "20 为满值"),
    statCard("在线 Bot / 对局", `${gla ? (gla.bots ?? "—") : "—"} / ${gla ? (gla.games ?? "—") : "—"}`,
      gla ? `training ${gla.training ?? "—"} · sessions ${gla.sessions ?? "—"}` : "gladiator status 不可用"),
    statCard("堆内存", gla && gla.heapUsedMb != null ? `${gla.heapUsedMb} MB` : "—",
      gla && gla.heapMaxMb != null ? `上限 ${gla.heapMaxMb} MB` : "—"),
  ];
  $("#server-cards").innerHTML = cards.join("");

  const detail = [];
  detail.push(kvItem("gladiator 版本", escapeHtml(gla ? gla.version ?? "—" : "—")));
  detail.push(kvItem("API", escapeHtml(gla ? gla.api ?? "—" : "—")));
  detail.push(kvItem("ready", gla ? boolPill(gla.ready) : boolPill(false, "就绪", "未就绪")));
  detail.push(kvItem("WorldGuard", escapeHtml(gla ? gla.worldguard ?? "—" : "—")));
  detail.push(kvItem("link", escapeHtml(gla ? gla.link ?? "—" : "—")));
  detail.push(kvItem("游戏端口", boolPill(ports.game, "监听", "未监听")));
  detail.push(kvItem("端口 8765 通道 A", boolPill(ports.bridge, "监听", "未监听")));
  detail.push(kvItem("端口 25575 RCON", boolPill(ports.rcon, "监听", "未监听")));
  detail.push(kvItem("服务端 jar", server.jarExists ? escapeHtml(server.jar) : '<span class="pill bad">缺失</span>'));
  if (server.javaReady) {
    detail.push(kvItem("Java 8", `${escapeHtml(server.java || "—")} <span class="muted small">v${escapeHtml(String(server.javaVersion ?? "?"))}</span>`));
  } else {
    const reason = server.javaVersion ? `Java ${server.javaVersion}（需要 8）` : "缺失";
    detail.push(kvItem("Java 8", `<span class="pill bad" title="${escapeHtml(server.javaHint || "")}">${escapeHtml(reason)}</span>`));
  }
  detail.push(kvItem("日志文件", escapeHtml(server.logPath || "—")));
  if (server.rconError) detail.push(kvItem("RCON 提示", `<span class="muted">${escapeHtml(server.rconError)}</span>`));
  $("#server-details").innerHTML = `<div class="kv">${detail.join("")}</div>`;

  const used = gla && gla.heapUsedMb;
  const max = gla && gla.heapMaxMb;
  if (used != null && max) {
    const pct = Math.min(100, Math.round((used / max) * 100));
    $("#server-heap").innerHTML = `
      <div class="heap-bar"><div class="heap-fill" style="width:${pct}%"></div></div>
      <div class="muted small">${used} MB / ${max} MB（${pct}%）</div>`;
  } else {
    $("#server-heap").innerHTML = '<p class="muted">暂无堆内存数据（需要 RCON 可用）。</p>';
  }
  $("#server-log").textContent = (server.logTail || []).join("\n") || "（暂无日志）";
  $("#server-log-path").textContent = server.logPath || "";
  $("#btn-server-start").disabled = !state.health || state.health.readOnly || server.running;
  $("#btn-server-stop").disabled = !state.health || state.health.readOnly || !server.managed;
  $("#overview-log").textContent = (server.logTail || []).join("\n") || "（日志为空）";
}

/* ---------------- 运行时 ---------------- */

function roleBadge(item) {
  const role = item.role || "unknown";
  const label = item.roleLabel || "未知来源";
  const hint = item.roleHint || "";
  const cls = role === "trained" ? "trained" : role === "policy" ? "policy" : role === "match" ? "match" : "unknown";
  return ` <span class="role-badge ${cls}" title="${escapeHtml(hint)}">${escapeHtml(label)}</span>`;
}

function renderRuntimes() {
  const list = state.runtimes || [];
  $("#runtime-count").textContent = `${list.filter((item) => item.alive).length} 个存活 / 共 ${list.length} 个`;
  const body = $("#runtime-body");
  body.innerHTML = list.map((item) => {
    let status;
    let cls;
    if (item.external) {
      status = "外部运行中";
      cls = "warn";
    } else if (item.state === "running") {
      status = "运行中";
      cls = "ok";
    } else {
      status = "已停止";
      cls = "neutral";
    }
    const canStop = item.alive && !item.external;
    return `<tr>
      <td>${escapeHtml(item.username || (item.external ? "（外部进程）" : "—"))}
        ${roleBadge(item)}</td>
      <td>${item.envPort ?? "—"}</td>
      <td><span class="pill ${cls}">${status}</span></td>
      <td>${item.pid ?? "—"}</td>
      <td>${item.uptimeSeconds == null ? "—" : fmtDuration(item.uptimeSeconds)}</td>
      <td>${boolPill(item.listening, "监听", "未监听")}</td>
      <td>
        <button class="btn tiny ghost" data-runtime-log="${escapeHtml(item.id)}" type="button" ${item.external ? "disabled" : ""}>日志</button>
        <button class="btn tiny danger" data-runtime-stop="${escapeHtml(item.id)}" type="button" ${canStop ? "" : "disabled"}>停止</button>
      </td>
    </tr>`;
  }).join("");
  $("#runtime-empty").classList.toggle("hidden", list.length > 0);

  if (state.runtimeLogId) {
    const item = list.find((entry) => entry.id === state.runtimeLogId);
    if (item) {
      $("#runtime-log-title").textContent = `${item.username || item.id} · ${item.id}`;
      $("#runtime-log").textContent = (item.tail || []).join("\n") || "（暂无日志）";
    }
  }
  const defaults = state.runtimeDefaults;
  if (defaults && !$("#runtime-username").dataset.touched) {
    $("#runtime-username").value = defaults.username || "Glad_Bot_1";
    $("#runtime-port").value = defaults.envPort || 8770;
  }
}

/* ---------------- 训练 ---------------- */

/* 计划项是唯一的回合来源：{opponentMode, episodes}，不再有独立「回合数」输入。 */
function planEntryValid(item) {
  if (!item || !item.opponentMode) return false;
  const episodes = Number(item.episodes);
  return Number.isInteger(episodes) && episodes >= 1 && episodes <= 500;
}

function planEpochs() {
  const el = $("#tr-epochs");
  const value = el ? Number(el.value) : 1;
  return Number.isInteger(value) && value >= 1 && value <= 200 ? value : 1;
}

function epochsInputValid() {
  const el = $("#tr-epochs");
  if (!el) return true;
  const value = Number(el.value);
  return Number.isInteger(value) && value >= 1 && value <= 200;
}

function planIsValid() {
  return epochsInputValid()
    && state.plan.length > 0
    && state.plan.length <= 50
    && state.plan.every(planEntryValid)
    && planTotalEpisodes() <= 5000;
}

function planTotalEpisodes() {
  return state.plan.reduce((sum, item) => sum + (planEntryValid(item) ? Number(item.episodes) : 0), 0);
}

function planText() {
  return state.plan.map((item) => {
    const value = planEntryValid(item) ? Number(item.episodes) : (item.episodes ?? "");
    return `${item.opponentMode}:${value}`;
  }).join(",");
}

function scriptInfo(mode) {
  return state.scripts.find((item) => item.id === mode)
    || { id: mode, name: mode || "未知", builtin: false, color: "§f" };
}

function updatePlanMeta() {
  const perEpoch = planTotalEpisodes();
  const epochs = planEpochs();
  const total = $("#tr-plan-total");
  if (!total) return;
  total.textContent = `共 ${state.plan.length} 段 · 每回合 ${perEpoch} 局 · 训练合计 ${perEpoch * epochs} 局`;
  total.className = `pill ${planIsValid() ? "ok" : "warn"}`;
  const text = $("#tr-plan-text");
  if (text) text.textContent = planText() || "—";
  const empty = $("#tr-plan-empty");
  if (empty) empty.classList.toggle("hidden", state.plan.length > 0);
  const start = $("#btn-training-start");
  if (start) {
    start.disabled = !planIsValid();
    start.title = planIsValid()
      ? ""
      : "训练回合需为 1–200 的整数；计划不能为空，每段轮次需为 1–500 的整数（每回合 ≤ 5000 局）";
  }
}

function renderPlan() {
  const list = $("#tr-plan-list");
  if (!list) return;
  list.innerHTML = state.plan.map((item, index) => {
    const info = scriptInfo(item.opponentMode);
    const valid = planEntryValid(item);
    const value = item.episodes === null || item.episodes === undefined ? "" : escapeHtml(item.episodes);
    return `<div class="plan-row${valid ? "" : " invalid"}" draggable="true" data-index="${index}">
      <span class="plan-grip" title="拖拽排序" aria-hidden="true">⠿</span>
      <span class="plan-mode" style="color:${mcColor(info.color)}">${escapeHtml(info.name)}</span>
      <span class="plan-mode-id mono">${escapeHtml(item.opponentMode)}</span>
      <input class="plan-episodes" type="number" min="1" max="500" step="1" value="${value}"
        data-index="${index}" aria-label="${escapeHtml(info.name)} 轮次" aria-invalid="${valid ? "false" : "true"}">
      <div class="plan-actions">
        <button class="btn tiny ghost" type="button" data-act="up" data-index="${index}" title="上移"${index === 0 ? " disabled" : ""}>↑</button>
        <button class="btn tiny ghost" type="button" data-act="down" data-index="${index}" title="下移"${index === state.plan.length - 1 ? " disabled" : ""}>↓</button>
        <button class="btn tiny danger" type="button" data-act="remove" data-index="${index}" title="删除">删除</button>
      </div>
    </div>`;
  }).join("");
  updatePlanMeta();
}

function setPlan(items, touched) {
  state.plan = (Array.isArray(items) ? items : []).map((item) => {
    const episodes = Number(item && item.episodes);
    return {
      opponentMode: String((item && item.opponentMode) || "scripted").trim().toLowerCase(),
      episodes: Number.isFinite(episodes) && episodes > 0 ? episodes : null,
    };
  });
  if (touched) state.planTouched = true;
  renderPlan();
}

function movePlan(index, delta) {
  const target = index + delta;
  if (index < 0 || target < 0 || target >= state.plan.length) return;
  const [item] = state.plan.splice(index, 1);
  state.plan.splice(target, 0, item);
  state.planTouched = true;
  renderPlan();
}

function parsePlanString(text) {
  const parts = String(text || "").split(",").map((part) => part.trim()).filter(Boolean);
  if (!parts.length) throw new Error("计划字符串为空");
  const known = new Set(state.scripts.map((script) => script.id));
  const items = parts.map((part) => {
    const split = part.indexOf(":");
    if (split <= 0) throw new Error(`格式应为 模式:回合数，收到 ${part}`);
    const mode = part.slice(0, split).trim().toLowerCase();
    const countText = part.slice(split + 1).trim();
    if (!known.has(mode)) throw new Error(`未知训练内容 ${mode}`);
    const count = Number(countText);
    if (!countText || !Number.isInteger(count) || count < 1 || count > 500) {
      throw new Error(`${mode} 的轮次需要 1–500 的整数`);
    }
    return { opponentMode: mode, episodes: count };
  });
  if (items.length > 50) throw new Error("计划最多 50 段");
  if (items.reduce((sum, item) => sum + item.episodes, 0) > 5000) throw new Error("总回合超过上限 5000");
  return items;
}

function renderScripts() {
  const list = $("#tr-script-list");
  if (!list) return;
  list.innerHTML = state.scripts.map((script) => `<div class="script-item${script.builtin ? "" : " custom"}">
      <span class="script-dot" style="background:${mcColor(script.color)}"></span>
      <span class="script-name" style="color:${mcColor(script.color)}">${escapeHtml(script.name || script.id)}</span>
      <span class="script-id mono">${escapeHtml(script.id)}</span>
    </div>`).join("");
  const empty = $("#tr-script-empty");
  if (empty) empty.classList.toggle("hidden", state.scripts.length > 0);
  const addModel = $("#tr-add-model");
  if (addModel) {
    addModel.disabled = !state.modelSupport;
    addModel.title = state.modelSupport ? "装载模型文件" : "模型装载待接入";
  }
}

function renderScriptOptions() {
  const select = $("#tr-add-script");
  if (!select) return;
  const previous = select.value;
  select.innerHTML = state.scripts.map((script) => `<option value="${escapeHtml(script.id)}" style="color:${mcColor(script.color)}">${escapeHtml(script.name || script.id)}</option>`).join("");
  if (state.scripts.some((script) => script.id === previous)) select.value = previous;
  select.style.color = mcColor(scriptInfo(select.value).color);
}

function applyScripts(scripts) {
  const incoming = Array.isArray(scripts) ? scripts : [];
  const next = incoming.length ? incoming : FALLBACK_SCRIPTS;
  if (JSON.stringify(next) === JSON.stringify(state.scripts)) return;
  state.scripts = next;
  renderScripts();
  renderScriptOptions();
}

function fillTrainingForm(options) {
  const d = options.defaults || {};
  const game = options.game || {};
  const note = $("#tr-game-note");
  if (note && game.port) note.textContent = ` 游戏端口 ${game.host || "127.0.0.1"}:${game.port}（自动传入 --mc-port）。`;
  const set = (id, value) => {
    const el = $(id);
    if (el && !el.dataset.touched) el.value = value;
  };
  set("#tr-users", (d.users || []).join(", "));
  set("#tr-ports", (d.ports || []).join(", "));
  set("#tr-epochs", d.epochs ?? 1);
  set("#tr-max-steps", d.maxSteps ?? 200);
  set("#tr-seed", d.seed ?? 7);
  set("#tr-eval", d.evalEpochs ?? d.evalEpisodes ?? 1);
  set("#tr-center", d.center ?? "0.5,4,0.5");
  set("#tr-spacing", d.spacing ?? 6);
  set("#tr-prep", d.prepSeconds ?? 3);
  set("#tr-pairs", d.pairs ?? 2);
  set("#tr-prefix", d.usersPrefix ?? "Bot_P");
  set("#tr-base-port", d.portsBase ?? 8770);
  const auto = $("#tr-auto-spawn");
  if (auto && !auto.dataset.touched) auto.checked = Boolean(d.autoSpawn);
  const cleanup = $("#tr-auto-cleanup");
  if (cleanup && !cleanup.dataset.touched) cleanup.checked = d.autoCleanup !== false;

  state.models = Array.isArray(options.models) ? options.models : [];
  state.modelSupport = Boolean(options.modelSupport);
  if (options.rewardPresetSchema) state.rewardSchema = options.rewardPresetSchema;
  if (Array.isArray(options.rewardPresets)) state.rewardPresets = options.rewardPresets;
  applyScripts(options.scripts);
  renderScripts();
  renderRewardPresetOptions();
  renderModelOptions();
  const modelSelect = $("#tr-model");
  if (modelSelect && !modelSelect.dataset.touched) {
    const wanted = d.modelId || "default";
    if ((state.models || []).some((item) => item.id === wanted)) modelSelect.value = wanted;
  }
  const rewardSelect = $("#tr-reward-preset");
  if (rewardSelect && !rewardSelect.dataset.touched) {
    const wanted = d.rewardPreset || "default";
    if (state.rewardPresets.some((item) => item.id === wanted)) rewardSelect.value = wanted;
  }

  // 用户编辑过计划（dataset.touched 等价物）后，不再被轮询覆盖。
  if (!state.planTouched && !state.planInitialized) {
    const plan = Array.isArray(d.plan) && d.plan.length
      ? d.plan
      : [{ opponentMode: d.opponentMode || "scripted", episodes: Number(d.episodes) || 4 }];
    setPlan(plan, false);
    state.planInitialized = true;
  }
}


/* ---------------- 训练预设 ---------------- */

function planFieldsFromForm() {
  return {
    epochs: planEpochs(),
    maxSteps: Number($("#tr-max-steps").value),
    seed: Number($("#tr-seed").value),
    evalEpochs: Number($("#tr-eval").value),
    center: $("#tr-center").value.trim(),
    spacing: Number($("#tr-spacing").value),
    prepSeconds: Number($("#tr-prep").value),
    autoSpawn: $("#tr-auto-spawn").checked,
    autoCleanup: $("#tr-auto-cleanup").checked,
  };
}

const PRESET_FIELDS = {
  epochs: "#tr-epochs",
  maxSteps: "#tr-max-steps",
  seed: "#tr-seed",
  evalEpochs: "#tr-eval",
  center: "#tr-center",
  spacing: "#tr-spacing",
  prepSeconds: "#tr-prep",
};

function applyPresetFields(fields) {
  Object.entries(PRESET_FIELDS).forEach(([key, selector]) => {
    const el = $(selector);
    if (el && fields[key] !== undefined && fields[key] !== null) {
      el.value = fields[key];
      el.dataset.touched = "1";
    }
  });
  const spawn = $("#tr-auto-spawn");
  if (spawn && typeof fields.autoSpawn === "boolean") {
    spawn.checked = fields.autoSpawn;
    spawn.dataset.touched = "1";
  }
  const cleanup = $("#tr-auto-cleanup");
  if (cleanup && typeof fields.autoCleanup === "boolean") {
    cleanup.checked = fields.autoCleanup;
    cleanup.dataset.touched = "1";
  }
}

async function loadPresets() {
  try {
    const data = await api("/api/presets");
    state.presets = data.presets || [];
    state.presetsAvailable = true;
  } catch (err) {
    // 仍在运行的旧版 Dashboard 进程没有该接口：降级为面板内提示，不弹红色错误。
    state.presets = [];
    state.presetsAvailable = false;
  }
  renderPresets();
}

function renderPresets() {
  const list = $("#tr-presets");
  if (!list) return;
  const presets = state.presets || [];
  list.innerHTML = presets.map((preset) => {
    const count = Array.isArray(preset.plan) ? preset.plan.length : 0;
    return `<div class="preset-item">
      <div class="preset-main">
        <div class="preset-name">${escapeHtml(preset.name)}</div>
        <div class="preset-sub">${count} 段 · 总回合 ${Number(preset.totalEpisodes) || 0}</div>
      </div>
      <div class="preset-actions">
        <button class="btn tiny" type="button" data-preset-load="${escapeHtml(preset.id)}">加载</button>
        <button class="btn tiny danger" type="button" data-preset-delete="${escapeHtml(preset.id)}">删除</button>
      </div>
    </div>`;
  }).join("");
  const empty = $("#tr-presets-empty");
  if (empty) empty.classList.toggle("hidden", presets.length > 0 || !state.presetsAvailable);
  const note = $("#tr-presets-note");
  if (note) {
    note.textContent = state.presetsAvailable
      ? ""
      : "当前 Dashboard 进程不支持预设接口（可能仍是旧版本）；请重启 Dashboard 后再试。";
    note.classList.toggle("hidden", state.presetsAvailable);
  }
}

async function savePreset(button) {
  if (!state.presetsAvailable) {
    toast("当前 Dashboard 进程不支持预设，请重启 Dashboard 后再试", "warn");
    return;
  }
  if (!planIsValid()) {
    toast("计划不能为空，且每段轮次需为 1–500 的整数", "err");
    return;
  }
  const input = window.prompt("预设名称", "");
  if (input === null) return;
  const name = input.trim();
  if (!name) {
    toast("预设名称不能为空", "err");
    return;
  }
  await withBusy(button, async () => {
    await api("/api/presets/save", {
      method: "POST",
      body: {
        name,
        plan: state.plan.map(({ opponentMode, episodes }) => ({ opponentMode, episodes })),
        fields: planFieldsFromForm(),
      },
    });
    toast(`预设已保存：${name}`, "ok");
    await loadPresets();
  });
}

async function loadPresetById(id, button) {
  await withBusy(button, async () => {
    const data = await api("/api/presets/load", { method: "POST", body: { id } });
    const preset = data.preset || {};
    setPlan(preset.plan || [], true);
    applyPresetFields(preset.fields || {});
    toast(`已加载预设：${preset.name || id}`, "ok");
  });
}

async function deletePresetById(id, button) {
  if (!window.confirm("删除该预设？")) return;
  await withBusy(button, async () => {
    await api("/api/presets/delete", { method: "POST", body: { id } });
    toast("预设已删除", "ok");
    await loadPresets();
  });
}



/* ---------------- 脚本注册表（含颜色） ---------------- */

const SCRIPT_ID_RE = /^[a-z0-9][a-z0-9_-]{0,23}$/;
const COLOR_CODES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f"];

function scriptColorOptions(selected) {
  return COLOR_CODES.map((code) => {
    const value = `\u00a7${code}`;
    const chosen = String(selected || "").replace(/^[&\u00a7]/, "").toLowerCase() === code;
    return `<option value="${value}"${chosen ? " selected" : ""}>§${code} ■</option>`;
  }).join("");
}

async function loadScriptRegistry() {
  try {
    const data = await api("/api/scripts");
    state.scriptItems = data.scripts || [];
  } catch (err) {
    state.scriptItems = [];
  }
  renderScriptEditorList();
}

function renderScriptEditorList() {
  const list = $("#script-editor-list");
  if (!list) return;
  const items = state.scriptItems || [];
  list.innerHTML = items.length ? items.map((item) => `
    <div class="script-editor-item${state.scriptEditing && state.scriptEditing.id === item.id ? " active" : ""}" data-script-id="${escapeHtml(item.id)}">
      <span class="script-dot" style="background:${mcColor(item.color)}"></span>
      <span class="script-name" style="color:${mcColor(item.color)}">${escapeHtml(item.name)}</span>
      <span class="script-id mono">${escapeHtml(item.id)}</span>
      ${item.builtin ? '<span class="pill neutral">内置</span>' : ""}
    </div>`).join("") : '<p class="muted small">还没有脚本。</p>';
}

function openScriptEditor(item) {
  state.scriptEditing = item ? { ...item } : { id: "", name: "", color: "§f", description: "", builtin: false };
  const editing = state.scriptEditing;
  $("#script-id").value = editing.id || "";
  $("#script-id").disabled = Boolean(editing.builtin);
  $("#script-name").value = editing.name || "";
  $("#script-color").innerHTML = scriptColorOptions(editing.color);
  $("#script-description").value = editing.description || "";
  $("#btn-script-delete").disabled = Boolean(editing.builtin);
  $("#script-status").textContent = editing.builtin ? "内置脚本：可改颜色/名称，不能删除。" : "";
  renderScriptEditorList();
  const dialog = $("#script-dialog");
  if (dialog && typeof dialog.showModal === "function") dialog.showModal();
  else if (dialog) dialog.setAttribute("open", "");
}

async function saveScriptEntry() {
  const payload = {
    id: $("#script-id").value.trim(),
    name: $("#script-name").value.trim(),
    color: $("#script-color").value,
    description: $("#script-description").value.trim(),
  };
  if (!SCRIPT_ID_RE.test(payload.id)) {
    toast("脚本 id 需为 1–24 位小写字母/数字/下划线/连字符，且以字母或数字开头", "err");
    return;
  }
  const data = await api("/api/scripts", { method: "POST", body: payload });
  state.scriptItems = data.scripts || state.scriptItems;
  await loadScriptRegistry();
  renderScriptEditorList();
  $("#script-status").textContent = `已保存：${payload.id}`;
  toast(`脚本已保存：${payload.name || payload.id}`, "ok");
  await refreshTrainingOptions();
}

async function deleteScriptEntry() {
  const ident = $("#script-id").value.trim();
  if (!ident) return;
  if (!window.confirm(`删除脚本 ${ident}？（内置脚本不可删除）`)) return;
  await api(`/api/scripts?id=${encodeURIComponent(ident)}`, { method: "DELETE" });
  toast(`已删除脚本：${ident}`, "ok");
  await loadScriptRegistry();
  const next = (state.scriptItems || [])[0];
  if (next) openScriptEditor(next);
  await refreshTrainingOptions();
}

function bindScriptEditor() {
  const manage = $("#btn-script-manage");
  if (manage) manage.addEventListener("click", (event) => withBusy(event.currentTarget, async () => {
    await loadScriptRegistry();
    openScriptEditor((state.scriptItems || [])[0]);
  }));
  const dialog = $("#script-dialog");
  if (!dialog) return;
  $("#btn-script-close").addEventListener("click", () => dialog.close());
  $("#btn-script-new").addEventListener("click", () => openScriptEditor(null));
  $("#btn-script-save").addEventListener("click", (event) => withBusy(event.currentTarget, saveScriptEntry));
  $("#btn-script-delete").addEventListener("click", (event) => withBusy(event.currentTarget, deleteScriptEntry));
  $("#script-editor-list").addEventListener("click", (event) => {
    const row = event.target.closest("[data-script-id]");
    if (!row) return;
    const item = (state.scriptItems || []).find((entry) => entry.id === row.dataset.scriptId);
    if (item) openScriptEditor(item);
  });
}

async function refreshTrainingOptions() {
  try {
    const data = await api("/api/training/options");
    state.options = data.options;
    fillTrainingForm(state.options);
    renderPlan();
  } catch (err) { /* 选项刷新失败不阻塞 */ }
}

/* ---------------- Reward 预设（权重 schema：base + terms） ---------------- */

const REWARD_ID_RE = /^[a-z0-9][a-z0-9_-]{0,31}$/;

function rewardBaseKeys() {
  const schema = state.rewardSchema;
  return Array.isArray(schema && schema.baseKeys) ? schema.baseKeys : [];
}

function rewardConditionKeys() {
  const schema = state.rewardSchema;
  return Array.isArray(schema && schema.conditionKeys) ? schema.conditionKeys : [];
}

function blankRewardPreset() {
  const base = {};
  rewardBaseKeys().forEach((item) => { base[item.key] = Number(item.default) || 0; });
  return {
    format: (state.rewardSchema && state.rewardSchema.format) || "gladiator.reward-preset/v1",
    name: "新预设",
    description: "",
    base,
    terms: [],
  };
}

function renderRewardPresetOptions() {
  const select = $("#tr-reward-preset");
  if (!select) return;
  const presets = state.rewardPresets || [];
  const current = select.value;
  select.innerHTML = presets.length
    ? presets.map((preset) => `<option value="${escapeHtml(preset.id)}">${escapeHtml(preset.name || preset.id)}${preset.termsCount ? ` · ${preset.termsCount} terms` : ""}</option>`).join("")
    : '<option value="">（没有可用预设）</option>';
  if (presets.some((preset) => preset.id === current)) select.value = current;
  const manage = $("#btn-reward-manage");
  if (manage) manage.disabled = !presets.length && !state.rewardSchema;
}

async function loadRewardPresets() {
  try {
    const data = await api("/api/reward-presets");
    state.rewardPresets = data.presets || [];
    if (data.schema) state.rewardSchema = data.schema;
  } catch (err) {
    state.rewardPresets = [];
  }
  renderRewardPresetOptions();
}

function currentRewardPresetId() {
  const select = $("#tr-reward-preset");
  return select ? select.value : "";
}

function rewardPresetLabel(job) {
  const preset = job && job.params && job.params.rewardPreset;
  if (!preset) return "";
  const suffix = preset.applied === false ? " · 本次未应用（训练侧未支持）" : "";
  return `${preset.name || preset.id}（${preset.id}）${suffix}`;
}

function renderRewardBase(base) {
  const host = $("#reward-base");
  if (!host) return;
  host.innerHTML = rewardBaseKeys().map((item) => `
    <label class="field"><span>${escapeHtml(item.label || item.key)} <code>${escapeHtml(item.key)}</code></span>
      <input id="reward-base-${escapeHtml(item.key)}" type="number" step="0.01" value="${escapeHtml(String(base[item.key] ?? item.default ?? 0))}">
    </label>`).join("");
}

function conditionKeyOptions(selected) {
  const items = rewardConditionKeys();
  const list = items.length ? items : [{ key: selected, label: selected }];
  return list.map((item) => `<option value="${escapeHtml(item.key)}"${item.key === selected ? " selected" : ""}>${escapeHtml(item.label || item.key)}（${escapeHtml(item.key)}）</option>`).join("");
}

function conditionValueControl(key, value) {
  if (key === "actionIn") {
    const text = Array.isArray(value) ? value.join(",") : String(value ?? "");
    return `<input class="cond-value" value="${escapeHtml(text)}" placeholder="block, jump">`;
  }
  if (key === "attacked") {
    const truthy = value === true || value === "true";
    return `<select class="cond-value">
      <option value="true"${truthy ? " selected" : ""}>是</option>
      <option value="false"${truthy ? "" : " selected"}>否</option>
    </select>`;
  }
  const text = value === undefined || value === null || Number.isNaN(Number(value)) ? "" : String(value);
  return `<input class="cond-value" type="number" step="0.1" value="${escapeHtml(text)}">`;
}

function renderRewardTerms(terms) {
  const host = $("#reward-terms");
  if (!host) return;
  const meta = $("#reward-terms-meta");
  if (meta) meta.textContent = terms.length ? `${terms.length} 条` : "还没有附加项";
  host.innerHTML = terms.map((term, index) => {
    const conds = Object.entries(term.when || {}).map(([key, value]) => `
      <div class="reward-cond">
        <select class="cond-key">${conditionKeyOptions(key)}</select>
        ${conditionValueControl(key, value)}
        <button class="btn tiny ghost" data-cond-remove="1" type="button" title="删除该条件">×</button>
      </div>`).join("");
    return `<div class="reward-term" data-index="${index}">
      <div class="reward-term-head">
        <input class="term-id" placeholder="term id（如 threat）" value="${escapeHtml(term.id || "")}">
        <input class="term-weight" type="number" step="0.01" placeholder="权重" value="${term.weight === undefined || term.weight === null ? "" : escapeHtml(String(term.weight))}">
        <select class="term-unit">
          <option value="second"${term.unit !== "step" ? " selected" : ""}>按秒（second）</option>
          <option value="step"${term.unit === "step" ? " selected" : ""}>按步（step）</option>
        </select>
        <button class="btn tiny danger" data-term-remove="${index}" type="button">删除</button>
      </div>
      <div class="reward-conds">${conds || '<p class="muted small">没有条件：该 term 每一步（或每秒）恒定结算。</p>'}</div>
      <button class="btn tiny ghost" data-cond-add="${index}" type="button">添加条件</button>
    </div>`;
  }).join("");
}

function renderRewardEditor() {
  const editor = state.rewardEditor;
  if (!editor) return;
  $("#reward-id").value = editor.id || "";
  $("#reward-name").value = editor.preset.name || "";
  $("#reward-description").value = editor.preset.description || "";
  const title = $("#reward-title");
  if (title) title.textContent = `Reward 预设 · ${editor.preset.name || editor.id || "新建"}`;
  renderRewardBase(editor.preset.base || {});
  renderRewardTerms(editor.preset.terms || []);
  const del = $("#btn-reward-delete");
  const builtin = editor.id === ((state.rewardSchema && state.rewardSchema.defaultId) || "default");
  if (del) del.disabled = builtin;
  if (builtin) setRewardStatus("内置 default 可以覆盖保存，但删除按钮已禁用。");
}

function setRewardStatus(text) {
  const el = $("#reward-status");
  if (el) el.textContent = text || "";
}

function readRewardEditor() {
  const base = {};
  rewardBaseKeys().forEach((item) => {
    const input = $(`#reward-base-${item.key}`);
    if (!input) return;
    const raw = input.value.trim();
    base[item.key] = raw === "" ? Number.NaN : Number(raw);
  });
  const terms = $$("#reward-terms .reward-term").map((row) => {
    const when = {};
    $$(".reward-cond", row).forEach((condRow) => {
      const key = $(".cond-key", condRow).value;
      const control = $(".cond-value", condRow);
      if (!key || !control) return;
      if (key === "actionIn") {
        when[key] = control.value.split(/[\s,]+/).map((part) => part.trim()).filter(Boolean);
      } else if (key === "attacked") {
        when[key] = control.value === "true";
      } else {
        const raw = control.value.trim();
        when[key] = raw === "" ? Number.NaN : Number(raw);
      }
    });
    const weightRaw = $(".term-weight", row).value.trim();
    return {
      id: $(".term-id", row).value.trim(),
      weight: weightRaw === "" ? Number.NaN : Number(weightRaw),
      unit: $(".term-unit", row).value,
      when,
    };
  });
  return {
    format: (state.rewardSchema && state.rewardSchema.format) || "gladiator.reward-preset/v1",
    name: $("#reward-name").value.trim(),
    description: $("#reward-description").value.trim(),
    base,
    terms,
  };
}

/* 结构性改动（增删 term/条件）前先把 DOM 状态收进 state，避免重渲染丢掉用户刚输入的 id。 */
function syncRewardEditorFromDom() {
  const preset = readRewardEditor();
  const id = ($("#reward-id").value || "").trim();
  state.rewardEditor = {
    id: id || (state.rewardEditor && state.rewardEditor.id) || "",
    preset,
  };
  return state.rewardEditor;
}

async function openRewardEditor(presetId) {
  const id = presetId || currentRewardPresetId()
    || ((state.rewardSchema && state.rewardSchema.defaultId) || "default");
  let preset = null;
  if (id) {
    try {
      preset = (await api(`/api/reward-presets/${encodeURIComponent(id)}`)).preset;
    } catch (err) {
      preset = null;
    }
  }
  state.rewardEditor = { id, preset: preset || blankRewardPreset() };
  renderRewardEditor();
  setRewardStatus(preset ? "" : "该 id 还没有预设文件，保存后会新建。");
  const dialog = $("#reward-dialog");
  if (dialog && typeof dialog.showModal === "function") dialog.showModal();
  else if (dialog) dialog.setAttribute("open", "");
}

async function saveRewardPreset() {
  const preset = readRewardEditor();
  const id = $("#reward-id").value.trim();
  if (!REWARD_ID_RE.test(id)) {
    toast("预设 id 需为 1–32 位小写字母/数字/下划线/连字符，且以字母或数字开头", "err");
    return;
  }
  const data = await api("/api/reward-presets", { method: "POST", body: { id, preset } });
  const saved = data.preset || preset;
  state.rewardEditor = { id, preset: saved };
  renderRewardEditor();
  await loadRewardPresets();
  await loadModels();
  const select = $("#tr-reward-preset");
  if (select) {
    select.value = id;
    select.dataset.touched = "1";
  }
  setRewardStatus(`已保存：${id}`);
  toast(`reward 预设已保存：${saved.name || id}`, "ok");
}

async function deleteRewardPreset() {
  const id = $("#reward-id").value.trim();
  if (!id) return;
  if (!window.confirm(`删除 reward 预设 ${id}？（会移到 var/reward-presets/trash/ 可恢复）`)) return;
  await api(`/api/reward-presets/${encodeURIComponent(id)}`, { method: "DELETE" });
  toast(`reward 预设已删除：${id}`, "ok");
  await loadRewardPresets();
  await loadModels();
  const fallback = currentRewardPresetId();
  if (fallback) await openRewardEditor(fallback);
  else {
    state.rewardEditor = { id: "", preset: blankRewardPreset() };
    renderRewardEditor();
  }
}

async function copyDefaultRewardPreset() {
  const defaultId = (state.rewardSchema && state.rewardSchema.defaultId) || "default";
  let preset;
  try {
    preset = (await api(`/api/reward-presets/${encodeURIComponent(defaultId)}`)).preset;
  } catch (err) {
    preset = blankRewardPreset();
  }
  const copy = JSON.parse(JSON.stringify(preset));
  copy.name = `${copy.name || "默认"} 副本`;
  state.rewardEditor = { id: `copy-${Date.now().toString(36)}`, preset: copy };
  renderRewardEditor();
  setRewardStatus("已载入默认预设内容；改 id / 名称后点「保存预设」即新建一份。");
}

function bindRewardEditor() {
  const dialog = $("#reward-dialog");
  if (!dialog) return;
  $("#btn-reward-close").addEventListener("click", () => dialog.close());
  $("#btn-reward-copy").addEventListener("click", (event) => withBusy(event.currentTarget, copyDefaultRewardPreset));
  $("#btn-reward-save").addEventListener("click", (event) => withBusy(event.currentTarget, saveRewardPreset));
  $("#btn-reward-delete").addEventListener("click", (event) => withBusy(event.currentTarget, deleteRewardPreset));
  $("#btn-reward-add-term").addEventListener("click", () => {
    const editor = syncRewardEditorFromDom();
    editor.preset.terms.push({ id: `term${editor.preset.terms.length + 1}`, weight: 0, unit: "step", when: {} });
    renderRewardEditor();
  });
  $("#reward-terms").addEventListener("click", (event) => {
    const removeTerm = event.target.closest("[data-term-remove]");
    if (removeTerm) {
      const editor = syncRewardEditorFromDom();
      editor.preset.terms.splice(Number(removeTerm.dataset.termRemove), 1);
      renderRewardEditor();
      return;
    }
    const addCond = event.target.closest("[data-cond-add]");
    if (addCond) {
      const editor = syncRewardEditorFromDom();
      const term = editor.preset.terms[Number(addCond.dataset.condAdd)];
      if (!term) return;
      const used = new Set(Object.keys(term.when || {}));
      const next = rewardConditionKeys().map((item) => item.key).find((key) => !used.has(key)) || "distanceMin";
      term.when[next] = next === "actionIn" ? ["block"] : (next === "attacked" ? true : 0);
      renderRewardEditor();
      return;
    }
    const removeCond = event.target.closest("[data-cond-remove]");
    if (removeCond) {
      const termRow = removeCond.closest(".reward-term");
      const condRow = removeCond.closest(".reward-cond");
      const key = $(".cond-key", condRow).value;
      const editor = syncRewardEditorFromDom();
      const term = editor.preset.terms[Number(termRow.dataset.index)];
      if (term && key) {
        delete term.when[key];
        renderRewardEditor();
      }
    }
  });
  ["#reward-id", "#reward-name", "#reward-description"].forEach((selector) => {
    const el = $(selector);
    if (el) el.addEventListener("input", () => { setRewardStatus(""); });
  });
  const manage = $("#btn-reward-manage");
  if (manage) manage.addEventListener("click", () => withBusy(manage, () => openRewardEditor()));
}



/* ---------------- 本机文件夹 ---------------- */

async function openFolder(kind, id) {
  try {
    const data = await api("/api/open-folder", { method: "POST", body: id ? { kind, id } : { kind } });
    toast(`已打开：${data.path}`, "ok");
    return data;
  } catch (err) {
    toast(err.message || "打开文件夹失败", "err");
    return null;
  }
}

/* ---------------- 模型注册表 ---------------- */

function fmtBytes(value) {
  const size = Number(value) || 0;
  if (size <= 0) return "—";
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / 1024 / 1024).toFixed(2)} MB`;
}

async function loadModels() {
  try {
    const data = await api("/api/models");
    state.models = data.models || [];
    state.gmodDir = data.gmodDir || null;
    state.gmodScanned = Number(data.scanned) || 0;
    state.gmodInvalid = data.invalid || [];
  } catch (err) {
    state.models = [];
    state.gmodDir = null;
    state.gmodScanned = 0;
    state.gmodInvalid = [];
  }
  renderModels();
  renderModelOptions();
  renderModelSourceRuns();
}

async function reloadModels(button) {
  await withBusy(button, async () => {
    const data = await api("/api/models/reload", { method: "POST", body: {} });
    state.models = data.models || [];
    state.gmodDir = data.gmodDir || state.gmodDir;
    state.gmodScanned = Number(data.scanned) || 0;
    state.gmodInvalid = data.invalid || [];
    renderModels();
    renderModelOptions();
    const bad = state.gmodInvalid.length;
    toast(`已刷新：扫描 ${state.gmodScanned} 个 .gmod，新增 ${data.added || 0} 个${bad ? `，${bad} 个失败` : ""}`, bad ? "err" : "ok");
    if (bad) state.gmodInvalid.forEach((item) => toast(`${item.file}：${item.error}`, "err"));
  });
}

function syncPersistControls() {
  const box = $("#tr-persist");
  if (!box) return;
  const select = $("#tr-model");
  const id = select ? select.value : "default";
  const model = (state.models || []).find((item) => item.id === id) || null;
  const kind = $("#tr-kind") ? $("#tr-kind").value : "train";
  const isGmod = Boolean(model && model.kind === "gmod");
  const isMulti = kind === "multi";
  // 多场地 + 融合（mean/weighted）才允许持久化写回融合结果；merge=none 仍禁用。
  const merge = $("#tr-merge") ? $("#tr-merge").value : "mean";
  const multiMergeEnabled = isMulti && merge && merge !== "none";
  const blocked = isGmod || (isMulti && !multiMergeEnabled) || Boolean(model && model.valid === false);
  state.importGmodId = isGmod ? id : null;
  box.disabled = blocked;
  if (blocked) box.checked = false;
  else if (!state.persistTouched) box.checked = true;  // 默认开启
  const note = $("#tr-persist-note");
  if (note) {
    note.textContent = isGmod
      ? "文件模型（.gmod）只读，写回已禁用；点「导入并继续训练」可生成可持久化的策略模型。"
      : isMulti
        ? (multiMergeEnabled
            ? "多场地已开启融合：训练结束会把多个场地的权重平均成唯一一份，并写回所选模型（累加局数/步数）；融合失败则自动降级为不写回。"
            : "多场地训练需先选「融合 = 平均/按步数加权」才能持久化；选「不融合」或融合失败会自动降级为不写回模型。")
        : "开启后，正常完成的训练会把最终权重写回所选模型（累加局数/步数）；失败或被停止的训练不写回。关闭则只读起点，产物仅留在本次 run 里。";
  }
  const actions = $("#tr-gmod-actions");
  if (actions) actions.classList.toggle("hidden", !isGmod);
  const gmodNote = $("#tr-gmod-note");
  if (gmodNote && isGmod) {
    gmodNote.textContent = `文件模型只读：${(model.source && model.source.file) || id}`;
  }
}

async function importGmodAndContinue(button) {
  const id = state.importGmodId;
  if (!id) return;
  const model = (state.models || []).find((item) => item.id === id);
  await withBusy(button, async () => {
    const data = await api("/api/models", {
      method: "POST",
      body: {
        name: `${(model && model.name) || id}（副本）`,
        gmodId: id,
        description: `从 ${(model && model.source && model.source.file) || id} 导入，用于持久化训练`,
      },
    });
    await loadModels();
    const select = $("#tr-model");
    if (select) {
      select.value = data.model.id;
      select.dataset.touched = "1";
    }
    state.persistTouched = false;
    syncPersistControls();
    toast(`已导入为策略模型 ${data.model.id}，现在可以持久化训练`, "ok");
  });
}

function modelSourceLabel(model) {
  if (model.builtin) return "内置 · default";
  const source = model.source || {};
  if (model.kind === "gmod") {
    return `文件 · ${escapeHtml(source.dir || "models")}/${escapeHtml(source.file || `${model.id}.gmod`)}`;
  }
  return source.runId ? `策略产物 · ${escapeHtml(source.runId)}` : "策略产物";
}

function renderModelOptions() {
  const select = $("#tr-model");
  if (!select) return;
  const current = select.value;
  const models = state.models || [];
  const option = (model) => {
    const dims = model.obsDim && model.numActions ? ` · ${model.obsDim}×${model.numActions}` : "";
    const invalid = model.valid === false;
    const label = `${model.name}${dims}${model.builtin ? "（默认）" : ""}`
      + (invalid ? ` · 不可用：${model.error || "校验失败"}` : "");
    return `<option value="${escapeHtml(model.id)}"${invalid ? " disabled" : ""}>${escapeHtml(label)}</option>`;
  };
  const group = (label, items) => items.length
    ? `<optgroup label="${escapeHtml(label)}">${items.map(option).join("")}</optgroup>` : "";
  const policy = models.filter((model) => model.kind !== "gmod");
  const policyGroup = policy.length ? group("策略产物", policy)
    : '<optgroup label="策略产物"><option value="default">default（全新初始化）</option></optgroup>';
  select.innerHTML = policyGroup
    + group("文件模型（.gmod）", models.filter((model) => model.kind === "gmod"))
    + '<option value="__import__">＋ 从 run 导入…</option>';
  const stillValid = models.some((model) => model.id === current && model.valid !== false);
  select.value = stillValid ? current : "default";
  syncPersistControls();
}

function renderModelSourceRuns() {
  const select = $("#model-source-run");
  if (!select) return;
  const runs = (state.runs || []).filter((run) => run.status !== "running");
  select.innerHTML = runs.length
    ? runs.map((run) => `<option value="${escapeHtml(run.id)}">${escapeHtml(run.id)} · ${escapeHtml(run.kind || "")} · ${num(run.episodes)} 局</option>`).join("")
    : '<option value="">（没有可导入的 run）</option>';
}

function renderModels() {
  const rows = $("#models-rows");
  if (!rows) return;
  const models = state.models || [];
  rows.innerHTML = models.map((model) => {
    const invalid = model.valid === false;
    const dims = model.obsDim && model.numActions ? `${model.obsDim} × ${model.numActions}` : "—";
    const size = model.builtin ? "无文件" : fmtBytes(model.sizeBytes);
    const status = invalid
      ? `<span class="pill bad" title="${escapeHtml(model.error || "校验失败")}">不可用</span>`
      : '<span class="pill ok">正常</span>';
    const warning = model.warning ? `<div class="muted small">${escapeHtml(model.warning)}</div>` : "";
    const error = invalid ? `<div class="model-error">${escapeHtml(model.error || "校验失败")}</div>` : "";
    const training = model.training;
    const trainLine = training && training.lastTrainedAt
      ? `<div class="muted small">最近训练：${escapeHtml(training.lastTrainedAt)} · 累计 ${num(training.totalEpisodes)} 局 / ${num(training.totalSteps)} 步 · 最近均值 reward ${training.lastMeanReward == null ? "—" : num(training.lastMeanReward, 2)}</div>`
      : (model.kind === "gmod" ? "" : '<div class="muted small">最近训练：尚未写回</div>');
    return `<tr class="${invalid ? "models-row-invalid" : ""}" data-model-row="${escapeHtml(model.id)}">
      <td>${escapeHtml(model.name)}${model.builtin ? ' <span class="pill neutral">内置</span>' : ""}${model.conflict ? ' <span class="pill neutral" title="与同名可持久化模型并存，本行只读">同名只读副本</span>' : ""}${warning}${error}${trainLine}</td>
      <td class="mono">${escapeHtml(model.id)}</td>
      <td class="mono">${modelSourceLabel(model)}${model.kind === "gmod" ? ' <span class="pill neutral">.gmod</span>' : ""}</td>
      <td>${size}</td>
      <td>${status}</td>
      <td>${dims}</td>
      <td>${num(model.episodes)}</td>
      <td>${model.meanReward == null ? "—" : num(model.meanReward, 2)}</td>
      <td>${fmtTime(model.updatedAt || model.createdAt)}</td>
      <td><button class="btn tiny ghost" data-model-detail="${escapeHtml(model.id)}" type="button">详情</button>${
        model.kind !== "gmod" && model.hasPolicy
          ? ` <button class="btn tiny ghost" data-model-export="${escapeHtml(model.id)}" type="button" title="把权重与训练统计打成 models/${escapeHtml(model.id)}.gmod">导出 .gmod</button>`
          : ""
      }</td>
    </tr>`;
  }).join("");
  const empty = $("#models-empty");
  if (empty) empty.classList.toggle("hidden", models.length > 0);
  const meta = $("#models-meta");
  if (meta) {
    const bad = (state.gmodInvalid || []).length;
    meta.textContent = `${models.length} 个（含内置 default） · .gmod 扫描 ${state.gmodScanned || 0} 个${bad ? ` · ${bad} 个失败` : ""}`;
  }
  const dir = $("#models-gmod-dir");
  if (dir && state.gmodDir) dir.textContent = `${state.gmodDir}/`;
}

async function openModelDetail(id) {
  const card = $("#model-detail");
  if (!card) return;
  try {
    const data = await api(`/api/models?id=${encodeURIComponent(id)}`);
    state.modelDetail = data.model;
    state.modelEditingId = id;
  } catch (err) {
    toast(err.message || "读取模型失败", "err");
    return;
  }
  const model = state.modelDetail;
  const isGmod = model.kind === "gmod";
  card.classList.remove("hidden");
  $("#model-detail-title").textContent = `模型详情 · ${model.name}`;
  const sha = model.sha256 ? `<code>${escapeHtml(model.sha256.slice(0, 16))}…</code>` : "—";
  const status = model.valid === false
    ? `<span class="pill bad">不可用</span> <span class="model-error">${escapeHtml(model.error || "校验失败")}</span>`
    : '<span class="pill ok">正常</span>';
  $("#model-detail-body").innerHTML = `<div class="kv">
    ${kvItem("id", `<code>${escapeHtml(model.id)}</code>`)}
    ${kvItem("类型 / 状态", `${isGmod ? "文件模型（.gmod）" : "策略产物"} · ${status}`)}
    ${kvItem("是否内置", model.builtin ? "是（default = 全新初始化）" : "否")}
    ${kvItem("来源", modelSourceLabel(model))}
    ${isGmod ? kvItem("包文件", `<code>${escapeHtml(model.packagePath || (model.source && model.source.file) || "—")}</code>`) : ""}
    ${isGmod ? kvItem("颜色", model.color ? `<code>${escapeHtml(model.color)}</code>（仅元数据展示，未下发到游戏）` : "未设置") : ""}
    ${model.warning ? kvItem("提示", escapeHtml(model.warning)) : ""}
    ${kvItem("obs × actions × hidden", `${num(model.obsDim)} × ${num(model.numActions)} × ${num(model.hidden)}`)}
    ${kvItem("来源 run", model.source && model.source.runId ? `<code>${escapeHtml(model.source.runId)}</code>` : "—")}
    ${kvItem("pair", model.source && model.source.pair ? String(model.source.pair) : "—")}
    ${kvItem("训练统计", `${num(model.episodes)} 局 / ${num(model.steps)} 步 / 均 reward ${model.meanReward == null ? "—" : num(model.meanReward, 2)}`)}
    ${isGmod ? kvItem("持久化训练", "只读文件模型，不写回；请先导入为策略模型") : kvItem("持久化训练", model.training && model.training.lastTrainedAt
      ? `最近 ${escapeHtml(model.training.lastTrainedAt)} · 累计 ${num(model.training.totalEpisodes)} 局 / ${num(model.training.totalSteps)} 步 · 写回 ${num(model.training.writeCount)} 次`
      : "尚未写回（首次持久化训练后会生成 policy.npz）")}
    ${kvItem("策略文件", model.policyPath ? `<code>${escapeHtml(model.policyPath)}</code>` : (isGmod ? "包内 policy.npz（训练时解到 gmod-cache）" : "尚未生成"))}
    ${kvItem("大小 / sha256", `${fmtBytes(model.sizeBytes)} / ${sha}`)}
    ${kvItem("更新时间", fmtTime(model.updatedAt))}
  </div>`;
  $("#model-edit-name").value = model.name || "";
  $("#model-edit-description").value = model.description || "";
  $("#model-edit-note").value = model.note || "";
  $("#btn-model-save").disabled = isGmod;
  $("#btn-model-delete").disabled = Boolean(model.builtin) || isGmod;
  const training = model.training || null;
  const writeRuns = training && Array.isArray(training.runs) ? training.runs.slice().reverse() : [];
  const recent = model.recentRuns || [];
  const list = $("#model-recent-runs");
  if (list) {
    if (isGmod) {
      list.innerHTML = '<p class="muted small">.gmod 是只读文件包，没有训练累计；导入为策略模型后可查看写回记录。</p>';
    } else if (writeRuns.length) {
      list.innerHTML = writeRuns.map((run) => `<div class="review-item"><div><div class="r-title mono">${escapeHtml(run.runId || "—")}</div>
        <div class="r-sub">${escapeHtml(run.at || "")} · ${num(run.episodes)} 局 / ${num(run.steps)} 步 · 均值 reward ${run.meanReward == null ? "—" : num(run.meanReward, 2)}</div></div>
        ${run.runId ? `<button class="btn tiny ghost" data-run-jump="${escapeHtml(run.runId)}" type="button">查看 run</button>` : ""}</div>`).join("");
    } else {
      list.innerHTML = recent.length
        ? recent.map((run) => `<div class="review-item"><div><div class="r-title mono">${escapeHtml(run.runId)}</div>
            <div class="r-sub">${escapeHtml(run.kind || "—")} · ${fmtTime(run.createdAt || "")}</div></div>
            <button class="btn tiny ghost" data-run-jump="${escapeHtml(run.runId)}" type="button">查看 run</button></div>`).join("")
        : '<p class="muted small">还没有持久化训练记录。</p>';
    }
  }
  $("#model-status").textContent = "";
  card.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

async function renderModelViz(id) {
  const card = $("#model-viz");
  const body = $("#model-viz-body");
  if (!card || !body) return;
  card.classList.remove("hidden");
  body.innerHTML = '<div class="muted">加载中…</div>';
  let data;
  try {
    data = await api(`/api/models/viz?id=${encodeURIComponent(id)}`);
  } catch (err) {
    body.innerHTML = `<div class="empty">可视化加载失败：${escapeHtml(err.message)}</div>`;
    return;
  }
  const m = data.model || {};
  const pct = (v) => (v == null ? "—" : (v * 100).toFixed(1) + "%");
  const f2 = (v) => (v == null ? "—" : Number(v).toFixed(2));
  const vizCard = (title, inner, wide) =>
    `<section class="viz-card${wide ? " viz-wide" : ""}"><h4>${escapeHtml(title)}</h4>${inner}</section>`;

  const info = `<div class="kv">
    ${kvItem("id", `<code>${escapeHtml(m.id || "")}</code>`)}
    ${kvItem("名称", escapeHtml(m.name || ""))}
    ${kvItem("维度 (obs×actions×hidden)", `${num(m.obsDim)} × ${num(m.numActions)} × ${num(m.hidden)}`)}
    ${kvItem("训练累计", `${num((m.training || {}).totalEpisodes)} 局 / ${num((m.training || {}).totalSteps)} 步 / 写回 ${num((m.training || {}).writeCount)} 次`)}
    ${kvItem("最近训练", escapeHtml((m.training || {}).lastTrainedAt || "—"))}
    ${kvItem("sha256", m.sha256 ? `<code>${escapeHtml(m.sha256.slice(0, 16))}…</code>` : "—")}
    ${kvItem("说明", escapeHtml(m.description || "—"))}
  </div>`;

  const w = data.weights || {};
  let weightsHtml;
  if (!w.available) {
    weightsHtml = `<div class="empty">暂无数据${w.note ? "：" + escapeHtml(w.note) : ""}</div>`;
  } else {
    const hist = (w.histogram && w.histogram.available)
      ? histogramChart(w.histogram.binEdges, w.histogram.counts)
      : '<div class="empty">暂无数据</div>';
    const norms = w.norms ? metricList(Object.keys(w.norms).map((k) => ({ label: `‖${k}‖₂`, value: f2(w.norms[k]) }))) : "";
    weightsHtml = `<div class="viz-sub">权重直方图（共 ${num(w.count)} 个参数）</div>
      <div class="chart">${hist}</div>
      <div class="viz-sub">各参数 L2 范数</div>${norms}`;
  }

  const bd = data.backupDiff || {};
  let backupHtml;
  if (!bd.available) {
    backupHtml = `<div class="empty">暂无数据${bd.note ? "：" + escapeHtml(bd.note) : ""}</div>`;
  } else {
    backupHtml = metricList([
      { label: "L2 相对变化 ‖w−w₀‖/‖w₀‖", value: f2(bd.l2RelativeChange) },
      { label: "余弦相似度 cos(w, w₀)", value: f2(bd.cosineSimilarity) },
    ]) + `<div class="muted small">备份：${escapeHtml(bd.backupPath || "")}</div>`;
  }

  const r = data.runs || {};
  let runsHtml;
  if (!r.available) {
    runsHtml = vizCard("训练统计（曲线 / 指标 / 动作分布 / 累计表）",
      `<div class="empty">暂无数据${r.note ? "：" + escapeHtml(r.note) : ""}</div>`, true);
  } else {
    const byRun = multiLineChart(r.byRun.series);
    const cumVals = r.cumulative.points.map((p) => ({
      value: p.cumulative, label: "#" + p.i, tip: `累计 ${p.cumulative.toFixed(2)}（本局 ${p.reward}）`,
    }));
    const cum = lineChart(cumVals);
    const mt = r.metrics;
    const metrics = metricList([
      { label: "胜率 winRate", value: pct(mt.winRate) },
      { label: "命中率 hitRate", value: pct(mt.hitRate) },
      { label: "空挥率 whiffRate", value: pct(mt.whiffRate) },
      { label: "伤害比 damageRatio", value: f2(mt.damageRatio) },
      { label: "平均时长 (s)", value: f2(mt.avgDuration) },
      { label: "样本 episode", value: num(mt.episodeCount) },
    ]);
    const oppRows = r.opponents.map((o) => `<tr><td>${escapeHtml(o.opponentMode)}</td><td>${num(o.nEpisodes)}</td><td>${pct(o.winRate)}</td><td>${pct(o.hitRate)}</td><td>${f2(o.damageRatio)}</td><td>${f2(o.avgReward)}</td><td>${f2(o.avgDuration)}</td></tr>`).join("");
    const oppTable = `<div class="table-wrap"><table><thead><tr><th>对手风格</th><th>样本</th><th>胜率</th><th>命中率</th><th>伤害比</th><th>均 reward</th><th>均时长</th></tr></thead><tbody>${oppRows}</tbody></table></div>`;
    const actBar = barChart(r.actions.map((a) => ({ label: a.action, value: a.fraction })));
    const t = r.table;
    const trows = t.rows.map((row) => `<tr><td class="mono">${escapeHtml(row.runId)}</td><td>${num(row.episodes)}</td><td>${num(row.steps)}</td><td>${f2(row.meanReward)}</td><td>${pct(row.winRate)}</td><td>${pct(row.hitRate)}</td><td>${f2(row.damageRatio)}</td><td>${f2(row.avgDuration)}</td></tr>`).join("");
    const tot = t.totals;
    const tfoot = `<tr class="totals"><td>合计</td><td>${num(tot.episodes)}</td><td>${num(tot.steps)}</td><td>${f2(tot.meanReward)}</td><td>${pct(tot.winRate)}</td><td>${pct(tot.hitRate)}</td><td>${f2(tot.damageRatio)}</td><td>${f2(tot.avgDuration)}</td></tr>`;
    const statTable = `<div class="table-wrap"><table><thead><tr><th>run</th><th>episodes</th><th>steps</th><th>均 reward</th><th>胜率</th><th>命中率</th><th>伤害比</th><th>均时长</th></tr></thead><tbody>${trows}</tbody><tfoot>${tfoot}</tfoot></table></div>`;
    runsHtml = [
      vizCard("累计 reward 曲线（按 run）", `<div class="chart">${byRun}</div>`),
      vizCard("累计 reward 曲线（累计和）", `<div class="chart">${cum}</div>`),
      vizCard("效果指标", metrics),
      vizCard("对手分组对比", oppTable),
      vizCard("动作分布（11 动作 · 占比）", `<div class="chart">${actBar}</div>`),
      vizCard("累计统计表", statTable, true),
    ].join("");
  }

  $("#model-viz-title").textContent = `模型可视化 · ${m.name || id}`;
  body.innerHTML = `<div class="viz-grid">
    ${vizCard("基本信息", info)}
    ${vizCard("权重直方图与范数", weightsHtml, true)}
    ${vizCard("与最近备份的权重差异", backupHtml)}
    ${runsHtml}
  </div>`;
  card.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

async function saveModelEdit(button) {
  const id = state.modelEditingId;
  if (!id) return;
  await withBusy(button, async () => {
    const data = await api("/api/models/update", {
      method: "POST",
      body: {
        id,
        name: $("#model-edit-name").value.trim(),
        description: $("#model-edit-description").value.trim(),
        note: $("#model-edit-note").value.trim(),
      },
    });
    state.models = data.models || state.models;
    renderModels();
    renderModelOptions();
    $("#model-status").textContent = "已保存";
    toast("模型信息已保存", "ok");
  });
}

async function deleteModel(button) {
  const id = state.modelEditingId;
  if (!id) return;
  if (!window.confirm(`删除模型 ${id}？（会移到 models-trash，可恢复）`)) return;
  await withBusy(button, async () => {
    const data = await api(`/api/models?id=${encodeURIComponent(id)}`, { method: "DELETE" });
    state.models = data.models || state.models;
    renderModels();
    renderModelOptions();
    $("#model-detail").classList.add("hidden");
    state.modelEditingId = null;
    toast(`模型已删除：${id}`, "ok");
  });
}

async function exportModelGmod(button) {
  const id = button.dataset.modelExport;
  if (!id) return;
  // 已存在同名 .gmod 时确认覆盖（后端默认 overwrite=false，会返回 409）。
  const existing = (state.models || []).find((item) => item.kind === "gmod" && item.id === id);
  let overwrite = false;
  if (existing) {
    const file = (existing.source && existing.source.file) || `${id}.gmod`;
    if (!window.confirm(`models/${file} 已存在，覆盖导出？`)) return;
    overwrite = true;
  }
  const label = button.textContent;
  button.disabled = true;
  button.textContent = "导出中…";
  try {
    const data = await api("/api/models/export", { method: "POST", body: { id, overwrite } });
    toast(`已导出：${data.relativePath || data.path}`, "ok");
    button.title = data.path || "";
    await loadModels();
  } catch (err) {
    toast(err.message || "导出 .gmod 失败", "err");
  } finally {
    button.disabled = false;
    button.textContent = label;
  }
}

async function importModel(button) {
  const runId = ($("#model-source-run").value || "").trim();
  const policyPath = ($("#model-policy-path").value || "").trim();
  const name = ($("#model-import-name").value || "").trim() || runId || "imported-model";
  await withBusy(button, async () => {
    const body = {
      name,
      description: ($("#model-import-description").value || "").trim(),
      note: ($("#model-import-note").value || "").trim(),
    };
    if (policyPath) body.policyPath = policyPath;
    else if (runId) body.sourceRunId = runId;
    else throw new Error("请选择来源 run 或填写 policyPath");
    const data = await api("/api/models", { method: "POST", body });
    state.models = data.models || state.models;
    renderModels();
    renderModelOptions();
    $("#model-import-status").textContent = `已导入：${data.model.id}`;
    toast(`模型已导入：${data.model.name}`, "ok");
    await openModelDetail(data.model.id);
  });
}

function bindModels() {
  const detail = $("#model-detail");
  const importCard = $("#model-import-card");
  $("#tr-persist")?.addEventListener("change", () => { state.persistTouched = true; });
  $("#btn-model-import-continue")?.addEventListener("click", (event) => importGmodAndContinue(event.currentTarget));
  $("#tr-kind")?.addEventListener("change", syncPersistControls);
  $("#tr-merge")?.addEventListener("change", syncPersistControls);
  const manage = $("#btn-model-manage");
  if (manage) manage.addEventListener("click", () => showTab("models"));
  $("#btn-model-import-open")?.addEventListener("click", () => {
    importCard?.classList.remove("hidden");
    renderModelSourceRuns();
    $("#model-import-card")?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  });
  $("#btn-model-import-close")?.addEventListener("click", () => importCard?.classList.add("hidden"));
  $("#btn-model-import")?.addEventListener("click", (event) => withBusy(event.currentTarget, () => importModel(event.currentTarget)));
  $("#btn-model-save")?.addEventListener("click", (event) => saveModelEdit(event.currentTarget));
  $("#btn-model-delete")?.addEventListener("click", (event) => deleteModel(event.currentTarget));
  $("#btn-model-detail-close")?.addEventListener("click", () => detail?.classList.add("hidden"));
  $("#btn-models-reload")?.addEventListener("click", (event) => reloadModels(event.currentTarget));
  $("#btn-models-folder")?.addEventListener("click", () => openFolder("gmod"));
  $("#btn-model-detail-folder")?.addEventListener("click", () =>
    openFolder(state.modelDetail && state.modelDetail.kind === "gmod" ? "gmod" : "models"));
  const rows = $("#models-rows");
  if (rows) rows.addEventListener("click", (event) => {
    const exportButton = event.target.closest("[data-model-export]");
    if (exportButton) {
      exportModelGmod(exportButton);
      return;
    }
    const button = event.target.closest("[data-model-detail]");
    if (button) openModelDetail(button.dataset.modelDetail);
  });
  $("#model-recent-runs")?.addEventListener("click", (event) => {
    const jump = event.target.closest("[data-run-jump]");
    if (!jump) return;
    showTab("data");
    selectRun(jump.dataset.runJump);
  });
  $("#btn-model-viz")?.addEventListener("click", () => {
    if (state.modelEditingId) renderModelViz(state.modelEditingId);
  });
  $("#btn-model-viz-close")?.addEventListener("click", () => $("#model-viz")?.classList.add("hidden"));
  $("#btn-model-viz-copy")?.addEventListener("click", () => {
    const id = state.modelEditingId || "";
    const link = `${location.origin}${location.pathname}?model=${encodeURIComponent(id)}#model-viz`;
    copyText(link);
  });
  const modelSelect = $("#tr-model");
  if (modelSelect) modelSelect.addEventListener("change", () => {
    if (modelSelect.value === "__import__") {
      // 只是个入口，不是可选模型：回到 default 并打开模型页的导入卡片。
      modelSelect.value = "default";
      showTab("models");
      const card = $("#model-import-card");
      if (card) card.classList.remove("hidden");
      renderModelSourceRuns();
      card?.scrollIntoView({ behavior: "smooth", block: "nearest" });
      return;
    }
    modelSelect.dataset.touched = "1";
    syncPersistControls();
  });
  syncPersistControls();
}

/* ---------------- 对局详情 ---------------- */

function renderEpisodeReview(review) {
  if (!review) {
    return '<p class="muted small">本局没有复盘（旧数据或评估局）。</p>';
  }
  const data = review.data || {};
  const parts = [];
  if (review.markdown) {
    parts.push(`<pre class="log episode-md">${escapeHtml(review.markdown)}</pre>`);
  }
  // 伤害交换：字段名与 training/gladiator_env/review.py 的 EpisodeReview.to_dict() 对齐。
  const damageRows = [
    ["造成伤害", data.damageDealt],
    ["承受伤害", data.damageTaken],
    ["伤害比", data.damageRatio],
    ["命中 / 空挥", (data.hits != null || data.whiffs != null) ? `${num(data.hits)} / ${num(data.whiffs)}` : null],
    ["命中率", data.hitRate != null ? `${(Number(data.hitRate) * 100).toFixed(1)}%` : null],
    ["连击峰值", data.maxCombo],
    ["平均距离", data.meanDistance != null ? num(data.meanDistance, 2) : null],
  ].filter((entry) => entry[1] !== null && entry[1] !== undefined && entry[1] !== "");
  if (damageRows.length) {
    parts.push(`<div class="card-head" style="margin-top:12px"><h3>伤害交换</h3></div>
      <div class="table-wrap"><table><thead><tr><th>指标</th><th>数值</th></tr></thead><tbody>
      ${damageRows.map(([name, value]) => `<tr><td>${escapeHtml(name)}</td><td>${escapeHtml(String(value))}</td></tr>`).join("")}
      </tbody></table></div>`);
  }
  const actionCounts = data.actionCounts || {};
  const actionEntries = Object.entries(actionCounts);
  if (actionEntries.length) {
    const total = actionEntries.reduce((sum, item) => sum + Number(item[1] || 0), 0) || 1;
    parts.push(`<div class="card-head" style="margin-top:12px"><h3>动作分布</h3><span class="muted small">共 ${num(total)} 次</span></div>
      <div class="table-wrap"><table><thead><tr><th>动作</th><th>次数</th><th>占比</th></tr></thead><tbody>
      ${actionEntries.sort((a, b) => Number(b[1]) - Number(a[1])).map(([name, count]) => `<tr><td>${escapeHtml(name)}</td><td>${num(count)}</td><td>${(Number(count) / total * 100).toFixed(1)}%</td></tr>`).join("")}
      </tbody></table></div>`);
  }
  const breakdown = data.rewardBreakdown || {};
  const breakdownEntries = Object.entries(breakdown);
  if (breakdownEntries.length) {
    parts.push(`<div class="card-head" style="margin-top:12px"><h3>奖励分解</h3></div>
      <div class="table-wrap"><table><thead><tr><th>项</th><th>贡献</th></tr></thead><tbody>
      ${breakdownEntries.map(([name, value]) => `<tr><td>${escapeHtml(name)}</td><td>${num(value, 3)}</td></tr>`).join("")}
      </tbody></table></div>`);
  }
  const events = data.events || [];
  if (events.length) {
    parts.push(`<div class="card-head" style="margin-top:12px"><h3>关键事件</h3><span class="muted small">${events.length} 条</span></div>
      <pre class="log small">${escapeHtml(events.map((event) => `step ${event.step ?? "?"}: ${event.type ?? "?"}` + Object.entries(event).filter(([key]) => !["step", "type"].includes(key)).map(([key, value]) => ` ${key}=${value}`).join("")).join("\n"))}</pre>`);
  }
  return parts.join("");
}

async function openEpisode(runId, episode) {
  if (!runId || !episode) return;
  state.episodeRunId = runId;
  const dialog = $("#episode-dialog");
  const body = $("#episode-body");
  if (!body) return;
  body.innerHTML = '<p class="muted">读取中…</p>';
  if (dialog && typeof dialog.showModal === "function") dialog.showModal();
  else if (dialog) dialog.setAttribute("open", "");
  try {
    const data = await api(`/api/training/episodes?run=${encodeURIComponent(runId)}&episode=${encodeURIComponent(episode)}`);
    $("#episode-title").textContent = `对局详情 · ${runId} · 第 ${data.episode}/${data.episodeCount} 局`;
    const model = data.model ? `${data.model.name}（${data.model.id}）` : "—";
    body.innerHTML = `<div class="kv">
      ${kvItem("结果", escapeHtml(data.winner || "未分胜负"))}
      ${kvItem("耗时 / 步数", `${fmtDuration(data.durationSeconds)} / ${num(data.steps)}`)}
      ${kvItem("reward", num(data.totalReward, 2))}
      ${kvItem("terminated / truncated", `${data.terminated ? "是" : "否"} / ${data.truncated ? "是" : "否"}`)}
      ${kvItem("阶段 / 场地", `${escapeHtml(data.opponentMode || "—")}${data.pairKey ? ` · ${escapeHtml(data.pairKey)}` : ""}`)}
      ${kvItem("模型", escapeHtml(model))}
      ${kvItem("开始时间", fmtTime(data.startedAt || ""))}
    </div>${renderEpisodeReview(data.review)}`;
  } catch (err) {
    body.innerHTML = `<p class="small" style="color:var(--danger)">${escapeHtml(err.message || "读取失败")}</p>`;
  }
}

function bindEpisodeDialog() {
  const dialog = $("#episode-dialog");
  if (!dialog) return;
  $("#btn-episode-close")?.addEventListener("click", () => dialog.close());
  $("#btn-episode-folder")?.addEventListener("click", () => {
    if (state.episodeRunId) openFolder("run-reviews", state.episodeRunId);
  });
  document.addEventListener("click", (event) => {
    const hit = event.target.closest("[data-episode-row]");
    if (hit) {
      openEpisode(hit.dataset.runId, hit.dataset.episodeRow);
    }
  });
}

function renderTraining() {
  const data = state.training;
  if (!data) return;
  const job = data.job;
  // 写回失败要让用户看见（但不把 job 变 failed）：同一条错误只提示一次。
  const persistInfo = job && job.persist;
  if (persistInfo && persistInfo.enabled && persistInfo.ok === false
      && !persistInfo.skipped && persistInfo.error) {
    const key = `${job.runId || ""}:${persistInfo.error}`;
    if (state.persistToastKey !== key) {
      state.persistToastKey = key;
      toast(`模型写回失败：${persistInfo.error}（训练本身已完成）`, "err");
    }
  }
  const statusEl = $("#training-status");
  if (!job) {
    statusEl.className = "pill neutral";
    statusEl.textContent = "未运行";
    const sessions = state.server && state.server.gladiator ? state.server.gladiator.training : null;
    const externalServer = state.server && state.server.external;
    const hints = [];
    if (externalServer) {
      hints.push(`检测到外部 Paper 进程（PID ${escapeHtml(String(state.server.pid ?? "未知"))}），Dashboard 只读接入、不会停止它。启动训练/对战前请确认外部训练已结束，避免与现有 Bot 争用端口。`);
    }
    if (sessions) {
      hints.push(`插件报告有 ${escapeHtml(String(sessions))} 个训练会话正在运行，可能是外部启动的训练；训练进度暂不可见，产物完成后会出现在「数据可视化」。`);
    }
    const externalHint = hints.length
      ? `<p class="small" style="color:var(--warn, #d9a441)">${hints.join("<br>")}</p>`
      : "";
    $("#training-progress-body").innerHTML = externalHint + '<p class="muted">还没有由 Dashboard 启动的训练任务。</p>';
    $("#training-progress-bar").style.width = "0%";
    $("#training-log").textContent = "—";
    $("#training-chart").innerHTML = '<div class="chart-empty">启动训练后显示 reward 曲线</div>';
    $("#training-report-card").style.display = "none";
    $("#training-episodes").innerHTML = "";
    $("#training-episodes-empty").classList.remove("hidden");
    $("#training-episodes-meta").textContent = "";
    return;
  }
  statusEl.className = `pill ${["running", "starting", "finished"].includes(job.status) ? "ok" : job.status === "failed" ? "bad" : "warn"}`;
  statusEl.textContent = STATUS_TEXT[job.status] || job.status;

  const total = Math.max(1, job.totalEpisodes || 1);
  const pct = Math.min(100, Math.round(((job.episodesCompleted || 0) / total) * 100));
  $("#training-progress-bar").style.width = `${pct}%`;

  const lastReward = job.rewards && job.rewards.length ? job.rewards[job.rewards.length - 1] : null;
  const mean = job.rewards && job.rewards.length
    ? job.rewards.reduce((a, b) => a + b, 0) / job.rewards.length : null;
  const isMulti = job.kind === "multi";
  const pairs = Number(job.params && job.params.pairs) || 0;
  const perPair = Number(job.params && job.params.episodes) || 0;
  const plan = Array.isArray(job.params && job.params.plan) ? job.params.plan : [];
  const stageCount = Number(job.stageCount) || plan.length;
  const stageIndex = Number(job.stageIndex) || 0;
  const stageDone = Number(job.stageCompleted) || 0;
  const stagePlan = stageIndex > 0 && plan[stageIndex - 1]
    ? Number(plan[stageIndex - 1].episodes) || 0 : 0;
  const epochCount = Number(job.epochCount) || 0;
  const epochIndex = Number(job.epochIndex) || 0;
  const epochPrefix = epochCount > 1 ? `回合 ${epochIndex || 1}/${epochCount} · ` : "";
  const stageSub = stagePlan
    ? `${epochPrefix}第 ${stageIndex}/${stageCount} 段 · 本段 ${Math.min(stageDone, stagePlan)}/${stagePlan} 局`
    : `${epochPrefix}第 ${stageIndex}/${stageCount} 段`;
  const stageValue = job.currentStage
    ? `<span>${escapeHtml(job.currentStage)}</span><small class="sub">${stageSub}</small>`
    : "—";
  const roundText = isMulti
    ? `<span>${job.episodesCompleted || 0} / ${job.totalEpisodes || 0}</span><small class="sub">已完成 / 计划（${pairs} 场地 × ${perPair} 局）</small>`
    : `<span>${job.episodesCompleted || 0} / ${job.totalEpisodes || 0}</span><small class="sub">训练总局数</small>`;
  const items = [
    kvItem("当前计划", stageValue),
    kvItem("回合", roundText),
    kvItem("步数", num(job.steps)),
    kvItem("最近 reward", lastReward == null ? "—" : num(lastReward, 2)),
    kvItem("平均 reward", mean == null ? "—" : num(mean, 2)),
    kvItem("开始时间", fmtTime(job.startedAt)),
    kvItem("结束时间", fmtTime(job.finishedAt)),
    kvItem("运行时长", fmtDuration(job.durationSeconds)),
  ];
  let body = `<div class="kv">${items.join("")}</div>`;
  if (!isMulti && plan.length) {
    const planText = plan.map((item, index) => `${index + 1}. ${escapeHtml(item.opponentMode)}×${item.episodes}`).join(" → ");
    const epochsTotal = Number(job.epochCount) || 1;
    const perEpoch = Number(job.params && job.params.episodesPerEpoch) || 0;
    body += `<p class="muted small stage-plan">训练计划：${planText} · ${epochsTotal} 回合 × 每回合 ${perEpoch} 局 = ${job.totalEpisodes || 0} 局</p>`;
  }
  if (job.runId) {
    body += `<p class="muted small" style="margin:10px 0 0">运行目录：<code>${escapeHtml(job.runDir || job.runId)}</code></p>`;
  }
  const jobModel = job.params && job.params.modelId
    ? `${job.params.modelName || job.params.modelId}（${job.params.modelId}）` : "";
  if (jobModel) {
    body += `<p class="muted small" style="margin:6px 0 0">训练模型：<b>${escapeHtml(jobModel)}</b></p>`;
  }
  if (persistInfo && persistInfo.enabled) {
    const persistText = persistInfo.ok === true
      ? `已写回模型 <b>${escapeHtml(persistInfo.modelId)}</b>（权重已更新，下次选它继续练）`
      : persistInfo.skipped
        ? `未写回模型 ${escapeHtml(persistInfo.modelId)}：${escapeHtml(persistInfo.error || "训练未正常完成")}`
        : `<span style="color:var(--danger)">写回失败：${escapeHtml(persistInfo.error || "未知原因")}</span>`;
    body += `<p class="muted small" style="margin:6px 0 0">持久化训练：${persistText}</p>`;
  }
  if (job.persistWarning) {
    body += `<p class="muted small" style="margin:6px 0 0">${escapeHtml(job.persistWarning)}</p>`;
  }
  const presetLabel = rewardPresetLabel(job);
  if (presetLabel) {
    body += `<p class="muted small" style="margin:6px 0 0">Reward 预设：<b>${escapeHtml(presetLabel)}</b></p>`;
  }
  if (job.cleanup && job.cleanup.message) {
    body += `<p class="muted small" style="margin:6px 0 0">${escapeHtml(job.cleanup.message)}</p>`;
  }
  if (job.lastEpisode) {
    body += `<p class="muted small" style="margin:10px 0 0">最近回合：${job.lastEpisode.steps} 步 · reward ${num(job.lastEpisode.reward, 2)} · winner ${escapeHtml(job.lastEpisode.winner ?? "未分胜负")} · ${fmtTime(job.lastEpisode.startedAt)} · ${fmtDuration(job.lastEpisode.durationSeconds)}</p>`;
  }
  if (job.error) body += `<p class="small" style="color:var(--danger)">${escapeHtml(job.error)}</p>`;
  $("#training-progress-body").innerHTML = body;
  renderTrainingEpisodes(job);

  const rewards = (job.rewards && job.rewards.length) ? job.rewards
    : ((job.report && job.report.summary) ? [] : []);
  if (rewards.length) {
    $("#training-chart").innerHTML = lineChart(rewards);
    $("#training-chart-meta").textContent = isMulti
      ? `合并 ${pairs} 个场地 · ${rewards.length} 个回合（每场地 ${perPair}）`
      : `${rewards.length} 个回合`;
  } else if (job.report && job.report.exists) {
    $("#training-chart").innerHTML = '<div class="chart-empty">训练已结束，完整曲线见「数据可视化」</div>';
    $("#training-chart-meta").textContent = "";
  } else {
    $("#training-chart").innerHTML = '<div class="chart-empty">等待第一个回合…</div>';
    $("#training-chart-meta").textContent = "";
  }
  $("#training-log").textContent = (job.logTail || []).join("\n") || "—";
  renderTrainingReport(job.report);
}

function renderTrainingEpisodes(job) {
  const episodes = (job.episodes && job.episodes.length)
    ? job.episodes
    : ((job.report && job.report.history) ? job.report.history : []);
  const runId = (state.training && state.training.job && state.training.job.runId) || "";
  const rows = episodes.map((item, index) => {
    const reward = item.reward ?? item.totalReward;
    return `<tr class="clickable" data-episode-row="${index + 1}" data-run-id="${escapeHtml(runId)}">
      <td>${index + 1}</td>
      <td>${fmtTime(item.startedAt)}</td>
      <td>${fmtDuration(item.durationSeconds)}</td>
      <td>${num(item.steps)}</td>
      <td>${num(reward, 2)}</td>
      <td>${escapeHtml(item.winner ?? "未分")}</td>
    </tr>`;
  }).join("");
  $("#training-episodes").innerHTML = rows;
  $("#training-episodes-empty").classList.toggle("hidden", episodes.length > 0);
  $("#training-episodes-meta").textContent = episodes.length ? `${episodes.length} 局` : "";
}

function renderTrainingReport(report) {
  const card = $("#training-report-card");
  if (!report || !report.exists) { card.style.display = "none"; return; }
  card.style.display = "";
  const s = report.summary || {};
  const items = [
    kvItem("回合", s.episodes ?? "—"),
    kvItem("总步数", num(s.totalSteps)),
    kvItem("平均 reward", num(s.meanReward, 2)),
    kvItem("最佳 / 最差", `${num(s.bestReward, 2)} / ${num(s.worstReward, 2)}`),
    kvItem("命中率", s.hitRate == null ? "—" : `${(s.hitRate * 100).toFixed(1)}%`),
    kvItem("报告", escapeHtml(report.path || "—")),
  ];
  $("#training-report").innerHTML = `<div class="kv">${items.join("")}</div>`;
}

/* ---------------- 人机对战 ---------------- */

function renderMatch() {
  const match = state.match;
  if (!match) return;
  const ready = (match.bots || []).filter((bot) => bot.ready).length;
  const cards = [
    statCard("对战状态", STATUS_TEXT[match.status] || match.status || "空闲",
      match.running ? `已运行 ${fmtDuration(match.durationSeconds)}` : "尚未启动",
      `<div style="margin-top:6px">${match.running ? '<span class="pill ok">进行中</span>' : '<span class="pill neutral">空闲</span>'}</div>`),
    statCard("Bot 就绪", `${ready} / ${match.count || 0}`, "ready=true 才算真正进服"),
    statCard("服务器", match.mcRunning ? "在线" : "离线", match.rconReachable ? "RCON 可达" : "RCON 不可达"),
    statCard("进服地址", escapeHtml(match.address || " — "), "Minecraft 1.8.9 · offline"),
  ];
  $("#match-cards").innerHTML = cards.join("");
  $("#match-address").textContent = match.address || "127.0.0.1:25566";

  const commands = [];
  commands.push(`<div class="command-row"><code data-copy="${escapeHtml(match.address || "")}">${escapeHtml(match.address || "")}</code><button class="btn tiny ghost" data-copy="${escapeHtml(match.address || "")}" type="button">复制地址</button></div>`);
  (match.bots || []).forEach((bot) => {
    const command = `/gladiator spectate ${bot.username}`;
    commands.push(`<div class="command-row"><code>${escapeHtml(command)}</code><button class="btn tiny ghost" data-copy="${escapeHtml(command)}" type="button">复制</button></div>`);
  });
  commands.push(`<div class="command-row"><code>/gladiator unspectate</code><button class="btn tiny ghost" data-copy="/gladiator unspectate" type="button">复制</button></div>`);
  $("#match-commands").innerHTML = commands.join("");

  $("#match-bots").innerHTML = (match.bots || []).map((bot) => `<tr>
    <td>${escapeHtml(bot.username)}</td>
    <td>${bot.envPort ?? "—"}</td>
    <td>${escapeHtml(bot.runtimeId || "—")}</td>
    <td>${boolPill(bot.ready, "ready", "未就绪")}</td>
    <td>${bot.pid ?? "—"}</td>
  </tr>`).join("");
  $("#match-bots-empty").classList.toggle("hidden", (match.bots || []).length > 0);
  $("#match-log").textContent = (match.log || []).map((entry) => `[${entry.at}] ${entry.text}`).join("\n") || "—";
  $("#btn-match-start").disabled = !state.health || state.health.readOnly || match.running;
  $("#btn-match-stop").disabled = !state.health || state.health.readOnly || !match.running;
}

/* ---------------- 数据可视化 ---------------- */

function renderArtifacts() {
  const data = state.artifacts;
  if (!data) return;
  const runs = state.runs || [];
  const isMerged = Boolean(data.merged);
  const runsMeta = $("#data-runs-meta");
  if (runsMeta) {
    runsMeta.textContent = runs.length
      ? `${runs.length} 次运行${state.selectedRuns.size ? ` · 已选 ${state.selectedRuns.size}` : ""}`
      : "";
  }
  const selectMeta = $("#data-select-meta");
  if (selectMeta) {
    selectMeta.textContent = state.selectedRuns.size
      ? (state.mergeMode ? `已合并 ${state.selectedRuns.size} 次运行` : `已选 ${state.selectedRuns.size} 次运行`)
      : "未选择";
  }
  const runRows = runs.map((run) => {
    const plan = (run.plan || []).map((item) => `${item.opponentMode || "?"}×${item.episodes ?? "?"}`).join(" → ") || "—";
    const statusClass = run.status === "finished" ? "ok" : run.status === "running" ? "warn" : "bad";
    const statusText = run.status === "finished" ? "已完成" : run.status === "running" ? "运行中" : "中断";
    const checked = state.selectedRuns.has(run.id) ? "checked" : "";
    return `<tr class="run-row ${state.selectedRun === run.id && !isMerged ? "selected" : ""}" data-run="${escapeHtml(run.id)}">
      <td class="check-col"><input type="checkbox" data-run-check="${escapeHtml(run.id)}" ${checked} aria-label="选择 ${escapeHtml(run.name)}"></td>
      <td><strong>${escapeHtml(run.name)}</strong>${run.legacy ? ' <span class="pill neutral">legacy</span>' : ""}<div class="muted small">${fmtTime(run.createdAt)}</div></td>
      <td>${escapeHtml(run.kind || "—")}</td>
      <td class="small">${escapeHtml(plan)}</td>
      <td>${num(run.episodes)}</td>
      <td>${num(run.steps)}</td>
      <td>${num(run.meanReward, 2)}</td>
      <td><span class="pill ${statusClass}">${statusText}</span></td>
    </tr>`;
  }).join("");
  const runsBody = $("#data-runs");
  if (runsBody) runsBody.innerHTML = runRows;
  $("#data-runs-empty").classList.toggle("hidden", runs.length > 0);
  const allChecked = runs.length > 0 && runs.every((run) => state.selectedRuns.has(run.id));
  ["#data-check-all", "#data-check-all-head"].forEach((selector) => {
    const box = $(selector);
    if (box) {
      box.checked = allChecked;
      box.indeterminate = !allChecked && state.selectedRuns.size > 0;
    }
  });
  renderRunDetail();

  const train = data.train || {};
  const multi = data.multi || {};
  const stress = data.stress || {};
  const selected = data.selectedRun || state.selectedRunData || {};
  const isMulti = Boolean(multi.exists);
  const report = isMulti ? multi : train;
  const summary = selected.summary || report.summary || {};
  const allReviews = data.reviews || [];
  const pairFilter = state.pairFilter;
  const reviews = pairFilter == null ? allReviews : allReviews.filter((item) => {
    if (item.pairKey) return item.pairKey === pairFilter;
    const raw = String(pairFilter);
    const suffix = raw.includes("::") ? raw.split("::").pop() : raw;
    const key = suffix.startsWith("pair-") ? suffix.slice(5) : suffix;
    const path = String(item.path || "");
    return path.includes(`pair-${key}/`) || path.includes(`pair${key}/`);
  });
  const reportName = (report.path || "").split(/[\\/]/).pop() || (isMulti ? "multi report" : "train report");
  const mergedCount = isMerged ? (data.selectedRunIds || selected.sources || []).length : 0;
  const cards = [
    statCard("训练回合", summary.episodes ?? "—",
      isMerged ? `合并 ${mergedCount} 次运行 · ${(report.history || []).length} 个数据点`
        : `${selected.kind === "multi" ? "多场地 · " : ""}${reportName}`),
    statCard("平均 reward", num(summary.meanReward, 2), `最佳 ${num(summary.bestReward, 2)}`),
    statCard("命中率", summary.hitRate == null ? "—" : `${(summary.hitRate * 100).toFixed(1)}%`,
      `${num(summary.hits)} 命中 / ${num(summary.whiffs)} 落空`),
    statCard("复盘", `${allReviews.length}`,
      isMerged ? `${mergedCount} 次运行合并` : (isMulti ? `多场地 ${ (multi.pairs || []).length } 组` : "reviews/*.json")),
  ];
  $("#data-cards").innerHTML = cards.join("");

  const history = report.history || [];
  const sourceIds = (data.sources || selected.sources || []).map((item) => item.id);
  const shortPointLabel = (item) => {
    if (!isMerged) return `#${item.episode ?? ""}`;
    const index = sourceIds.indexOf(item.runId);
    return `R${index >= 0 ? index + 1 : "?"}#${item.episode ?? ""}`;
  };
  if (history.length) {
    const values = history.map((item) => Number(item.totalReward) || 0);
    const labels = history.map(shortPointLabel);
    const tooltips = history.map((item) => [
      `${item.runName || selected.name || "运行"} · 回合 ${item.episode ?? "—"}`,
      `reward ${num(item.totalReward, 2)} · ${num(item.steps)} 步 · 耗时 ${fmtDuration(item.durationSeconds)}`,
      `开始 ${fmtTime(item.startedAt)} · winner ${item.winner ?? "未分"}`,
    ].join("\n"));
    const runIds = history.map((item) => item.runId || selected.id || "");
    $("#data-chart").innerHTML = lineChart(values, { labels, tooltips, runIds });
    const pairs = multi.pairs || [];
    const perPair = pairs.length ? Math.round(history.length / pairs.length) : 0;
    let scope;
    if (isMerged) {
      scope = `合并 ${mergedCount} 次运行${isMulti ? ` / ${pairs.length} 个场地` : ""} · ${history.length} 个回合`;
    } else if (isMulti) {
      scope = `合并 ${pairs.length} 个场地 · ${history.length} 个回合（每场地 ${perPair}）`;
    } else {
      scope = `${history.length} 个回合`;
    }
    $("#data-chart-meta").textContent = `${scope} · 悬停看详情，点击数据点查看运行`;
  } else {
    $("#data-chart").innerHTML = '<div class="chart-empty">选中的运行没有 report.json / history</div>';
    $("#data-chart-meta").textContent = selected.name || "";
  }

  const consistency = $("#data-consistency");
  if (consistency) {
    const showHint = !selected.legacy && history.length > 0 && allReviews.length !== history.length;
    consistency.classList.toggle("hidden", !showHint);
    consistency.textContent = showHint
      ? `口径提示：report.json history 有 ${history.length} 局，reviews 目录有 ${allReviews.length} 份复盘；页面以 history 为准，差异可能来自中途失败或未计入 report 的局。`
      : "";
  }

  const liveTps = state.server && state.server.tps;
  const stressItems = (stress.metrics || [])
    .filter((metric) => metric.value != null && metric.value !== "")
    .map((metric) => ({ label: metric.label, value: metric.value, unit: metric.unit || "" }));
  if (liveTps !== null && liveTps !== undefined && liveTps !== "") {
    stressItems.push({ label: "TPS（实时）", value: liveTps, unit: "" });
  }
  const stressNote = stress.derived
    ? `<div class="muted small" style="margin-top:8px">${escapeHtml(stress.note || "未找到 stress-report.json，指标由选中 run 派生；TPS 为服务端实时值。")}</div>`
    : (stress.exists ? `<div class="muted small" style="margin-top:8px">来源：${escapeHtml(stress.path || "stress-report.json")}</div>`
      : '<div class="muted small" style="margin-top:8px">无压力测试报告；TPS 取自 /api/server 实时值。</div>');
  $("#data-stress").innerHTML = (stressItems.length
    ? metricList(stressItems)
    : '<div class="chart-empty">该运行没有可展示的 TPS / 步频数据</div>') + stressNote;

  if (isMulti && (multi.pairs || []).length) {
    const rows = multi.pairs.map((pair) => {
      const key = String(pair.pairKey || pair.pair || "");
      return `<tr class="pair-row ${pairFilter === key ? "selected" : ""}" data-pair="${escapeHtml(key)}">
        <td>${pair.pair ?? "—"}</td>
        <td>${escapeHtml((pair.users || []).join(", "))}${isMerged ? `<div class="muted small">${escapeHtml(pair.runName || pair.runId || "")}</div>` : ""}<div class="muted small">${fmtTime(pair.startedAt || report.startedAt)} · ${fmtDuration(pair.durationSeconds ?? report.durationSeconds)}</div></td>
        <td>${pair.episodes ?? "—"}</td>
        <td>${num(pair.steps)}</td>
        <td>${num(pair.meanReward, 2)}</td>
        <td>${num(pair.hits)} / ${num(pair.whiffs)}</td>
        <td>${pair.error ? `<span class="pill bad">失败</span>` : `<span class="pill ok">完成</span>`}</td>
        <td><button class="btn tiny ghost" data-pair="${escapeHtml(key)}" type="button">复盘</button></td>
      </tr>`;
    }).join("");
    $("#data-multi").innerHTML = `<div class="table-wrap"><table><thead><tr><th>场地</th><th>用户</th><th>回合</th><th>步数</th><th>mean reward</th><th>命中 / 落空</th><th>状态</th><th></th></tr></thead><tbody>${rows}</tbody></table></div>`;
  } else {
    $("#data-multi").innerHTML = '<div class="chart-empty">当前选中的运行不是多场地训练（没有 multi report / pairs）。</div>';
  }

  $("#data-review-count").textContent = `${reviews.length} 份${pairFilter == null ? "" : ` · ${pairFilter.includes("::") ? pairFilter.split("::")[1] : `场地 ${pairFilter}`}`}`;
  $("#data-reviews").innerHTML = reviews.slice(0, 200).map((item) => `
    <div class="review-item">
      <div>
        <div class="r-title">${escapeHtml(item.path)}${isMerged && item.runName ? ` <span class="muted small">· ${escapeHtml(item.runName)}</span>` : ""}</div>
        <div class="r-sub">${escapeHtml(item.username || "—")} · 回合 ${item.episode ?? "—"} · ${fmtTime(item.startedAt || "")} · 耗时 ${fmtDuration(item.durationSeconds)} · winner ${escapeHtml(item.winner ?? "—")} · reward ${num(item.totalReward, 2)}</div>
      </div>
      <span class="pill ${item.winner ? "ok" : "neutral"}">${item.winner ? "已分胜负" : "未分"}</span>
      <button class="btn tiny ghost" data-review="${escapeHtml(item.markdownPath || "")}" type="button" ${item.markdownPath ? "" : "disabled"}>查看复盘</button>
    </div>`).join("");
  $("#data-reviews-empty").classList.toggle("hidden", reviews.length > 0);
}

function renderRunDetail() {
  const card = $("#data-run-detail");
  if (!card) return;
  const run = state.selectedRunData;
  if (!run || (state.artifacts && state.artifacts.merged)) { card.classList.add("hidden"); return; }
  card.classList.remove("hidden");
  $("#data-run-title").textContent = `运行详情 · ${run.name || run.id}`;
  const s = run.summary || {};
  const plan = (run.plan || []).map((item) => `${item.opponentMode || "?"}×${item.episodes ?? "?"}`).join(" → ") || "—";
  $("#data-run-body").innerHTML = `<div class="kv">
    ${kvItem("kind", run.kind || "—")}
    ${kvItem("状态", run.status || "—")}
    ${kvItem("阶段", plan)}
    ${kvItem("开始时间", fmtTime(run.startedAt || run.createdAt))}
    ${kvItem("结束时间", fmtTime(run.finishedAt))}
    ${kvItem("耗时", fmtDuration(run.durationSeconds))}
    ${kvItem("回合 / 步数", `${num(run.episodes)} / ${num(run.steps)}`)}
    ${kvItem("平均 / 最终 reward", `${num(run.meanReward, 2)} / ${num(run.finalReward, 2)}`)}
    ${kvItem("Reward 预设", run.rewardPreset ? `${run.rewardPreset.name || run.rewardPreset.id}（${run.rewardPreset.id}）` : "—")}
    ${kvItem("训练模型", run.model ? `${run.model.name}（${run.model.id}）` : "—")}
    ${kvItem("持久化写回", run.persist
      ? (run.persist.ok === true
          ? `已写回模型 ${run.persist.modelId || "—"}`
          : run.persist.skipped
            ? `未写回（${run.persist.error || "训练未正常完成"}）`
            : `失败：${run.persist.error || "未知原因"}`)
      : "未启用")}
    ${kvItem("日志", run.logState === "no-log" ? "本次运行没有日志（历史数据）" : (run.logPath || "—"))}
    ${kvItem("报告", run.reportPath || "—")}
  </div>`;
  const history = run.history || [];
  $("#data-run-chart").innerHTML = history.length
    ? lineChart(history.map((item) => Number(item.totalReward) || 0), { labels: history.map((item) => `#${item.episode ?? ""}`) })
    : '<div class="chart-empty">该运行还没有 report.json 或 history</div>';
  const runEpisodes = run.history || [];
  $("#data-run-episodes").innerHTML = runEpisodes.map((item, index) => `<tr class="clickable" data-episode-row="${index + 1}" data-run-id="${escapeHtml(run.id || run.name)}">
    <td>${index + 1}</td>
    <td>${fmtTime(item.startedAt)}</td>
    <td>${fmtDuration(item.durationSeconds)}</td>
    <td>${num(item.steps)}</td>
    <td>${num(item.totalReward, 2)}</td>
    <td>${escapeHtml(item.winner ?? "未分")}</td>
    <td><button class="btn tiny ghost" type="button" data-episode-row="${index + 1}" data-run-id="${escapeHtml(run.id || run.name)}">详情</button></td>
  </tr>`).join("");
  $("#data-run-episodes-empty").classList.toggle("hidden", runEpisodes.length > 0);
  $("#data-run-episodes-meta").textContent = runEpisodes.length ? `${runEpisodes.length} 局（点行看详情）` : "";

  const reviews = run.reviews || [];
  $("#data-run-review-count").textContent = `${reviews.length} 份`;
  $("#data-run-reviews").innerHTML = reviews.length ? reviews.map((item) => `
    <div class="review-item">
      <div>
        <div class="r-title">${escapeHtml(item.path)}</div>
        <div class="r-sub">${escapeHtml(item.username || "—")} · 回合 ${item.episode ?? "—"} · reward ${num(item.totalReward, 2)}</div>
      </div>
      <button class="btn tiny ghost" data-review="${escapeHtml(item.markdownPath || "")}" type="button" ${item.markdownPath ? "" : "disabled"}>查看复盘</button>
    </div>`).join("") : '<p class="muted small">该运行还没有复盘文件。</p>';
}

async function openReview(path) {
  if (!path) return;
  try {
    const data = await api(`/api/artifacts/review?path=${encodeURIComponent(path)}`);
    $("#review-title").textContent = path;
    $("#review-content").innerHTML = markdownToHtml(data.markdown || "");
    const dialog = $("#review-dialog");
    if (typeof dialog.showModal === "function") dialog.showModal();
    else dialog.setAttribute("open", "");
  } catch (err) {
    toast(err.message, "err");
  }
}

/* ---------------- 图表 ---------------- */

function chartId() {
  state.charts += 1;
  return `chart${state.charts}`;
}

function lineChart(values, options = {}) {
  if (!values || !values.length) return '<div class="chart-empty">暂无数据</div>';
  const width = 760;
  const height = options.height || 240;
  const pad = { left: 52, right: 22, top: 18, bottom: 30 };
  const items = values.map((raw, index) => {
    if (raw && typeof raw === "object") {
      return { ...raw, value: Number(raw.value ?? raw.totalReward) || 0 };
    }
    return {
      value: Number(raw) || 0,
      label: options.labels ? options.labels[index] : `#${index + 1}`,
      tip: options.tooltips ? options.tooltips[index] : "",
      runId: options.runIds ? options.runIds[index] : "",
    };
  });
  const nums = items.map((item) => item.value);
  let min = Math.min(...nums, 0);
  let max = Math.max(...nums, 0);
  if (max === min) max = min + 1;
  const span = max - min;
  min -= span * 0.06;
  max += span * 0.06;
  const innerW = width - pad.left - pad.right;
  const innerH = height - pad.top - pad.bottom;
  const x = (index) => pad.left + (nums.length === 1 ? innerW / 2 : (index / (nums.length - 1)) * innerW);
  const y = (value) => pad.top + innerH - ((value - min) / (max - min)) * innerH;
  const id = chartId();

  const points = nums.map((value, index) => `${x(index).toFixed(1)},${y(value).toFixed(1)}`);
  const path = `M${points.join(" L")}`;
  const area = `${path} L${x(nums.length - 1).toFixed(1)},${(pad.top + innerH).toFixed(1)} L${x(0).toFixed(1)},${(pad.top + innerH).toFixed(1)} Z`;

  const grid = [];
  for (let i = 0; i <= 4; i += 1) {
    const value = min + ((max - min) * i) / 4;
    const gy = y(value);
    grid.push(`<line class="gridline" x1="${pad.left}" y1="${gy.toFixed(1)}" x2="${width - pad.right}" y2="${gy.toFixed(1)}"/>`);
    grid.push(`<text class="label" x="${pad.left - 8}" y="${(gy + 3).toFixed(1)}" text-anchor="end">${value.toFixed(value >= 100 || value <= -100 ? 0 : 1)}</text>`);
  }
  const tipFor = (item, index) => {
    if (item.tip) return item.tip;
    const label = item.label || `#${index + 1}`;
    return `${label}\nreward ${item.value}`;
  };
  const band = innerW / Math.max(1, nums.length);
  const hits = items.map((item, index) => {
    const left = Math.max(pad.left, x(index) - band / 2);
    const right = Math.min(width - pad.right, x(index) + band / 2);
    const runId = item.runId || "";
    return `<rect class="chart-hit" x="${left.toFixed(1)}" y="${pad.top}" width="${Math.max(6, right - left).toFixed(1)}" height="${innerH.toFixed(1)}" data-chart-tip="${escapeHtml(tipFor(item, index))}" data-chart-run="${escapeHtml(runId)}"></rect>`;
  }).join("");
  const dots = items.map((item, index) => {
    const label = item.label || `#${index + 1}`;
    const runId = item.runId || "";
    return `<circle class="dot" cx="${x(index).toFixed(1)}" cy="${y(item.value).toFixed(1)}" r="4" data-chart-tip="${escapeHtml(tipFor(item, index))}" data-chart-run="${escapeHtml(runId)}"><title>${escapeHtml(label)}: ${item.value}</title></circle>`;
  }).join("");

  const xLabels = [];
  const count = Math.min(6, nums.length);
  for (let i = 0; i < count; i += 1) {
    const index = count === 1 ? 0 : Math.round((i / (count - 1)) * (nums.length - 1));
    const label = items[index].label || `${index + 1}`;
    xLabels.push(`<text class="label" x="${x(index).toFixed(1)}" y="${height - 10}" text-anchor="middle">${escapeHtml(label)}</text>`);
  }

  return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="reward 曲线">
    <defs>
      <linearGradient id="${id}fill" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#57d68d" stop-opacity=".45"/>
        <stop offset="100%" stop-color="#57d68d" stop-opacity="0"/>
      </linearGradient>
    </defs>
    ${grid.join("")}
    <path d="${area}" fill="url(#${id}fill)"/>
    <path class="line" d="${path}"/>
    ${dots}
    ${hits}
    ${xLabels.join("")}
  </svg>`;
}

function chartTooltip() {
  let node = $("#chart-tooltip");
  if (!node) {
    node = document.createElement("div");
    node.id = "chart-tooltip";
    node.className = "chart-tooltip hidden";
    document.body.appendChild(node);
  }
  return node;
}

function showChartTooltip(event, hit) {
  const node = chartTooltip();
  const text = hit.dataset.chartTip || "";
  if (!text) return;
  node.textContent = text;
  node.classList.remove("hidden");
  const gap = 14;
  const width = node.offsetWidth || 240;
  const height = node.offsetHeight || 80;
  let left = event.clientX + gap;
  let top = event.clientY + gap;
  if (left + width > window.innerWidth - 8) left = event.clientX - width - gap;
  if (top + height > window.innerHeight - 8) top = event.clientY - height - gap;
  node.style.left = `${Math.max(8, left)}px`;
  node.style.top = `${Math.max(8, top)}px`;
}

function hideChartTooltip() {
  const node = $("#chart-tooltip");
  if (node) node.classList.add("hidden");
}

function metricList(items) {
  const rows = items.map((item) => `<div class="metric-row">
    <span class="metric-label">${escapeHtml(item.label || "指标")}</span>
    <span class="metric-value">${num(item.value, item.unit === "step/s" ? 1 : 2)}${item.unit ? ` <span class="muted">${escapeHtml(item.unit)}</span>` : ""}</span>
  </div>`).join("");
  return `<div class="metric-list">${rows}</div>`;
}

function barChart(items) {
  if (!items || !items.length) return '<div class="chart-empty">暂无数据</div>';
  const width = 760;
  const rowH = 44;
  const height = items.length * rowH + 26;
  const labelW = 120;
  const valueW = 80;
  const barW = width - labelW - valueW - 24;
  const max = Math.max(...items.map((item) => Math.abs(item.value)), 1);
  const id = chartId();
  const rows = items.map((item, index) => {
    const y = 14 + index * rowH;
    const w = Math.max(2, (Math.abs(item.value) / max) * barW);
    const value = `${num(item.value, item.unit === "step/s" ? 1 : 2)}${item.unit ? ` ${item.unit}` : ""}`;
    return `
      <text class="label" x="${labelW - 10}" y="${y + 18}" text-anchor="end">${escapeHtml(item.label)}</text>
      <rect class="bar" x="${labelW}" y="${y + 4}" width="${w.toFixed(1)}" height="20" rx="4" fill="url(#${id}bar)"/>
      <text class="label" x="${labelW + w + 8}" y="${y + 18}">${escapeHtml(value)}</text>`;
  }).join("");
  return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="指标柱状图">
    <defs><linearGradient id="${id}bar" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#2f9e63"/><stop offset="100%" stop-color="#57d68d"/>
    </linearGradient></defs>
    <line class="axis" x1="${labelW}" y1="8" x2="${labelW}" y2="${height - 8}"/>
    ${rows}
  </svg>`;
}

/* 多条折线（按 run 分组的 reward 曲线）。series: [{runId, points:[{i, reward}]}] */
function multiLineChart(series, options = {}) {
  const all = series.flatMap((s) => s.points.map((p) => Number(p.reward) || 0));
  if (!all.length) return '<div class="chart-empty">暂无数据</div>';
  const width = 760;
  const height = options.height || 260;
  const pad = { left: 52, right: 22, top: 18, bottom: 38 };
  let min = Math.min(...all, 0);
  let max = Math.max(...all, 0);
  if (max === min) max = min + 1;
  const span = max - min;
  min -= span * 0.06;
  max += span * 0.06;
  const innerW = width - pad.left - pad.right;
  const innerH = height - pad.top - pad.bottom;
  const colors = ["#57d68d", "#4aa3ff", "#ffb454", "#ff6b6b", "#b78bff", "#3fd0c9", "#f06595", "#94d82d"];
  const maxLen = series.reduce((m, s) => Math.max(m, s.points.length), 1);
  const x = (idx) => pad.left + (maxLen === 1 ? innerW / 2 : (idx / (maxLen - 1)) * innerW);
  const y = (v) => pad.top + innerH - ((v - min) / (max - min)) * innerH;
  const grid = [];
  for (let i = 0; i <= 4; i += 1) {
    const value = min + ((max - min) * i) / 4;
    const gy = y(value);
    grid.push(`<line class="gridline" x1="${pad.left}" y1="${gy.toFixed(1)}" x2="${width - pad.right}" y2="${gy.toFixed(1)}"/>`);
    grid.push(`<text class="label" x="${pad.left - 8}" y="${(gy + 3).toFixed(1)}" text-anchor="end">${value.toFixed(value >= 100 || value <= -100 ? 0 : 1)}</text>`);
  }
  const paths = series.map((s, si) => {
    const color = colors[si % colors.length];
    if (!s.points.length) return "";
    const pts = s.points.map((p, i) => `${x(i).toFixed(1)},${y(Number(p.reward) || 0).toFixed(1)}`);
    const path = `M${pts.join(" L")}`;
    const dots = s.points.map((p, i) => `<circle class="dot" cx="${x(i).toFixed(1)}" cy="${y(Number(p.reward) || 0).toFixed(1)}" r="3"><title>${escapeHtml(s.runId)} #${p.i}: ${Number(p.reward) || 0}</title></circle>`).join("");
    const shortId = s.runId.length > 16 ? s.runId.slice(0, 16) + "…" : s.runId;
    return `<path class="line" d="${path}" stroke="${color}" fill="none"/>${dots}`
      + `<text class="label" x="${pad.left}" y="${pad.top + 4 + si * 14}" fill="${color}">${escapeHtml(shortId)}</text>`;
  }).join("");
  const xLabels = [];
  const count = Math.min(6, maxLen);
  for (let i = 0; i < count; i += 1) {
    const idx = count === 1 ? 0 : Math.round((i / (count - 1)) * (maxLen - 1));
    xLabels.push(`<text class="label" x="${x(idx).toFixed(1)}" y="${height - 12}" text-anchor="middle">#${idx + 1}</text>`);
  }
  return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="reward 曲线（按 run）">
    ${grid.join("")}
    ${paths}
    ${xLabels.join("")}
  </svg>`;
}

/* 直方图：binEdges 长度 n+1，counts 长度 n。 */
function histogramChart(binEdges, counts) {
  if (!counts || !counts.length) return '<div class="chart-empty">暂无数据</div>';
  const width = 760;
  const height = 240;
  const pad = { left: 44, right: 16, top: 16, bottom: 36 };
  const maxC = Math.max(...counts, 1);
  const innerW = width - pad.left - pad.right;
  const innerH = height - pad.top - pad.bottom;
  const n = counts.length;
  const bw = innerW / n;
  const y = (v) => pad.top + innerH - (v / maxC) * innerH;
  const bars = counts.map((c, i) => {
    const h = (c / maxC) * innerH;
    const bx = pad.left + i * bw;
    const by = pad.top + innerH - h;
    const lo = binEdges[i], hi = binEdges[i + 1];
    return `<rect class="bar" x="${bx.toFixed(1)}" y="${by.toFixed(1)}" width="${(bw - 1).toFixed(1)}" height="${h.toFixed(1)}" rx="1"><title>${lo.toFixed(3)} ~ ${hi.toFixed(3)} : ${c}</title></rect>`;
  }).join("");
  const grid = [];
  for (let i = 0; i <= 4; i += 1) {
    const value = (maxC * i) / 4;
    const gy = y(value);
    grid.push(`<line class="gridline" x1="${pad.left}" y1="${gy.toFixed(1)}" x2="${width - pad.right}" y2="${gy.toFixed(1)}"/>`);
    grid.push(`<text class="label" x="${pad.left - 6}" y="${(gy + 3).toFixed(1)}" text-anchor="end">${value.toFixed(0)}</text>`);
  }
  const edgeLabels = [binEdges[0], binEdges[Math.floor(n / 2)], binEdges[n]];
  const xl = edgeLabels.map((e, i) => {
    const bx = pad.left + (i === 0 ? 0 : i === 1 ? innerW / 2 : innerW);
    return `<text class="label" x="${bx.toFixed(1)}" y="${height - 12}" text-anchor="${i === 0 ? "start" : i === 2 ? "end" : "middle"}">${Number(e).toFixed(2)}</text>`;
  }).join("");
  return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="权重直方图">
    ${grid.join("")}${bars}${xl}
  </svg>`;
}

/* ---------------- Markdown（安全渲染） ---------------- */

function inlineMarkdown(text) {
  let out = escapeHtml(text);
  out = out.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  out = out.replace(/`([^`]+)`/g, "<code>$1</code>");
  out = out.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (match, label) => label);
  return out;
}

function markdownToHtml(markdown) {
  const lines = String(markdown || "").replace(/\r\n/g, "\n").split("\n");
  const html = [];
  let paragraph = [];
  let listType = null;
  let inCode = false;
  let codeLines = [];

  const flushParagraph = () => {
    if (paragraph.length) {
      html.push(`<p>${inlineMarkdown(paragraph.join(" "))}</p>`);
      paragraph = [];
    }
  };
  const closeList = () => {
    if (listType) { html.push(`</${listType}>`); listType = null; }
  };

  for (let i = 0; i < lines.length; i += 1) {
    const raw = lines[i];
    const line = raw.trimEnd();

    if (line.startsWith("```")) {
      flushParagraph(); closeList();
      if (inCode) {
        html.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
        codeLines = []; inCode = false;
      } else {
        inCode = true;
      }
      continue;
    }
    if (inCode) { codeLines.push(raw); continue; }

    if (!line.trim()) { flushParagraph(); closeList(); continue; }

    const heading = /^(#{1,4})\s+(.*)$/.exec(line);
    if (heading) {
      flushParagraph(); closeList();
      const level = heading[1].length;
      html.push(`<h${level}>${inlineMarkdown(heading[2])}</h${level}>`);
      continue;
    }

    // 表格：当前行和下一行是分隔行
    if (line.trim().startsWith("|") && i + 1 < lines.length && /^\s*\|?[\s:|-]+\|?\s*$/.test(lines[i + 1]) && lines[i + 1].includes("-")) {
      flushParagraph(); closeList();
      const cells = line.trim().replace(/^\||\|$/g, "").split("|").map((cell) => cell.trim());
      const rows = [];
      i += 1;
      while (i + 1 < lines.length && lines[i + 1].trim().startsWith("|")) {
        i += 1;
        rows.push(lines[i].trim().replace(/^\||\|$/g, "").split("|").map((cell) => cell.trim()));
      }
      html.push(`<table><thead><tr>${cells.map((cell) => `<th>${inlineMarkdown(cell)}</th>`).join("")}</tr></thead><tbody>${rows.map((row) => `<tr>${row.map((cell) => `<td>${inlineMarkdown(cell)}</td>`).join("")}</tr>`).join("")}</tbody></table>`);
      continue;
    }

    const bullet = /^\s*[-*+]\s+(.*)$/.exec(line);
    if (bullet) {
      flushParagraph();
      if (listType !== "ul") { closeList(); html.push("<ul>"); listType = "ul"; }
      html.push(`<li>${inlineMarkdown(bullet[1])}</li>`);
      continue;
    }
    const ordered = /^\s*\d+[.)]\s+(.*)$/.exec(line);
    if (ordered) {
      flushParagraph();
      if (listType !== "ol") { closeList(); html.push("<ol>"); listType = "ol"; }
      html.push(`<li>${inlineMarkdown(ordered[1])}</li>`);
      continue;
    }

    if (/^\s*>\s?/.test(line)) {
      flushParagraph(); closeList();
      html.push(`<blockquote>${inlineMarkdown(line.replace(/^\s*>\s?/, ""))}</blockquote>`);
      continue;
    }

    paragraph.push(line.trim());
  }
  if (inCode) html.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
  flushParagraph(); closeList();
  return html.join("\n");
}

/* ---------------- 实时终端 ---------------- */

function terminalTargets(source) {
  if (source === "runtime") {
    return (state.runtimes || []).filter((item) => item.listening || item.alive)
      .map((item) => ({ value: item.username || item.id, label: `${item.username || item.id} · ${item.envPort ?? "?"} · ${item.state || ""}` }));
  }
  if (source === "training") {
    const job = (state.training || {}).job;
    const runs = state.runs || [];
    const out = [];
    if (job && job.runId) out.push({ value: job.runId, label: `当前训练 ${job.runId}` });
    runs.filter((run) => run.kind === "train" || run.kind === "multi").slice(0, 20).forEach((run) => {
      if (!out.some((item) => item.value === run.id)) out.push({ value: run.id, label: `${run.name}（${run.status}）` });
    });
    return out;
  }
  return [];
}

function refreshTerminalTargets() {
  const source = $("#terminal-source") ? $("#terminal-source").value : "server";
  const nameWrap = $("#terminal-name-wrap");
  const select = $("#terminal-name");
  if (!select || !nameWrap) return;
  const options = terminalTargets(source);
  nameWrap.classList.toggle("hidden", source === "server");
  if (source === "server") {
    select.innerHTML = "";
    state.terminal.name = "";
    return;
  }
  const previous = state.terminal.name;
  select.innerHTML = options.length
    ? options.map((item) => `<option value="${escapeHtml(item.value)}">${escapeHtml(item.label)}</option>`).join("")
    : '<option value="">（暂无可用目标）</option>';
  if (options.some((item) => item.value === previous)) select.value = previous;
  state.terminal.name = select.value;
}

function appendTerminalLine(text) {
  state.terminal.lines.push(String(text ?? ""));
  if (state.terminal.lines.length > 5000) state.terminal.lines = state.terminal.lines.slice(-4000);
  if (state.terminal.paused) return;
  const output = $("#terminal-output");
  if (!output) return;
  output.textContent = state.terminal.lines.join("\n");
  if ($("#terminal-autoscroll").checked) output.scrollTop = output.scrollHeight;
}

function stopTerminalStream() {
  if (state.terminal.es) {
    state.terminal.es.close();
    state.terminal.es = null;
  }
}

function setTerminalStatus(text) {
  const el = $("#terminal-status");
  if (el) el.textContent = text;
}

function connectTerminal() {
  stopTerminalStream();
  const source = $("#terminal-source").value;
  const name = source === "server" ? "" : ($("#terminal-name").value || "");
  state.terminal.source = source;
  state.terminal.name = name;
  state.terminal.lines = [];
  state.terminal.paused = false;
  $("#terminal-output").textContent = "正在连接…";
  $("#btn-terminal-pause").textContent = "暂停";
  const params = new URLSearchParams({ source });
  if (name) params.set("name", name);
  const es = new EventSource(`/api/terminal/stream?${params.toString()}`);
  state.terminal.es = es;
  setTerminalStatus("连接中…（只读）");
  es.addEventListener("meta", (event) => {
    try {
      const meta = JSON.parse(event.data);
      setTerminalStatus(`${meta.label || source} · ${meta.note || "已连接"}${meta.path ? ` · ${meta.path}` : ""}`);
      // 历史 run 没有 run.log：给一句明确结论，而不是空白等待。
      if (meta.noLog) appendTerminalLine("[dashboard] 本次运行没有日志（历史数据）");
    } catch (err) { setTerminalStatus("已连接（只读）"); }
  });
  es.onmessage = (event) => appendTerminalLine(event.data);
  es.onerror = () => {
    if (state.terminal.es === es) setTerminalStatus("连接中断，浏览器会自动重连；也可点「连接」重试。");
  };
}

function initTerminal() {
  if (!$("#terminal-source")) return;
  refreshTerminalTargets();
  // 离开终端页时连接会被主动关闭；回来后即使已有历史行也要重新建立 SSE。
  if (!state.terminal.es) connectTerminal();
}

/* ---------------- 数据加载 ---------------- */

async function loadHealth() {
  if (state.inflight.has("health")) return;
  state.inflight.add("health");
  try {
    state.health = await api("/api/health");
    if (state.health.readOnly) toast("Dashboard 处于只读模式，启动/停止操作已禁用", "err");
  } catch (err) {
    toast(`无法连接 Dashboard 后端: ${err.message}`, "err");
  } finally {
    state.inflight.delete("health");
  }
}

async function loadServer(render = true) {
  if (state.inflight.has("server")) return;
  state.inflight.add("server");
  try {
    const data = await api("/api/server");
    state.server = data.server;
    if (render) { renderServer(); renderOverview(); }
    renderTop();
  } catch (err) {
    toast(`读取服务器状态失败: ${err.message}`, "err");
  } finally {
    state.inflight.delete("server");
  }
}

async function loadRuntimes() {
  if (state.inflight.has("runtimes")) return;
  state.inflight.add("runtimes");
  try {
    const data = await api("/api/runtimes");
    state.runtimes = data.runtimes || [];
    state.runtimeDefaults = data.defaults || state.runtimeDefaults;
    renderRuntimes();
    renderOverview();
    renderTop();
    if (state.tab === "terminal") refreshTerminalTargets();
  } catch (err) {
    toast(`读取运行时失败: ${err.message}`, "err");
  } finally {
    state.inflight.delete("runtimes");
  }
}

async function loadTraining() {
  if (state.inflight.has("training")) return;
  state.inflight.add("training");
  try {
    const data = await api("/api/training");
    state.training = data;
    state.options = data.options || state.options;
    if (state.options && state.tab === "training") fillTrainingForm(state.options);
    renderTraining();
    renderOverview();
    renderTop();
    if (state.tab === "terminal") refreshTerminalTargets();
  } catch (err) {
    toast(`读取训练状态失败: ${err.message}`, "err");
  } finally {
    state.inflight.delete("training");
  }
}

async function loadMatch() {
  if (state.inflight.has("match")) return;
  state.inflight.add("match");
  try {
    const data = await api("/api/match");
    state.match = data.match;
    renderMatch();
    renderOverview();
    renderTop();
  } catch (err) {
    toast(`读取对战状态失败: ${err.message}`, "err");
  } finally {
    state.inflight.delete("match");
  }
}

async function loadArtifacts(notify = false) {
  if (state.mergeMode) {
    await mergeSelected(notify);
    return;
  }
  if (state.inflight.has("artifacts")) return;
  state.inflight.add("artifacts");
  try {
    let data = await api("/api/artifacts");
    const defaultRun = data.selectedRun && data.selectedRun.id;
    if (!state.selectedRun) state.selectedRun = defaultRun || null;
    if (state.selectedRun && defaultRun !== state.selectedRun) {
      data = await api(`/api/artifacts?run=${encodeURIComponent(state.selectedRun)}`);
    }
    state.artifacts = data;
    state.runs = data.runs || [];
    if (state.selectedRun) {
      try {
        const detail = await api(`/api/artifacts/run?id=${encodeURIComponent(state.selectedRun)}`);
        state.selectedRunData = detail.run;
      } catch (err) {
        state.selectedRun = defaultRun || null;
        state.selectedRunData = null;
      }
    }
    renderArtifacts();
    if (notify) toast("训练产物已重新读取", "ok");
  } catch (err) {
    toast(`读取训练产物失败: ${err.message}`, "err");
  } finally {
    state.inflight.delete("artifacts");
  }
}

async function selectRun(runId) {
  if (!runId || runId === "merged") return;
  state.mergeMode = false;
  state.selectedRun = runId;
  state.selectedRunData = null;
  state.pairFilter = null;
  renderArtifacts();
  try {
    const [summary, detail] = await Promise.all([
      api(`/api/artifacts?run=${encodeURIComponent(runId)}`),
      api(`/api/artifacts/run?id=${encodeURIComponent(runId)}`),
    ]);
    state.artifacts = summary;
    state.runs = summary.runs || state.runs;
    state.selectedRunData = detail.run;
    renderArtifacts();
  } catch (err) {
    toast(`读取运行详情失败: ${err.message}`, "err");
  }
}

function selectedRunIds() {
  return (state.runs || []).filter((run) => state.selectedRuns.has(run.id)).map((run) => run.id);
}

async function mergeSelected(notify = false) {
  const ids = selectedRunIds();
  if (!ids.length) {
    toast("请先勾选至少一个运行", "warn");
    return;
  }
  if (state.inflight.has("artifacts")) return;
  state.inflight.add("artifacts");
  try {
    const query = ids.map((id) => encodeURIComponent(id)).join(",");
    const data = await api(`/api/artifacts/merge?runs=${query}`);
    state.artifacts = data;
    state.runs = data.runs || state.runs;
    state.selectedRun = null;
    state.selectedRunData = null;
    state.mergeMode = true;
    state.pairFilter = null;
    renderArtifacts();
    $("#data-cards")?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    if (notify) toast(`已合并 ${ids.length} 次运行`, "ok");
  } catch (err) {
    toast(`合并运行失败: ${err.message}`, "err");
  } finally {
    state.inflight.delete("artifacts");
  }
}

async function pollTick() {
  renderTop();
  await Promise.allSettled([
    loadServer(false),
    loadRuntimes(),
    loadTraining(),
    (state.tab === "match" || (state.match && state.match.running) || state.tab === "overview") ? loadMatch() : Promise.resolve(),
  ]);
  if (state.tab === "overview") {
    renderOverview();
  }
  if (state.tab === "server") renderServer();
  if (state.tab === "runtime") renderRuntimes();
  if (state.tab === "training") renderTraining();
  if (state.tab === "match") renderMatch();
  // 数据页每轮只重渲染一次，让“TPS（实时）”跟上 /api/server，不重复请求产物。
  if (state.tab === "data") renderArtifacts();
}

/* ---------------- 事件绑定 ---------------- */

function bindTabs() {
  $$(".tab").forEach((tab) => tab.addEventListener("click", () => showTab(tab.dataset.tab)));
  document.addEventListener("mouseover", (event) => {
    const hit = event.target.closest("[data-chart-tip]");
    if (hit) showChartTooltip(event, hit);
  });
  document.addEventListener("mousemove", (event) => {
    const hit = event.target.closest("[data-chart-tip]");
    if (hit) showChartTooltip(event, hit);
  });
  document.addEventListener("mouseout", (event) => {
    const hit = event.target.closest("[data-chart-tip]");
    if (hit && !hit.contains(event.relatedTarget)) hideChartTooltip();
  });
  document.addEventListener("click", (event) => {
    const goto = event.target.closest("[data-goto]");
    if (goto) { showTab(goto.dataset.goto); return; }
    const copy = event.target.closest("[data-copy]");
    if (copy) {
      let text = copy.dataset.copy || copy.textContent.trim();
      if (text.startsWith("#")) {
        const target = $(text);
        text = target ? target.textContent.trim() : text;
      }
      copyText(text);
      return;
    }
    const chartHit = event.target.closest("[data-chart-run]");
    if (chartHit && chartHit.dataset.chartRun) {
      hideChartTooltip();
      selectRun(chartHit.dataset.chartRun);
      $("#data-run-detail")?.scrollIntoView({ behavior: "smooth", block: "nearest" });
      return;
    }
    const pairRow = event.target.closest("[data-pair]");
    if (pairRow) {
      const pair = pairRow.dataset.pair;
      state.pairFilter = state.pairFilter === pair ? null : pair;
      renderArtifacts();
      if (state.pairFilter != null) {
        const target = $("#data-reviews");
        if (target) target.scrollIntoView({ behavior: "smooth", block: "nearest" });
      }
      return;
    }
    const runRow = event.target.closest("[data-run]");
    if (runRow) {
      if (event.target.closest("input")) return;
      selectRun(runRow.dataset.run);
      return;
    }
    if (event.target.closest("#btn-run-close")) {
      state.selectedRun = null;
      state.selectedRunData = null;
      state.mergeMode = false;
      state.pairFilter = null;
      loadArtifacts();
      return;
    }
    const review = event.target.closest("[data-review]");
    if (review) { openReview(review.dataset.review); return; }
    const logButton = event.target.closest("[data-runtime-log]");
    if (logButton) {
      state.runtimeLogId = logButton.dataset.runtimeLog;
      renderRuntimes();
      return;
    }
    const stopButton = event.target.closest("[data-runtime-stop]");
    if (stopButton) {
      withBusy(stopButton, async () => {
        await api("/api/runtimes/stop", { method: "POST", body: { id: stopButton.dataset.runtimeStop } });
        toast("运行时已停止", "ok");
        await loadRuntimes();
      });
    }
  });
}

function bindServer() {
  $("#btn-refresh").addEventListener("click", async () => {
    await Promise.allSettled([loadServer(), loadRuntimes(), loadTraining(), loadMatch()]);
    toast("已刷新", "ok");
  });
  $("#btn-server-start").addEventListener("click", (event) => withBusy(event.currentTarget, async () => {
    toast("正在启动 Paper，请等待端口就绪…", "ok");
    await api("/api/server/start", { method: "POST" });
    toast("启动命令已下发", "ok");
    await loadServer();
  }));
  $("#btn-server-stop").addEventListener("click", (event) => {
    if (!window.confirm("停止 Paper 会结束当前对局/训练，最多等待 40 秒。继续？")) return;
    withBusy(event.currentTarget, async () => {
      toast("正在停止 Paper…", "ok");
      await api("/api/server/stop", { method: "POST" });
      toast("服务器已停止", "ok");
      await loadServer();
    });
  });
  $("#console-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const line = $("#console-line").value.trim();
    if (!line) return;
    withBusy(event.submitter || null, async () => {
      await api("/api/server/console", { method: "POST", body: { line } });
      $("#console-line").value = "";
      toast("命令已发送", "ok");
      await loadServer();
    });
  });
  $("#btn-quick-server").addEventListener("click", (event) => withBusy(event.currentTarget, async () => {
    await api("/api/server/start", { method: "POST" });
    toast("启动命令已下发，等待端口就绪…", "ok");
    await loadServer();
  }));
}

function bindRuntimes() {
  $("#runtime-username").addEventListener("input", (event) => { event.target.dataset.touched = "1"; });
  $("#runtime-port").addEventListener("input", (event) => { event.target.dataset.touched = "1"; });
  $("#runtime-form").addEventListener("submit", (event) => {
    event.preventDefault();
    withBusy(event.submitter || null, async () => {
      const payload = {
        username: $("#runtime-username").value.trim(),
        envPort: Number($("#runtime-port").value),
      };
      const data = await api("/api/runtimes/start", { method: "POST", body: payload });
      toast(`运行时已启动（PID ${data.runtime.pid}）`, "ok");
      state.runtimeLogId = data.runtime.id;
      await loadRuntimes();
    });
  });
  $("#btn-runtime-stop-all").addEventListener("click", (event) => {
    if (!window.confirm("停止所有由 Dashboard 托管的 bot-runtime？")) return;
    withBusy(event.currentTarget, async () => {
      await api("/api/runtimes/stop-all", { method: "POST" });
      toast("全部运行时已停止", "ok");
      state.runtimeLogId = null;
      await loadRuntimes();
    });
  });
}

function bindTraining() {
  $("#tr-kind").addEventListener("change", () => {
    const kind = $("#tr-kind").value;
    $$("#training-form [data-kind]").forEach((block) => {
      block.classList.toggle("hidden", block.dataset.kind !== kind);
    });
    $("#btn-training-start").textContent = kind === "multi" ? "开始多场地训练" : "开始训练";
    // 多场地（multi_train）没有评估阶段：字段隐藏 + 说明，避免“填了但不生效”。
    const isMulti = kind === "multi";
    const evalField = $("#tr-eval-field");
    if (evalField) evalField.classList.toggle("hidden", isMulti);
    const evalNote = $("#tr-eval-note");
    if (evalNote) evalNote.classList.toggle("hidden", !isMulti);
  });
  ["#tr-users", "#tr-ports", "#tr-epochs", "#tr-max-steps", "#tr-seed", "#tr-eval", "#tr-center",
    "#tr-spacing", "#tr-prep", "#tr-pairs", "#tr-prefix", "#tr-base-port"].forEach((selector) => {
    const el = $(selector);
    if (el) el.addEventListener("input", () => { el.dataset.touched = "1"; });
  });
  const rewardSelect = $("#tr-reward-preset");
  if (rewardSelect) rewardSelect.addEventListener("change", () => { rewardSelect.dataset.touched = "1"; });
  $("#tr-auto-spawn").addEventListener("change", (event) => { event.target.dataset.touched = "1"; });
  $("#tr-auto-cleanup").addEventListener("change", (event) => { event.target.dataset.touched = "1"; });

  // 计划列表：行内轮次编辑（不重渲染，避免打断输入）
  const planList = $("#tr-plan-list");
  planList.addEventListener("input", (event) => {
    const input = event.target.closest(".plan-episodes");
    if (!input) return;
    const index = Number(input.dataset.index);
    const item = state.plan[index];
    if (!item) return;
    const raw = input.value.trim();
    item.episodes = raw === "" ? null : Number(raw);
    state.planTouched = true;
    const valid = planEntryValid(item);
    const row = input.closest(".plan-row");
    if (row) row.classList.toggle("invalid", !valid);
    input.setAttribute("aria-invalid", valid ? "false" : "true");
    updatePlanMeta();
  });

  // 计划列表：上移 / 下移 / 删除
  planList.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-act]");
    if (!button) return;
    const index = Number(button.dataset.index);
    const act = button.dataset.act;
    if (act === "up") movePlan(index, -1);
    else if (act === "down") movePlan(index, 1);
    else if (act === "remove") {
      if (index < 0 || index >= state.plan.length) return;
      state.plan.splice(index, 1);
      state.planTouched = true;
      renderPlan();
    }
  });

  // 计划列表：HTML5 拖拽排序
  planList.addEventListener("dragstart", (event) => {
    const row = event.target.closest(".plan-row");
    if (!row) return;
    state.dragPlanIndex = Number(row.dataset.index);
    row.classList.add("dragging");
    if (event.dataTransfer) {
      event.dataTransfer.effectAllowed = "move";
      try { event.dataTransfer.setData("text/plain", String(state.dragPlanIndex)); } catch (err) { /* 部分浏览器只读 */ }
    }
  });
  planList.addEventListener("dragover", (event) => {
    const row = event.target.closest(".plan-row");
    if (!row || state.dragPlanIndex === null) return;
    event.preventDefault();
    $$(".plan-row.drag-over", planList).forEach((el) => el.classList.remove("drag-over"));
    row.classList.add("drag-over");
  });
  planList.addEventListener("dragleave", (event) => {
    const row = event.target.closest(".plan-row");
    if (row) row.classList.remove("drag-over");
  });
  planList.addEventListener("drop", (event) => {
    const row = event.target.closest(".plan-row");
    if (!row || state.dragPlanIndex === null) return;
    event.preventDefault();
    const from = state.dragPlanIndex;
    const to = Number(row.dataset.index);
    state.dragPlanIndex = null;
    if (Number.isInteger(from) && Number.isInteger(to) && from !== to) {
      const [item] = state.plan.splice(from, 1);
      state.plan.splice(to, 0, item);
      state.planTouched = true;
      renderPlan();
    }
    $$(".plan-row", planList).forEach((el) => el.classList.remove("drag-over", "dragging"));
  });
  planList.addEventListener("dragend", () => {
    state.dragPlanIndex = null;
    $$(".plan-row", planList).forEach((el) => el.classList.remove("drag-over", "dragging"));
  });

  $("#tr-add-script").addEventListener("change", (event) => {
    event.target.style.color = mcColor(scriptInfo(event.target.value).color);
  });
  $("#btn-plan-add").addEventListener("click", () => {
    const select = $("#tr-add-script");
    const mode = select.value || (state.scripts[0] && state.scripts[0].id);
    if (!mode) {
      toast("脚本列表尚未加载", "err");
      return;
    }
    const raw = $("#tr-add-episodes").value.trim();
    const count = Number(raw);
    if (!raw || !Number.isInteger(count) || count < 1 || count > 500) {
      toast("轮次需要 1–500 的整数", "err");
      return;
    }
    if (state.plan.length >= 50) {
      toast("计划最多 50 段", "err");
      return;
    }
    if (planTotalEpisodes() + count > 5000) {
      toast("总回合超过上限 5000", "err");
      return;
    }
    state.plan.push({ opponentMode: mode, episodes: count });
    state.planTouched = true;
    renderPlan();
  });

  $("#btn-plan-import").addEventListener("click", () => {
    const input = $("#tr-plan-input");
    try {
      const items = parsePlanString(input.value);
      setPlan(items, true);
      input.value = planText();
      toast(`已加载 ${items.length} 段计划`, "ok");
    } catch (err) {
      // 解析失败不覆盖原计划
      toast(err.message || "计划字符串解析失败", "err");
    }
  });
  $("#btn-plan-copy").addEventListener("click", () => {
    const text = planText();
    if (!text) {
      toast("计划为空，没有可复制的内容", "warn");
      return;
    }
    copyText(text);
  });
  $("#btn-plan-clear").addEventListener("click", () => {
    if (!state.plan.length) return;
    if (!window.confirm("清空当前训练计划？")) return;
    setPlan([], true);
  });

  $("#btn-preset-save").addEventListener("click", (event) => savePreset(event.currentTarget));
  $("#tr-presets").addEventListener("click", (event) => {
    const loadBtn = event.target.closest("[data-preset-load]");
    if (loadBtn) {
      loadPresetById(loadBtn.dataset.presetLoad, loadBtn);
      return;
    }
    const deleteBtn = event.target.closest("[data-preset-delete]");
    if (deleteBtn) deletePresetById(deleteBtn.dataset.presetDelete, deleteBtn);
  });

  $("#btn-training-start").addEventListener("click", async (event) => {
    const button = event.currentTarget;
    if (!planIsValid()) {
      toast("请先修正计划：每段轮次需为 1–500 的整数，且至少一段", "err");
      return;
    }
    await withBusy(button, async () => {
      const kind = $("#tr-kind").value;
      const split = (value) => value.split(/[\s,]+/).map((item) => item.trim()).filter(Boolean);
      const payload = {
        kind,
        users: split($("#tr-users").value),
        ports: split($("#tr-ports").value).map(Number),
        plan: state.plan.map(({ opponentMode, episodes }) => ({ opponentMode, episodes })),
        epochs: planEpochs(),
        maxSteps: Number($("#tr-max-steps").value),
        seed: Number($("#tr-seed").value),
        autoSpawn: $("#tr-auto-spawn").checked,
        autoCleanup: $("#tr-auto-cleanup").checked,
        center: $("#tr-center").value.trim(),
        spacing: Number($("#tr-spacing").value),
        prepSeconds: Number($("#tr-prep").value),
        pairs: Number($("#tr-pairs").value),
        usersPrefix: $("#tr-prefix").value.trim(),
        portsBase: Number($("#tr-base-port").value),
      };
      // 多场地没有评估阶段：不发送 evalEpochs（后端也会拒绝 >0）。
      if (kind !== "multi") payload.evalEpochs = Number($("#tr-eval").value);
      const rewardId = currentRewardPresetId();
      if (rewardId) payload.rewardPreset = rewardId;
      const modelId = $("#tr-model") ? $("#tr-model").value : "";
      if (modelId) payload.modelId = modelId;
      const persistBox = $("#tr-persist");
      payload.persist = persistBox ? Boolean(persistBox.checked && !persistBox.disabled) : true;
      const data = await api("/api/training/start", { method: "POST", body: payload });
      toast("训练已启动", "ok");
      state.training = data;
      renderTraining();
      renderTop();
    });
    updatePlanMeta();
  });
  $("#btn-training-stop").addEventListener("click", (event) => {
    if (!window.confirm("停止当前训练任务？")) return;
    withBusy(event.currentTarget, async () => {
      await api("/api/training/stop", { method: "POST" });
      toast("训练已停止", "ok");
      await loadTraining();
    });
  });
  $("#btn-training-start").textContent = "开始训练";
  updatePlanMeta();
}


function bindMatch() {
  $("#btn-match-start").addEventListener("click", (event) => withBusy(event.currentTarget, async () => {
    const payload = {
      count: Number($("#match-count").value),
      usersPrefix: $("#match-prefix").value.trim(),
      portsBase: Number($("#match-base").value),
    };
    const data = await api("/api/match/start", { method: "POST", body: payload });
    state.match = data.match;
    renderMatch();
    toast("正在拉起 Bot，进度见启动日志", "ok");
    setTimeout(loadMatch, 1500);
  }));
  $("#btn-match-stop").addEventListener("click", (event) => {
    if (!window.confirm("结束对战并停止本场拉起的 Bot？")) return;
    withBusy(event.currentTarget, async () => {
      await api("/api/match/stop", { method: "POST" });
      toast("对战已结束", "ok");
      await loadMatch();
      await loadRuntimes();
    });
  });
}

function bindData() {
  $("#btn-data-refresh").addEventListener("click", () => loadArtifacts(true));
  const toggleAll = (checked) => {
    state.selectedRuns = checked ? new Set((state.runs || []).map((run) => run.id)) : new Set();
    renderArtifacts();
  };
  $("#data-check-all").addEventListener("change", (event) => toggleAll(event.target.checked));
  $("#data-check-all-head").addEventListener("change", (event) => toggleAll(event.target.checked));
  $("#data-runs").addEventListener("change", (event) => {
    const box = event.target.closest("[data-run-check]");
    if (!box) return;
    const id = box.dataset.runCheck;
    if (box.checked) state.selectedRuns.add(id);
    else state.selectedRuns.delete(id);
    renderArtifacts();
  });
  $("#btn-data-range").addEventListener("click", () => {
    const fromValue = $("#data-from").value;
    const toValue = $("#data-to").value;
    const from = fromValue ? new Date(fromValue).getTime() : null;
    const to = toValue ? new Date(toValue).getTime() : null;
    if (from == null && to == null) {
      toast("请先填写开始或结束时间", "warn");
      return;
    }
    const chosen = (state.runs || []).filter((run) => {
      const created = new Date(run.startedAt || run.createdAt || "").getTime();
      if (Number.isNaN(created)) return false;
      if (from != null && created < from) return false;
      if (to != null && created > to) return false;
      return true;
    });
    state.selectedRuns = new Set(chosen.map((run) => run.id));
    renderArtifacts();
    toast(`按时间选中 ${chosen.length} 次运行`, chosen.length ? "ok" : "warn");
  });
  $("#btn-data-merge").addEventListener("click", () => mergeSelected(true));
  $("#btn-data-clear").addEventListener("click", () => {
    state.selectedRuns = new Set();
    state.mergeMode = false;
    state.pairFilter = null;
    state.selectedRun = null;
    state.selectedRunData = null;
    loadArtifacts();
  });
  $("#terminal-source").addEventListener("change", () => {
    state.terminal.name = "";
    refreshTerminalTargets();
    stopTerminalStream();
    setTerminalStatus("来源已切换，点「连接」开始。");
  });
  $("#terminal-name").addEventListener("change", (event) => { state.terminal.name = event.target.value; });
  $("#btn-terminal-connect").addEventListener("click", connectTerminal);
  $("#btn-terminal-pause").addEventListener("click", (event) => {
    state.terminal.paused = !state.terminal.paused;
    event.currentTarget.textContent = state.terminal.paused ? "继续" : "暂停";
    if (!state.terminal.paused) {
      const output = $("#terminal-output");
      output.textContent = state.terminal.lines.join("\n");
      output.scrollTop = output.scrollHeight;
    }
  });
  $("#btn-terminal-clear").addEventListener("click", () => {
    state.terminal.lines = [];
    $("#terminal-output").textContent = "—";
    setTerminalStatus("已清空浏览器缓冲（不影响服务端日志）。");
  });
  $("#btn-review-close").addEventListener("click", () => {
    const dialog = $("#review-dialog");
    if (typeof dialog.close === "function") dialog.close();
    else dialog.removeAttribute("open");
  });
  $("#review-dialog").addEventListener("click", (event) => {
    if (event.target === $("#review-dialog")) {
      const dialog = $("#review-dialog");
      if (typeof dialog.close === "function") dialog.close();
    }
  });
}


/* ---------------- 打开文件夹按钮 ---------------- */

function bindFolderButtons() {
  $("#btn-script-folder")?.addEventListener("click", () => openFolder("scripts"));
  $("#btn-preset-folder")?.addEventListener("click", () => openFolder("presets"));
  $("#btn-reward-folder")?.addEventListener("click", () => openFolder("reward-presets"));
  $("#btn-run-folder")?.addEventListener("click", () => {
    const runId = (state.selectedRunData && (state.selectedRunData.id || state.selectedRunData.name)) || state.selectedRun;
    if (runId) openFolder("run", runId);
  });
  $("#btn-run-reviews-folder")?.addEventListener("click", () => {
    const runId = (state.selectedRunData && (state.selectedRunData.id || state.selectedRunData.name)) || state.selectedRun;
    if (runId) openFolder("run-reviews", runId);
  });
}

async function boot() {
  bindTabs();
  bindServer();
  bindRuntimes();
  bindTraining();
  bindRewardEditor();
  bindScriptEditor();
  bindModels();
  bindEpisodeDialog();
  bindFolderButtons();
  bindMatch();
  bindData();
  await loadHealth();
  await Promise.allSettled([loadServer(), loadRuntimes(), loadTraining(), loadMatch()]);
  renderOverview();
  renderTop();
  try {
    const opts = await api("/api/training/options");
    state.options = opts.options;
    fillTrainingForm(state.options);
  } catch (err) { /* 训练选项失败不阻塞页面 */ }
  await loadRewardPresets();
  await loadModels();
  loadArtifacts(false);
  if (location.pathname.replace(/\/+$/, "") === "/terminal") showTab("terminal");
  // 直达链接：/?model=<id>#model-viz → 打开模型页并直接渲染可视化。
  const deepParams = new URLSearchParams(location.search);
  const deepModel = deepParams.get("model");
  if (deepModel && (location.hash === "#model-viz" || deepParams.get("viz") === "1")) {
    showTab("models");
    try {
      await openModelDetail(deepModel);
      await renderModelViz(deepModel);
    } catch (err) {
      toast(err.message || "打开可视化失败", "err");
    }
  }
  setInterval(pollTick, 4000);
}

document.addEventListener("DOMContentLoaded", boot);
