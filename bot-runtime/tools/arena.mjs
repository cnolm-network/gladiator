// 临时工具：批量起 4 场地 8 个运行时并 spawn（VecEnv 真机验收用）。
// 用法：node bot-runtime/tmp-arena.mjs start|stop [arenas]
import { execFile } from "node:child_process";

const DASH = "http://127.0.0.1:8080";
const args = process.argv.slice(2);
const action = args[0] || "start";
const arenas = Number(args[1] || 4);
const PORTS_BASE = 8770;
const PREFIX = "Bot_V";

const rcon = (cmd) => new Promise((resolve) => {
  execFile("node", ["bot-runtime/tmp-rcon.mjs", cmd], (err, stdout) => {
    resolve(stdout || (err ? String(err) : ""));
  });
});

const call = async (path, body) => {
  const res = await fetch(DASH + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  return res.json().catch(() => ({}));
};

const users = [];
for (let i = 0; i < arenas; i += 1) {
  users.push({ name: `${PREFIX}${i}A`, port: PORTS_BASE + i * 2 });
  users.push({ name: `${PREFIX}${i}B`, port: PORTS_BASE + i * 2 + 1 });
}

if (action === "start") {
  for (const u of users) {
    const started = await call("/api/runtimes/start", { username: u.name, envPort: u.port });
    console.log(`start ${u.name}:${u.port} ok=${started.ok} ${started.error || ""}`);
  }
  await new Promise((r) => setTimeout(r, 15000));
  for (const u of users) {
    const out = await rcon(`gladiator spawn ${u.name}`);
    console.log(`spawn ${u.name}: ${out.split("\n").filter((l) => l.includes("spawn")).join(" | ")}`);
  }
  await new Promise((r) => setTimeout(r, 12000));
  const list = await rcon("list");
  console.log("online:", list.split("\n").find((l) => l.includes("players online")));
} else {
  const listing = await fetch(`${DASH}/api/runtimes`).then((r) => r.json());
  for (const runtime of listing.runtimes || []) {
    const match = users.find((u) => String(runtime.username || "").toLowerCase() === u.name.toLowerCase());
    if (!match) continue;
    const stopped = await call("/api/runtimes/stop", { id: runtime.id });
    console.log(`stop ${match.name} (${runtime.id}): ${JSON.stringify(stopped).slice(0, 120)}`);
  }
}
