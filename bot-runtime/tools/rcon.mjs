// 最小 RCON 客户端：node rcon.mjs "command" [--port 25575] [--password devpass]
import net from 'node:net';

const args = process.argv.slice(2);
const command = args[0];
let port = 25575;
let password = 'devpass';
for (let i = 1; i < args.length; i += 2) {
  if (args[i] === '--port') port = Number(args[i + 1]);
  if (args[i] === '--password') password = args[i + 1];
}

const pack = (id, type, body) => {
  const payload = Buffer.from(body, 'utf8');
  const buf = Buffer.alloc(14 + payload.length);
  buf.writeInt32LE(10 + payload.length, 0);
  buf.writeInt32LE(id, 4);
  buf.writeInt32LE(type, 8);
  payload.copy(buf, 12);
  buf.writeInt16LE(0, 12 + payload.length);
  return buf;
};

const socket = net.createConnection({ host: '127.0.0.1', port }, () => {
  socket.write(pack(1, 3, password));
});

let stage = 0;
let acc = Buffer.alloc(0);
socket.on('data', (chunk) => {
  acc = Buffer.concat([acc, chunk]);
  while (acc.length >= 12) {
    const len = acc.readInt32LE(0);
    if (acc.length < len + 4) return;
    const id = acc.readInt32LE(4);
    const type = acc.readInt32LE(8);
    const body = acc.subarray(12, len + 2).toString('utf8');
    acc = acc.subarray(len + 4);
    if (type === 2 && id === 1) {
      socket.write(pack(2, 2, command));
    } else if (type === 0) {
      console.log(body);
      stage = 1;
      socket.end();
    }
  }
});

socket.on('error', (error) => {
  console.error('rcon error:', error.message);
  process.exit(1);
});
socket.on('close', () => process.exit(stage ? 0 : 1));
