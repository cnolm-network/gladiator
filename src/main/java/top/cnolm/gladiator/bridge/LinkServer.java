package top.cnolm.gladiator.bridge;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;

import java.io.IOException;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.security.MessageDigest;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

/**
 * 通道 A 的插件侧 WebSocket 服务端（G-101 / G-102 / G-103 链路骨架）。
 *
 * <p>约束：只绑定回环地址；握手 token 取自 config；协议主版本不匹配即拒绝；
 * 30 秒无活动回收会话。所有游戏逻辑通过 {@link LinkHandler} 交给编排层，
 * 本类不引用 Bukkit，可在普通 JVM 中完整测试。
 */
public final class LinkServer {

    private static final long SWEEP_INTERVAL_SECONDS = 1L;

    private final String host;
    private final int requestedPort;
    private final String token;
    private final long heartbeatTimeoutMillis;
    private final LinkHandler handler;
    private final SessionRegistry sessions = new SessionRegistry();
    private final AtomicLong outboundSeq = new AtomicLong();

    private volatile ServerSocket serverSocket;
    private volatile boolean running;
    private volatile int boundPort = -1;
    private Thread acceptThread;
    private ScheduledExecutorService sweeper;

    /**
     * @param host                     必须是回环地址（127.0.0.1 / ::1 / localhost）
     * @param port                     0 表示由系统分配，便于测试
     * @param token                    空串表示不校验（仅限本机调试）
     * @param heartbeatTimeoutSeconds  无活动判定死亡的秒数
     */
    public LinkServer(String host, int port, String token,
                      long heartbeatTimeoutSeconds, LinkHandler handler) {
        this.host = host == null ? "127.0.0.1" : host;
        this.requestedPort = port;
        this.token = token == null ? "" : token;
        this.heartbeatTimeoutMillis = Math.max(1L, heartbeatTimeoutSeconds) * 1000L;
        if (handler == null) {
            throw new IllegalArgumentException("handler");
        }
        this.handler = handler;
    }

    public synchronized void start() throws IOException {
        if (running) {
            return;
        }
        InetAddress address = InetAddress.getByName(host);
        if (!address.isLoopbackAddress()) {
            throw new IllegalArgumentException("LinkServer 仅允许绑定回环地址，拒绝: " + host);
        }
        ServerSocket socket = new ServerSocket();
        socket.bind(new InetSocketAddress(address, requestedPort), 16);
        serverSocket = socket;
        boundPort = socket.getLocalPort();
        running = true;

        acceptThread = new Thread(new Runnable() {
            @Override
            public void run() {
                acceptLoop();
            }
        }, "gladiator-link-accept");
        acceptThread.setDaemon(true);
        acceptThread.start();

        sweeper = Executors.newSingleThreadScheduledExecutor(new ThreadFactory() {
            @Override
            public Thread newThread(Runnable r) {
                Thread t = new Thread(r, "gladiator-link-heartbeat");
                t.setDaemon(true);
                return t;
            }
        });
        sweeper.scheduleAtFixedRate(new Runnable() {
            @Override
            public void run() {
                sweep();
            }
        }, SWEEP_INTERVAL_SECONDS, SWEEP_INTERVAL_SECONDS, TimeUnit.SECONDS);

        handler.log("LinkServer listening on " + host + ":" + boundPort
                + " (token=" + (token.length() == 0 ? "disabled" : "required")
                + ", heartbeat=" + (heartbeatTimeoutMillis / 1000L) + "s)");
    }

    public synchronized void stop() {
        if (!running) {
            return;
        }
        running = false;
        try {
            ServerSocket socket = serverSocket;
            if (socket != null && !socket.isClosed()) {
                socket.close();
            }
        } catch (IOException ignored) {
            // ignore
        }
        if (sweeper != null) {
            sweeper.shutdownNow();
            sweeper = null;
        }
        for (SessionRegistry.Session session : sessions.all()) {
            notifyDisconnect(session, "server-stop");
        }
        sessions.clear();
        if (acceptThread != null) {
            acceptThread.interrupt();
            acceptThread = null;
        }
        handler.log("LinkServer stopped");
    }

    public boolean isRunning() {
        return running;
    }

    /** 实际监听端口；port=0 时返回系统分配值。 */
    public int getPort() {
        return boundPort;
    }

    public InetAddress getBoundAddress() {
        ServerSocket socket = serverSocket;
        return socket == null ? null : socket.getInetAddress();
    }

    public String getHost() {
        return host;
    }

    public SessionRegistry getSessions() {
        return sessions;
    }

    /** 向已注册会话发送一条出站报文；会话不存在或写失败返回 false。 */
    public boolean send(String botId, String type, JsonObject payload, Long replyTo) {
        SessionRegistry.Session session = sessions.findByBotId(botId);
        if (session == null) {
            return false;
        }
        return write(session, Envelope.of(type, outboundSeq.incrementAndGet(), botId, replyTo, payload));
    }

    /**
     * 直接向某个会话发送出站报文（G-107：SPAWN / BOT_BOUND 不依赖 botId 反查，
     * 因为会话可能尚未进入 {@link SessionRegistry} 的正常索引路径或已被替换）。
     */
    public boolean send(SessionRegistry.Session session, String type, JsonObject payload,
                        Long replyTo) {
        if (session == null) {
            return false;
        }
        return write(session, Envelope.of(type, outboundSeq.incrementAndGet(),
                session.getBotId(), replyTo, payload));
    }

    private boolean write(SessionRegistry.Session session, Envelope envelope) {
        try {
            session.connection().sendText(envelope.toJson());
            return true;
        } catch (IOException e) {
            handler.log("link write failed for " + session.getBotId() + ": " + e.getMessage());
            return false;
        }
    }

    private void acceptLoop() {
        while (running) {
            Socket socket;
            try {
                socket = serverSocket.accept();
            } catch (IOException e) {
                if (running) {
                    handler.log("link accept error: " + e.getMessage());
                }
                break;
            }
            if (!socket.getInetAddress().isLoopbackAddress()) {
                handler.log("rejected non-loopback connection from " + socket.getInetAddress());
                try {
                    socket.close();
                } catch (IOException ignored) {
                    // ignore
                }
                continue;
            }
            Thread worker = new Thread(new ConnectionTask(socket), "gladiator-link-client");
            worker.setDaemon(true);
            worker.start();
        }
    }

    private final class ConnectionTask implements Runnable {

        private final Socket socket;

        ConnectionTask(Socket socket) {
            this.socket = socket;
        }

        @Override
        public void run() {
            WebSocketConnection connection = null;
            SessionRegistry.Session session = null;
            try {
                connection = WebSocketConnection.upgrade(socket);
                while (running) {
                    String text = connection.readText();
                    if (text == null) {
                        break;
                    }
                    Envelope envelope;
                    try {
                        envelope = Envelope.parse(text);
                    } catch (Envelope.MalformedException e) {
                        handler.log("malformed envelope: " + e.getMessage());
                        reject(connection, "INTERNAL", e.getMessage());
                        break;
                    }
                    if (envelope.version() != Envelope.VERSION) {
                        reject(connection, "VERSION_UNSUPPORTED",
                                "expected v" + Envelope.VERSION + ", got v" + envelope.version());
                        break;
                    }
                    String type = envelope.type();
                    if ("HELLO".equals(type)) {
                        if (!authenticate(connection, envelope)) {
                            break;
                        }
                        session = handshake(connection, envelope);
                    } else if ("PING".equals(type)) {
                        if (session != null) {
                            session.touch();
                        }
                        JsonObject empty = new JsonObject();
                        writeTo(connection, "PONG", envelope.botId(),
                                Long.valueOf(envelope.seq()), empty);
                    } else if (session == null) {
                        reject(connection, "SESSION_UNKNOWN", "HELLO required before " + type);
                        break;
                    } else if ("QUERY_PVP".equals(type)) {
                        session.touch();
                        handler.onQueryPvp(LinkServer.this, session, envelope);
                    } else {
                        session.touch();
                        handler.onMessage(LinkServer.this, session, envelope);
                    }
                }
            } catch (IOException e) {
                if (running) {
                    handler.log("link connection closed: " + e.getMessage());
                }
            } finally {
                if (connection != null) {
                    connection.close();
                } else {
                    try {
                        socket.close();
                    } catch (IOException ignored) {
                        // ignore
                    }
                }
                if (session != null) {
                    sessions.remove(session);
                    notifyDisconnect(session, "disconnected");
                }
            }
        }
    }

    private boolean authenticate(WebSocketConnection connection, Envelope hello) throws IOException {
        if (token.length() == 0) {
            return true;
        }
        String presented = hello.payloadString("token", "");
        boolean ok = MessageDigest.isEqual(
                token.getBytes("UTF-8"), presented.getBytes("UTF-8"));
        if (!ok) {
            reject(connection, "AUTH_FAILED", "token mismatch");
        }
        return ok;
    }

    private SessionRegistry.Session handshake(WebSocketConnection connection, Envelope hello)
            throws IOException {
        String botId = hello.payloadString("botId", "");
        if (botId.length() == 0) {
            reject(connection, "INTERNAL", "HELLO payload.botId is required");
            return null;
        }
        String username = hello.payloadString("username", botId);
        SessionRegistry.Session previous = sessions.findByBotId(botId);
        if (previous != null) {
            sessions.remove(previous);
            previous.connection().close();
            notifyDisconnect(previous, "replaced-by-reconnect");
        }
        SessionRegistry.Session session = sessions.register(botId, username, connection);
        handler.onHello(LinkServer.this, session, hello);

        JsonObject payload = new JsonObject();
        payload.addProperty("sessionId", session.getSessionId());
        // G-103/G-104：已绑定 BotHandle 时为真实 Player UUID；玩家实体尚未生成时为空串，
        // 运行时需以此判断是否需要等待/重试（见 protocol.md §2.2）。
        String playerUuid = handler.playerUuid(username);
        payload.addProperty("playerUuid", playerUuid == null ? "" : playerUuid);
        payload.addProperty("serverVersion", "unknown");
        JsonArray features = new JsonArray();
        features.add(new JsonPrimitive("state.machine"));
        features.add(new JsonPrimitive("analyze"));
        payload.add("features", features);
        JsonObject limits = new JsonObject();
        limits.addProperty("maxActionsPerSecond", Integer.valueOf(10));
        limits.addProperty("observationHz", Integer.valueOf(5));
        limits.addProperty("idleAfterSeconds", Integer.valueOf(15));
        payload.add("limits", limits);

        JsonObject info = handler.serverInfo();
        if (info != null) {
            for (Map.Entry<String, JsonElement> entry : info.entrySet()) {
                payload.add(entry.getKey(), entry.getValue());
            }
        }
        writeTo(connection, "WELCOME", botId, Long.valueOf(hello.seq()), payload);
        return session;
    }

    private void reject(WebSocketConnection connection, String code, String message)
            throws IOException {
        JsonObject payload = new JsonObject();
        payload.addProperty("code", code);
        payload.addProperty("message", message);
        writeTo(connection, "REJECT", "", null, payload);
    }

    private void writeTo(WebSocketConnection connection, String type, String botId,
                         Long replyTo, JsonObject payload) throws IOException {
        connection.sendText(Envelope.of(type, outboundSeq.incrementAndGet(),
                botId, replyTo, payload).toJson());
    }

    private void sweep() {
        if (!running) {
            return;
        }
        for (SessionRegistry.Session session : sessions.sweepExpired(heartbeatTimeoutMillis)) {
            sessions.remove(session);
            session.connection().close();
            notifyDisconnect(session, "heartbeat-timeout");
            handler.log("session expired: " + session.getBotId());
        }
    }

    private void notifyDisconnect(SessionRegistry.Session session, String reason) {
        if (!session.markDisconnected()) {
            return;
        }
        try {
            handler.onDisconnect(LinkServer.this, session, reason);
        } catch (RuntimeException e) {
            handler.log("onDisconnect handler error: " + e.getMessage());
        }
    }
}
