package top.cnolm.gladiator.bridge;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * 通道 A 的会话表（G-102）：botId / sessionId → 连接与最后活动时间。
 * 心跳扫描在 LinkServer 的调度线程执行，读写全部线程安全；本类不引用 Bukkit。
 */
public final class SessionRegistry {

    /** 一个已通过 HELLO 注册的运行时连接。 */
    public static final class Session {

        private final String sessionId;
        private final String botId;
        private final String username;
        private final WebSocketConnection connection;
        private final AtomicBoolean disconnectNotified = new AtomicBoolean(false);
        private volatile long lastActivityMillis;

        private Session(String sessionId, String botId, String username, WebSocketConnection connection) {
            this.sessionId = sessionId;
            this.botId = botId;
            this.username = username;
            this.connection = connection;
            this.lastActivityMillis = System.currentTimeMillis();
        }

        public String getSessionId() {
            return sessionId;
        }

        public String getBotId() {
            return botId;
        }

        public String getUsername() {
            return username;
        }

        /** 仅供同包的链路层读写帧。 */
        WebSocketConnection connection() {
            return connection;
        }

        public void touch() {
            lastActivityMillis = System.currentTimeMillis();
        }

        public long getLastActivityMillis() {
            return lastActivityMillis;
        }

        public boolean isExpired(long now, long timeoutMillis) {
            return now - lastActivityMillis > timeoutMillis;
        }

        /** 保证断连回调只派发一次（连接线程与心跳线程可能同时收尾）。 */
        boolean markDisconnected() {
            return disconnectNotified.compareAndSet(false, true);
        }
    }

    private final ConcurrentMap<String, Session> byBotId = new ConcurrentHashMap<String, Session>();
    private final ConcurrentMap<String, Session> bySessionId = new ConcurrentHashMap<String, Session>();

    public Session register(String botId, String username, WebSocketConnection connection) {
        Session session = new Session(UUID.randomUUID().toString(), botId, username, connection);
        byBotId.put(botId, session);
        bySessionId.put(session.getSessionId(), session);
        return session;
    }

    public Session findByBotId(String botId) {
        return botId == null ? null : byBotId.get(botId);
    }

    public Session findBySessionId(String sessionId) {
        return sessionId == null ? null : bySessionId.get(sessionId);
    }

    public void remove(Session session) {
        if (session == null) {
            return;
        }
        byBotId.remove(session.getBotId(), session);
        bySessionId.remove(session.getSessionId(), session);
    }

    public Collection<Session> all() {
        return byBotId.values();
    }

    public int size() {
        return byBotId.size();
    }

    /** 返回心跳超时会话，由调用方负责关闭连接与派发断连回调。 */
    public List<Session> sweepExpired(long timeoutMillis) {
        long now = System.currentTimeMillis();
        List<Session> expired = new ArrayList<Session>();
        for (Session session : byBotId.values()) {
            if (session.isExpired(now, timeoutMillis)) {
                expired.add(session);
            }
        }
        return expired;
    }

    /** 关闭全部连接并清空表；仅在插件 onDisable 时调用。 */
    public void clear() {
        for (Session session : byBotId.values()) {
            session.connection().close();
        }
        byBotId.clear();
        bySessionId.clear();
    }
}
