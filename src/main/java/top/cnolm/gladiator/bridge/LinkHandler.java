package top.cnolm.gladiator.bridge;

import com.google.gson.JsonObject;

/**
 * 链路层向编排层投递事件的回调。实现类可以把消息切到 Bukkit 主线程；
 * 链路层自身只处理网络、握手、心跳与信封校验，不调用任何 Bukkit API。
 */
public interface LinkHandler {

    /**
     * WELCOME 的补充字段（serverVersion / features / limits），返回 null 表示使用默认空值。
     */
    JsonObject serverInfo();

    void onHello(LinkServer server, SessionRegistry.Session session, Envelope hello);

    void onMessage(LinkServer server, SessionRegistry.Session session, Envelope message);

    void onDisconnect(LinkServer server, SessionRegistry.Session session, String reason);

    /**
     * 主动规则查询（协议 §2.6，G-204）。实现方必须在主线程完成判定后，
     * 用 {@code server.send(session, "QUERY_RESULT", payload, envelope.seq())} 回包。
     *
     * <p>默认实现返回 degraded 放行，保证链路层在未接线时仍然可用。
     */
    void onQueryPvp(LinkServer server, SessionRegistry.Session session, Envelope query);

    /**
     * WELCOME 中的 {@code playerUuid}：已绑定则返回真实 Player UUID，
     * 玩家实体尚未生成时返回 null（链路层写空串）。默认 null 保持 M1 之前的语义。
     */
    String playerUuid(String username);

    void log(String message);

    /** 空实现，便于测试与渐进接入。 */
    abstract class Adapter implements LinkHandler {

        @Override
        public JsonObject serverInfo() {
            return null;
        }

        @Override
        public void onHello(LinkServer server, SessionRegistry.Session session, Envelope hello) {
        }

        @Override
        public void onMessage(LinkServer server, SessionRegistry.Session session, Envelope message) {
        }

        @Override
        public void onDisconnect(LinkServer server, SessionRegistry.Session session, String reason) {
        }

        @Override
        public void onQueryPvp(LinkServer server, SessionRegistry.Session session, Envelope query) {
            JsonObject payload = new JsonObject();
            payload.addProperty("allowed", Boolean.TRUE);
            payload.add("regionIds", new com.google.gson.JsonArray());
            payload.addProperty("source", "degraded");
            server.send(session, "QUERY_RESULT", payload, Long.valueOf(query.seq()));
        }

        @Override
        public String playerUuid(String username) {
            return null;
        }

        @Override
        public void log(String message) {
        }
    }
}
