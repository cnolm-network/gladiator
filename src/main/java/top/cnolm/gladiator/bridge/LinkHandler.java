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
        public void log(String message) {
        }
    }
}
