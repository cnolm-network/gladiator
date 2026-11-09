package top.cnolm.gladiator.bridge;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.BufferedInputStream;
import java.io.ByteArrayOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * G-101 / G-102 / G-103 链路测试。
 *
 * <p>这里的 WebSocket 客户端是**独立实现**（纯 JDK Socket + 手工掩码帧），
 * 不复用被测的 WebSocketConnection，用于验证 RFC 6455 互操作性。
 */
public class LinkServerTest {

    private static final String TOKEN = "s3cr3t";

    private LinkServer server;
    private WsClient client;
    private RecordingHandler handler;

    @Before
    public void setUp() throws Exception {
        handler = new RecordingHandler();
        server = new LinkServer("127.0.0.1", 0, TOKEN, 30L, handler);
        server.start();
    }

    @After
    public void tearDown() {
        if (client != null) {
            client.close();
            client = null;
        }
        if (server != null) {
            server.stop();
            server = null;
        }
    }

    @Test
    public void bindsToLoopbackOnly() {
        assertTrue(server.isRunning());
        assertTrue(server.getPort() > 0);
        assertNotNull(server.getBoundAddress());
        assertTrue(server.getBoundAddress().isLoopbackAddress());
    }

    @Test
    public void rejectsNonLoopbackHost() {
        LinkServer bad = new LinkServer("0.0.0.0", 0, "", 30L, new LinkHandler.Adapter() {
        });
        try {
            bad.start();
            fail("0.0.0.0 必须被拒绝");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("回环"));
        } catch (IOException e) {
            fail("应抛出 IllegalArgumentException，实际为 " + e);
        } finally {
            bad.stop();
        }
    }

    @Test
    public void helloWithValidTokenReceivesWelcome() throws Exception {
        client = connect();
        client.send(hello(TOKEN, "glad-bot-1"));

        JsonObject welcome = parse(client.readText());
        assertEquals("WELCOME", welcome.get("type").getAsString());
        assertEquals(1, welcome.get("v").getAsInt());
        assertEquals("glad-bot-1", welcome.get("botId").getAsString());
        JsonObject payload = welcome.getAsJsonObject("payload");
        assertTrue(payload.get("sessionId").getAsString().length() > 0);
        assertNotNull(payload.get("limits"));
        assertNotNull(payload.get("features"));
        // G-103：接口默认不提供 playerUuid，WELCOME 必须写空串而不是 null/缺失。
        assertEquals("", payload.get("playerUuid").getAsString());
        assertEquals(1, server.getSessions().size());
        assertEquals(1, handler.hellos.size());
    }

    @Test
    public void welcomeCarriesPlayerUuidWhenBound() throws Exception {
        final String uuid = "3f1c2a44-8f7e-3f0a-9d6b-1c2d3e4f5a6b";
        LinkServer boundServer = new LinkServer("127.0.0.1", 0, "", 30L,
                new LinkHandler.Adapter() {
                    @Override
                    public String playerUuid(String username) {
                        return "Glad_Bot_1".equals(username) ? uuid : null;
                    }
                });
        boundServer.start();
        WsClient boundClient = null;
        try {
            boundClient = new WsClient(boundServer.getPort());
            boundClient.send(hello("", "glad-bot-1"));
            JsonObject welcome = parse(boundClient.readText());
            assertEquals(uuid, welcome.getAsJsonObject("payload").get("playerUuid").getAsString());
        } finally {
            if (boundClient != null) {
                boundClient.close();
            }
            boundServer.stop();
        }
    }

    @Test
    public void helloWithWrongTokenIsRejected() throws Exception {
        client = connect();
        client.send(hello("wrong-token", "glad-bot-1"));

        JsonObject reject = parse(client.readText());
        assertEquals("REJECT", reject.get("type").getAsString());
        assertEquals("AUTH_FAILED",
                reject.getAsJsonObject("payload").get("code").getAsString());
        assertEquals(0, server.getSessions().size());
    }

    @Test
    public void versionMismatchIsRejected() throws Exception {
        client = connect();
        client.send(envelope(2, "HELLO", helloPayload(TOKEN, "glad-bot-1")));

        JsonObject reject = parse(client.readText());
        assertEquals("REJECT", reject.get("type").getAsString());
        assertEquals("VERSION_UNSUPPORTED",
                reject.getAsJsonObject("payload").get("code").getAsString());
    }

    @Test
    public void messageBeforeHelloIsRejected() throws Exception {
        client = connect();
        client.send(envelope(1, "STATE", "{}"));

        JsonObject reject = parse(client.readText());
        assertEquals("REJECT", reject.get("type").getAsString());
        assertEquals("SESSION_UNKNOWN",
                reject.getAsJsonObject("payload").get("code").getAsString());
    }

    @Test
    public void pingReceivesPong() throws Exception {
        client = connect();
        client.send(hello(TOKEN, "glad-bot-1"));
        parse(client.readText()); // WELCOME
        client.send(envelope(1, "PING", "{}"));

        JsonObject pong = parse(client.readText());
        assertEquals("PONG", pong.get("type").getAsString());
    }

    @Test
    public void idleSessionIsSwept() throws Exception {
        RecordingHandler localHandler = new RecordingHandler();
        LinkServer shortServer = new LinkServer("127.0.0.1", 0, "", 1L, localHandler);
        shortServer.start();
        WsClient localClient = null;
        try {
            localClient = new WsClient(shortServer.getPort());
            localClient.send(hello("", "glad-short"));
            assertEquals("WELCOME", parse(localClient.readText()).get("type").getAsString());
            assertEquals(1, shortServer.getSessions().size());
            Thread.sleep(3500L);
            assertEquals(0, shortServer.getSessions().size());
            assertTrue(localHandler.disconnects.toString().contains("heartbeat-timeout"));
        } finally {
            if (localClient != null) {
                localClient.close();
            }
            shortServer.stop();
        }
    }

    private WsClient connect() throws IOException {
        return new WsClient(server.getPort());
    }

    private static JsonObject parse(String text) {
        assertNotNull("服务端未返回文本帧", text);
        return new JsonParser().parse(text).getAsJsonObject();
    }

    private static String hello(String token, String botId) {
        return envelope(1, "HELLO", helloPayload(token, botId));
    }

    private static String helloPayload(String token, String botId) {
        return "{\"token\":\"" + token + "\",\"botId\":\"" + botId + "\","
                + "\"username\":\"Glad_Bot_1\",\"mcVersion\":\"1.8.8\","
                + "\"runtimeVersion\":\"0.1.0\"}";
    }

    private static String envelope(int version, String type, String payload) {
        return "{\"v\":" + version + ",\"type\":\"" + type + "\",\"seq\":1,"
                + "\"ts\":0,\"botId\":\"\",\"replyTo\":null,\"payload\":" + payload + "}";
    }

    private static final class RecordingHandler extends LinkHandler.Adapter {
        private final List<String> hellos = new ArrayList<String>();
        private final List<String> disconnects = new ArrayList<String>();

        @Override
        public void onHello(LinkServer server, SessionRegistry.Session session, Envelope hello) {
            hellos.add(session.getBotId());
        }

        @Override
        public void onDisconnect(LinkServer server, SessionRegistry.Session session, String reason) {
            disconnects.add(reason);
        }
    }

    /** 独立的极简 RFC 6455 客户端：升级握手 + 掩码文本帧 + 解析服务端帧。 */
    private static final class WsClient {

        private final Socket socket;
        private final InputStream in;
        private final OutputStream out;
        private final SecureRandom random = new SecureRandom();

        WsClient(int port) throws IOException {
            socket = new Socket();
            socket.connect(new InetSocketAddress("127.0.0.1", port), 3000);
            socket.setSoTimeout(5000);
            in = new BufferedInputStream(socket.getInputStream());
            out = socket.getOutputStream();

            byte[] key = new byte[16];
            random.nextBytes(key);
            String request = "GET /gladiator HTTP/1.1\r\n"
                    + "Host: 127.0.0.1:" + port + "\r\n"
                    + "Upgrade: websocket\r\n"
                    + "Connection: Upgrade\r\n"
                    + "Sec-WebSocket-Key: " + Base64.getEncoder().encodeToString(key) + "\r\n"
                    + "Sec-WebSocket-Version: 13\r\n"
                    + "\r\n";
            out.write(request.getBytes("US-ASCII"));
            out.flush();
            if (!readHeaders().startsWith("HTTP/1.1 101")) {
                throw new IOException("WebSocket upgrade was rejected");
            }
        }

        private String readHeaders() throws IOException {
            ByteArrayOutputStream buffer = new ByteArrayOutputStream();
            while (true) {
                int c = in.read();
                if (c < 0) {
                    throw new EOFException("closed during handshake");
                }
                buffer.write(c);
                byte[] bytes = buffer.toByteArray();
                int n = bytes.length;
                if (n >= 4 && bytes[n - 4] == '\r' && bytes[n - 3] == '\n'
                        && bytes[n - 2] == '\r' && bytes[n - 1] == '\n') {
                    return new String(bytes, "US-ASCII");
                }
                if (n > 16384) {
                    throw new IOException("response header too large");
                }
            }
        }

        void send(String json) throws IOException {
            byte[] payload = json.getBytes("UTF-8");
            byte[] mask = new byte[4];
            random.nextBytes(mask);
            ByteArrayOutputStream frame = new ByteArrayOutputStream(payload.length + 16);
            frame.write(0x81); // FIN + TEXT
            int length = payload.length;
            if (length < 126) {
                frame.write(0x80 | length);
            } else if (length < 65536) {
                frame.write(0x80 | 126);
                frame.write((length >>> 8) & 0xFF);
                frame.write(length & 0xFF);
            } else {
                frame.write(0x80 | 127);
                for (int shift = 56; shift >= 0; shift -= 8) {
                    frame.write((int) ((length >>> shift) & 0xFF));
                }
            }
            frame.write(mask, 0, 4);
            for (int i = 0; i < length; i++) {
                frame.write(payload[i] ^ mask[i & 3]);
            }
            out.write(frame.toByteArray());
            out.flush();
        }

        String readText() throws IOException {
            while (true) {
                int b0 = in.read();
                if (b0 < 0) {
                    return null;
                }
                int b1 = in.read();
                if (b1 < 0) {
                    return null;
                }
                int opcode = b0 & 0x0F;
                long length = readLength(b1);
                byte[] mask = ((b1 & 0x80) != 0) ? readFully(4) : null;
                byte[] payload = readFully((int) length);
                if (mask != null) {
                    for (int i = 0; i < payload.length; i++) {
                        payload[i] = (byte) (payload[i] ^ mask[i & 3]);
                    }
                }
                if (opcode == 0x1 || opcode == 0x0) {
                    return new String(payload, "UTF-8");
                }
                if (opcode == 0x8) {
                    return null;
                }
                // PING / PONG 控制帧跳过
            }
        }

        private long readLength(int b1) throws IOException {
            int first = b1 & 0x7F;
            if (first < 126) {
                return first;
            }
            if (first == 126) {
                byte[] ext = readFully(2);
                return ((ext[0] & 0xFF) << 8) | (ext[1] & 0xFF);
            }
            byte[] ext = readFully(8);
            long value = 0L;
            for (int i = 0; i < 8; i++) {
                value = (value << 8) | (ext[i] & 0xFF);
            }
            return value;
        }

        private byte[] readFully(int length) throws IOException {
            byte[] data = new byte[length];
            int offset = 0;
            while (offset < length) {
                int read = in.read(data, offset, length - offset);
                if (read < 0) {
                    throw new EOFException("closed mid-frame");
                }
                offset += read;
            }
            return data;
        }

        void close() {
            try {
                socket.close();
            } catch (IOException ignored) {
                // ignore
            }
        }
    }
}
