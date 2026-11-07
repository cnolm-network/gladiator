package top.cnolm.gladiator.bridge;

import java.io.BufferedInputStream;
import java.io.ByteArrayOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.Socket;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Base64;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

/**
 * RFC 6455 服务端连接的**最小可用子集**：HTTP Upgrade 握手、文本帧、分片、
 * 客户端掩码校验、PING/PONG、CLOSE。
 *
 * <p>选择纯 JDK 实现而非引入 Java-WebSocket / Netty 的理由见 ADR-017：
 * 1.8.8 服务端自带 Netty 版本不固定，shade 第三方库又会与同服插件冲突；
 * 通道只服务本机、己方 mineflayer/ws 客户端，帧子集足以覆盖且便于审计。
 *
 * <p>本类不引用 Bukkit，可在普通 JVM 中测试。二进制帧在协议 v1 中不使用。
 */
final class WebSocketConnection {

    private static final String WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
    private static final int MAX_HEADER_BYTES = 16 * 1024;
    private static final int MAX_MESSAGE_BYTES = 1024 * 1024;
    private static final byte[] EMPTY = new byte[0];

    /** HTTP Upgrade 失败。 */
    static final class HandshakeException extends IOException {
        private static final long serialVersionUID = 1L;

        HandshakeException(String message) {
            super(message);
        }
    }

    private final Socket socket;
    private final InputStream in;
    private final OutputStream out;
    private final Object writeLock = new Object();
    private volatile boolean closed;

    private WebSocketConnection(Socket socket, InputStream in, OutputStream out) {
        this.socket = socket;
        this.in = in;
        this.out = out;
    }

    /** 读取请求头、校验并完成 101 Switching Protocols。 */
    static WebSocketConnection upgrade(Socket socket) throws IOException {
        socket.setTcpNoDelay(true);
        InputStream rawIn = socket.getInputStream();
        BufferedInputStream in = new BufferedInputStream(rawIn);
        OutputStream out = socket.getOutputStream();

        byte[] header = readHeader(in);
        String text = new String(header, "US-ASCII");
        String[] lines = text.split("\r\n");
        if (lines.length == 0 || !lines[0].toUpperCase(Locale.ROOT).startsWith("GET ")) {
            throw new HandshakeException("not a WebSocket GET request");
        }
        Map<String, String> headers = new HashMap<String, String>();
        for (int i = 1; i < lines.length; i++) {
            int colon = lines[i].indexOf(':');
            if (colon <= 0) {
                continue;
            }
            headers.put(lines[i].substring(0, colon).trim().toLowerCase(Locale.ROOT),
                    lines[i].substring(colon + 1).trim());
        }
        String upgrade = lower(headers.get("upgrade"));
        String connection = lower(headers.get("connection"));
        String version = headers.get("sec-websocket-version");
        String key = headers.get("sec-websocket-key");
        if (!"websocket".equals(upgrade) || connection == null || !connection.contains("upgrade")) {
            throw new HandshakeException("missing Upgrade/Connection headers");
        }
        if (!"13".equals(version)) {
            throw new HandshakeException("unsupported Sec-WebSocket-Version: " + version);
        }
        byte[] decodedKey;
        try {
            decodedKey = Base64.getDecoder().decode(key == null ? "" : key);
        } catch (IllegalArgumentException e) {
            throw new HandshakeException("invalid Sec-WebSocket-Key");
        }
        if (decodedKey.length != 16) {
            throw new HandshakeException("Sec-WebSocket-Key must decode to 16 bytes");
        }

        String accept = computeAccept(key);
        String response = "HTTP/1.1 101 Switching Protocols\r\n"
                + "Upgrade: websocket\r\n"
                + "Connection: Upgrade\r\n"
                + "Sec-WebSocket-Accept: " + accept + "\r\n"
                + "\r\n";
        out.write(response.getBytes("US-ASCII"));
        out.flush();
        return new WebSocketConnection(socket, in, out);
    }

    private static String computeAccept(String key) throws IOException {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-1");
            byte[] hash = digest.digest((key + WS_GUID).getBytes("US-ASCII"));
            return Base64.getEncoder().encodeToString(hash);
        } catch (NoSuchAlgorithmException e) {
            throw new IOException("SHA-1 unavailable", e);
        }
    }

    private static String lower(String value) {
        return value == null ? "" : value.toLowerCase(Locale.ROOT);
    }

    private static byte[] readHeader(InputStream in) throws IOException {
        ByteArrayOutputStream buffer = new ByteArrayOutputStream();
        while (true) {
            int c = in.read();
            if (c < 0) {
                throw new EOFException("connection closed during handshake");
            }
            buffer.write(c);
            byte[] bytes = buffer.toByteArray();
            int n = bytes.length;
            if (n >= 4 && bytes[n - 4] == '\r' && bytes[n - 3] == '\n'
                    && bytes[n - 2] == '\r' && bytes[n - 1] == '\n') {
                return bytes;
            }
            if (n > MAX_HEADER_BYTES) {
                throw new HandshakeException("request header too large");
            }
        }
    }

    /**
     * 阻塞读取一条完整文本消息。控制帧在内部处理（PING 自动回 PONG）。
     * 返回 null 表示收到 CLOSE 或对端断开。
     */
    String readText() throws IOException {
        ByteArrayOutputStream message = null;
        while (true) {
            int b0 = in.read();
            if (b0 < 0) {
                return null;
            }
            int b1 = in.read();
            if (b1 < 0) {
                return null;
            }
            boolean fin = (b0 & 0x80) != 0;
            int opcode = b0 & 0x0F;
            boolean masked = (b1 & 0x80) != 0;
            long length = readPayloadLength(b1 & 0x7F);
            if (length > MAX_MESSAGE_BYTES) {
                throw new IOException("WebSocket frame too large: " + length);
            }
            if (!masked) {
                throw new IOException("client frames must be masked (RFC 6455 §5.3)");
            }
            byte[] mask = readFully(4);
            byte[] payload = readFully((int) length);
            for (int i = 0; i < payload.length; i++) {
                payload[i] = (byte) (payload[i] ^ mask[i & 3]);
            }

            switch (opcode) {
                case 0x1: // TEXT
                    if (message != null) {
                        throw new IOException("unexpected new text frame during fragmentation");
                    }
                    if (fin) {
                        return new String(payload, "UTF-8");
                    }
                    message = new ByteArrayOutputStream();
                    message.write(payload, 0, payload.length);
                    break;
                case 0x0: // CONTINUATION
                    if (message == null) {
                        throw new IOException("continuation without start frame");
                    }
                    if (message.size() + payload.length > MAX_MESSAGE_BYTES) {
                        throw new IOException("fragmented message too large");
                    }
                    message.write(payload, 0, payload.length);
                    if (fin) {
                        return new String(message.toByteArray(), "UTF-8");
                    }
                    break;
                case 0x8: // CLOSE
                    return null;
                case 0x9: // PING
                    sendControl(0xA, payload);
                    break;
                case 0xA: // PONG
                    break;
                default:
                    throw new IOException("unsupported WebSocket opcode: " + opcode);
            }
        }
    }

    private long readPayloadLength(int first) throws IOException {
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
        if (value < 0 || value > MAX_MESSAGE_BYTES) {
            throw new IOException("invalid 64-bit frame length");
        }
        return value;
    }

    private byte[] readFully(int length) throws IOException {
        byte[] data = new byte[length];
        int offset = 0;
        while (offset < length) {
            int read = in.read(data, offset, length - offset);
            if (read < 0) {
                throw new EOFException("connection closed mid-frame");
            }
            offset += read;
        }
        return data;
    }

    /** 发送一条 UTF-8 文本帧（服务端帧不掩码）。 */
    void sendText(String text) throws IOException {
        if (text == null) {
            return;
        }
        sendFrame(0x1, text.getBytes("UTF-8"));
    }

    private void sendControl(int opcode, byte[] payload) throws IOException {
        sendFrame(opcode, payload == null ? EMPTY : payload);
    }

    private void sendFrame(int opcode, byte[] payload) throws IOException {
        ByteArrayOutputStream frame = new ByteArrayOutputStream(payload.length + 16);
        frame.write(0x80 | opcode);
        int length = payload.length;
        if (length < 126) {
            frame.write(length);
        } else if (length < 65536) {
            frame.write(126);
            frame.write((length >>> 8) & 0xFF);
            frame.write(length & 0xFF);
        } else {
            frame.write(127);
            for (int shift = 56; shift >= 0; shift -= 8) {
                frame.write((int) ((length >>> shift) & 0xFF));
            }
        }
        frame.write(payload, 0, payload.length);
        synchronized (writeLock) {
            out.write(frame.toByteArray());
            out.flush();
        }
    }

    boolean isLoopback() {
        return socket.getInetAddress() != null && socket.getInetAddress().isLoopbackAddress();
    }

    boolean isClosed() {
        return closed;
    }

    /** 先发 CLOSE 帧，再关闭底层 socket；可安全重复调用。 */
    void close() {
        if (closed) {
            return;
        }
        closed = true;
        try {
            sendFrame(0x8, EMPTY);
        } catch (IOException ignored) {
            // 对端已断开时仍要完成本地关闭
        }
        try {
            socket.close();
        } catch (IOException ignored) {
            // ignore
        }
    }
}
