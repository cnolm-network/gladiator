package top.cnolm.gladiator.core;

import top.cnolm.gladiator.api.GameSession;
import top.cnolm.gladiator.api.TrainingSession;
import top.cnolm.gladiator.api.User;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Map;
import java.util.Objects;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

/**
 * 对局 / 训练场会话登记表（G-105）。
 *
 * <p><strong>只做登记与查询，不实现任何对局规则</strong>：不传送、不发套件、不判断胜负。
 * 传送、套件、地图、匹配与结算全部由调用方负责；这里只维护
 * 「谁在哪个会话里」「会话何时开始」。
 */
public final class GameRegistry {

    private final Map<String, GameSession> games = new ConcurrentHashMap<String, GameSession>();
    private final Map<String, TrainingSession> training =
            new ConcurrentHashMap<String, TrainingSession>();
    private final Map<UUID, String> sessionByUser = new ConcurrentHashMap<UUID, String>();
    private final Map<String, Long> startedAt = new ConcurrentHashMap<String, Long>();

    public boolean hasSessionFor(UUID uuid) {
        return uuid != null && sessionByUser.containsKey(uuid);
    }

    public String sessionIdOf(UUID uuid) {
        return uuid == null ? null : sessionByUser.get(uuid);
    }

    public Long startedAt(String sessionId) {
        return sessionId == null ? null : startedAt.get(sessionId);
    }

    public GameSession registerGame(GameSession session) {
        Objects.requireNonNull(session, "session");
        games.put(session.getSessionId(), session);
        startedAt.put(session.getSessionId(), Long.valueOf(session.getStartedAt()));
        for (User user : session.getUsers()) {
            sessionByUser.put(user.getUniqueId(), session.getSessionId());
        }
        return session;
    }

    public TrainingSession registerTraining(TrainingSession session) {
        Objects.requireNonNull(session, "session");
        training.put(session.getSessionId(), session);
        startedAt.put(session.getSessionId(), Long.valueOf(session.getStartedAt()));
        sessionByUser.put(session.getBot().getUniqueId(), session.getSessionId());
        return session;
    }

    /**
     * 移除会话并清理参与者索引；会话不存在返回 {@code false}。
     * 不派发事件——事件由编排层统一发出。
     */
    public boolean remove(String sessionId) {
        if (sessionId == null) {
            return false;
        }
        GameSession game = games.remove(sessionId);
        TrainingSession place = game == null ? training.remove(sessionId) : null;
        if (game == null && place == null) {
            return false;
        }
        startedAt.remove(sessionId);
        if (game != null) {
            for (User user : game.getUsers()) {
                sessionByUser.remove(user.getUniqueId(), sessionId);
            }
        } else {
            sessionByUser.remove(place.getBot().getUniqueId(), sessionId);
        }
        return true;
    }

    public Collection<GameSession> games() {
        return new ArrayList<GameSession>(games.values());
    }

    public Collection<TrainingSession> trainingSessions() {
        return new ArrayList<TrainingSession>(training.values());
    }

    public boolean isGame(String sessionId) {
        return sessionId != null && games.containsKey(sessionId);
    }

    public boolean isTraining(String sessionId) {
        return sessionId != null && training.containsKey(sessionId);
    }

    public int gameCount() {
        return games.size();
    }

    public int trainingCount() {
        return training.size();
    }

    public void clear() {
        games.clear();
        training.clear();
        sessionByUser.clear();
        startedAt.clear();
    }
}
