package top.cnolm.arena.status;

public enum MatchStatusImpl implements MatchStatus {
    PREPARING,
    COUNTDOWN,
    GAMING,
    END,
    CLOSED
    ;

    @Override
    public String getName() {
        return toString();
    }

    @Override
    public Status fromName(String name) {
        return valueOf(name);
    }

    @Override
    public Status defaultStatus() {
        return MatchStatusImpl.PREPARING;
    }
}
