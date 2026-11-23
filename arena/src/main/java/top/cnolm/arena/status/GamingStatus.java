package top.cnolm.arena.status;

public interface GamingStatus extends Status {
    GamingStatus eliminate();
    GamingStatus fighting();
    GamingStatus countdown();
    @Override
    GamingStatus defaultStatus();
}
