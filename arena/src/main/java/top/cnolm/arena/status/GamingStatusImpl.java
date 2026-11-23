package top.cnolm.arena.status;

/**
 * 游戏状态实现 </br>
 * 如果需要自定义状态，请继承 {@link GamingStatus}
 */
public enum GamingStatusImpl implements GamingStatus {
    PREPARING {
        @Override
        public String description() {
            return "正在准备中";
        }
    },
    COUNTDOWN {
        @Override
        public String description() {
            return "倒计时中";
        }
    },
    FIGHTING {
        @Override
        public String description() {
            return "正在对局";
        }
    },
    DEAD {
        @Override
        public String description() {
            return "已死亡";
        }
    }


    ;

    @Override
    public GamingStatus defaultStatus() {
        return PREPARING;
    }

    @Override
    public GamingStatus eliminate() {
        return DEAD;
    }

    @Override
    public GamingStatus fighting() {
        return FIGHTING;
    }

    @Override
    public GamingStatus countdown() {
        return COUNTDOWN;
    }

    @Override
    public String getName() {
        return toString();
    }

    @Override
    public GamingStatus fromName(String name) {
        return valueOf(name);
    }
}
