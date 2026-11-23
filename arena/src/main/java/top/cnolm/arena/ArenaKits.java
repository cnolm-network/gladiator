package top.cnolm.arena;

import org.bukkit.Material;
import org.bukkit.entity.Player;
import org.bukkit.inventory.ItemStack;
import org.bukkit.inventory.PlayerInventory;
import org.bukkit.potion.PotionEffect;
import org.bukkit.potion.PotionEffectType;

/**
 * 套件与免伤窗口：平地对战里「发什么」和「什么时候开始算伤害」的规则都在这里。
 *
 * <p>为什么独立成类：{@link FlatMatchManager} 管流程，套件是数据；以后加模式只在这里加一行，
 * 不去动回合引擎。{@code kitId} 来自 {@code RoundSpec#getKitId()}，未知值回退 {@link #CLASSIC}
 * 并打一条告警——静默回退会让「我明明选了 nodebuff」变成难查的错。
 *
 * <p>免伤用抗性 V（{@code amplifier = 4}）：1.8 里这是完全免伤，正好覆盖倒计时那 5 秒，
 * 让玩家有时间穿装备，而不是一出生就被砍。
 */
final class ArenaKits {

    /** 默认套件：钻石甲 + 钻剑 + 鱼竿 + 5 金苹果 + 16 熟牛肉。 */
    static final String CLASSIC = "classic";
    /** 只有甲和剑，用来练点击/对刀。 */
    static final String SWORD = "sword";
    /** 多发金苹果，用来练消耗与蹭药节奏。 */
    static final String GAPPLE = "gapple";
    /** 什么都不发（只清空背包），用于「空手」规则或调试。 */
    static final String EMPTY = "empty";

    /** 抗性 V 的 amplifier；1.8 的等级 = amplifier + 1。 */
    private static final int SHIELD_AMPLIFIER = 4;

    private ArenaKits() {
    }

    /**
     * 按 kitId 发套件（会先清空背包与护甲，调用方不需要自己清）。
     *
     * @return 实际使用的 kitId（未知值回退成 {@link #CLASSIC}）
     */
    static String give(Player target, String kitId) {
        String kit = normalize(kitId);
        PlayerInventory inv = target.getInventory();
        inv.clear();
        inv.setArmorContents(new ItemStack[]{null, null, null, null});
        switch (kit) {
            case EMPTY:
                break;
            case SWORD:
                armor(inv);
                inv.setItemInHand(new ItemStack(Material.DIAMOND_SWORD));
                break;
            case GAPPLE:
                armor(inv);
                inv.setItemInHand(new ItemStack(Material.DIAMOND_SWORD));
                inv.setItem(1, new ItemStack(Material.FISHING_ROD));
                inv.setItem(2, new ItemStack(Material.GOLDEN_APPLE, 16));
                inv.setItem(3, new ItemStack(Material.COOKED_BEEF, 16));
                break;
            case CLASSIC:
            default:
                armor(inv);
                inv.setItemInHand(new ItemStack(Material.DIAMOND_SWORD));
                inv.setItem(1, new ItemStack(Material.FISHING_ROD));
                inv.setItem(2, new ItemStack(Material.GOLDEN_APPLE, 5));
                inv.setItem(3, new ItemStack(Material.COOKED_BEEF, 16));
                break;
        }
        target.updateInventory();
        return kit;
    }

    /** 未知/空 kitId 一律回退 classic（并让调用方去记日志）。 */
    static String normalize(String kitId) {
        if (kitId == null) {
            return CLASSIC;
        }
        String value = kitId.trim().toLowerCase(java.util.Locale.ROOT);
        if (value.isEmpty() || "default".equals(value) || "diamond".equals(value)) {
            return CLASSIC;
        }
        if (EMPTY.equals(value) || SWORD.equals(value) || GAPPLE.equals(value) || CLASSIC.equals(value)) {
            return value;
        }
        return null;
    }

    /** 倒计时免伤：抗性 V，覆盖倒计时 + 1 秒（多给一秒是为了容忍客户端延迟）。 */
    static void applyShield(Player target, int seconds) {
        if (seconds <= 0) {
            return;
        }
        target.addPotionEffect(new PotionEffect(
                PotionEffectType.DAMAGE_RESISTANCE, (seconds + 1) * 20, SHIELD_AMPLIFIER, true, false));
    }

    /** 开打瞬间解除免伤；没有这个效果时是空操作。 */
    static void clearShield(Player target) {
        target.removePotionEffect(PotionEffectType.DAMAGE_RESISTANCE);
    }

    private static void armor(PlayerInventory inv) {
        inv.setArmorContents(new ItemStack[]{
            new ItemStack(Material.DIAMOND_BOOTS),
            new ItemStack(Material.DIAMOND_LEGGINGS),
            new ItemStack(Material.DIAMOND_CHESTPLATE),
            new ItemStack(Material.DIAMOND_HELMET)
        });
    }
}
