package top.cnolm.arena.status;

public interface Status {
    /** 状态名称 */
    String getName();

    /** 从名称获取状态 */
    Status fromName(String name);

    /** 默认状态 **/
    Status defaultStatus();

    /** 状态描述 */
    default String description() {
        return null;
    }
}
