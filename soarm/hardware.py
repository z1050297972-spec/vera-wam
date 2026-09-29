"""硬件层操作：自检、深度探测、状态快照、力矩安全预处理。

`check_arm` 是本项目**最重要的诊断工具**：它用原始 Feetech 协议逐个 ping
舵机并读错误位。为什么不能只信 lerobot 的报错 —— lerobot 的
`broadcast_ping()` 会**丢弃带错误状态的舵机**，所以"能通信但过压"在它眼里
长得像"找不到舵机"，报错方向完全错。原始协议不做这个过滤。
"""

from __future__ import annotations

import time

from .bus import Arm
from .constants import (
    ARM_INFO,
    ERR_BITS,
    MOTORS,
)
from .ports import port_of, role_serial


# ---------------------------------------------------------------- 硬件自检
def check_arm(role: str) -> str:
    """读电压/温度/错误位。返回 'OK' / 'DEAD' / 'VOLTAGE_ERR'。"""
    info = ARM_INFO[role]
    print(f"检查 {info['cn']}   (序列号 {role_serial(role)})")
    print(f"  该臂规格: 舵机 {info['servo_v']}，供电 {info['supply']}")
    with Arm(role) as a:
        print(f"  端口: {a.port}")
        found = a.ping_all()
        if not found:
            print("  ❌ 无响应：一个舵机都找不到")
            print(f"     -> 舵机没上电，或总线线缆/接头断开。该臂应用 {info['supply']} 供电。")
            return "DEAD"
        print(f"  ✓ 找到 {len(found)} 个舵机")

        bad = []
        for name, mid in MOTORS.items():
            v = a.rd(mid, "Present_Voltage")
            lo = a.rd(mid, "Min_Voltage_Limit")
            hi = a.rd(mid, "Max_Voltage_Limit")
            st = a.rd(mid, "Status")
            tp = a.rd(mid, "Present_Temperature")
            if v is None:
                print(f"    {name:<15} 读取失败")
                continue
            flag = ""
            if st:
                bits = [ERR_BITS.get(b, f"bit{b}") for b in range(6) if st & (1 << b)]
                flag = f"  ★错误位 0x{st:02x} ({'/'.join(bits)})"
                bad.append(name)
            if hi is not None and v > hi:
                flag += f"  过压! 超上限 {hi/10}V"
            elif lo is not None and v < lo:
                flag += f"  欠压! 低于下限 {lo/10}V"
            print(f"    {name:<15} {v/10:>5.1f}V  限值[{lo/10},{hi/10}]V  {tp}°C{flag}")

        if bad:
            print(f"\n  ⚠️  报错舵机: {bad}")
            print(f"     -> 电压不对。该臂舵机是 {info['servo_v']} 规格，应用 {info['supply']} 供电。")
            return "VOLTAGE_ERR"
        print("  ✅ 全部正常，无错误状态")
        return "OK"


def check_all() -> dict:
    """检查两条臂，返回 {角色: 结果}。"""
    results = {}
    for role in ("leader", "follower"):
        print("=" * 66)
        results[role] = check_arm(role)
        print()
    print("结论: " + "   ".join(f"{r}={v}" for r, v in results.items()))
    return results


# ---------------------------------------------------------------- 深度探测
def deep_probe(role: str) -> int:
    """原始协议探测：两种协议 × 全部可用波特率。只读。

    比 check_arm 更底层：能看到错误位（err=0x01 就是电压错误），也能确认
    是不是波特率不对。14400/128000/250000 会被 CH343 适配器拒绝，正常。
    """
    from scservo_sdk import PacketHandler, PortHandler

    bauds = [1_000_000, 500_000, 250_000, 128_000, 115_200, 57_600,
             38_400, 19_200, 14_400, 9_600, 4_800]
    port = port_of(role)
    print(f"深度探测 {ARM_INFO[role]['cn']}   {port}\n")
    h = PortHandler(port)
    if not h.openPort():
        print("  ❌ 打不开端口（权限不足？重新登录，或用 sg dialout 包一层）")
        return 4
    hits = 0
    try:
        for baud in bauds:
            try:
                h.setBaudRate(baud)
            except Exception:  # noqa: BLE001
                print(f"  {baud:>9}  -    适配器不支持该波特率（跳过）")
                continue
            time.sleep(0.05)
            row = []
            for proto in (0, 1):
                ph = PacketHandler(proto)
                found = {}
                for mid in range(1, 7):
                    model, comm, err = ph.ping(h, mid)
                    if comm == 0 and model:
                        found[mid] = (model, err)
                if found:
                    hits += 1
                    detail = ", ".join(
                        f"ID{k}={v[0]}" + (f"(err=0x{v[1]:02x})" if v[1] else "")
                        for k, v in sorted(found.items())
                    )
                    row.append(f"协议{proto} ★ {detail}")
                else:
                    row.append(f"协议{proto} 无")
            print(f"  {baud:>9}  " + "   ".join(row))
    finally:
        h.closePort()
    print()
    if hits:
        print(f"  ✅ 有 {hits} 个组合有响应")
        print("     若带 err=0x01，是电压错误位 —— 供电电压不符合该臂规格。")
        return 0
    print("  ❌ 两种协议 × 全部可用波特率 均无响应")
    print("     => 舵机没上电，或总线线缆/接头断开。")
    return 1


# ---------------------------------------------------------------- 状态快照
def read_state(role: str) -> dict:
    """力矩 / Lock / Goal / 当前位置快照，并判断"开力矩会不会猛冲"。

    `Goal_Position` 在 SRAM 里（上电为 0），手动移动机械臂**不会更新它**，
    所以掰过臂之后 Goal 就是个过时的旧姿态 —— 任何开启力矩的操作
    都会让臂朝它冲过去。实测见过 1938 刻度（约 170°）的差值。
    """
    with Arm(role) as a:
        print(f"{ARM_INFO[role]['cn']} 状态快照\n")
        print(f"  {'ID':>3} {'关节':<15} {'力矩':>5} {'Lock':>6} {'Goal':>7} "
              f"{'当前位置':>8}  {'差值':>6}  说明")
        print("  " + "-" * 80)
        rows, risky = {}, []
        for name, mid in MOTORS.items():
            tq = a.rd(mid, "Torque_Enable")
            lk = a.rd(mid, "Lock")
            gp = a.rd(mid, "Goal_Position", 2)
            pp = a.rd(mid, "Present_Position", 2)
            tq_s = {0: "关", 1: "开", 2: "松弛"}.get(tq, str(tq))
            lk_s = {0: "解锁", 1: "锁定"}.get(lk, str(lk))
            note, diff = "", None
            if isinstance(gp, int) and isinstance(pp, int):
                diff = abs(gp - pp)
                if tq == 1 and diff > 100:
                    note = "★力矩已开且差值大，会转动"
                elif tq == 0 and diff > 100:
                    note = "⚠️ 开力矩后会朝 Goal 猛冲"
            if isinstance(lk, int) and lk == 1:
                note += "  Lock=1 会挡住 EEPROM 写入"
            print(f"  {mid:>3} {name:<15} {tq_s:>5} {lk_s:>6} {str(gp):>7} "
                  f"{str(pp):>8}  {str(diff) if diff is not None else '?':>6}  {note}")
            rows[name] = {"torque": tq, "lock": lk, "goal": gp,
                          "present": pp, "diff": diff}
            if diff is not None and diff > 100:
                risky.append(name)
        print()
        if risky:
            print(f"  ⚠️  {len(risky)} 个关节 Goal 与当前位置相差较大: {risky}")
            print(f"     开力矩前先跑:  python -m soarm prep {role}")
        else:
            print("  ✅ Goal 与当前位置基本一致，开力矩不会明显移动。")
        return rows


# ---------------------------------------------------------------- 力矩安全预处理
def prep_goal_safe(role: str) -> bool:
    """把 Goal_Position 设成当前位置，然后关掉力矩。

    `Goal_Position` 在 SRAM，上电为 0；STS3215 在力矩 0->1 时会朝它驱动，
    所以不预设会让机械臂朝刻度 0 猛冲。

    ⚠️ **顺序关键**：写 `Goal_Position` 会让固件**自动开启力矩**（实测 6/6），
    所以必须在写完 Goal 之后【再关一次】力矩，否则臂会变硬、手掰不动。
    """
    with Arm(role) as a:
        a.disable_torque()
        pos = a.positions()
        if any(v is None for v in pos.values()):
            print(f"  ✗ 有舵机读不到位置: {[k for k, v in pos.items() if v is None]}")
            return False
        for name, mid in MOTORS.items():
            a.wr(mid, "Goal_Position", pos[name], length=2)

        goal = a.goals()
        for name in MOTORS:
            ok = goal.get(name) == pos.get(name)
            print(f"  {name:<15} 位置={pos[name]:>5}  Goal={goal.get(name):>5}  "
                  f"{'✓' if ok else '✗'}")
        if any(goal.get(n) != pos.get(n) for n in MOTORS):
            print("\n  ⚠️  有舵机未对齐")
            return False

        a.disable_torque()   # 写 Goal 会打开力矩，这里补关（顺序不能颠倒）
        time.sleep(0.2)
        if not a.is_limp():
            print("\n  ⚠️  力矩未完全关闭:")
            for n, v in a.torques().items():
                if v != 0:
                    print(f"       {n} Torque_Enable={v}")
            return False
        print("\n  ✅ Goal 已对齐当前位置；力矩已关闭，可以用手掰")
        return True
