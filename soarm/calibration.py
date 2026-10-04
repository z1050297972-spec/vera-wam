"""标定：两趟法 + 验证。

为什么不是一趟（lerobot 原版流程）—— 原版把"摆到行程中间"实现成舵机自己的
半圈刻度 2047，而 2047 是舵机转整圈的一半，关节的机械行程只是整圈的一小段。
对行程偏在一侧的关节，2047 可能离**机械中点**很远。实测后果是主从零点完全
不对应（从臂夹爪 32%、主臂 0%）。

两趟法的做法与证明：
  第一趟 measure_travel         —— 手推测出每个关节的真实行程 [min,max]
  第二趟 show_angles_to_midpoints —— 按各自实测中点摆位
  第三趟 finalize_two_pass      —— 算出并写入

若摆位在实测中点 (min+max)/2，则偏移 o = 中点 − 2047，量程变成
[min−o, max−o]，其中点恰好是 2047 —— 所以写入后 `2047位置` 全是 50%。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from .bus import Arm
from .constants import (
    ARM_INFO,
    FULL_TURN,
    MID,
    MOTORS,
    MOVABLE_ALL,
    RES,
    calibration_path,
    state_file,
)
from .encoding import (
    decode_sign_magnitude,
    encode_sign_magnitude,
    shortest_arc,
    ticks_from,
    ticks_to_deg,
)
from .ui import REFRESH_INTERVAL_S, KeyWatcher, Screen, range_bar

# 一次力矩关闭指令的有效期。摆位/推行程时臂必须保持松，每 2 秒重发一次，
# 防止别的进程（或任何写 Goal 的操作）把力矩又打开。
LIMP_REFRESH_S = 2.0

# 行程小于这个角度就认为无效（没推到位或机械卡住）
MIN_TRAVEL_DEG = 20.0

# 行程大于这个角度就可疑：SO-ARM 单个关节转不了这么多，
# 出现这种值说明读数被错误包污染了（半双工总线偶发损坏状态包）。
IMPLAUSIBLE_TRAVEL_DEG = 300.0

# 取连续 N 次采样的中位数作为可信值，抑制偶发错误包。
# 单次跳变会被中位数滤掉；真实的极值因为会被推到位顶住，不受影响。
TRUST_WINDOW = 3

# 行程文件的格式版本。旧格式（raw min/max）不识别编码器回绕，会算出接近
# 360° 的假行程，所以必须区分并拒绝。
TRAVEL_FORMAT = 2


# ---------------------------------------------------------------- 行程数据工具
def travel_mid_raw(t: dict) -> float:
    """行程中点在原始刻度空间的读数（可能跨过 0 点，所以取模）。"""
    return (t["ref"] + t["mid_u"]) % RES


def travel_width_ticks(t: dict) -> float:
    """行程宽度（刻度）。展开空间的极差，与起点无关，不受回绕影响。"""
    return t["umax"] - t["umin"]


def travel_delta_from_mid(t: dict, raw: float) -> float:
    """当前读数相对行程中点的带符号偏移（刻度），按最短弧算。"""
    return shortest_arc(raw - travel_mid_raw(t))


def travel_wraps(t: dict) -> bool:
    """行程是否跨过编码器 0 点。用于在界面上提示，不影响计算。"""
    lo = (t["ref"] + t["umin"]) % RES
    hi = (t["ref"] + t["umax"]) % RES
    return lo > hi


def travel_from_unrolled(ref: float, umin: float, umax: float) -> dict:
    """由展开空间的极值构造行程记录。"""
    return {
        "ref": ref,                       # 测量起点的原始读数
        "umin": umin,                     # 相对起点的累计位移下限（展开空间）
        "umax": umax,
        "mid_u": (umin + umax) / 2,
        "width_ticks": umax - umin,
        "width_deg": ticks_to_deg(umax - umin),
    }


# ================================================================ 第一趟
def measure_travel(role: str) -> dict:
    """用手把每个关节推到两端，测出真实行程。**不做任何写入。**

    把每个关节从一端推到另一端推满，然后按 `x` 结束（Ctrl-C 也可以，同样
    保留已测数据）。没有时间限制，也没有自动判定 —— 推多久由你决定。

    结果存到 `~/.cache/soarm/travel_<role>.json`，第二趟读它。
    """
    movable = [m for m in MOTORS if m != FULL_TURN]
    print("用手把每个关节【从一端推到另一端】，走满真实行程。")
    print("推满之后按 x 结束（Ctrl-C 也可以）。")
    print(f"注意：{FULL_TURN}（手腕旋转）是整圈连续旋转关节，**没有行程端点**，")
    print(f"      所以不在下面的表里，也不需要推 —— 它的量程固定为 0–4095，")
    print(f"      零位在第二趟和其余关节一起摆。")
    print()
    print("说明：下表的角度都相对【行程】算，不是舵机整圈角度 ——")
    print("      「已测宽度」是关节真正走过的角度，「距下限」是当前位置离下限多远。")
    print()

    with Arm(role) as a, KeyWatcher() as keys:
        a.disable_torque()
        last_limp = time.time()
        pos = a.positions()
        start = {m: pos[m] for m in movable if pos[m] is not None}
        # 展开空间：按最短弧累积位移，这样行程跨过编码器 0 点也能算对宽度。
        # 旧做法是记 raw 的 min/max，一旦行程跨 0 点，max-min 会接近 4096，
        # 显示成 ~360° 的假行程。
        unrolled = dict.fromkeys(start, 0.0)
        prev = dict(start)
        umin = dict.fromkeys(start, 0.0)
        umax = dict.fromkeys(start, 0.0)
        history: dict[str, list] = {m: [start[m]] for m in start}
        scr = Screen()
        finished_by_key = False
        try:
            while True:
                if time.time() - last_limp > LIMP_REFRESH_S:
                    a.disable_torque()
                    last_limp = time.time()

                key = keys.poll()
                if key and key.lower() == "x":
                    finished_by_key = True

                pos = a.positions()
                trusted = {}
                for m in movable:
                    v = pos.get(m)
                    if v is None:
                        continue
                    h = history.setdefault(m, [])
                    h.append(v)
                    del h[:-TRUST_WINDOW]
                    t = sorted(h)[len(h) // 2]      # 中位数，滤掉单次错误包
                    trusted[m] = t
                    unrolled[m] += shortest_arc(t - prev[m])   # 走最短弧
                    prev[m] = t
                    umin[m] = min(umin[m], unrolled[m])
                    umax[m] = max(umax[m], unrolled[m])

                lines = [
                    f"{ARM_INFO[role]['cn']} — 测量真实行程",
                    "",
                    f"  {'关节':<15} {'行程宽度':>9} {'距下限':>8}  位置条（o 当前）"
                    f"            {'当前(raw)':>9}  备注",
                    "  " + "-" * 84,
                ]
                for m in movable:
                    if m not in trusted:
                        lines.append(f"  {m:<15}   读取失败")
                        continue
                    t = travel_from_unrolled(start[m], umin[m], umax[m])
                    note = "跨过 0 点" if travel_wraps(t) else ""
                    w = umax[m] - umin[m]
                    bar = range_bar(unrolled[m] - umin[m], 0, w)
                    lines.append(
                        f"  {m:<15} {t['width_deg']:>8.1f}° "
                        f"{ticks_from(unrolled[m], umin[m]):>7.1f}°  [{bar}]  "
                        f"{trusted[m]:>9}  {note}"
                    )
                lines.append("")
                lines.append("  >>> 把每个关节推到两端；推满后按 x 结束")
                lines.append("  （角度和位置条都相对行程算，不是舵机整圈角度）")
                scr.draw(lines)

                if finished_by_key:
                    print("\n  已按 x 结束。")
                    break
                time.sleep(REFRESH_INTERVAL_S)
        except KeyboardInterrupt:
            print("\n\n  ■ 已中断，用当前已测得的结果。")
        finally:
            scr.close()

    travel = {m: travel_from_unrolled(start[m], umin[m], umax[m]) for m in start}
    print("  测得的真实行程:")
    print(f"    {'关节':<15} {'起始(raw)':>10} {'行程宽度':>10} {'中点(raw)':>10}  备注")
    print("    " + "-" * 62)
    for m in movable:
        t = travel[m]
        note = "行程跨过编码器 0 点（已按最短弧正确计算）" if travel_wraps(t) else ""
        print(f"    {m:<15} {t['ref']:>10} {t['width_deg']:>9.1f}° "
              f"{travel_mid_raw(t):>10.0f}  {note}")

    narrow = [m for m in movable if travel[m]["width_deg"] < MIN_TRAVEL_DEG]
    if narrow:
        print(f"\n  ⚠️  这些关节行程不到 {MIN_TRAVEL_DEG:.0f}°，可疑（没推到位或机械卡住）: {narrow}")

    too_wide = [m for m in movable if travel[m]["width_deg"] > IMPLAUSIBLE_TRAVEL_DEG]
    if too_wide:
        print(f"\n  ❌ 这些关节测出的行程超过 {IMPLAUSIBLE_TRAVEL_DEG:.0f}°，"
              f"单个关节转不了这么多: {too_wide}")
        print("     多半是读数被错误包污染了（总线偶发损坏状态包），"
              "或者有另一个进程同时在读同一条总线。")
        print("     请确认没有别的程序占用串口，然后重新测量。**不要用这份数据往下做标定。**")

    out = state_file(role)
    out.parent.mkdir(parents=True, exist_ok=True)
    # 覆盖前先备份：否则一次跑空（比如刚启动就退出）就会把上次好的测量
    # 数据冲掉，而且无声无息。
    if out.exists() and out.stat().st_size > 0:
        bak = out.with_name(out.name + ".bak")
        bak.write_bytes(out.read_bytes())
        print(f"  上一次的结果已备份到 {bak.name}")
    out.write_text(json.dumps({"_format": TRAVEL_FORMAT, **travel}, indent=2))
    print(f"\n  已保存到 {state_file(role)}（第二趟会读它）")
    return travel


def load_travel(role: str) -> dict:
    """读第一趟的结果并校验可用性，不合格就退出并说明原因。

    这里必须挡住：行程宽度为 0 时"任何位置都算中点"，`mid` 会误报成功，
    一路把无效数据带进标定。
    """
    f = state_file(role)
    if not f.exists():
        sys.exit(f"还没测行程。先运行:  python -m soarm measure {role}")
    data = json.loads(f.read_text())
    if data.get("_format") != TRAVEL_FORMAT:
        sys.exit(
            f"{f} 是旧格式的行程数据，不能直接用。\n"
            f"  旧格式按 raw min/max 记录，没有处理编码器回绕 —— 行程跨过 0 点的\n"
            f"  关节会被算成接近 360° 的假行程。\n"
            f"  请重新测量:  python -m soarm measure {role}"
        )
    travel = {k: v for k, v in data.items() if k != "_format"}

    narrow = [m for m, t in travel.items()
              if ticks_to_deg(travel_width_ticks(t)) < MIN_TRAVEL_DEG]
    if narrow:
        print("❌ 行程数据不可用，这些关节的行程太小:")
        for m in narrow:
            print(f"     {m:<15} {ticks_to_deg(travel_width_ticks(travel[m])):.1f}°")
        sys.exit(
            f"\n  常见原因：measure 时没把关节推到两端，或跑到一半就退出了。\n"
            f"  请重新测量:  python -m soarm measure {role}\n"
            f"  （每个关节都要从一端推到另一端，走满真实行程）"
        )
    return travel


# ================================================================ 第二趟
def show_angles_to_midpoints(role: str, travel: dict, tolerance_deg: float = 8.0) -> bool:
    """按每个关节【各自实测的中点】摆位。

    和 measure 一样：没有计时也没有时限，摆好后**按 `x` 结束**（Ctrl-C 也可以）。
    5 个行程关节都进入容差时会自动结束。

    `wrist_roll` 没有行程中点可对，所以它那一行只显示当前 raw 和将要写入的偏移
    —— 但它的零位同样要定，而且**主从必须摆成同一个物理方向**，否则遥操作时
    手腕旋转会一直偏。
    """
    print("把每个关节摆到它自己行程的中点；位置条的 | 是中点，o 是当前位置。")
    print(f"摆好后按 x 结束（Ctrl-C 也可以）；5 个行程关节都进入 ±{tolerance_deg:.0f}° 会自动结束。")
    print(f"⚠️ 主臂和从臂的 {FULL_TURN} 要摆成【同一个物理方向】（比如夹爪开口都竖直朝上）。")
    print()

    with Arm(role) as a, KeyWatcher() as keys:
        a.disable_torque()
        last_limp = time.time()
        scr = Screen()
        finished_by_key = False
        try:
            while True:
                if time.time() - last_limp > LIMP_REFRESH_S:
                    a.disable_torque()
                    last_limp = time.time()

                key = keys.poll()
                if key and key.lower() == "x":
                    finished_by_key = True

                pos = a.positions()
                lines = [
                    f"{ARM_INFO[role]['cn']} — 摆到【各自真实行程的中点】  容差 ±{tolerance_deg:.0f}°",
                    "",
                    f"  {'关节':<15} {'相对中点':>9} {'行程位置':>8}  位置条（| 中点  o 当前）  状态",
                    "  " + "-" * 78,
                ]
                all_ok = True
                for m in MOVABLE_ALL:
                    v = pos.get(m)
                    if v is None:
                        lines.append(f"  {m:<15}   读取失败")
                        all_ok = False
                        continue

                    if m == FULL_TURN:
                        off = int(round(v - MID))
                        note = f"无行程可对；raw {v}，将写入偏移 {off:+}"
                        if abs(off) > 2047:
                            note += "  ★超出 ±2047，转一下再记录"
                            all_ok = False
                        lines.append(f"  {m:<15} {'—':>9} {'—':>8}  {'':<25}  {note}")
                        continue

                    t = travel[m]
                    w = travel_width_ticks(t)
                    dev_ticks = travel_delta_from_mid(t, v)
                    dev = ticks_to_deg(dev_ticks)
                    pct = (dev_ticks + w / 2) / w * 100 if w else float("nan")
                    bar = range_bar(dev_ticks + w / 2, 0, w, mark=w / 2)

                    if abs(dev_ticks) > w / 2 + RES * 0.02:
                        state = "★已超出测量行程"
                        all_ok = False
                    elif abs(dev) <= tolerance_deg:
                        state = "OK"
                    else:
                        state = "偏离中点"
                        all_ok = False
                    lines.append(f"  {m:<15} {dev:>+8.1f}° {pct:>7.0f}%  [{bar}]  {state}")

                lines.append("")
                if all_ok:
                    lines.append("  ✅ 5 个行程关节都在各自中点附近，可以结束。")
                else:
                    lines.append("  >>> 用手调整；位置条的 | 是该关节自己行程的中点。")

                scr.draw(lines)

                if all_ok:
                    return True
                if finished_by_key:
                    print("\n  已按 x 结束（注意：并不是所有行程关节都到了中点）。")
                    return False
                time.sleep(REFRESH_INTERVAL_S)
        except KeyboardInterrupt:
            print("\n\n  ■ 已中断。")
            return False
        finally:
            scr.close()


# ================================================================ 第三趟
def compute_calibration(pos: dict, travel: dict) -> tuple[dict, dict, dict, list]:
    """由实测行程算出标定，返回 (offsets, mins, maxes, problems)。

    纯计算，不碰硬件 —— 便于单独测试和 dry-run。

    **零点取自实测的行程中点，不是当前位置**：

        mid_raw = 行程中点的原始读数（由 ref + mid_u 取模算得）
        o       = mid_raw - 2047          写进 Homing_Offset
        量程    = [2047 - W/2, 2047 + W/2]

    这样写完之后**行程中点**读数恰好是 2047，量程恒定居中 —— 位置只要落在
    测量到的行程内就行，不要求你摆得准。（早期版本把 `o` 定成 `pos - 2047`，
    相当于强制"当前位置 = 中点"；位置一偏量程就被推出 [0,4095]，导致
    "偏移后量程超出"的报错。）

    附带的好处：归一化零点是精确的行程中点，主从对应更好。

    所以 `pos` 在这里只用于**健全性检查**：如果当前位置跑出了实测行程，
    说明行程数据过时了（臂被动过、或换过装配），这时拒绝写入。

    `wrist_roll` 例外：整圈旋转没有行程可测，它的零点只能取自当前位置，
    所以主从必须摆成同一个物理方向。
    """
    offsets, mins, maxes, problems = {}, {}, {}, []
    movable = [m for m in MOTORS if m != FULL_TURN]

    for m in movable:
        t = travel[m]
        w = travel_width_ticks(t)
        w_deg = ticks_to_deg(w)
        if w_deg < MIN_TRAVEL_DEG:
            problems.append(
                f"{m}: 测得行程只有 {w_deg:.1f}°，太小 —— 第一趟没推到位，重跑 measure"
            )
            continue
        if w_deg > IMPLAUSIBLE_TRAVEL_DEG:
            problems.append(
                f"{m}: 测得行程 {w_deg:.1f}° 超过 {IMPLAUSIBLE_TRAVEL_DEG:.0f}°，"
                f"单个关节转不了这么多 —— 读数多半被错误包污染，重跑 measure"
            )
            continue

        # 健全性检查：当前位置应该在实测行程内
        delta = travel_delta_from_mid(t, pos[m])
        if abs(delta) > w / 2 + RES * 0.02:
            problems.append(
                f"{m}: 当前位置离行程中点 {ticks_to_deg(delta):+.1f}° "
                f"（行程半宽 {ticks_to_deg(w / 2):.1f}°），已跑出实测行程 —— "
                f"行程数据可能过时（臂被动过？），重跑 measure"
            )
            continue

        mid_raw = travel_mid_raw(t)
        o = int(round(mid_raw - MID))
        offsets[m] = o
        mins[m] = int(round(MID - w / 2))
        maxes[m] = int(round(MID + w / 2))

        if abs(o) > 2047:
            problems.append(f"{m}: homing 偏移 {o} 超出 ±2047（舵机会报 ValueError）")
        if mins[m] < 0 or maxes[m] > RES - 1:
            problems.append(
                f"{m}: 量程 [{mins[m]},{maxes[m]}] 超出 [0,{RES-1}]"
                f"（行程宽度 {ticks_to_deg(w):.1f}° 太大装不下）"
            )

    if pos.get(FULL_TURN) is not None:
        offsets[FULL_TURN] = int(round(pos[FULL_TURN] - MID))
    else:
        offsets[FULL_TURN] = 0
    mins[FULL_TURN], maxes[FULL_TURN] = 0, RES - 1
    return offsets, mins, maxes, problems


def calibration_dict(offsets: dict, mins: dict, maxes: dict) -> dict:
    """构造 lerobot 的 MotorCalibration JSON 结构。

    字段顺序和 draccus.dump 的输出逐字节一致，lerobot 能直接读。
    """
    return {
        m: {"id": mid, "drive_mode": 0, "homing_offset": offsets[m],
            "range_min": mins[m], "range_max": maxes[m]}
        for m, mid in MOTORS.items()
    }


def finalize_two_pass(role: str, arm_id: str, travel: dict, dry_run: bool = False):
    """根据实测行程 + 当前摆位算出最终标定，校验后写入 EEPROM 和 JSON。"""
    with Arm(role) as a:
        a.disable_torque()
        pos = a.positions()
        if any(pos.get(m) is None for m in MOTORS if m != FULL_TURN):
            print("  ✗ 有舵机读不到位置")
            return None

        offsets, mins, maxes, problems = compute_calibration(pos, travel)

        if problems:
            print("  ❌ 校验未通过，未写入:")
            for p in problems:
                print(f"     - {p}")
            return None

        print(f"  {'关节':<15} {'位置':>6} {'偏移':>7} {'最终量程':>16} "
              f"{'宽度':>8} {'2047位置':>9}")
        print("  " + "-" * 70)
        for m in MOVABLE_ALL:
            rng = maxes[m] - mins[m]
            pct = (MID - mins[m]) / rng * 100 if rng else float("nan")
            print(f"  {m:<15} {str(pos.get(m)):>6} {offsets[m]:>+7} "
                  f"[{mins[m]:>5},{maxes[m]:>5}] {ticks_to_deg(rng):>7.1f}° {pct:>8.0f}%")
        print()

        if not dry_run:
            for name, mid in MOTORS.items():          # mid 是 ID，别用关节名
                a.wr(mid, "Homing_Offset", encode_sign_magnitude(offsets[name]), length=2)
                a.wr(mid, "Min_Position_Limit", mins[name], length=2)
                a.wr(mid, "Max_Position_Limit", maxes[name], length=2)
            back = {name: a.rd(mid, "Homing_Offset", 2) for name, mid in MOTORS.items()}
            bad = [name for name, _ in MOTORS.items()
                   if back.get(name) != encode_sign_magnitude(offsets[name])]
            if bad:
                print(f"  ⚠️  {bad} 的 Homing_Offset 回读不一致")

        calib = calibration_dict(offsets, mins, maxes)
        p = calibration_path(role, arm_id)
        if not dry_run:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(calib, indent=4, ensure_ascii=False))
        print(f"  ✅ 标定{'（dry-run，未写入）' if dry_run else '已写入'}: {p}")

        pcts = [(MID - mins[m]) / (maxes[m] - mins[m]) * 100
                for m in MOVABLE_ALL if maxes[m] > mins[m]]
        print(f"  2047 在各关节量程中的位置: {' '.join(f'{x:.0f}%' for x in pcts)}"
              f"   （应接近 50%）")
        return calib


# ================================================================ wrist_roll 零点对齐
# 五个行程关节的零点都锚在【各自实测行程的中点】上，两条臂因此天然对应 ——
# 数据上的证据：shoulder_lift 在三个数据集里都被推到了同一个物理下限，两臂读数
# −104.0 / −103.2，只差 0.8°。
#
# wrist_roll 没有行程可测（lerobot 写死量程 0..4095），零点只能取"写偏移那一刻
# 手腕的朝向"—— 两条臂在不同时间、不同姿态写入，就会差出一个固定角度。
# 实测 2026-10-03：从臂在自己的物理尽头读 −80.1°，主臂在同一物理尽头读 −98.9°，
# 差 18.8°（214 刻度）。后果：采数据时从臂提前顶住，observation 的最低限位与
# action 对不上（−80 vs −99），那一段的 action 从臂物理上执行不了。
#
# 原理（lerobot feetech.py: Present_Position = Actual_Position − Homing_Offset）：
# 同一个物理方向上 raw_从 − raw_主 恒等于两臂的零点差。所以在两个共同的物理
# 方向上各读一次、取平均，Δh = mean(raw_从 − raw_主)，把从臂偏移加上 Δh 即可。
# 取两端平均 = 对齐两臂行程的中点（和五个行程关节同一套思路），线缆松紧造成的
# 宽度差被平摊到两端。
#
# 只动从臂这一个寄存器：其余 5 个关节现在的对应关系是好的（上面 shoulder_lift
# 的证据），不能因为重跑整套 finalize 而被 10-03 那份重新手测的行程数据改动。
WRIST_FIX_REFUSE_TICKS = 200.0   # 两端读数差差这么多：多半没推到同一对物理方向
WRIST_FIX_WARN_TICKS = 30.0      # 超过它提示两臂行程宽度有差（每端差一半）

# lerobot 的 DEGREES 归一化用 分辨率−1 = 4095 当整圈，这里必须和它一致，
# 否则显示的度数和数据集里记的差一点点（0.02%，不影响判断，但没必要有偏差）。
LEROBOT_MAX_RES = RES - 1


def wrist_deg(raw: int, calib: dict) -> float:
    """按标定把 raw 换算成 lerobot 记录的度数（相对量程中点，见 motors_bus._normalize）。"""
    c = calib[FULL_TURN]
    mid = (c["range_min"] + c["range_max"]) / 2
    return (raw - mid) * 360.0 / LEROBOT_MAX_RES


def compute_wrist_fix(pairs: list, old_offset: int) -> tuple[int, list, list]:
    """由两次"共同方向"读数算出从臂 wrist_roll 的新 Homing_Offset。纯计算。

    pairs = [(主臂 raw, 从臂 raw), (主臂 raw, 从臂 raw)]，两个方向各一次。
    返回 (new_offset, notes, problems)；problems 非空时不要写入。
    """
    notes, problems = [], []
    deltas = []
    for i, (rl, rf) in enumerate(pairs, 1):
        d = shortest_arc(rf - rl)
        deltas.append(d)
        notes.append(f"方向{i}: 主臂 {rl:>5} / 从臂 {rf:>5} → 读数差 "
                     f"{d:+.0f} 刻度 ({ticks_to_deg(d):+.1f}°)")

    spread = abs(deltas[0] - deltas[1])
    if spread > WRIST_FIX_REFUSE_TICKS:
        problems.append(
            f"两个方向的读数差相差 {spread:.0f} 刻度（{ticks_to_deg(spread):.1f}°）——"
            f"两条臂是同一套硬件，端点上不该差这么多。多半两次没推到同一对物理"
            f"方向（有一条臂推反了？），重新测一次。"
        )
    elif spread > WRIST_FIX_WARN_TICKS:
        notes.append(
            f"⚠️ 两端的读数差相差 {spread:.0f} 刻度（{ticks_to_deg(spread):.1f}°）："
            f"两臂手腕的可转范围宽度不同（线缆松紧），按中点对齐后两端各差约 "
            f"{ticks_to_deg(spread / 2):.1f}°。"
        )

    delta = sum(deltas) / 2
    new_offset = int(round(old_offset + delta))
    if abs(new_offset) > 2047:
        problems.append(f"新偏移 {new_offset} 超出 ±2047（Homing_Offset 幅值上限），拒绝写入。")
    notes.append(f"从臂 wrist_roll 偏移: {old_offset:+} → {new_offset:+}"
                 f"（Δ {delta:+.0f} 刻度 = {ticks_to_deg(delta):+.1f}°）")
    return new_offset, notes, problems


def write_wrist_eeprom(a: Arm, new_offset: int) -> bool:
    """写从臂 wrist_roll 的 Homing_Offset（EEPROM），并做安全收尾。

    顺序是关键（舵机行为实测，见 prep_goal_safe 的说明）：
      解锁 → 写偏移 → 回读 → Goal := 新读数 → 关力矩（最后）
    写 Goal 会自行打开力矩，所以必须补关；写偏移后读数会整体平移、而 Goal 还是
    旧值，不先把 Goal 对齐就开力矩会让臂朝旧位置冲。
    """
    mid = MOTORS[FULL_TURN]
    a.wr(mid, "Lock", 0)
    if not a.wr(mid, "Homing_Offset", encode_sign_magnitude(new_offset), length=2):
        print("  ✗ 写 Homing_Offset 失败（总线无响应？）")
        return False
    back = a.rd(mid, "Homing_Offset", 2)
    if back is None or decode_sign_magnitude(back) != new_offset:
        print(f"  ✗ 回读不一致：期望 {new_offset}，读到 {back}")
        return False
    now = a.rd(mid, "Present_Position", 2)
    if now is not None:
        a.wr(mid, "Goal_Position", now, length=2)
    a.wr(mid, "Torque_Enable", 0)
    return True


def update_wrist_json(role: str, arm_id: str, new_offset: int) -> Path:
    """只改 lerobot 标定 JSON 里 wrist_roll 的 homing_offset，其余原样写回。

    必须和 EEPROM 一致：lerobot 连接时会用文件里的标定写舵机，
    只改 EEPROM 不改文件的话，下次连接就被改回去了。
    """
    p = calibration_path(role, arm_id)
    calib = json.loads(p.read_text())
    calib[FULL_TURN]["homing_offset"] = new_offset
    p.write_text(json.dumps(calib, indent=4, ensure_ascii=False))
    return p


def fix_wrist(arm_id: str, dry_run: bool = False) -> int:
    """把从臂 wrist_roll 的零点对齐到主臂（只动这一个关节）。

    交互：两条臂的手腕一起推到同一个物理尽头 → 按 x；再推到另一个尽头 → 按 x。
    然后算 Δh 写入从臂 EEPROM + lerobot 的标定 JSON，最后进入验证显示。
    """
    role = "follower"
    p = calibration_path(role, arm_id)
    if not p.exists():
        print(f"❌ 找不到从臂标定文件: {p}")
        print("   先做标定，或确认 --id 与文件名一致。")
        return 1
    calib = json.loads(p.read_text())

    print(f"对齐 {FULL_TURN} 零点（以{ARM_INFO['leader']['cn']}为准，只改"
          f"{ARM_INFO[role]['cn']} 的这一个关节）")
    print("  为什么单独修它：其余 5 个关节的零点取自各自【实测行程的中点】，")
    print(f"  天生对应；{FULL_TURN} 整圈旋转、没有行程可测，零点只能取写入那一刻的")
    print("  朝向，两条臂不同时间写入就会差出一个固定角度。")
    print()
    print("把【两条臂的手腕】一起推到【同一个物理尽头】（哪一头都行，但两条臂要同")
    print("一头），推到底后按 x 记录；然后再一起推到另一头，按 x。")
    print()

    wl = wf = MOTORS[FULL_TURN]
    caps: list[tuple[int, int]] = []
    new_offset = None

    with Arm("leader") as lead, Arm("follower") as fol, KeyWatcher() as keys:
        lead.wr(wl, "Torque_Enable", 0)      # 手腕要能用手转
        fol.wr(wf, "Torque_Enable", 0)
        eeprom_old = decode_sign_magnitude(fol.rd(wf, "Homing_Offset", 2))
        json_old = calib[FULL_TURN]["homing_offset"]

        # ---------- 两次取点 ----------
        scr = Screen()
        last_limp = 0.0
        try:
            while len(caps) < 2:
                if time.time() - last_limp > LIMP_REFRESH_S:
                    lead.wr(wl, "Torque_Enable", 0)
                    fol.wr(wf, "Torque_Enable", 0)
                    last_limp = time.time()

                rl = lead.rd(wl, "Present_Position", 2)
                rf = fol.rd(wf, "Present_Position", 2)
                lines = [
                    f"{FULL_TURN} 零点对齐 —— 第 {len(caps) + 1}/2 个方向",
                    "",
                    f"  {'臂':<6} {'当前(raw)':>10} {'当前(°)':>9}",
                    "  " + "-" * 30,
                    f"  {'主臂':<6} {rl:>10} {wrist_deg(rl, calib):>8.1f}°"
                    if rl is not None else f"  {'主臂':<6}   读取失败",
                    f"  {'从臂':<6} {rf:>10} {wrist_deg(rf, calib):>8.1f}°"
                    if rf is not None else f"  {'从臂':<6}   读取失败",
                ]
                if rl is not None and rf is not None:
                    d = shortest_arc(rf - rl)
                    lines += [
                        "",
                        f"  当前读数差 {d:+.0f} 刻度 = {ticks_to_deg(d):+.1f}°"
                        f"（两臂摆到同一物理方向后，这就是零点差）",
                    ]
                lines += [
                    "",
                    "  >>> 两条臂手腕一起推到同一个物理尽头，按 x 记录"
                    if not caps else
                    "  >>> 再一起推到另一个物理尽头，按 x 记录",
                ]
                scr.draw(lines)

                key = keys.poll()
                if key and key.lower() in ("x", " ", "\r", "\n"):
                    if rl is None or rf is None:
                        pass                    # 读不到读数，忽略这次按键
                    else:
                        caps.append((rl, rf))
                        while keys.poll() is not None:   # 清掉缓冲里剩的按键
                            pass
                time.sleep(REFRESH_INTERVAL_S)
        finally:
            scr.close()

        if eeprom_old is None:
            print("  ✗ 读不到从臂 wrist_roll 的 Homing_Offset，未做任何写入。")
            return 1

        # ---------- 计算 ----------
        print("  两个方向的读数：")
        new_offset, notes, problems = compute_wrist_fix(caps, eeprom_old)
        for n in notes:
            print(f"    {n}")
        if json_old != eeprom_old:
            print(f"    ⚠️  标定文件里的偏移 ({json_old:+}) 与舵机 EEPROM ({eeprom_old:+}) "
                  f"不一致 —— 现在两处一起写成新值。")
        if problems:
            print("  ❌ 校验未通过，未写入：")
            for x in problems:
                print(f"     - {x}")
            return 1
        print()

        if dry_run:
            print("  （dry-run：只测只算，未写入任何寄存器/文件。去掉 --dry-run 即写入。）")
            return 0

        # ---------- 写入 ----------
        if not write_wrist_eeprom(fol, new_offset):
            return 1
        jp = update_wrist_json(role, arm_id, new_offset)
        print(f"  ✅ 已写入：舵机 EEPROM + {jp}")
        print(f"     效果：两臂在同一物理方向的读数差从原来那个值变成 0；"
              f"从臂的可用行程与主臂重合，")
        print(f"           不会再提前顶住（采数据时最低限位两边一致）。")
        print()

        # ---------- 验证显示 ----------
        print("  验证：把两条臂的手腕再摆到同一个物理方向，两行的度数应该一致。")
        print("  按 x 结束（Ctrl-C 也可以）。")
        print()
        scr = Screen()
        try:
            while True:
                rl = lead.rd(wl, "Present_Position", 2)
                rf = fol.rd(wf, "Present_Position", 2)
                lines = [
                    f"{FULL_TURN} 零点对齐 —— 验证（按 x 结束）",
                    "",
                    f"  {'臂':<6} {'当前(raw)':>10} {'当前(°)':>9}",
                    "  " + "-" * 30,
                    f"  {'主臂':<6} {rl:>10} {wrist_deg(rl, calib):>8.1f}°"
                    if rl is not None else f"  {'主臂':<6}   读取失败",
                    f"  {'从臂':<6} {rf:>10} {wrist_deg(rf, calib):>8.1f}°"
                    if rf is not None else f"  {'从臂':<6}   读取失败",
                ]
                if rl is not None and rf is not None:
                    d = shortest_arc(rf - rl)
                    mark = "✓ 已对齐" if abs(ticks_to_deg(d)) < 2.0 else "← 还没摆到同一方向？"
                    lines += ["", f"  度数差 {ticks_to_deg(d):+.1f}°   {mark}"]
                scr.draw(lines)
                key = keys.poll()
                if key and key.lower() in ("x", " ", "\r", "\n"):
                    break
                time.sleep(REFRESH_INTERVAL_S)
        finally:
            scr.close()
        print("  完成。力矩已关（手腕可以继续用手掰）。")
    return 0


# ================================================================ 验证
def verify_arm(role: str, arm_id: str, move: bool = False,
               delta_deg: float = 8.0) -> int:
    """验证标定。阶段 1 只读检查；阶段 2（--move）小幅驱动。

    阶段 2 自己控制力矩时序：先把 Goal 设成当前位置再开力矩，所以不会像
    lerobot 的 configure() 那样猛冲；结束时把 Goal 恢复并关力矩。
    """
    p = calibration_path(role, arm_id)
    if not p.exists():
        print(f"❌ 找不到标定文件: {p}")
        print("   先做标定，或确认 --id 与文件名一致。")
        return 1
    calib = json.loads(p.read_text())
    print(f"标定文件: {p}\n")

    with Arm(role) as a:
        a.disable_torque()

        # ---------- 阶段 1 ----------
        print("=" * 70)
        print("阶段 1：当前位置 vs 标定量程（不做任何运动）")
        print("=" * 70)
        raw = a.positions()
        print(f"\n  {'关节':<15} {'当前位置':>8} {'量程':>16} {'位置(量程%)':>12}  判定")
        print("  " + "-" * 64)
        problems = []
        for m in MOVABLE_ALL:
            c, v = calib.get(m), raw.get(m)
            if c is None or v is None:
                print(f"  {m:<15}   缺标定或缺读数")
                problems.append(m)
                continue
            lo, hi = c["range_min"], c["range_max"]
            rng = hi - lo
            pct = (v - lo) / rng * 100 if rng else float("nan")
            if v < lo or v > hi or abs(c["homing_offset"]) > 2047:
                verdict = "★超出量程/偏移异常"
                problems.append(m)
            elif pct < 5 or pct > 95:
                verdict = "接近端点"
            else:
                verdict = "OK"
            print(f"  {m:<15} {v:>8} [{lo:>5},{hi:>5}] {pct:>10.0f}%  {verdict}")
        print()
        if problems:
            print(f"  ⚠️  有问题的关节: {problems}")
        else:
            print("  ✅ 所有关节都在标定量程内，标定数据自洽。")

        if not move:
            print()
            print("阶段 2（小幅运动测试）默认不跑。要跑就加 --move：")
            print(f"  python -m soarm verify {role} --id {arm_id} --move")
            return 1 if problems else 0

        # ---------- 阶段 2 ----------
        print()
        print("=" * 70)
        print(f"阶段 2：小幅运动（每个关节朝量程中心移动 {delta_deg}° 后返回）")
        print("=" * 70)
        print("⚠️  请勿触碰机械臂。3 秒后开始...")
        for i in (3, 2, 1):
            print(f"  {i}...", end="", flush=True)
            time.sleep(1)
        print("\n")

        a.disable_torque()
        start = a.positions()                # {关节名: 刻度}
        for name, mid in MOTORS.items():     # 先对齐 Goal 再开力矩，避免猛冲
            a.wr(mid, "Goal_Position", start[name], length=2)
        a.enable_torque()
        time.sleep(0.3)

        print(f"  {'关节':<15} {'起始':>7} {'移动量':>9} {'实际变化':>9}  判定")
        print("  " + "-" * 58)
        bad = []
        try:
            for name, mid in MOTORS.items():
                lo, hi = calib[name]["range_min"], calib[name]["range_max"]
                cur = a.rd(mid, "Present_Position", 2)
                step = int(delta_deg / 360 * RES)
                if cur > (lo + hi) // 2:
                    step = -step
                a.wr(mid, "Goal_Position", cur + step, length=2)
                time.sleep(0.8)
                after = a.rd(mid, "Present_Position", 2)
                a.wr(mid, "Goal_Position", cur, length=2)
                time.sleep(0.5)
                d = (after - cur) if (after is not None and cur is not None) else 0
                if abs(d) < abs(step) * 0.3:
                    verdict = "✗ 几乎没动（力矩/卡住？）"
                    bad.append(name)
                elif d * step <= 0:
                    verdict = "✗ 方向相反（标定异常）"
                    bad.append(name)
                else:
                    verdict = "✓ 正常"
                print(f"  {name:<15} {cur:>7} {step:>+9} {d:>+9}  {verdict}")
        finally:
            for mid in MOTORS.values():
                p_now = a.rd(mid, "Present_Position", 2)
                if p_now is not None:
                    a.wr(mid, "Goal_Position", p_now, length=2)
            a.disable_torque()
        print()
        if bad:
            print(f"  ⚠️  异常关节: {bad} —— 先查该关节线缆，再考虑重标。")
        else:
            print("  ✅ 全部正常：能接受指令、方向正确、标定映射自洽。")
        print("     力矩已关闭。")
        return 1 if (bad or problems) else 0


# ================================================================ 主从对比
def compare_arms() -> int:
    """检查主从零点对齐：归一化零点（2047）落在各自量程的哪个百分比。

    两趟标定做对了应该都是 50%。
    """
    from .constants import calibration_dir

    base = calibration_dir()
    files = sorted(base.rglob("*.json")) if base.exists() else []
    if not files:
        print("还没有任何标定文件。")
        return 1
    print("已找到的标定文件:")
    for f in files:
        print(f"  {f}")
    print()

    by_role = {}
    for f in files:
        role = "follower" if "so_follower" in str(f) else "leader"
        by_role[role] = json.loads(Path(f).read_text())
    if len(by_role) < 2:
        print("只有一条臂的标定文件，无法对比主从一致性。")
        return 1

    f, l = by_role["follower"], by_role["leader"]
    print(f"  {'关节':<15} {'从臂行程':>8} {'主臂行程':>8} "
          f"{'从2047位置':>11} {'主2047位置':>11}  判定")
    print("  " + "-" * 72)
    bad = []
    for k in MOTORS:
        fr = f[k]["range_max"] - f[k]["range_min"]
        lr = l[k]["range_max"] - l[k]["range_min"]

        def frac(d):
            rng = d["range_max"] - d["range_min"]
            return (MID - d["range_min"]) / rng * 100 if rng else float("nan")

        ff, lf = frac(f[k]), frac(l[k])
        gap = abs(ff - lf)
        if gap <= 12:
            verdict = "✓ 对齐"
        elif gap <= 25:
            verdict = "△ 有偏差"
        else:
            verdict = "✗ 差太多，建议重标"
            bad.append(k)
        print(f"  {k:<15} {ticks_to_deg(fr):>7.0f}° {ticks_to_deg(lr):>7.0f}° "
              f"{ff:>10.0f}% {lf:>10.0f}%  {verdict}")
    print()
    print("  「2047位置」= 归一化的零点落在量程的百分比。")
    print("  两趟标定做对了应该都是 50%；差距大的关节遥操作会偏移。")
    return 1 if bad else 0
