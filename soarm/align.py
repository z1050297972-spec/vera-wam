"""主从对齐：读主臂的位姿，把从臂驱动到同一个姿态。

主臂是基准、只读 —— 它的力矩全程关着，用手摆；从臂由本工具驱动跟随。
归一化用的是各自标定文件里的 range_min/range_max（和 lerobot 归一化用的同一组
数），所以这里算出的偏差为 0，在 lerobot 眼里相对位姿也就是 0。

遥操作启动时从臂会朝主臂姿态移动，`--robot.max_relative_target` 只是限速、不是
限位（4.4），所以先对齐过，启动就只是小范围微调。

按 x 结束：从臂停在当前姿态、保持力矩，可以直接起遥操作（不要再跑 prep）。
`--dry-run` 只显示偏差，不动任何东西。

安全：写 `Goal_Position` 会让固件自动开力矩（3.1），所以开力矩前先把 Goal 对齐到
当前位置（否则会朝 SRAM 里的 0 猛冲）。

驱动是**积分式**的：命令值持续朝目标累积，直到实测位置到位 —— 因为带负载的关节
有静差（实测 elbow 托着前臂时差 30 刻度 = 2.6°），每周期只推一小步会落在静差里，
关节永远不动。见 `advance()`。
"""

from __future__ import annotations

import json
import sys
import time
from unicodedata import east_asian_width

from .bus import Arm
from .constants import (
    ARM_INFO,
    FULL_TURN,
    MOTORS,
    calibration_dir,
    calibration_path,
    calibration_subdir,
)
from .encoding import deg_to_ticks, shortest_arc, ticks_to_deg
from .ui import KeyWatcher, Screen, range_bar

# 控制周期（20 Hz）。实际速率受半双工总线限制，可能略慢 —— 慢只会让移动更
# 保守，是安全的方向。
CONTROL_INTERVAL_S = 0.05

# 主臂必须保持松（用手摆），每 2 秒重发一次关力矩
LIMP_REFRESH_S = 2.0

# 判定「已对齐」的角度容差
TOLERANCE_DEG = 5.0

# 每个控制周期最多移动的角度。40°/s 量级：跟得上手摆主臂，又不会猛冲。
DEFAULT_MAX_STEP_DEG = 2.0

# 到位判定：实测位置离目标小于这么多刻度就不再推进命令值（2 刻度 ≈ 0.18°）
SETTLE_TICKS = 2

# 卡住判定：目标还差这么多刻度、而且位置这么久没变 → 顶到限位或被挡住
STUCK_TICKS = 11          # ≈1°
STUCK_S = 1.0

# 命令值最多领先实测位置这么多刻度（≈7.9°）。用途：
#   ① 覆盖重力静差（实测 elbow 需要 30 刻度）——正常运动时领先量由负载自己决定，
#      远小于这个上限，所以它只影响"顶住"的情形；
#   ② 反饱和：关节被挡住时命令值不会无限累积，放开时也不会突然冲很远。
LEAD_MAX_TICKS = 90

# 表格列宽（显示宽度）。表头是中文、数据是数字，所以列宽必须按显示宽度算，
# 不能按字符个数算 —— 否则中文表头会整体左移，和数字列对不上。
COL_NAME, COL_PCT, COL_MOVE, COL_BAR = 15, 9, 9, 27


def _w(s: str) -> int:
    """字符串的显示宽度：CJK 字符占 2 列。"""
    return sum(2 if east_asian_width(c) in "WF" else 1 for c in s)


def _cell(s: str, width: int, right: bool = False) -> str:
    pad = " " * max(0, width - _w(s))
    return pad + s if right else s + pad


# ---------------------------------------------------------------- 标定文件
def resolve_id(role: str, given: str | None) -> str:
    """确定用哪个标定 id：显式给就用给的，否则用该角色目录下唯一的那个。"""
    d = calibration_dir() / calibration_subdir(role)
    files = sorted(p.stem for p in d.glob("*.json")) if d.exists() else []

    if given:
        if given not in files:
            sys.exit(
                f"{ARM_INFO[role]['cn']} 找不到 id 为 {given} 的标定文件: {d / (given + '.json')}\n"
                f"  该目录下现有: {files or '（空）'}\n"
                f"  先标定:  python -m soarm measure {role} / mid {role} / "
                f"finalize {role} --id {given}"
            )
        return given
    if not files:
        sys.exit(
            f"{ARM_INFO[role]['cn']} 还没有标定文件: {d}\n"
            f"  先按 3.3–3.6 标定:  python -m soarm finalize {role} --id my_{role}"
        )
    if len(files) > 1:
        sys.exit(
            f"{d} 下有多个标定文件，分不清用哪个: {files}\n"
            f"  用 --{role}-id 指定遥操作真正在用的那个 id。"
        )
    return files[0]


def _check_drive_mode(cal: dict, role: str) -> None:
    """确认标定里的 drive_mode 都是 0。

    本工具的映射按 drive_mode=0 算 —— 本项目的标定（3.5）一律写 0。若某个关节被
    手工改成了 1，lerobot 会把它的归一化读数反过来（`-norm`），而本工具不知道，
    对齐出来的结果就会和 lerobot 不一致，所以这里直接拒绝而不是猜。
    """
    bad = [n for n in MOTORS if cal.get(n, {}).get("drive_mode", 0) != 0]
    if bad:
        sys.exit(
            f"{ARM_INFO[role]['cn']} 的标定里这些关节 drive_mode 不是 0: {bad}\n"
            f"  lerobot 会把它们的归一化读数反过来，本工具按 drive_mode=0 算，\n"
            f"  结果会不一致。重新标定（3.5 / 3.6）会把 drive_mode 写回 0。"
        )


def load_pair(follower_id: str | None, leader_id: str | None):
    """读两条臂的标定文件，返回 (从臂标定, 主臂标定)。"""
    f_id = resolve_id("follower", follower_id)
    l_id = resolve_id("leader", leader_id)
    fcal = json.loads(calibration_path("follower", f_id).read_text())
    lcal = json.loads(calibration_path("leader", l_id).read_text())
    _check_drive_mode(fcal, "follower")
    _check_drive_mode(lcal, "leader")
    print(f"从臂标定: {calibration_path('follower', f_id)}")
    print(f"主臂标定: {calibration_path('leader', l_id)}")
    return fcal, lcal


# ---------------------------------------------------------------- 对齐计算
def frac_of(cal: dict, raw) -> float | None:
    """原始刻度 -> 归一化位置（0 = 行程下限，1 = 上限）。"""
    if raw is None:
        return None
    lo, hi = cal["range_min"], cal["range_max"]
    return (raw - lo) / (hi - lo) if hi > lo else None


def delta_ticks(name: str, target: float, raw: float) -> float:
    """从当前位置到目标刻度要走的量（刻度，带符号）。

    行程关节直接相减：标定后的量程一定落在 [0,4095] 内（finalize 会拒绝越界的
    量程），所以量程内部不会绕圈，直接相减就是那条正确的路径。

    `wrist_roll` 是例外 —— 整圈旋转、没有行程端点，可以一直转，所以走**最短弧**，
    免得为了差几度转一整圈。
    """
    d = target - raw
    return shortest_arc(d) if name == FULL_TURN else d


def joint_row(name: str, lcal: dict, fcal: dict, l_raw, f_raw) -> dict:
    """一个关节：把主臂的归一化位置映射到**从臂**量程上，得到从臂的目标刻度。

    `delta_deg` = 从臂还要转的角度（+ 表示往它自己量程的上限方向）。
    """
    row = {"name": name, "target": None, "delta_deg": None, "l_frac": None,
           "f_frac": None, "f_raw": f_raw, "lo": None, "range": None,
           "clamped": False, "stuck": False}
    if l_raw is None or f_raw is None:
        return row

    l_frac = frac_of(lcal[name], l_raw)
    f_frac = frac_of(fcal[name], f_raw)
    if l_frac is None or f_frac is None:
        return row

    fc = fcal[name]
    f_lo, f_hi = fc["range_min"], fc["range_max"]
    target = f_lo + l_frac * (f_hi - f_lo)
    # 主臂摆到了自己的行程之外时，映射过来的目标会落在从臂量程外 —— 只能到端点
    clamped = False
    if target < f_lo:
        target, clamped = float(f_lo), True
    elif target > f_hi:
        target, clamped = float(f_hi), True

    row.update({
        "l_frac": l_frac, "f_frac": f_frac, "target": target, "clamped": clamped,
        "lo": f_lo, "range": f_hi - f_lo,
        "delta_deg": ticks_to_deg(delta_ticks(name, target, f_raw)),
    })
    return row


def build_rows(lcal: dict, fcal: dict, l_pos: dict, f_pos: dict) -> list[dict]:
    return [joint_row(n, lcal, fcal, l_pos.get(n), f_pos.get(n)) for n in MOTORS]


def state_of(row: dict) -> str:
    if row["delta_deg"] is None:
        return "读取失败"
    if row["clamped"]:
        return "★主臂超量程"
    if row.get("stuck") and abs(row["delta_deg"]) > TOLERANCE_DEG:
        return "★没动（顶住/被挡住）"
    return "OK" if abs(row["delta_deg"]) <= TOLERANCE_DEG else "偏离"


def count_ok(rows: list[dict]) -> int:
    return sum(1 for r in rows if state_of(r) == "OK")


# ---------------------------------------------------------------- 驱动一步
def advance(goal: float, target: float, present: int, step_ticks: int,
            lo: int, hi: int) -> float:
    """把命令值 `goal` 朝目标推进一步，返回新的命令值。

    **必须积分，不能每周期从当前位置重算。** 带负载的关节有明显静差：实测从臂
    elbow 停下时离命令值差 30 刻度（2.6°）、wrist_flex 差 15 刻度 —— 重力把它压
    在下面，舵机自己走不到命令值。如果每周期下发「当前位置 ± 一小步」，命令值就
    一直落在静差里 → 舵机认为已经到了 → **关节永远不动**（实测：每周期 +1° 推
    elbow 完全不动，一次写 13° 立刻就动）。

    积分让命令值持续朝目标方向累积，直到**实测位置**追上目标：静差被自动补偿
    （命令值最后停在目标上方，正好抵消重力），到位后残差只剩舵机自身分辨率。

    `goal` 夹在本关节量程内，不会越界；`step_ticks` 是每周期最大推进量；命令值最多
    领先实测位置 `LEAD_MAX_TICKS`（反饱和：关节被顶住时不会无限累积）。
    """
    if abs(target - present) <= SETTLE_TICKS:
        return goal
    goal = goal + max(-step_ticks, min(step_ticks, target - present))
    goal = max(present - LEAD_MAX_TICKS, min(present + LEAD_MAX_TICKS, goal))
    return max(lo, min(hi, goal))


# ---------------------------------------------------------------- 界面
def frame(rows: list[dict], dry_run: bool, started: bool = True) -> list[str]:
    """实时帧。位置条里的 | 是主臂位置映射到从臂量程上的目标，o 是从臂当前位置。"""
    lines = [
        "主臂 → 从臂 对齐    "
        + ("只读预览（--dry-run，不驱动任何东西）" if dry_run else "从臂跟随主臂")
        + f"    容差 ±{TOLERANCE_DEG:.1f}°",
        "",
        ("  " + _cell("关节", COL_NAME) + _cell("主臂位置", COL_PCT, True)
         + _cell("从臂位置", COL_PCT, True) + _cell("需移动", COL_MOVE, True)
         + "   " + _cell("位置条", COL_BAR) + "  " + "状态"),
        "  " + "-" * (COL_NAME + COL_PCT * 2 + COL_MOVE + 3 + COL_BAR + 2 + 4),
    ]
    for r in rows:
        if r["target"] is None:
            lines.append("  " + _cell(r["name"], COL_NAME) + _cell("—", COL_PCT, True) * 2
                         + _cell("—", COL_MOVE, True) + "   " + _cell("—", COL_BAR)
                         + "  读取失败")
            continue
        bar = range_bar(r["f_raw"] - r["lo"], 0, r["range"],
                        mark=r["target"] - r["lo"])
        lines.append(
            "  " + _cell(r["name"], COL_NAME)
            + _cell(f"{r['l_frac'] * 100:.0f}%", COL_PCT, True)
            + _cell(f"{r['f_frac'] * 100:.0f}%", COL_PCT, True)
            + _cell(f"{r['delta_deg']:+.1f}°", COL_MOVE, True)
            + "   " + _cell(f"[{bar}]", COL_BAR) + "  " + state_of(r))
    n_ok, n = count_ok(rows), len(rows)
    lines += [
        "",
        f"  对齐 {n_ok}/{n} 个关节（偏差在 ±{TOLERANCE_DEG:.1f}° 内算对齐）",
        "  位置 = 各自行程的百分比（0% = 行程下限，100% = 上限；lerobot 也这么归一化）",
        "  需移动 = 从臂要转的角度，+ 表示往它自己量程的上限方向",
        "  位置条 = 从臂的行程：| 是主臂位置映射过来的目标，o 是从臂现在的位置",
        "",
    ]
    if dry_run:
        lines.append("  >>> 用手把从臂摆到与主臂一致的姿态；按 x 结束")
    else:
        lines.append("  >>> 用手摆主臂（它的力矩是关的），从臂会跟着动；按 x 结束（停在当前姿态）")
        if n_ok == n:
            lines.append("  ✅ 已对齐，可以按 x 结束")
    return lines


# ---------------------------------------------------------------- 主流程
def align_arms(follower_id: str | None = None, leader_id: str | None = None,
               dry_run: bool = False,
               max_step_deg: float = DEFAULT_MAX_STEP_DEG, hold: bool = True) -> int:
    """读主臂位姿，把从臂驱动到同一个姿态。返回 0 表示退出时所有关节都在容差内。

    `dry_run=True`：只读预览，两条臂都关力矩，用手摆。
    `hold=True`  ：结束时从臂保持力矩，停在当前姿态（否则关力矩，会因重力下垂）。
    """
    fcal, lcal = load_pair(follower_id, leader_id)
    if not dry_run and not sys.stdin.isatty():
        print("\n❌ 会驱动从臂的模式需要交互终端：要能按 x 结束、按 Ctrl-C 停下。")
        print("   输出被重定向时，请先跑一次 --dry-run（只读预览）。")
        return 1
    if dry_run:
        print("只读预览：两条臂都关力矩，用手摆。要驱动从臂就去掉 --dry-run。\n")
    else:
        print("主臂是基准、只读（力矩关，用手摆）；从臂由本工具驱动跟随。")
        print(f"⚠️  从臂马上开始朝主臂姿态移动，每周期最多 {max_step_deg:.1f}°。\n")

    step_ticks = max(1, deg_to_ticks(max_step_deg))

    with Arm("follower") as follower, Arm("leader") as leader, KeyWatcher() as keys:
        leader.disable_torque()
        follower.disable_torque()

        if not dry_run:
            # 开力矩前先把 Goal 对齐当前位置 —— 顺序不能反（写 Goal 会让固件自己
            # 开力矩；而 Goal 在 SRAM 里、上电为 0，不先对齐会朝刻度 0 猛冲，见 3.1）。
            pos = follower.positions()
            if any(v is None for v in pos.values()):
                print("  ✗ 从臂有舵机读不到位置，不敢开力矩。先跑: python -m soarm check follower")
                return 1
            for name in MOTORS:
                follower.wr(MOTORS[name], "Goal_Position", pos[name], length=2)
            time.sleep(0.2)
            follower.enable_torque()
            time.sleep(0.2)

        scr = Screen()
        bad_writes = 0
        last_limp = 0.0
        # 每个关节的命令值（积分项）、上次读数、上次动过的时间 —— 见 advance()
        goals = {n: pos[n] for n in MOTORS} if not dry_run else {}
        last_raw = {}
        last_move = {}
        try:
            while True:
                t0 = time.time()
                if time.time() - last_limp > LIMP_REFRESH_S:
                    # 主臂必须一直松着（写 Goal 会自动开力矩，别的进程也可能动它）
                    leader.disable_torque()
                    last_limp = time.time()

                now = time.time()
                rows = build_rows(lcal, fcal, leader.positions(), follower.positions())

                if not dry_run:
                    for r in rows:
                        name = r["name"]
                        if r["target"] is None:
                            continue
                        if last_raw.get(name) is None or \
                                abs(shortest_arc(r["f_raw"] - last_raw[name])) > 1:
                            last_move[name] = now          # 动过（或第一次）→ 重新计时
                        last_raw[name] = r["f_raw"]
                        r["stuck"] = (abs(goals[name] - r["f_raw"]) > STUCK_TICKS
                                      and now - last_move.get(name, now) > STUCK_S)

                        goal = int(round(advance(goals[name], r["target"], r["f_raw"],
                                                 step_ticks, r["lo"], r["lo"] + r["range"])))
                        if goal != goals[name]:
                            goals[name] = goal
                            if not follower.wr(MOTORS[name], "Goal_Position",
                                               goal, length=2):
                                bad_writes += 1

                lines = frame(rows, dry_run)
                if bad_writes:
                    lines.append(f"  ⚠️  写入失败 {bad_writes} 次（总线偶发丢包，不影响大局）")
                scr.draw(lines)

                key = keys.poll()
                if key and key.lower() == "x":
                    break

                dt = time.time() - t0
                if dt < CONTROL_INTERVAL_S:
                    time.sleep(CONTROL_INTERVAL_S - dt)
        except KeyboardInterrupt:
            print("\n\n  ■ 已中断。")
        finally:
            scr.close()

        # ---------------- 收尾：从臂定住（或松开），并报最终状态 ----------------
        f_pos = follower.positions()
        rows = build_rows(lcal, fcal, leader.positions(), f_pos)
        if not dry_run and all(v is not None for v in f_pos.values()):
            for name in MOTORS:                  # Goal := 当前位置，停在原地不动
                follower.wr(MOTORS[name], "Goal_Position", f_pos[name], length=2)
            if hold:
                follower.enable_torque()
            else:
                follower.disable_torque()

    note = "从臂力矩没开过（--dry-run），仍是松的" if dry_run else (
        "从臂保持力矩，停在此姿态" if hold else "从臂已松手")
    print(f"\n  最终状态（{note}）:")
    print("  " + _cell("关节", COL_NAME) + _cell("主臂位置", COL_PCT, True)
          + _cell("从臂位置", COL_PCT, True) + _cell("偏差", COL_MOVE, True))
    print("  " + "-" * (COL_NAME + COL_PCT * 2 + COL_MOVE))
    for r in rows:
        if r["target"] is None:
            print("  " + _cell(r["name"], COL_NAME) + _cell("—", COL_PCT, True) * 2
                  + _cell("—", COL_MOVE, True))
            continue
        print("  " + _cell(r["name"], COL_NAME)
              + _cell(f"{r['l_frac'] * 100:.0f}%", COL_PCT, True)
              + _cell(f"{r['f_frac'] * 100:.0f}%", COL_PCT, True)
              + _cell(f"{r['delta_deg']:+.1f}°", COL_MOVE, True)
              + "  " + state_of(r))

    n_ok = count_ok(rows)
    print()
    if n_ok == len(rows):
        print(f"  ✅ 主从姿态已对齐（{n_ok}/{len(rows)} 个关节都在 ±{TOLERANCE_DEG:.1f}° 内）")
    else:
        print(f"  ⚠️  还有 {len(rows) - n_ok} 个关节没对上")
        print("     按 x 是【停在当前姿态】，不会继续追目标 —— 再跑一次让它多走一会儿；")
        print("     某个关节一直差很多（接近整段行程宽度）就是装配方向问题，先按 3.8 的方向自检")
    if not dry_run and hold and n_ok == len(rows):
        print("     从臂保持力矩、停在主臂姿态：可以直接起遥操作（4.4），不要再跑 prep。")
    return 0 if n_ok == len(rows) else 1
