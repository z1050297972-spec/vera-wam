"""命令行入口：子命令定义与分发。

    python -m soarm <子命令> [参数]

每个子命令都是对 calibration / hardware / ports 里函数的薄封装。
"""

from __future__ import annotations

import argparse
import sys

from . import align, calibration, camera, hardware
from .constants import ROLES
from .ports import discover_roles, load_arms

EPILOG = """\
常用流程：

  # 1. 识别与自检
  python -m soarm arms                    # 自动发现两条臂、按电压判定主从
  python -m soarm check all               # 电压/温度/错误位

  # 2. 标定（每条臂三趟；从臂用 follower，主臂换成 leader）
  python -m soarm measure  follower       # 第一趟：手推测真实行程
  python -m soarm mid      follower       # 第二趟：摆到各自实测中点
  python -m soarm finalize follower --id my_follower --dry-run   # 先空跑
  python -m soarm finalize follower --id my_follower             # 写入
  python -m soarm compare                 # 主从零点对齐（应都是 50%）

  # 3. 验证与遥操作
  python -m soarm verify follower --id my_follower          # 只读检查
  python -m soarm verify follower --id my_follower --move   # 小幅运动测试
  python -m soarm fix-wrist --id my_follower                # 修 wrist_roll 零点（两臂手腕推到同一对方向）
  python -m soarm align                   # 读主臂位姿，把从臂对齐过去（按 x 结束）
  python -m soarm prep follower           # 开力矩前的安全预处理

  # 4. 摄像头（VERA 采数据前跑一遍）
  python -m soarm cam                     # 体检（所有相机）：自动项 + 实测帧率/亮度稳定性
  python -m soarm cam --lock              # 锁曝光/白平衡（拔插后要重跑）
  python -m soarm cam --simul             # 多相机同时开流体检（USB2 带宽够不够）
  python -m soarm cam --preview           # 本地预览 http://127.0.0.1:8099

退出码：0 正常 / 1 有问题或被拒绝 / 4 权限不足。

详细说明见仓库根目录 README.md。
"""


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m soarm",
        description="SO-ARM101 硬件自检 / 力矩安全 / 两趟标定 / 标定验证",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EPILOG,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    # ---------------------------------------------------------- 识别与检查
    p = sub.add_parser("arms", help="自动发现两条臂并按电压判定主从")
    p.set_defaults(func=_cmd_arms)

    p = sub.add_parser("check", help="硬件自检：电压/温度/错误位")
    p.add_argument("role", choices=ROLES + ["all"], nargs="?", default="all")
    p.set_defaults(func=_cmd_check)

    p = sub.add_parser("probe", help="深度探测：两种协议 × 全部波特率（无响应时用）")
    p.add_argument("role", choices=ROLES)
    p.set_defaults(func=_cmd_probe)

    p = sub.add_parser("state", help="力矩/Goal/位置快照")
    p.add_argument("role", choices=ROLES)
    p.set_defaults(func=_cmd_state)

    # ---------------------------------------------------------- 力矩安全
    p = sub.add_parser("prep", help="开力矩前的安全预处理（Goal := 当前位置）")
    p.add_argument("role", choices=ROLES)
    p.set_defaults(func=_cmd_prep)

    # ---------------------------------------------------------- 标定
    p = sub.add_parser("measure", help="标定第一趟：测出每个关节的真实行程")
    p.add_argument("role", choices=ROLES)
    p.set_defaults(func=_cmd_measure)

    p = sub.add_parser("mid", help="标定第二趟：摆到各自实测中点（按 x 结束）")
    p.add_argument("role", choices=ROLES)
    p.add_argument("--tolerance", type=float, default=8.0, help="容差角度（默认 8）")
    p.set_defaults(func=_cmd_mid)

    p = sub.add_parser("finalize", help="标定第三趟：写入 EEPROM 和 JSON")
    p.add_argument("role", choices=ROLES)
    p.add_argument("--id", required=True, help="标定文件名（如 my_follower）")
    p.add_argument("--dry-run", action="store_true", help="只算不写")
    p.set_defaults(func=_cmd_finalize)

    p = sub.add_parser("verify", help="验证标定：范围检查 +（可选）小幅运动测试")
    p.add_argument("role", choices=ROLES)
    p.add_argument("--id", required=True)
    p.add_argument("--move", action="store_true", help="跑阶段 2（会驱动机械臂）")
    p.add_argument("--delta", type=float, default=8.0,
                   help="阶段 2 的移动角度（默认 8）")
    p.set_defaults(func=_cmd_verify)

    p = sub.add_parser("compare", help="主从零点对齐检查")
    p.set_defaults(func=_cmd_compare)

    p = sub.add_parser("fix-wrist",
                       help="对齐 wrist_roll 零点：两臂手腕推到同一对方向，只改从臂这一个关节")
    p.add_argument("--id", required=True, help="从臂标定文件名（如 my_follower）")
    p.add_argument("--dry-run", action="store_true", help="只测只算，不写入")
    p.set_defaults(func=_cmd_fix_wrist)

    p = sub.add_parser("align", help="读主臂位姿，把从臂对齐到同一个姿态（按 x 结束）")
    p.add_argument("--dry-run", action="store_true",
                   help="只显示偏差，不驱动任何东西")
    p.add_argument("--follower-id", default=None, help="从臂标定 id（默认用唯一的那个）")
    p.add_argument("--leader-id", default=None, help="主臂标定 id（默认用唯一的那个）")
    p.add_argument("--max-step", type=float, default=align.DEFAULT_MAX_STEP_DEG,
                   help=f"每个控制周期最多移动的角度（默认 "
                        f"{align.DEFAULT_MAX_STEP_DEG:.0f}，约 40°/s）")
    p.add_argument("--limp", action="store_true",
                   help="结束时关掉从臂力矩（默认保持力矩、停在当前姿态）")
    p.set_defaults(func=_cmd_align)

    # ---------------------------------------------------------- 摄像头
    p = sub.add_parser("cam", help="摄像头：体检 / 锁定设置 / 存帧 / 本地预览")
    p.add_argument("--device", default=None,
                   help="只操作这一只（默认对找到的**所有**相机操作）")
    p.add_argument("--list", dest="list_only", action="store_true",
                   help="列出所有摄像头（序列号 + 能不能出帧）")
    p.add_argument("--simul", action="store_true",
                   help="同时开流体检：多只相机能不能一起跑（VERA 多视角的硬要求）")
    p.add_argument("--lock", action="store_true",
                   help="锁定曝光/白平衡/对焦（UVC 设置拔插即失效，每次采数据前都要重跑）")
    p.add_argument("--exposure", type=int, default=None,
                   help="锁定时用的曝光绝对值（默认保持当前值）")
    p.add_argument("--wb", type=int, default=None, help="锁定时用的色温 K（默认保持当前值）")
    p.add_argument("--brightness", type=int, default=None,
                   help="数字增益（本机有效，但会牺牲对比度；默认不动）")
    p.add_argument("--snap", default=None, metavar="路径",
                   help="存一帧（同时存 VERA 用的 128×192 版本）")
    p.add_argument("--preview", action="store_true",
                   help="本地预览 http://127.0.0.1:<port>（会独占摄像头）")
    p.add_argument("--port", type=int, default=8099, help="预览端口（默认 8099）")
    p.add_argument("--seconds", type=float, default=5.0, help="体检时长（默认 5 秒）")
    p.set_defaults(func=_cmd_cam)

    return ap


# ------------------------------------------------------------------ 各子命令
def _cmd_arms(args) -> int:
    return 0 if discover_roles() else 1


def _cmd_check(args) -> int:
    roles = list(ROLES) if args.role == "all" else [args.role]
    results = {}
    for r in roles:
        print("=" * 66)
        results[r] = hardware.check_arm(r)
        print()
    print("结论: " + "   ".join(f"{r}={v}" for r, v in results.items()))
    return 0 if all(v == "OK" for v in results.values()) else 1


def _cmd_probe(args) -> int:
    return hardware.deep_probe(args.role)


def _cmd_state(args) -> int:
    hardware.read_state(args.role)
    return 0


def _cmd_prep(args) -> int:
    return 0 if hardware.prep_goal_safe(args.role) else 1


def _cmd_measure(args) -> int:
    t = calibration.measure_travel(args.role)
    narrow = [m for m in t if t[m]["width_deg"] < calibration.MIN_TRAVEL_DEG]
    wide = [m for m in t if t[m]["width_deg"] > calibration.IMPLAUSIBLE_TRAVEL_DEG]
    if wide:
        print(f"\n不可信的数据（行程过大）: {wide} —— 请重新测量，不要用它做标定。")
    return 1 if (narrow or wide) else 0


def _cmd_mid(args) -> int:
    travel = calibration.load_travel(args.role)
    print(f"读入行程: {calibration.state_file(args.role)}\n")
    ok = calibration.show_angles_to_midpoints(args.role, travel, args.tolerance)
    return 0 if ok else 1


def _cmd_finalize(args) -> int:
    travel = calibration.load_travel(args.role)
    calib = calibration.finalize_two_pass(args.role, args.id, travel,
                                         dry_run=args.dry_run)
    return 0 if calib else 1


def _cmd_verify(args) -> int:
    return calibration.verify_arm(args.role, args.id, args.move, args.delta)


def _cmd_compare(args) -> int:
    return calibration.compare_arms()


def _cmd_fix_wrist(args) -> int:
    return calibration.fix_wrist(args.id, dry_run=args.dry_run)


def _cmd_align(args) -> int:
    return align.align_arms(args.follower_id, args.leader_id, dry_run=args.dry_run,
                            max_step_deg=args.max_step, hold=not args.limp)


def _cmd_cam(args) -> int:
    return camera.run_camera(device=args.device, list_only=args.list_only,
                             lock=args.lock, exposure=args.exposure, wb=args.wb,
                             brightness=args.brightness, snap=args.snap,
                             do_preview=args.preview, simul=args.simul,
                             port=args.port, seconds=args.seconds)


# ------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if not load_arms():
        # 缓存缺失时给一句提示，但不阻断 —— 有些子命令自己会报得更清楚
        print("（提示：还没确定两条臂的序列号，先运行 python -m soarm arms）\n",
              file=sys.stderr)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n已中断。")
        return 130
