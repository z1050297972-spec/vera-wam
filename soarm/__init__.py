"""SO-ARM101 工具包 —— 硬件自检、力矩安全、两趟标定、标定验证、主从对齐、摄像头。

只依赖 `feetech-servo-sdk`（纯 Python，仅拉 `pyserial`），不需要 lerobot
或 torch，因此可以装在 VERA 的 conda 环境里直接用：

    pip install feetech-servo-sdk

命令行用法：

    python -m soarm --help

作为库使用：

    from soarm import Arm, check_arm, prep_goal_safe, measure_travel

模块划分：

    constants   常量、寄存器地址、角色定义与路径
    encoding    舵机编码与角度换算
    ports       适配器发现、角色判定、端口解析
    bus         串口总线会话（Arm 类）
    hardware    自检 / 深度探测 / 状态快照 / 力矩安全预处理
    calibration 两趟标定 + 验证 + 主从对比
    align       主从姿态对齐（以主臂为基准）
    camera      摄像头：设备发现 / 设置锁定 / 体检 / 存帧 / 预览
    ui          终端实时刷新
    cli         命令行入口
"""

from __future__ import annotations

__version__ = "2.1.0"

from .align import align_arms
from .bus import Arm
from .camera import check_camera, check_simultaneous, find_devices, run_camera
from .calibration import (
    compare_arms,
    finalize_two_pass,
    measure_travel,
    show_angles_to_midpoints,
    verify_arm,
)
from .constants import (
    ARM_INFO,
    FULL_TURN,
    MID,
    MOTORS,
    RES,
    ROLES,
    calibration_path,
    state_file,
)
from .encoding import (
    decode_sign_magnitude,
    deg_to_ticks,
    encode_sign_magnitude,
    ticks_to_deg,
)
from .hardware import check_all, check_arm, deep_probe, prep_goal_safe, read_state
from .ports import discover_roles, find_adapters, load_arms, port_of, role_serial

__all__ = [
    "__version__",
    # 常量
    "ARM_INFO", "FULL_TURN", "MID", "MOTORS", "RES", "ROLES",
    "calibration_path", "state_file",
    # 编码
    "decode_sign_magnitude", "deg_to_ticks", "encode_sign_magnitude", "ticks_to_deg",
    # 端口
    "discover_roles", "find_adapters", "load_arms", "port_of", "role_serial",
    # 总线
    "Arm",
    # 硬件
    "check_all", "check_arm", "deep_probe", "prep_goal_safe", "read_state",
    # 标定
    "compare_arms", "finalize_two_pass", "measure_travel",
    "show_angles_to_midpoints", "verify_arm",
    # 主从对齐
    "align_arms",
    # 摄像头
    "check_camera", "check_simultaneous", "find_devices", "run_camera",
]
