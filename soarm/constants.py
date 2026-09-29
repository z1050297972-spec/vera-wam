"""常量、寄存器地址、角色定义与路径。

寄存器地址取自 lerobot 的 sts3215 控制表（用 `get_address(MODEL_CONTROL_TABLE,
"sts3215", ...)` 逐项核对过），不要凭记忆改。
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------- 编码器与中位
RES = 4096            # STS3215 编码器一圈 4096 刻度（12 位）
MID = 2047            # 半圈；标定以它作为中位参考值
MID_DEG = 180.0
HOMING_SIGN_BIT = 11  # Homing_Offset 用符号幅值编码，幅值上限 2^11-1 = 2047

# ---------------------------------------------------------------- 关节
MOTORS = {
    "shoulder_pan": 1,
    "shoulder_lift": 2,
    "elbow_flex": 3,
    "wrist_flex": 4,
    "wrist_roll": 5,
    "gripper": 6,
}
MOVABLE_ALL = list(MOTORS)
FULL_TURN = "wrist_roll"   # 整圈旋转关节，量程写死 0..4095，不参与行程测量

# ---------------------------------------------------------------- 寄存器
REG = {
    "Return_Delay_Time": 7,
    "Acceleration": 41,
    "Maximum_Acceleration": 85,
    # Phase 的 bit4 控制角度反馈模式：置位时 STS3215 的位置读数会溢出或取负，
    # 表现为读数在整圈范围内跳动（行程看起来接近 360°）。lerobot 在
    # configure_motors() 里会清掉它，注释写明"只有 STS3215 需要"。
    "Phase": 18,
    "Torque_Enable": 40,
    "Goal_Position": 42,
    "Lock": 55,
    "Homing_Offset": 31,
    "Min_Position_Limit": 9,
    "Max_Position_Limit": 11,
    "Operating_Mode": 33,
    "Present_Position": 56,
    "Present_Voltage": 62,
    "Present_Temperature": 63,
    "Status": 65,
    "Min_Voltage_Limit": 15,
    "Max_Voltage_Limit": 14,
}

# Status 字节的位含义
ERR_BITS = {0: "电压", 1: "角度", 2: "过热", 3: "过流", 4: "过载", 5: "电气"}

# ---------------------------------------------------------------- 通信
BAUDRATE = 1_000_000
PACKET_TIMEOUT_MS = 100
ADAPTER_VID_PID = "1a86_USB_Single_Serial"   # 沁恒 CH343 适配器
REG_MAX_VOLT_LIMIT = 14
REG_PRESENT_VOLT = 62

# ---------------------------------------------------------------- 角色
ARM_INFO = {
    "leader": {"cn": "主臂 leader", "supply": "5V 4A", "servo_v": "7.4V"},
    "follower": {"cn": "从臂 follower", "supply": "12V 2A", "servo_v": "12V"},
}
ROLES = list(ARM_INFO)

# 电压画像判据：12V 舵机的 Max_Voltage_Limit 约 14.0V，7.4V 舵机在 8~12V
VOLT_TWELVE_MIN = 13.5
VOLT_SEVEN_MAX = 12.5

# ---------------------------------------------------------------- 路径
# 机器配置（序列号等）放用户缓存目录
CACHE_DIR = Path(os.getenv("SOARM_CACHE", "~/.cache/soarm")).expanduser()
ARMS_FILE = CACHE_DIR / "arms.json"

# 测量结果放仓库里的 measure/ 目录 —— 直接可见、方便取用和检查，
# 不用去翻 ~/.cache。可用 SOARM_MEASURE_DIR 覆盖。
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
MEASURE_DIR = Path(os.getenv("SOARM_MEASURE_DIR", _PROJECT_ROOT / "measure")).expanduser()


def state_file(role: str) -> Path:
    """第一趟（measure）测得的结果写这里，第二/三趟（mid / finalize）读它。"""
    return MEASURE_DIR / f"{role}.json"


def lerobot_home() -> Path:
    return Path(os.getenv("HF_LEROBOT_HOME", "~/.cache/huggingface/lerobot")).expanduser()


def calibration_dir() -> Path:
    """复刻 lerobot 的标定文件位置。"""
    return lerobot_home() / "calibration"


def calibration_subdir(role: str) -> str:
    """角色 -> lerobot 标定文件所在的子目录。"""
    return "robots/so_follower" if role == "follower" else "teleoperators/so_leader"


def calibration_path(role: str, arm_id: str) -> Path:
    """角色 + id -> lerobot 的标定文件路径。"""
    return calibration_dir() / calibration_subdir(role) / f"{arm_id}.json"
