"""舵机编码与角度换算。

`encode_sign_magnitude` 必须与 lerobot 的实现逐位一致，否则写进舵机的
Homing_Offset 会和 lerobot 的读回值对不上，lerobot 会认为该臂未标定。
已用 592 个整数用例与 `lerobot.motors.encoding_utils` 比对，结果全部相同。
"""

from __future__ import annotations

from .constants import HOMING_SIGN_BIT, RES


def encode_sign_magnitude(value: int, sign_bit_index: int = HOMING_SIGN_BIT) -> int:
    """符号幅值编码：负值置最高位，剩余位存幅值。

    与 `lerobot.motors.encoding_utils.encode_sign_magnitude` 等价。
    `Homing_Offset` 寄存器用 sign_bit_index=11，所以幅值上限是 2047 ——
    超出会抛 ValueError，这也是「关节停在刻度顶端会让标定崩溃」的原因。
    """
    max_magnitude = (1 << sign_bit_index) - 1
    magnitude = abs(value)
    if magnitude > max_magnitude:
        raise ValueError(
            f"幅值 {magnitude} 超过 {max_magnitude}（sign_bit_index={sign_bit_index}）"
        )
    return ((1 if value < 0 else 0) << sign_bit_index) | magnitude


def decode_sign_magnitude(encoded: int, sign_bit_index: int = HOMING_SIGN_BIT) -> int:
    """符号幅值解码，用于回读校验。"""
    direction = (encoded >> sign_bit_index) & 1
    magnitude = encoded & ((1 << sign_bit_index) - 1)
    return -magnitude if direction else magnitude


def shortest_arc(delta_ticks: float, res: int = RES) -> float:
    """把刻度差折到 [-res/2, res/2)，即圆上的**最短弧**（带符号）。

    为什么需要它：编码器读数是 0..4095 的单圈值，关节行程**可能跨过 0 点**
    （从 4095 继续转到 0 再往上）。此时直接算 `max - min` 会得到接近 4096
    的值，显示成 ~360°，而真实行程可能只有 180°。

    正确做法是按最短弧累积位移（"展开"到实数轴上），行程宽度取展开后的
    极差 —— 与起点在哪里无关。

    前提：相邻两次采样之间关节不会真的转过半圈（>180°）。以 5Hz 采样手动
    推臂，这个前提总是成立。
    """
    half = res / 2.0
    return (delta_ticks + half) % res - half


def ticks_to_deg(ticks: float) -> float:
    """编码器刻度 -> **舵机整圈**角度（0–360°）。

    ⚠️ 这不是关节角度，只是"读数落在整圈里的哪个位置"。关节的机械行程只是
    整圈的一小段，所以一个只能转 154° 的关节，它的绝对读数完全可能是
    170°~320° 这种值 —— 拿它当关节角度显示会得到"接近 360°"这种不可能的数。

    **表达关节角度必须相对行程来算**：用 `ticks_from()` 求相对某个参考点的
    偏移。标定工具里所有面向用户的角度列都走这两个函数。
    """
    return ticks / RES * 360.0


def ticks_from(ticks: float, reference: float) -> float:
    """刻度相对某个参考点的偏移，换算成角度（带符号）。

    参考点取行程下限时得到 0..行程宽度；取中点时得到 ±行程宽度/2。
    这才是关节角度的正确表达方式。
    """
    return (ticks - reference) / RES * 360.0


def deg_to_ticks(deg: float) -> int:
    """角度 -> 编码器刻度。"""
    return int(round(deg / 360.0 * RES))
