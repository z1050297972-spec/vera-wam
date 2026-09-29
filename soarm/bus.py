"""串口总线会话：打开一条臂，读写寄存器。

只做最底层的事 —— 端口开关、单寄存器读写、批量读位置。所有上层功能
（自检、标定、验证）都通过它操作硬件。
"""

from __future__ import annotations

import os
import sys
import time

from scservo_sdk import PacketHandler, PortHandler

from .constants import (
    ARM_INFO,
    BAUDRATE,
    MOTORS,
    PACKET_TIMEOUT_MS,
    REG,
)
from .ports import port_of


class Arm:
    """一条臂的总线会话。用 with 语法自动关闭。"""

    def __init__(self, role: str):
        self.role = role
        self.port = port_of(role)
        if not os.path.exists(self.port):
            sys.exit(
                f"端口不存在: {self.port}\n"
                f"  USB 没插，或适配器没被识别。先执行:  python -m soarm arms"
            )
        self.h = PortHandler(self.port)
        if not self.h.openPort():
            sys.exit(
                f"打不开 {self.port}\n"
                f"  权限不足。执行 sudo usermod -aG dialout $USER 后【重新登录】，\n"
                f"  或本次用:  sg dialout -c 'python -m soarm ...'"
            )
        self.h.setBaudRate(BAUDRATE)
        self.h.setPacketTimeout(PACKET_TIMEOUT_MS)
        time.sleep(0.1)
        self.ph = PacketHandler(0)

    # ------------------------------------------------------------ 生命周期
    def close(self) -> None:
        try:
            self.h.closePort()
        except Exception:  # noqa: BLE001
            pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def __repr__(self) -> str:
        return f"Arm({ARM_INFO[self.role]['cn']}, {self.port})"

    # ------------------------------------------------------------ 寄存器读写
    @staticmethod
    def _need_id(mid) -> int:
        """确保传进来的是舵机 ID（int），不是关节名。

        MOTORS 是 {关节名: ID}，很容易写成 `a.wr("shoulder_pan", ...)`。
        那样字符串会进到数据包里，SDK 算校验和时报
        `TypeError: unsupported operand type(s) for +=: 'int' and 'str'` ——
        错误位置在 SDK 内部，很难看出是自己的调用错了，所以在这里拦掉。
        """
        if not isinstance(mid, int) or isinstance(mid, bool):
            raise TypeError(
                f"舵机 ID 必须是 int，收到 {mid!r}（{type(mid).__name__}）。"
                f"  你可能把关节名当 ID 传进来了 —— MOTORS 是 {{关节名: ID}}，"
                f"用 MOTORS.items() 同时拿两者。"
            )
        return mid

    def rd(self, mid: int, reg: str, length: int = 1):
        """读寄存器。失败返回 None（不抛异常，便于逐项容错）。"""
        self._need_id(mid)
        data, comm, err = self.ph.readTxRx(self.h, mid, REG[reg], length)
        if comm != 0 or data is None:
            return None
        return data[0] if length == 1 else data[0] | (data[1] << 8)

    def wr(self, mid: int, reg: str, val: int, length: int = 1) -> bool:
        """写寄存器。返回是否成功。"""
        self._need_id(mid)
        payload = [val & 0xFF] if length == 1 else [val & 0xFF, (val >> 8) & 0xFF]
        comm, err = self.ph.writeTxRx(self.h, mid, REG[reg], length, payload)
        return comm == 0

    # ------------------------------------------------------------ 常用操作
    def positions(self) -> dict:
        """读全部关节的原始刻度。某个读失败时该项为 None。"""
        return {n: self.rd(m, "Present_Position", 2) for n, m in MOTORS.items()}

    def goals(self) -> dict:
        return {n: self.rd(m, "Goal_Position", 2) for n, m in MOTORS.items()}

    def torques(self) -> dict:
        return {n: self.rd(m, "Torque_Enable") for n, m in MOTORS.items()}

    def ping_all(self) -> dict:
        """逐个 ping，返回 {关节名: 型号}。"""
        found = {}
        for name, mid in MOTORS.items():
            model, comm, err = self.ph.ping(self.h, mid)
            if comm == 0 and model:
                found[name] = model
        return found

    def disable_torque(self) -> None:
        """关闭力矩并解锁 EEPROM。

        注意：写 `Goal_Position` 会让固件**自己把力矩打开**（实测 6/6）。
        所以任何写完 Goal 的地方都必须再关一次，顺序不能颠倒。
        """
        for mid in MOTORS.values():
            self.wr(mid, "Torque_Enable", 0)
            self.wr(mid, "Lock", 0)

    def enable_torque(self) -> None:
        for mid in MOTORS.values():
            self.wr(mid, "Torque_Enable", 1)

    def is_limp(self) -> bool:
        """回读确认力矩是否全关。"""
        return all(v == 0 for v in self.torques().values())
