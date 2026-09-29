"""终端交互小工具：实时刷新 + 单键读取。

`Screen` 负责多行原地重绘（终端下用 ANSI 光标上移；输出被重定向到文件时
退化为每 2 秒打印一次，这样 `... > log.txt` 也能看到进度）。

`KeyWatcher` 负责非阻塞读单个按键 —— 用户按一下 `x` 就结束，不需要回车。
"""

from __future__ import annotations

import select
import sys
import termios
import time
import tty

REFRESH_INTERVAL_S = 0.2      # 终端下的重绘间隔
NON_TTY_INTERVAL_S = 2.0      # 非终端下打印一帧的间隔
BAR_WIDTH = 25                # 位置条长度


class Screen:
    """多行原地重绘，避免刷屏。"""

    def __init__(self) -> None:
        self.n = 0
        self.tty = sys.stdout.isatty()
        self.last = 0.0

    def draw(self, lines: list[str]) -> None:
        if not self.tty:
            now = time.time()
            if now - self.last < NON_TTY_INTERVAL_S:
                return
            self.last = now
            print("\n".join(lines))
            return
        if self.n:
            sys.stdout.write(f"\033[{self.n}A")
        for ln in lines:
            sys.stdout.write("\033[K" + ln + "\n")
        self.n = len(lines)
        sys.stdout.flush()

    def close(self) -> None:
        if self.tty:
            sys.stdout.write("\n")
        else:
            print()
        self.n = 0
        sys.stdout.flush()


class KeyWatcher:
    """非阻塞读单个按键，用来实现"按 x 结束"。

    用 termios cbreak 模式，所以不需要按回车；退出时恢复终端原状态。
    输入不是终端（被重定向）时 poll() 永远返回 None —— 那种情况下用
    Ctrl-C 结束（走 KeyboardInterrupt，同样会保留已测数据）。
    """

    def __init__(self) -> None:
        self.enabled = sys.stdin.isatty()
        self.fd = sys.stdin.fileno() if self.enabled else None
        self.saved = None

    def __enter__(self):
        if self.enabled:
            try:
                self.saved = termios.tcgetattr(self.fd)
                tty.setcbreak(self.fd)
            except Exception:  # noqa: BLE001
                self.enabled = False
        return self

    def __exit__(self, *a):
        if self.enabled and self.saved is not None:
            try:
                termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)
            except Exception:  # noqa: BLE001
                pass

    def poll(self) -> str | None:
        """返回刚按下的字符；没按键返回 None。"""
        if not self.enabled:
            return None
        try:
            if select.select([sys.stdin], [], [], 0)[0]:
                return sys.stdin.read(1)
        except Exception:  # noqa: BLE001
            return None
        return None


def range_bar(value: float, lo: float, hi: float,
              mark: float | None = None, width: int = BAR_WIDTH) -> str:
    """把 value 在 [lo, hi] 里的位置画成位置条，当前位置是 `o`，mark 处是 `|`。

    坐标轴是**该关节的实际范围**（行程），不是编码器整圈 —— 否则一个只能转
    187° 的关节会被画在整圈的一小段里，看不出名堂。
    """
    if hi <= lo:
        return "-" * width
    span = hi - lo

    def idx_of(x: float) -> int:
        return min(width - 1, max(0, int((x - lo) / span * width)))

    bar = ["-"] * width
    if mark is not None and lo <= mark <= hi:
        bar[idx_of(mark)] = "|"
    bar[idx_of(value)] = "o"
    return "".join(bar)
