"""适配器发现、角色判定与端口解析。

两条臂的 USB 适配器是同一型号（WCH CH343，`1a86:55d3`），只能靠芯片出厂
序列号区分，所以序列号必须自动发现而不是写死在代码里。

角色判定依据：Pro 套装的主从臂舵机电压规格不同 —— 12V 舵机（电压上限约
14.0V）的是从臂，7.4V 舵机（8~12V）的是主臂。判定结果缓存到
`~/.cache/soarm/arms.json`，之后所有命令都读它。
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
import time

from .constants import (
    ADAPTER_VID_PID,
    ARMS_FILE,
    BAUDRATE,
    CACHE_DIR,
    VOLT_SEVEN_MAX,
    VOLT_TWELVE_MIN,
)


def find_adapters() -> list[dict]:
    """扫描 /dev/serial/by-id，列出所有 SO-ARM 适配器。"""
    out = []
    for p in sorted(glob.glob("/dev/serial/by-id/*")):
        if ADAPTER_VID_PID not in os.path.basename(p):
            continue
        m = re.search(r"Serial_(\w+)-if", os.path.basename(p))
        out.append({
            "by_id": p,
            "serial": m.group(1) if m else "?",
            "tty": os.path.realpath(p),
        })
    return out


def probe_profile(by_id: str) -> dict:
    """读一个端口的舵机电压画像，用于推断是 7.4V 臂还是 12V 臂。"""
    from scservo_sdk import PacketHandler, PortHandler

    from .constants import PACKET_TIMEOUT_MS, REG_MAX_VOLT_LIMIT, REG_PRESENT_VOLT

    res = {"servos": 0, "max_v": None, "present_v": None, "kind": "?", "perm": False}
    try:
        h = PortHandler(by_id)
        if not h.openPort():
            res["kind"] = "打不开端口（权限不足？）"
            res["perm"] = True
            return res
        h.setBaudRate(BAUDRATE)
        h.setPacketTimeout(PACKET_TIMEOUT_MS)
        time.sleep(0.05)
        ph = PacketHandler(0)
        maxvs, volts, n = [], [], 0
        for mid in range(1, 7):
            model, comm, _ = ph.ping(h, mid)
            if comm != 0 or not model:
                continue
            n += 1
            d, c, _ = ph.readTxRx(h, mid, REG_MAX_VOLT_LIMIT, 1)
            if c == 0 and d:
                maxvs.append(d[0])
            d, c, _ = ph.readTxRx(h, mid, REG_PRESENT_VOLT, 1)
            if c == 0 and d:
                volts.append(d[0])
        h.closePort()
        res["servos"] = n
        if not n:
            res["kind"] = "无舵机响应（未上电 / 总线线缆？）"
            return res
        res["max_v"] = sum(maxvs) / len(maxvs) / 10 if maxvs else None
        res["present_v"] = sum(volts) / len(volts) / 10 if volts else None
        if res["max_v"] is not None:
            if res["max_v"] >= VOLT_TWELVE_MIN:
                res["kind"] = "12V 舵机"
            elif res["max_v"] <= VOLT_SEVEN_MAX:
                res["kind"] = "7.4V 舵机"
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        if "Permission denied" in msg or "could not open port" in msg:
            res["kind"] = "打不开端口（权限不足）"
            res["perm"] = True
        else:
            res["kind"] = f"探测异常 {type(exc).__name__}"
    return res


def load_arms() -> dict:
    try:
        return json.loads(ARMS_FILE.read_text())
    except Exception:  # noqa: BLE001
        return {}


def save_arms(d: dict) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        ARMS_FILE.write_text(json.dumps(d, indent=2))
    except Exception:  # noqa: BLE001
        pass


def role_serial(role: str) -> str:
    """取某角色的序列号，取不到或格式明显不对时直接退出并给出修复命令。"""
    from .constants import ARM_INFO

    s = load_arms().get(role)
    if not s:
        sys.exit(
            f"还没确定{ARM_INFO[role]['cn']}的 USB 序列号。\n"
            f"  先运行:   python -m soarm arms\n"
            f"  或手动写: {ARMS_FILE}"
        )
    if not re.fullmatch(r"[0-9A-Fa-f]{4,32}", str(s)):
        sys.exit(
            f"{ARMS_FILE} 里 {role} 的值不像是真实序列号: {s!r}\n"
            f"  它应该是一串十六进制字符（例如 5B3D048490），从设备的 USB 序列号里来。\n"
            f"  最常见的错误是把文档里的占位符原样抄了进去。\n"
            f"\n"
            f"  直接跑自动发现就能覆盖掉它:\n"
            f"      python -m soarm arms\n"
            f"\n"
            f"  想看真实序列号:  ls /dev/serial/by-id/"
        )
    return str(s)


def port_of(role: str) -> str:
    """角色 -> 稳定端口路径（按序列号，插拔顺序变化不影响）。"""
    return f"/dev/serial/by-id/usb-1a86_USB_Single_Serial_{role_serial(role)}-if00"


def discover_roles(detect_voltage: bool = True) -> dict:
    """发现适配器并按电压画像判定主从，成功则写入缓存。

    detect_voltage=False 时只列设备、不读电压（用于纯排查）。
    """
    from .constants import ARM_INFO

    adaps = find_adapters()
    print(f"发现 {len(adaps)} 个 SO-ARM 适配器\n")
    profiles = []
    for a in adaps:
        p = probe_profile(a["by_id"]) if detect_voltage else {
            "servos": 0, "max_v": None, "present_v": None, "kind": "-", "perm": False}
        profiles.append(p)
        pv = f"{p['present_v']:.1f}V" if p["present_v"] is not None else "?"
        mv = f"{p['max_v']:.1f}V" if p["max_v"] is not None else "?"
        print(f"  序列号 {a['serial']}  ->  {a['tty']}")
        print(f"     {p['servos']} 个舵机   实测 {pv}   电压上限 {mv}   {p['kind']}")

    if any(p.get("perm") for p in profiles):
        _print_permission_help()
        return {}

    if len(adaps) != 2:
        print(f"\n⚠️  期望 2 个适配器，实际 {len(adaps)} 个。")
        print("    检查接线后重跑；或手动写入配置文件:")
        print(f'      echo \'{{"leader":"序列号","follower":"序列号"}}\' > {ARMS_FILE}')
        return {}

    twelve = [a for a, p in zip(adaps, profiles) if p["max_v"] and p["max_v"] >= VOLT_TWELVE_MIN]
    seven = [a for a, p in zip(adaps, profiles) if p["max_v"] and p["max_v"] <= VOLT_SEVEN_MAX]
    if len(twelve) == 1 and len(seven) == 1:
        d = {"leader": seven[0]["serial"], "follower": twelve[0]["serial"]}
        save_arms(d)
        print()
        print(f"✅ 自动判定: leader = {d['leader']} ({ARM_INFO['leader']['servo_v']} 舵机)  "
              f"follower = {d['follower']} ({ARM_INFO['follower']['servo_v']} 舵机)")
        print(f"   已缓存到 {ARMS_FILE}")
        return d

    print()
    print("⚠️  无法按电压自动判定主从（两条臂规格相同，或某条臂没上电）。")
    print("    手动方法：拔掉其中一条臂的 USB，看哪个节点消失即可确定映射，然后:")
    print(f'      echo \'{{"leader":"...","follower":"..."}}\' > {ARMS_FILE}')
    return {}


def _print_permission_help() -> None:
    print()
    print("=" * 66)
    print("❌ 打不开串口：权限不足（不是电压规格问题，也不是没插好）")
    print("=" * 66)
    print("  当前进程不在 dialout 组里。组成员在【登录时】固定，")
    print("  所以 usermod 之后必须重新登录才生效。三种解法：")
    print()
    print("   (1) 注销桌面会话再重新登录（最彻底）")
    print("   (2) 不重新登录，用带组的 shell 跑：")
    print("         sg dialout -c 'python -m soarm arms'")
    print("   (3) 装 udev 规则把端口权限设成 0666（见 README 第 2.4 节）")


__all__ = [
    "find_adapters", "load_arms", "port_of", "probe_profile", "role_serial",
    "save_arms",
]
