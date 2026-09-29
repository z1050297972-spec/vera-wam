"""摄像头：设备发现、设置锁定、体检、本地预览、存帧。

为什么单独成模块：VERA 对画面的**全局外观**很敏感 —— 光流和视频模型分不清
"物体在动"和"整个画面变亮/变黄了"。而 UVC 的这些设置是**易失**的：摄像头一拔
一插就回出厂默认（自动曝光 + 自动白平衡）。所以采数据/部署前必须重新锁一遍。

本机实测（JoyandAI JYU2C-2083，静态场景 10 秒）：

| 项 | 自动（出厂） | 锁成手动后 |
|---|---|---|
| 画面亮度漂移 | **42 级** | 0.1 级 |

| 格式 | fps | 帧间隔抖动 |
|---|---|---|
| **MJPG 1280×720** | **29.8** | 2.0ms |
| MJPG 640×480 | 24.8 | 1.3ms |
| YUYV 1280×720 | 9.9 | 2.0ms |

所以默认用 MJPG 1280×720（这一档反而最快，小尺寸并不会更快 —— 是驱动在该模式
下的默认帧率低）。

`auto_exposure` 的取值（V4L2）：0=自动，1=手动，2=快门优先，3=光圈优先。出厂是 3，
也就是"相机会自己调曝光去凑目标亮度"，正是亮度漂移的来源。

**多相机（实测）**：两只同型号 JYU2C-2083 一起插上时，**只能有一只开流成功**，
而且与分辨率无关（连 320×240 都一样）——谁先开流谁活。
原因不是带宽总量不够，而是这类相机的流接口提供高带宽等时档（端点最高
`3×1020 字节/微帧`），而 USB 2.0 一条总线的等时预算约 6000 字节/微帧，两条这样
的流挤不下。**换更好的 hub 没用**（USB 2.0 的等时带宽是主机控制器的，不是 hub 的；
本机只有一个控制器，所有 USB2 口共用 `usb1`）。要同时用两只：
① 加一张 PCIe USB 卡（多一个控制器）；② 换一只 **USB 3.0** 相机（走独立的
SuperSpeed 总线）；③ 只用一只（PushT 的 IDM 就是单视角）。
`python -m soarm cam --simul` 就是测这个的。
"""

from __future__ import annotations

import errno
import fcntl
import glob
import os
import re
import struct
import sys
import time

# V4L2 ioctl：结构大小必须对，否则内核直接 EINVAL
_IOC_RW = 3
VIDIOC_QUERYCTRL = (_IOC_RW << 30) | (68 << 16) | (ord("V") << 8) | 36
VIDIOC_G_CTRL = (_IOC_RW << 30) | (8 << 16) | (ord("V") << 8) | 27
VIDIOC_S_CTRL = (_IOC_RW << 30) | (8 << 16) | (ord("V") << 8) | 28
V4L2_CTRL_FLAG_NEXT_CTRL = 0x80000000

# 常用控制项 ID（UVC 标准）
CTRL = {
    "brightness": 0x00980900,
    "contrast": 0x00980901,
    "gain": 0x00980913,
    "power_line_frequency": 0x00980918,
    "white_balance_automatic": 0x0098090C,
    "white_balance_temperature": 0x0098091A,
    "exposure_auto": 0x009A0901,
    "exposure_absolute": 0x009A0902,
    "focus_auto_continuous": 0x009A090C,
}
EXPOSURE_MEANING = {0: "自动", 1: "手动", 2: "快门优先", 3: "光圈优先"}

# 采集默认分辨率（本机实测最快的一档）
WIDTH, HEIGHT, FOURCC = 1280, 720, "MJPG"

# 体检判定
MIN_FPS = 15.0            # VERA 控制循环 15Hz
MAX_BRIGHT_RANGE = 3.0    # 静态场景下亮度极差超过这么多就说明画面在漂


# ---------------------------------------------------------------- 设备发现
def find_devices(explicit: str | None = None) -> list[str]:
    """找摄像头节点，返回**全部**（或只返回显式指定的那个）。

    优先用 /dev/v4l/by-id 的稳定路径（插拔/重启不会换位）。`video0`/`video1` 按
    枚举顺序排、`videoN-index1` 通常是 metadata 节点（读不出帧），所以不要写死
    `/dev/video0`。
    """
    if explicit:
        if not os.path.exists(explicit):
            sys.exit(f"设备不存在: {explicit}")
        return [explicit]
    cands = sorted(glob.glob("/dev/v4l/by-id/*video-index0"))
    if cands:
        return cands
    if os.path.exists("/dev/video0"):
        print("（提示：没找到 /dev/v4l/by-id 下的稳定路径，退回 /dev/video0；"
              "插拔后编号可能变）")
        return ["/dev/video0"]
    sys.exit("找不到摄像头：/dev/v4l/by-id/* 和 /dev/video0 都没有。")


def serial_of(device: str) -> str:
    """从 by-id 路径里取序列号，用来区分同型号的多只相机。"""
    m = re.search(r"-([0-9A-Za-z]+)-video-index\d+$", device)
    return m.group(1) if m else device


def _cv2():
    import cv2
    return cv2


def open_stream(device: str, cv2=None):
    """按采集参数（MJPG 1280×720）打开一路流。"""
    cv2 = cv2 or _cv2()
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*FOURCC))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    return cap


# ---------------------------------------------------------------- V4L2 控制
class Controls:
    """直接读写 V4L2 控制项（不需要 v4l2-ctl）。

    UVC 的这些设置保存在相机里、**掉电/拔出即失效**，所以每次都要重设。
    """

    def __init__(self, device: str):
        self.device = device
        try:
            self.fd = os.open(device, os.O_RDWR)
        except PermissionError:
            sys.exit(f"打不开 {device}（权限）。看下 ls -l {device} 的组/ACL。")
        except OSError as e:
            sys.exit(f"打不开 {device}: {e}")

    def close(self) -> None:
        try:
            os.close(self.fd)
        except OSError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def available(self) -> dict:
        """枚举设备支持的控制项。

        键尽量用本模块的短名（`exposure_auto` 等），认不出来的就用驱动给的原名。
        """
        by_id = {v: k for k, v in CTRL.items()}
        out, cid = {}, V4L2_CTRL_FLAG_NEXT_CTRL
        while True:
            try:
                buf = fcntl.ioctl(self.fd, VIDIOC_QUERYCTRL,
                                  struct.pack("II32siiiiiII", cid, 0, b"", 0, 0, 0, 0, 0, 0, 0))
            except OSError as e:
                if e.errno == errno.EINVAL:
                    break
                raise
            qid = struct.unpack_from("I", buf, 0)[0]
            driver_name = buf[8:40].split(b"\x00")[0].decode(errors="replace")
            mn, mx, st, df = struct.unpack_from("iiii", buf, 40)
            out[by_id.get(qid, driver_name)] = {
                "driver_name": driver_name, "id": qid, "min": mn, "max": mx,
                "step": st, "default": df, "current": self.get(qid),
            }
            cid = qid | V4L2_CTRL_FLAG_NEXT_CTRL
        return out

    def get(self, cid: int):
        try:
            buf = fcntl.ioctl(self.fd, VIDIOC_G_CTRL, struct.pack("Ii", cid, 0))
            return struct.unpack_from("i", buf, 4)[0]
        except OSError:
            return None

    def set(self, cid: int, value: int):
        """写控制项。驱动可能拒绝某些值（本机 `exposure_auto=0` 就会被拒），
        这种情况返回 None 并把原因报到 stderr，而不是让命令崩掉。"""
        try:
            fcntl.ioctl(self.fd, VIDIOC_S_CTRL, struct.pack("Ii", cid, int(value)))
        except OSError as e:
            print(f"    ⚠️  写控制项 0x{cid:08x}={value} 被驱动拒绝"
                  f"（{errno.errorcode.get(e.errno, e.errno)}）", file=sys.stderr)
            return None
        return self.get(cid)


def lock_settings(ctrl: Controls, exposure: int | None = None,
                  wb: int | None = None, brightness: int | None = None,
                  quiet: bool = False) -> list[str]:
    """把会自动漂的项锁掉，返回改动说明。

    - `exposure_auto` → 1（手动 = 不再自己调曝光）。**这一条才是关键**：实测本机在
      光圈优先模式下静态场景 10 秒亮度漂 42 级，手动后 0.1 级。
    - `exposure_absolute`／`white_balance_temperature`：跟着一起写，但**本机这两项
      可能是桩控制** —— 实测 `exposure_absolute` 78 与 10000 画面完全不变；色温同理
      待验证。所以别指望它们调亮度，亮度靠**环境光和 `brightness`**。
    - `brightness`：数字增益，本机有效（64 把亮度从 179 抬到 228），但会牺牲对比度
      （实测清晰度 77 → 67），能不用就不用。
    - `focus_auto_continuous` → 0（本机无对焦马达，桩控制，锁上只是省心）
    - `power_line_frequency` 不动（本机已是 50Hz，抗灯光频闪）
    """
    avail = ctrl.available()
    changes = []

    def apply(name: str, want: int, label: str) -> None:
        c = avail.get(name)
        if not c:
            return
        before = c["current"]
        if before == want:
            changes.append(f"{label:<24} 已是 {want}，无需改")
            return
        after = ctrl.set(c["id"], want)
        changes.append(f"{label:<24} {before} → {after}")

    if "exposure_auto" in avail:
        apply("exposure_auto", 1, "曝光模式（1=手动）")
        keep = exposure if exposure is not None else (
            avail.get("exposure_absolute", {}).get("current"))
        if keep is not None and "exposure_absolute" in avail:
            if ctrl.set(avail["exposure_absolute"]["id"], keep) is not None:
                changes.append(f"{'  └ 曝光值':<22} → {keep}（本机可能是桩控制）")

    if "white_balance_automatic" in avail:
        keep_wb = wb if wb is not None else avail.get("white_balance_temperature", {}).get("current")
        apply("white_balance_automatic", 0, "自动白平衡")
        if keep_wb is not None and "white_balance_temperature" in avail:
            if ctrl.set(avail["white_balance_temperature"]["id"], keep_wb) is not None:
                changes.append(f"{'  └ 色温 K':<22} → {keep_wb}")

    if "focus_auto_continuous" in avail:
        apply("focus_auto_continuous", 0, "连续自动对焦")

    if brightness is not None and "brightness" in avail:
        if ctrl.set(avail["brightness"]["id"], brightness) is not None:
            changes.append(f"{'brightness（数字增益）':<22} → {brightness}（会牺牲对比度）")

    if not quiet:
        print("  锁定以下设置（UVC 设置是易失的，拔插后要重跑）:")
        for c in changes:
            print(f"    {c}")
    return changes


def lock_report(avail: dict) -> list[str]:
    """只读检查：哪些自动项还开着。"""
    problems = []
    exp = avail.get("exposure_auto", {}).get("current")
    if exp is not None and exp != 1:
        problems.append(f"曝光模式={exp}（{EXPOSURE_MEANING.get(exp, '?')}）→ 画面亮度会自己漂")
    if avail.get("white_balance_automatic", {}).get("current") == 1:
        problems.append("自动白平衡开着 → 整幅画面色温会漂")
    if avail.get("focus_auto_continuous", {}).get("current") == 1:
        problems.append("连续自动对焦开着（本机是桩控制，无实际影响）")
    return problems


# ---------------------------------------------------------------- 体检
def check_camera(device: str, seconds: float = 5.0, cv2=None) -> int:
    """打开摄像头按采集配置实测：fps、帧间隔抖动、亮度稳定性、清晰度。

    只读 —— 不改任何设置。返回 0 表示满足 VERA 采集要求。
    """
    import numpy as np
    if cv2 is None:
        import cv2 as _cv2
        cv2 = _cv2

    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*FOURCC))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    if not cap.isOpened():
        print(f"❌ 打不开 {device}")
        return 1

    for _ in range(8):                       # 预热：AE/驱动切换要几帧才稳
        cap.read()
    t0, means, gaps, sharp = time.time(), [], [], []
    last, i, fails = time.time(), 0, 0
    while time.time() - t0 < seconds:
        ok, frame = cap.read()
        now = time.time()
        if not ok:
            fails += 1
            continue
        gaps.append(now - last)
        last = now
        i += 1
        means.append(float(frame.mean()))
        if i % 4 == 0:
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharp.append(float(cv2.Laplacian(g, cv2.CV_64F).var()))
    shape = frame.shape if i else None
    cap.release()

    if not gaps:
        print("❌ 一帧都没读到")
        return 1
    g, m, s = np.array(gaps), np.array(means), np.array(sharp or [0.0])
    fps = len(g) / seconds
    print(f"  实际分辨率 {shape[1]}×{shape[0]}   {FOURCC}   {fps:.1f} fps"
          f"   抖动 {g.std() * 1000:.1f}ms（最慢 {g.max() * 1000:.0f}ms）   读失败 {fails}")
    print(f"  亮度 {m.mean():.1f}   极差 {m.max() - m.min():.1f}   清晰度（Laplacian 方差）{s.mean():.0f}")

    problems = []
    if fps < MIN_FPS:
        problems.append(f"帧率 {fps:.1f} < {MIN_FPS:.0f}（VERA 控制循环要 15Hz）")
    if m.max() - m.min() > MAX_BRIGHT_RANGE:
        problems.append(f"亮度极差 {m.max() - m.min():.1f} > {MAX_BRIGHT_RANGE}（画面在漂；"
                        f"先跑 --lock，若已锁则是光源在变）")
    if fails:
        problems.append(f"读了 {fails} 帧失败（USB 带宽/线缆？）")
    return 1 if problems else 0


def check_simultaneous(devices: list[str], seconds: float = 5.0) -> int:
    """同时开流体检：多相机能不能**一起**跑（VERA 多视角的硬要求）。

    USB 2.0 的等时带宽是主机控制器的，两条高带宽流挤不下 —— 表现就是"谁先开流谁活"，
    另一只一帧都读不到，而且和分辨率无关。这里如实报出来并给出可选方案。
    """
    import threading
    import numpy as np
    cv2 = _cv2()
    if len(devices) < 2:
        print("只有一只相机，无需同时开流体检。")
        return 0

    fps, fails = {}, {}

    def worker(dev: str) -> None:
        cap = open_stream(dev, cv2)
        if not cap.isOpened():
            fps[dev], fails[dev] = 0.0, -1
            return
        for _ in range(8):
            cap.read()
        t0, n, bad, last, gaps = time.time(), 0, 0, time.time(), []
        while time.time() - t0 < seconds:
            ok, _f = cap.read()
            t = time.time()
            if not ok:
                bad += 1
                if bad > 300:
                    break
                continue
            gaps.append(t - last)
            last = t
            n += 1
            cv2.resize(_f, (192, 128), interpolation=cv2.INTER_AREA)   # 模拟 VERA 预处理
        cap.release()
        fps[dev] = n / seconds if gaps else 0.0
        fails[dev] = bad

    threads = [threading.Thread(target=worker, args=(d,)) for d in devices]
    print(f"  同时开流 {seconds:.0f} 秒（每帧含 resize 到 128×192）:")
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for d in devices:
        mark = "✅" if fps[d] >= MIN_FPS else "✗"
        extra = "（读失败 " + str(fails[d]) + "）" if fails.get(d, 0) else ""
        print(f"    {mark} {serial_of(d):<12} {fps[d]:5.1f} fps{extra}")
    if all(fps[d] >= MIN_FPS for d in devices):
        print("  ✅ 多路可以同时跑。")
        return 0
    print("\n  ❌ 有相机跑不起来 —— USB 2.0 等时带宽不够，两条高带宽流挤不下。")
    print("     注意：**换更好的 hub 没用**（带宽归主机控制器管），换端口也没用。")
    print("     要同时用多路，三条路：")
    print("       ① 加一张 PCIe USB 卡（多一个控制器）")
    print("       ② 换一只 USB 3.0 相机（走独立的 SuperSpeed 总线，不抢 USB2 的预算）")
    print("       ③ 只用一只相机（VERA 的 PushT IDM 就是单视角，可以先跑通）")
    return 1


# ---------------------------------------------------------------- 存帧 / 预览
def snapshot(device: str, path: str, cv2=None) -> int:
    """存一帧：原始尺寸 + VERA 实际喂给模型的 128×192（BGR→RGB 后存）。"""
    if cv2 is None:
        import cv2 as _cv2
        cv2 = _cv2
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*FOURCC))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    for _ in range(10):
        ok, frame = cap.read()
    cap.release()
    if not ok:
        print("❌ 读不到帧")
        return 1
    big = path
    small = os.path.splitext(path)[0] + "_128x192.jpg"
    cv2.imwrite(big, frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    rgb = frame[..., ::-1].copy()                      # VERA 契约是 RGB
    cv2.imwrite(small, cv2.resize(rgb, (192, 128), interpolation=cv2.INTER_AREA),
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"  已存 {big}（{frame.shape[1]}×{frame.shape[0]}）")
    print(f"  已存 {small}（VERA 喂给模型的 128×192）")
    return 0


def preview(devices: list[str], port: int = 8099) -> int:
    """本地预览：把所有相机的 MJPEG 流转发到 http://127.0.0.1:<port>（只监听本机）。

    预览会独占摄像头 —— 开着的时候别的程序（lerobot 录数据等）读不到画面。
    **多相机时要注意**：USB 2.0 的多只相机通常只有一只开流成功（见模块头部说明），
    页面会如实显示哪一路是黑的。
    """
    if isinstance(devices, str):
        devices = [devices]
    cv2 = _cv2()
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    states = {}
    caps = []
    for i, dev in enumerate(devices):
        cap = open_stream(dev, cv2)
        if not cap.isOpened():
            print(f"❌ 打不开 {dev}")
            continue
        states[i] = {"dev": dev, "jpeg": b"", "n": 0, "fps": 0.0, "err": 0}
        caps.append((i, cap))

    if not states:
        return 1

    def pump(i: int, cap) -> None:
        st = states[i]
        t0, n, last = time.time(), 0, time.time()
        while True:
            ok, frame = cap.read()
            if not ok:
                st["err"] += 1
                time.sleep(0.05)
                continue
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                st["jpeg"] = buf.tobytes()
                st["n"] += 1
                n += 1
                now = time.time()
                if now - t0 >= 1.0:
                    st["fps"] = n / (now - t0)
                    t0, n = now, 0

    for i, cap in caps:
        threading.Thread(target=pump, args=(i, cap), daemon=True).start()

    cards = "\n".join(
        f'<div class="c"><h4>{serial_of(states[i]["dev"])} <span id="f{i}"></span></h4>'
        f'<img src="/stream/{i}"><p><a href="/snap/{i}.jpg">存当前帧</a></p></div>'
        for i in states
    )
    page = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>SO-ARM 摄像头</title><style>
body{{background:#111;color:#ddd;font-family:sans-serif;margin:16px}}
.c{{display:inline-block;vertical-align:top;margin:8px}}
img{{max-width:640px;border:1px solid #444;border-radius:4px}} code{{color:#8bf}}
h4{{margin:4px 0}} .dim{{color:#777}}</style></head><body>
<h3>摄像头预览 · {WIDTH}×{HEIGHT} {FOURCC} · {len(states)} 路</h3>
{cards}
<p class="dim">多路同时预览时，USB 2.0 的情况下通常只有一路有画面（见 README 2.7）。
关闭预览：<code>pkill -f "soarm cam"</code></p>
<script>
setInterval(async () => {{
  for (const i of {list(states)}) {{
    try {{ const r = await fetch('/fps/'+i); const t = await r.text();
      document.getElementById('f'+i).textContent = t; }} catch (e) {{}}
  }}
}}, 1000);
</script></body></html>""".encode()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body: bytes, ctype: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                self._send(page, "text/html; charset=utf-8")
            elif self.path.startswith("/fps/"):
                i = int(self.path.rsplit("/", 1)[1])
                st = states.get(i)
                txt = "—" if not st else (
                    f"{st['fps']:.1f} fps" + (f" ⚠️ 读失败 {st['err']}" if st["err"] else ""))
                self._send(txt.encode(), "text/plain; charset=utf-8")
            elif self.path.startswith("/snap/"):
                i = int(self.path.split("/")[2].split(".")[0])
                self._send(states[i]["jpeg"], "image/jpeg")
            elif self.path.startswith("/stream/"):
                i = int(self.path.rsplit("/", 1)[1])
                st = states[i]
                self.send_response(200)
                self.send_header("Content-Type",
                                 "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                seen = -1
                try:
                    while True:
                        if st["n"] != seen and st["jpeg"]:
                            seen = st["n"]
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                             b"Content-Length: "
                                             + str(len(st["jpeg"])).encode()
                                             + b"\r\n\r\n" + st["jpeg"] + b"\r\n")
                        time.sleep(0.05)
                except Exception:
                    return
            else:
                self.send_response(404)
                self.end_headers()

    print(f"预览: http://127.0.0.1:{port}   （{len(states)} 路；Ctrl-C 结束；会独占摄像头）")
    try:
        ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
    except KeyboardInterrupt:
        print("\n已结束预览。")
    return 0


# ---------------------------------------------------------------- 主流程
def run_camera(device: str | None = None, list_only: bool = False,
               lock: bool = False, exposure: int | None = None,
               wb: int | None = None, brightness: int | None = None,
               snap: str | None = None, do_preview: bool = False,
               simul: bool = False, port: int = 8099,
               seconds: float = 5.0) -> int:
    """`python -m soarm cam` 的实现：体检（默认）/ 列出 / 同时开流 / 锁定 / 存帧 / 预览。

    不加 `--device` 时对**所有**找到的相机操作（多相机场景）。
    """
    devices = find_devices(device)

    if list_only:
        print(f"找到 {len(devices)} 只摄像头:\n")
        cv2 = _cv2()
        for d in devices:
            cap = open_stream(d, cv2)
            info = "打不开"
            ok = False
            if cap.isOpened():
                for _ in range(5):
                    ok, frame = cap.read()
                if ok:
                    info = f"{frame.shape[1]}×{frame.shape[0]}  亮度 {frame.mean():.1f}"
            cap.release()
            print(f"  {'✅' if ok else '✗'} 序列号 {serial_of(d):<14} {info}")
            print(f"     {d}")
        return 0

    if simul:
        return check_simultaneous(devices, seconds)

    if snap:
        rc = 0
        for d in devices:
            name = os.path.splitext(snap)[0] + (
                "" if len(devices) == 1 else "_" + serial_of(d)) + os.path.splitext(snap)[1]
            print(f"  相机 {serial_of(d)}:")
            rc = snapshot(d, name) or rc
        return rc

    if do_preview:
        return preview(devices, port)

    rc = 0
    for dev in devices:
        print(f"=== 摄像头 {serial_of(dev)} ===")
        print(f"    {dev}")
        rc = _check_one(dev, lock=lock, exposure=exposure, wb=wb,
                        brightness=brightness, seconds=seconds) or rc
        print()
    return rc


def _check_one(dev: str, lock: bool = False, exposure: int | None = None,
               wb: int | None = None, brightness: int | None = None,
               seconds: float = 5.0) -> int:
    """单只相机的体检（+ 可选锁定）。"""
    rc = 0
    with Controls(dev) as ctrl:
        avail = ctrl.available()
        print("  当前的自动项:")
        exp = avail.get("exposure_auto", {}).get("current")
        if exp is not None:
            print(f"    曝光模式            {exp}（{EXPOSURE_MEANING.get(exp, '?')}）")
        for name, label in (("exposure_absolute", "曝光值"),
                            ("white_balance_automatic", "自动白平衡"),
                            ("white_balance_temperature", "色温"),
                            ("focus_auto_continuous", "连续自动对焦"),
                            ("power_line_frequency", "抗闪烁频率"),
                            ("brightness", "亮度"), ("gain", "增益")):
            if name in avail:
                print(f"    {label:<18} {avail[name]['current']}")
        if lock:
            print()
            lock_settings(ctrl, exposure=exposure, wb=wb, brightness=brightness)
            avail = ctrl.available()
        problems = lock_report(avail)
        if problems:
            print("\n  ⚠️  还没锁的项（会让画面整体漂移，VERA 会把它当运动）:")
            for p in problems:
                print(f"     - {p}")
            print("     → 跑:  python -m soarm cam --lock")
            if not lock:
                rc = 1
    print("\n  实测:")
    rc = check_camera(dev, seconds) or rc
    if rc == 0:
        print("  ✅ 满足 VERA 采集要求（帧率、亮度稳定性都过关）")
    return rc
