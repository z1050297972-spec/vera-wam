"""Render the IDM teaching video (1280x720, H.264, silent, ~10 min).

Built as a course rather than a summary: every concept is introduced from
scratch and then tied to a **specific line of the real repository**, extracted at
build time (never retyped), so a viewer can go read that function afterwards.

    python video/make_assets.py          # real tensors (runs the model)
    python video/make_idm_video.py       # renders this

Run `make_idm_video.py --audit-only` to check layout without encoding: it measures
every text bounding box for canvas overflow and pairwise overlap, and refuses to
encode if anything is wrong. There is no way to eyeball 15k frames, so geometry
is the gate.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
VERA = ROOT / "vera-main"
ASSETS = HERE / "assets"

W, H, FPS = 1280, 720, 25
CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
CJK_B = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
MONO_B = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"

BG = (13, 16, 21)
PANEL = (19, 24, 32)
FG = (234, 239, 246)
DIM = (141, 154, 172)
FAINT = (86, 97, 113)
ACC = (95, 178, 255)
ACC2 = (255, 172, 95)
ACC3 = (110, 214, 160)
WARN = (255, 116, 116)
VIOLET = (186, 160, 255)

_fc: dict[tuple[str, int, bool], ImageFont.FreeTypeFont] = {}
CODE_LH = 1.5          # 代码行距倍数；panel() 用它按行数算高度


def F(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    k = ("m" if mono else "c", size, bold)
    if k not in _fc:
        path = (MONO_B if bold else MONO) if mono else (CJK_B if bold else CJK)
        _fc[k] = ImageFont.truetype(path, size)
    return _fc[k]


def C(c, a=255):
    return (c[0], c[1], c[2], a)


BOXES: list = []


def layer() -> Image.Image:
    return Image.new("RGBA", (W, H), (0, 0, 0, 0))


def base(d) -> None:
    d.rectangle([0, 0, W, H], fill=C(BG))


def t(d, xy, s, size=28, color=FG, bold=False, anchor="la", track=False, mono=False):
    d.text(xy, s, font=F(size, bold, mono), fill=C(color), anchor=anchor)
    if track:
        BOXES.append((d.textbbox(xy, s, font=F(size, bold, mono), anchor=anchor), (s[:26], size)))


def wrap(d, s, size, width, bold=False):
    f = F(size, bold)
    out, cur = [], ""
    for ch in s:
        if ch == "\n":
            out.append(cur)
            cur = ""
            continue
        if d.textlength(cur + ch, font=f) > width and cur:
            if ch.isalnum() and cur[-1].isalnum() and " " in cur.strip():
                head, _, tail = cur.rpartition(" ")
                if d.textlength(head + " " + ch, font=f) <= width:
                    out.append(head)
                    cur = ch + tail
                    continue
            out.append(cur)
            cur = ch
        else:
            cur += ch
    if cur:
        out.append(cur)
    return out


def para(d, xy, s, size=24, color=DIM, width=None, lh=1.6, bold=False, track=True):
    width = width or (W - xy[0] - 80)
    for i, ln in enumerate(wrap(d, s, size, width, bold)):
        y = xy[1] + i * int(size * lh)
        d.text((xy[0], y), ln, font=F(size, bold), fill=C(color), anchor="la")
        if track:
            BOXES.append((d.textbbox((xy[0], y), ln, font=F(size, bold), anchor="la"), (ln[:26], size)))


def rr(d, box, r=12, fill=None, outline=None, width=2):
    d.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=width)


def arrow(d, p0, p1, color=DIM, w=3, h=11):
    d.line([p0, p1], fill=C(color), width=w)
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    n = max(1e-6, (dx * dx + dy * dy) ** 0.5)
    ux, uy = dx / n, dy / n
    px, py = -uy, ux
    d.polygon([p1,
               (p1[0] - ux * h + px * h * .55, p1[1] - uy * h + py * h * .55),
               (p1[0] - ux * h - px * h * .55, p1[1] - uy * h - py * h * .55)], fill=C(color))


# ---------------------------------------------------------------- 代码读取
_cache: dict[str, list[str]] = {}


def src(rel: str) -> list[str]:
    if rel not in _cache:
        _cache[rel] = (ROOT / rel).read_text().splitlines()
    return _cache[rel]


def grab(rel: str, anchor: str, before: int = 0, after: int = 12) -> tuple[str, list[str]]:
    """从真实仓库里截取包含 anchor 的代码段（找不到就报错，绝不手抄）。"""
    lines = src(rel)
    hit = None
    for i, ln in enumerate(lines):
        if anchor in ln:
            hit = i
            break
    if hit is None:
        raise SystemExit(f"代码锚点没找到: {rel} :: {anchor}")
    a = max(0, hit - before)
    b = min(len(lines), hit + after)
    return f"{rel}:{a+1}-{b}", lines[a:b]


KW = {"def", "return", "if", "else", "for", "while", "in", "not", "and", "or", "None",
      "True", "False", "import", "from", "with", "as", "class", "self", "raise", "try"}
TOKEN = re.compile(r"(#.*$)|(\"[^\"]*\"|'[^']*')|(\b\d+\.?\d*\b)|([A-Za-z_][A-Za-z_0-9]*)|(\s+)|(.)")


def code(d, xy, lines, size=17, lh=1.5, start_no=None, maxw=None):
    """逐 token 上色 + 逐字符选字体（ASCII 用等宽、CJK 用黑体），保证列对齐。

    每行都登记**实际绘制范围**（从 x0 到最后一个字符的右边界），审计才能发现
    真实的溢出/压字 —— 早先用固定 maxw 登记会让重叠面积算成整行宽，噪声极大。
    """
    x0, y0 = xy
    fm, fmb = F(size, mono=True), F(size, mono=True, bold=True)
    for k, raw in enumerate(lines):
        ln = raw.replace("\t", "    ")
        y = y0 + k * int(size * lh)
        if start_no is not None:
            d.text((x0 - 12, y), str(start_no + k), font=fm, fill=C(FAINT), anchor="ra")
        cx = x0
        for m in TOKEN.finditer(ln):
            kind, s = m.lastgroup, m.group()
            if kind == 1:
                col, bold = (94, 112, 92), False
            elif kind == 2:
                col, bold = (222, 176, 106), False
            elif kind == 3:
                col, bold = (150, 190, 255), False
            elif kind == 4:
                col, bold = (ACC if s in KW else FG), s in KW
            else:
                col, bold = FG, False
            for ch in s:
                f = (fmb if bold else fm) if ord(ch) < 128 else F(size, bold)
                d.text((cx, y), ch, font=f, fill=C(col), anchor="la")
                cx += d.textlength(ch, font=f)
        BOXES.append(((x0, y, cx, y + int(size * lh)), (f"code:{ln.strip()[:20]}", size)))


def panel(img, where, lines, x, y, w, size=17, pad=18, start_no=None, limit=None):
    """Auto-height code panel. Returns its bottom y so callers can flow below it.

    Height is derived from the line count instead of being guessed — guessed
    panel heights were the single largest source of layout errors here.
    ``limit`` is the y coordinate the panel must not cross (defaults to the
    canvas bottom minus a small margin).
    """
    limit = H - 24 if limit is None else limit
    h = 30 + len(lines) * int(size * CODE_LH) + pad
    if y + h > limit:
        raise SystemExit(f"代码面板放不下: {where} 需要 {h}px，从 y={y} 到 {limit} 只有 {limit-y}px")
    d = ImageDraw.Draw(img)
    rr(d, [x, y, x + w, y + h], 12, fill=C(PANEL), outline=C((52, 64, 84)), width=2)
    rr(d, [x, y, x + w, y + 30], 12, fill=C((27, 34, 46)))
    t(d, (x + pad, y + 5), where, 15, ACC, mono=True)
    code(d, (x + pad, y + 42), lines, size=size, start_no=start_no, maxw=w - 2 * pad)
    return y + h


def fit(lines, d, size, width):
    """挑一个能放进面板的最大字号。"""
    fm = F(size, mono=True)
    w = max((d.textlength(l.replace("\t", "    "), font=fm) for l in lines), default=0)
    while size > 10 and w > width:
        size -= 1
        fm = F(size, mono=True)
        w = max((d.textlength(l.replace("\t", "    "), font=fm) for l in lines), default=0)
    return size


# ---------------------------------------------------------------- 真实素材
def npz(name):
    return np.load(ASSETS / f"{name}.npy")


def flow_img(flow, max_mag=None):
    import matplotlib.colors as mcolors
    u, v = flow[0], flow[1]
    mag = np.sqrt(u * u + v * v)
    ang = np.arctan2(-v, -u) / np.pi
    m = mag / (max_mag or max(float(mag.max()), 1e-6))
    rgb = mcolors.hsv_to_rgb(np.stack([(ang + 1) / 2, np.ones_like(m), np.clip(m, 0, 1)], -1))
    return Image.fromarray((rgb * 255).astype(np.uint8))


def dots(x, n=7, color=(255, 255, 255), alpha=64):
    ov = Image.new("RGBA", x.size, (0, 0, 0, 0))
    dd = ImageDraw.Draw(ov)
    for i in range(n):
        for j in range(n):
            cx = (j + .5) * x.size[0] / n
            cy = (i + .5) * x.size[1] / n
            r = max(1, x.size[0] // (n * 6))
            dd.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(*color, alpha))
    return Image.alpha_composite(x.convert("RGBA"), ov)


def put(dst, img, box, r=8):
    im = img.convert("RGBA")
    m = Image.new("L", im.size, 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, im.size[0] - 1, im.size[1] - 1], radius=r, fill=255)
    dst.alpha_composite(Image.composite(im, Image.new("RGBA", im.size, (0, 0, 0, 0)), m), (box[0], box[1]))


# ---------------------------------------------------------------- 页面骨架
CHAPTER = ["开场", "第 1 章  控制什么", "第 2 章  什么是逆动力学", "第 3 章  雅可比场",
           "第 4 章  怎么训练", "第 5 章  工程踩坑", "第 6 章  推理落地",
           "第 7 章  别人会问什么", "现状"]


def page(sec: int, title: str, sub: str = "") -> Image.Image:
    img = layer()
    d = ImageDraw.Draw(img)
    base(d)
    t(d, (66, 40), CHAPTER[sec], 19, ACC, bold=True)
    t(d, (66, 72), title, 36, FG, bold=True, track=True)
    if sub:
        t(d, (66, 122), sub, 21, DIM, track=True)
    return img


def section(no: int, name: str, points: list[str]) -> Image.Image:
    img = layer()
    d = ImageDraw.Draw(img)
    base(d)
    d.rectangle([0, 0, 12, H], fill=C(ACC))
    t(d, (100, 210), no, 100, (36, 46, 62), bold=True)
    t(d, (250, 236), name, 56, FG, bold=True, track=True)
    for i, p in enumerate(points):
        y = 350 + i * 56
        d.ellipse([256, y + 10, 270, y + 24], fill=C(ACC))
        t(d, (292, y), p, 25, DIM, track=True)
    return img


# ================================================================ 场景
def s_title():
    img = layer(); d = ImageDraw.Draw(img); base(d)
    d.rectangle([0, 0, 14, H], fill=C(ACC))
    t(d, (96, 150), "VERA 的 IDM", 76, FG, bold=True)
    t(d, (96, 246), "从零讲透", 76, ACC, bold=True)
    t(d, (100, 360), "原理 → 代码 → 我们项目里怎么做的、为什么这么做", 27, DIM)
    rr(d, [96, 424, 700, 476], 12, fill=C(ACC, 22), outline=C(ACC, 140), width=2)
    t(d, (122, 438), "看完能讲给别人听 · 全部素材来自本项目真实运行", 22, ACC)
    t(d, (96, 560), "SO-ARM101 · 2×RTX 3090 · VERA video-to-action", 19, FAINT)
    t(d, (96, 600), "配套代码：vera-main/vera/idm/jacobian/ · vera-main/vera/datasets/core/", 19, FAINT)
    return img


def s_roadmap():
    img = page(0, "这份视频怎么用")
    para(img and ImageDraw.Draw(img), (66, 170),
         "这是一堂课，不是摘要。每个概念都按「先讲清楚它是什么 → 再看它在仓库里对应哪几行代码 → "
         "最后说我们为什么这么选」的顺序讲。", 24, DIM, 1140)
    boxes = [("第 1 章", "我们要控制什么", ACC), ("第 2 章", "什么是逆动力学", ACC),
             ("第 3 章", "雅可比场（核心）", ACC2), ("第 4 章", "怎么训练", ACC2),
             ("第 5 章", "工程踩过的坑", WARN), ("第 6 章", "推理怎么落地", ACC3),
             ("第 7 章", "别人会问什么", VIOLET)]
    for i, (n, s, c) in enumerate(boxes):
        x = 66 + (i % 4) * 296
        y = 300 + (i // 4) * 128
        rr(ImageDraw.Draw(img), [x, y, x + 268, y + 104], 14, fill=C(c, 18), outline=C(c, 130), width=2)
        t(ImageDraw.Draw(img), (x + 20, y + 16), n, 20, c, bold=True)
        t(ImageDraw.Draw(img), (x + 20, y + 48), s, 23, FG, bold=True)
    para(ImageDraw.Draw(img), (66, 590),
         "第 3 章是整篇的心脏。看完第 7 章，你应该能回答「IDM 到底在干嘛」这类问题。", 23, ACC3, 1140)
    return img


def s_ch1_intro():
    img = page(1, "1.1  我们要控制的东西：6 个数字")
    st = npz("du")
    rgb = Image.fromarray((npz("rgb")[4].transpose(1, 2, 0) * 255).astype(np.uint8))
    put(img, dots(rgb, 7), (66, 180, 66 + 210, 180 + 118))
    t(ImageDraw.Draw(img), (66, 306), "手腕相机看到的一帧（真实数据）", 18, FAINT)
    para(ImageDraw.Draw(img), (320, 178),
         "这条机械臂有 5 个可转的关节 + 1 个夹爪，所以「它现在的姿态」就是 6 个数字。"
         "遥操作时你手里那根主臂也是这 6 个数。", 23, DIM, 620)
    rr(ImageDraw.Draw(img), [320, 300, 1210, 470], 14, fill=C(PANEL), outline=C((52, 64, 84)), width=2)
    t(ImageDraw.Draw(img), (348, 322), "每一帧记录下来的 6 个数（action / observation.state）", 22, FG, bold=True)
    for i, (n, e) in enumerate(zip(["shoulder_pan", "shoulder_lift", "elbow_flex",
                                    "wrist_flex", "wrist_roll", "gripper"], [1, 2, 3, 4, 5, 6])):
        x = 348 + i * 140
        rr(ImageDraw.Draw(img), [x, 366, x + 124, 452], 10, fill=C((26, 33, 44)))
        t(ImageDraw.Draw(img), (x + 62, 380), f"ID {e}", 17, FAINT, anchor="ma")
        t(ImageDraw.Draw(img), (x + 62, 408), n.split("_")[-1] if i < 5 else "夹爪", 17, ACC, bold=True, anchor="ma")
    para(ImageDraw.Draw(img), (66, 500),
         "注意量纲不统一：5 个关节在 lerobot 里是「度」，夹爪是 0–100 的百分比。"
         "这个细节后面会咬人（第 5 章）。", 23, WARN, 1140)
    para(ImageDraw.Draw(img), (66, 590),
         "我们不输出这 6 个数本身，而是输出「它们该怎么变」——一个 6 维增量 du。", 24, ACC, 1140)
    return img


def s_ch1_du():
    img = page(1, "1.2  输出增量 du，而不是绝对角度")
    where, lines = grab("vera-main/vera/datasets/core/actions.py", "du = (q1 - q0)", before=8, after=2)
    d = ImageDraw.Draw(img)
    b = panel(img, where, lines, 66, 178, 1144)
    para(d, (66, b + 26), "训练时不需要任何动作文件：du 由相邻两帧的关节角差分得到，"
                          "所以每条遥操作轨迹自带监督信号。", 23, DIM, 1140)
    rr(d, [66, b + 92, 1210, b + 208], 14, fill=C(ACC3, 14), outline=C(ACC3, 130), width=2)
    t(d, (96, b + 112), "为什么用增量而不是绝对角度？", 25, FG, bold=True)
    t(d, (96, b + 156), "绝对角度会把「起始姿态」这个无关信息也编码进去；增量只关心「该怎么动」。",
      21, DIM)
    return img


def s_ch1_data():
    img = page(1, "1.3  手上的数据：13 集、够干什么")
    st = npz("du")
    d = ImageDraw.Draw(img)
    facts = [("13", "集遥操作演示", ACC), ("10.5", "分钟有效画面", ACC),
             ("128×128", "训练分辨率", ACC2), ("39/39", "帧校验对齐", ACC3)]
    for i, (big, small, c) in enumerate(facts):
        x = 66 + i * 296
        rr(d, [x, 180, x + 268, 300], 14, fill=C(c, 16), outline=C(c, 130), width=2)
        t(d, (x + 134, 200), big, 44, c, bold=True, anchor="ma")
        t(d, (x + 134, 258), small, 20, DIM, anchor="ma")
    para(d, (66, 340),
         "13 集不足以让模型学会「叠方块」这个任务，但足够让它学会一件与任务无关的事："
         "这台机械臂的关节一动，画面会怎么变。", 25, FG, 1140, bold=True)
    para(d, (66, 486),
         "这就是 VERA 把问题拆成两段的原因 —— 任务知识放在 planner（跨具身、用别人的大数据训），"
         "而 IDM 只负责一件机械上很具体的事。", 23, DIM, 1140)
    rr(d, [66, 540, 1210, 650], 14, fill=C(PANEL), outline=C((52, 64, 84)), width=2)
    t(d, (96, 562), "类比", 22, ACC, bold=True)
    t(d, (96, 600), "就像学会「手怎么动」和学会「把杯子放到盘子边」是两件事 —— 前者一次学会就一直能用。", 21, DIM)
    return img


def s_ch2_fwd():
    img = page(2, "2.1  正向动力学：最经典的定义")
    d = ImageDraw.Draw(img)
    rr(d, [66, 175, 600, 330], 14, fill=C(ACC, 16), outline=C(ACC, 140), width=2)
    t(d, (96, 196), "正向动力学（经典机器人学）", 24, ACC, bold=True)
    para(d, (96, 250), "已知每个关节转了多少 → 算出机械臂末端在空间里的位置。", 23, FG, 460)
    rr(d, [680, 175, 1214, 330], 14, fill=C(ACC2, 16), outline=C(ACC2, 140), width=2)
    t(d, (710, 196), "逆向动力学（IDM）", 24, ACC2, bold=True)
    para(d, (710, 250), "看到末端在哪儿 → 反推每个关节各转了多少。", 23, FG, 460)
    para(d, (66, 370),
         "在真实机器人上，正向可以直接用运动学公式算出来；反过来却算不出来 —— "
         "因为末端位置对 6 个关节的映射是多对一的（手腕转 + 肘转可能得到同样的位置）。", 24, DIM, 1140)
    rr(d, [66, 500, 1210, 640], 14, fill=C(PANEL), outline=C((52, 64, 84)), width=2)
    t(d, (96, 522), "为什么机器人学里逆动力学很难、但这里反而容易？", 24, FG, bold=True)
    para(d, (96, 572),
         "因为我们不在关节空间里求解，而在「画面」里求解。画面给了几万个像素的约束，"
         "把多解问题变成了超定问题。", 22, ACC3, 1080)
    return img



def s_ch2_flow():
    img = page(2, "2.2  换个地方求解：像素也会动")
    d = ImageDraw.Draw(img)
    gt = npz("gt_flow")
    rgb = Image.fromarray((npz("rgb")[4].transpose(1, 2, 0) * 255).astype(np.uint8))
    put(img, dots(rgb, 7), (66, 190, 276, 325))
    t(d, (66, 332), "第 4 帧画面", 18, FAINT)
    put(img, dots(flow_img(gt[4]), 7), (296, 190, 506, 325))
    t(d, (296, 332), "同两帧之间的光流（真实 MegaFlow）", 18, FAINT)
    where, lines = grab("vera-main/vera/idm/jacobian/models/vggt_jacobian_field.py",
                        "flow = einsum(", before=1, after=3)
    b = panel(img, where, lines, 66, 400, 1144)
    t(d, (600, 186), "光流（optical flow）= 相邻两帧之间", 23, FG, bold=True)
    t(d, (600, 226), "每个像素移动了多少。", 23, FG, bold=True)
    para(d, (600, 276), "它只看像素自身的位移，不看「这是什么物体」：手往左移、画面整体变亮，"
                       "都只是像素位移。", 21, DIM, 600)
    t(d, (66, b + 24), "于是「关节动了多少」和「画面动了多少」之间存在一张映射表 —— 我们要学的就是它。",
      22, ACC)
    return img


def s_ch2_sup():
    img = page(2, "2.3  为什么在画面里解就容易了")
    d = ImageDraw.Draw(img)
    rr(d, [66, 180, 1210, 330], 14, fill=C(ACC2, 14), outline=C(ACC2, 140), width=2)
    t(d, (640, 208), "128 × 128 × 2  =  32 768 个方程", 40, FG, bold=True, anchor="ma")
    t(d, (640, 272), "去解        6 个未知数", 30, ACC2, bold=True, anchor="ma")
    para(d, (66, 370),
         "每条训练数据都变成一次极其「超定」的测量：几万个像素同时投票决定 6 个关节该动多少。"
         "就算某些像素看不清、某些区域被手臂挡住，剩下的像素仍然足以给出答案。", 24, DIM, 1140)
    para(d, (66, 500),
         "对比一下：如果直接学「画面 → 动作」，监督信号只有 6 个数，网络要同时理解手在哪、"
         "抓没抓住、物体往哪走 —— 数据效率极低。", 24, DIM, 1140)
    rr(d, [66, 590, 1210, 664], 14, fill=C(ACC3, 14), outline=C(ACC3, 130), width=2)
    t(d, (96, 612), "关键转向：不学「动作是什么」，而学「动作对画面做了什么」。", 25, FG, bold=True)
    return img


def s_ch3_def():
    img = page(3, "3.1  雅可比：先从一维的「斜率」说起")
    d = ImageDraw.Draw(img)
    para(d, (66, 175), "雅可比就是「局部斜率」的高维版本：", 25, FG, 1140, bold=True)
    para(d, (66, 245), "一维里，f′(x) 告诉你「x 动一点点，f 会变多少」。", 23, DIM, 1140)
    gx, gy, gw, gh = 120, 320, 460, 210
    rr(d, [gx, gy, gx + gw, gy + gh], 10, fill=C(PANEL), outline=C((52, 64, 84)), width=2)
    pts = [(gx + 40 + i * (gw - 80) / 9, gy + 30 + 150 * (0.5 + 0.45 * np.sin(i / 9 * 3.0)) / 1.0)
           for i in range(10)]
    d.line(pts, fill=C(ACC), width=4)
    a, b = pts[2], pts[6]
    d.line([a, b], fill=C(ACC2, 110), width=2)
    a2 = (a[0], a[1]); b2 = (b[0], a[1])
    d.line([a2, b2], fill=C(ACC2), width=3)
    t(d, (a2[0], a2[1] - 34), "Δx", 20, ACC2, bold=True)
    t(d, (b2[0] + 10, a2[1] - 14), "Δf(x)", 20, ACC2, bold=True)
    t(d, (gx + gw / 2, gy + gh - 30), "曲线在某点的斜率 = 那一小段的 Δf / Δx", 19, FAINT, anchor="ma")
    para(d, (640, 320),
         "放到我们的问题里：\n\n"
         "「关节角」是自变量，\n「每个像素的位移」是因变量。\n\n"
         "雅可比 J[i, p] 就是：第 i 个关节动 1 度，像素 p 会移动多少（一个二维小向量）。", 23, FG, 540)
    rr(d, [640, 520, 1214, 640], 14, fill=C(PANEL), outline=C((52, 64, 84)), width=2)
    t(d, (668, 542), "形状 = [6 个关节, 2 个方向, 128, 128]", 23, ACC2, bold=True)
    t(d, (668, 584), "一共 6 × 2 × 128 × 128 ≈ 19.7 万个数，非常稠密", 20, DIM)
    return img


def s_ch3_real():
    img = page(3, "3.2  这就是模型真学出来的 J")
    J = npz("jacobian")[0]
    sz, gap = 186, 14
    x0 = (W - (6 * sz + 5 * gap)) // 2
    y0 = 178
    import matplotlib.cm as cm
    mag = float(np.abs(J).max())
    d = ImageDraw.Draw(img)
    for i in range(6):
        m = np.clip(np.abs(J[i, 0]) / max(mag, 1e-9), 0, 1)
        tile = Image.fromarray((cm.inferno(m)[:, :, :3] * 255).astype(np.uint8)).resize((sz, sz), Image.NEAREST)
        x = x0 + i * (sz + gap)
        rr(d, [x - 3, y0 - 3, x + sz + 3, y0 + sz + 3], 10, outline=C(ACC, 120), width=2)
        put(img, tile, (x, y0), r=6)
        t(d, (x + sz // 2, y0 + sz + 16), ["肩旋转", "肩俯仰", "肘", "腕俯仰", "腕旋转", "夹爪"][i], 21, FG, bold=True, anchor="ma")
        t(d, (x + sz // 2, y0 + sz + 44), f"J[{i}]  ∂u/∂q", 16, FAINT, mono=True, anchor="ma")
    para(d, (66, y0 + sz + 86),
         "颜色越亮 = 该关节动一度，这个像素在水平方向移动越多。这六张图全部由「用真实动作解释真实光流」"
         "这一个损失函数学出来，没有任何人工标注。", 22, DIM, 1140)
    rr(d, [66, 590, 1210, 664], 14, fill=C(ACC3, 14), outline=C(ACC3, 130), width=2)
    t(d, (96, 612), "注意各关节的图案完全不同 —— 这就是「谁影响哪些像素」，模型自己分清楚了。", 23, FG, bold=True)
    return img


def s_ch3_why():
    img = page(3, "3.3  为什么学 J 比学动作划算")
    d = ImageDraw.Draw(img)
    J = npz("jacobian")[0]
    mags = np.abs(J[:, 0]).reshape(6, -1).mean(1)
    order = np.argsort(-mags)
    rr(d, [66, 180, 600, 480], 14, fill=C(PANEL), outline=C((52, 64, 84)), width=2)
    t(d, (96, 200), "本模型里每个关节的平均 |J|", 22, FG, bold=True)
    for r, i in enumerate(order):
        y = 250 + r * 38
        t(d, (96, y), ["肩旋转", "肩俯仰", "肘", "腕俯仰", "腕旋转", "夹爪"][i], 20, FG)
        t(d, (200, y), f"J[{i}]", 18, FAINT, mono=True)
        bw = 250 * float(mags[i]) / float(mags.max())
        rr(d, [260, y + 4, 260 + max(5, bw), y + 22], 7, fill=C(ACC2 if r == 0 else ACC, 200))
        t(d, (530, y), f"{mags[i]:.2f}", 19, DIM)
    para(d, (640, 180),
         "三件事让 J 特别好学：\n\n"
         "① 它是局部的 —— 一个关节只影响画面上一小片区域，不用理解整个场景。\n\n"
         "② 它是光滑的 —— 相邻帧的 J 几乎一样，所以每条轨迹提供了大量相似的样本。\n\n"
         "③ 它与任务无关 —— 叠方块和拧螺丝用的是同一张 J。", 22, DIM, 560)
    rr(d, [66, 510, 1210, 650], 14, fill=C(ACC2, 14), outline=C(ACC2, 130), width=2)
    t(d, (96, 532), "这就是数据效率的来源", 24, FG, bold=True)
    para(d, (96, 578),
         "任务知识留在 planner 那一头（它见过成千上万条演示）。IDM 只需要学会「这台机器人自己的运动学」，"
         "13 集就够起步。", 21, DIM, 1080)
    return img



def s_ch4_fwd():
    img = page(4, "4.1  正向：一行 einsum 就是全部")
    where, lines = grab("vera-main/vera/idm/jacobian/models/vggt_jacobian_field.py",
                        "flow = einsum(", before=5, after=4)
    d = ImageDraw.Draw(img)
    b = panel(img, where, lines, 66, 176, 1144)
    rr(d, [66, b + 26, 1210, b + 140], 14, fill=C(ACC, 14), outline=C(ACC, 130), width=2)
    t(d, (640, b + 46), "predicted_flow = J · du", 34, FG, bold=True, anchor="ma")
    para(d, (100, b + 104), "给定雅可比场，动作 du 进去，光流出来 —— 对 du 是线性的。", 22, DIM, 1080)
    para(d, (66, b + 176), "训练时不需要反解：数据里 du 是已知的（相邻帧关节角差分），"
                           "只要让 J·du 逼近实际观测到的光流。", 23, DIM, 1140)
    return img



def s_ch4_loss():
    img = page(4, "4.2  损失函数：为什么不是 MSE")
    where, lines = grab("vera-main/vera/idm/jacobian/image_jacobian.py",
                        'if self.cfg.flow_loss == "mse"', before=1, after=7)
    d = ImageDraw.Draw(img)
    b = panel(img, where, lines, 66, 176, 1144)
    para(d, (66, b + 22),
         "Charbonnier = √(差² + ε²)。和 MSE 的区别在小误差处：MSE 会被极小误差惩罚得极狠，"
         "而这里 ε 相当于一个死区 —— 差一点点基本不罚，模型能把精力花在大误差上。", 22, DIM, 1140)
    where2, lines2 = grab("vera-main/vera/configurations/algorithm/model/soarm_vggt_jacobian.yaml",
                           "command_dim:", before=1, after=3)
    panel(img, where2, lines2, 66, b + 104, 660)
    t(d, (770, b + 128), "动作维度只在这个文件里出现，", 21, ACC2)
    t(d, (770, b + 162), "不会从数据推断 ——", 21, ACC2)
    t(d, (770, b + 196), "换机械臂就必��改它。", 21, ACC2)
    return img



def s_ch4_norm():
    img = page(4, "4.3  归一化：必须按训练分辨率统计")
    d = ImageDraw.Draw(img)
    rr(d, [66, 176, 600, 322], 14, fill=C(WARN, 14), outline=C(WARN, 140), width=2)
    t(d, (96, 196), "坑", 22, WARN, bold=True)
    para(d, (96, 240), "光流以「像素」为单位。我们存 256×256，但训练时缩放到 128×128 —— "
                       "同一物理位移，像素数差一倍。", 21, FG, 470)
    rr(d, [640, 176, 1214, 322], 14, fill=C(ACC3, 14), outline=C(ACC3, 140), width=2)
    t(d, (670, 196), "所以", 22, ACC3, bold=True)
    para(d, (670, 240), "归一化常数必须用「训练分辨率下量出来的」统计，否则目标值差一倍，"
                        "模型学到的尺度就是错的。", 21, FG, 510)
    w1, l1 = grab("vera-main/vera/configurations/dataset/soarm_packed.yaml",
                  "oflow_abs_scale:", before=1, after=3)
    b1 = panel(img, w1, l1, 66, 352, 560)
    w2, l2 = grab("vera-main/vera/configurations/dataset/soarm_packed.yaml",
                  "action_abs_scale:", before=1, after=7)
    b2 = panel(img, w2, l2, 664, 352, 550)
    low = max(b1, b2) + 16
    t(d, (66, low), "实测值（13 集、4399 步）", 20, ACC)
    para(d, (66, low + 34), "6 个通道量纲不同：5 个关节是「度」、夹爪是 0–100。共用一个常数会让夹爪"
                           "那一维几乎不参与训练 —— 所以必须逐通道。", 21, DIM, 1140)
    return img


def s_ch4_reg():
    img = page(4, "4.4  四个正则项，各治一种病")
    where, lines = grab("vera-main/vera/idm/jacobian/image_jacobian.py", 'losses["flow_gradient"]', before=2, after=12)
    d = ImageDraw.Draw(img)
    panel(img, where, lines, 66, 176, 1144)
    items = [("flow_gradient", "让预测光流的梯度对齐真值的梯度 —— 保住边缘结构，防止 J 被抹平", ACC),
             ("jacobian_tv", "边缘感知 TV，直接罚 J 上的孤立大值 —— 专治「静态区域里的孤立高响应」", ACC2),
             ("flow_aleatoric", "逐像素置信度：模型自动在光流不可预测的地方降低自信", ACC3),
             ("view/motion 平衡", "按真正在动的像素加权 —— 否则大面积静止区域会稀释损失", VIOLET)]
    for i, (k, v, c) in enumerate(items):
        y = 405 + i * 66
        rr(d, [66, y, 1210, y + 56], 10, fill=C(c, 14), outline=C(c, 110), width=2)
        t(d, (92, y + 14), k, 21, c, bold=True, mono=True)
        t(d, (300, y + 16), v, 20, FG)
    return img



def s_ch4_inv():
    img = page(4, "4.5  反向：推理时必须反解")
    d = ImageDraw.Draw(img)
    rr(d, [66, 176, 1210, 302], 14, fill=C(ACC2, 14), outline=C(ACC2, 150), width=2)
    t(d, (640, 198), "du = argmin ‖ J·du − flow* ‖²", 34, FG, bold=True, anchor="ma")
    t(d, (640, 252), "闭式解：  du = (JᵀJ + λI)⁻¹ Jᵀ flow*", 24, ACC2, anchor="ma")
    where, lines = grab("vera-main/vera/idm/jacobian/image_jacobian.py",
                        "ridge = (", before=5, after=3)
    b = panel(img, where, lines, 66, 332, 1144)
    para(d, (66, b + 22), "λ 是阻尼（ridge）：当某些区域的 J 几乎线性相关时，纯解会爆炸；"
                          "加上 λ 后解变得保守稳定。", 22, DIM, 1140)
    para(d, (66, b + 78), "整段在 float32 里算 —— autocast 在这里被显式关掉：这是求逆，"
                          "数值稳定性优先于速度。", 22, DIM, 1140)
    return img



def s_ch4_inv_train():
    img = page(4, "4.6  反解也要训练（最容易被忽略）")
    where, lines = grab("vera-main/vera/idm/jacobian/image_jacobian.py",
                        "loss = float(self.cfg.inverse_action_weight)", before=4, after=4)
    d = ImageDraw.Draw(img)
    b = panel(img, where, lines, 66, 176, 1144)
    para(d, (66, b + 22), "训练时把上面那个求逆也跑一遍，得到 du_hat，再和数据里真实的 du 做回归"
                          "损失（我们的权重 0.1）。", 22, DIM, 1140)
    rr(d, [66, b + 86, 1210, b + 292], 14, fill=C(PANEL), outline=C((52, 64, 84)), width=2)
    t(d, (96, b + 106), "为什么必须这样？", 24, FG, bold=True)
    para(d, (96, b + 154), "只用「正向拟合光流」的话，J 会有大量简并解：一组完全不同的 J 可以产生"
                           "相同的 J·du，正向损失查不出来，但反解时会给出错误答案。", 21, DIM, 1080)
    t(d, (96, b + 256), "加了这个损失，等于额外要求「J 必须能被反解回 du」。", 21, ACC3)
    return img


def s_ch5_pipeline():
    img = page(5, "5.1  数据链路：为什么中间必须有「打包」这一步")
    d = ImageDraw.Draw(img)
    nodes = [("lerobot 数据集", "parquet + mp4\n13 集实拍", ACC),
             ("pack_lerobot.py", "切帧 → JPEG\nMegaFlow 光流\ntraj_state", ACC2),
             ("okto_packed", "一集一个 NPZ\n664 MB", ACC2),
             ("IDM 训练", "直接读 NPZ\n解码即用", ACC3)]
    for i, (t1, t2, c) in enumerate(nodes):
        x = 66 + i * 296
        rr(d, [x, 190, x + 250, 360], 14, fill=C(c, 16), outline=C(c, 140), width=2)
        t(d, (x + 125, 212), t1, 22, c, bold=True, anchor="ma")
        for j, ln in enumerate(t2.split("\n")):
            t(d, (x + 125, 258 + j * 28), ln, 18, DIM, anchor="ma")
        if i < 3:
            arrow(d, (x + 256, 275), (x + 290, 275), DIM, 3, 10)
    rr(d, [66, 395, 1210, 500], 14, fill=C(WARN, 12), outline=C(WARN, 130), width=2)
    t(d, (96, 415), "不能跳过打包", 23, WARN, bold=True)
    para(d, (96, 458), "仓库里光流只在「packed」这条读取路径上实现；直接读 mp4 的那条路径 load_flow 直接返回 None，"
                       "IDM 训练硬编码需要光流。所以打包不是可选优化，是必经步骤。", 21, FG, 1080)
    where, lines = grab("vera-main/vera/datasets/core/view_loader.py", "def load_flow", after=2)
    rr(d, [66, 530, 1210, 664], 14, fill=C(PANEL), outline=C((52, 64, 84)), width=2)
    t(d, (92, 548), "对照：另一条读取路径（原始 mp4）", 20, WARN)
    t(d, (92, 584), "DroidViewLoader.load_flow  →  return None", 22, FG, mono=True)
    t(d, (92, 626), "只有 PackedViewLoader 真正解码 qint8 光流 → 所以必须打包", 20, DIM)
    return img



def s_ch5_bug1():
    img = page(5, "5.2  坑一：多集 mp4 切片会「静默」写错帧")
    where, lines = grab("vera-main/scripts/data/pack_lerobot.py",
                        "self._streams[key] = (container, frames, pos)", before=6, after=3)
    d = ImageDraw.Draw(img)
    para(d, (66, 172), "lerobot 的一个 mp4 里塞了多集。整文件读不现实（第一个文件 8942 帧、"
                       "uint8 约 25 GB），所以按集用绝对帧号流式切片。", 22, DIM, 1140)
    b = panel(img, where, lines, 66, 258, 1144)
    rr(d, [66, b + 24, 1210, b + 160], 14, fill=C(WARN, 12), outline=C(WARN, 140), width=2)
    t(d, (96, b + 44), "真实发生过的事故", 23, WARN, bold=True)
    para(d, (96, b + 86), "位置计数器只写在局部变量、没回写缓存 → 计数器和解码迭代器脱节 → "
                          "13 集里有 4 集存的是别的集的画面，而且不报任何错，能正常加载。", 21, FG, 1080)
    return img



def s_ch5_bug2():
    img = page(5, "5.3  坑二：校验不能用「绝对差值」阈值")
    where, lines = grab("vera-main/scripts/data/check_packed.py",
                        "at0 = next(", before=3, after=6)
    d = ImageDraw.Draw(img)
    para(d, (66, 172), "打包存的是 JPEG，解出来和原始像素天然差 ~1/255；而「错一帧」的差值在手腕"
                       "相机上只有 ~1.07 —— 两者分不开，绝对阈值形同虚设。", 22, DIM, 1140)
    b = panel(img, where, lines, 66, 268, 1144)
    rr(d, [66, b + 24, 1210, b + 148], 14, fill=C(ACC3, 14), outline=C(ACC3, 130), width=2)
    t(d, (96, b + 44), "正确的判据", 23, FG, bold=True)
    para(d, (96, b + 84), "搜索 ±3 帧偏移，要求「偏移 0 严格最优」。当前结果：39/39 次帧比对全部对齐，"
                          "13 集轨迹与原始 parquet 逐位一致。", 21, FG, 1080)
    return img


def s_ch5_bug3():
    img = page(5, "5.4  坑三：第二个视角的光流是噪声")
    gt = npz("gt_flow")
    d = ImageDraw.Draw(img)
    sz = 148
    for r, fi in enumerate([1, 3, 5]):
        put(img, flow_img(gt[fi]), (66, 180 + r * 168, 66 + sz, 180 + r * 168 + sz))
    t(d, (66, 690), "手腕相机：颜色=方向，明暗=大小", 19, FAINT)
    bx = 300
    rr(d, [bx, 180, 1214, 640], 14, fill=C((255, 255, 255), 6), outline=C((52, 64, 84)), width=2)
    t(d, (bx + 28, 202), "两个视角的信号量（本项目实测）", 24, FG, bold=True)
    for i, (n, m, p, c) in enumerate([("手腕视角  desk", 5.38, "73%", ACC3),
                                      ("固定机位  wide", 0.05, "1%", WARN)]):
        y = 262 + i * 156
        t(d, (bx + 28, y), n, 23, FG)
        t(d, (bx + 28, y + 36), f"平均 |光流| = {m:.2f} px", 20, DIM)
        rr(d, [bx + 28, y + 70, bx + 28 + 540, y + 98], 8, fill=C((255, 255, 255), 10))
        rr(d, [bx + 28, y + 70, bx + 28 + max(6, 540 * m / 6.0), y + 98], 8, fill=C(c, 205))
        t(d, (bx + 28, y + 110), f"{p} 的像素超过 0.5 px", 20, DIM)
    para(d, (bx + 28, 566), "固定机位里方块只有约 11 像素宽，0.1 秒的位移远小于一个像素。", 21, WARN, 860)
    para(d, (bx + 28, 602), "那不是监督信号，是噪声 —— 所以只用手腕视角。", 21, WARN, 860)
    return img


def s_ch5_bug4():
    img = page(5, "5.5  坑四：显存 / worker / 量纲")
    d = ImageDraw.Draw(img)
    cards = [("显存", "12.6 亿参数不冻结时，AdamW 光是优化器状态就要约 20 GB，3090 上 batch 2 即 OOM。"
                      "→ 冻结骨干，只训 3280 万参数的解码器，batch 4 轻松。", WARN),
             ("DataLoader", "worker 默认 fork，会继承父进程已初始化的 CUDA 上下文，worker 里任何 CUDA 调用都报 "
                            "initialization error，而且只在训练结束后的拆解阶段才暴露。→ 改 spawn。", ACC2),
             ("量纲", "5 个关节是「度」、夹爪是 0–100。共用一个归一化常数会让夹爪那一维几乎不参与训练。"
                      "→ 逐通道 action_abs_scale。", ACC3)]
    for i, (k, v, c) in enumerate(cards):
        y = 180 + i * 165
        rr(d, [66, y, 1210, y + 148], 14, fill=C(c, 13), outline=C(c, 130), width=2)
        t(d, (96, y + 18), k, 24, c, bold=True)
        para(d, (96, y + 60), v, 21, FG, 1080)
    return img


def s_ch6_chain():
    img = page(6, "6.1  推理时的完整链条")
    steps = [("当前画面", ACC), ("planner\n幻想未来帧", ACC), ("算光流\nflow*", ACC2),
             ("IDM 反解\ndu", ACC2), ("积分式\n下发", ACC3), ("机械臂动", ACC3)]
    bw, gap, y0 = 178, 24, 180
    d = ImageDraw.Draw(img)
    for i, (s, c) in enumerate(steps):
        x = 56 + i * (bw + gap)
        rr(d, [x, y0, x + bw, y0 + 120], 12, fill=C(c, 18), outline=C(c, 150), width=2)
        lines = s.split("\n")
        for j, ln in enumerate(lines):
            t(d, (x + bw // 2, y0 + 30 + j * 32), ln, 20, c, bold=True, anchor="ma")
        if i < len(steps) - 1:
            arrow(d, (x + bw + 3, y0 + 60), (x + bw + gap - 3, y0 + 60), DIM, 3, 9)
    rr(d, [66, 340, 1210, 420], 14, fill=C(PANEL), outline=C((52, 64, 84)), width=2)
    t(d, (96, 362), "前三步是 VERA 提供的；后三步要我们自己写 —— 这就是第 7 章「还没做」的部分。", 22, FG)
    rr(d, [66, 450, 1210, 664], 14, fill=C(WARN, 12), outline=C(WARN, 140), width=2)
    t(d, (96, 472), "最容易踩的坑：不能每周期写 Goal = 上次Goal + du", 25, FG, bold=True)
    para(d, (96, 520),
         "实测：带负载时舵机停在离命令值约 30 刻度（2.6°）的地方。每周期只推一小步，"
         "这一步整个落在静差里 → 舵机认为「已经到了」→ 关节纹丝不动。", 21, DIM, 1080)
    where, lines = grab("soarm/align.py", "def advance", after=16)
    t(d, (96, 606), "正确做法（soarm/align.py：命令值持续朝目标累积）", 19, ACC2)
    return img



def s_ch6_servo():
    img = page(6, "6.2  积分式下发：真实代码")
    where, lines = grab("soarm/align.py", "goal = goal + max(", before=6, after=4)
    d = ImageDraw.Draw(img)
    b = panel(img, where, lines, 66, 176, 1144)
    para(d, (66, b + 26), "命令值持续朝目标累积，直到实测位置追上 —— 而不是每步只推一点点然后重算。"
                          "这是我们 soarm 工具包和 lerobot 的 max_relative_target 最实质的区别。", 22, DIM, 1140)
    return img


def s_ch7_qa():
    img = page(7, "7.1  别人可能会问的（1/2）")
    d = ImageDraw.Draw(img)
    qa = [("「IDM 到底在干嘛？」", "学一张雅可比场 J：每个关节动一下，每个像素会移动多少。然后用它把「想要的画面」反解成关节增量。"),
          ("「它和普通 inverse dynamics model 有啥区别？」", "经典 IDM 在关节空间里从状态差反推动作，多解、少样本；这个在像素空间里做，几万个方程解 6 个未知数，超定所以稳。"),
          ("「为什么不用普通神经网络直接输出动作？」", "那样监督信号只有 6 个数；用光流做监督，同样的数据变成 3 万多个方程，数据效率完全不同。"),
          ("「wrist_roll 的零点是干嘛的？」", "它是唯一没有行程可测的关节，零点只能取写入那一刻的朝向。我们实测两条臂差了 19°，表现为采数据时 observation 提前饱和。")]
    for i, (q, a) in enumerate(qa):
        y = 176 + i * 128
        t(d, (66, y), "Q" + str(i + 1), 22, ACC2, bold=True)
        t(d, (110, y), q, 23, FG, bold=True)
        para(d, (110, y + 36), a, 20, DIM, 1080)
    return img


def s_ch7_qa2():
    img = page(7, "7.2  别人可能会问的（2/2）")
    d = ImageDraw.Draw(img)
    qa = [("「为什么只用一个相机？」", "固定机位的方块只有约 11 像素宽，0.1 秒位移远小于一个像素，光流实测只有 1% 的像素超过 0.5 px —— 是噪声不是信号。数据量上来后可以重新评估。"),
          ("「13 集够吗？」", "够验证管线、够训出 J，但不够让策略真正会做任务。PushT 的参照是 206 集，建议补到 50–100 集。"),
          ("「训练时冻结了骨干，为什么？」", "12.6 亿参数不冻结时 AdamW 的优化器状态就要约 20 GB，3090 放不下。冻结后只训解码器（3280 万参数）。"),
          ("「整个东西什么时候能真的让机械臂动起来？」", "还差两步：把 soarm 注册进服务端的 embodiment 适配表，以及写机械臂侧的 backend。数据这一侧已经通了。")]
    for i, (q, a) in enumerate(qa):
        y = 176 + i * 128
        t(d, (66, y), "Q" + str(i + 5), 22, ACC2, bold=True)
        t(d, (110, y), q, 23, FG, bold=True)
        para(d, (110, y + 36), a, 20, DIM, 1080)
    return img


def s_end():
    img = page(8, "现状与下一步")
    d = ImageDraw.Draw(img)
    rows = [("数据", "13 集已录并打包；39/39 帧校验对齐；轨迹逐位一致", True),
            ("训练链路", "配置就绪，单卡 3.4 it/s，过拟合自检损失下降", True),
            ("理解", "这份视频 + 三个脚本（打包 / 校验 / 过拟合自检）", True),
            ("服务端", "还没做：把 soarm 加进 embodiment 适配表", False),
            ("机械臂侧", "还没做：写 SoArmBackend（读状态 / 发指令）", False),
            ("数据量", "建议补到 50–100 集再做正式训练", False)]
    for i, (k, v, done) in enumerate(rows):
        y = 190 + i * 74
        c = ACC3 if done else DIM
        d.ellipse([70, y + 9, 86, y + 25], fill=C(c))
        t(d, (110, y), k, 25, FG, bold=True)
        t(d, (360, y + 2), v, 22, ACC if done else DIM)
    rr(d, [66, 630, 1210, 690], 12, fill=C(PANEL), outline=C((52, 64, 84)), width=2)
    t(d, (96, 648), "配套脚本：scripts/data/pack_lerobot.py · check_packed.py · overfit_check.py · video/", 20, ACC, mono=True)
    return img


# ================================================================ 场景表
SECTIONS = [
    ("s0", "开场", lambda: section("0", "开场", ["这份视频讲什么", "怎么配合代码看", "素材从哪来"]), 7.0),
    ("s0", "开场", s_roadmap, 13.0),
    ("s1", "第 1 章", lambda: section("1", "控制什么", ["机械臂的 6 个数字", "输出增量而不是绝对角度", "手上的数据够干什么"]), 7.0),
    ("s1", "第 1 章", s_ch1_intro, 15.0),
    ("s1", "第 1 章", s_ch1_du, 16.0),
    ("s1", "第 1 章", s_ch1_data, 15.0),
    ("s2", "第 2 章", lambda: section("2", "什么是逆动力学", ["正向 vs 逆向", "把求解换个地方：像素也会动", "为什么换完就容易了"]), 8.0),
    ("s2", "第 2 章", s_ch2_fwd, 16.0),
    ("s2", "第 2 章", s_ch2_flow, 16.0),
    ("s2", "第 2 章", s_ch2_sup, 14.0),
    ("s3", "第 3 章", lambda: section("3", "雅可比场（核心）", ["从一维的斜率说起", "模型真学出来的 J", "为什么这个量特别好学"]), 9.0),
    ("s3", "第 3 章", s_ch3_def, 18.0),
    ("s3", "第 3 章", s_ch3_real, 16.0),
    ("s3", "第 3 章", s_ch3_why, 18.0),
    ("s4", "第 4 章", lambda: section("4", "怎么训练", ["正向：一行 einsum", "损失与归一化", "正则项与反向反解"]), 8.0),
    ("s4", "第 4 章", s_ch4_fwd, 15.0),
    ("s4", "第 4 章", s_ch4_loss, 16.0),
    ("s4", "第 4 章", s_ch4_norm, 18.0),
    ("s4", "第 4 章", s_ch4_reg, 18.0),
    ("s4", "第 4 章", s_ch4_inv, 16.0),
    ("s4", "第 4 章", s_ch4_inv_train, 15.0),
    ("s5", "第 5 章", lambda: section("5", "工程踩过的坑", ["数据链路", "四个真实事故与对策", "为什么这么选"]), 8.0),
    ("s5", "第 5 章", s_ch5_pipeline, 17.0),
    ("s5", "第 5 章", s_ch5_bug1, 16.0),
    ("s5", "第 5 章", s_ch5_bug2, 16.0),
    ("s5", "第 5 章", s_ch5_bug3, 16.0),
    ("s5", "第 5 章", s_ch5_bug4, 16.0),
    ("s6", "第 6 章", lambda: section("6", "推理落地", ["完整链条", "积分式下发与舵机静差"]), 7.0),
    ("s6", "第 6 章", s_ch6_chain, 16.0),
    ("s6", "第 6 章", s_ch6_servo, 15.0),
    ("s7", "第 7 章", lambda: section("7", "别人会问什么", ["八个最可能被问到的问题", "以及怎么答"]), 7.0),
    ("s7", "第 7 章", s_ch7_qa, 20.0),
    ("s7", "第 7 章", s_ch7_qa2, 20.0),
    ("s8", "现状", s_end, 15.0),
]


def overlay(img, frac, chapter):
    """顶部章节条 + 底部进度条（每帧重画，所以静态场景只渲染一次）。"""
    d = ImageDraw.Draw(img)
    d.rectangle([0, H - 5, int(W * frac), H], fill=C(ACC))
    d.rectangle([0, H - 5, W, H], outline=C((38, 46, 60)))
    d.rectangle([0, 0, 12, H], fill=C((26, 34, 48)))
    d.text((W - 24, H - 30), chapter, font=F(17), fill=C(FAINT), anchor="ra")


def audit(verbose=True) -> int:
    BOXES.clear()
    n = 0
    for _sid, _sec, fn, _d in SECTIONS:
        BOXES.clear()
        try:
            fn()
        except Exception as exc:
            print(f"  ❌ 场景构建失败: {type(exc).__name__}: {exc}")
            n += 1
            continue
        found = []
        for bb, (s, _sz) in BOXES:
            if bb[2] > W + 1 or bb[3] > H + 1 or bb[0] < -1 or bb[1] < -1:
                found.append(f"越界 {s!r} bbox=({bb[0]:.0f},{bb[1]:.0f},{bb[2]:.0f},{bb[3]:.0f})")
        for i in range(len(BOXES)):
            for j in range(i + 1, len(BOXES)):
                (a, ta), (b, tb) = BOXES[i], BOXES[j]
                ox = min(a[2], b[2]) - max(a[0], b[0])
                oy = min(a[3], b[3]) - max(a[1], b[1])
                if ox > 4 and oy > 4:
                    found.append(f"重叠 {ta[0]!r} × {tb[0]!r} ({ox:.0f}×{oy:.0f}px)")
        if verbose:
            print(("  ❌ " if found else "  ✓ ") + f"{_sid:6s} {len(BOXES):3d} 段"
                  + (f"  ← {len(found)} 问题" if found else ""))
            for f in found[:6]:
                print("       -", f)
        n += len(found)
    print(f"  合计排版问题: {n}")
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "vera_idm.mp4")
    ap.add_argument("--audit-only", action="store_true")
    ap.add_argument("--skip-audit", action="store_true")
    args = ap.parse_args()

    if not args.skip_audit:
        print("[audit] 排版自检")
        if audit():
            print("[abort] 存在排版问题，未编码。")
            return 1
    if args.audit_only:
        return 0

    import imageio_ffmpeg
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
           "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(args.out)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    # 逐帧流式写盘，不在内存里堆帧（长视频会吃掉几十 GB）
    total_frames = sum(int(d * FPS) for *_, d in SECTIONS)
    total_dur = total_frames / FPS
    done_frames = 0
    cache: dict[int, Image.Image] = {}
    print(f"[render] {total_frames} 帧, {total_dur:.0f}s, {W}x{H}@{FPS}")
    for si, (_sid, sec, fn, dur) in enumerate(SECTIONS):
        n = int(dur * FPS)
        try:
            base_img = fn()
            cache[si] = base_img
        except TypeError:
            base_img = None
        for i in range(n):
            img = cache[si] if base_img is not None else fn()
            frame = img.copy()
            overlay(frame, done_frames / total_frames, sec)
            p.stdin.write(frame.convert("RGB").tobytes())
            done_frames += 1
            if done_frames % 500 == 0:
                print(f"  {done_frames}/{total_frames} ({done_frames/total_frames*100:.0f}%)", flush=True)
    p.stdin.close()
    if p.wait() != 0:
        print("[fail] 编码失败")
        return 1
    print(f"[done] {args.out}  ({args.out.stat().st_size/1e6:.1f} MB, {total_dur:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
