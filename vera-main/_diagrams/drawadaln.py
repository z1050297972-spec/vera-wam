"""单独画 AdaLN（自适应层归一化）的示意图。

全部对照 vera/video_model/algorithms/wan/modules/model.py 的实际实现：
  WanModel.time_embedding / time_projection、block.modulation、
  WanAttentionBlock.forward 里的 e[0..5]、Head.modulation。
框高按内容自动计算，避免文字溢出。
"""
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

avail = {f.name for f in font_manager.fontManager.ttflist}
CJK = next((n for n in ("Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans CJK TC",
                        "Droid Sans Fallback", "Noto Serif CJK SC") if n in avail), None)
if CJK is None:
    raise SystemExit("找不到中文字体")
plt.rcParams["font.family"] = CJK
plt.rcParams["axes.unicode_minus"] = False

COND = dict(fc="#DBEAFE", ec="#3B82F6", tc="#1E3A8A")
MOD = dict(fc="#FEF3C7", ec="#F59E0B", tc="#78350F")
ACT = dict(fc="#EEF2F7", ec="#94A3B8", tc="#1E293B")
RES = dict(fc="#DCFCE7", ec="#22C55E", tc="#14532D")
PLAIN = dict(fc="#F1F5F9", ec="#CBD5E1", tc="#475569")
ZERO = dict(fc="#FEE2E2", ec="#EF4444", tc="#991B1B")

fig, ax = plt.subplots(figsize=(16.5, 14.5))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

GAP = 1.75


def _split(style):
    st = dict(style)
    return {k: v for k, v in st.items() if k != "tc"}, st.get("tc", "#1E293B")


def box(x, ybot, w, title, lines=(), style=ACT, fs=10, ts=12, gap=GAP, pad_t=1.15, pad_b=1.0):
    """按内容算高度：h = 标题 + n 行 × 行距 + 上下留白。ybot 是框底。"""
    h = pad_t + len(lines) * gap + pad_b
    pk, tc = _split(style)
    ax.add_patch(FancyBboxPatch((x, ybot), w, h, boxstyle="round,pad=0.25,rounding_size=0.45",
                                lw=1.6, zorder=2, **pk))
    ax.text(x + w / 2, ybot + h - pad_t + 0.45, title, ha="center", va="top",
            fontsize=ts, fontweight="bold", zorder=3, color=tc)
    for i, (txt, mono) in enumerate(lines):
        ax.text(x + w / 2, ybot + h - pad_t - 0.95 - i * gap, txt, ha="center", va="top",
                fontsize=fs, zorder=3, color=tc, family="monospace" if mono else None)
    return dict(l=x, r=x + w, t=ybot + h, b=ybot, cx=x + w / 2, h=h)


def arrow(a, b, label="", color="#475569", rad=0.0, ls="-", fs=9.5, lx=0, ly=0, lw=1.7):
    ax.add_patch(FancyArrowPatch((a[0], a[1]), (b[0], b[1]), arrowstyle="-|>",
                                 mutation_scale=14, lw=lw, color=color, linestyle=ls,
                                 zorder=1, connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=1, shrinkB=1))
    if label:
        ax.text((a[0] + b[0]) / 2 + lx, (a[1] + b[1]) / 2 + ly, label, ha="center",
                va="center", fontsize=fs, color=color, zorder=6,
                bbox=dict(fc="white", ec="none", alpha=0.93, pad=1.5))


ax.text(50, 99.2, "AdaLN —— 自适应层归一化", ha="center", va="top",
        fontsize=23, fontweight="bold", color="#0F172A")
ax.text(50, 96.2, "让归一化的 scale / shift 由「时间步 t」算出来 —— "
                  "于是同一个网络在不同去噪阶段表现出不同行为",
        ha="center", va="top", fontsize=12, color="#475569")

# ================= ① 普通 LN vs AdaLN =================
ax.text(5, 92.4, "① 为什么需要它", ha="left", va="top", fontsize=13.5,
        fontweight="bold", color="#0F172A")
ln = box(5, 82.0, 42, "普通 LayerNorm",
         [("y = γ ⊙ normalize(x) + β", False),
          ("γ、β 写死在参数里，对所有输入一视同仁", False)], PLAIN, ts=12)
adln = box(5, 72.0, 42, "AdaLN（DiT 提出，WAN 用这个）",
           [("y = (1 + scale) ⊙ normalize(x) + shift", False),
            ("scale、shift 由条件信号算出 → 条件变，行为就变", False)], MOD, ts=12)
arrow((ln["cx"], ln["b"]), (adln["cx"], adln["t"]), "让归一化「知道现在在第几步」", ly=0.3)
ax.text(48.5, 76.0, "本图的条件 ＝ 时间步 t\n（去噪到第几步）", ha="left", va="center",
        fontsize=10.5, color="#1E3A8A")

# ================= ② t → 六个调制量 =================
ax.text(53, 92.4, "② 六个调制量怎么算出来（30 层里每层都一样）",
        ha="left", va="top", fontsize=13.5, fontweight="bold", color="#0F172A")

t = box(53, 85.5, 11, "时间步 t", [("标量", False)], COND, ts=11, fs=9.5)
emb = box(67, 85.5, 16, "time_embedding", [("Linear → SiLU → Linear", True),
                                          ("dim 1536", False)], ACT, ts=11, fs=9.5)
proj = box(53, 78.5, 16, "time_projection", [("SiLU + Linear", True),
                                             ("1536 → 1536×6", True)], ACT, ts=11, fs=9.5)
modp = box(73, 78.5, 15, "本层 modulation", [("可学习参数 (1,6,1536)", False)],
           MOD, ts=11, fs=9.5)
e = box(53, 71.0, 30, "相加得到 e", [("e = time_projection(·) + per-layer offsets", True),
                                     ("形状 [batch, 6, 1536]", False)], MOD, ts=11.5, fs=10)

arrow((t["r"], (t["b"] + t["t"]) / 2), (emb["l"], (emb["b"] + emb["t"]) / 2))
arrow((emb["cx"], emb["b"]), (proj["cx"] - 3.5, proj["t"]))
arrow((emb["r"] + 5, (emb["b"] + emb["t"]) / 2), (modp["l"] - 1, (modp["b"] + modp["t"]) / 2),
      "每层各自的偏移", rad=-0.2, lx=2, ly=1.2, fs=9)
arrow((proj["cx"], proj["b"]), (e["cx"] - 8, e["t"]))
arrow((modp["cx"], modp["b"]), (e["cx"] + 9, e["t"]))

# ================= ③ 六个量分给三条支路 =================
ax.text(53, 67.0, "③ 六个数分给三条残差支路", ha="left", va="top",
        fontsize=13.5, fontweight="bold", color="#0F172A")

xbox = box(51, 47.5, 9, "输入 x", [("token 序列", False)], PLAIN, ts=10.5, fs=9.5)

b1 = box(64, 55.0, 24, "① 自注意力",
         [("scale e[0]   shift e[1]   gate e[2]", False),
          ("norm1(x)·(1+e0)+e1 → attn → ×e2 → +x", True)], COND, ts=11.5, fs=9.2)
b2 = box(64, 47.6, 24, "② 交叉注意力（读文本）",
         [("这一支不做 AdaLN 调制", False),
          ("norm3(x) → cross-attn(text) → +x", True)], PLAIN, ts=11.5, fs=9.2)
b3 = box(64, 40.2, 24, "③ FFN",
         [("scale e[3]   shift e[4]   gate e[5]", False),
          ("norm2(x)·(1+e3)+e4 → ffn → ×e5 → +x", True)], COND, ts=11.5, fs=9.2)
out = box(64, 34.5, 24, "输出", [], RES, ts=11.5)

for b in (b1, b2, b3):
    arrow((xbox["r"], (xbox["b"] + xbox["t"]) / 2), (b["l"], (b["b"] + b["t"]) / 2), lw=1.4)
arrow((b3["cx"], b3["b"]), (out["cx"], out["t"]), lw=1.4)

# e → 三支
ax.add_patch(FancyArrowPatch((e["cx"] + 15, e["b"]), (70, b1["t"] + 0.5),
                             arrowstyle="-|>", mutation_scale=12, lw=1.5, color="#F59E0B",
                             zorder=1, connectionstyle="arc3,rad=0.22"))
ax.add_patch(FancyArrowPatch((e["cx"] + 15, e["b"]), (76, b2["t"]),
                             arrowstyle="-", lw=1.5, color="#F59E0B", zorder=1))
ax.add_patch(FancyArrowPatch((e["cx"] + 15, e["b"]), (70, b3["t"] - 0.3),
                             arrowstyle="-|>", mutation_scale=12, lw=1.5, color="#F59E0B",
                             zorder=1, connectionstyle="arc3,rad=-0.22"))
ax.text(89.2, b1["t"] - 0.8, "e[0..2]", ha="left", va="center", fontsize=9.5,
        color="#B45309", bbox=dict(fc="white", ec="none", alpha=0.9, pad=1))
ax.text(89.2, b2["cx"] if False else (b2["b"] + b2["t"]) / 2, "无调制", ha="left",
        va="center", fontsize=9.5, color="#94A3B8")
ax.text(89.2, b3["b"] + 3.2, "e[3..5]", ha="left", va="center", fontsize=9.5,
        color="#B45309", bbox=dict(fc="white", ec="none", alpha=0.9, pad=1))

# ================= ④ gate / AdaLN-Zero =================
ax.text(5, 67.0, "④ gate 是干什么的", ha="left", va="top",
        fontsize=13.5, fontweight="bold", color="#0F172A")
z = box(5, 51.0, 42, "AdaLN-Zero（DiT 的技巧）",
        [("残差分支输出要乘一个 gate：x = x + y · gate", False),
         ("训练初期让 gate ≈ 0 → 每条分支输出 ≈ 0", False),
         ("30 层网络此时等于「什么都不做」的恒等映射", False),
         ("深层网络因此训得稳、收敛快、不易发散", False),
         ("WAN 的 gate 是学出来的，没有做零初始化", False)], ZERO, ts=12, fs=10)

w = box(5, 36.5, 42, "为什么视频模型特别需要它",
        [("同一个网络，去噪到第 1 步和第 999 步该做的事完全不同：", False),
         ("　早期 → 定布局、定运动趋势　　后期 → 抠细节、修边缘", False),
         ("AdaLN 就是把这个「分阶段任务」显式喂进网络", False)], ACT, ts=12, fs=10)
arrow((z["cx"], z["b"]), (w["cx"], w["t"]))

# ================= 源码对照 =================
src = box(5, 4.5, 90, "对应源码（vera/video_model/algorithms/wan/modules/model.py）",
          [("time_embedding   = Linear → SiLU → Linear                    (L478)", True),
           ("time_projection  = SiLU → Linear(dim → dim×6)               (L481)", True),
           ("block.modulation = randn(1, 6, dim)   per-layer offsets   (L273)", True),
           ("self-attn: norm1(x)·(1+e[1]) + e[0] → ×e[2]                (L303-305)", True),
           ("ffn:        norm2(x)·(1+e[4]) + e[3] → ×e[5]                (L311-312)", True),
           ("head:       norm(x)·(1+e[1]) + e[0]   scale/shift only    (L354)", True)],
          dict(fc="#F8FAFC", ec="#94A3B8", tc="#334155"), ts=12, fs=10.5)

path = "/home/shawn/桌面/vera-wam/vera-main/_diagrams/adaln_diagram.png"
fig.savefig(path, dpi=145, bbox_inches="tight", facecolor="white")
print("已保存:", path)