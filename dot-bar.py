import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Circle
from matplotlib.pyplot import scatter

df = pd.read_excel("path",
                   sheet_name=' ')

metrics = df["metric"].tolist()
x = np.arange(len(metrics))
offset = 0.16

x_ext = x - offset
x_int = x + offset

# 自动处理 low/high 顺序
ext_low = np.minimum(df["external_low"], df["external_high"])
ext_high = np.maximum(df["external_low"], df["external_high"])
int_low = np.minimum(df["internal_low"], df["internal_high"])
int_high = np.maximum(df["internal_low"], df["internal_high"])

# 误差线长度：相对 mean 的上下距离
ext_yerr = np.vstack([
    df["external_mean"] - ext_low,
    ext_high - df["external_mean"]
])

int_yerr = np.vstack([
    df["internal_mean"] - int_low,
    int_high - df["internal_mean"]
])

# ===== 每个 metric 一组颜色：浅色给 external，深色给 internal =====
light_colors = [
    "#B5C3D7",  # Accuracy
    "#B6C9C0",  # F1
    "#F5D8B7",  # AUROC
    "#F1D0C6",  # Sensitivity
    "#DDAEAB",  # Specificity
    "#D0E0EF",  # PPV
    "#E9CDDF",  # NPV
    "#C8BFD9",  # AUPRC
    "#E4E3BF",  # Kappa
    "#D9E8C8",  # MCC
    "#F3D9C9",  # Net benefit
]

dark_colors = [
    "#6E8FB2",  # Accuracy
    "#7DA494",  # F1
    "#EAB67A",  # AUROC
    "#E5A79A",  # Sensitivity
    "#C16E71",  # Specificity
    "#ABC8E5",  # PPV
    "#D8A0C1",  # NPV
    "#9B8DB8",  # AUPRC
    "#D0D08A",  # Kappa
    "#A9C77A",  # MCC
    "#D89C84",  # Net benefit
]

# 如果 metric 数量超过颜色数量，就自动循环
if len(metrics) > len(light_colors):
    import itertools
    light_colors = list(itertools.islice(itertools.cycle(light_colors), len(metrics)))
    dark_colors = list(itertools.islice(itertools.cycle(dark_colors), len(metrics)))
else:
    light_colors = light_colors[:len(metrics)]
    dark_colors = dark_colors[:len(metrics)]

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# ===== 一个 metric 一个 metric 地画 =====
for i in range(len(metrics)):
    ax.errorbar(
        x_ext[i], df.loc[i, "external_mean"],
        yerr=[[ext_yerr[0, i]], [ext_yerr[1, i]]],
        fmt="s",
        color=light_colors[i],
        ecolor=light_colors[i],
        elinewidth=2.5,
        capsize=4,
        markersize=10,
        markeredgecolor="none",
        label="_nolegend_"
    )

    ax.errorbar(
        x_int[i], df.loc[i, "internal_mean"],
        yerr=[[int_yerr[0, i]], [int_yerr[1, i]]],
        fmt="o",
        color=dark_colors[i],
        ecolor=dark_colors[i],
        elinewidth=2.5,
        capsize=4,
        markersize=10,
        markeredgecolor="none",
        label="_nolegend_"
    )

# ===== 坐标轴 =====
ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.6)

ax.set_xticks(x)
ax.set_xticklabels(metrics, rotation=30, ha="right", fontsize=13)
ax.set_ylim(-0.5, 0.5)
ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5])
ax.tick_params(axis = 'y', labelsize = 13)
ax.set_ylabel("Increased performance (95%CI)", fontsize=15)
# ax.set_xlabel("Metric", fontsize=12)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# # ===== 把 X 轴标签文字改成对应 metric 的颜色（用深色更清楚）=====
# for ticklabel, color in zip(ax.get_xticklabels(), dark_colors):
#     ticklabel.set_color(color)
#     ticklabel.set_fontweight("bold")

# ===== 手动画顶部两行色块图例 =====
# 位置参数都是 ax 坐标系(0~1)
fig_width, fig_height = 14, 6
aspect_ratio = fig_width / fig_height
start_x = 0.02      # 色块起始 x
start_y1 = 1.00     # 第一行 y
start_y2 = 0.94     # 第二行 y
box_w = 0.015       # 每个色块宽度
box_h = 0.015*aspect_ratio       # 每个色块高度
gap = 0.004         # 色块之间间距

# 第一行：External（浅色）
for i, c in enumerate(light_colors[:len(metrics)]):
    rect = Rectangle(
        (start_x + i * (box_w + gap), start_y1),
        box_w, box_h,
        transform=ax.transAxes,
        facecolor=c,
        edgecolor='none',
        clip_on=False
    )
    ax.add_patch(rect)

# 第二行：Internal（深色）
#如果用circle画出来是椭圆，因为会收到图形整体长宽的影响
for i, c in enumerate(dark_colors[:len(metrics)]):
    ax.scatter(
        start_x + i * (box_w + gap) + box_w/2, start_y2 + box_h/2,
        s=90,
        color=[c],
        marker="o",
        transform = ax.transAxes,
        clip_on=False
    )

# 在每行色块后面加文字
text_x = start_x + len(metrics) * (box_w + gap) + 0.01

ax.text(
    text_x, start_y1 + box_h / 2,
    "External",
    transform=ax.transAxes,
    va='center', ha='left',
    fontsize=11.5
)

ax.text(
    text_x, start_y2 + box_h / 2,
    "Internal",
    transform=ax.transAxes,
    va='center', ha='left',
    fontsize=11.5
)

plt.tight_layout()
plt.savefig("path")
plt.show()