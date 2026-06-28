import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import FormatStrFormatter

# ========= 参数区 =========
EXCEL_PATH = "../results/绘图数据整理.xlsx"   # 路径
SHEET_NAME = "senior_f"          # sheet
OUT_PATH = "../senior_f.pdf"

Y_LABEL = "Values"

# 颜色和点形状，可按需要继续加
colors = ["#BC6F45", "#E48DBC", "#e5a6b8", "#7aa6c2", "#8dbf7f"]
markers = ["o", "v", "s", "D", "^"]

# ========= 读取数据 =========
df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)

# 默认第一列为横坐标名称，后面所有列为不同组
metric_col = df.columns[0]
value_cols = df.columns[1:]

df[metric_col] = df[metric_col].astype(str)

# 转成数值，避免 Excel 里有字符串数字
for col in value_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

x = np.arange(len(df))

# ========= 绘图 =========
fig, ax = plt.subplots(figsize=(7, 3))

# 每个指标下画灰色竖线：连接该指标所有组的 min 和 max
for i, row in df.iterrows():
    vals = row[value_cols].dropna().values.astype(float)
    if len(vals) >= 2:
        ax.vlines(
            x=i,
            ymin=np.min(vals),
            ymax=np.max(vals),
            color="#c9c9c9",
            linewidth=1.2,
            zorder=1
        )

# 画散点

legend_names = {
    "without_mean": "Without_model",
    "with_mean": "With_model"
}

for j, col in enumerate(value_cols):
    ax.scatter(
        x,
        df[col],
        s=42,
        marker=markers[j % len(markers)],
        color=colors[j % len(colors)],
        label=legend_names.get(col, col),
        zorder=3,
        edgecolors="none"
    )

# ========= 样式 =========
ax.set_xticks(x)
ax.set_xticklabels(df[metric_col], rotation=45, fontsize=9)

ax.set_ylabel(Y_LABEL, rotation=90, fontsize=11)

# y 轴范围自动留白
all_vals = df[value_cols].to_numpy(dtype=float)
y_min = np.nanmin(all_vals)
y_max = np.nanmax(all_vals)
pad = max((y_max - y_min) * 0.18, 0.01)
ax.set_ylim(y_min - pad, y_max + pad)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))




# 图例放右侧
ax.legend(
    frameon=False,
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    fontsize=8,
    handletextpad=0.4
)

# 去掉上、右边框，类似示例图
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.tick_params(axis="both", length=3, width=1)
ax.grid(False)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
plt.show()
