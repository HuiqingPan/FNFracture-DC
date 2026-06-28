import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# ======================

# 1. 读取数据

# ======================

file_path = "../internal_predictions.csv"   # 路径
df = pd.read_csv(file_path, encoding='utf-8')
df = df[df['true_fracture'] != 0]

true_col = "true_angle"
pred_col = "pred_angle"

# 如果有类别列，比如 angle_type / param / name，可在这里填写

# 如果没有类别列，就设为 None

group_col = None

# ======================

# 2. Bland-Altman 绘图函数

# ======================

def bland_altman_plot(ax, data, title=""):
    true_angle = data[true_col].astype(float)
    pred_angle = data[pred_col].astype(float)

    mean_value = (true_angle + pred_angle) / 2
    diff_value = pred_angle - true_angle
    
    mean_diff = np.mean(diff_value)
    sd_diff = np.std(diff_value, ddof=1)
    
    upper_limit = mean_diff + 1.96 * sd_diff
    lower_limit = mean_diff - 1.96 * sd_diff
    
    ax.scatter(mean_value, diff_value, color = '#2E7D63', s=12, alpha=0.6)
    
    ax.axhline(mean_diff, color="#222222", linewidth=2, label="Mean Difference")
    ax.axhline(upper_limit, color="#D33F49", linestyle="--", linewidth=2, label="±1.96 SD")
    ax.axhline(lower_limit, color="#D33F49", linestyle="--", linewidth=2)
    
    text = (f"+1.96SD = {upper_limit:.2f}, "
            f"-1.96SD = {lower_limit:.2f},"
            f"Mean Diff. = {mean_diff:.2f}")
    
    ax.text(
        0.1, 1.04, text,
        transform=ax.transAxes,
        fontsize=10,
        va="bottom"
    )
    
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Mean")
    ax.set_ylabel("Difference")
    x_ticks = [0, 20, 40, 60, 80]
    ax.set_xticks(x_ticks)
    ax.set_xlim(-5, 92)
    ax.set_ylim(-50, 50)
    ax.legend(fontsize=9)
    
    return mean_diff, upper_limit, lower_limit


# ======================

# 3. 绘图

# ======================

if group_col is None:
    fig, ax = plt.subplots(figsize=(6, 5))
    bland_altman_plot(ax, df)

else:
    groups = list(df[group_col].dropna().unique())
    n = len(groups)

    ncols = 4
    nrows = math.ceil(n / ncols)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)
    
    for i, g in enumerate(groups):
        sub_df = df[df[group_col] == g]
        bland_altman_plot(axes[i], sub_df, title=str(g))
    
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

# ======================

# 4. 总图例与保存

# ======================

plt.grid(False)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("../resultsaltman_internal.pdf",
            dpi=300,
            bbox_inches="tight")
plt.show()
