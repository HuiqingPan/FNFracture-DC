import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ======================
# 读取数据
# ======================
file_path = "path"
df = pd.read_csv(file_path, encoding="utf-8")
df = df[df['true_angle']!=0]

# # ======================
# # 抽取需要的数量
# # ======================
# # 每一类要抽取的数量
# sample_nums = {
#     0: 58,
#     1: 69,
#     2: 95
# }
#
# # 分类别随机抽取
# df = pd.concat([
#     df[df['true_Pauwels'] == label].sample(
#         n=num,
#         random_state=42
#     )
#     for label, num in sample_nums.items()
# ], ignore_index=True)
#
# # 如果希望最终顺序也随机打乱
# new_df = df.sample(frac=1, random_state=42).reset_index(drop=True)
#
# # 查看新 df 中各类数量
# print(new_df['true_Pauwels'].value_counts())
#
# df.to_excel(
#     "../results/multi_task/mobilenet/external_sampled.xlsx",
#     index=False
# )

# 真实标签和预测标签列名
true_col = "true_angle"
pred_col = "pred_angle"

# 如果有分组列，例如不同角度类型，可在这里填写列名
# 如果没有分组列，保持 None
group_col = None
# group_col = "angle_type"

df = df[[true_col, pred_col] + ([group_col] if group_col else [])].dropna()

# ======================
# 绘图函数
# ======================
def plot_regression(ax, data, title=""):
    x = data[pred_col].values.astype(float)   # Prediction
    y = data[true_col].values.astype(float)   # Ground truth

    # 线性回归
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    # 排序后的 x，用于画线
    x_line = np.linspace(x.min(), x.max(), 200)
    y_fit = slope * x_line + intercept

    # Perfect line: y = x
    y_perfect = x_line

    # ======================
    # 95% Prediction Band
    # ======================
    n = len(x)
    y_hat = slope * x + intercept
    residuals = y - y_hat

    mse = np.sum(residuals ** 2) / (n - 2)
    x_mean = np.mean(x)
    sxx = np.sum((x - x_mean) ** 2)

    t_value = stats.t.ppf(0.975, df=n - 2)

    pred_se = np.sqrt(
        mse * (1 + 1 / n + (x_line - x_mean) ** 2 / sxx)
    )

    upper = y_fit + t_value * pred_se
    lower = y_fit - t_value * pred_se

    # ======================
    # 绘图
    # ======================
    ax.scatter(
        x, y,
        s=12,
        alpha=0.65,
        color="#2E7D63",
        label="Data"
    )

    ax.plot(
        x_line, y_perfect,
        color="#F79E7B",
        linewidth=2,
        label="Perfect Line"
    )

    ax.plot(
        x_line, y_fit,
        color="#222222",
        linewidth=2,
        label=f"y = {slope:.2f}x + {intercept:.2f}"
    )

    ax.plot(
        x_line, upper,
        color="#D33F49",
        linestyle="--",
        linewidth=2,
        label="95% Prediction Band"
    )

    ax.plot(
        x_line, lower,
        color="#D33F49",
        linestyle="--",
        linewidth=2
    )

    ax.set_xlabel("Prediction (Degree)")
    ax.set_ylabel("Ground truth (Degree)")
    ax.set_title(title)

    # ax.grid(True, alpha=0.3)

    # 让 x/y 轴范围一致，方便和 y=x 比较
    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())
    margin = (max_val - min_val) * 0.08

    # ax.set_xlim(min_val - margin, max_val + margin)
    ax.set_ylim(min_val - margin, max_val + margin)
    x_ticks = [0, 20, 40, 60, 80]
    ax.set_xticks(x_ticks)
    ax.set_xlim(-5, 92)

    ax.legend(fontsize=9)

    # 打印评价指标
    mae = np.mean(np.abs(y - x))
    rmse = np.sqrt(np.mean((y - x) ** 2))

    print(f"{title}")
    print(f"Regression: y = {slope:.4f}x + {intercept:.4f}")
    print(f"R² = {r_value ** 2:.4f}")
    print(f"MAE = {mae:.4f}")
    print(f"RMSE = {rmse:.4f}")
    print("-" * 40)


# ======================
# 单图或多子图
# ======================
if group_col is None:
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_regression(ax, df, title="Linear Regression Analysis")

else:
    groups = list(df[group_col].dropna().unique())
    n_groups = len(groups)

    n_cols = 4
    n_rows = int(np.ceil(n_groups / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.5 * n_cols, 4 * n_rows)
    )

    axes = np.array(axes).reshape(-1)

    for i, g in enumerate(groups):
        sub_df = df[df[group_col] == g]
        plot_regression(axes[i], sub_df, title=str(g))

    # 删除多余空白子图
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.grid(False)

plt.tight_layout()
plt.savefig("path", dpi=300)
plt.show()

# ======================
# 计算回归指标
# ======================
def regression_metrics(data, name="Total"):
    x = data[pred_col].values.astype(float)   # Prediction
    y = data[true_col].values.astype(float)   # Ground truth

    n = len(x)

    if n < 3:
        return {
            "Parameter": name,
            "N": n,
            "R²": np.nan,
            "p-value": np.nan,
            "Regression equation": np.nan,
            "Slope coefficient": np.nan,
            "Regression line slope in Degree": np.nan,
            "Standard error of prediction difference": np.nan,
            "MAE": np.nan,
            "RMSE": np.nan,
        }

    # 线性回归: y = slope * x + intercept
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    y_fit = slope * x + intercept

    # R²
    r2 = r_value ** 2

    # 回归线角度，单位 degree
    slope_degree = np.degrees(np.arctan(slope))

    # prediction difference: true - pred
    diff = y - x

    # 标准差形式的 prediction difference
    std_prediction_diff = np.std(diff, ddof=1)

    # 如果你更想用回归残差的标准误，可以使用下面这个：
    residuals = y - y_fit
    residual_standard_error = np.sqrt(np.sum(residuals ** 2) / (n - 2))

    # MAE / RMSE
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff ** 2))

    return {
        "Parameter": name,
        "N": n,
        "R²": r2,
        "p-value": p_value,
        "Regression equation": f"y = {slope:.2f}x + {intercept:.2f}",
        "Slope coefficient": slope,
        "Regression line slope in Degree": slope_degree,
        "Standard error of prediction difference": std_prediction_diff,
        "Residual standard error": residual_standard_error,
        "MAE": mae,
        "RMSE": rmse,
    }

# ======================
# 输出单组或多组结果
# ======================
results = []

if group_col is None:
    results.append(regression_metrics(df, name="Total"))
else:
    for name, sub_df in df.groupby(group_col):
        results.append(regression_metrics(sub_df, name=name))

    # 输出 Total
    results.append(regression_metrics(df, name="Total"))

result_df = pd.DataFrame(results)
def format_p_value(p):
    if pd.isna(p):
        return ""
    elif p < 0.001:
        return "<0.001"
    else:
        return f"{p:.3f}"

display_df = result_df.copy()
display_df["R²"] = display_df["R²"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
display_df["p-value"] = display_df["p-value"].map(format_p_value)
display_df["Slope coefficient"] = display_df["Slope coefficient"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
display_df["Regression line slope in Degree"] = display_df["Regression line slope in Degree"].map(
    lambda x: f"{x:.2f}°" if pd.notna(x) else ""
)
display_df["Standard error of prediction difference"] = display_df[
    "Standard error of prediction difference"
].map(lambda x: f"{x:.2f}°" if pd.notna(x) else "")

display_df["Residual standard error"] = display_df["Residual standard error"].map(
    lambda x: f"{x:.2f}°" if pd.notna(x) else ""
)
display_df["MAE"] = display_df["MAE"].map(lambda x: f"{x:.2f}°" if pd.notna(x) else "")
display_df["RMSE"] = display_df["RMSE"].map(lambda x: f"{x:.2f}°" if pd.notna(x) else "")
paper_table = display_df[
    [
        "Parameter",
        "R²",
        "p-value",
        "Regression line slope in Degree",
        "Standard error of prediction difference",
    ]
]

display_df.to_excel("path")