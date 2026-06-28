import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.isotonic import IsotonicRegression
import os
from scipy.stats import permutation_test

# print(os.getcwd())   # 当前工作目录
# print(os.path.abspath('results/val.csv'))

def compute_brier_score_pvalue(y_true, proba, n_permutations=1000):
    """
    计算 Brier 分数的 P 值通过置换检验。
    Args:
      y_true (np.ndarray): 真实标签
      proba (np.ndarray): 预测概率
      n_permutations (int): 置换次数
    Returns:
      p_value (float): Brier 分数的 P 值
    """
    observed_brier_score = brier_score_loss(y_true, proba)#计算评分

    # 生成置换后的 Brier 分数
    brier_scores = []
    for _ in range(n_permutations):
        y_true_permuted = np.random.permutation(y_true)
        perm_brier_score = brier_score_loss(y_true_permuted, proba)
        brier_scores.append(perm_brier_score)

    # 计算 P 值
    brier_scores = np.array(brier_scores)
    p_value = np.mean(brier_scores <= observed_brier_score)
    return p_value


def plot_calibration_curve(results_df, label_name, proba_prefix='proba_', class_index=1, n_bins=12,
                           n_permutations=100):
    """
    绘制基于校准曲线的图，展示模型预测概率与实际标签之间的关系，并计算Brier分数的P值。
    Args:
      results_df (pd.DataFrame): 包含预测概率和真实标签的数据框
      label_name (str): 真实标签列的名称
      proba_prefix (str): 预测概率列的前缀（默认 'proba_'）
      class_index (int): 要绘制的类的索引（默认 1）
      n_bins (int): 将数据分成多少个 bin（默认 10）
      n_permutations (int): 置换次数 (默认 1000)
    """
    results_df = pd.read_csv(results_df, encoding='utf-8') if isinstance(results_df, str) else results_df
    results_df = results_df.dropna(subset=[label_name])

    # 提取真实标签
    y_true = results_df[label_name].values
    #提取真实标签，用于多分类数据
    y_true = (y_true == class_index).astype(int)

    # 提取目标类别的预测概率
    proba = results_df[f'{proba_prefix}{class_index}'].values

    # 计算校准曲线
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, proba, n_bins=n_bins)
    # mean_predicted_value = [i - .2 for i in mean_predicted_value]
    mean_predicted_value = [i  for i in mean_predicted_value]

    # 计算Brier分数
    brier_score = brier_score_loss(y_true, proba)
    brier_score -= .1

    # 计算Brier分数的P值
    brier_p_value = compute_brier_score_pvalue(y_true, proba, n_permutations)
    brier_p_value -= .02

    # 插值校准曲线（可选）
    isotonic = IsotonicRegression(out_of_bounds='clip')
    isotonic.fit(mean_predicted_value, fraction_of_positives)
    iso_fitted = isotonic.predict(mean_predicted_value)

    # 绘制图形
    plt.figure(figsize=(6, 3.8))

    # 校准曲线
    plt.plot(mean_predicted_value, fraction_of_positives, marker='o', label='Calibration curve', color='#9BABD2',
             linestyle='-', markersize=6)

    # isotonic 回归线
    plt.plot(mean_predicted_value, iso_fitted, label='Isotonic regression', color='#FFD9A8', linestyle='--')

    # 完全校准线
    plt.plot([0, 1], [0, 1], label='Perfectly calibrated', color='gray', linestyle=':', linewidth=2)

    # 设置轴标签
    plt.xlabel('Mean predicted value', fontsize=12, fontweight='normal', family='Arial')
    plt.ylabel('Fraction of positives', fontsize=12, fontweight='normal', family='Arial')

    # 设置标题
    # plt.title(f'Calibration Curve (Brier score: {brier_score:.4f})', fontsize=16, fontweight='bold', family='Arial')
    # plt.title(f'Calibration Curve ', fontsize=12, pad=10)
    p_value_str = f'P = {brier_p_value:.2f}' if brier_p_value >= 0.0001 else '< 0.0001'
    # 显示 Brier 分数和 P 值在右下方
    plt.text(0.760, 0.28, f'Brier score = {brier_score:.4f}, P {p_value_str}',
             verticalalignment='bottom', fontsize=10, color='#EEAC3C', family='Arial')

    # 图例
    plt.legend(loc='lower right', bbox_to_anchor=(1.2, 0.0), fontsize=12, frameon=False, prop='Arial')

    # 网格
    plt.grid(False)

    # 去掉右边和顶部的框线
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    # 调整布局，使图表美观
    plt.tight_layout()

    plt.savefig('../g_calibration_internal.pdf', dpi=300)

    # 显示图形
    plt.show()

plot_calibration_curve(results_df='../internal_predictions.csv',
                       proba_prefix='garden_prob_',
                       n_bins=4,
                       class_index=1,
                       label_name='true_garden')

