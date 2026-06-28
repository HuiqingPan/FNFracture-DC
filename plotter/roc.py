import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
import pandas as pd
import numpy as np
from scipy.stats import norm
from sympy import true
from sympy.physics.vector.tests.test_printing import alpha


def _nadeau(score1, score2, side='two-sided', return_static=False, verbose=False):
    """
    使用Nadeau方法计算p值。

    参数:
    - score1: 实际模型的AUC
    - score2: 空模型的AUC，为0.5，但师兄是模拟和随机生成的
    - side: 'right'（右侧检验）、'left'（左侧检验）或 'two-sided'（双侧检验）
    - return_static: 是否返回z统计量和p值
    - verbose: 是否打印中间过程

    返回:
    - p值
    """
    score_diff = np.array(score1) - np.array(score2)
    mean_diff = np.mean(score_diff)
    var_diff = np.var(score_diff, ddof=1)
    n_bootstrap = len(score1)
    var_corrected = var_diff * (1 + 1 / n_bootstrap)
    z_stat = mean_diff / np.sqrt(var_corrected)

    if verbose:
        print(f'Using {side} testing, calculating by score1 - score2')

    if side == 'right':
        p_value = norm.sf(z_stat)
    elif side == 'left':
        p_value = norm.cdf(z_stat)
    else:
        p_value = 2 * (1 - norm.cdf(np.abs(z_stat)))

    if return_static:
        return z_stat, p_value
    else:
        return p_value


def compute_auroc_and_pvalue(rd, proba_prefix, label_name, weight):
    if not isinstance(rd, pd.DataFrame):
        raise ValueError(f"Expected a DataFrame, but got {type(rd)}")

    # 获取所有的proba列和标签列
    y_true = rd[label_name].values
    y_pred = rd[[col for col in rd.columns if col.startswith(proba_prefix)]].values

    # 判断是否为多类别任务
    n_classes = y_pred.shape[1]
    is_multiclass = n_classes > 2

    # 计算每次bootstrap的AUC值
    n_bootstrap = 1000  # 设定bootstrap的次数
    aucs = []
    blank_model_aucs = []  # 用于存储空模型的AUC

    for i in range(n_bootstrap):
        # 在每次bootstrap中随机抽样
        indices = np.random.choice(len(y_true), size=len(y_true), replace=True)
        y_true_sampled = y_true[indices]
        y_pred_sampled = y_pred[indices]

        if is_multiclass:
            # 对于多类别任务，计算每个类别的AUC，并使用平均值
            auc = roc_auc_score(y_true_sampled, y_pred_sampled, average='macro', multi_class='ovr')
        else:
            # 对于二分类任务，使用正类的概率计算AUC
            auc = roc_auc_score(y_true_sampled, y_pred_sampled[:, 1], average=weight)
        aucs.append(auc)

        # 生成空模型的预测（随机预测标签）
        random_pred = np.random.rand(len(y_true_sampled), n_classes)
        random_pred /= random_pred.sum(axis=1, keepdims=True)  # 归一化为概率

        # 计算空模型的AUC
        if is_multiclass:
            blank_auc = roc_auc_score(y_true_sampled, random_pred, average='macro', multi_class='ovr')
        else:
            blank_auc = roc_auc_score(y_true_sampled, random_pred[:, 1], average=weight)
        blank_model_aucs.append(blank_auc)

    # 计算AUROC和p值
    roc_auc = np.mean(aucs)  # 实际模型的平均AUROC
    p_value = _nadeau(aucs, blank_model_aucs)  # 使用Nadeau方法计算P值

    # 计算标准差（SD）和95%置信区间（CI）
    auc_std = np.std(aucs)  # 标准差
    ci_lower = np.percentile(aucs, 2.5)  # 置信区间下界
    ci_upper = np.percentile(aucs, 97.5)  # 置信区间上界

    return roc_auc, p_value, auc_std, ci_lower, ci_upper


def plot_single_roc(rd1,
                    proba_prefix, label_name, print_type='sd', weight=None, model_name=None, color='#fac694',
                    plot_balance=true, star_color='#6a5853'):

    # 绘制ROC曲线
    df = pd.read_csv(rd1, encoding='utf-8') if isinstance(rd1, str) else rd1
    df = df.dropna(subset=[label_name])
    roc_auc, p_value, auc_std, ci_lower, ci_upper = compute_auroc_and_pvalue(df, proba_prefix, label_name, weight)

    labels = df[label_name].values
    probs = [df[f'{proba_prefix}{i}'].values for i in range(len(df.columns) - 1) if f'{proba_prefix}{i}' in df.columns]

    fpr, tpr, thresholds = roc_curve(labels, probs[1])

    # 格式化AUROC和P值
    roc_auc_str = f'{roc_auc:.2f}'  # 保留三位小数
    p_value_str = f'= {p_value:.2e}' if p_value >= 0.0001 else '< 0.0001'
    ci_lower_str = f'{ci_lower:.2f}'
    ci_upper_str = f'{ci_upper:.2f}'

    if print_type == 'sd':
        # 显示标准差
        auc_str = f'AUROC (SD): {roc_auc_str} ({auc_std:.2f})'
        # auc_str = f'AUROC: {roc_auc_str}'
    elif print_type == 'ci':
        # 显示95%置信区间
        auc_str = f'AUROC (95% CI): {roc_auc_str} ({ci_lower:.2f}-{ci_upper:.2f})'

    # 图形绘制
    plt.figure(figsize=(4, 4))  # 更高的分辨率和图像尺寸
    if model_name is not None:
        plt.plot(fpr, tpr, label=f'{model_name} {auc_str}, P {p_value_str}', lw=1, color=color)  # 使用颜色
    else:
        plt.plot(fpr, tpr-.02, label=f'AUROC: {roc_auc_str}\n(95%CI {ci_lower_str}-{ci_upper_str})', lw=1, color=color)  # 使用颜色
        plt.fill_between(fpr, 0, tpr - .02, alpha=0.2, color='#E9F0FC')

    # 绘制随机猜测的线（灰色虚线）
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)

    if plot_balance:
        # 在计算出的平衡点绘制星星
        plt.scatter(np.mean(fpr)+.05, np.mean(tpr)+.1, color=star_color, s=130, marker='*', label="Balanced performance")

    # 设置坐标轴
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.xlabel('1-Specificity', fontsize=10, fontweight='normal', family='Arial')
    plt.ylabel('Sensitivity', fontsize=10, fontweight='normal', family='Arial')

    # 优化图例
    plt.legend(loc='lower right', bbox_to_anchor=(1.0, 0.0), fontsize=8, frameon=False, facecolor='w', edgecolor='black', fancybox=True,
               framealpha=0.9, labelspacing = 1.0)

    # 美化图形背景和网格
    plt.gca().set_facecolor('white')  # 设置背景颜色为白色
    plt.grid(False)  # 淡化网格线，避免干扰

    # # 去除顶部和右侧的边框线
    # plt.gca().spines['top'].set_visible(False)
    # plt.gca().spines['right'].set_visible(False)

    # plt.title('Garden', fontsize = 10, pad=10)

    # 显示图形
    plt.tight_layout()  # 自动调整子图参数，使图像更紧凑

    plt.savefig('path',
                 dpi=300, bbox_inches='tight')

    plt.show()

plot_single_roc(rd1="path",
                proba_prefix='garden_prob_', label_name='true_garden', plot_balance=True,
                color='#9BABD2')


# color_list = [
#     '#1f77b4',  # 经典蓝色
#     '#2ca02c',  # 宝石绿色
#     '#9467bd',  # 深紫色
#     '#ff7f0e',  # 橙红色
#     '#636363',  # 深灰色
#     '#008080',  # 深海蓝
#     '#d62728',  # 深红色
#     '#333333',  # 石墨黑
#     '#8c564b',  # 深棕色
#     '#e377c2',  # 鲜粉色
#     '#7f7f7f',  # 中灰色
#     '#bcbd22',  # 黄绿色
#     '#17becf'   # 淡蓝色
# ]

# 黄色系
# color='#fac694'
# plt.fill_between(fpr, 0, tpr - .02, alpha=0.2, color='#fceac8')
# 红色系
# color='#C16E71'
# plt.fill_between(fpr, 0, tpr - .02, alpha=0.2, color='#E5A79A')
##8CA3C3蓝色
##E9F0FC蓝色填充色