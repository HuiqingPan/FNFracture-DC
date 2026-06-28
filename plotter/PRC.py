import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from polars.selectors import alpha
from scipy.stats import norm
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

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


def compute_auroc_and_pvalue(rd, proba_prefix, label_name, weight, classes = None):
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
    auprcs = []
    baseline_auprcs = []

    for i in range(n_bootstrap):
        # 在每次bootstrap中随机抽样
        indices = np.random.choice(len(y_true), size=len(y_true), replace=True)
        y_true_sampled = y_true[indices]
        y_pred_sampled = y_pred[indices]

        y_true_sampled_bin = label_binarize(y_true_sampled, classes=classes)

        if is_multiclass:
            # 对于多类别任务，计算每个类别的AUC，并使用平均值
            auprc =  average_precision_score(y_true_sampled_bin, y_pred_sampled, average='macro', multi_class='ovr')
        else:
            # 对于二分类任务，使用正类的概率计算AUC
            auprc =  average_precision_score(y_true_sampled, y_pred_sampled[:, 1], average=weight)
        auprcs.append(auprc)

        #基线模型
        baseline_auprc = np.mean(y_true_sampled_bin)
        baseline_auprcs.append(baseline_auprc)

    # 计算AUROC和p值
    auprc_model = np.mean(auprcs)  # 实际模型的平均AUROC
    p_value = _nadeau(auprcs, baseline_auprcs)  # 使用Nadeau方法计算P值

    # 计算标准差（SD）和95%置信区间（CI）
    auprc_std = np.std(auprcs, ddof=1)
    ci_lower = np.percentile(auprcs, 2.5)
    ci_upper = np.percentile(auprcs, 97.5)

    return auprc_model, p_value, auprc_std, ci_lower, ci_upper, baseline_auprc


def plot_pr_curve(
        rd,
        encoding,
        proba_prefix,
        label_name,
        weight = None,
        print_type='sd',
        title=None,
        model_name = None,
        figsize=(4,4),
        color='#fac694',
        dpi=300
                  ):

    # 绘制AUPR曲线
    df = pd.read_csv(rd, encoding=encoding) if isinstance(rd, str) else rd
    df = df.dropna(subset=[label_name])

    labels = df[label_name].values
    classes = np.sort(np.unique(labels))
    auprc_model, p_value, auprc_std, ci_lower, ci_upper, baseline_auprc = compute_auroc_and_pvalue(df, proba_prefix,
                                                                                                   label_name, weight, classes =  classes)
    probs = [df[f'{proba_prefix}{i}'].values for i in range(len(df.columns) - 1) if f'{proba_prefix}{i}' in df.columns]

    pr, rc, thresholds = precision_recall_curve(labels, probs[1])

    # 格式化AUPRC和P值
    auprc_str = f'{auprc_model:.2f}'  # 保留三位小数
    auprc_std_str = f'{auprc_std:.2f}'
    p_value_str = f'= {p_value:.2e}' if p_value >= 0.0001 else '< 0.0001'
    ci_lower_str = f'{ci_lower:.2f}'
    ci_upper_str = f'{ci_upper:.2f}'

    if print_type == 'sd':
        # 显示标准差
        auprc_str = f'AUPRC (SD): {auprc_str} ({auprc_std_str})'
    elif print_type == 'ci':
        # 显示95%置信区间
        auprc_str = f'AUPRC (95% CI): {auprc_str} ({ci_lower:.2f}-{ci_upper:.2f})'

    # 图形绘制
    plt.figure(figsize=figsize, dpi=dpi)  # 更高的分辨率和图像尺寸
    if model_name is not None:
        plt.plot(pr, rc, label=f'{model_name} {auprc_str}, P {p_value_str}', lw=1, color=color)  # 使用颜色
    else:
        plt.plot(pr, rc, label=f'{auprc_str}', lw=1, color=color)  # 使用颜色

        pr_fill = np.insert(pr, 0, 0)
        rc_fill = np.insert(rc, 0, rc[0])
        # plt.fill_between(pr_fill, 0, rc_fill, color='#F9F3F2')
        plt.fill_between(pr_fill, 0, rc_fill, alpha=0.2, color='#E9F0FC')

    # baseline
    plt.axhline(baseline_auprc, linestyle=':', color='gray',
                label=f" Baseline: {baseline_auprc:.3f}")

    # 对角线
    plt.plot([0, 1], [1, 0], linestyle=':', color='gray')

    # 设置坐标轴
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.xlabel('Recall', fontsize=10, fontweight='normal', family='Arial')
    plt.ylabel('Precision', fontsize=10, fontweight='normal', family='Arial')


    # 优化图例
    plt.legend(loc='lower left', bbox_to_anchor=(0.0, 0.0), fontsize=8, frameon=False, facecolor='w', edgecolor='black', fancybox=True,
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

plot_pr_curve(rd="path",
              encoding='utf-8',
              proba_prefix='garden_prob_', label_name='true_garden',
              color = '#9BABD2')


# color_list = [
#fceac8'填充色，透明度0.2，黄色
#fac694线黄色
#C25759\#C16E71砖红色
#F9F3F2红色面积填充色，无透明度
# #E5A79A红色面积填充色，有透明度]
##8CA3C3, #9BABD2蓝色
##E9F0FC蓝色填充色