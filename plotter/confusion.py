import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap
from matplotlib.colors import BoundaryNorm
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns


def plot_confusion_matrices(
        results_df,
        label_name,
        class_name,
        proba_prefix='fracture_prob_',
        normalize=False,
        cmap=['#FFF5E7', '#FEEBED', '#E5A79A', '#C16E71']):
    """
    为多个任务分别画混淆矩阵(带 Precision/Recall 柱状图 + Heatmap + colorbar)，
    并且从 {task}_proba_i 列中取 argmax 当预测标签；colorbar 放在热图外面。

    Args:
      results_df (pd.DataFrame):
        必须包含:
          - labels_{task}: 真实标签 (int类别)
          - {task}_proba_0, {task}_proba_1, ...: 各类别概率列
      tasks (list): 要绘制的任务列表 (如 ['diag_code','anxiety'] )
      class_name_map (dict or None): {task: [...]}, 类名映射；否则用数字序列
      normalize (bool): 是否行归一化(默认False)
      cmap (str): 热图的颜色映射(默认 'YlGnBu')，自定义从浅到深写
    """
    results_df = pd.read_csv(results_df, encoding='utf-8') if isinstance(results_df, str) else results_df
    results_df = results_df.dropna(subset=[label_name])
    # 1) 提取真值

    y_true = results_df[label_name].values

    # 2) 提取 proba 列

    proba_array =  results_df.filter(regex=proba_prefix).to_numpy()
    # 3) preds = argmax
    y_pred = np.argmax(proba_array, axis=1)#跨行计算，取的是每行的最大值
    nclass = proba_array.shape[1]
    classes = np.unique(y_true)

    # 5) 混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    if normalize:
        cm = cm.astype(float)
        row_sum = cm.sum(axis=1, keepdims=True)
        cm = cm / (row_sum + 1e-12)

    # 6) Precision / Recall
    col_sum = cm.sum(axis=0, keepdims=True)
    row_sum = cm.sum(axis=1, keepdims=True)
    diag = np.diag(cm)
    precision = diag / (col_sum[0] + 1e-12)
    recall = diag / (row_sum[:, 0] + 1e-12)
    precision = np.nan_to_num(precision)
    recall = np.nan_to_num(recall)

    # 7) figure + gridspec(2 x 3)
    #    widths = [4,1,0.3] => col0=4, col1=1, col2=0.3 for colorbar
    #    heights = [1,4]
    fig = plt.figure(figsize=(6, 6))
    gs = GridSpec(2, 3,
                  width_ratios=[4, .5, 0.3],
                  height_ratios=[.5, 4],
                  wspace=0.05, hspace=0.05)

    # (0,0): precision, (1,0): heatmap, (1,1): recall, (1,2): colorbar
    # (0,1) / (0,2) 空置不用
    #[ (0,0) | (0,1) | (0,2) ]网格布局
    # [ (1,0) | (1,1) | (1,2) ]网格布局

    # 7a) Precision 柱状图
    ax_prec = fig.add_subplot(gs[0, 0], frame_on=False)#frame_on=False隐藏图表边框
    ax_prec.bar(range(nclass), precision, color='#D5D3DE', alpha=0.9)#绘制柱状图
    ax_prec.set_xlim([-0.5, nclass - 0.5])#x轴的范围，确保柱子居中
    ax_prec.set_ylim(0, 1)#y轴的刻度范围
    ax_prec.set_xticks([])#隐藏x轴的刻度标签
    ax_prec.set_yticks([0, 0.5, 1])#设置y轴刻度线的位置
    ax_prec.set_yticklabels(['0', '0.5', '1'], fontsize=10)#设置y轴的刻度标签和字体大小
    ax_prec.set_ylabel('Precision', fontsize=12, labelpad=10, fontfamily='Arial')#labelpad=10 控制标签与轴的距离
    ax_prec.tick_params(axis='y', which='both', direction='in', length=3)
    #设置y轴的刻度样式，direction='in'：刻度线朝内（指向图表内部），length=3：刻度线长度（像素），which='both'：同时调整主刻度和次刻度（此处仅主刻度生效）

    # 7b) Recall 柱状图
    ax_recall = fig.add_subplot(gs[1, 1], frame_on=False)
    ax_recall.barh(range(nclass), recall, color='#C2BCB3', alpha=0.9)
    ax_recall.set_ylim([-0.5, nclass - 0.5])
    ax_recall.set_xlim(0, 1)
    ax_recall.set_yticks([])
    ax_recall.set_xticks([0, 0.5, 1])
    ax_recall.set_xticklabels(['0', '0.5', '1'], fontsize=10)
    ax_recall.set_xlabel('Recall', fontsize=12, labelpad=10, fontfamily='Arial')
    ax_recall.tick_params(axis='x', which='both', direction='in', length=3)

    # 7c) Colorbar Ax
    # ax_cbar = fig.add_subplot(gs[1,2])  # 第1行,第2列, col2
    # 7d) Heatmap
    ax_matrix = fig.add_subplot(gs[1, 0])
    bounds = [0, 50, 80, 100, 200]#颜色边界
    norm = BoundaryNorm(bounds, 4)
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap=ListedColormap(cmap),  # Subtler heatmap gradient
        norm = norm,
        xticklabels=class_name,
        yticklabels=class_name,
        cbar=False,
        ax=ax_matrix,
        linewidths=0.5,
        linecolor='white',
        annot_kws={"fontsize": 12},
        alpha=0.9  # Slight transparency for a softer look
    )
    ax_matrix.set_xlabel('Predicted label', fontsize=12, labelpad=10, fontfamily='Arial')
    ax_matrix.set_ylabel('True label', fontsize=12, labelpad=10, fontfamily='Arial')
    ax_matrix.tick_params(axis='both', which='major', labelsize=10)
    ax_matrix.tick_params(axis='x', rotation=0)
    ax_matrix.tick_params(axis='y', rotation=90)

    # 隐藏 precision / recall 坐标轴的边框
    for spine in ['top', 'right', 'bottom', 'left']:
        ax_prec.spines[spine].set_visible(False)
        ax_recall.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig('../g_confusion_external.pdf',
                dpi=300)

    plt.show()

plot_confusion_matrices(results_df="../external_predictions.csv",
                        label_name='true_garden',
                        proba_prefix='garden_prob_',
                        class_name=['Garden I&II', 'Garden III&IV'],
                        cmap=['#E9F0FC', '#C3D7F2', '#93B0DB', '#074B90'])
