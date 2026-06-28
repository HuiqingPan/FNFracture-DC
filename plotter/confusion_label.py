import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap
from matplotlib.colors import BoundaryNorm
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns


def _format_label_value(x):
    """把 0.0 / 1.0 这类标签显示成 0 / 1，避免坐标轴标签太长。"""
    try:
        xf = float(x)
        if xf.is_integer():
            return str(int(xf))
    except Exception:
        pass
    return str(x)


def plot_confusion_matrices(
        results_df,
        label_name='true_g0p0',
        pred_name='pred_g0p0',
        class_name=None,
        normalize=False,
        save_path='g0p0_confusion_external.pdf',
        cmap=['#E9F0FC', '#C3D7F2', '#93B0DB', '#074B90']):
    """
    绘制混淆矩阵(带 Precision/Recall 柱状图 + Heatmap)，
    直接使用真实标签列 label_name 和预测标签列 pred_name。

    Args:
      results_df (str or pd.DataFrame):
        CSV路径或DataFrame，必须包含:
          - true_g0p0: 真实标签
          - pred_g0p0: 预测标签
      label_name (str): 真实标签列名，默认 'true_g0p0'
      pred_name (str): 预测标签列名，默认 'pred_g0p0'
      class_name (list or None): 类别显示名称；默认按标签值自动生成
      normalize (bool): 是否对热图按行归一化(默认False)
      save_path (str): 图片保存路径，建议 .pdf / .png / .tiff
      cmap (list): 自定义从浅到深的热图颜色
    """
    # 0) 读取数据
    results_df = pd.read_csv(results_df, encoding='utf-8') if isinstance(results_df, str) else results_df

    # 1) 检查列名
    need_cols = [label_name, pred_name]
    missing_cols = [c for c in need_cols if c not in results_df.columns]
    if missing_cols:
        raise ValueError(f"数据中缺少列: {missing_cols}")

    # 2) 去掉真实标签或预测标签缺失的样本
    results_df = results_df.dropna(subset=[label_name, pred_name]).copy()

    # 3) 提取真值和预测值
    y_true = results_df[label_name].values
    y_pred = results_df[pred_name].values

    # 4) 如果标签本身是 0.0 / 1.0 这类整数型浮点数，转成 int，避免显示 0.0
    try:
        if np.all(pd.Series(y_true).dropna().astype(float) % 1 == 0) and \
           np.all(pd.Series(y_pred).dropna().astype(float) % 1 == 0):
            y_true = y_true.astype(float).astype(int)
            y_pred = y_pred.astype(float).astype(int)
    except Exception:
        pass

    # 5) 类别顺序：同时考虑真实标签和预测标签，防止某一类只出现在预测中
    classes = np.array(sorted(pd.unique(np.concatenate([y_true, y_pred]))))
    nclass = len(classes)

    if class_name is None:
        class_name = [_format_label_value(x) for x in classes]
    if len(class_name) != nclass:
        raise ValueError(f"class_name长度({len(class_name)})必须等于类别数({nclass})，当前类别为: {classes.tolist()}")

    # 6) 混淆矩阵
    cm_raw = confusion_matrix(y_true, y_pred, labels=classes)
    acc = accuracy_score(y_true, y_pred)

    if normalize:
        cm = cm_raw.astype(float)
        row_sum = cm.sum(axis=1, keepdims=True)
        cm = cm / (row_sum + 1e-12)
    else:
        cm = cm_raw

    # 7) Precision / Recall 用原始计数计算
    col_sum = cm_raw.sum(axis=0, keepdims=True)
    row_sum = cm_raw.sum(axis=1, keepdims=True)
    diag = np.diag(cm_raw)
    precision = diag / (col_sum[0] + 1e-12)
    recall = diag / (row_sum[:, 0] + 1e-12)
    precision = np.nan_to_num(precision)
    recall = np.nan_to_num(recall)

    # 8) figure + gridspec(2 x 3)
    #    widths = [4, .5, 0.3] => col0=heatmap/precision, col1=recall, col2预留
    #    heights = [.5, 4]
    fig = plt.figure(figsize=(6, 6))
    gs = GridSpec(2, 3,
                  width_ratios=[4, .5, 0.3],
                  height_ratios=[.5, 4],
                  wspace=0.05, hspace=0.05)

    # [ (0,0) Precision | (0,1) 空 | (0,2) 空 ]
    # [ (1,0) Heatmap   | (1,1) Recall | (1,2) 空 ]

    # 8a) Precision 柱状图
    ax_prec = fig.add_subplot(gs[0, 0], frame_on=False)  # frame_on=False隐藏图表边框
    ax_prec.bar(range(nclass), precision, color='#D5D3DE', alpha=0.9)
    ax_prec.set_xlim([-0.5, nclass - 0.5])
    ax_prec.set_ylim(0, 1)
    ax_prec.set_xticks([])
    ax_prec.set_yticks([0, 0.5, 1])
    ax_prec.set_yticklabels(['0', '0.5', '1'], fontsize=10)
    ax_prec.set_ylabel('Precision', fontsize=12, labelpad=10, fontfamily='Arial')
    ax_prec.tick_params(axis='y', which='both', direction='in', length=3)

    # 8b) Recall 横向柱状图
    ax_recall = fig.add_subplot(gs[1, 1], frame_on=False)
    ax_recall.barh(range(nclass), recall, color='#C2BCB3', alpha=0.9)
    ax_recall.set_ylim([-0.5, nclass - 0.5])
    ax_recall.set_xlim(0, 1)
    ax_recall.set_yticks([])
    ax_recall.set_xticks([0, 0.5, 1])
    ax_recall.set_xticklabels(['0', '0.5', '1'], fontsize=10)
    ax_recall.set_xlabel('Recall', fontsize=12, labelpad=10, fontfamily='Arial')
    ax_recall.tick_params(axis='x', which='both', direction='in', length=3)

    # 8c) Heatmap
    ax_matrix = fig.add_subplot(gs[1, 0])

    # 颜色边界：参考原脚本BoundaryNorm写法，按当前数据动态设置
    if normalize:
        bounds = np.linspace(0, 1, len(cmap) + 1)
    else:
        cm_max = int(np.max(cm_raw)) if np.max(cm_raw) > 0 else 1
        bounds = np.linspace(0, cm_max, len(cmap) + 1)
        bounds[-1] = cm_max + 1e-9  # 确保最大值落入最后一个颜色区间

    bounds = [0, 50, 100, 150, 500]#颜色边界
    norm = BoundaryNorm(bounds, len(cmap))

    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap=ListedColormap(cmap),
        norm=norm,
        xticklabels=class_name,
        yticklabels=class_name,
        cbar=False,
        ax=ax_matrix,
        linewidths=0.5,
        linecolor='white',
        annot_kws={"fontsize": 12},
        alpha=0.9
    )

    ax_matrix.set_xlabel('Predicted label', fontsize=12, labelpad=10, fontfamily='Arial')
    ax_matrix.set_ylabel('True label', fontsize=12, labelpad=10, fontfamily='Arial')
    # ax_matrix.set_title(f'Accuracy = {acc:.3f}', fontsize=12, pad=10, fontfamily='Arial')
    ax_matrix.tick_params(axis='both', which='major', labelsize=10)
    ax_matrix.tick_params(axis='x', rotation=0)
    ax_matrix.tick_params(axis='y', rotation=90)

    # 9) 隐藏 precision / recall 坐标轴的边框
    for spine in ['top', 'right', 'bottom', 'left']:
        ax_prec.spines[spine].set_visible(False)
        ax_recall.spines[spine].set_visible(False)

    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Saved confusion matrix to: {save_path}")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion matrix:")
    print(cm_raw)

    return cm_raw, precision, recall, acc


def main():
    parser = argparse.ArgumentParser(description='Plot confusion matrix.')
    parser.add_argument('--csv', default='.path', help='输入CSV路径')
    parser.add_argument('--label_name', default='true_g0p0', help='真实标签列名')
    parser.add_argument('--pred_name', default='pred_g0p0', help='预测标签列名')
    parser.add_argument('--output', default='path', help='输出图片路径')
    parser.add_argument('--normalize', action='store_true', help='是否按行归一化热图')
    parser.add_argument('--class_names', nargs='*', default=['Negative', 'Positive'],
                        help='类别名称，例如: --class_names "Class 0" "Class 1"')
    args = parser.parse_args()

    plot_confusion_matrices(
        results_df=args.csv,
        label_name=args.label_name,
        pred_name=args.pred_name,
        class_name=args.class_names,
        normalize=args.normalize,
        save_path=args.output,
        cmap=['#f2f5e6', '#c5e0b2', '#ddefd1', '#fff8d4']
    )


if __name__ == '__main__':
    main()

