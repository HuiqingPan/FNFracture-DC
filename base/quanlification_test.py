import pandas as pd
import numpy as np
from scipy.stats import norm

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from scipy.stats import pearsonr

class quanlification_test():

    def _nadeau(self, score1, score2, side='two-sided', return_static=False, verbose=False):
        """
        使用 Nadeau and Bengio 方法计算p值。

        参数:
        - bootstrapped_aucs: 实际模型通过bootstrap得到的AUC列表
        - blank_model_aucs: 空模型通过bootstrap得到的AUC列表
        - side: 'right'（右侧检验）、'left'（左侧检验）或 'two-sided'（双侧检验）
        - alpha: 显著性水平，默认0.05

        返回:
        - p值
        """
        # 计算两组Score的差异
        score_diff = np.array(score1) - np.array(score2)

        # 计算差异的均值和方差
        mean_diff = np.mean(score_diff)
        var_diff = np.var(score_diff, ddof=1)

        # 使用Nadeau方法修正方差，公式: var_corrected = var_diff * (1 + 1/n_bootstrap)
        n_bootstrap = len(score1)
        var_corrected = var_diff * (1 + 1 / n_bootstrap)

        # 计算Z统计量
        z_stat = mean_diff / np.sqrt(var_corrected)
        if verbose:
            print(f'Using {side} testing, calculating by score1 - score2')

        # 计算p值
        if side == 'right':
            # 右侧检验
            p_value = norm.sf(z_stat)  # 使用生存函数计算右侧p值
        elif side == 'left':
            # 左侧检验
            p_value = norm.cdf(z_stat)
        else:
            # 双侧检验
            p_value = 2 * (1 - norm.cdf(np.abs(z_stat)))

        # 返回p值和显著性水平的结果
        if return_static:
            return z_stat, p_value
        else:
            return p_value

    # =========================
    # 去除缺失值
    # =========================
    def delete_missing_values(self, df, true_reg_col , pred_reg_col):
        eval_df = df[
            [true_reg_col, pred_reg_col]
        ].dropna()

        y_true = eval_df[true_reg_col].astype(float)
        y_pred = eval_df[pred_reg_col].astype(float)

        return y_true, y_pred

    def cal_mae(self, y_true, y_pred):
        return mean_absolute_error(y_true, y_pred)

    def cal_abs_error(self, y_true, y_pred):
        errors = y_pred - y_true
        abs_errors = np.abs(errors)

        bias = np.mean(errors)
        median_ae = np.median(abs_errors)

        return abs_errors, bias, median_ae

    def cal_rmse(self, y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    def cal_r2(self, y_true, y_pred):
        return r2_score(y_true, y_pred)

    def cal_pearson(self, y_true, y_pred):

        if len(y_true) > 1:
            pearson_r, pearson_p = pearsonr(y_true, y_pred)
        else:
            pearson_r, pearson_p = np.nan, np.nan

        return pearson_r, pearson_p

    def _print_mode(self, array: np.ndarray, mode: str = 'ci', round: int = 3, name: str = ''):
        means = np.mean(array)
        if mode == 'ci':
            lower, upper = np.percentile(array, [2.5, 97.5])
            print(f'{name}: {means:.{round}f} ({lower:.{round}f}–{upper:.{round}f})')
        elif mode == 'sd':
            std = np.std(array)
            print(f"{name}: {means:.{round}f} ({std:.{round}f})")

    def bootstrap_resample(self, y_true, y_pred, n_iterations=1000, frac=.8,
                           thre: tuple = None, normalized: bool = True):
        """
        使用bootstrap方法重采样并计算每次重采样的指标。
        :param y_pred: 预测概率
        :param y_true: 真实标签
        :param n_iterations: bootstrap的次数
        :param weight: mae/R2等加权方法
        :return: 重采样得到的所有指标
        """
        metrics_dict = {
            'mae': [],
            'rmse': [],
            'median ae': [],
            'errors': [],
            'bias': [],
            'R2': [],
            'Person r': [],
            'Person p': []
        }

        # 对每次迭代进行重采样
        for _ in range(n_iterations):
            # 从原始数据中随机抽取样本（80%样本）
            indices = np.random.choice(len(y_true), int(frac * len(y_true)), replace=True)
            pred_resampled = y_pred[indices]
            labels_resampled = y_true[indices]

            # 计算各项指标，并保存到字典中
            metrics_dict['mae'].append(self.cal_mae(labels_resampled, pred_resampled))
            metrics_dict['rmse'].append(self.cal_rmse(labels_resampled, pred_resampled))
            metrics_dict['median ae'].append(self.cal_abs_error(labels_resampled, pred_resampled)[2])
            metrics_dict['errors'].append(self.cal_abs_error(labels_resampled, pred_resampled)[0])
            metrics_dict['bias'].append(self.cal_abs_error(labels_resampled, pred_resampled)[1])
            metrics_dict['R2'].append(self.cal_r2(labels_resampled, pred_resampled))
            metrics_dict['Person r'].append(self.cal_pearson(labels_resampled, pred_resampled)[0])
            metrics_dict['Person p'].append(self.cal_pearson(labels_resampled, pred_resampled)[1])

        return metrics_dict


