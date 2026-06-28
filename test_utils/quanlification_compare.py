import numpy as np
import pandas as pd
from 分型与坏死预测.base.quanlification_test import quanlification_test


class Compare(quanlification_test):

    def compare(self, path1, path2,
                label_name1, label_name2,
                print_type='sd', side='two-sided',
                true_angle = 'true_angle',
                bootstrap=1000, verbose=False, frac=.8):
        """
        比较两个结果数据框的指标差异，并计算p值
        :param path1: 第一个数据路径
        :param path2: 第二个数据路径
        :param metric_name: 需要比较的指标名，默认比较所有指标
        :param print_type: 输出类型，'ci' 或 'sd'
        :param side: 检验类型，'right'、'left' 或 'two-sided'
        :param n_iterations: bootstrap重采样次数
        """
        print(f'================Comparing between {path1} and {path2}, using {side} with (former - later)================')
        # 读取两个数据
        results_df1 = pd.read_excel(path1, sheet_name= 'Sheet2')
        results_df1=results_df1.dropna(subset=[true_angle])
        y_pred1 = results_df1.filter(regex=f"^{label_name1}").to_numpy()
        y_true1 = results_df1.filter(regex=true_angle).to_numpy()

        results_df2 = pd.read_excel(path2, sheet_name= 'Sheet2')
        results_df2=results_df2.dropna(subset=[true_angle])
        y_pred2 = results_df2.filter(regex=f"^{label_name2}").to_numpy()
        y_true2 = results_df2.filter(regex=true_angle).to_numpy()

        metrics1, metrics2 = self.bootstrap_resample(y_pred1, y_true1), \
                             self.bootstrap_resample(y_pred2, y_true2)

        # 获取指标列表
        metric_list = ['mae', 'rmse', 'median ae', 'errors', 'bias',
                       'R2', 'Person r', 'Person p']

        # 对每个指标进行计算
        for metric in metric_list:
            # 提取结果
            score1 = metrics1.get(metric)
            score2 = metrics2.get(metric)

            # 计算差值
            score_diff = np.array(score1) - np.array(score2)

            # 使用 _nadeau 计算 p 值
            p_value = self._nadeau(score1, score2, side=side, verbose=verbose)

            # 打印差值的CI或SD
            self._print_mode(score_diff, mode=print_type, round=3, name=f'{metric} difference, P={p_value:.3f}: ')
