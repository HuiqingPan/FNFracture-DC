import pandas as pd
import numpy as np
from sklearn.utils import resample
from 分型与坏死预测.base.quanlification_test import quanlification_test


class Tester(quanlification_test):


    def calc_all_metrics(self, path, sheet_name, y_true, y_pred, bootstrap=1000, print_type = 'sd', round_digits=3, frac=.8):
        """
        计算并输出所有评估指标：准确率，F1分数，AUC等。
        :param path: 结果文件的路径
        :param y_ture: 标签列的名称
        :param y_pred: 预测概率的列前缀
        :param bootstrap: bootstrap重采样次数
        :param round_digits: 输出的小数位数
        """
        print(f'================Calculating results from {path}================')
        # 读取数据
        results_df = pd.read_excel(path, sheet_name=sheet_name)
        results_df = results_df.dropna(subset=[y_true])
        y_pred = results_df.filter(regex=f"^{y_pred}").to_numpy()
        y_true = results_df.filter(regex=y_true).to_numpy()

        # print(f"y_pred shape: {y_pred.shape}, y_ture shape: {y_true.shape}")
        # print(f"y_pred sample: {y_pred[:5]}")
        # print(f"y_ture sample: {y_true[:5]}")
        # print(f"y_pred std: {np.std(y_pred)}, y_ture std: {np.std(y_true)}")

        # 执行bootstrap并计算指标
        metrics_dict = self.bootstrap_resample(y_true, y_pred, n_iterations=bootstrap, frac=frac)

        # 打印每个指标
        self._print_mode(np.array(metrics_dict['mae']), mode=print_type, round=round_digits, name='mae')
        self._print_mode(np.array(metrics_dict['rmse']), mode=print_type, round=round_digits, name='emse')
        self._print_mode(np.array(metrics_dict['median ae']), mode=print_type, round=round_digits, name='median ae')
        self._print_mode(np.array(metrics_dict['errors']), mode=print_type, round=round_digits, name='errors')
        self._print_mode(np.array(metrics_dict['bias']), mode=print_type, round=round_digits, name='bias')
        self._print_mode(np.array(metrics_dict['R2']), mode=print_type, round=round_digits, name='R2')
        self._print_mode(np.array(metrics_dict['Person r']), mode=print_type, round=round_digits, name='Person r')
        self._print_mode(np.array(metrics_dict['Person p']), mode=print_type, round=round_digits, name='Person p')
