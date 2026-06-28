import pandas as pd
import numpy as np
from sklearn.utils import resample
from 分型与坏死预测.base.base import Base


class Tester(Base):


    def calc_all_metrics(self, path, label_name, proba_prefix='prob_class_', print_type='ci', average='macro',
                         bootstrap=1000, round_digits=3, k=2, frac=.8):
        """
        计算并输出所有评估指标：准确率，F1分数，AUC等。
        :param path: 结果文件的路径
        :param label_name: 标签列的名称
        :param proba_prefix: 预测概率的列前缀
        :param print_type: 输出类型 ('ci' 或 'sd')
        :param average: 用于多分类计算的加权方式，默认为'macro'
        :param bootstrap: bootstrap重采样次数
        :param round_digits: 输出的小数位数
        :param k: Top-k准确率的k值
        """
        print(f'================Calculating results from {path}================')
        # 读取数据
        probas, labels = self.read_rd(path, proba_prefix, label_name)

        # 执行bootstrap并计算指标
        metrics_dict = self.bootstrap_resample(probas, labels, n_iterations=bootstrap, weight=average, k=k, frac=frac)

        # 打印每个指标
        self._print_mode(np.array(metrics_dict['accuracy']), mode=print_type, round=round_digits, name='Accuracy')
        self._print_mode(np.array(metrics_dict['f1']), mode=print_type, round=round_digits, name='F1 score')
        self._print_mode(np.array(metrics_dict['auroc']), mode=print_type, round=round_digits, name='AUROC')
        self._print_mode(np.array(metrics_dict['sensitivity']), mode=print_type, round=round_digits, name='Sensitivity')
        self._print_mode(np.array(metrics_dict['specificity']), mode=print_type, round=round_digits, name='Specificity')
        self._print_mode(np.array(metrics_dict['ppv']), mode=print_type, round=round_digits, name='PPV')
        self._print_mode(np.array(metrics_dict['npv']), mode=print_type, round=round_digits, name='NPV')
        self._print_mode(np.array(metrics_dict['auprc']), mode=print_type, round=round_digits, name='AUPRC')
        self._print_mode(np.array(metrics_dict['kappa']), mode=print_type, round=round_digits, name='Kappa')
        self._print_mode(np.array(metrics_dict['mcc']), mode=print_type, round=round_digits, name='MCC')
        if probas.shape[1] <= 2:
            self._print_mode(np.array(metrics_dict['youden_index']), mode=print_type, round=round_digits, name='Youden')
            self._print_mode(np.array(metrics_dict['net_benefit']), mode=print_type,
                             round=round_digits, name='Clinical utility')
        if probas.shape[1] >= 3:
            self._print_mode(np.array(metrics_dict['topk_accuracy']), mode=print_type, round=round_digits,
                             name=f'Top-{k} Accuracy')
