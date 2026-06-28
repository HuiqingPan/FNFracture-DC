import numpy as np
from 分型与坏死预测.base.test_base_label import test_base_label

#本部分适用于只有预测类别，没有预测概率的指标计算
class Tester_label(test_base_label):

    def calc_label_metrics(self, path, label_name='true_Pauwels', pred_name='pauwels_type',
                           print_type='ci', bootstrap=1000, round_digits=3, frac=.8):
        """
        计算只有真实标签和预测标签时的评估指标。
        适用于Pauwels这类没有概率输出的结果。

        :param path: 结果文件路径
        :param label_name: 真实标签列名
        :param pred_name: 预测标签列名
        :param print_type: 输出类型，'ci' 或 'sd'
        :param bootstrap: bootstrap重采样次数
        :param round_digits: 小数位数
        :param frac: 每次bootstrap抽样比例
        """
        print(f'================Calculating label results from {path}================')

        preds, labels, label_list = self.read_label_rd(
            path=path,
            label_name=label_name,
            pred_name=pred_name
        )

        print(f'Valid samples: {len(labels)}')
        print(f'Labels: {label_list}')

        metrics_dict = self.bootstrap_resample_label(
            preds=preds,
            labels=labels,
            label_list=label_list,
            n_iterations=bootstrap,
            frac=frac
        )

        print('----------------Overall metrics----------------')
        self._print_mode(np.array(metrics_dict['accuracy']), mode=print_type,
                         round=round_digits, name='Accuracy')
        self._print_mode(np.array(metrics_dict['macro_f1']), mode=print_type,
                         round=round_digits, name='Macro-F1')
        self._print_mode(np.array(metrics_dict['weighted_f1']), mode=print_type,
                         round=round_digits, name='Weighted-F1')
        self._print_mode(np.array(metrics_dict['kappa']), mode=print_type,
                         round=round_digits, name='Kappa')

        print('----------------Total metrics, macro-average----------------')
        self._print_mode(np.array(metrics_dict['total_sensitivity']), mode=print_type,
                         round=round_digits, name='Total Sensitivity')
        self._print_mode(np.array(metrics_dict['total_specificity']), mode=print_type,
                         round=round_digits, name='Total Specificity')
        self._print_mode(np.array(metrics_dict['total_ppv']), mode=print_type,
                         round=round_digits, name='Total PPV')
        self._print_mode(np.array(metrics_dict['total_npv']), mode=print_type,
                         round=round_digits, name='Total NPV')
        self._print_mode(np.array(metrics_dict['total_f1']), mode=print_type,
                         round=round_digits, name='Total F1')

        print('----------------Per-class metrics----------------')
        for label in label_list:
            print(f'Class {label}')
            self._print_mode(np.array(metrics_dict[f'class_{label}_sensitivity']), mode=print_type,
                             round=round_digits, name='Sensitivity')
            self._print_mode(np.array(metrics_dict[f'class_{label}_specificity']), mode=print_type,
                             round=round_digits, name='Specificity')
            self._print_mode(np.array(metrics_dict[f'class_{label}_ppv']), mode=print_type,
                             round=round_digits, name='PPV')
            self._print_mode(np.array(metrics_dict[f'class_{label}_npv']), mode=print_type,
                             round=round_digits, name='NPV')
            self._print_mode(np.array(metrics_dict[f'class_{label}_f1']), mode=print_type,
                             round=round_digits, name='F1')

        return metrics_dict
