import numpy as np

from 分型与坏死预测.base.base import Base


class Compare(Base):

    def compare(self, path1, path2, label_name, metric_name: str = 'all', print_type='ci', side='two-sided',
                proba_prefix = 'prob_class_',
                bootstrap=1000, verbose=False, frac=.8,
                thre: tuple = None, normalized: bool = True):
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
        probas1, labels1 = self.read_rd(path1, label_name=label_name, proba_prefix=proba_prefix)
        probas2, labels2 = self.read_rd(path2, label_name=label_name,  proba_prefix=proba_prefix)
    
        metrics1, metrics2 = self.bootstrap_resample(probas1, labels1, bootstrap, frac=frac, normalized=normalized,
                                                     thre=thre), \
                             self.bootstrap_resample(probas2, labels2, bootstrap, frac=frac, normalized=normalized,
                                                     thre=thre)
        # 获取指标列表
        metric_list = ['accuracy', 'f1', 'auroc', 'sensitivity', 'specificity',
                       'ppv', 'npv', 'auprc', 'kappa', 'mcc']
    
        if len(np.unique(labels1)) >= 3:
            metric_list.append('topk_accuracy')
        else:
            metric_list.append('net_benefit')
    
        if metric_name != 'all':
            metric_list = [metric_name]  # 只比较指定的指标
    
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
            self._print_mode(score_diff, mode=print_type, round=3, name=f'{metric} difference, P={p_value:.4f}: ')
