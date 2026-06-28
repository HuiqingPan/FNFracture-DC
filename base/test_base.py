from scipy.stats import norm
from sklearn import metrics
from sklearn.preprocessing import label_binarize
import numpy as np


class Test_base:

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


    def _is_multiclass(self, labels):
        """
        判断是否是多分类任务
        :param labels: 真实标签数组
        :return: True 如果是多分类任务，False 如果是二分类任务
        """
        return len(np.unique(labels)) > 2

    def _acc(self, preds, labels):
        """
        计算准确率
        :param preds: 模型的预测结果
        :param labels: 真实标签
        :return: 准确率
        """
        return metrics.accuracy_score(labels, preds)

    def _topk_acc(self, all_probas, labels, k=2):#输入的all_probas的数据类型是ndarray吗
        """
        计算 Top-k 准确率
        :param all_probas: 所有类别的预测概率[0.2, 0.5, 0.8]，对应索引[0, 1, 2]，返回索引才可以与真实标签比较
        :param labels: 真实标签[1]
        :param k: 选择的 Top-k，默认是 2
        :return: Top-k 准确率
        """
        topk_preds = np.argsort(all_probas, axis=1)[:, -k:]  # 取前 k 个概率最大的类别
        topk_correct = np.any(topk_preds == labels[:, None], axis=1)  # 判断预测是否正确，[:, None]是给转成ndarray格式的labels增加一个维度，方便与topk_preds进行广播比较
        # topk_correct = np.any(topk_preds == labels.to_numpy()[:, None], axis=1)
        return np.mean(topk_correct)

    def _f1(self, preds, labels, weight='macro'):
        """
        计算 F1 分数
        :param preds: 模型的预测结果
        :param labels: 真实标签
        :param weight: 加权方法，默认为 'macro'
        :return: F1 分数
        """
        if self._is_multiclass(labels):  # 如果是多分类任务
            return metrics.f1_score(labels, preds, average=weight)
        else:  # 二分类时不加权
            return metrics.f1_score(labels, preds)

    def _auroc(self, all_probas, labels, weight='macro'):
        """
        计算 AUC-ROC 分数
        :param all_probas: 所有类别的预测概率
        :param labels: 真实标签
        :param weight: 加权方法，默认为 'macro'
        :return: AUC-ROC 分数
        """
        if self._is_multiclass(labels):  # 多分类任务
            y_true_bin = label_binarize(labels, classes=np.unique(labels))  # 将标签二值化
            return metrics.roc_auc_score(y_true_bin, all_probas, average=weight, multi_class="ovr")  # 使用 OVR 方法
        else:  # 二分类时不加权
            return metrics.roc_auc_score(labels, all_probas[:, 1])  # 二分类 AUC，通常all_probas第列对应的是正列的概率值

    def _sensitivity(self, preds, labels, weight='macro'):
        """
        计算灵敏度（召回率）
        :param preds: 模型的预测结果
        :param labels: 真实标签
        :param weight: 加权方法，默认为 'macro'
        :return: 灵敏度（召回率）
        """
        if self._is_multiclass(labels):  # 多分类任务
            return metrics.recall_score(labels, preds, average=weight)
        else:  # 二分类时不加权
            return metrics.recall_score(labels, preds)

    def _specifity(self, preds, labels):
        """ 计算特异性（Specificity） :
        param preds: 模型的预测结果 :
        param labels: 真实标签 :
        return: 特异性 """
        # cm = metrics.confusion_matrix(labels, preds)
        # tn = np.diag(cm)  # True negatives
        # fp = cm.sum(axis=0) - tn # False positives
        # specificity_per_class = tn / (tn + fp + np.finfo(float).eps) # 防止除零错误
        # return np.mean(specificity_per_class)

        cm = metrics.confusion_matrix(labels, preds)
        num_classes = cm.shape[0]
        spec_per_class = []
        total = cm.sum()

        for i in range(num_classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = total - tp - fp - fn
            spec = tn / (tn + fp + np.finfo(float).eps)
            spec_per_class.append(spec)

        return np.mean(spec_per_class)

    def _ppv(self, preds, labels, weight='macro'):
        """
        计算精确率（Positive Predictive Value）
        :param preds: 模型的预测结果
        :param labels: 真实标签
        :param weight: 加权方法，默认为 'macro'
        :return: 精确率
        """
        if self._is_multiclass(labels):  # 多分类任务
            return metrics.precision_score(labels, preds, average=weight)
        else:  # 二分类时不加权
            return metrics.precision_score(labels, preds)

    def _npv(self, preds, labels):
        """
        计算阴性预测值（Negative Predictive Value）
        :param preds: 模型的预测结果
        :param labels: 真实标签
        :return: 阴性预测值
        """
        cm = metrics.confusion_matrix(labels, preds)
        tn = np.diag(cm)  # True negatives
        fn = cm.sum(axis=1) - tn  # False negatives
        npv_per_class = tn / (tn + fn + np.finfo(float).eps)  # 防止除零错误
        return np.mean(npv_per_class)

    def _auprc(self, all_probas, labels, weight='macro'):
        """
        计算 AUC-PR（Precision-Recall曲线下的面积）
        :param all_probas: 所有类别的预测概率
        :param labels: 真实标签
        :param weight: 加权方法，默认为 'macro'
        :return: AUC-PR
        """
        if self._is_multiclass(labels):  # 多分类任务
            y_true_bin = label_binarize(labels, classes=np.unique(labels))  # 将标签二值化
            return metrics.average_precision_score(y_true_bin, all_probas, average=weight)  # 使用 OVR 方法
        else:  # 二分类时不加权
            return metrics.average_precision_score(labels, all_probas[:, 1])  # 二分类 AUC-PR

    def _kappa(self, preds, labels):
        """
        计算 Cohen's Kappa
        :param preds: 模型的预测结果
        :param labels: 真实标签
        :return: Cohen's Kappa
        """
        return metrics.cohen_kappa_score(labels, preds)

    def _mcc(self, preds, labels):
        """
        计算 MCC（Matthews Correlation Coefficient）
        :param preds: 模型的预测结果
        :param labels: 真实标签
        :return: MCC
        """
        return metrics.matthews_corrcoef(labels, preds)

    def calculate_best_threshold(self, probas, labels):
        """
        计算最佳分类阈值，该阈值对应最大约登指数。
        :param probas: 预测概率
        :param labels: 真实标签
        :return: 最佳分类阈值和对应的约登指数
        """
        # 初始化
        best_j_index = -1
        best_threshold = 0.5

        # 假设概率列为proba_1
        proba = probas[:, 1]  # 获取正类的预测概率

        thresholds = np.linspace(0, 1, 100)  # 从0到1之间取100个阈值
        for threshold in thresholds:
            # 根据阈值转换为预测标签
            preds = (proba >= threshold).astype(int)

            # 计算灵敏度和特异性
            sensitivity = self._sensitivity(preds, labels)
            specificity = self._specifity(preds, labels)

            # 计算约登指数
            j_index = sensitivity + specificity - 1

            # 更新最大约登指数和最佳阈值
            if j_index > best_j_index:
                best_j_index = j_index
                best_threshold = threshold

        return best_threshold, best_j_index

    def _youden(self, sensitivity, specificity):
        return sensitivity + specificity - 1

    def _net_benefit(self, probas, labels, thre: tuple, normalized: bool=True):#为什么不直接用_net_benefit计算得到的约登指数呢
        # 计算基于约登指数的净效益（Net Benefit）
        threshold = self.calculate_best_threshold(probas,
                                                  labels) if thre is None else thre  # 可以使用约登指数作为阈值
        # 根据最佳阈值重新计算预测标签
        preds_best_threshold = (probas[:, 1] >= threshold[0]).astype(int)  # TODO: 多分类也可以计算临床效用

        # 计算基于最佳阈值的净效益
        tp = np.sum((preds_best_threshold == 1) & (labels == 1))  # 真阳性
        fp = np.sum((preds_best_threshold == 1) & (labels == 0))  # 假阳性

        # 净效益计算公式：True Positives - False Positives
        if normalized:
            return (tp - fp) / np.sum(labels == 1)
        else:
            return (tp - fp)

    def bootstrap_resample(self, probas, labels, n_iterations=1000, weight='macro', k=2, frac=.8,
                           thre: tuple = None, normalized: bool = True):
        """
        使用bootstrap方法重采样并计算每次重采样的指标。
        :param probas: 预测概率
        :param labels: 真实标签
        :param n_iterations: bootstrap的次数
        :param weight: F1、AUC等加权方法
        :param k: Top-k准确率的k值
        :return: 重采样得到的所有指标
        """
        metrics_dict = {
            'accuracy': [],
            'topk_accuracy': [],
            'f1': [],
            'auroc': [],
            'sensitivity': [],
            'specificity': [],
            'ppv': [],
            'npv': [],
            'auprc': [],
            'kappa': [],
            'mcc': [],
            'youden_index': [],  # 新增约登指数
            'net_benefit': []  # 新增净效益
        }

        # 对每次迭代进行重采样
        for _ in range(n_iterations):
            # 从原始数据中随机抽取样本（80%样本）
            indices = np.random.choice(len(labels), int(frac * len(labels)), replace=True)
            probas_resampled = probas[indices]
            labels_resampled = labels[indices]
            preds_resampled = np.argmax(probas_resampled, axis=1)  # 获取预测的标签

            # 计算各项指标，并保存到字典中
            metrics_dict['accuracy'].append(self._acc(preds_resampled, labels_resampled))
            metrics_dict['f1'].append(self._f1(preds_resampled, labels_resampled, weight=weight))
            metrics_dict['auroc'].append(self._auroc(probas_resampled, labels_resampled, weight=weight))
            metrics_dict['sensitivity'].append(self._sensitivity(preds_resampled, labels_resampled, weight=weight))
            metrics_dict['specificity'].append(self._specifity(preds_resampled, labels_resampled))
            metrics_dict['ppv'].append(self._ppv(preds_resampled, labels_resampled, weight=weight))
            metrics_dict['npv'].append(self._npv(preds_resampled, labels_resampled))
            metrics_dict['auprc'].append(self._auprc(probas_resampled, labels_resampled, weight=weight))
            metrics_dict['kappa'].append(self._kappa(preds_resampled, labels_resampled))
            metrics_dict['mcc'].append(self._mcc(preds_resampled, labels_resampled))
            if probas.shape[1] <= 2:
                metrics_dict['youden_index'].append(self._youden(sensitivity=metrics_dict['sensitivity'][-1],
                                                                 specificity=metrics_dict['specificity'][-1]))

                metrics_dict['net_benefit'].append(
                    self._net_benefit(probas=probas_resampled, labels=labels_resampled, thre=thre,
                                      normalized=normalized))

            if probas.shape[1] >= 3:
                metrics_dict['topk_accuracy'].append(self._topk_acc(probas_resampled, labels_resampled, k=k))

        return metrics_dict
