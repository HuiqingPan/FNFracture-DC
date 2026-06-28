import pandas as pd
import numpy as np
from sklearn import metrics


class test_base_label:

    def _safe_divide(self, a, b):
        """
        安全除法，避免分母为0。
        """
        if b == 0:
            return np.nan
        return a / b

    def _print_mode(self, data, mode='ci', round=3, name='Metric'):
        """
        按照均值 ± 标准差或95%CI打印结果。
        """
        data = np.array(data, dtype=float)
        data = data[~np.isnan(data)]

        if len(data) == 0:
            print(f'{name}: NA')
            return

        if mode == 'sd':
            print(f'{name}: {np.mean(data):.{round}f} ± {np.std(data):.{round}f}')
        else:
            lower = np.percentile(data, 2.5)
            upper = np.percentile(data, 97.5)
            print(f'{name}: {np.mean(data):.{round}f} ({lower:.{round}f}-{upper:.{round}f})')

    def _read_table(self, path):
        """
        根据文件后缀读取csv或excel。
        """
        if path.endswith('.csv'):
            return pd.read_csv(path, encoding='utf-8')
        elif path.endswith('.xlsx') or path.endswith('.xls'):
            return pd.read_excel(path)
        else:
            raise ValueError('Only csv, xlsx and xls files are supported.')

    def _format_label_series(self, x):
        """
        将标签统一格式化。
        如果是0.0、1.0、2.0这种标签，则转为0、1、2。
        """
        x_num = pd.to_numeric(x, errors='coerce')

        if x_num.notna().all():
            if np.all(np.isclose(x_num, np.round(x_num))):
                return x_num.astype(int).to_numpy()
            else:
                return x_num.astype(float).to_numpy()
        else:
            return x.astype(str).to_numpy()

    def read_label_rd(self, path, label_name, pred_name):
        """
        读取只有真实标签和预测标签的结果文件。
        :param path: 结果文件路径
        :param label_name: 真实标签列名
        :param pred_name: 预测标签列名
        :return: preds, labels, label_list
        """
        df = self._read_table(path)

        data = df[[label_name, pred_name]].dropna().copy()
        labels = self._format_label_series(data[label_name])
        preds = self._format_label_series(data[pred_name])

        label_list = np.unique(np.concatenate([labels, preds]))

        try:
            label_list = np.sort(label_list)
        except Exception:
            label_list = sorted(label_list)

        return preds, labels, label_list

    def _calc_label_metrics_once(self, preds, labels, label_list):
        """
        对一次预测结果计算：
        ACC, macro-F1, weighted-F1, Kappa,
        总体Sensitivity / Specificity / PPV / NPV / F1,
        以及每一类的Sensitivity / Specificity / PPV / NPV / F1。
        """
        cm = metrics.confusion_matrix(labels, preds, labels=label_list)
        total = cm.sum()

        per_class_metrics = {}

        for i, label in enumerate(label_list):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = total - tp - fn - fp

            sensitivity = self._safe_divide(tp, tp + fn)
            specificity = self._safe_divide(tn, tn + fp)
            ppv = self._safe_divide(tp, tp + fp)
            npv = self._safe_divide(tn, tn + fn)
            f1 = self._safe_divide(2 * ppv * sensitivity, ppv + sensitivity)

            per_class_metrics[label] = {
                'sensitivity': sensitivity,
                'specificity': specificity,
                'ppv': ppv,
                'npv': npv,
                'f1': f1
            }

        sensitivity_list = [per_class_metrics[label]['sensitivity'] for label in label_list]
        specificity_list = [per_class_metrics[label]['specificity'] for label in label_list]
        ppv_list = [per_class_metrics[label]['ppv'] for label in label_list]
        npv_list = [per_class_metrics[label]['npv'] for label in label_list]
        f1_list = [per_class_metrics[label]['f1'] for label in label_list]

        overall_metrics = {
            'accuracy': metrics.accuracy_score(labels, preds),
            'macro_f1': metrics.f1_score(labels, preds, labels=label_list, average='macro', zero_division=0),
            'weighted_f1': metrics.f1_score(labels, preds, labels=label_list, average='weighted', zero_division=0),
            'kappa': metrics.cohen_kappa_score(labels, preds, labels=label_list),

            # 这里的 total 表示多分类 one-vs-rest 后的 macro-average
            'total_sensitivity': np.nanmean(sensitivity_list),
            'total_specificity': np.nanmean(specificity_list),
            'total_ppv': np.nanmean(ppv_list),
            'total_npv': np.nanmean(npv_list),
            'total_f1': np.nanmean(f1_list)
        }

        return overall_metrics, per_class_metrics

    def bootstrap_resample_label(self, preds, labels, label_list, n_iterations=1000, frac=.8):
        """
        针对只有预测标签和真实标签的结果进行bootstrap重采样。
        :param preds: 预测标签
        :param labels: 真实标签
        :param label_list: 所有类别
        :param n_iterations: bootstrap次数
        :param frac: 每次抽样比例，保持和原来代码风格一致，默认0.8
        :return: metrics_dict
        """
        metrics_dict = {
            'accuracy': [],
            'macro_f1': [],
            'weighted_f1': [],
            'kappa': [],

            'total_sensitivity': [],
            'total_specificity': [],
            'total_ppv': [],
            'total_npv': [],
            'total_f1': []
        }

        for label in label_list:
            metrics_dict[f'class_{label}_sensitivity'] = []
            metrics_dict[f'class_{label}_specificity'] = []
            metrics_dict[f'class_{label}_ppv'] = []
            metrics_dict[f'class_{label}_npv'] = []
            metrics_dict[f'class_{label}_f1'] = []

        for _ in range(n_iterations):
            indices = np.random.choice(len(labels), int(frac * len(labels)), replace=True)

            preds_resampled = preds[indices]
            labels_resampled = labels[indices]

            overall_metrics, per_class_metrics = self._calc_label_metrics_once(
                preds_resampled,
                labels_resampled,
                label_list
            )

            metrics_dict['accuracy'].append(overall_metrics['accuracy'])
            metrics_dict['macro_f1'].append(overall_metrics['macro_f1'])
            metrics_dict['weighted_f1'].append(overall_metrics['weighted_f1'])
            metrics_dict['kappa'].append(overall_metrics['kappa'])

            metrics_dict['total_sensitivity'].append(overall_metrics['total_sensitivity'])
            metrics_dict['total_specificity'].append(overall_metrics['total_specificity'])
            metrics_dict['total_ppv'].append(overall_metrics['total_ppv'])
            metrics_dict['total_npv'].append(overall_metrics['total_npv'])
            metrics_dict['total_f1'].append(overall_metrics['total_f1'])

            for label in label_list:
                metrics_dict[f'class_{label}_sensitivity'].append(per_class_metrics[label]['sensitivity'])
                metrics_dict[f'class_{label}_specificity'].append(per_class_metrics[label]['specificity'])
                metrics_dict[f'class_{label}_ppv'].append(per_class_metrics[label]['ppv'])
                metrics_dict[f'class_{label}_npv'].append(per_class_metrics[label]['npv'])
                metrics_dict[f'class_{label}_f1'].append(per_class_metrics[label]['f1'])

        return metrics_dict
