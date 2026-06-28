import numpy as np
import pandas as pd

from 分型与坏死预测.base.base import Base


class Create_baseline(Base):

    def create_baseline_proba(self, labels):
        """
        根据标签的比例生成proba作为基线数据。
        :param result_df: 包含标签的结果数据框
        :param label_name: 标签列的名称，默认为 'labels'
        :param proba_prefix: 基线proba的前缀，默认为 'proba_'
        :return: 基线概率（numpy array）
        """

        # 获取标签的类别分布（频率）
        unique_labels, counts = np.unique(labels, return_counts=True)
        label_dist = counts / len(labels)  # 计算每个标签类别的频率

        return np.array([label_dist for _ in range(len(labels))])

    def create(self, path, label_name, save_path, proba_prefix='proba_'):
        """
        创建基线proba并保存到指定路径
        :param path: 原始数据文件路径
        :param label_name: 标签列的名称
        :param save_path: 保存基线proba数据的路径
        :param proba_prefix: 基线proba的前缀，默认为 'proba_'
        """
        # 读取数据
        _, labels = self.read_rd(path, label_name=label_name)

        # 生成基线概率
        baseline_proba = self.create_baseline_proba(labels)

        result_df = pd.DataFrame({label_name: labels})
        # 将基线概率加入数据框
        for i in range(baseline_proba.shape[1]):
            result_df[f'{proba_prefix}{i}'] = baseline_proba[:, i]

        # 保存结果
        self.save_df(result_df, save_path)
