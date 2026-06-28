import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

from 分型与坏死预测.base.base import Base


class DCA(Base):
    """ Simple usage:
    dca = DCA('probas', 'labels')
    nbs = list()
    root_list = [...]
    model_names = [...]
    nb_all = dca.net_benifit_all('any results.csv will be applicable')
    for root in root_list:
        nbs.append(dca.net_benifit(root))
    dca.plot_dca(nbs, model_names) """

    def __init__(self, proba_name, label_name, class_index = 1):
        self.proba_name = proba_name
        self.label_name = label_name
        self.nb_all = None
        self.class_index = class_index

    def net_benifit(self, root, thre_limit=100):
        """ Calculate a net benifit for one model """
        df = self.load_df(root)
        df = df.dropna(subset=[self.label_name])
        probas = df[self.proba_name].values
        labels = df[self.label_name].values
        labels = (labels == self.class_index).astype(int)
        total = len(labels)
        thre_group = []
        for i in range(0, thre_limit, 1):
            thre_group.append(i / 100)
        net_b = np.array([])  # (len of thre_limit, )
        for thre in thre_group:
            preds = probas > thre
            tn, fp, fn, tp = metrics.confusion_matrix(labels, preds).ravel()#ravel()是展平
            nb_tmp = tp / total - fp / total * (thre / (1 - thre))
            net_b = np.append(net_b, nb_tmp)
        return net_b

    def net_benefit_all(self, root, thre_limit=100, true_positive_reward=1, false_positive_cost=1):
        """ Net benefit for all at different thresholds """
        df = self.load_df(root)
        df = df.dropna(subset=[self.label_name])
        labels = df[self.label_name].values
        labels = (labels == self.class_index).astype(int)
        total = len(labels)

        # 初始化净收益数组
        self.nb_all = np.zeros(thre_limit)

        # 遍历不同的阈值
        for i, thre in enumerate(np.linspace(0, 1, thre_limit)):
            # 根据阈值将预测转换为二分类结果

            tp = np.sum(labels == 1)  # True positives
            fp = np.sum(labels == 0)  # False positives

            # 计算净收益
            nb = tp / total - fp / total * (thre / (1 - thre))
            # 注意：通常净收益不需要除以total，除非你想得到标准化的净收益
            # 如果需要标准化，则应该是 (nb / total)

            # 将净收益存储到数组中
            self.nb_all[i] = nb

        return self.nb_all

    def single_benifit(self, root, threshold=.5, model_name=None):
        """ Calculate a net benifit for one preset threshold, which can be used for model interpretation """
        df = self.load_df(root)
        df = df.dropna(subset=[self.label_name])
        probas = df[self.proba_name].values
        labels = df[self.label_name].values
        labels = (labels == self.class_index).astype(int)
        total = len(labels)
        preds = probas > threshold
        tn, fp, fn, tp = metrics.confusion_matrix(labels, preds).ravel()
        net_benifit = tp / total - fp / total * (threshold / (1 - threshold))
        model_name = model_name if model_name is not None else root
        print(f' For model [{model_name}] with a threshold of {threshold}, the net benifit is {net_benifit} ')
        return net_benifit

    def calc_youden(self, root):
        df = self.load_df(root)
        df = df.dropna(subset=[self.label_name])
        probas = df[self.proba_name].values
        labels = df[self.label_name].values
        labels = (labels == self.class_index).astype(int)
        fpr, tpr, thre = metrics.roc_curve(labels, probas)
        self.youden_cutoff = thre[np.argmax(tpr - fpr)] * 100
        if self.youden_cutoff < 7:
            self.youden_cutoff += 1
        print(f'The Youden cutoff is {self.youden_cutoff}')
        return self.youden_cutoff

    def plot_dca(self, nbs, model_names=None, color='#DE7833', thre_limit=100, all_color='gray', ylim=[-.1, 0.5],
                 fontdict={'family': 'Arial', 'fontsize': 15}, grid=False,
                 legend_loc='upper right', show=True, figure_size=[6.8, 6], normalized=False):
        """ Plot DCA curve(s) @ zez
        nbs, model_names, and color reguire lists """

        plt.figure(figsize=(figure_size[0], figure_size[1]))
        thre_group = []
        for i in range(0, thre_limit, 1):
            thre_group.append(i / 100)

        if normalized:
            min_values = [np.min(nb) for nb in nbs]  # 获取每个数组的最小值
            max_values = [np.max(nb) for nb in nbs]  # 获取每个数组的最大值
            nbs = [(nb - min_val) / (max_val - min_val) if max_val != min_val else np.zeros_like(nb)
                   for nb, min_val, max_val in zip(nbs, min_values, max_values)]

            # 对 self.nb_all 进行标准化/归一化
            min_all = np.min(self.nb_all)
            max_all = np.max(self.nb_all)
            if max_all != min_all:
                self.nb_all = (self.nb_all - min_all) / (max_all - min_all)
            else:
                self.nb_all = np.zeros_like(self.nb_all)  # 避免除以零

        plt.plot(thre_group, self.nb_all, color=all_color, lw=2, label='All')
        plt.plot(thre_group, np.zeros_like(thre_group), color='k', lw=2, linestyle=':', label='None')

        plt.plot(thre_group, nbs, color=color, label=model_names, lw = 2.5)
        youden_x = self.youden_cutoff / 100  # 因为 thre_group 使用了百分比转换为小数
        youden_y = np.interp(youden_x, thre_group, nbs)  # 使用插值找到对应的净收益
        # 绘制指向Youden阈值的箭头
        plt.annotate(f"Youden's index: {round(self.youden_cutoff, 3)}%", xy=(youden_x, youden_y),
                     xytext=(youden_x + .04, youden_y+.05),
                     arrowprops=dict(facecolor='red', arrowstyle='->', connectionstyle="arc3,rad=.2"),
                     fontsize = 12)

        # plt.ylim(ylim[0], ylim[1])
        plt.ylim(-0.1, 1.0)
        plt.xlabel(xlabel='Threshold probability', fontdict=fontdict, fontsize=13)
        plt.ylabel(ylabel='Net benefit', fontdict=fontdict, fontsize = 13)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(grid)
        plt.legend(loc=legend_loc)

        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.legend(frameon = False, fontsize = 10)

        plt.savefig('../results/efficientnet/g_dca_external.pdf', dpi=300)

        if show:
            plt.show()

a = DCA('garden_prob_1', 'true_garden', class_index = 1)
net_b = a.net_benifit("../results/efficientnet/external_predictions.csv")
net_all = a.net_benefit_all("../results/efficientnet/external_predictions.csv")
net_benefit=a.single_benifit("../results/efficientnet/external_predictions.csv")
youden_cutoff = a.calc_youden("../results/efficientnet/external_predictions.csv")
a.plot_dca(net_b, model_names='Efficientnetv2', color='#9BABD2')

#黄色 #DE7833
#红色 #C16E71
#蓝色 #8CA3C3
#绿色#b8d0c3
