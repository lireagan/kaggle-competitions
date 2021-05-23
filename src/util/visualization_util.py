#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = "reagan.llx"

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap


def plot_2d_decision_regions(X, y, classifiers, test_idx=None):
    """多个分类模型决策边界二维可视化.

    评估多个分类模型，并可视化效果. 其中输入的标注集需要提前进行降维，降低到2维才能进行可视化

    Args:
        X: 标注集的特征矩阵. [sample_size, 2]
        y: 标注集的标签向量. [sample_size,]
        classifiers: sklearn中定义好的分类模型. {'name1': classifier1, 'name2': classifier2 ...}
        test_idx: 测试集序号

    """
    figure = plt.figure(figsize=(10, 4))
    # 设置类别图例和颜色
    markers = ('s', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    for i, (name, clf) in enumerate(classifiers.items()):
        # 创建子图
        ax = plt.subplot(1, len(classifiers), i + 1)

        # 训练分类器
        X_train, y_train = X, y
        if test_idx:
            X_train, y_train = X[:test_idx.start, :], y[:test_idx.start]
        clf.fit(X_train, y_train)

        # 通过分类器预测构造决策边界内所有点
        # 1) 枚举出X特征集合内的所有可能的特征组合，形成枚举样本集
        # 2) 用分类器对枚举样本集进行预测，得到每个样本的预测label
        # 3) 此时，每个预测label下的所有枚举样本组成了该label的决策边界
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1  # 第一个特征取值范围作为横轴
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1  # 第二个特征取值范围作为纵轴
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.05),
                               np.arange(x2_min, x2_max, 0.05))  # resolution是网格剖分粒度
        Z = clf.predict(np.array([xx1.ravel(), xx2.ravel()]).T)  # ravel数组展平
        Z = Z.reshape(xx1.shape)  # Z是列向量

        # 通过等高线来描绘决策边界
        ax.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)  # alpha是透明度，0透明1不透明

        # 描绘标记样本的样本点
        for idx, label in enumerate(np.unique(y)):
            ax.scatter(x=X[y == label, 0], y=X[y == label, 1],
                       alpha=0.8,
                       color=cmap(idx),
                       marker=markers[idx], label=label)

        # 如有，凸显出测试样本
        if test_idx:
            X_test, y_test = X[test_idx, :], y[test_idx]
            ax.scatter(X_test[:, 0],
                       X_test[:, 1],
                       color=None,
                       alpha=1.0,
                       linewidths=1,
                       marker='x',
                       s=55, label='test set')  # c设置颜色，测试集不同类别的实例点画图不区别颜色

        ax.set_xlim(xx1.min(), xx1.max())
        ax.set_ylim(xx2.min(), xx2.max())
        ax.set_title(name)
        ax.legend(loc='upper left')

    figure.subplots_adjust(left=.04, right=.98)
    plt.show()
