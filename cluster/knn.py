# ！/usr/bin/python
# -*- coding: utf-8 -*-
# Created by PJZero on 17-5-3

from sklearn.datasets import make_multilabel_classification
import numpy as np
import matplotlib.pyplot as plt
import operator

_color_table = ['#99CC99', '#FFCC00', '#006699', '#33CC99', '#66CCCC',
                '#CC6600', '#333399', '#99CC00', '#FF6600', '#FFCC99',
                '#FF9900', '#009999', '#CC3366', '#FFFFFF', '#99CCFF',
                '#FFFF00', '#FFCC33', '#FF9933', '#FFFF33', '#CCCCFF',
                '#FFCCCC', '#FF0033', '#CC0033', '#003399', '#666699',
                '#CCFF99', '#FFFFCC', '#CCCC00', '#99CC33', '#FF9966',
                '#336699', '#CCFFCC', '#99CCCC', '#CCCC44', '#0099CC',
                '#CCFFFF', '#0066CC', '#FFFF99', '#CC3333', '#CCCCCC',
                '#6699CC', '#FF6666', '#66CCFF', '#663399', '#339933']


def get_color(i):
    return _color_table[i % len(_color_table)]


def knn(x, y, k, predict_x):
    """
    knn算法实现，使用欧氏距离
    :param x: 样本值
    :param y: 标签
    :param k: 个数
    :return:
    """
    assert isinstance(y, np.ndarray)
    y = y.flatten('F')
    
    def cal_distance(a, b):
        return np.sqrt(np.sum(np.power(a - b, 2), axis=0))
    
    dists = {
        
    }
    for (index, sample) in enumerate(x):
        dists[index] = cal_distance(sample, predict_x)
    
    k_sample = sorted(dists.items(), key=operator.itemgetter(1))[:k]
    k_labels = y[[key for (key, value) in k_sample]]
    counters = {
        
    }
    for k in k_labels:
        if k not in counters.keys():
            counters[k] = 1
        else:
            counters[k] += 1
    return sorted(counters.items(), key=operator.itemgetter(1))[0]


sample_num = 30
x, y = make_multilabel_classification(n_samples=sample_num, n_features=2,
                                      n_classes=1, n_labels=1, random_state=2)

c_1 = x[np.where(y == 0)[0]]  # 类别1
c_2 = x[np.where(y == 1)[0]]  # 类别2
predict_x = np.array([30, 30])
result = knn(x, y, 5, predict_x)

plt.figure()
plt.subplot(111)
s_c1 = plt.scatter(c_1[:, 0], c_1[:, 1], c=get_color(0), )
s_c2 = plt.scatter(c_2[:, 0], c_2[:, 1], c=get_color(1))
s_c3 = plt.scatter(predict_x[0], predict_x[1], c=get_color(2))
plt.legend((s_c1, s_c2, s_c3),
           ('label 0', 'label 1', 'pre points'),
           scatterpoints=1,
           loc='lower left',
           fontsize=8)

plt.annotate("belong to '{}'".format(result[0]),
             xy=(predict_x[0], predict_x[1]), xytext=(-20, 20),
             textcoords='offset points', ha='right', va='bottom',
             bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

plt.show()
