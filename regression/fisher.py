# ！/usr/bin/python
# -*- coding: utf-8 -*-
# Created by PJZero on 17-5-3


from mlcvkit.gen_data import make_multilabel_classification
import numpy as np
import matplotlib.pyplot as plt

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


x, y = make_multilabel_classification(n_samples=20, n_features=2,
                                      n_labels=1, n_classes=1,
                                      random_state=2)


def cal_cov_and_avg(samples):
    """
    给定一个类别的数据，计算协方差矩阵和平均向量
    :param samples:
    :return:
    """
    u1 = np.mean(samples, axis=0)
    cov_m = np.zeros((samples.shape[1], samples.shape[1]))
    for s in samples:
        t = s - u1
        cov_m += t * t.reshape(2, 1)
    return cov_m, u1


def fisher(c_1, c_2):
    """
    fisher算法实现(请参考上面推导出来的公式，那个才是精华部分)
    :param c_1:
    :param c_2:
    :return:
    """
    cov_1, u1 = cal_cov_and_avg(c_1)
    cov_2, u2 = cal_cov_and_avg(c_2)
    s_w = cov_1 + cov_2
    u, s, v = np.linalg.svd(s_w)  # 奇异值分解
    s_w_inv = np.dot(np.dot(v.T, np.linalg.inv(np.diag(s))), u.T)
    return np.dot(s_w_inv, u1 - u2)


def judge(sample, w, c_1, c_2):
    """
    true 属于1
    false 属于2
    :param sample:
    :param w:
    :param center_1:
    :param center_2:
    :return:
    """
    u1 = np.mean(c_1, axis=0)
    u2 = np.mean(c_2, axis=0)
    center_1 = np.dot(w.T, u1)
    center_2 = np.dot(w.T, u2)
    pos = np.dot(w.T, sample)
    return abs(pos - center_1) < abs(pos - center_2)


index1 = np.array([index for (index, value) in enumerate(y) if value == 0])
index2 = np.array([index for (index, value) in enumerate(y) if value == 1])

c_1 = x[index1]
c_2 = x[index2]
w = fisher(c_1, c_2)  # 得到参数w
out = judge(c_1[1], w, c_1, c_2)
print(out)

plt.scatter(c_1[:, 0], c_1[:, 1], c=get_color(0))
plt.scatter(c_2[:, 0], c_2[:, 1], c=get_color(1))
line_x = np.arange(min(np.min(c_1[:, 0]), np.min(c_2[:, 0])),
                   max(np.max(c_1[:, 0]), np.max(c_2[:, 0])),
                   step=1)

line_y = - (w[0] * line_x) / w[1]
plt.plot(line_x, line_y)
plt.show()
