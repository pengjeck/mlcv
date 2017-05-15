# ！/usr/bin/python
# -*- coding: utf-8 -*-
# Created by PJZero on 17-4-30

"""
通过scikit-learn库实现的梯度下降法
"""
import numpy as np
from sklearn.linear_model import SGDRegressor


def gen_line_data(sample_num=100):
    """
    y = 3*x1 + 4*x2
    :return:
    """
    x1 = np.linspace(0, 9, sample_num)
    x2 = np.linspace(4, 13, sample_num)
    x = np.concatenate(([x1], [x2]), axis=0).T
    y = np.dot(x, np.array([3, 4]).T)  # y 列向量
    return x, y


def sgd(samples, y):
    """
    随机梯度下降
    :param samples:
    :param y:
    :return:
    """
    c_sgd = SGDRegressor()
    c_sgd.fit(samples, y)
    return c_sgd.coef_


if __name__ == '__main__':
    x, y = gen_line_data()
    print(sgd(x, y))  # [ 2.84988732  4.04590773] 非常接近[3, 4]
