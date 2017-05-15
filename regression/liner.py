# ！/usr/bin/python
# -*- coding: utf-8 -*-
# Created by PJZero on 17-4-22

import numpy as np


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


def line_regression_mat(samples, labels):
    """
    线性回归：具体请参照西瓜书
    :param samples:
    :param labels:
    :return:
    """
    assert isinstance(samples, np.ndarray)
    assert isinstance(labels, np.ndarray)
    return np.dot(np.linalg.inv(np.dot(samples.T, samples)),
                  np.dot(samples.T, labels))


if __name__ == '__main__':
    x, y = gen_line_data()
    weight = line_regression_mat(x, y)
    print(weight)  # [ 3.  4.]
