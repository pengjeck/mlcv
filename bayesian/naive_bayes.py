# ！/usr/bin/python
# -*- coding: utf-8 -*-
# Created by PJZero on 17-5-5

"""
naive bayes(朴素贝叶斯)
    describe:基于贝叶斯公式，并添加了属性条件独立性假设。
    blog: http://blog.csdn.net/pengjian444/article/details/72179416
"""

import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()
y = iris.target
x = iris.data


def naive_bayes(x, y, predict):
    unique_y = list(set(y))
    label_num = len(unique_y)
    sample_num, dim = x.shape
    joint_p = [1] * label_num
    # 把所有的类别都过一遍，计算P(c)
    for (label_index, label) in enumerate(unique_y):
        p_c = len(y[y == label]) / sample_num
        for (feature_index, x_i) in enumerate(predict):
            tmp = x[y == label]
            joint_p[label_index] *= len(
            [t for t in tmp[:, feature_index] if t == x_i]) / len(tmp)
        joint_p[label_index] *= p_c
    
    tmp = joint_p[0]
    max_index = 0
    for (i, p) in enumerate(joint_p):
        if tmp < p:
            tmp = p
            max_index = i
    
    return unique_y[max_index]


out = naive_bayes(x, y, np.array([5.9, 3., 5.1, 1.8]))
print(out)
