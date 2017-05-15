# ！/usr/bin/python
# -*- coding: utf-8 -*-
# Created by PJZero on 17-5-5

"""
画出正态分布的图。
"""
import numpy as np
import matplotlib.pyplot as plt


def cal_f(x, u=0, sigma=1):
    """
    计算正态分布的概率密度值
    :param x:
    :param u:
    :param sigma:
    :return:
    """
    return 1 / np.sqrt(2 * np.pi * np.power(sigma, 2)) \
           * np.power(np.e, - np.power(x - u, 2) / (2 * np.power(sigma, 2)))


u = 0
sigma = 1
x = np.arange(-4 * sigma, 4 * sigma, 0.1)
y = cal_f(x, u, sigma)
# y = 1 / np.sqrt(2 * np.pi * np.power(sigma, 2)) \
#     * np.power(np.e, - np.power(x - u, 2) / (2 * np.power(sigma, 2)))

plt.figure()
plt.plot(x, y)
plt.show()
