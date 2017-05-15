# ！/usr/bin/python
# -*- coding: utf-8 -*-
# Created by PJZero on 17-4-30

"""
梯度下降法的各种具体实现，包括：
    BGD：批量梯度下降法（Batch Gradient Descent，简称BGD）
    SGD：随机梯度下降法（Stochastic Gradient Descent，简称SGD）
    MBGD：小批量梯度下降法（Mini-batch Gradient Descent，简称MBGD）
"""
import numpy as np
import random


def gen_Rosenbrock_func_data(sample_num=100):
    """
    f(x, y) = (1 - x)^2 + 100(y - x^2)^2
    wiki: https://zh.wikipedia.org/wiki/Rosenbrock%E5%87%BD%E6%95%B8
    :param sample_num:
    :return:
    """
    x1 = np.random.randint(-(sample_num // 3),
                           2 * (sample_num // 3),
                           sample_num)
    x2 = np.random.randint(-sample_num * 2, sample_num * 2, sample_num)
    x = np.concatenate(([x1], [x2]), axis=0).T
    y = np.power(1 - x1, 2) + 100 * np.power((x2 - np.power(x1, 2)), 2)
    return x, y


def bgd(samples, y, step_size=0.01, max_iter_count=10000):
    sample_num, dim = samples.shape
    y = y.flatten()
    w = np.ones((dim,), dtype=np.float32)
    loss = 10
    iter_count = 0
    while loss > 0.001 and iter_count < max_iter_count:
        loss = 0
        error = np.zeros((dim,), dtype=np.float32)
        for i in range(sample_num):
            predict_y = np.dot(w.T, samples[i])
            for j in range(dim):
                error[j] += (y[i] - predict_y) * samples[i][j]
        
        for j in range(dim):
            w[j] += step_size * error[j] / sample_num
        
        for i in range(sample_num):
            predict_y = np.dot(w.T, samples[i])
            error = (1 / (sample_num * dim)) * np.power((predict_y - y[i]), 2)
            loss += error
        
        print("iter_count: ", iter_count, "the loss:", loss)
        iter_count += 1
    return w


def sgd(samples, y, step_size=0.01, max_iter_count=10000):
    """
    随机梯度下降法
    :param samples: 样本
    :param y: 结果value
    :param step_size: 每一接迭代的步长
    :param max_iter_count: 最大的迭代次数
    :param batch_size: 随机选取的相对于总样本的大小
    :return:
    """
    sample_num, dim = samples.shape
    y = y.flatten()
    w = np.ones((dim,), dtype=np.float32)
    loss = 10
    iter_count = 0
    while loss > 0.001 and iter_count < max_iter_count:
        loss = 0
        error = np.zeros((dim,), dtype=np.float32)
        for i in range(sample_num):
            predict_y = np.dot(w.T, samples[i])
            for j in range(dim):
                error[j] += (y[i] - predict_y) * samples[i][j]
                w[j] += step_size * error[j] / sample_num
        
        # for j in range(dim):
        #     w[j] += step_size * error[j] / sample_num
        
        for i in range(sample_num):
            predict_y = np.dot(w.T, samples[i])
            error = (1 / (sample_num * dim)) * np.power((predict_y - y[i]), 2)
            loss += error
        
        print("iter_count: ", iter_count, "the loss:", loss)
        iter_count += 1
    return w


def mbgd(samples, y, step_size=0.01, max_iter_count=10000, batch_size=0.2):
    """
    MBGD（Mini-batch gradient descent）小批量梯度下降：每次迭代使用b组样本
    :param samples:
    :param y:
    :param step_size:
    :param max_iter_count:
    :param batch_size:
    :return:
    """
    sample_num, dim = samples.shape
    y = y.flatten()
    w = np.ones((dim,), dtype=np.float32)
    # batch_size = np.ceil(sample_num * batch_size)
    loss = 10
    iter_count = 0
    while loss > 0.001 and iter_count < max_iter_count:
        loss = 0
        error = np.zeros((dim,), dtype=np.float32)
        
        # batch_samples, batch_y = select_random_samples(samples, y,
        # batch_size)
        
        index = random.sample(range(sample_num),
                              int(np.ceil(sample_num * batch_size)))
        batch_samples = samples[index]
        batch_y = y[index]
        
        for i in range(len(batch_samples)):
            predict_y = np.dot(w.T, batch_samples[i])
            for j in range(dim):
                error[j] += (batch_y[i] - predict_y) * batch_samples[i][j]
        
        for j in range(dim):
            w[j] += step_size * error[j] / sample_num
        
        for i in range(sample_num):
            predict_y = np.dot(w.T, samples[i])
            error = (1 / (sample_num * dim)) * np.power((predict_y - y[i]), 2)
            loss += error
        
        print("iter_count: ", iter_count, "the loss:", loss)
        iter_count += 1
    return w


# region sosenbrock test

def cal_rosenbrock(x1, x2):
    """
    计算rosenbrock函数的值
    :param x1:
    :param x2:
    :return:
    """
    return (1 - x1) ** 2 + 100 * (x2 - x1 ** 2) ** 2


def cal_rosenbrock_prax(x1, x2):
    return -2 + 2 * x1 - 400 * (x2 - x1 ** 2) * x1


def cal_rosenbrock_pray(x1, x2):
    return 200 * (x2 - x1 ** 2)


def for_rosenbrock_func(max_iter_count=100000, step_size=0.001):
    pre_x = np.zeros((2,), dtype=np.float32)
    loss = 10
    iter_count = 0
    while loss > 0.001 and iter_count < max_iter_count:
        error = np.zeros((2,), dtype=np.float32)
        error[0] = cal_rosenbrock_prax(pre_x[0], pre_x[1])
        error[1] = cal_rosenbrock_pray(pre_x[0], pre_x[1])
        
        for j in range(2):
            pre_x[j] -= step_size * error[j]
        
        loss = cal_rosenbrock(pre_x[0], pre_x[1])  # 最小值为0
        
        # print("iter_count: ", iter_count, "the loss:", loss)
        iter_count += 1
    print(loss)
    return pre_x


# endregion


if __name__ == '__main__':
    samples, y = gen_Rosenbrock_func_data()
    w = for_rosenbrock_func()
    print(w)
