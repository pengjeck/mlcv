# ！/usr/bin/python
# -*- coding: utf-8 -*-
# Created by PJZero on 17-5-5


def select_b_from_a(a, b):
    """
    c(a, b)
    :return:
    """
    assert a > b
    if b > a - b:
        b = a - b
    
    def accumulate(arr, method='multi'):
        """
        连乘，连加
        :param arr:
        :param method: add or multi
        :return:
        """
        if method == 'multi':
            result = 1
            for i in arr:
                result *= i
            return result
        elif method == 'add':
            result = 0
            for i in arr:
                result += i
            return result
        else:
            raise AttributeError("only support add and multi")
    
    up = [i for i in range(a - b + 1, a + 1)]
    down = [i for i in range(1, b + 1)]
    return accumulate(up) / accumulate(down)
