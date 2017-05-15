# --coding: utf-8--
# BY pj
# Date: 17-4-21
import math
import numpy as np
from PIL import Image

"""简单的高斯滤波"""


def gaussian1d(x, sigma):
    """
    formula:
        (1/sqrt(2*pi*sigma^2))*exp(-x^2/(2*sigma^2))
    :param x:
    :param sigma:
    :return:
    """
    sqrt_2_pi = 2.5066282746310002
    return (1 / sqrt_2_pi * sigma) * math.exp(-(x * x) / (2 * sigma * sigma))


def gaussian2d(x, y, sigma):
    """
    formula:
        (1/(2*pi*sigma))*exp(-(x^2+y^2)/(2*sigma^2))
    :param x:
    :param y:
    :param sigma:
    :return:
    """
    return (1 / (2 * np.pi * sigma * sigma)) * np.exp(
        -(x * x + y * y) / (2 * sigma * sigma))


def gen_gaussian_kernel(sigma, size):
    """
    generate gaussian kernel, this method could be slowly.
    I will promote this after.
    :param sigma:
    :param size:
    :return:
    """
    grid = np.zeros((size, size))
    center = size // 2
    for i in range(size):
        for j in range(size):
            grid[i, j] = gaussian2d(i - center, j - center, sigma)
    
    grid_sum = sum(sum(grid))
    return grid / grid_sum


def single_pixel_filter(im, x, y, filter_kernel):
    half_kernel_size = filter_kernel.shape[0] // 2
    out_value = 0
    for i in range(filter_kernel.shape[0]):
        for j in range(filter_kernel.shape[1]):
            out_value += filter_kernel[i, j] * im[x + i - half_kernel_size,
                                                  y + j - half_kernel_size]
    
    return out_value


def gray_gaussian_filter(im, sigma, size):
    gaussian_kernel = gen_gaussian_kernel(sigma, size)
    border = size // 2
    work_im = np.lib.pad(im, border,
                         'constant', constant_values=(0, 0))
    out = np.copy(im)
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            out[i, j] = single_pixel_filter(work_im, i + border, j + border,
                                            gaussian_kernel)
    
    return out


a = Image.open("/home/pj/scode/pjdlib/examples/faces/2008_002470.jpg")
a = a.convert('L')  # makes it greyscale
a = np.array(a)

out = gray_gaussian_filter(a, 1, 3)

print(out.shape)
