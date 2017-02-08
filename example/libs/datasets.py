# encoding: utf-8

"""
@author: liubo
@software: PyCharm
@file: batch_norm.py
@contact: ustb_liubo@qq.com
@annotation: batch_norm
"""

import tensorflow.examples.tutorials.mnist.input_data as input_data
from .dataset_utils import *


def MNIST(one_hot=True):
    """Returns the MNIST dataset.

    Returns
        mnist : DataSet
            DataSet object w/ convenienve props for accessing
            train/validation/test sets and batches.
    """
    return input_data.read_data_sets('MNIST_data/', one_hot=one_hot)


def CIFAR10():
    # plt.imshow(np.transpose(np.reshape(cifar.train.images[10], (3, 32, 32)), [1, 2, 0]))
    Xs, ys = cifar10_load()
    return Dataset(Xs, ys)
