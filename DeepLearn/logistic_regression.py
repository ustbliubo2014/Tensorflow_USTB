# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: logistic_regression.py
@time: 2016/7/22 20:28
@contact: ustb_liubo@qq.com
@annotation: logistic_regression
"""
import sys

reload(sys)
sys.setdefaultencoding("utf-8")
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='logistic_regression.log',
                    filemode='a+')
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('data/', one_hot=True)
trainimg = mnist.train.images
trainlabel = mnist.train.labels
testimg = mnist.test.images
testlabel = mnist.test.labels

if __name__ == '__main__':
    pass
