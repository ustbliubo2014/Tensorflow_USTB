# encoding: utf-8

"""
@author: liubo
@software: PyCharm
@file: 02_polynomial_regression.py
@time: 2017/1/24 15:45
@contact: ustb_liubo@qq.com
@annotation: 02_polynomial_regression
"""
import sys
import logging
from logging.config import fileConfig
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

reload(sys)
sys.setdefaultencoding("utf-8")
# fileConfig('logger_config.ini')
# logger_error = logging.getLogger('errorhandler')


n_observations = 100
train_x = np.linspace(-3, 3, n_observations)
train_y = np.sin(train_x) + np.random.uniform(-0.5, 0.5, n_observations)

dtype = tf.float32
X = tf.placeholder(dtype)
Y = tf.placeholder(dtype)

# 函数: W1*x + W2*x2 + W2*x3 + W4*x4 + b
Y_predict = tf.Variable(tf.random_normal([1]), name='bias')
for pow_i in range(1, 5):
    W = tf.Variable(tf.random_normal([1]), name='weight_{}'.format(pow_i))
    Y_predict = tf.add(tf.mul(tf.pow(X, pow_i), X), Y_predict)

loss = tf.reduce_sum(tf.pow(tf.sub(Y, Y_predict), 2))

optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

for e in range(10):     # 10个epoch
    for i in range(100):
        sess.run(optimizer, feed_dict={X: train_x[i], Y: train_y[i]})
    print sess.run(loss, feed_dict={X: train_x, Y: train_y})
