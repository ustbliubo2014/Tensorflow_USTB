# encoding: utf-8

"""
@author: liubo
@software: PyCharm
@file: 01_linear_regression.py
@time: 2017/1/24 9:54
@contact: ustb_liubo@qq.com
@annotation: 01_linear_regression
"""
import sys
import logging
from logging.config import fileConfig
import os
import tensorflow as tf
import numpy as np
import pdb

reload(sys)
sys.setdefaultencoding("utf-8")
# fileConfig('logger_config.ini')
# logger_error = logging.getLogger('errorhandler')


# y = 2*x
train_x = np.asarray(np.linspace(-1, 1, 10000))
train_y = 2 * train_x

# 输入输出使用tf.placeholder, 要求的参数用tf.Variable
dtype = tf.float32
X = tf.placeholder(dtype)
Y = tf.placeholder(dtype)
W = tf.Variable(0.0, name='W')

# 线性函数
predict = tf.mul(W, X)
# 损失函数
loss = tf.pow(tf.sub(Y, predict), 2)
# 优化函数
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

# 定义session
sess = tf.Session()
# 初始化所有变量
init = tf.initialize_all_variables()
sess.run(init)

# 迭代 : 用sess.run(optimizer进行迭代, 用sess.run(loss获得loss, 用sess.run(predict, 进行预测, 用sess.run(W) 得到参数值
# 在这里, optimizer, loss, predict 都是一个op, 用sess.run进行实际执行(用feed_dict进行提供数据)
for i in range(10000):
    sess.run(optimizer, feed_dict={X: train_x[i], Y: train_y[i]})
    if i % 500 == 0:
        print i, sess.run(loss, feed_dict={X: train_x[i], Y: train_y[i]}), sess.run(W)
