# encoding: utf-8

"""
@author: liubo
@software: PyCharm
@file: 04_softmax.py
@time: 2017/1/24 17:20
@contact: ustb_liubo@qq.com
@annotation: 04_softmax
"""
import sys
import logging
from logging.config import fileConfig
import os
import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data
import pdb

reload(sys)
sys.setdefaultencoding("utf-8")
# fileConfig('logger_config.ini')
# logger_error = logging.getLogger('errorhandler')


mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

print(mnist.train.num_examples, mnist.test.num_examples, mnist.validation.num_examples)
# (55000, 10000, 5000)
print(mnist.train.images.shape, mnist.train.labels.shape)
# ((55000, 784), (55000, 10))
print(np.min(mnist.train.images), np.max(mnist.train.images))
# (0.0, 1.0)

train_x, train_y, test_x, test_y = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
valid_x, valid_y = mnist.validation.images, mnist.validation.labels

n_input = 784
n_output = 10
dtype = tf.float32
X = tf.placeholder(dtype, [None, n_input])
Y = tf.placeholder(dtype, [None, n_output])
W = tf.Variable(tf.random_normal([n_input, n_output], stddev=0.1))
b = tf.Variable(tf.zeros([n_output]))

predict = tf.nn.softmax(tf.add(tf.matmul(X, W), b))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predict, Y))
correct_prediction = tf.equal(tf.argmax(predict, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype))

optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

batch_size = 100
train_batch_num = 550
valid_batch_num = 50
test_batch_num = 100
epoch_num = 30

for e in range(epoch_num):
    all_train_loss = []
    all_valid_loss = []
    all_test_loss = []
    all_train_acc = []
    all_valid_acc = []
    all_test_acc = []
    for i in range(train_batch_num):
        batch_x = train_x[i*batch_size:i*batch_size+batch_size]
        batch_y = train_y[i*batch_size:i*batch_size+batch_size]
        sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y})
        train_loss = sess.run(loss, feed_dict={X: batch_x, Y: batch_y})
        train_acc = sess.run(accuracy, feed_dict={X: batch_x, Y: batch_y})
        all_train_loss.append(train_loss)
        all_train_acc.append(train_acc)
    for i in range(valid_batch_num):
        batch_x = valid_x[i*batch_size:i*batch_size+batch_size]
        batch_y = valid_y[i*batch_size:i*batch_size+batch_size]
        sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y})
        valid_loss = sess.run(loss, feed_dict={X: batch_x, Y: batch_y})
        valid_acc = sess.run(accuracy, feed_dict={X: batch_x, Y: batch_y})
        all_valid_loss.append(valid_loss)
        all_valid_acc.append(valid_acc)
    for i in range(test_batch_num):
        batch_x = test_x[i*batch_size:i*batch_size+batch_size]
        batch_y = test_y[i*batch_size:i*batch_size+batch_size]
        sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y})
        test_loss = sess.run(loss, feed_dict={X: batch_x, Y: batch_y})
        test_acc = sess.run(accuracy, feed_dict={X: batch_x, Y: batch_y})
        all_test_loss.append(test_loss)
        all_test_acc.append(test_acc)
    print 'epoch :', e
    print 'accuracy ', np.mean(all_train_acc), np.mean(all_valid_acc), np.mean(all_test_acc)
    print 'loss ', np.mean(all_train_loss), np.mean(all_valid_loss), np.mean(all_test_loss)
    print

# accuracy  0.940236 0.947 0.9464
# loss  1.52833 1.52179 1.52241