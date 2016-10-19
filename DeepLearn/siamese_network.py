# encoding: utf-8

"""
@author: liubo
@software: PyCharm Community Edition
@file: siamese.py
@time: 2016/10/19 12:26
@contact: ustb_liubo@qq.com
@annotation: siamese_network
"""
import sys
import logging
from logging.config import fileConfig
import os
from tensorflow.examples.tutorials.mnist import input_data # for data
import tensorflow as tf
import numpy as np

reload(sys)
sys.setdefaultencoding("utf-8")
# fileConfig('logger_config.ini')
# logger_error = logging.getLogger('errorhandler')

class Siamese:
    # Create model
    def __init__(self):
        self.channel_num = 1
        self.x1 = tf.placeholder(tf.float32, [None, 28, 28, self.channel_num])
        self.x2 = tf.placeholder(tf.float32, [None, 28, 28, self.channel_num])

        # 构建共享网络
        with tf.variable_scope("siamese") as scope:
            self.o1 = self.network(self.x1)
            scope.reuse_variables()
            self.o2 = self.network(self.x2)

        # Create loss
        self.y_ = tf.placeholder(tf.float32, [None])
        self.loss = self.loss_with_spring()


    def network(self, x):
        initer = tf.truncated_normal_initializer(stddev=0.01)
        w1 = tf.get_variable('la1W', dtype=tf.float32, shape=[3, 3, self.channel_num, 32], initializer=initer)

        l1a = tf.nn.relu(tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='SAME'))
        l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        l1 = tf.nn.lrn(l1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
        l1 = tf.nn.dropout(l1, 0.5)

        w2 = tf.get_variable('la2W', dtype=tf.float32, shape=[3, 3, 32, 64], initializer=initer)
        l2a = tf.nn.relu(tf.nn.conv2d(l1, w2, strides=[1, 1, 1, 1], padding='SAME'))
        l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        l2 = tf.nn.lrn(l2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
        l2 = tf.nn.dropout(l2, 0.5)

        w3 = tf.get_variable('la3W', dtype=tf.float32, shape=[3, 3, 64, 128], initializer=initer)
        l3a = tf.nn.relu(tf.nn.conv2d(l2, w3, strides=[1, 1, 1, 1], padding='SAME'))
        l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        l3 = tf.nn.lrn(l3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

        w4 = tf.get_variable('la4W', dtype=tf.float32, shape=[3, 3, 128, 128], initializer=initer)
        l4a = tf.nn.relu(tf.nn.conv2d(l3, w4, strides=[1, 1, 1, 1], padding='SAME'))
        l4a = tf.nn.lrn(l4a, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

        concat = tf.concat(3, [l3, l4a])
        # pdb.set_trace()
        w5 = tf.get_variable('w5', dtype=tf.float32, shape=[4096, 1024], initializer=initer)
        concat = tf.reshape(concat, [-1, w5.get_shape().as_list()[0]])

        l5 = tf.matmul(concat, w5)

        return l5

    def loss_with_spring(self):
        margin = 3.0
        labels_t = self.y_
        labels_f = tf.sub(1.0, self.y_, name='1-y')
        eucd = tf.pow(tf.sub(self.o1, self.o2), 2)
        eucd = tf.reduce_sum(eucd, 1)
        # Dw = ||Gw(X1)-Gw(X2)||2
        eucd2 = tf.sqrt(eucd + 1e-6, name='eucd')
        # yi*||CNN(p1i)-CNN(p2i)||^2 + (1-yi)*max(0, C-||CNN(p1i)-CNN(p2i)||^2)
        C = tf.constant(margin, name='C')
        pos = tf.mul(labels_t, eucd, name='yi_x_eucd2')
        neg = tf.mul(labels_f, tf.pow(tf.maximum(tf.sub(C, eucd2), 0), 2), name='1-yi_max')
        losses = tf.add(pos, neg, name='losses')
        loss = tf.reduce_mean(losses, name='loss')
        return loss


    def loss_with_step(self):
        margin = 3.0
        labels_t = self.y_
        labels_f = tf.sub(1.0, self.y_, name='1-yi')
        eucd2 = tf.pow(tf.sub(self.o1 - self.o2), 2)
        eucd2 = tf.reduce_sum(eucd2, 1)
        eucd = tf.sqrt(eucd2 + 1e-6, name='eucd')
        C = tf.constant(margin, name='C')
        pos = tf.mul(labels_t, eucd, name='y_x_eucd')
        neg = tf.mul(labels_f, tf.maximum(tf.sub(C, eucd), 0), name='Ny_C-eucd')
        losses = tf.add(pos, neg, name='losses')
        loss = tf.reduce_mean(losses, name='loss')
        return loss


if __name__ == '__main__':
    print('load data')
    mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
    sess = tf.InteractiveSession()

    print('build model')
    # setup siamese network
    siamese = Siamese()
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(siamese.loss)
    saver = tf.train.Saver()
    tf.initialize_all_variables().run()

    # if you just want to load a previously trainmodel?
    new = True
    model_ckpt = 'models/model.ckpt'

    saver.restore(sess, 'models/model.ckpt')

    if new:
        # 训练10万个batch
        for step in range(100000):
            batch_x1, batch_y1 = mnist.train.next_batch(128)
            batch_x2, batch_y2 = mnist.train.next_batch(128)
            batch_x1 = batch_x1.reshape([128, 28, 28, 1])
            batch_x2 = batch_x2.reshape([128, 28, 28, 1])
            batch_y = (batch_y1 == batch_y2).astype('float')

            _, loss_v = sess.run([train_step, siamese.loss], feed_dict={
                siamese.x1: batch_x1,
                siamese.x2: batch_x2,
                siamese.y_: batch_y})

            if np.isnan(loss_v):
                print('Model diverged with loss = NaN')
                quit()

            if step % 10 == 0:
                print ('step %d: loss %.3f' % (step, loss_v))

            if step % 1000 == 0 and step > 0:
                saver.save(sess, 'models/model.ckpt')

    else:
        saver.restore(sess, 'models/model.ckpt')
