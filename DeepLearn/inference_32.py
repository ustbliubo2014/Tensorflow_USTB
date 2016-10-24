# encoding: utf-8

"""
@author: liubo
@software: PyCharm Community Edition
@file: inference.py
@time: 2016/10/19 17:40
@contact: ustb_liubo@qq.com
@annotation: inference
"""
import sys
import logging
from logging.config import fileConfig
import os
import tensorflow as tf
import pdb

reload(sys)
sys.setdefaultencoding("utf-8")
# fileConfig('logger_config.ini')
# logger_error = logging.getLogger('errorhandler')


def inference(images):
    # images : Tensor
    initer = tf.truncated_normal_initializer(stddev=0.01)
    w1 = tf.get_variable('la1W', dtype=tf.float32, shape=[3, 3, 3, 32], initializer=initer)

    l1a = tf.nn.relu(tf.nn.conv2d(images, w1, strides=[1, 1, 1, 1], padding='SAME'))
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.lrn(l1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    l1 = tf.nn.dropout(l1, 0.5)

    w2 = tf.get_variable('la2W', dtype=tf.float32, shape=[3, 3, 32, 64], initializer=initer)
    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2, strides=[1, 1, 1, 1], padding='SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.lrn(l2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    l2 = tf.nn.dropout(l2, 0.5)

    w3 = tf.get_variable('la3W', dtype=tf.float32, shape=[3, 3, 64, 256], initializer=initer)
    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3, strides=[1, 1, 1, 1], padding='SAME'))
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l3 = tf.nn.lrn(l3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

    w4 = tf.get_variable('la4W', dtype=tf.float32, shape=[3, 3, 256, 256], initializer=initer)
    l4a = tf.nn.relu(tf.nn.conv2d(l3, w4, strides=[1, 1, 1, 1], padding='SAME'))
    l4a = tf.nn.lrn(l4a, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

    concat = tf.concat(3, [l3, l4a])

    # 如果不能确定flatten之后的大小,可以填一个素数,程序报错后就可以得到输出的维度(总数/batch_size)
    w5 = tf.get_variable('w5', dtype=tf.float32, shape=[2304*2, 1024], initializer=initer)
    concat = tf.reshape(concat, [-1, w5.get_shape().as_list()[0]])

    l5 = tf.matmul(concat, w5)
    w_o = tf.get_variable('softmaxW', dtype=tf.float32, shape=[1024, 10], initializer=initer)
    b_o = tf.get_variable('softmaxb', dtype=tf.float32, shape=[10], initializer=initer)
    softmax_linear = tf.add(tf.matmul(l5, w_o), b_o, name='softmax')

    return softmax_linear


def loss(logits, labels):
    """Add L2Loss to all the trainable variables.

    Add summary for "Loss" and "Loss/avg".
    Args:
        logits: Logits from inference().
        labels: Labels from distorted_inputs or inputs(). 1-D tensor of shape [batch_size]
    Returns:
        Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits, labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


if __name__ == '__main__':
    pass
