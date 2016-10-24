# encoding: utf-8

"""
@author: liubo
@software: PyCharm Community Edition
@file: inference.py
@time: 2016/10/20 16:41
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

    initer = tf.truncated_normal_initializer(stddev=0.01)
    w1 = tf.get_variable('la1W', dtype=tf.float32, shape=[3, 3, 3, 32], initializer=initer)
    l1a = tf.nn.relu(tf.nn.conv2d(images, w1, strides=[1, 1, 1, 1], padding='SAME'))
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.lrn(l1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    l1 = tf.nn.dropout(l1, 0.5)

    w2a = tf.get_variable('la2aW', dtype=tf.float32, shape=[3, 3, 32, 64], initializer=initer)
    w2b = tf.get_variable('la2bW', dtype=tf.float32, shape=[3, 3, 64, 64], initializer=initer)
    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2a, strides=[1, 1, 1, 1], padding='SAME'))
    l2b = tf.nn.relu(tf.nn.conv2d(l2a, w2b, strides=[1, 1, 1, 1], padding='SAME'))
    l2 = tf.nn.max_pool(l2b, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.lrn(l2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    l2 = tf.nn.dropout(l2, 0.5)

    w3a = tf.get_variable('la3aW', dtype=tf.float32, shape=[3, 3, 64, 128], initializer=initer)
    w3b = tf.get_variable('la3bW', dtype=tf.float32, shape=[3, 3, 128, 128], initializer=initer)
    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3a, strides=[1, 1, 1, 1], padding='SAME'))
    l3b = tf.nn.relu(tf.nn.conv2d(l3a, w3b, strides=[1, 1, 1, 1], padding='SAME'))
    l3 = tf.nn.max_pool(l3b, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l3 = tf.nn.lrn(l3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    l3 = tf.nn.dropout(l3, 0.5)

    w4a = tf.get_variable('la4aW', dtype=tf.float32, shape=[3, 3, 128, 256], initializer=initer)
    w4b = tf.get_variable('la4bW', dtype=tf.float32, shape=[3, 3, 256, 512], initializer=initer)
    l4a = tf.nn.relu(tf.nn.conv2d(l3, w4a, strides=[1, 1, 1, 1], padding='SAME'))
    l4b = tf.nn.relu(tf.nn.conv2d(l4a, w4b, strides=[1, 1, 1, 1], padding='SAME'))
    l4 = tf.nn.max_pool(l4b, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l4 = tf.nn.lrn(l4, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    l4 = tf.nn.dropout(l4, 0.5)


    w5a = tf.get_variable('la5aW', dtype=tf.float32, shape=[3, 3, 512, 512], initializer=initer)
    w5b = tf.get_variable('la5bW', dtype=tf.float32, shape=[3, 3, 512, 512], initializer=initer)
    l5a = tf.nn.relu(tf.nn.conv2d(l4, w5a, strides=[1, 1, 1, 1], padding='SAME'))
    l5b = tf.nn.relu(tf.nn.conv2d(l5a, w5b, strides=[1, 1, 1, 1], padding='SAME'))
    l5 = tf.nn.max_pool(l5b, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l5 = tf.nn.lrn(l5, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)


    w6a = tf.get_variable('la6aW', dtype=tf.float32, shape=[3, 3, 512, 512], initializer=initer)
    w6b = tf.get_variable('la6bW', dtype=tf.float32, shape=[3, 3, 512, 512], initializer=initer)
    l6a = tf.nn.relu(tf.nn.conv2d(l5, w6a, strides=[1, 1, 1, 1], padding='SAME'))
    l6b = tf.nn.relu(tf.nn.conv2d(l6a, w6b, strides=[1, 1, 1, 1], padding='SAME'))
    l6 = tf.nn.lrn(l6b, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    l6 = tf.nn.dropout(l6, 0.5)

    concat = tf.concat(3, [l5, l6])
    # print l4, l5a
    # 如果不能确定flatten之后的大小,可以填一个素数,程序报错后就可以得到输出的维度(总数/batch_size)
    # 可以根据l3, l4a的shape确定
    w5 = tf.get_variable('w5', dtype=tf.float32, shape=[25600, 1024], initializer=initer)
    concat = tf.reshape(concat, [-1, w5.get_shape().as_list()[0]])

    l5 = tf.matmul(concat, w5)
    w_o = tf.get_variable('softmaxW', dtype=tf.float32, shape=[1024, 20], initializer=initer)
    b_o = tf.get_variable('softmaxb', dtype=tf.float32, shape=[20], initializer=initer)
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
