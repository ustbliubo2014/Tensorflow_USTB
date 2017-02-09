# encoding: utf-8

"""
@author: liubo
@software: PyCharm
@file: speed_test.py
@time: 2017/2/8 17:47
@contact: ustb_liubo@qq.com
@annotation: speed_test
"""
import sys
import logging
from logging.config import fileConfig
import os
import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data
from time import time
from collections import namedtuple
from libs.connections import conv2d, linear
from math import sqrt

reload(sys)
sys.setdefaultencoding("utf-8")
# fileConfig('logger_config.ini')
# logger_error = logging.getLogger('errorhandler')


#
def residual_network(x, n_outputs,
                     activation=tf.nn.relu):
    """Builds a residual network.

    Parameters
    ----------
        x : Placeholder  Input to the network
        n_outputs : TYPE    Number of outputs of final softmax
        activation : Attribute, optional    Nonlinearity to apply after each convolution

    Returns
    -------
        net : Tensor    Description

    Raises
    ------
    ValueError
        If a 2D Tensor is input, the Tensor must be square or else
        the network can't be converted to a 4D Tensor.
    """
    #
    LayerBlock = namedtuple(
        'LayerBlock', ['num_repeats', 'num_filters', 'bottleneck_size'])
    blocks = [LayerBlock(1, 128, 32),
              LayerBlock(1, 256, 64),
              LayerBlock(1, 512, 128),
              LayerBlock(1, 1024, 256)]

    #
    input_shape = x.get_shape().as_list()
    if len(input_shape) == 2:
        ndim = int(sqrt(input_shape[1]))
        if ndim * ndim != input_shape[1]:
            raise ValueError('input_shape should be square')
        x = tf.reshape(x, [-1, ndim, ndim, 1])

    # First convolution expands to 64 channels and downsamples
    net = conv2d(x, 64, k_h=3, k_w=3, name='conv1', activation=activation)

    # Max pool and downsampling
    net = tf.nn.max_pool(net, [1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Setup first chain of resnets
    net = conv2d(net, blocks[0].num_filters, k_h=1, k_w=1, stride_h=1, stride_w=1, padding='VALID', name='conv2')

    # Loop through all res blocks
    for block_i, block in enumerate(blocks):
        for repeat_i in range(block.num_repeats):

            name = 'block_%d/repeat_%d' % (block_i, repeat_i)
            conv = conv2d(net, block.bottleneck_size, k_h=1, k_w=1,
                          padding='VALID', stride_h=1, stride_w=1,
                          activation=activation,
                          name=name + '/conv_in')

            conv = conv2d(conv, block.bottleneck_size, k_h=3, k_w=3,
                          padding='SAME', stride_h=1, stride_w=1,
                          activation=activation,
                          name=name + '/conv_bottleneck')

            conv = conv2d(conv, block.num_filters, k_h=1, k_w=1,
                          padding='VALID', stride_h=1, stride_w=1,
                          activation=activation,
                          name=name + '/conv_out')

            net = conv + net
        try:
            # upscale to the next block size
            next_block = blocks[block_i + 1]
            net = conv2d(net, next_block.num_filters, k_h=1, k_w=1,
                         padding='SAME', stride_h=1, stride_w=1, bias=False,
                         name='block_%d/conv_upscale' % block_i)
        except IndexError:
            pass

    #
    net = tf.nn.avg_pool(net,
                         ksize=[1, net.get_shape().as_list()[1],
                                net.get_shape().as_list()[2], 1],
                         strides=[1, 1, 1, 1], padding='VALID')
    net = tf.reshape(
        net,
        [-1, net.get_shape().as_list()[1] *
         net.get_shape().as_list()[2] *
         net.get_shape().as_list()[3]])

    net = linear(net, n_outputs, activation=tf.nn.softmax)

    #
    return net


def test_mnist():
    """Test the resnet on MNIST."""

    trainX = np.random.random(size=1024*10000)
    trainY = np.random.random(size=1024*10000)
    trainX = np.reshape(trainX, (10000, 1024))
    trainY = np.reshape(trainY, (10000, 1024))

    x = tf.placeholder(tf.float32, [None, 1024])
    y = tf.placeholder(tf.float32, [None, 1024])
    y_pred = residual_network(x, 1024)

    #  Define loss/eval/training functions
    cross_entropy = -tf.reduce_sum(y * tf.log(y_pred))
    optimizer = tf.train.AdamOptimizer().minimize(cross_entropy)

    #  Monitor accuracy
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    #  We now create a new session to actually perform the initialization the variables:
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    # TensorBoard log目录
    log_dir = 'speed_log'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    writer = tf.train.SummaryWriter(log_dir, sess.graph)
    writer.close()


    #  We'll train in minibatches and report accuracy:
    for batch_size in range(1, 510, 50):
        n_epochs = 1
        all_time = []
        for epoch_i in range(n_epochs):
            for batch_i in range(len(trainX) // batch_size):
                batch_xs = trainX[batch_i*batch_size:(batch_i+1)*batch_size]
                batch_ys = trainY[batch_i*batch_size:(batch_i+1)*batch_size]
                start = time()
                sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys})
                end = time()
                all_time.append(end-start)
        mean_time = np.mean(all_time)
        print 'batch_size :', batch_size, 'per_time :', (mean_time / batch_size)







if __name__ == '__main__':
    test_mnist()
