# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: tmp.py
@time: 2016/8/17 11:10
@contact: ustb_liubo@qq.com
@annotation: tmp
"""

import tensorflow as tf
import numpy as np
from time import time
from sklearn.cross_validation import train_test_split
import os
import msgpack_numpy
import sys
from scipy.misc import imresize, imread
import traceback
from keras.utils import np_utils

# 训练originalimages, 200个人, 每个人14张图片
learning_rate = 0.001
training_epochs = 100
display_step = 1
batch_size = 128
test_size = 256
nb_classes = 181
channel = 3
pic_shape = 128


model_data, model_label = msgpack_numpy.load(open('/data/liubo/face/originalimages/originalimages_model.p', 'rb'))
model_label = np_utils.to_categorical(model_label, nb_classes)
print model_data.shape, model_label.shape


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w, w2, w3, w4, w5, w_o, p_keep_conv, p_keep_hidden):
    l1a = tf.nn.relu(tf.nn.conv2d(X, w, strides=[1, 1, 1, 1], padding='SAME'))
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.lrn(l1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    # l1 = tf.nn.dropout(l1, p_keep_conv)

    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2, strides=[1, 1, 1, 1], padding='SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.lrn(l2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    # l2 = tf.nn.dropout(l2, p_keep_conv)

    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3, strides=[1, 1, 1, 1], padding='SAME'))
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l3 = tf.nn.lrn(l3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    l4a = tf.nn.relu(tf.nn.conv2d(l3, w4, strides=[1, 1, 1, 1], padding='SAME'))
    l4a = tf.nn.lrn(l4a, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

    concat = tf.concat(3, [l3, l4a])
    concat = tf.reshape(concat, [-1, w5.get_shape().as_list()[0]])
    concat = tf.nn.dropout(concat, p_keep_conv)

    l5 = tf.nn.relu(tf.matmul(concat, w5))
    l5 = tf.nn.dropout(l5, p_keep_hidden)

    pyx = tf.matmul(l5, w_o)
    return pyx

global_step = tf.Variable(0, name='global_step', trainable=False)
saver = tf.train.Saver()


X = tf.placeholder("float", [None, pic_shape, pic_shape, channel])
Y = tf.placeholder("float", [None, nb_classes])

w = init_weights([3, 3, channel, 64])       # 3x3x1 conv, 64 outputs
w2 = init_weights([3, 3, 64, 128])     # 3x3x32 conv, 64 outputs
w3 = init_weights([3, 3, 128, 256])    # 3x3x64 conv, 128 outputs
w4 = init_weights([3, 3, 256, 256])   # 3x3x128 conv, 128 outputs
w5 = init_weights([(256+256) * 16 * 16, 2048]) # FC 128 * 16 * 16 inputs, 1024 outputs
w_o = init_weights([2048, nb_classes])         # FC 1024 inputs, 1800 outputs (labels)

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(X, w, w2, w3, w4, w5, w_o, p_keep_conv, p_keep_hidden)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
train_op = tf.train.RMSPropOptimizer(0.01, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)



# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.initialize_all_variables().run()
    last_acc = 0

    for i in range(training_epochs):
        global_step.assign(i).eval()
        print 'epoch :', i
        all_train_result = []
        start_time = time()
        training_batch = zip(range(0, len(model_data), batch_size),
                             range(batch_size, len(model_label), batch_size))

        for start, end in training_batch:
            sess.run(train_op, feed_dict={X: model_data[start:end], Y: model_label[start:end],
                                p_keep_conv: 0.8, p_keep_hidden: 0.5})


