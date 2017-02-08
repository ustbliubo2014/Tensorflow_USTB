# encoding: utf-8

"""
@author: liubo
@software: PyCharm
@file: 06_cnn.py
@time: 2017/1/25 9:57
@contact: ustb_liubo@qq.com
@annotation: 06_cnn
"""
import sys
import logging
from logging.config import fileConfig
import os
import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data

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


learning_rate = 0.001
training_epochs = 20
display_step = 10

# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

dtype = tf.float32
# tf Graph input
X = tf.placeholder(dtype, [None, n_input])
Y = tf.placeholder(dtype, [None, n_classes])
dropout_keep_prob = tf.placeholder(dtype)


def conv2d(x, W, b, strides=1):
    # strides=1时维度不变, strides=2时维度减半
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],  padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv1 = maxpool2d(conv1, k=2)

    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, k=2)

    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)

    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


Weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

Biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


predict = conv_net(X, Weights, Biases, dropout_keep_prob)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predict, Y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
correct_prediction = tf.equal(tf.argmax(predict, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype))

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)


batch_size = 100
train_batch_num = 550
valid_batch_num = 50
test_batch_num = 100
epoch_num = 30
train_dropout_prob = 0.5
valid_dropout_prob = 1.0


for epoch in range(training_epochs):
    # shuffle数据
    shuffle_list = np.asarray(range(train_x.shape[0]))
    np.random.shuffle(shuffle_list)
    all_train_loss = []
    all_valid_loss = []
    all_test_loss = []
    all_train_acc = []
    all_valid_acc = []
    all_test_acc = []
    for i in range(train_batch_num):
        batch_x = train_x[shuffle_list[i*batch_size:i*batch_size+batch_size]]
        batch_y = train_y[shuffle_list[i*batch_size:i*batch_size+batch_size]]
        sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y, dropout_keep_prob: train_dropout_prob})
        train_loss = sess.run(loss, feed_dict={X: batch_x, Y: batch_y, dropout_keep_prob: train_dropout_prob})
        train_acc = sess.run(accuracy, feed_dict={X: batch_x, Y: batch_y, dropout_keep_prob: train_dropout_prob})
        all_train_loss.append(train_loss)
        all_train_acc.append(train_acc)
    for i in range(valid_batch_num):
        batch_x = valid_x[i*batch_size:i*batch_size+batch_size]
        batch_y = valid_y[i*batch_size:i*batch_size+batch_size]
        sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y, dropout_keep_prob: valid_dropout_prob})
        valid_loss = sess.run(loss, feed_dict={X: batch_x, Y: batch_y, dropout_keep_prob: valid_dropout_prob})
        valid_acc = sess.run(accuracy, feed_dict={X: batch_x, Y: batch_y, dropout_keep_prob: valid_dropout_prob})
        all_valid_loss.append(valid_loss)
        all_valid_acc.append(valid_acc)
    for i in range(test_batch_num):
        batch_x = test_x[i*batch_size:i*batch_size+batch_size]
        batch_y = test_y[i*batch_size:i*batch_size+batch_size]
        sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y, dropout_keep_prob: valid_dropout_prob})
        test_loss = sess.run(loss, feed_dict={X: batch_x, Y: batch_y, dropout_keep_prob: valid_dropout_prob})
        test_acc = sess.run(accuracy, feed_dict={X: batch_x, Y: batch_y, dropout_keep_prob: valid_dropout_prob})
        all_test_loss.append(test_loss)
        all_test_acc.append(test_acc)
    print 'epoch :', epoch
    print 'accuracy ', np.mean(all_train_acc), np.mean(all_valid_acc), np.mean(all_test_acc)
    print 'loss ', np.mean(all_train_loss), np.mean(all_valid_loss), np.mean(all_test_loss)
    print

