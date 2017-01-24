# encoding: utf-8

"""
@author: liubo
@software: PyCharm
@file: 05_mlp.py
@time: 2017/1/24 17:32
@contact: ustb_liubo@qq.com
@annotation: 05_mlp
"""
import sys
import logging
from logging.config import fileConfig
import os
import numpy as np
import tensorflow as tf
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


# Parameters
learning_rate = 0.001
training_epochs = 20

# Network Parameters
n_input = 784           # MNIST data input (img shape: 28*28)
n_hidden_1 = 256        # 1st layer num features
n_hidden_2 = 256        # 2nd layer num features
n_hidden_3 = 256        # 3rd layer num features
n_hidden_4 = 256        # 4th layer num features
n_classes = 10          # MNIST total classes (0-9 digits)

dtype = tf.float32

# 创建网络输入输出节点(tf.placeholder)
X = tf.placeholder(dtype, [None, n_input])
Y = tf.placeholder(dtype, [None, n_classes])
dropout_keep_prob = tf.placeholder(dtype)

# 创建网络参数(tf.Variable)
stddev = 0.1
Weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=stddev)),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=stddev)),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], stddev=stddev)),
    'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4], stddev=stddev)),
    'out': tf.Variable(tf.random_normal([n_hidden_4, n_classes], stddev=stddev))
}

Biases = {
    'h1': tf.Variable(tf.random_normal([n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_3])),
    'h4': tf.Variable(tf.random_normal([n_hidden_4])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# 创建网络
def multilayer_perception_dropout(X, Weights, Biases, keep_prob):
    layer1 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(X, Weights['h1']), Biases['h1'])), keep_prob)
    layer2 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(layer1, Weights['h2']), Biases['h2'])), keep_prob)
    layer3 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(layer2, Weights['h3']), Biases['h3'])), keep_prob)
    layer4 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(layer3, Weights['h4']), Biases['h4'])), keep_prob)
    return tf.add(tf.matmul(layer4, Weights['out']), Biases['out'])


predict = multilayer_perception_dropout(X, Weights, Biases, dropout_keep_prob)
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

# accuracy  0.967436 0.9922 0.996
# loss  0.124456 0.0304988 0.0156934