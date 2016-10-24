# encoding: utf-8
__author__ = 'liubo'

"""
@version: 
@author: 刘博
@license: Apache Licence 
@contact: ustb_liubo@qq.com
@software: PyCharm
@file: mlp_simple.py
@time: 2016/7/23 13:44
"""

import logging
import os

if not os.path.exists('log'):
    os.mkdir('log')

logging.basicConfig(level=logging.ERROR,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='log/mlp.log',
                    filemode='w')

import numpy as np
import tensorflow as tf
import pdb
from read_data import read_data_sets


# 读入数据 -- 转换成numpy的格式进行训练(大规模数据可以使用batch的方法进行训练)
train_images, train_labels, test_images, test_labels = read_data_sets('/home/liubo-it/siamese_tf_mnist/MNIST_data/', one_hot=True)


# Parameters
learning_rate = 0.001
training_epochs = 20
batch_size = 128
valid_step = 1

# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_hidden_1 = 256 # 1st layer num features
n_hidden_2 = 256 # 2nd layer num features
n_hidden_3 = 256 # 3rd layer num features
n_hidden_4 = 256 # 4th layer num features
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])
dropout_keep_prob = tf.placeholder("float")


# Create model
def multilayer_perceptron(_X, _weights, _biases, _keep_prob):
    layer_1 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1'])), _keep_prob)
    layer_2 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(layer_1, _weights['h2']), _biases['b2'])), _keep_prob)
    layer_3 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(layer_2, _weights['h3']), _biases['b3'])), _keep_prob)
    layer_4 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(layer_3, _weights['h4']), _biases['b4'])), _keep_prob)
    return (tf.matmul(layer_4, _weights['out']) + _biases['out'])


# Store layers weight & bias
stddev = 0.1
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=stddev)),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=stddev)),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], stddev=stddev)),
    'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4], stddev=stddev)),
    'out': tf.Variable(tf.random_normal([n_hidden_4, n_classes], stddev=stddev))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'b4': tf.Variable(tf.random_normal([n_hidden_4])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# Construct model
pred = multilayer_perceptron(x, weights, biases, dropout_keep_prob)


# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y)) # Softmax loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer
# optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.8).minimize(cost) # Adam Optimizer


# Accuracy
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# Initializing the variables
init = tf.initialize_all_variables()

print ("Network Ready")

# Launch the graph
sess = tf.Session()
sess.run(init)

# Training cycle
for epoch in range(training_epochs):
    avg_cost = 0.
    num_batch = train_images.shape[0] / batch_size
    shuffle_list = np.asarray(range(train_images.shape[0]))
    np.random.shuffle(shuffle_list)


    for i in range(num_batch):
        batch_xs, batch_ys = train_images[shuffle_list[i*batch_size:(i+1)*batch_size]], \
                                 train_labels[shuffle_list[i*batch_size:(i+1)*batch_size]]

        # Fit training using batch data
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, dropout_keep_prob: 0.7})
        # Compute average loss
        avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, dropout_keep_prob:1.})/num_batch

        # Display logs per epoch step
    if epoch % valid_step == 0:
        # 对train和valid进行验证
        all_train_avg_cost = []
        for i in range(num_batch):
            batch_xs, batch_ys = train_images[shuffle_list[i*batch_size:(i+1)*batch_size]], \
                                 train_labels[shuffle_list[i*batch_size:(i+1)*batch_size]]
            train_acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, dropout_keep_prob:1.})
            all_train_avg_cost.append(train_acc)
        all_valid_avg_cost = []
        valid_num_batch = test_images.shape[0] / batch_size
        for i in range(valid_num_batch):
            batch_xs, batch_ys = test_images[i*batch_size:(i+1)*batch_size], \
                                 test_labels[i*batch_size:(i+1)*batch_size]
            valid_acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, dropout_keep_prob:1.})
            all_valid_avg_cost.append(valid_acc)
        print ("Epoch: %03d/%03d train_acc: %.3f valid_acc: %.3f "
                   % (epoch, training_epochs, np.mean(all_train_avg_cost), np.mean(all_valid_avg_cost)))


print ("Optimization Finished!")


if __name__ == '__main__':
    pass