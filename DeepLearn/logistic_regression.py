# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: logistic_regression.py
@time: 2016/7/22 20:28
@contact: ustb_liubo@qq.com
@annotation: logistic_regression
"""
import sys

reload(sys)
sys.setdefaultencoding("utf-8")
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='logistic_regression.log',
                    filemode='a+')
import numpy as np
import tensorflow as tf
import pdb
from read_data import read_data_sets


# 读入数据 -- 转换成numpy的格式进行训练(大规模数据可以使用batch的方法进行训练)
train_images, train_labels, test_images, test_labels = read_data_sets('data/', one_hot=True)


# 设置参数
# Parameters of Logistic Regression
learning_rate = 0.001
training_epochs = 500
batch_size = 128
valid_step = 1 # 每valid_step个epoch进行一次验证

# 设置placeholder和Variable[输入的参数用tf.placeholder来初始,参数W,b用tf.Variable来初始(这里需要根据指定维度)]
# Create Graph for Logistic Regression
x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float", [None, 10])  # None is for infinite or unspecified length
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# softmax的计算公式
# Activation, Cost, and Optimizing functions
activation = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

# softmax对应的损失函数
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(activation), reduction_indices=1)) # Cross entropy

# 优化函数
# * is an element-wise product in numpy (in Matlab, it should be .*)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) # Gradient Descent

# 验证函数(计算准确率)
correct_prediction = tf.equal(tf.argmax(activation, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# Optimize with TensorFlow
# Initializing the variables
# 在sess初始前需要初始化所有参数
init = tf.initialize_all_variables()


# Launch the graph
with tf.Session() as sess:
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
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})

            # Compute average loss
            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys}) / num_batch

        # Display logs per epoch step
        if epoch % valid_step == 0:
            # 对train和valid进行验证
            all_train_avg_cost = []
            for i in range(num_batch):
                batch_xs, batch_ys = train_images[shuffle_list[i*batch_size:(i+1)*batch_size]], \
                                 train_labels[shuffle_list[i*batch_size:(i+1)*batch_size]]
                train_acc = accuracy.eval({x: batch_xs, y: batch_ys})
                all_train_avg_cost.append(train_acc)
            all_valid_avg_cost = []
            valid_num_batch = test_images.shape[0] / batch_size
            for i in range(valid_num_batch):
                batch_xs, batch_ys = test_images[i*batch_size:(i+1)*batch_size], \
                                 test_labels[i*batch_size:(i+1)*batch_size]
                valid_acc = accuracy.eval({x: batch_xs, y: batch_ys})
                all_valid_avg_cost.append(valid_acc)
            print ("Epoch: %03d/%03d train_acc: %.3f valid_acc: %.3f "
                   % (epoch, training_epochs, np.mean(all_train_avg_cost), np.mean(all_valid_avg_cost)))


print ("Done.")


