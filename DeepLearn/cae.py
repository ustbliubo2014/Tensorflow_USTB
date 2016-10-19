# encoding: utf-8
__author__ = 'liubo'

"""
@version: 
@author: 刘博
@license: Apache Licence 
@contact: ustb_liubo@qq.com
@software: PyCharm
@file: cae.py
@time: 2016/7/23 21:03
"""

import logging
import os

if not os.path.exists('log'):
    os.mkdir('log')

logging.basicConfig(level=logging.ERROR,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='log/cae.log',
                    filemode='w')


import tensorflow as tf
import numpy as np
import sys
import os
import math
import input_data

class cae():
    def __init__(self,
                 config = {'CheckPoint' : 'sample.ckpt',
                           'Activate' : 'Relu',
                           'MaxoutNum' : 3,
                           'data' : None,
                           'HiddenNodes' : 3,
                           'Initialize' : True,
                           'TrainNum' : 10000,
                           'Algorithm' : '',
                           'LogPeriod' : 100,
                           'Noise' : True,
                           'NoiseLevel' : 0.1,
                           'LearningRate' : 0.1,
                           'BatchSize' : 50,
                           'Sparse' : True,
                           'SparseBeta' : 3
                           }
                 ):
        self.checkpoint = config['CheckPoint']
        self.activate = config['Activate']
        self.learning_rate = config['LearningRate']
        self.data = config['data']
        self.hidden_nodes = config['HiddenNodes']
        self.initialize = config['Initialize']
        self.train_num = config['TrainNum']
        self.algo = config['Algorithm']
        self.ckpt = config['LogPeriod']
        self.noise = config['Noise']
        self.noise_level = config['NoiseLevel']
        self.batch_size = config['BatchSize']
        self.sparse = config['Sparse']
        self.beta = config['SparseBeta']
        self.n = config['MaxoutNum']
        self.filter = config['Filter']
        self.strides = config['Strides']
        self.padding = config['Padding']

        self.sess = tf.InteractiveSession()


        self.x = tf.placeholder("float", shape=[None, len(self.data[0]), len(self.data[0][0]), len(self.data[0][0][0])])
        self.y_ = tf.placeholder("float", shape=[None, len(self.data[0]), len(self.data[0][0]), len(self.data[0][0][0])])


        self.interface()

        self.loss()

        self.training()

        self.saver = tf.train.Saver()
        self.restore()

        self.hidden, self.W_p, self.b_p, self.output = self.learning()

    def session_close(self):
        self.sess.close()

    def get_params(self):
        return self.hidden, self.W_p, self.b_p, self.output

    def add_noise(self, labels):
        if self.noise:
            y = []
            for d0 in labels:
                x = []
                for d1 in d0:
                    x.append(d1* (1.0 + np.random.normal(0,self.noise_level)))
                y.append(x)
        else:
            y = labels
        return y


    def restore(self):
        if os.path.exists(self.checkpoint) and self.init == False:
            try:
                self.saver.restore(self.sess, self.checkpoint)
                print 'restored'
            except:
                print 'WARNING: Model is changed'
                self.sess.run(tf.initialize_all_variables())
        else:
            print "Initialize"
            self.sess.run(tf.initialize_all_variables())


    def learning(self):
        j = 0
        if self.initialize:
            labels = self.data
            ylabels = self.add_noise(labels)
            for i in range(self.train_num):
                d0, d1 = [], []
                for p in range(self.batch_size):
                    d0.append(self.data[j % len(self.data)])
                    d1.append(ylabels[j % len(self.data)])
                    j += 1
                xTrain = np.array(d0)
                yTrain = np.array(d1)
                _, loss_val = self.sess.run([self.train_op, self.loss_function], feed_dict={self.x: xTrain, self.y_: yTrain})
                if i % int(self.ckpt) == 0:
                    print "Step:", i, "Current loss:", loss_val
            save_path = self.saver.save(self.sess, self.checkpoint)
            print "Model saved in file: ", save_path
        xTrain = np.array(self.data)
        l, y, w, b = self.sess.run([self.logits[1], self.y, self.W[0], self.b[0]], feed_dict={self.x: xTrain})
        return l, w, b, y



    def training(self):
        if self.algo == 'Adam':
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_function)
        elif self.algo == 'GD':
            self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss_function)
        elif self.algo == 'Adagrad':
            self.train_op = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss_function)
        else:
            self.train_op = tf.train.AdamOptimizer().minimize(self.loss_function)


    def loss(self):
        if not self.sparse:
            self.loss_function = tf.nn.l2_loss(self.y_ - self.y) / float(self.batch_size)
        else:
            eps = 0.00001
            rho = tf.reduce_mean(self.logits[1])
            kl = self.beta * rho
            self.loss_function = (tf.nn.l2_loss(self.y_ - self.y)  + self.beta * kl) / float(self.batch_size)


    def interface(self):
        self.W, self.b, self.logits = [], [], [self.x]
        with tf.name_scope('main_layer') as scope:
            if self.activate == 'Maxout':
                w_shape = [self.filter[0], self.filter[1], self.filter[2], self.filter[3] * self.n]
                b_shape = [self.hidden_nodes * self.n]
                self.W.append(weight_variable(w_shape))
                self.b.append(bias_variable(b_shape))
                t0 = tf.nn.conv2d(self.logits[0], self.W[0], strides = self.strides, padding = self.padding) + self.b[0]
                s = t0.get_shape()
                t1 = tf.reshape(t0, [-1, int(s.dims[1]), int(s.dims[2]), self.hidden_nodes, self.n])
                t2 = tf.reduce_max(t1, 4)
                self.logits.append(t2)
                w_shape = [self.filter[0], self.filter[1], self.hidden_nodes, self.filter[2] * self.n]
                b_shape = [self.filter[2] * self.n]
                self.W.append(weight_variable(w_shape))
                self.b.append(bias_variable(b_shape))
                t0 = tf.nn.conv2d(self.logits[1], self.W[1], strides = self.strides, padding = self.padding) + self.b[1]
                s = t0.get_shape()
                t1 = tf.reshape(t0, [-1, int(s.dims[1]), int(s.dims[2]), self.filter[2], self.n])
                t2 = tf.reduce_max(t1, 4)
                self.logits.append(t2)
                self.y = self.logits[len(self.logits) - 1]
            else:
                w_shape = self.filter
                b_shape = [self.hidden_nodes]
                self.W.append(weight_variable(w_shape))
                self.b.append(bias_variable(b_shape))
                self.logits.append(tf.nn.relu(tf.nn.conv2d(self.logits[0], self.W[0], strides = self.strides, padding = self.padding) + self.b[0]))
                w_shape = [self.filter[0], self.filter[1], self.hidden_nodes, self.filter[2]]
                b_shape = [self.filter[2]]
                self.W.append(weight_variable(w_shape))
                self.b.append(bias_variable(b_shape))
                self.logits.append(tf.nn.relu(tf.nn.conv2d(self.logits[1], self.W[1], strides = self.strides, padding = self.padding) + self.b[1]))
                self.y = self.logits[len(self.logits) - 1]


def weight_variable(shape):
    w = tf.truncated_normal(shape, stddev=1./math.sqrt(shape[0]))
    return tf.Variable(w)


def bias_variable(shape):
    b = tf.zeros(shape)
    return tf.Variable(b)


if __name__ == '__main__':
    pre_training_data = 50
    mnist = input_data.read_data_sets("/home/liubo-it/siamese_tf_mnist/MNIST_data/", one_hot=True)
    data = mnist.train.next_batch(pre_training_data)
    test_data = []
    for d in data[0]:
        test_data.append(np.reshape(d, (28, 28, 1)))
    config = {'CheckPoint' : 'sample_cae.ckpt',
            'Activate' : 'Maxout',
            'MaxoutNum' : 10,
            'data' : test_data,
            'HiddenNodes' : 5,
            'Filter' : [5, 5, 1, 5],
            'Strides' : [1, 1, 1, 1],
            'Padding' : 'SAME',
            'Initialize' : True,
            'TrainNum' : 10000,
            'Algorithm' : '',
            'LogPeriod' : 100,
            'Noise' : True,
            'NoiseLevel' : 0.1,
            'LearningRate' : 0.1,
            'BatchSize' : 50,
            'Sparse' : False,
            'SparseBeta' : 3
            }
    c = cae(config = config)
    c.learning()



