# encoding: utf-8

"""
@author: liubo
@software: PyCharm Community Edition
@file: mnist_main.py
@time: 2016/10/26 19:01
@contact: ustb_liubo@qq.com
@annotation: mnist_main
"""
import sys
import logging
from logging.config import fileConfig
import os
import time
import logging
logger = logging.getLogger()
logger.setLevel("DEBUG")
import numpy as np
import tensorflow as tf
import model
from tensorflow.examples.tutorials.mnist import input_data
reload(sys)
sys.setdefaultencoding("utf-8")

sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True

def evaluation(y_pred, y):
    correct = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    return accuracy


def mlogloss(predicted, actual):
    '''
      args.
         predicted : predicted probability
                    (sum of predicted proba should be 1.0)
         actual    : actual value, label
    '''
    def inner_fn(item):
        eps = 1.e-15
        item1 = min(item, (1 - eps))
        item1 = max(item, eps)
        res = np.log(item1)

        return res

    nrow = actual.shape[0]
    ncol = actual.shape[1]

    mysum = sum([actual[i, j] * inner_fn(predicted[i, j])
        for i in range(nrow) for j in range(ncol)])

    ans = -1 * mysum / nrow

    return ans


mnist = input_data.read_data_sets("/home/liubo-it/siamese_tf_mnist/MNIST_data", one_hot=True)
chkpt_file = '../model/mnist_cnn.ckpt'

TASK = 'train'
LEARNING_RATE = 0.0001

with tf.Graph().as_default():
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)
    phase_train = tf.placeholder(tf.bool, name='phase_train')

    y_pred   = model.inference(x, keep_prob, phase_train)
    accuracy = evaluation(y_pred, y_)
    loss     = model.loss(y_pred, y_)
    train_op = model.training(loss, LEARNING_RATE)
    init_op  = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())

    if TASK == 'test' or os.path.exists(chkpt_file):
        restore_call = True
    elif TASK == 'train':
        restore_call = False
    else:
        print('Check task switch.')

    saver = tf.train.Saver()

    sess = tf.Session(config=sess_config)
    if TASK == 'train':
        sess.run(init_op)

    if restore_call:
        # Restore variables from disk.
        saver.restore(sess, chkpt_file)
    if TASK == 'train':
        print('\n Training...')
        total_train_loss = []
        duration = 0

        for i in range(5001):
            start_time = time.time()
            batch_xs, batch_ys = mnist.train.next_batch(100)
            _, loss_value, accuracy_value = sess.run([train_op, loss, accuracy], {x: batch_xs, y_: batch_ys, keep_prob: 0.5, phase_train: True})
            duration += time.time() - start_time
            total_train_loss.append(loss_value)

            if (i % 100 == 0) and (i != 0):
                print('step, loss, accuracy = %6d: %8.4f / %8.4f (%.3f sec)' % (i, np.mean(total_train_loss), accuracy_value, duration) )

    # Test trained model
    test_loss, test_y_pred, test_accuracy_value = sess.run([loss, y_pred, accuracy],{x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0, phase_train: False})
    # Multiclass Log Loss
    print(' accuracy = %8.4f' % test_accuracy_value)
    act = mnist.test.labels
    print(' multiclass logloss = %8.4f' % mlogloss(test_y_pred, act))

    # Save the variables to disk.
    if TASK == 'train':
        save_path = saver.save(sess, chkpt_file)
        print("Model saved in file: %s" % save_path)
