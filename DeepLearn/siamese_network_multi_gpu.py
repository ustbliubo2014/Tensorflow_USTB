# encoding: utf-8

"""
@author: liubo
@software: PyCharm Community Edition
@file: siamese_network_multi_gpu.py
@time: 2016/10/20 18:04
@contact: ustb_liubo@qq.com
@annotation: siamese_network_multi_gpu
"""

import sys
import logging
from logging.config import fileConfig
import os
import os.path
import re
import time
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import siamese_network
import pdb
import input_data


TOWER_NAME = "tower"
gpu_num = 3
MOVING_AVERAGE_DECAY = 0.9999
GPU_MEMERY_ALLOCATE = 0.4
batch_size = 128
train_dir = './train'
if not os.path.exists(train_dir):
    os.makedirs(train_dir)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('num_gpus', 3,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

# 在inference中构造自己的网络结构和loss函数
def inference(batch_x1, batch_x2):
      return siamese_network.distance_model(batch_x1, batch_x2)


def loss(distance, labels):
    return siamese_network.siamese_loss(distance, labels)


def tower_loss(scope, batch_x1, batch_x2, labels):
    # Build inference Graph.
    model1, model2, distance = inference(batch_x1, batch_x2)

    _ = loss(distance, labels)

    # Assemble all of the losses for the current tower only.
    losses = tf.get_collection('losses', scope)

    # Calculate the total loss for the current tower.
    total_loss = tf.add_n(losses, name='total_loss')

    pdb.set_trace()

    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    for l in losses + [total_loss]:
        loss_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', l.op.name)
        tf.scalar_summary(loss_name + ' (raw)', l)
        tf.scalar_summary(loss_name, loss_averages.average(l))

    with tf.control_dependencies([loss_averages_op]):
        total_loss = tf.identity(total_loss)
    return total_loss, distance


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(0, grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def network(epochs=200, predict=False):
    global_step_val = int(100)
    """Train for a number of steps."""
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # Create a variable to count the number of train() calls. This equals the
        # number of batches processed * FLAGS.num_gpus.
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(global_step_val), trainable=False)

        # Calculate the learning rate schedule.
        starter_learning_rate = 0.0001
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 10, 0.1, staircase=True)

        # Create an optimizer that performs gradient descent.
        # opt = tf.train.GradientDescentOptimizer(lr)
        opt = tf.train.AdamOptimizer(learning_rate)

        # Calculate the gradients for each model tower.
        tower_grads = []
        tower_acc = []
        tower_feeds = []
        tower_logits = []
        for i in xrange(gpu_num):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('%s_%d' % (TOWER_NAME, i)) as scope:
                    # all towers.
                    images_L = tf.placeholder(tf.float32, shape=([None, 28, 28, 1]), name='L')
                    images_R = tf.placeholder(tf.float32, shape=([None, 28, 28, 1]), name='R')
                    labels = tf.placeholder(tf.float32, shape=([None, 1]), name='gt')
                    tower_feeds.append((images_L, images_R, labels))
                    loss, logits = tower_loss(scope, images_L, images_R, labels)

                    tower_logits.append(logits)

                    # all accuracy
                    tower_acc.append(tf.get_collection('accuracy', scope)[0])

                    # Reuse variables for the next tower.
                    tf.get_variable_scope().reuse_variables()

                    # Retain the summaries from the final tower.
                    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                    # Calculate the gradients for the batch of data on this CIFAR tower.
                    grads = opt.compute_gradients(loss)

                    # Keep track of the gradients across all towers.
                    tower_grads.append(grads)

        # average accuracy
        accuracy = tf.add_n(tower_acc) / len(tower_acc)

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = average_gradients(tower_grads)


        # Add a summary to track the learning rate.
        summaries.append(tf.scalar_summary('learning_rate', learning_rate))

        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                summaries.append(
                    tf.histogram_summary(var.op.name + '/gradients', grad))

        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            summaries.append(tf.histogram_summary(var.op.name, var))

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        # Group all updates to into a single train op.
        train_op = tf.group(apply_gradient_op, variables_averages_op)

        # Create a saver.
        # saver = tf.train.Saver(tf.all_variables())
        saver = tf.train.Saver()

        # Build the summary operation from the last tower summaries.
        summary_op = tf.merge_summary(summaries)

        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()

        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=GPU_MEMERY_ALLOCATE)
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            gpu_options=gpu_options)
        )
        sess.run(init)


        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        graph_def = sess.graph.as_graph_def(add_shapes=True)
        summary_writer = tf.train.SummaryWriter(train_dir,
                                                graph_def=graph_def)
        return sess, saver, summary_writer, train_op, loss, accuracy, global_step, learning_rate, tower_feeds, tower_logits


def train_network():

    mnist = input_data.read_data_sets("/home/liubo-it/siamese_tf_mnist/MNIST_data", one_hot=False)
    X_train = mnist.train._images
    y_train = mnist.train._labels
    X_test = mnist.test._images
    y_test = mnist.test._labels
    X_train = np.reshape(X_train, (X_train.shape[0], 28, 28, 1))
    X_test = np.reshape(X_test, (X_test.shape[0], 28, 28, 1))  # load model
    digit_indices = [np.where(y_train == i)[0] for i in range(10)]
    tr_pairs, tr_y = siamese_network.create_pairs(X_train, digit_indices)
    digit_indices = [np.where(y_test == i)[0] for i in range(10)]
    te_pairs, te_y = siamese_network.create_pairs(X_test, digit_indices)


    sess, saver, summary_writer, train_op, loss, accuracy, global_step, lr, tower_feeds, tower_logits = network()
    train_step = 0

    # fit model by mini batch
    avg_loss, avg_acc = 0.0, 0.0
    for epoch in range(30):
        avg_loss = 0.
        avg_acc = 0.
        total_batch = int(X_train.shape[0] / batch_size)
        total_batch = 30
        start_time = time.time()
        # Loop over all batches

        train_step += 1
        feeds = {}
        for i in range(0, total_batch, gpu_num):
            for gpu_id in xrange(gpu_num):
                s = i * batch_size
                e = (i + gpu_id + 1) * batch_size
                input1, input2, y = siamese_network.next_batch(s, e, tr_pairs, tr_y)
                # tower_feeds.append((images_L, images_R, labels))
                feeds[tower_feeds[gpu_id][0]] = input1
                feeds[tower_feeds[gpu_id][1]] = input2
                feeds[tower_feeds[gpu_id][2]] = y
            tmp, global_step_val, loss_val, acc_val = sess.run([train_op, global_step, loss, accuracy], feed_dict=feeds)
            avg_loss += loss_val
        avg_loss = avg_loss / total_batch
        print 'avg_loss :', avg_loss


if __name__ == "__main__":
    train_network()
