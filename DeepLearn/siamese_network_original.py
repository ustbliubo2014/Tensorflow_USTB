# encoding: utf-8

"""
@author: liubo
@software: PyCharm Community Edition
@file: siamese_network_1.py
@time: 2016/10/20 17:53
@contact: ustb_liubo@qq.com
@annotation: siamese_network_original
"""
import random
import numpy as np
import time
import tensorflow as tf
import input_data
import math
import sys
import inference_28
import pdb

mnist = input_data.read_data_sets("/home/liubo-it/siamese_tf_mnist/MNIST_data",one_hot=False)



# 将读入的数据根据label生成正负样本(保证正负样本均衡[所有可能的正样本和相同数量的负样本])
def create_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(10)]) - 1
    for d in range(10):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i+1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, 10)
            dn = (d + inc) % 10
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)


def build_model(X_):
    model = inference_28.inference(X_)
    return model


def contrastive_loss(y_true, y_pred):
    margin = 1
    return tf.reduce_mean(y_true * tf.square(y_pred) + (1 - y_true) * tf.square(tf.maximum(margin - y_pred, 0)))


def compute_accuracy(prediction, labels):
    return labels[prediction.ravel() < 0.5].mean()


def next_batch(s, e, inputs, labels):
    input1 = inputs[s:e, 0]
    input2 = inputs[s:e, 1]
    y = np.reshape(labels[s:e], (len(range(s, e)), 1))
    return input1, input2, y


def distance_model(batch_x1, batch_x2):
    with tf.variable_scope("siamese") as scope:
        model1 = build_model(batch_x1)
        scope.reuse_variables()
        model2 = build_model(batch_x2)
    distance = tf.sqrt(tf.reduce_sum(tf.pow(tf.sub(model1, model2), 2), 1, keep_dims=True))
    return model1, model2, distance


def siamese_loss(distance, labels):
    loss = contrastive_loss(labels, distance)
    tf.add_to_collection('losses', loss)
    tf.add_to_collection('accuracy', loss)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


if __name__ == '__main__':

    # with tf.device('/gpu:0'):
    # Initializing the variables
    init = tf.initialize_all_variables()
    # the data, shuffled and split between train and test sets
    X_train = mnist.train._images
    y_train = mnist.train._labels
    X_test = mnist.test._images
    y_test = mnist.test._labels
    X_train = np.reshape(X_train, (X_train.shape[0], 28, 28, 1))
    X_test = np.reshape(X_test, (X_test.shape[0], 28, 28, 1))

    batch_size = 128
    global_step = tf.Variable(0, trainable=False)
    # 调siamese网络时, 学习率要比分类网络小至少一个数量级
    starter_learning_rate = 0.0001
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 10, 0.1, staircase=True)
    # create training+test positive and negative pairs
    digit_indices = [np.where(y_train == i)[0] for i in range(10)]
    tr_pairs, tr_y = create_pairs(X_train, digit_indices)
    digit_indices = [np.where(y_test == i)[0] for i in range(10)]
    te_pairs, te_y = create_pairs(X_test, digit_indices)

    images_L = tf.placeholder(tf.float32, shape=([None, 28, 28, 1]), name='L')
    images_R = tf.placeholder(tf.float32, shape=([None, 28, 28, 1]), name='R')
    labels = tf.placeholder(tf.float32, shape=([None, 1]), name='gt')
    dropout_f = tf.placeholder("float")

    # 构建一个梦想, 返回distance, 直接和label计算loss就可以了

    # distance < tf.Tensor 'Sqrt:0' shape = (?, 1) dtype = float32 >
    # labels < tf.Tensor 'gt:0' shape = (?, 1) dtype = float32 >
    model1, model2, distance = distance_model(images_L, images_R)

    loss = contrastive_loss(labels, distance)

    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'l' in var.name]
    batch = tf.Variable(0)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
    session = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        gpu_options=gpu_options)
    )

    with session as sess:
        # sess.run(init)
        tf.initialize_all_variables().run()
        # Training cycle
        for epoch in range(30):
            avg_loss = 0.
            avg_acc = 0.
            total_batch = int(X_train.shape[0]/batch_size)
            start_time = time.time()
            # Loop over all batches
            batch_size = 128
            for i in range(total_batch):
                s = i * batch_size
                e = (i+1) *batch_size
                # Fit training using batch data
                input1, input2, y = next_batch(s, e, tr_pairs, tr_y)
                _, loss_value, predict = sess.run([optimizer, loss, distance],
                                               feed_dict={images_L:input1, images_R:input2, labels:y, dropout_f:0.9})
                feature1 = model1.eval(feed_dict={images_L:input1, dropout_f:0.9})
                feature2 = model2.eval(feed_dict={images_R:input2, dropout_f:0.9})

                tr_acc = compute_accuracy(predict, y)
                if math.isnan(tr_acc) and epoch != 0:
                    print('tr_acc %0.2f' % tr_acc)
                    print('nan')
                    sys.exit()
                avg_loss += loss_value
                avg_acc += tr_acc * 100

            duration = time.time() - start_time
            print('epoch %d  time: %f loss %0.5f acc %0.2f' %(epoch, duration, avg_loss/(total_batch), avg_acc/total_batch))
        y = np.reshape(tr_y,(tr_y.shape[0],1))
        predict = distance.eval(feed_dict={images_L:tr_pairs[:,0],images_R:tr_pairs[:,1],labels:y,dropout_f:1.0})
        tr_acc = compute_accuracy(predict,y)
        print('Accuract training set %0.2f' % (100 * tr_acc))

        # Test model
        predict = distance.eval(feed_dict={images_L:te_pairs[:,0], images_R:te_pairs[:,1], labels:y, dropout_f:1.0})
        y = np.reshape(te_y,(te_y.shape[0],1))
        te_acc = compute_accuracy(predict,y)
        print('Accuract test set %0.2f' % (100 * te_acc))

