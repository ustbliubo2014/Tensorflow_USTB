import sys
import os
from tensorflow.python.sparkflow.sparkflow import SparkFlow
from tensorflow.python.sparkflow.sparkflow import IteratorCached
from tensorflow.python.sparkflow.sparkflow import loadImg
import numpy as np
import tensorflow as tf
import time

global cluster
global g_result_output_path
global test_x
global test_y

flags = tf.app.flags
FLAGS = flags.FLAGS 
tf.app.flags.DEFINE_string('data_path', 'hdfs://namenode.safe.lycc.qihoo.net:9000/tmp/sparkflow/dnn_demo/data', 'The hdfs directory for read training data.')
tf.app.flags.DEFINE_string('test_data_path', 'hdfs://namenode.safe.lycc.qihoo.net:9000/tmp/sparkflow/dnn_demo/testdata', 'The hdfs directory for read test data.')
tf.app.flags.DEFINE_string('save_path', 'hdfs://namenode.safe.lycc.qihoo.net:9000/tmp/sparkflow/dnn_demo/saveModel', 'The hdfs directory for save model and summary writer files.')
tf.app.flags.DEFINE_integer('worker_number', 4, 'The number of tensorflow worker.')
tf.app.flags.DEFINE_integer('ps_number', 2, 'The number of tensorflow ps.')

def extractLabel(filename):
    arr=filename.split("/")
    return arr[len(arr)-1][0:1]

def oneHot(sLabel):
    arr = np.zeros(10,np.float32)
    arr[int(''.join(sLabel))] = int(''.join(sLabel))
    return arr.transpose()

def startWorker( index, iterator):
    print "index is ", index
    global cluster
    print "in startWorker, global cluster is  ", str(cluster)
    gpu_options = tf.GPUOptions(allow_growth = True)
    server = tf.train.Server(cluster, job_name="worker", task_index=index, 
                             config=tf.ConfigProto(gpu_options = gpu_options,is_spark_application=True, allow_soft_placement = True))
    learning_rate = 0.002
    training_epochs = 2000
    batch_size = 120
    
    with tf.device(tf.train.replica_device_setter(worker_device=("/job:worker/task:%d"%(index)), cluster=cluster)):
        # count the number of updates
        global_step = tf.get_variable('global_step', [],initializer = tf.constant_initializer(0), trainable = False)
        # input images
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, shape=[None, 784], name="x-input")
            y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y-input")
        # model parameters will change during training so we use tf.Variable
        tf.set_random_seed(1)
        with tf.name_scope("weights"):
            W1 = tf.Variable(tf.random_normal([784, 100]))
            W2 = tf.Variable(tf.random_normal([100, 10]))
        # bias
        with tf.name_scope("biases"):
            b1 = tf.Variable(tf.zeros([100]))
            b2 = tf.Variable(tf.zeros([10]))
        # implement model
        with tf.name_scope("softmax"):
            # y is our prediction
            z2 = tf.add(tf.matmul(x,W1),b1)
            a2 = tf.nn.sigmoid(z2)
            z3 = tf.add(tf.matmul(a2,W2),b2)
            y = tf.nn.softmax(z3)
        # specify cost function
        with tf.name_scope('cross_entropy'):
            # this is our cost
            cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
        # specify optimizer
        with tf.name_scope('train'):
            # optimizer is an "operation" which we can execute in a session
            grad_op = tf.train.GradientDescentOptimizer(learning_rate)
            train_op = grad_op.minimize(cross_entropy, global_step=global_step)
        with tf.name_scope('Accuracy'):
            # accuracy
            correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # create a summary for our cost and accuracy
        tf.scalar_summary("cost", cross_entropy)
        tf.scalar_summary("accuracy", accuracy)
        # merge all summaries into a single "operation" which we can execute in a session 
        summary_op = tf.merge_all_summaries()
        init_op = tf.initialize_all_variables()
        print("Variables initialized ...")
        sv = tf.train.Supervisor(is_chief=(index == 0),
                                global_step=global_step,
                                init_op=init_op)
        begin_time = time.time()
        with sv.prepare_or_wait_for_session(server.target,config=tf.ConfigProto(allow_soft_placement = True,log_device_placement=True)) as sess:
            # create log writer object (this will log on every machine)
            writer = tf.train.SummaryWriter(SparkFlow.getLogPath(), graph=tf.get_default_graph())
            # perform training cycles
            start_time = time.time()
            cycle = 0
            itercache = IteratorCached(iterator, batch_size)
            for epoch in range(training_epochs):
                # number of batches in one epoch                
                cycle += 1
                for step in range(itercache.batchCount()):
                    iterator_curr = itercache.nextBatch()
                    flag = 0
                    for iter in iterator_curr:
                        if 0 == flag:
                            train_x = iter[1].reshape(1,784)
                            train_y = oneHot(iter[0]).reshape(1,10)
                        train_x = np.concatenate((train_x, iter[1].reshape(1,784)))
                        train_y = np.concatenate((train_y, oneHot(iter[0]).reshape(1,10)))
                        flag = 1
                    print "this is the ", cycle, " th epoch"
                    _, cost, summary, gstep = sess.run(
                            [train_op, cross_entropy, summary_op, global_step],
                            feed_dict={x: train_x, y_: train_y})
                    writer.add_summary(summary, gstep)                    
                    elapsed_time = time.time() - start_time
                    start_time = time.time()
                    print("Step: %d," % (gstep+1),
                        " Epoch: %2d," % (epoch+1),                            
                        " Cost: %.4f," % cost,
                        " AvgTime: %3.2fms" % float(elapsed_time*1000))
            print("Final Cost: %.4f" % cost)
            
            global test_x
            global test_y
            test_accuracy = sess.run(accuracy, feed_dict={x: np.asarray(test_x).reshape(111,784), y_: np.asarray(test_y).reshape(111,10)}) 
            print("Test-Accuracy: %2.2f" % test_accuracy)
            print("Total Time: %3.2fs" % float(time.time() - begin_time))

            if(index == 0):
                print "saving model..."
                global g_result_output_path
                SparkFlow.saveModel(sv.saver, sess, g_result_output_path)
                print "saving model, done!"
            sv.stop()        
            print("done")     
    yield ""

def Main():
    global g_result_output_path 
    g_result_output_path = FLAGS.save_path

    sf= SparkFlow(FLAGS.worker_number, FLAGS.ps_number)

    global cluster
    cluster = sf.getCluster()
    print "in Main() global cluster is ", str(cluster)

    import cv2    
    imgRDD = loadImg(sf.sc, FLAGS.data_path)
    imgArrRDD = imgRDD.map( lambda (x,y): (extractLabel(x), np.multiply(cv2.imdecode(y,0),1.0/255.0)) ).cache()

    testimgRDD = loadImg(sf.sc, FLAGS.test_data_path)
    testimgArrRDD = testimgRDD.map(lambda (x,y): (extractLabel(x), np.multiply(cv2.imdecode(y,0),1.0/255.0))).cache()
    global test_x
    global test_y
    test_x = testimgArrRDD.map(lambda (x,y): y).collect()
    test_y = testimgArrRDD.map(lambda (x,y): oneHot(x)).collect()

    sf.train(imgArrRDD, startWorker)

if __name__ == "__main__":
    Main()    

