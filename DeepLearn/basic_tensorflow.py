# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: basic_tensorflow.py
@time: 2016/7/22 19:31
@contact: ustb_liubo@qq.com
@annotation: basic_tensorflow
"""
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='basic_tensorflow.log',
                    filemode='a+')
import pdb
import numpy as np
import tensorflow as tf
print ("packs loaded")


"""
 In order to make things happen, we need 'session'!
"""
sess = tf.Session()


"""
 Run session with tf variable
"""
hello = tf.constant("Hello, it's me.")
print ("hello is %s" % (hello))

hello_out = sess.run(hello)
print ("\nType of 'hello': %s" % (type(hello)))
print ("Type of 'hello_out': %s" % (type(hello_out)))
print
print (hello_out)

"""
 Until you run session, nothing happens!
"""
print ("\nUntil you run session, nothing happens!")

"""
 There are other types as well
  1. Constant types
  2. Operators
  3. Variables
  4. Placeholder (Buffers)
"""

"""
 Constant types
"""
print ("Constant types (numpy)")
a = tf.constant(1.5)
b = tf.constant(2.5)
print ("'a': %s \n     Type: %s" % (a, type(a)))
print ("'a': %s \n     Type: %s" % (b, type(b)))
a_out = sess.run(a)
b_out = sess.run(b)
print
print ("Type of 'a_out' is %s" % (type(a_out)))
print ("Type of 'b_out' is %s" % (type(b_out)))
print ("a_out is %.1f b_out is %.1f a_out+b_out is %.1f"
       % (a_out, b_out, a_out+b_out))


"""
 Operators are also tf variables
 先用tf定义操作,在通过sess.run获得结果(输入Tensor类型的数据,run后得到numpy类型的结果)
"""
print ("Operators (tf.add, tf.mul)")
add = tf.add(a, b)
print (" 'add' is %s \n    type is %s" % (add, type(add)))
add_out = sess.run(add)
print (" 'add_out' is %s \n    type is %s" % (add_out, type(add_out)))
mul = tf.mul(a, b)
print (" 'mul' is %s \n    type is %s" % (mul, type(mul)))
mul_out = sess.run(mul)
print (" 'mul_out' is %s \n    type is %s" % (mul_out, type(mul_out)))



"""
 Variables & PlaceHolder
    定义变量 Weight, Bias,
    然后初始化所有参数,
    init = tf.initialize_all_variables()
    sess.run(init)
    这样就可以通过Weight.eval(sess)获得变量的值
"""
print ("Variables & PlaceHolders")
X = np.random.rand(1, 20)
Input = tf.placeholder(tf.float32, [None, 20])
Weight = tf.Variable(tf.random_normal([20, 10], stddev=0.5))
Bias = tf.Variable(tf.zeros([1, 10]))
print (" 'Input': %s \n    Type is %s" % (Input, type(Input)))
print (" 'Weight': %s \n    Type is %s" % (Weight, type(Weight)))
print (" 'Bias': %s \n    Type is %s" % (Bias, type(Bias)))
# Weight_out = sess.run(Weight) # <= This is not allowed!
# print Weight.eval(sess) # <= This is not also allowed! (Do You Know Why??)


"""
 Initialize Variables
"""
print ("Initialize Variables")
init = tf.initialize_all_variables()
sess.run(init)
print (" 'Weight': %s \n  Type is %s" % (Weight, type(Weight)))
print (" 'Bias': %s \n  Type is %s" % (Bias, type(Bias)))
print
print (Weight.eval(sess))

"""
 Operators with PlaceHolder
 (This is very important !)
 (Remember 'feed_dict' !!!)
 先用placeholder定义变量,然后定义Graph(使用预先定义的placeholder),然后通过run获得结果(使用feed_dict将参数值传入(dict的形式))
"""
print ("Operators with PlaceHolder (tf.add, tf.mul)")
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
add = tf.add(x, y)
mul = tf.mul(x, y)
add_out = sess.run(add, feed_dict={x:5.0, y:6.0})
mul_out = sess.run(mul, feed_dict={x:5.0, y:6.0})
print (" 'add_out' is %s \n  Type is %s" % (add_out, type(add_out)))
print (" 'mul_out' is %s \n  Type is %s" % (mul_out, type(mul_out)))

