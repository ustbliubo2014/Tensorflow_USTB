# encoding: utf-8

"""
@author: liubo
@software: PyCharm
@file: tmp.py
@time: 2017/2/8 18:08
@contact: ustb_liubo@qq.com
@annotation: tmp
"""
import sys
import logging
from logging.config import fileConfig
import os

reload(sys)
sys.setdefaultencoding("utf-8")
# fileConfig('logger_config.ini')
# logger_error = logging.getLogger('errorhandler')


import tensorflow as tf
import os
import tarfile
import requests

inception_pretrain_model_url = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'

# 下载inception pretrain模型
inception_pretrain_model_dir = "inception_pretrain"
if not os.path.exists(inception_pretrain_model_dir):
	os.makedirs(inception_pretrain_model_dir)

filename = inception_pretrain_model_url.split('/')[-1]
filepath = os.path.join(inception_pretrain_model_dir, filename)

if not os.path.exists(filepath):
	print("开始下载: ", filename)
	r = requests.get(inception_pretrain_model_url, stream=True)
	with open(filepath, 'wb') as f:
		for chunk in r.iter_content(chunk_size=1024):
			if chunk:
				f.write(chunk)

print("下载完成, 开始解压: ", filename)
tarfile.open(filepath, 'r:gz').extractall(inception_pretrain_model_dir)

# TensorBoard log目录
log_dir = 'inception_log'
if not os.path.exists(log_dir):
	os.makedirs(log_dir)

# 加载inception graph
inception_graph_def_file = os.path.join(inception_pretrain_model_dir, 'classify_image_graph_def.pb')
with tf.Session() as sess:
	with tf.gfile.FastGFile(inception_graph_def_file, 'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
		tf.import_graph_def(graph_def, name='')
	writer = tf.train.SummaryWriter(log_dir, sess.graph)
	writer.close()
