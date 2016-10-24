# encoding: utf-8

"""
@author: liubo
@software: PyCharm Community Edition
@file: data_process.py
@time: 2016/10/21 18:23
@contact: ustb_liubo@qq.com
@annotation: data_process
"""
import sys
import logging
from logging.config import fileConfig
import os
import cv2
import msgpack_numpy
import numpy as np
import traceback
from sklearn.cross_validation import train_test_split

reload(sys)
sys.setdefaultencoding("utf-8")
# fileConfig('logger_config.ini')
# logger_error = logging.getLogger('errorhandler')


root_folder = '/data/liubo/face/all_pic_data/FaceScrub'
person_list = os.listdir(root_folder)
all_data = []
all_label = []
index = 0
for person in person_list[:20]:
    print person, index
    pic_list = os.listdir(os.path.join(root_folder, person))
    for pic in pic_list:
        try:
            pic_path = os.path.join(root_folder, person, pic)
            im = cv2.resize(cv2.imread(pic_path), (150, 150))
            all_data.append(im)
            all_label.append(index)
        except:
            traceback.print_exc()
            continue
    index += 1

all_data = np.asarray(all_data)
all_label = np.asarray(all_label)


train_data, test_data, train_label, test_label = train_test_split(all_data, all_label)
print train_data.shape, train_label.shape, test_data.shape, test_label.shape
msgpack_numpy.dump((train_data, test_data, train_label, test_label),
                   open('/data/liubo/face/all_pic_data/FaceScrub.p', 'wb'))
