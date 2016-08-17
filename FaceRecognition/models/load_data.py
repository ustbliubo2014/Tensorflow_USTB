# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: load_data.py
@time: 2016/8/17 11:44
@contact: ustb_liubo@qq.com
@annotation: load_data
"""
import sys
import logging
from logging.config import fileConfig
import os
from scipy.misc import imread, imresize
import numpy as np
import traceback
import shutil
import pdb
import msgpack_numpy

reload(sys)
sys.setdefaultencoding("utf-8")
fileConfig('logger_config.ini')
logger_error = logging.getLogger('errorhandler')


def load_originalimages(folder, pic_shape=(128, 128, 3)):
    # 前180个人进行训练,后面的20人人进行人脸验证
    pic_list = os.listdir(folder)
    model_data = []
    model_label = []
    for pic in pic_list:
        try:
            this_label = int(pic.split('-')[0])
            this_data = imresize(imread(os.path.join(folder, pic)), pic_shape)
            model_data.append(this_data)
            model_label.append(this_label)
        except:
            traceback.print_exc()
            continue

    model_data = np.asarray(model_data) / 255.0
    model_label = np.asarray(model_label)
    return model_data, model_label


if __name__ == '__main__':
    model_data, model_label = load_originalimages(folder='/data/liubo/face/originalimages/originalimages_model')
    msgpack_numpy.dump((model_data, model_label), open('/data/liubo/face/originalimages/originalimages_model.p', 'wb'))
