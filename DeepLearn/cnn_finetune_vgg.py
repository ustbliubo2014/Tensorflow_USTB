# encoding: utf-8
__author__ = 'liubo'

"""
@version: 
@author: 刘博
@license: Apache Licence 
@contact: ustb_liubo@qq.com
@software: PyCharm
@file: cnn_finetune_vgg.py
@time: 2016/7/23 19:23
"""

import logging
import os

if not os.path.exists('log'):
    os.mkdir('log')

logging.basicConfig(level=logging.ERROR,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='log/cnn_finetune_vgg.log',
                    filemode='w')


def func():
    pass


class Main():
    def __init__(self):
        pass


if __name__ == '__main__':
    pass