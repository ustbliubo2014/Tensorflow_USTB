# encoding: utf-8

"""
@author: liubo
@software: PyCharm Community Edition
@file: MissingValueProcess.py
@time: 2016/10/20 10:28
@contact: ustb_liubo@qq.com
@annotation: MissingValueProcess
"""
import sys
import logging
from logging.config import fileConfig
import os
from sklearn.preprocessing import Imputer
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

reload(sys)
sys.setdefaultencoding("utf-8")
# fileConfig('logger_config.ini')
# logger_error = logging.getLogger('errorhandler')


# 使用sklearn中的分类算法时,必须进行缺失值填充(knn, most_freq, medium, mean)
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
X = np.array([[1, 2], [np.nan, 3], [7, 6]])
Y = [[np.nan, 2], [6, np.nan], [7, np.nan]]
imp.fit(X)
print imp.transform(Y)


# 使用xgboost可以不用处理缺失值
from sklearn.datasets import load_iris, load_boston
import xgboost as xgb

# loading database
iris = load_iris()
data = iris['data']
lable = iris['target']
data[0][0] = np.nan
xgb_cmodel = xgb.XGBClassifier().fit(data, lable)

