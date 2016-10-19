# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: DeepId_keras.py
@time: 2016/8/17 14:44
@contact: ustb_liubo@qq.com
@annotation: DeepId_keras
"""
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers import BatchNormalization, Flatten, Dense, Dropout
from keras.layers import Input, merge
from keras.models import Model
from keras import regularizers
import msgpack_numpy
from keras.utils import np_utils
import numpy as np
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
import os


# global constants
NB_CLASS = 181  # number of classes
DIM_ORDERING = 'th'  # 'th' (channels, width, height) or 'tf' (width, height, channels)
WEIGHT_DECAY = 0.0005  # L2 regularization factor
USE_BN = True  # whether to use batch normalization
PIC_SHAPE = 128
CHANNEL_NUM = 3


def conv2D_bn(x, nb_filter, nb_row, nb_col,
              border_mode='same', subsample=(1, 1),
              activation='relu', batch_norm=USE_BN,
              weight_decay=WEIGHT_DECAY, dim_ordering=DIM_ORDERING):
    '''Utility function to apply to a tensor a module conv + BN
    with optional weight decay (L2 weight regularization).
    '''
    if weight_decay:
        W_regularizer = regularizers.l2(weight_decay)
        b_regularizer = regularizers.l2(weight_decay)
    else:
        W_regularizer = None
        b_regularizer = None
    x = Convolution2D(nb_filter, nb_row, nb_col,
                      subsample=subsample,
                      activation=activation,
                      border_mode=border_mode,
                      W_regularizer=W_regularizer,
                      b_regularizer=b_regularizer,
                      dim_ordering=dim_ordering)(x)
    if batch_norm:
        x = BatchNormalization()(x)
    return x

# Define image input layer

if DIM_ORDERING == 'th':
    img_input = Input(shape=(CHANNEL_NUM, PIC_SHAPE, PIC_SHAPE))
    CONCAT_AXIS = 1
elif DIM_ORDERING == 'tf':
    img_input = Input(shape=(PIC_SHAPE, PIC_SHAPE, CHANNEL_NUM))
    CONCAT_AXIS = 3
else:
    raise Exception('Invalid dim ordering: ' + str(DIM_ORDERING))


# 128x128 -> 64x64
conv1 = conv2D_bn(img_input, 64, 3, 3, subsample=(1, 1), border_mode='same')
pool1 = MaxPooling2D((2, 2), strides=(2, 2), dim_ordering=DIM_ORDERING)(conv1)
# 64x64 -> 32x32
conv2 = conv2D_bn(pool1, 128, 3, 3, subsample=(1, 1), border_mode='same')
pool2 = MaxPooling2D((2, 2), strides=(2, 2), dim_ordering=DIM_ORDERING)(conv2)
# 32x32 -> 16x16
conv3 = conv2D_bn(pool2, 256, 3, 3, subsample=(1, 1), border_mode='same')
pool3 = MaxPooling2D((2, 2), strides=(2, 2), dim_ordering=DIM_ORDERING)(conv3)
# 16x16 -> 8x8
conv4 = conv2D_bn(pool3, 512, 3, 3, subsample=(1, 1), border_mode='same')
pool4 = MaxPooling2D((2, 2), strides=(2, 2), dim_ordering=DIM_ORDERING)(conv4)
# 8x8 -> 8x8
conv5 = conv2D_bn(pool4, 512, 3, 3, subsample=(1, 1), border_mode='same')
#
flatten = merge([conv5, pool4], mode='concat', concat_axis=CONCAT_AXIS)
flatten = Flatten()(flatten)
flatten = Dropout(0.5)(flatten)
# flatten = Flatten()(conv4)

fc1 = Dense(NB_CLASS, activation='relu')(flatten)
fc1 = Dropout(0.5)(fc1)
fc2 = Dense(NB_CLASS, activation='relu')(fc1)
fc2 = Dropout(0.5)(fc2)

preds = Dense(NB_CLASS, activation='softmax')(fc2)

# Define model

model = Model(input=img_input, output=preds)
model.compile(optimizer=SGD(momentum=0.9), loss='categorical_crossentropy')

print model.summary()

model_data, model_label = msgpack_numpy.load(open('/data/liubo/face/originalimages/originalimages_model.p', 'rb'))
model_data = np.transpose(model_data, (0, 3, 1, 2))
model_label = np_utils.to_categorical(model_label, NB_CLASS)
print model_data.shape, model_label.shape

weight_file = 'originalimages_model.weight'
if os.path.exists(weight_file):
    model.load_weights(weight_file)

checkpointer = ModelCheckpoint(filepath=weight_file, monitor='val_loss', save_best_only=True)
model.fit(model_data, model_label, batch_size=64, nb_epoch=10, verbose=1, shuffle=True,
          validation_split=0.1, callbacks=[checkpointer])
