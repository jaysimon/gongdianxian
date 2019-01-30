# /usr/bin/env python
# -*- coding: utf-8 -*- #
# Copyright (C) 2018 Houwei and Tuhang
# FileName : train.py
# Author : Hou Wei
# Version : V1.0
# Date: 2018-01-08
# Description: Train
# History:


import os
import datetime
import random
import numpy as np
from keras.models import Model
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers import MaxPooling2D, Conv2D, Reshape, Input, CuDNNLSTM
from keras.utils import np_utils
from keras.utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint
import tensorflow as tf

import prepare_data as pd


def prepare_train_data(sPos, sNeg):
    """Prepare data for training
    """
    # sDataPath = "../data/mel_train.npy"
    # sTagPath = "../list/metadata_train.csv"
    # npX, npY = pd.prepare_data(sDataPath, sTagPath)
    npX, npY = pd.prepare_enh_data(sPos, sNeg)

    iLen = npX.shape[0]
    jIndex = list(range(iLen))
    random.shuffle(jIndex)
    npY = np_utils.to_categorical(npY, 2)
    npX = np.reshape(npX, (npX.shape[0], npX.shape[1], npX.shape[2], 1))
    iLen = npX.shape[0]
    jIndex = list(range(iLen))
    random.shuffle(jIndex)
    npY = npY[jIndex, :]
    npX = npX[jIndex, :]

    return npX, npY


def model_1():
    """Train , ori maxpooling
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         (None, 64, 1251, 1)       0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 64, 1251, 64)      640       
    _________________________________________________________________
    batch_normalization_1 (Batch (None, 64, 1251, 64)      256       
    _________________________________________________________________
    activation_1 (Activation)    (None, 64, 1251, 64)      0         
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 16, 312, 64)       0         
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 16, 312, 64)       0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 16, 312, 64)       36928     
    _________________________________________________________________
    batch_normalization_2 (Batch (None, 16, 312, 64)       256       
    _________________________________________________________________
    activation_2 (Activation)    (None, 16, 312, 64)       0         
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 4, 62, 64)         0         
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 4, 62, 64)         0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 4, 62, 64)         36928     
    _________________________________________________________________
    batch_normalization_3 (Batch (None, 4, 62, 64)         256       
    _________________________________________________________________
    activation_3 (Activation)    (None, 4, 62, 64)         0         
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 1, 12, 64)         0         
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 1, 12, 64)         0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 768)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 2)                 1538      
    =================================================================
    Total params: 76,802
    Trainable params: 76,418
    Non-trainable params: 384
    _________________________________________________________________
    """

    inputs = Input(shape=(64, 1251, 1))
    x = Conv2D(
        64, input_shape=[64, 1251, 1], data_format="channels_last",
        padding="same", kernel_size=(3, 3),
        kernel_initializer="glorot_uniform")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(4, 4))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(64, kernel_size=(3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(4, 5))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(64, kernel_size=(3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(4, 5))(x)
    x = Dropout(0.3)(x)

    x = Flatten()(x)
    predictions = Dense(2, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=predictions)
    return model


def model_2():
    """Model, same size maxpooling [3, 3]
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         (None, 64, 1251, 1)       0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 64, 1251, 64)      640       
    _________________________________________________________________
    batch_normalization_1 (Batch (None, 64, 1251, 64)      256       
    _________________________________________________________________
    activation_1 (Activation)    (None, 64, 1251, 64)      0         
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 21, 417, 64)       0         
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 21, 417, 64)       0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 21, 417, 64)       36928     
    _________________________________________________________________
    batch_normalization_2 (Batch (None, 21, 417, 64)       256       
    _________________________________________________________________
    activation_2 (Activation)    (None, 21, 417, 64)       0         
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 7, 139, 64)        0         
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 7, 139, 64)        0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 7, 139, 64)        36928     
    _________________________________________________________________
    batch_normalization_3 (Batch (None, 7, 139, 64)        256       
    _________________________________________________________________
    activation_3 (Activation)    (None, 7, 139, 64)        0         
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 2, 46, 64)         0         
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 2, 46, 64)         0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 5888)              0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 2)                 11778     
    =================================================================
    Total params: 87,042
    Trainable params: 86,658
    Non-trainable params: 384
    _________________________________________________________________
    """
    inputs = Input(shape=(64, 1251, 1))
    x = Conv2D(
        64, input_shape=[64, 1251, 1], data_format="channels_last",
        padding="same", kernel_size=(3, 3),
        kernel_initializer="glorot_uniform")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(3, 3))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(64, kernel_size=(3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(3, 3))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(64, kernel_size=(3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(3, 3))(x)
    x = Dropout(0.3)(x)

    x = Flatten()(x)
    predictions = Dense(2, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=predictions)
    return model


def model_lstm():
    """Model, lstm model

    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         (None, 64, 1251, 1)       0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 64, 1251, 64)      640       
    _________________________________________________________________
    batch_normalization_1 (Batch (None, 64, 1251, 64)      256       
    _________________________________________________________________
    activation_1 (Activation)    (None, 64, 1251, 64)      0         
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 1, 1251, 64)       0         
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 1, 1251, 64)       0         
    _________________________________________________________________
    reshape_1 (Reshape)          (None, 1251, 64)          0         
    _________________________________________________________________
    cu_dnnlstm_1 (CuDNNLSTM)     (None, 1251, 32)          12544     
    _________________________________________________________________
    cu_dnnlstm_2 (CuDNNLSTM)     (None, 1251, 16)          3200      
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 20016)             0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 2)                 40034     
    =================================================================
    Total params: 56,674
    Trainable params: 56,546
    Non-trainable params: 128
    _________________________________________________________________
    """
    inputs = Input(shape=(64, 1251, 1))
    x = Conv2D(
        64, input_shape=[64, 1251, 1], data_format="channels_last",
        padding="same", kernel_size=(3, 3),
        kernel_initializer="glorot_uniform")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(64, 1))(x)
    x = Dropout(0.3)(x)
    x = Reshape((1251, 64))(x)

    x = CuDNNLSTM(32, return_sequences=True)(x)
    x = CuDNNLSTM(16, return_sequences=True)(x)
    x = Flatten()(x)
    predictions = Dense(2, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=predictions)
    return model


def train(npX, npY):
    """Train, same size maxpooling [3, 3]
    """
    tmStart = datetime.datetime.now()

    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    model = model_2()
    model.summary()

    # parallel_model = multi_gpu_model(model, 3)
    parallel_model = model
    parallel_model.compile(
        optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    filepath = "../model/cnn-64-{epoch:02d}-{val_acc:.3f}.hdf5"
    checkpoint = ModelCheckpoint(
        filepath, monitor='val_acc', verbose=1,
        save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    parallel_model.fit(
        npX, npY, epochs=100, batch_size=32, callbacks=callbacks_list,
        shuffle=True, validation_split=0.2)

    tmEnd = datetime.datetime.now()
    tmDuration = tmEnd - tmStart
    print(
        "[Log]time:\n[Log]\tStart:\t\t%s"
        "\n[Log]\tEnd:\t\t%s\n[Log]\tDuration:\t%s" %
        (tmStart, tmEnd, tmDuration))


def main():
    sPos = "../data/k_folder/mel_pos_enh_0.npy"
    sNeg = "../data/k_folder/mel_neg_enh_0.npy"
    npX, npY = prepare_train_data(sPos, sNeg)
    train(npX, npY)


if __name__ == "__main__":
    main()
