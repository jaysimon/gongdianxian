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
from keras.layers import MaxPooling2D, Conv2D, Reshape, Input
from keras.utils import np_utils
from keras.utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint
import tensorflow as tf

import prepare_data as pd


def prepare_train_data():
    """Prepare data for training
    """
    sDataPath = "../data/mel_train.npy"
    sTagPath = "../list/metadata_train.csv"
    npX, npY = pd.prepare_data(sDataPath, sTagPath)

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


def train(npX, npY):
    """Train
    """
    tmStart = datetime.datetime.now()

    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

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
    model.summary()

    # parallel_model = multi_gpu_model(model, 4)
    parallel_model = model
    parallel_model.compile(
        optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    filepath = "../model/64-{epoch:02d}-{val_acc:.3f}.hdf5"
    checkpoint = ModelCheckpoint(
        filepath, monitor='val_acc', verbose=1,
        save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    parallel_model.fit(
        npX, npY, epochs=1, batch_size=32, callbacks=callbacks_list,
        shuffle=True, validation_split=0.2)

    tmEnd = datetime.datetime.now()
    tmDuration = tmEnd - tmStart
    print(
        "[Log]time:\n[Log]\tStart:\t\t%s"
        "\n[Log]\tEnd:\t\t%s\n[Log]\tDuration:\t%s" %
        (tmStart, tmEnd, tmDuration))


def main():
    npX, npY = prepare_train_data()
    train(npX, npY)


if __name__ == "__main__":
    main()
