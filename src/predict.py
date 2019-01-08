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
import numpy as np
from keras.models import load_model

import prepare_data as pd


def prepare_test_data(sDataPath):
    """Prepare data for testing
    """
    npX = np.load(sDataPath)
    npX = np.reshape(npX, (npX.shape[0], npX.shape[1], npX.shape[2], 1))
    return npX


def predict(sModelPath, npX):
    """Predict
    """
    model = load_model(sModelPath)
    print("[Log] Model %s Predicting..." % sModelPath)
    npY = model.predict(npX)
    return npY


def save_result(sPath, npY, iAdd):
    """Save predict in ordered format
    """
    fileRes = open(sPath, "w")
    sLine = "signal_id,target\n"
    fileRes.write(sLine)

    npY = npY[:, 1]
    npY = npY>0.5
    npY = npY + 0
    
    for iIndex in range(len(npY)):
        sLine = str(iIndex + iAdd) + "," + str(npY[iIndex]) + "\n"
        fileRes.write(sLine)
    fileRes.close()
    print("[Log] Sum is %d" % np.sum(npY))
    print("[Log] Write result to %s" % sPath)

def main():
    sDataPath = "../data/mel_test.npy"
    sModelPath = "../model/cnn-64-66-0.972.hdf5"
    sResultPath = "../list/result.csv"

    npX = prepare_test_data(sDataPath)
    npY = predict(sModelPath, npX)
    save_result(sResultPath, npY, 8712)


if __name__ == "__main__":
    main()
