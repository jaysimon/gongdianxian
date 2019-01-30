# /usr/bin/env python
# -*- coding: utf-8 -*- #
# Copyright (C) 2018 Houwei and Tuhang
# FileName : predict.py
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
from postprocess import evaluate as ev


def load_data(sDataPath):
    """Prepare data for testing
    """
    npX = np.load(sDataPath)
    print("[Log] Read %s shape:%s" % (sDataPath, str(npX.shape)))
    npX = np.reshape(npX, (npX.shape[0], npX.shape[1], npX.shape[2], 1))
    return npX


def predict(sModelPath, npX):
    """Predict
    """
    model = load_model(sModelPath)
    # model.summary()
    print("[Log] Model %s Predicting..." % sModelPath)
    npY = model.predict(npX)
    npY = npY[:, 1]
    return npY


def save_result(sPath, npY, fThre, iAdd):
    """Save predict in ordered format
    """
    fileRes = open(sPath, "w")
    sLine = "signal_id,target\n"
    fileRes.write(sLine)

    npY = npY > fThre
    npY = npY + 0
    
    for iIndex in range(len(npY)):
        sLine = str(iIndex + iAdd) + "," + str(npY[iIndex]) + "\n"
        fileRes.write(sLine)
    fileRes.close()
    print("[Log] Sum is %d" % np.sum(npY))
    print("[Log] Write result to %s" % sPath)


def predict_test(sModelPath):
    """Predict for test data, save result to "../list/result.csv"
    """
    sDataPath = "../data/mel_test.npy"
    sResultPath = "../list/result.csv"

    npX = load_data(sDataPath)
    npY = predict(sModelPath, npX)
    save_result(sResultPath, npY, 0.5, 8712)


def predict_train(sModelPath):
    """Predict for all train data
    """
    sDataPath = "../data/mel_train.npy"
    sTagPath = "../list/metadata_train.csv"

    npX = load_data(sDataPath)
    npPredict = predict(sModelPath, npX)
    npTag = pd.read_tag(sTagPath)
    fBEP = ev.PRCurve(npPredict, npTag)
    ev.FScore(npPredict, npTag, fBEP)
    ev.ROC_AUC(npPredict, npTag)
    mcc = ev.mcc(npPredict, npTag, 0.5) 
    mcc = ev.mcc(npPredict, npTag, fBEP) 


def predict_val(sModelDir):
    """Predict val data by all models in sModelDir
    """
    sDataPath = "../data/k_folder/mel_val_0.npy"
    sTagPath = "../data/k_folder/valY_0.npy"
    
    fMaxMcc = 0
    sBestModel = ""
    for sDirPath, lDirNames, lFileNames in os.walk(sModelDir):
        for sFileName in lFileNames:
            if sFileName.find("hdf5") == -1:
                continue
            sModelPath = os.path.join(sDirPath, sFileName)
            npX = load_data(sDataPath)
            npPredict = predict(sModelPath, npX)
            npTag = np.load(sTagPath)
            fBEP = ev.PRCurve(npPredict, npTag)
            ev.FScore(npPredict, npTag, fBEP)
            ev.ROC_AUC(npPredict, npTag)
            mcc = ev.mcc(npPredict, npTag, 0.5)
            if mcc > fMaxMcc:
                fMaxMcc = mcc
                sBestModel = sFileName
            mcc = ev.mcc(npPredict, npTag, fBEP)
            if mcc > fMaxMcc:
                fMaxMcc = mcc
                sBestModel = sFileName
            print()
    print("[Log] Best mcc: %f, Best model:%s" % (fMaxMcc, sBestModel))


def main():
    sModelPath1 = "../model/cnn_no_enhance/cnn-64-100-0.975.hdf5"
    sModelPath2 = "../model/excellent/cnn-64-66-0.972-n592-p0.529.hdf5"
    sModelDir = "../model/cnn_no_enhance/"
    predict_val(sModelDir)
    # predict_train(sModelPath2)
    # predict_test(sModelPath)


if __name__ == "__main__":
    main()
