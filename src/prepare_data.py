# /usr/bin/env python
# -*- coding: utf-8 -*- #
# Copyright (C) 2018 Houwei and Tuhang
# FileName : read_data.py
# Author : Hou Wei
# Version : V1.0
# Date: 2018-01-03
# Description: Prepare data for training and testting
# History:


import os
import random
import soundfile
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import matplotlib.pyplot as plt

import feature_extractor as fe


def read_table(sPath):
    """ Read parquet data, and return a numpy array
    """
    pdData = pq.read_table(sPath).to_pandas()
    return pdData.values.T


def save_subset(sPath, sSub1, sSub2):
    """Read parquet data, and save subset
    """
    pdDataOri = pq.read_table(sPath).to_pandas()
    print(pdDataOri.shape)
    iLen = pdDataOri.shape[1]
    pdData = pdDataOri.iloc[:, list(range(int(iLen/2)))]
    print(pdData.shape)
    pqData = pa.Table.from_pandas(pdData)
    pq.write_table(pqData, sSub1)

    pdData = pdDataOri.iloc[:, list(range(int(iLen/2), iLen))]
    print(pdData.shape)
    pqData = pa.Table.from_pandas(pdData)
    pq.write_table(pqData, sSub2)


def save_sub_wav(npData):
    """Save to subtract to wav file
    """
    lNp = []
    for iIndex in range(3):
        npY = npData[iIndex]
        npY = npY.astype(np.int16)*256
        lNp.append(npY)
        print(npY)

    for iIndex in range(len(lNp)):
        iStop = iIndex + 1
        if iStop == len(lNp):
            iStop = 0
        npSub = lNp[iIndex] - lNp[iStop]
        print(npSub)
        soundfile.write(
            os.path.join(
                "../data/sound", str(iIndex) + "_sub.wav"), npSub, 16000)


def save_wav(npData):
    """Save to wav file
    """
    for iIndex in range(3):
        npY = npData[iIndex]
        npY = npY.astype(np.int16)*256
        print(npY)
        soundfile.write(
            os.path.join("../data/sound", str(iIndex) + ".wav"), npY, 16000)


def read_tag(sPath):
    """Read tag list, return a numpy array
    """
    lLines = open(sPath).read().splitlines()
    lLines = lLines[1:]
    npY = np.zeros((len(lLines),))
    for iIndex in range(len(lLines)):
        lWords = lLines[iIndex].split(",")
        npY[iIndex] = int(lWords[3])
    # print(npY[0:10])
    return npY


def save_mel_npy(sPath, sSavePath):
    """Convert orignal signal to mel frequency
    """
    npData = np.load(sPath)
    npMelData = np.zeros((npData.shape[0], 64, 1251))
    for iIndex in range(npData.shape[0]):
        npTmp = npData[iIndex]
        npTmp = npTmp.astype(np.int16)
        # TODO:Test differen npTmp*100 npTmp*256
        npMel = fe.get_mel(
            npTmp, win_length_seconds=0.08, hop_length_seconds=0.04, n_mels=64)
        npMelData[iIndex] = npMel
    np.save(sSavePath, npMelData)
    return npMelData


def prepare_data(sData, sTag):
    """Read mel numpy array and tag, return as numpy array
    """
    npX = np.load(sData)
    npY = read_tag(sTag)
    npY = npY[0:npX.shape[0]]
    return npX, npY


def split_pos_neg(npX, npY):
    """Split data into pos and neg
    """
    iPos = int(np.sum(npY))
    npNeg = np.zeros((npX.shape[0] - iPos, npX.shape[1]), dtype="int8")
    npPos = np.zeros((iPos, npX.shape[1]), dtype="int8")

    iIndexPos = 0
    iIndexNeg = 0
    for iIndex in range(npX.shape[0]):
        if npY[iIndex] == 0:
            npNeg[iIndexNeg] = npX[iIndex]
            iIndexNeg += 1
        elif npY[iIndex] == 1:
            npPos[iIndexPos] = npX[iIndex]
            iIndexPos += 1
    return npPos, npNeg


def enhance_data(npOri, iTimes):
    """Enhance data by roll
    """
    # npOri = np.load(sPath)
    npEnh = np.zeros((npOri.shape[0]*iTimes, npOri.shape[1]), dtype="int8")

    for iIndex in range(npOri.shape[0]):
        for jIndex in range(iTimes):
            npTmp = np.roll(npOri[iIndex], random.randint(0, npOri.shape[1]))
            npEnh[iIndex * iTimes + jIndex] = npTmp
    return npEnh


def prepare_enh_data(sPos, sNeg):
    """Read pos_enh and neg data
    """
    npPos = np.load(sPos)
    npNeg = np.load(sNeg)

    npX = np.concatenate((npPos, npNeg), axis=0)
    npY = np.zeros((npPos.shape[0] + npNeg.shape[0]))
    npY[0:npPos.shape[0]] = 1
    return npX, npY


def add_name(sLine, sExt, iNum):
    """Add num to a sLine
        "../data/pos.npy" -> "../data/pos_1.npy"
    """
    iPost = sLine.find(sExt)
    sFir = sLine[0: iPost]
    sSec = sLine[iPost:]
    return sFir + "_" + str(iNum) + sSec


def enhance_extract():
    """Read parquet data, translate to npy, enhance, and extract mel feature
    """
    sTagPath = "../list/metadata_train.csv"
    sTrainPq = "../data/train.parquet"
    sTrainNp = "../data/train.npy"
    sPos = "../data/k_folder/pos.npy"
    sNeg = "../data/k_folder/neg.npy"
    sNegEnh = "../data/k_folder/neg_enh.npy"
    sNegEnhMel = "../data/k_folder/mel_neg_enh.npy"
    sPosEnhMel = "../data/k_folder/mel_pos_enh.npy"
    sPosEnh = "../data/k_folder/pos_enh.npy"
    sPosEnhMel = "../data/k_folder/mel_pos_enh.npy"
    sNegMel = "../data/k_folder/mel_neg.npy"
    sValMel = "../data/k_folder/mel_val.npy"

    sKfolder = "../data/k_folder"
    sTrainX = "../data/k_folder/trainX.npy"
    sTrainY = "../data/k_folder/trainY.npy"
    sValX = "../data/k_folder/valX.npy"
    sValY = "../data/k_folder/valY.npy"

    sTestPq = "../data/test.parquet"
    sTestNp = "../data/test.npy"

    # Train data
    # npTrain = read_table(sTrainPq)
    # np.save(sTrainNp, npTrain)
    npX, npY = prepare_data(sTrainNp, sTagPath)
    iLen = npX.shape[0]
    iSpl = int(iLen/10)
    for iIndex in range(10):
        print("[Log] Start")
        """
        npXTrain = np.concatenate(
            (npX[0:iSpl*iIndex], npX[iSpl*(iIndex+1)-iLen:]))
        npXVal = npX[iSpl*iIndex:iSpl*(iIndex+1)]
        npYTrain = np.concatenate(
            (npY[0:iSpl*iIndex], npY[iSpl*(iIndex+1)-iLen:]))
        npYVal = npY[iSpl*iIndex:iSpl*(iIndex+1)]
        np.save(add_name(sTrainX, ".npy", iIndex), npXTrain)
        np.save(add_name(sValX, ".npy", iIndex), npXVal)
        np.save(add_name(sTrainY, ".npy", iIndex), npYTrain)
        np.save(add_name(sValY, ".npy", iIndex), npYVal)

        npPos, npNeg = split_pos_neg(npXTrain, npYTrain)
        np.save(add_name(sPos, ".npy", iIndex), npPos)
        np.save(add_name(sNeg, ".npy", iIndex), npNeg)

        npPosEnh = enhance_data(npPos, 10)
        np.save(add_name(sPosEnh, ".npy", iIndex), npPosEnh)
        save_mel_npy(
            add_name(
            sPosEnh, ".npy", iIndex), add_name(sPosEnhMel, ".npy", iIndex))
        save_mel_npy(
            add_name(sNeg, ".npy", iIndex), add_name(sNegMel, ".npy", iIndex))
        save_mel_npy(
            add_name(sValX, ".npy", iIndex), add_name(sValMel, ".npy", iIndex))
        """
        npNeg = np.load(add_name(sNeg, ".npy", iIndex))
        print("[Log] Load %s" % add_name(sNeg, ".npy", iIndex))
        npNegEnh = enhance_data(npNeg, 2)
        print("[Log] Enhanced :%s" % str(npNegEnh.shape))
        np.save(add_name(sNegEnh, ".npy", iIndex), npNegEnh)
        print("[Log] Saving to %s" % add_name(sNegEnh, ".npy", iIndex))
        save_mel_npy(
            add_name(
            sNegEnh, ".npy", iIndex), add_name(sNegEnhMel, ".npy", iIndex))
        print("[Log] %d folder finished!" % iIndex)


def main():
    sOriPath = "../data/test.parquet"
    sS1 = "../data/test1.parquet"
    sS2 = "../data/test2.parquet"
    # save_subset(sOriPath, sS1, sS2)

    # npData = np.load("../data/train.npy")
    # save_sub_wav(npData)
    # save_wav(npData)
    enhance_extract()
    # prepare_enh_data()


if __name__ == "__main__":
    main()
