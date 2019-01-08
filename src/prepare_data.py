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


def save_subset(sPath, sNewPath):
    """Read parquet data, and save subset
    """
    pdData = pq.read_table(sPath).to_pandas()
    pdData = pdData.iloc[:, list(range(10))]
    pqData = pa.Table.from_pandas(pdData)
    pq.write_table(pqData, sNewPath)


def save_wav(npData):
    """Save to wav file
    """
    for iIndex in range(200, 203):
        npY = npData[iIndex]
        npY = npY.astype(np.int16)*256
        soundfile.write("../data/" + str(iIndex) + ".wav", npY, 16000)


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
    npData = read_table(sPath)
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


def main():
    sTrainPath = "../data/train.parquet"
    sTestPath = "../data/test.parquet"
    sSubsetPath = "../data/subset.parquet"
    sSubTraPath = "../data/train_subset.parquet"
    sTagPath = "../list/metadata_train.csv"
    sSavePath = "../data/mel_test.npy"
    # save_subset(sTrainPath, "../data/train_subset.parquet")
    # npData = read_table(sTrainPath)
    # read_tag(sTagPath)
    save_mel_npy(sTestPath, sSavePath)


if __name__ == "__main__":
    main()
