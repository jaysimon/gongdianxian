#/usr/bin/env python
# -*- coding: utf-8 -*- #
# Copyright (C) 2018 ShenZhen Hian Speech S&T Co.,Ltd. All rights reserved.
# FileName : read_data.py
# Author : Hou Wei
# Version : V1.0
# Date: 2018-01-03
# Description: Read parquet data
# History:


import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def read_table(sPath):
    """ Read parquet data, and return a numpy array
    """
    pdData = pq.read_table(sPath).to_pandas()
    pdData.info()
    print("")
    return pdData.values.T


def main():
    sTrainPath = "../data/train.parquet"
    sTestPath = "../data/test.parquet"
    sSubsetPath = "../data/subset.parquet"
    npData = read_table(sSubsetPath)
    iStart = 10000
    print(npData[0,iStart:iStart + 1000])


if __name__ == "__main__":
    main()
