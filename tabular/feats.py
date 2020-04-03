# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import lightgbm as lgb
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import gc
import os
import time
import datetime
import path
from joblib import Parallel, delayed

JOBS = 4


# def cate_map(ss):
#     mps = list(ss.value_counts().index)
#     mps = {mps[i]: i for i in range(len(mps))}
#     ss = ss.map(mps)
#     return ss


# cates
def feat_cates_valuecounts(df, data_name):
    new_cols = []
    for col in df.columns:
        new_col = col + '_valuecounts'

        new_cols.append(new_col)

    if not new_cols:
        return

    def func(series):
        ss = series.value_counts()
        sss = series.map(ss)
        return sss

    res = Parallel(n_jobs=JOBS, require='sharedmem')(delayed(func)(df[col]) for col in df.columns)
    tmp = pd.concat(res, axis=1)
    tmp.columns = new_cols
