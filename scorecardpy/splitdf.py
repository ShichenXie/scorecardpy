# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import warnings
from .condition_fun import *


#' Split a dataset
#'
#' @param dt A data frame.
#' @param y Name of y variable, default is NULL. The input data will split based on the predictor y, if it is provide.
#' @param ratio A numeric value, default is 0.7. It indicates the ratio of total rows contained in one split, must less than 1.
#' @param seed A random seed, default is 186.
#'
#' @examples
#' # Load German credit data
#' data(germancredit)
#'
#' dt_list = split_df(germancredit, y="creditability")
#' train = dt_list$train
#' test = dt_list$test
#'
#' @import data.table
#' @export
def split_df(dt, y=None, ratio=0.7, seed=186):
    # remove date/time col
    dt = rm_datetime_col(dt)
    # replace "" by NA
    dt = rep_blank_na(dt)
    # set ratio range
    if not isinstance(ratio, (float, int)) or ratio > 1 or ratio <= 0:
        warnings.warn("Incorrect inputs; ratio must be a numeric that length equal to 1 and less than 1. It was set to 0.7.")
        ratio = 0.7
    # split into train and test
    if y is None:
        train = dt.sample(frac=ratio, random_state=seed).sort_index()
        test = dt.iloc[list(set(dt.index.tolist()).difference(set(train.index.tolist())))].sort_index()
    else:
        train = dt.groupby(y)\
          .apply(lambda x: x.sample(frac=ratio, random_state=seed))\
          .reset_index(level=y, drop=True)\
          .sort_index()
        test = dt.iloc[list(set(dt.index.tolist()).difference(set(train.index.tolist())))].sort_index()
    return {'train': train, 'test': test}

