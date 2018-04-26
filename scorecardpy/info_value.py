# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from .condition_fun import *


def iv(dt, y, x=None, positive='bad|1', order=True):
    '''
    Information Value
    ------
    This function calculates information value (IV) for multiple x variables.
    
    Params
    ------
    dt: A data frame with both x (predictor/feature) and 
      y (response/label) variables.
    y: Name of y variable.
    x: Name of x variables. Default is NULL. If x is NULL, then 
      all variables except y are counted as x variables.
    positive: Value of positive class, default is "bad|1".
    order: Logical, default is TRUE. If it is TRUE, the output 
      will descending order via iv.
    
    Returns
    ------
    DataFrame
        Information Value
    
    Examples
    ------
    import scorecardpy as sc
    
    # load data
    dat = sc.germancredit()
    
    # information values
    dt_info_value = sc.iv(dat, y = "creditability")
    '''
    
    dt = dt.copy(deep=True)
    # remove date/time col
    dt = rm_datetime_col(dt)
    # replace "" by NA
    dt = rep_blank_na(dt)
    # check y
    dt = check_y(dt, y, positive)
    # x variable names
    x = x_variable(dt, y, x)
    # info_value
    ivlist = pd.DataFrame({
        'variable': x,
        'info_value': [iv_xy(dt[i], dt[y]) for i in x]
    }, columns=['variable', 'info_value'])
    # sorting iv
    if order: 
        ivlist = ivlist.sort_values(by='info_value', ascending=False)
    return ivlist
# ivlist = iv(dat, y='creditability')

#' @import data.table
def iv_xy(x, y):
    # good bad func
    def goodbad(df):
        names = {'good': (df['y']==0).sum(),'bad': (df['y']==1).sum()}
        return pd.Series(names)
    # iv calculation
    iv_total = pd.DataFrame({'x':x,'y':y}) \
      .groupby('x') \
      .apply(goodbad) \
      .replace(0, 0.9) \
      .assign(
        DistrBad = lambda x: x.bad/sum(x.bad),
        DistrGood = lambda x: x.good/sum(x.good)
      ) \
      .assign(iv = lambda x: (x.DistrBad-x.DistrGood)*np.log(x.DistrBad/x.DistrGood)) \
      .iv.sum()
    # return iv
    return iv_total

# print(iv_xy(x,y))


# #' Information Value
# #'
# #' calculating IV of total based on good and bad vectors
# #'
# #' @param good vector of good numbers
# #' @param bad vector of bad numbers
# #'
# #' @examples
# #' # iv_01(good, bad)
# #' dtm = melt(dt, id = 'creditability')[, .(
# #' good = sum(creditability=="good"), bad = sum(creditability=="bad")
# #' ), keyby = c("variable", "value")]
# #'
# #' dtm[, .(iv = lapply(.SD, iv_01, bad)), by="variable", .SDcols# ="good"]
# #'
# #' @import data.table
#' @import data.table
#'
def iv_01(good, bad):
    # iv calculation
    iv_total = pd.DataFrame({'good':good,'bad':bad}) \
      .replace(0, 0.9) \
      .assign(
        DistrBad = lambda x: x.bad/sum(x.bad),
        DistrGood = lambda x: x.good/sum(x.good)
      ) \
      .assign(iv = lambda x: (x.DistrBad-x.DistrGood)*np.log(x.DistrBad/x.DistrGood)) \
      .iv.sum()
    # return iv
    return iv_total


# #' miv_01
# #'
# #' calculating IV of each bin based on good and bad vectors
# #'
# #' @param good vector of good numbers
# #' @param bad vector of bad numbers
# #'
# #' @import data.table
# #'
#' @import data.table
#'
def miv_01(good, bad):
    # iv calculation
    infovalue = pd.DataFrame({'good':good,'bad':bad}) \
      .replace(0, 0.9) \
      .assign(
        DistrBad = lambda x: x.bad/sum(x.bad),
        DistrGood = lambda x: x.good/sum(x.good)
      ) \
      .assign(iv = lambda x: (x.DistrBad-x.DistrGood)*np.log(x.DistrBad/x.DistrGood)) \
      .iv
    # return iv
    return infovalue


# #' woe_01
# #'
# #' calculating WOE of each bin based on good and bad vectors
# #'
# #' @param good vector of good numbers
# #' @param bad vector of bad numbers
# #'
# #' @import data.table
#' @import data.table
#'
def woe_01(good, bad):
    # woe calculation
    woe = pd.DataFrame({'good':good,'bad':bad}) \
      .replace(0, 0.9) \
      .assign(
        DistrBad = lambda x: x.bad/sum(x.bad),
        DistrGood = lambda x: x.good/sum(x.good)
      ) \
      .assign(woe = lambda x: np.log(x.DistrBad/x.DistrGood)) \
      .woe
    # return woe
    return woe
