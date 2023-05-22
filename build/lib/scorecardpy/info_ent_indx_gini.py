# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from .condition_fun import *

# information entropy ------
# ent(D) = -\sum_k(p_k*log_2(p_k)), if p_k=0 then p_k*log_2(p_k)=0

# information gain (ID3)
# gain = ent(D) - \sum(abs(\frac{D^v}{D})*ent(D^v))

# gain ratio (C4.5)
# gain_ratio(D,a) = Gain(D,a)/IV(a)
# instrinsic value: IV(a) = -\sum_v( abs(\frac{D^v}{D})*log_2(abs(\frac{D^v}{D})) )


# #' Information Entropy
# #'
# #' This function calculates information entropy (ie) for # multiple x variables.
# #'
# #' @param dt A data frame with both x (predictor/feature) and y # (response/label) variables.
# #' @param y Name of y variable.
# #' @param x Name of x variables. Default is NULL. If x is NULL, # then all variables except y are counted as x variables.
# #' @param order Logical, default is TRUE. If it is TRUE, the # output will descending order via ie.
# #'
# #' @return Information Entropy
# # #' @details
# #'
# #' @examples
# #' # Load German credit data
# #' data(germancredit)
# #'
# #' # Information Entropy
# #' dt_info_ent = ie(germancredit, y = "creditability")
# #'
# #' @import data.table
# #' @export

# Information Entropy
def ie(dt, y='creditability', x=None, order=True):
    # remove date/time col
    dt = rmcol_datetime_unique1(dat)
    # replace "" by NA
    dt = rep_blank_na(dt)
    # check y
    # dt = check_y(dt, y, positive)
    # x variable names
    x = x_variable(dt, y, x)
    # info_ent
    ielist = pd.DataFrame({
        'variable': x,
        'info_ent': [ie_xy(dt[i], dt[y]) for i in x]
    }, columns=['variable', 'info_ent'])
    # sorting
    if order:
        ielist = ielist.sort_values(by='info_ent', ascending=False)
    # return
    return ielist

# #' @import data.table
def ie_xy(x, y):
    # if x is None: x=0
    # xy_N
    df_xy = pd.DataFrame({'x':x,'y':y}).groupby(['x','y']).size().reset_index(name='xy_N')
    # x_N
    df_xy['x_N'] = df_xy.groupby('x')['xy_N'].transform(np.sum)
    # p
    df_xy['p'] = df_xy.xy_N/df_xy.x_N
    df_xy['enti'] = df_xy.p.apply(lambda x: 0 if x==0 else x*np.log2(x))
    # ent
    df_enti = df_xy.groupby('x')\
      .agg({'xy_N':'sum', 'enti': lambda x:-sum(x)})\
      .rename(columns={'xy_N':'x_N','enti':'ent'}).replace(np.nan, 0)
    df_enti['xN_distr'] = df_enti.apply({'x_N':lambda x: x/sum(x)})
    # return
    return sum(df_enti.ent*df_enti.xN_distr)
# x = ['A','B','B','A','C',"A","B","B","B","A","C","C","A","C","B","C","A"]
# y = np.repeat(np.array([1,0]), [8,9])
# ie_xy(x,y)


# #' Information Entropy
# #'
# #' calculating ie of total based on good and bad vectors
# #'
# #' @param good vector of good numbers
# #' @param bad vector of bad numbers
# #'
# #' @examples
# #' # ie_01(good, bad)
# #' dtm = melt(dt, id = 'creditability')[, .(
# #' good = sum(creditability=="good"), bad = sum(creditability=="bad")
# #' ), keyby = c("variable", "value")]
# #'
# #' dtm[, .(ie = lapply(.SD, ie_01, bad)), by="variable", .SDcols# ="good"]
# #'
# #' @import data.table
#' @import data.table
#'
def ie_01(good, bad):
    # enti function
    enti = lambda x: 0 if x==0 else x*np.log2(x)
    # df_enti
    df_enti=pd.DataFrame({'good':good,'bad':bad})\
    .assign(
        p0 = lambda x: x.good/(x.good+x.bad),
        p1 = lambda x: x.bad/(x.good+x.bad),
        count = lambda x: x.good+x.bad
    ) \
    .assign(
        enti = lambda x: -(x.p0.apply(enti)+x.p1.apply(enti))
    )
    # xN_distr
    df_enti['xN_distr'] = df_enti.apply({'count':lambda x: x/sum(x)})
    # return
    return sum(df_enti.enti*df_enti.xN_distr)


# gini impurity (CART) ------
# gini(D) = 1-\sum_k(p_k^2)
# gini_impurity(D) = \sum_v(abs(\frac{D^v}{D})*gini(D^v))

# #' Impurity Gini
# #'
# #' This function calculates gini impurity (used by the CART # Decision Tree) for multiple x variables.
# #'
# #' @param dt A data frame with both x (predictor/feature) and y (response/label) variables.
# #' @param y Name of y variable.
# #' @param x Name of x variables. Default is NULL. If x is NULL, then all variables except y are counted as x variables.
# #' @param order Logical, default is TRUE. If it is TRUE, the output will descending order via gini
# #'
# #' @return gini impurity
# # #' @details
# #'
# #' @examples
# #' # Load German credit data
# #' data(germancredit)
# #'
# #' # gini impurity
# #' dt_gini = ig(germancredit, y = "creditability")
# #'
# #' @import data.table
# #' @export
# #'
# impurity gini
def ig(dt, y, x=None, order=True):
    # remove date/time col
    dt = rmcol_datetime_unique1(dat)
    # replace "" by NA
    dt = rep_blank_na(dt)
    # check y
    # dt = check_y(dt, y, positive)
    # x variable names
    x = x_variable(dt, y, x)
    # info_ent
    iglist = pd.DataFrame({
        'variable': x,
        'gini_impurity': [ig_xy(dt[i], dt[y]) for i in x]
    }, columns=['variable', 'gini_impurity'])
    # sorting
    if order:
        iglist = iglist.sort_values(by='gini_impurity', ascending=False)
    # return
    return iglist
    
    
#' @import data.table
def ig_xy(x, y):
    # if x is None: x=0
    # xy_N
    df_xy = pd.DataFrame({'x':x,'y':y}).groupby(['x','y']).size().reset_index(name='xy_N')
    # x_N
    df_xy['x_N'] = df_xy.groupby('x')['xy_N'].transform(np.sum)
    # p
    df_xy['p'] = df_xy.xy_N/df_xy.x_N
    df_xy['enti'] = df_xy.p.apply(lambda x: x**2)
    # ent
    df_gini = df_xy.groupby('x')\
      .agg({'xy_N':'sum', 'enti': lambda x:1-sum(x)})\
      .rename(columns={'xy_N':'x_N','enti':'ent'}).replace(np.nan, 0)
    df_gini['xN_distr'] = df_gini.apply({'x_N':lambda x: x/sum(x)})
    # return
    return sum(df_gini.ent*df_gini.xN_distr)


#' @import data.table
def ig_01(good, bad):
    # df_gini
    df_gini=pd.DataFrame({'good':good,'bad':bad})\
    .assign(
        p0 = lambda x: x.good/(x.good+x.bad),
        p1 = lambda x: x.bad/(x.good+x.bad),
        count = lambda x: x.good+x.bad
    ) \
    .assign(
        bin_ig = lambda x: 1-(x.p0**2+x.p1**2)
    )
    # xN_distr
    df_gini['xN_distr'] = df_gini.apply({'count':lambda x: x/sum(x)})
    # return
    return sum(df_gini.bin_ig*df_gini.xN_distr)
    


