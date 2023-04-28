# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import re
import warnings
import multiprocessing as mp
import matplotlib.pyplot as plt
import time
import os
import platform
from .condition_fun import *
from .info_value import *

# converting vector (breaks & special_values) to dataframe
def split_vec_todf(vec):
    '''
    Create a dataframe based on provided vector. 
    Split the rows that including '%,%' into multiple rows. 
    Replace 'missing' by np.nan.
    
    Params
    ------
    vec: list
    
    Returns
    ------
    pandas.DataFrame
        returns a dataframe with three columns
        {'bin_chr':orginal vec, 'rowid':index of vec, 'value':splited vec}
    '''
    if vec is not None:
        vec = [str(i) for i in vec]
        a = pd.DataFrame({'bin_chr':vec}).assign(rowid=lambda x:x.index)
        b = pd.DataFrame([i.split('%,%') for i in vec], index=vec)\
        .stack().replace('missing', np.nan) \
        .reset_index(name='value')\
        .rename(columns={'level_0':'bin_chr'})[['bin_chr','value']]
        # return
        return pd.merge(a,b,on='bin_chr')


def add_missing_spl_val(dtm, breaks, spl_val):
    '''
    add missing to spl_val if there is nan in dtm.value and 
    missing is not specified in breaks and spl_val
    
    Params
    ------
    dtm: melt dataframe
    breaks: breaks list
    spl_val: speical values list
    
    Returns
    ------
    list
        returns spl_val list
    '''
    if dtm.value.isnull().any():
        if breaks is None:
            if spl_val is None:
                spl_val=['missing']
            elif any([('missing' in str(i)) for i in spl_val]):
                spl_val=spl_val
            else:
                spl_val=['missing']+spl_val
        elif any([('missing' in str(i)) for i in breaks]):
            spl_val=spl_val
        else:
            if spl_val is None:
                spl_val=['missing']
            elif any([('missing' in str(i)) for i in spl_val]):
                spl_val=spl_val
            else:
                spl_val=['missing']+spl_val
    # return
    return spl_val


# count number of good or bad in y
def n0(x): return sum(x==0)
def n1(x): return sum(x==1)
# split dtm into bin_sv and dtm (without speical_values)
def dtm_binning_sv(dtm, breaks, spl_val):
    '''
    Split the orginal dtm (melt dataframe) into 
    binning_sv (binning of special_values) and 
    a new dtm (without special_values).
    
    Params
    ------
    dtm: melt dataframe
    spl_val: speical values list
    
    Returns
    ------
    list
        returns a list with binning_sv and dtm
    '''
    spl_val = add_missing_spl_val(dtm, breaks, spl_val)
    if spl_val is not None:
        # special_values from vector to dataframe
        sv_df = split_vec_todf(spl_val)
        # value 
        if is_numeric_dtype(dtm['value']):
            sv_df['value'] = sv_df['value'].astype(dtm['value'].dtypes)
            # sv_df['bin_chr'] = sv_df['bin_chr'].astype(dtm['value'].dtypes).astype(str)
            sv_df['bin_chr'] = np.where(
              np.isnan(sv_df['value']), sv_df['bin_chr'], 
              sv_df['value'].astype(dtm['value'].dtypes).astype(str))
            # sv_df = sv_df.assign(value = lambda x: x.value.astype(dtm['value'].dtypes))
        # dtm_sv & dtm
        # dtm_sv = pd.merge(dtm.fillna("missing"), sv_df[['value']].fillna("missing"), how='inner', on='value', right_index=True)
        if None in dtm.index.names and len(dtm.index.names)==1: 
            dtm_index='index'
        elif None in dtm.index.names and len(dtm.index.names)>1:
            raise ValueError("multiindex's names contain Nonetype")
        else:
            dtm_index=dtm.index.names #update by ZK
        
        dtm_sv = pd.merge(
          dtm.fillna("missing").reset_index(),
          sv_df[['value']].fillna("missing"), 
          how='inner', on='value'
        ).set_index(dtm_index) #update by ZK

        dtm = dtm[~dtm.index.isin(dtm_sv.index)].reset_index() if len(dtm_sv.index) < len(dtm.index) else None
        # dtm_sv = dtm.query('value in {}'.format(sv_df['value'].tolist()))
        # dtm    = dtm.query('value not in {}'.format(sv_df['value'].tolist()))
        
        if dtm_sv.shape[0] == 0:
            return {'binning_sv':None, 'dtm':dtm}
        # binning_sv
        binning_sv = pd.merge(
          dtm_sv.fillna('missing').groupby(['variable','value'], group_keys=False)['y'].agg([n0, n1])\
          .reset_index().rename(columns={'n0':'good','n1':'bad'}),
          sv_df.fillna('missing'), 
          on='value'
        ).groupby(['variable', 'rowid', 'bin_chr'], group_keys=False).agg({'bad':sum,'good':sum})\
        .reset_index().rename(columns={'bin_chr':'bin'})\
        .drop('rowid', axis=1)
    else:
        binning_sv = None
    # return
    return {'binning_sv':binning_sv, 'dtm':dtm}
    
# check empty bins for unmeric variable
def check_empty_bins(dtm, binning):
    # check empty bins
    bin_list = np.unique(dtm.bin.astype(str)).tolist()
    if 'nan' in bin_list: 
        bin_list.remove('nan')
    binleft = set([re.match(r'\[(.+),(.+)\)', i).group(1) for i in bin_list]).difference(set(['-inf', 'inf']))
    binright = set([re.match(r'\[(.+),(.+)\)', i).group(2) for i in bin_list]).difference(set(['-inf', 'inf']))
    if binleft != binright:
        bstbrks = sorted(list(map(float, ['-inf'] + list(binright) + ['inf'])))
        labels = ['[{},{})'.format(bstbrks[i], bstbrks[i+1]) for i in range(len(bstbrks)-1)]
        dtm.loc[:,'bin'] = pd.cut(dtm['value'], bstbrks, right=False, labels=labels)
        binning = dtm.groupby(['variable','bin'], group_keys=False)['y'].agg([n0, n1])\
          .reset_index().rename(columns={'n0':'good','n1':'bad'})
        # warnings.warn("The break points are modified into '[{}]'. There are empty bins based on the provided break points.".format(','.join(binright)))
        # binning
        # dtm['bin'] = dtm['bin'].astype(str)
    # return
    return binning

# required in woebin2 # return binning if breaks provided
#' @import data.table
def woebin2_breaks(dtm, breaks, spl_val):
    '''
    get binning if breaks is provided
    
    Params
    ------
    dtm: melt dataframe
    breaks: breaks list
    spl_val: speical values list
    
    Returns
    ------
    DataFrame
        returns a binning datafram
    '''
    
    # breaks from vector to dataframe
    bk_df = split_vec_todf(breaks)
    # dtm $ binning_sv
    dtm_binsv_list = dtm_binning_sv(dtm, breaks, spl_val)
    dtm = dtm_binsv_list['dtm']
    binning_sv = dtm_binsv_list['binning_sv']
    if dtm is None: return {'binning_sv':binning_sv, 'binning':None}
    
    # binning
    if is_numeric_dtype(dtm['value']):
        # best breaks
        bstbrks = ['-inf'] + list(set(bk_df.value.tolist()).difference(set([np.nan, '-inf', 'inf', 'Inf', '-Inf']))) + ['inf']
        bstbrks = sorted(list(map(float, bstbrks)))
        # cut
        labels = ['[{},{})'.format(bstbrks[i], bstbrks[i+1]) for i in range(len(bstbrks)-1)]
        dtm.loc[:,'bin'] = pd.cut(dtm['value'], bstbrks, right=False, labels=labels)
        dtm['bin'] = dtm['bin'].astype(str)
        
        binning = dtm.groupby(['variable','bin'], group_keys=False)['y'].agg([n0, n1])\
          .reset_index().rename(columns={'n0':'good','n1':'bad'})
        # check empty bins for unmeric variable
        binning = check_empty_bins(dtm, binning)
        
        # sort bin
        binning = pd.merge(
          binning.assign(value=lambda x: [float(re.search(r"^\[(.*),(.*)\)", i).group(2)) if i != 'nan' else np.nan for i in binning['bin']] ),
          bk_df.assign(value=lambda x: x.value.astype(float)), 
          how='left',on='value'
        ).sort_values(by="rowid").reset_index(drop=True)
        # merge binning and bk_df if nan isin value
        if bk_df['value'].isnull().any():
            binning = binning.assign(bin=lambda x: [i if i != 'nan' else 'missing' for i in x['bin']])\
              .fillna('missing').groupby(['variable','rowid'])\
              .agg({'bin':lambda x: '%,%'.join(x), 'good':sum, 'bad':sum})\
              .reset_index()
    else:
        # merge binning with bk_df
        binning = pd.merge(
          dtm, 
          bk_df.assign(bin=lambda x: x.bin_chr),
          how='left', on='value'
        ).fillna('missing').groupby(['variable', 'rowid', 'bin'], group_keys=False)['y'].agg([n0,n1])\
        .rename(columns={'n0':'good','n1':'bad'})\
        .reset_index().drop('rowid', axis=1)
    # return
    return {'binning_sv':binning_sv, 'binning':binning}
    

# required in woebin2_init_bin # return pretty breakpoints
def pretty(low, high, n):
    '''
    pretty breakpoints, the same as pretty function in R
    
    Params
    ------
    low: minimal value 
    low: maximal value 
    n: number of intervals
    
    Returns
    ------
    numpy.ndarray
        returns a breakpoints array
    '''
    # nicenumber
    def nicenumber(x):
        exp = np.floor(np.log10(abs(x)))
        f   = abs(x) / 10**exp
        if f < 1.5:
            nf = 1.
        elif f < 3.:
            nf = 2.
        elif f < 7.:
            nf = 5.
        else:
            nf = 10.
        return np.sign(x) * nf * 10.**exp
    # pretty breakpoints
    d     = abs(nicenumber((high-low)/(n-1)))
    miny  = np.floor(low  / d) * d
    maxy  = np.ceil (high / d) * d
    return np.arange(miny, maxy+0.5*d, d)
# required in woebin2 # return initial binning
def woebin2_init_bin(dtm, init_count_distr, breaks, spl_val):
    '''
    initial binning
    
    Params
    ------
    dtm: melt dataframe
    init_count_distr: the minimal precentage in the fine binning process
    breaks: breaks
    breaks: breaks list
    spl_val: speical values list
    
    Returns
    ------
    dict
        returns a dict with initial binning and special_value binning
    '''
    
    # dtm $ binning_sv
    dtm_binsv_list = dtm_binning_sv(dtm, breaks, spl_val)
    dtm = dtm_binsv_list['dtm']
    binning_sv = dtm_binsv_list['binning_sv']
    if dtm is None: return {'binning_sv':binning_sv, 'initial_binning':None}
    # binning
    if is_numeric_dtype(dtm['value']): # numeric variable
        xvalue = dtm['value'].astype(float)
        # breaks vector & outlier
        iq = xvalue.quantile([0.01, 0.25, 0.75, 0.99])
        iqr = iq[0.75] - iq[0.25]
        if iqr == 0:
          prob_down = 0.01
          prob_up = 0.99
        else:
          prob_down = 0.25
          prob_up = 0.75
        xvalue_rm_outlier = xvalue[(xvalue >= iq[prob_down]-3*iqr) & (xvalue <= iq[prob_up]+3*iqr)]
        # number of initial binning
        n = np.trunc(1/init_count_distr)
        len_uniq_x = len(np.unique(xvalue_rm_outlier))
        if len_uniq_x < n: n = len_uniq_x
        # initial breaks
        brk = np.unique(xvalue_rm_outlier) if len_uniq_x < 10 else pretty(min(xvalue_rm_outlier), max(xvalue_rm_outlier), n)
        
        brk = list(filter(lambda x: x>np.nanmin(xvalue) and x<=np.nanmax(xvalue), brk))
        brk = [float('-inf')] + sorted(brk) + [float('inf')]
        # initial binning datatable
        # cut
        labels = ['[{},{})'.format(brk[i], brk[i+1]) for i in range(len(brk)-1)]
        dtm.loc[:,'bin'] = pd.cut(dtm['value'], brk, right=False, labels=labels)#.astype(str)
        # init_bin
        init_bin = dtm.groupby('bin', group_keys=False)['y'].agg([n0, n1])\
        .reset_index().rename(columns={'n0':'good','n1':'bad'})
        # check empty bins for unmeric variable
        init_bin = check_empty_bins(dtm, init_bin)
        
        init_bin = init_bin.assign(
          variable = dtm['variable'].values[0],
          brkp = lambda x: [float(re.match('^\[(.*),.+', i).group(1)) for i in x['bin']],
          badprob = lambda x: x['bad']/(x['bad']+x['good'])
        )[['variable', 'bin', 'brkp', 'good', 'bad', 'badprob']]
    else: # other type variable
        # initial binning datatable
        init_bin = dtm.groupby('value', group_keys=False)['y'].agg([n0,n1])\
        .rename(columns={'n0':'good','n1':'bad'})\
        .assign(
          variable = dtm['variable'].values[0],
          badprob = lambda x: x['bad']/(x['bad']+x['good'])
        ).reset_index()
        # order by badprob if is.character
        if dtm.value.dtype.name not in ['category', 'bool']:
            init_bin = init_bin.sort_values(by='badprob').reset_index()
        # add index as brkp column
        init_bin = init_bin.assign(brkp = lambda x: x.index)\
            [['variable', 'value', 'brkp', 'good', 'bad', 'badprob']]\
            .rename(columns={'value':'bin'})
    
    # remove brkp that good == 0 or bad == 0 ------
    while len(init_bin.query('(good==0) or (bad==0)')) > 0:
        # brkp needs to be removed if good==0 or bad==0
        rm_brkp = init_bin.assign(count = lambda x: x['good']+x['bad'])\
        .assign(
          count_lag  = lambda x: x['count'].shift(1).fillna(len(dtm)+1),
          count_lead = lambda x: x['count'].shift(-1).fillna(len(dtm)+1)
        ).assign(merge_tolead = lambda x: x['count_lag'] > x['count_lead'])\
        .query('(good==0) or (bad==0)')\
        .query('count == count.min()').iloc[0,]
        # set brkp to lead's or lag's
        shift_period = -1 if rm_brkp['merge_tolead'] else 1
        init_bin = init_bin.assign(brkp2  = lambda x: x['brkp'].shift(shift_period))\
        .assign(brkp = lambda x:np.where(x['brkp'] == rm_brkp['brkp'], x['brkp2'], x['brkp']))
        # groupby brkp
        init_bin = init_bin.groupby('brkp', group_keys=False).agg({
          'variable':lambda x: np.unique(x)[0],
          'bin': lambda x: '%,%'.join(x),
          'good': sum,
          'bad': sum
        }).assign(badprob = lambda x: x['bad']/(x['good']+x['bad']))\
        .reset_index()
    # format init_bin
    if is_numeric_dtype(dtm['value']):
        init_bin = init_bin\
        .assign(bin = lambda x: [re.sub(r'(?<=,).+%,%.+,', '', i) if ('%,%' in i) else i for i in x['bin']])\
        .assign(brkp = lambda x: [float(re.match('^\[(.*),.+', i).group(1)) for i in x['bin']])
    # return 
    return {'binning_sv':binning_sv, 'initial_binning':init_bin}


# required in woebin2_tree # add 1 best break for tree-like binning
def woebin2_tree_add_1brkp(dtm, initial_binning, count_distr_limit, bestbreaks=None):
    '''
    add a breakpoint into provided bestbreaks
    
    Params
    ------
    dtm
    initial_binning
    count_distr_limit
    bestbreaks
    
    Returns
    ------
    DataFrame
        a binning dataframe with updated breaks
    '''
    # dtm removed values in spl_val
    # total_iv for all best breaks
    def total_iv_all_breaks(initial_binning, bestbreaks, dtm_rows):
        # best breaks set
        breaks_set = set(initial_binning.brkp).difference(set(list(map(float, ['-inf', 'inf']))))
        if bestbreaks is not None: breaks_set = breaks_set.difference(set(bestbreaks))
        breaks_set = sorted(breaks_set)
        # loop on breaks_set
        init_bin_all_breaks = initial_binning.copy(deep=True)
        for i in breaks_set:
            # best break + i
            bestbreaks_i = [float('-inf')]+sorted(bestbreaks+[i] if bestbreaks is not None else [i])+[float('inf')]
            # best break datatable
            labels = ['[{},{})'.format(bestbreaks_i[i], bestbreaks_i[i+1]) for i in range(len(bestbreaks_i)-1)]
            init_bin_all_breaks.loc[:,'bstbin'+str(i)] = pd.cut(init_bin_all_breaks['brkp'], bestbreaks_i, right=False, labels=labels)#.astype(str)
        # best break dt
        total_iv_all_brks = pd.melt(
          init_bin_all_breaks, id_vars=["variable", "good", "bad"], var_name='bstbin', 
          value_vars=['bstbin'+str(i) for i in breaks_set])\
          .groupby(['variable', 'bstbin', 'value'], group_keys=False)\
          .agg({'good':sum, 'bad':sum}).reset_index()\
          .assign(count=lambda x: x['good']+x['bad'])
          
        total_iv_all_brks['count_distr'] = total_iv_all_brks.groupby(['variable', 'bstbin'], group_keys=False)\
          ['count'].apply(lambda x: x/dtm_rows).reset_index(drop=True)
        total_iv_all_brks['min_count_distr'] = total_iv_all_brks.groupby(['variable', 'bstbin'], group_keys=False)\
          ['count_distr'].transform(lambda x: min(x))
          
        total_iv_all_brks = total_iv_all_brks\
          .assign(bstbin = lambda x: [float(re.sub('^bstbin', '', i)) for i in x['bstbin']] )\
          .groupby(['variable','bstbin','min_count_distr'], group_keys=False)\
          .apply(lambda x: iv_01(x['good'], x['bad'])).reset_index(name='total_iv')
        # return 
        return total_iv_all_brks
    # binning add 1best break
    def binning_add_1bst(initial_binning, bestbreaks):
        if bestbreaks is None:
            bestbreaks_inf = [float('-inf'),float('inf')]
        else:
            if not is_numeric_dtype(dtm['value']):
                bestbreaks = [i for i in bestbreaks if int(i) != min(initial_binning.brkp)]
            bestbreaks_inf = [float('-inf')]+sorted(bestbreaks)+[float('inf')]
        
        labels = ['[{},{})'.format(bestbreaks_inf[i], bestbreaks_inf[i+1]) for i in range(len(bestbreaks_inf)-1)]
        binning_1bst_brk = initial_binning.assign(
          bstbin = lambda x: pd.cut(x['brkp'], bestbreaks_inf, right=False, labels=labels)
        )
        if is_numeric_dtype(dtm['value']):
            binning_1bst_brk = binning_1bst_brk.groupby(['variable', 'bstbin'], group_keys=False)\
            .agg({'good':sum, 'bad':sum}).reset_index().assign(bin=lambda x: x['bstbin'])\
            [['bstbin', 'variable', 'bin', 'good', 'bad']]
        else:
            binning_1bst_brk = binning_1bst_brk.groupby(['variable', 'bstbin'], group_keys=False)\
            .agg({'good':sum, 'bad':sum, 'bin':lambda x:'%,%'.join(x)}).reset_index()\
            [['bstbin', 'variable', 'bin', 'good', 'bad']]
        # format
        binning_1bst_brk['total_iv'] = iv_01(binning_1bst_brk.good, binning_1bst_brk.bad)
        binning_1bst_brk['bstbrkp'] = [float(re.match("^\[(.*),.+", i).group(1)) for i in binning_1bst_brk['bstbin']]
        # return
        return binning_1bst_brk
    # dtm_rows
    dtm_rows = len(dtm.index)
    # total_iv for all best breaks
    total_iv_all_brks = total_iv_all_breaks(initial_binning, bestbreaks, dtm_rows)
    # bestbreaks: total_iv == max(total_iv) & min(count_distr) >= count_distr_limit
    bstbrk_maxiv = total_iv_all_brks.loc[lambda x: x['min_count_distr'] >= count_distr_limit]
    if len(bstbrk_maxiv.index) > 0:
        bstbrk_maxiv = bstbrk_maxiv.loc[lambda x: x['total_iv']==max(x['total_iv'])]
        bstbrk_maxiv = bstbrk_maxiv['bstbin'].tolist()[0]
    else:
        bstbrk_maxiv = None
    # bestbreaks
    if bstbrk_maxiv is not None:
        # add 1best break to bestbreaks
        bestbreaks = bestbreaks+[bstbrk_maxiv] if bestbreaks is not None else [bstbrk_maxiv]
    # binning add 1best break
    bin_add_1bst = binning_add_1bst(initial_binning, bestbreaks)
    # return
    return bin_add_1bst
    
    
# required in woebin2 # return tree-like binning
def woebin2_tree(dtm, init_count_distr=0.02, count_distr_limit=0.05, 
                 stop_limit=0.1, bin_num_limit=8, breaks=None, spl_val=None):
    '''
    binning using tree-like method
    
    Params
    ------
    dtm:
    init_count_distr:
    count_distr_limit:
    stop_limit:
    bin_num_limit:
    breaks:
    spl_val:
    
    Returns
    ------
    dict
        returns a dict with initial binning and special_value binning
    '''
    # initial binning
    bin_list = woebin2_init_bin(dtm, init_count_distr=init_count_distr, breaks=breaks, spl_val=spl_val)
    initial_binning = bin_list['initial_binning']
    binning_sv = bin_list['binning_sv']
    if initial_binning is None: # fix next condition npe
        return {'binning_sv':binning_sv, 'binning':None}
    if len(initial_binning.index)==1: 
        return {'binning_sv':binning_sv, 'binning':initial_binning}
    # initialize parameters
    len_brks = len(initial_binning.index)
    bestbreaks = None
    IVt1 = IVt2 = 1e-10
    IVchg = 1 ## IV gain ratio
    step_num = 1
    # best breaks from three to n+1 bins
    binning_tree = None
    while (IVchg >= stop_limit) and (step_num+1 <= min([bin_num_limit, len_brks])):
        binning_tree = woebin2_tree_add_1brkp(dtm, initial_binning, count_distr_limit, bestbreaks)
        # best breaks
        bestbreaks = binning_tree.loc[lambda x: x['bstbrkp'] != float('-inf'), 'bstbrkp'].tolist()
        # information value
        IVt2 = binning_tree['total_iv'].tolist()[0]
        IVchg = IVt2/IVt1-1 ## ratio gain
        IVt1 = IVt2
        # step_num
        step_num = step_num + 1
    if binning_tree is None: binning_tree = initial_binning
    # return 
    return {'binning_sv':binning_sv, 'binning':binning_tree}
    
    
# examples
# import time
# start = time.time()
# # binning_dict = woebin2_init_bin(dtm, init_count_distr=0.02, breaks=None, spl_val=None) 
# # woebin2_tree_add_1brkp(dtm, binning_dict['initial_binning'], count_distr_limit=0.05) 
# # woebin2_tree(dtm, binning_dict['initial_binning'], count_distr_limit=0.05)
# end = time.time()
# print(end - start)





# required in woebin2 # return chimerge binning
#' @importFrom stats qchisq
def woebin2_chimerge(dtm, init_count_distr=0.02, count_distr_limit=0.05, 
                     stop_limit=0.1, bin_num_limit=8, breaks=None, spl_val=None):
    '''
    binning using chimerge method
    
    Params
    ------
    dtm:
    init_count_distr:
    count_distr_limit:
    stop_limit:
    bin_num_limit:
    breaks:
    spl_val:
    
    Returns
    ------
    dict
        returns a dict with initial binning and special_value binning
    '''
    # [chimerge](http://blog.csdn.net/qunxingvip/article/details/50449376)
    # [ChiMerge:Discretization of numeric attributs](http://www.aaai.org/Papers/AAAI/1992/AAAI92-019.pdf)
    # chisq = function(a11, a12, a21, a22) {
    #   A = list(a1 = c(a11, a12), a2 = c(a21, a22))
    #   Adf = do.call(rbind, A)
    #
    #   Edf =
    #     matrix(rowSums(Adf), ncol = 1) %*%
    #     matrix(colSums(Adf), nrow = 1) /
    #     sum(Adf)
    #
    #   sum((Adf-Edf)^2/Edf)
    # }
    # function to create a chisq column in initial_binning
    def add_chisq(initial_binning):
        chisq_df = pd.melt(initial_binning, 
          id_vars=["brkp", "variable", "bin"], value_vars=["good", "bad"],
          var_name='goodbad', value_name='a')\
        .sort_values(by=['goodbad', 'brkp']).reset_index(drop=True)
        ###
        chisq_df['a_lag'] = chisq_df.groupby('goodbad', group_keys=False)['a'].apply(lambda x: x.shift(1))#.reset_index(drop=True)
        chisq_df['a_rowsum'] = chisq_df.groupby('brkp', group_keys=False)['a'].transform(lambda x: sum(x))#.reset_index(drop=True)
        chisq_df['a_lag_rowsum'] = chisq_df.groupby('brkp', group_keys=False)['a_lag'].transform(lambda x: sum(x))#.reset_index(drop=True)
        ###
        chisq_df = pd.merge(
          chisq_df.assign(a_colsum = lambda df: df.a+df.a_lag), 
          chisq_df.groupby('brkp', group_keys=False).apply(lambda df: sum(df.a+df.a_lag)).reset_index(name='a_sum'))\
        .assign(
          e = lambda df: df.a_rowsum*df.a_colsum/df.a_sum,
          e_lag = lambda df: df.a_lag_rowsum*df.a_colsum/df.a_sum
        ).assign(
          ae = lambda df: (df.a-df.e)**2/df.e + (df.a_lag-df.e_lag)**2/df.e_lag
        ).groupby('brkp').apply(lambda x: sum(x.ae)).reset_index(name='chisq')
        # return
        return pd.merge(initial_binning.assign(count = lambda x: x['good']+x['bad']), chisq_df, how='left')
    # initial binning
    bin_list = woebin2_init_bin(dtm, init_count_distr=init_count_distr, breaks=breaks, spl_val=spl_val)
    initial_binning = bin_list['initial_binning']
    binning_sv = bin_list['binning_sv']
    # return initial binning if its row number equals 1
    if len(initial_binning.index)==1: 
        return {'binning_sv':binning_sv, 'binning':initial_binning}

    # dtm_rows
    dtm_rows = len(dtm.index)    
    # chisq limit
    from scipy.special import chdtri
    chisq_limit = chdtri(1, stop_limit)
    # binning with chisq column
    binning_chisq = add_chisq(initial_binning)
    
    # param
    bin_chisq_min = binning_chisq.chisq.min()
    bin_count_distr_min = min(binning_chisq['count']/dtm_rows)
    bin_nrow = len(binning_chisq.index)
    # remove brkp if chisq < chisq_limit
    while bin_chisq_min < chisq_limit or bin_count_distr_min < count_distr_limit or bin_nrow > bin_num_limit:
        # brkp needs to be removed
        if bin_chisq_min < chisq_limit:
            rm_brkp = binning_chisq.assign(merge_tolead = False).sort_values(by=['chisq', 'count']).iloc[0,]
        elif bin_count_distr_min < count_distr_limit:
            rm_brkp = binning_chisq.assign(
              count_distr = lambda x: x['count']/sum(x['count']),
              chisq_lead = lambda x: x['chisq'].shift(-1).fillna(float('inf'))
            ).assign(merge_tolead = lambda x: x['chisq'] > x['chisq_lead'])
            # replace merge_tolead as True
            rm_brkp.loc[np.isnan(rm_brkp['chisq']), 'merge_tolead']=True
            # order select 1st
            rm_brkp = rm_brkp.sort_values(by=['count_distr']).iloc[0,]
        elif bin_nrow > bin_num_limit:
            rm_brkp = binning_chisq.assign(merge_tolead = False).sort_values(by=['chisq', 'count']).iloc[0,]
        else:
            break
        # set brkp to lead's or lag's
        shift_period = -1 if rm_brkp['merge_tolead'] else 1
        binning_chisq = binning_chisq.assign(brkp2  = lambda x: x['brkp'].shift(shift_period))\
        .assign(brkp = lambda x:np.where(x['brkp'] == rm_brkp['brkp'], x['brkp2'], x['brkp']))
        # groupby brkp
        binning_chisq = binning_chisq.groupby('brkp', group_keys=False).agg({
          'variable':lambda x:np.unique(x),
          'bin': lambda x: '%,%'.join(x),
          'good': sum,
          'bad': sum
        }).assign(badprob = lambda x: x['bad']/(x['good']+x['bad']))\
        .reset_index()
        # update
        ## add chisq to new binning dataframe
        binning_chisq = add_chisq(binning_chisq)
        ## param
        bin_nrow = len(binning_chisq.index)
        if bin_nrow == 1:
            break
        bin_chisq_min = binning_chisq.chisq.min()
        bin_count_distr_min = min(binning_chisq['count']/dtm_rows)
        
    # format init_bin # remove (.+\\)%,%\\[.+,)
    if is_numeric_dtype(dtm['value']):
        binning_chisq = binning_chisq\
        .assign(bin = lambda x: [re.sub(r'(?<=,).+%,%.+,', '', i) if ('%,%' in i) else i for i in x['bin']])\
        .assign(brkp = lambda x: [float(re.match('^\[(.*),.+', i).group(1)) for i in x['bin']])
    # return 
    return {'binning_sv':binning_sv, 'binning':binning_chisq}
     
     

# required in woebin2 # # format binning output
def binning_format(binning):
    '''
    format binning dataframe
    
    Params
    ------
    binning: with columns of variable, bin, good, bad
    
    Returns
    ------
    DataFrame
        binning dataframe with columns of 'variable', 'bin', 
        'count', 'count_distr', 'good', 'bad', 'badprob', 'woe', 
        'bin_iv', 'total_iv',  'breaks', 'is_special_values'
    '''
    
    binning['count'] = binning['good'] + binning['bad']
    binning['count_distr'] = binning['count']/sum(binning['count'])
    binning['badprob'] = binning['bad']/binning['count']
    # binning = binning.assign(
    #   count = lambda x: (x['good']+x['bad']),
    #   count_distr = lambda x: (x['good']+x['bad'])/sum(x['good']+x['bad']),
    #   badprob = lambda x: x['bad']/(x['good']+x['bad']))
    # new columns: woe, iv, breaks, is_sv
    binning['woe'] = woe_01(binning['good'],binning['bad'])
    binning['bin_iv'] = miv_01(binning['good'],binning['bad'])
    binning['total_iv'] = binning['bin_iv'].sum()
    # breaks
    binning['breaks'] = binning['bin']
    if any([r'[' in str(i) for i in binning['bin']]):
        def re_extract_all(x): 
            gp23 = re.match(r"^\[(.*), *(.*)\)((%,%missing)*)", x)
            breaks_string = x if gp23 is None else gp23.group(2)+gp23.group(3)
            return breaks_string
        binning['breaks'] = [re_extract_all(i) for i in binning['bin']]
    # is_sv    
    binning['is_special_values'] = binning['is_sv']
    # return
    return binning[['variable', 'bin', 'count', 'count_distr', 'good', 'bad', 'badprob', 'woe', 'bin_iv', 'total_iv',  'breaks', 'is_special_values']]


# woebin2
# This function provides woe binning for only two columns (one x and one y) dataframe.
def woebin2(dtm, breaks=None, spl_val=None, 
            init_count_distr=0.02, count_distr_limit=0.05, 
            stop_limit=0.1, bin_num_limit=8, method="tree"):
    '''
    provides woe binning for only two series
    
    Params
    ------
    
    
    Returns
    ------
    DataFrame
        
    '''
    # binning
    if breaks is not None:
        # 1.return binning if breaks provided
        bin_list = woebin2_breaks(dtm=dtm, breaks=breaks, spl_val=spl_val)
    else:
        if stop_limit == 'N':
            # binning of initial & specialvalues
            bin_list = woebin2_init_bin(dtm, init_count_distr=init_count_distr, breaks=breaks, spl_val=spl_val)
        else:
            if method == 'tree':
                # 2.tree-like optimal binning
                bin_list = woebin2_tree(
                  dtm, init_count_distr=init_count_distr, count_distr_limit=count_distr_limit, 
                  stop_limit=stop_limit, bin_num_limit=bin_num_limit, breaks=breaks, spl_val=spl_val)
            elif method == "chimerge":
                # 2.chimerge optimal binning
                bin_list = woebin2_chimerge(
                  dtm, init_count_distr=init_count_distr, count_distr_limit=count_distr_limit, 
                  stop_limit=stop_limit, bin_num_limit=bin_num_limit, breaks=breaks, spl_val=spl_val)
    # rbind binning_sv and binning
    binning = pd.concat(bin_list, keys=bin_list.keys()).reset_index()\
              .assign(is_sv = lambda x: x.level_0 =='binning_sv')
    # return
    return binning_format(binning)


def bins_to_breaks(bins, dt, to_string=False, save_string=None):
    if isinstance(bins, dict):
        bins = pd.concat(bins, ignore_index=True)

    # x variables
    xs_all = bins['variable'].unique()
    # dtypes of  variables
    vars_class = pd.DataFrame({
      'variable': xs_all,
      'not_numeric': [not is_numeric_dtype(dt[i]) for i in xs_all]
    })
    
    # breakslist of bins
    bins_breakslist = bins[~bins['breaks'].isin(["-inf","inf","missing"]) & ~bins['is_special_values']]
    bins_breakslist = pd.merge(bins_breakslist[['variable', 'breaks']], vars_class, how='left', on='variable')
    bins_breakslist.loc[bins_breakslist['not_numeric'], 'breaks'] = '\''+bins_breakslist.loc[bins_breakslist['not_numeric'], 'breaks']+'\''
    bins_breakslist = bins_breakslist.groupby('variable', group_keys=False)['breaks'].agg(lambda x: ','.join(x))
    
    if to_string:
        bins_breakslist = "breaks_list={\n"+', \n'.join('\''+bins_breakslist.index[i]+'\': ['+bins_breakslist[i]+']' for i in np.arange(len(bins_breakslist)))+"}"
        if save_string is not None:
            brk_lst_name = '{}_{}.py'.format(save_string, time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time())))
            with open(brk_lst_name, 'w') as f:
                f.write(bins_breakslist)
            print('[INFO] The breaks_list is saved as {}'.format(brk_lst_name))
            return 
    return bins_breakslist

    

def woebin(dt, y, x=None, 
           var_skip=None, breaks_list=None, special_values=None, 
           stop_limit=0.1, count_distr_limit=0.05, bin_num_limit=8, 
           # min_perc_fine_bin=0.02, min_perc_coarse_bin=0.05, max_num_bin=8, 
           positive="bad|1", no_cores=None, print_step=0, method="tree",
           ignore_const_cols=True, ignore_datetime_cols=True, 
           check_cate_num=True, replace_blank=True, 
           save_breaks_list=None, **kwargs):
    '''
    WOE Binning
    ------
    `woebin` generates optimal binning for numerical, factor and categorical 
    variables using methods including tree-like segmentation or chi-square 
    merge. woebin can also customizing breakpoints if the breaks_list or 
    special_values was provided.
    
    The default woe is defined as ln(Distr_Bad_i/Distr_Good_i). If you 
    prefer ln(Distr_Good_i/Distr_Bad_i), please set the argument `positive` 
    as negative value, such as '0' or 'good'. If there is a zero frequency 
    class when calculating woe, the zero will replaced by 0.99 to make the 
    woe calculable.
    
    Params
    ------
    dt: A data frame with both x (predictor/feature) and y (response/label) variables.
    y: Name of y variable.
    x: Name of x variables. Default is None. If x is None, 
      then all variables except y are counted as x variables.
    var_skip: Name of variables that will skip for binning. Defaults to None.
    breaks_list: List of break points, default is None. 
      If it is not None, variable binning will based on the 
      provided breaks.
    special_values: the values specified in special_values 
      will be in separate bins. Default is None.
    count_distr_limit: The minimum percentage of final binning 
      class number over total. Accepted range: 0.01-0.2; default 
      is 0.05.
    stop_limit: Stop binning segmentation when information value 
      gain ratio less than the stop_limit, or stop binning merge 
      when the minimum of chi-square less than 'qchisq(1-stoplimit, 1)'. 
      Accepted range: 0-0.5; default is 0.1.
    bin_num_limit: Integer. The maximum number of binning.
    positive: Value of positive class, default "bad|1".
    no_cores: Number of CPU cores for parallel computation. 
      Defaults None. If no_cores is None, the no_cores will 
      set as 1 if length of x variables less than 10, and will 
      set as the number of all CPU cores if the length of x variables 
      greater than or equal to 10.
    print_step: A non-negative integer. Default is 1. If print_step>0, 
      print variable names by each print_step-th iteration. 
      If print_step=0 or no_cores>1, no message is print.
    method: Optimal binning method, it should be "tree" or "chimerge". 
      Default is "tree".
    ignore_const_cols: Logical. Ignore constant columns. Defaults to True.
    ignore_datetime_cols: Logical. Ignore datetime columns. Defaults to True.
    check_cate_num: Logical. Check whether the number of unique values in 
      categorical columns larger than 50. It might make the binning process slow 
      if there are too many unique categories. Defaults to True.
    replace_blank: Logical. Replace blank values with None. Defaults to True.
    save_breaks_list: The file name to save breaks_list. Default is None.
    
    Returns
    ------
    dictionary
        Optimal or customized binning dataframe.
    
    Examples
    ------
    import scorecardpy as sc
    import pandas as pd
    
    # load data
    dat = sc.germancredit()
    
    # Example I
    # binning of two variables in germancredit dataset
    bins_2var = sc.woebin(dat, y = "creditability", 
      x = ["credit_amount", "purpose"])
    
    # Example II
    # binning of the germancredit dataset
    bins_germ = sc.woebin(dat, y = "creditability")
    
    # Example III
    # customizing the breakpoints of binning
    dat2 = pd.DataFrame({'creditability':['good','bad']}).sample(50, replace=True)
    dat_nan = pd.concat([dat, dat2], ignore_index=True)
    
    breaks_list = {
      'age_in_years': [26, 35, 37, "Inf%,%missing"],
      'housing': ["own", "for free%,%rent"]
    }
    special_values = {
      'credit_amount': [2600, 9960, "6850%,%missing"],
      'purpose': ["education", "others%,%missing"]
    }
    
    bins_cus_brk = sc.woebin(dat_nan, y="creditability",
      x=["age_in_years","credit_amount","housing","purpose"],
      breaks_list=breaks_list, special_values=special_values)
    '''
    # start time
    start_time = time.time()
    
    # arguments
    ## print_info
    print_info = kwargs.get('print_info', True)
    ## init_count_distr
    min_perc_fine_bin = kwargs.get('min_perc_fine_bin', None)
    init_count_distr = kwargs.get('init_count_distr', min_perc_fine_bin)
    if init_count_distr is None: init_count_distr = 0.02
    ## count_distr_limit
    min_perc_coarse_bin = kwargs.get('min_perc_coarse_bin', None)
    if min_perc_coarse_bin is not None: count_distr_limit = min_perc_coarse_bin
    ## bin_num_limit
    max_num_bin = kwargs.get('max_num_bin', None)
    if max_num_bin is not None: bin_num_limit = max_num_bin
    
    # print infomation
    if print_info: print('[INFO] creating woe binning ...')
    
    dt = dt.copy(deep=True)
    if isinstance(y, str):
        y = [y]
    if isinstance(x, str) and x is not None:
        x = [x]
    if x is not None: 
        dt = dt[y+x]
    # check y
    dt = check_y(dt, y, positive)
    # remove constant columns
    if ignore_const_cols: dt = check_const_cols(dt)
    # remove date/time col
    if ignore_datetime_cols: dt = check_datetime_cols(dt)
    # check categorical columns' unique values
    if check_cate_num: check_cateCols_uniqueValues(dt, var_skip)
    # replace black with na
    if replace_blank: dt = rep_blank_na(dt)
      
    # x variable names
    xs = x_variable(dt, y, x, var_skip)
    xs_len = len(xs)
    # print_step
    print_step = check_print_step(print_step)
    # breaks_list
    breaks_list = check_breaks_list(breaks_list, xs)
    # special_values
    special_values = check_special_values(special_values, xs)
    ### ### 
    # stop_limit range
    if stop_limit<0 or stop_limit>0.5 or not isinstance(stop_limit, (float, int)):
        warnings.warn("Incorrect parameter specification; accepted stop_limit parameter range is 0-0.5. Parameter was set to default (0.1).")
        stop_limit = 0.1
    # init_count_distr range
    if init_count_distr<0.01 or init_count_distr>0.2 or not isinstance(init_count_distr, (float, int)):
        warnings.warn("Incorrect parameter specification; accepted init_count_distr parameter range is 0.01-0.2. Parameter was set to default (0.02).")
        init_count_distr = 0.02
    # count_distr_limit
    if count_distr_limit<0.01 or count_distr_limit>0.2 or not isinstance(count_distr_limit, (float, int)):
        warnings.warn("Incorrect parameter specification; accepted count_distr_limit parameter range is 0.01-0.2. Parameter was set to default (0.05).")
        count_distr_limit = 0.05
    # bin_num_limit
    if not isinstance(bin_num_limit, (float, int)):
        warnings.warn("Incorrect inputs; bin_num_limit should be numeric variable. Parameter was set to default (8).")
        bin_num_limit = 8
    # method
    if method not in ["tree", "chimerge"]:
        warnings.warn("Incorrect inputs; method should be tree or chimerge. Parameter was set to default (tree).")
        method = "tree"
    ### ### 
    # binning for each x variable
    # loop on xs
    if (no_cores is None) or (no_cores < 1):
        all_cores = mp.cpu_count() - 1
        no_cores = int(np.ceil(xs_len/5 if xs_len/5 < all_cores else all_cores*0.9))
    if platform.system() == 'Windows': 
        no_cores = 1
            
    # ylist to str
    y = y[0]
    # binning for variables
    if no_cores == 1:
        # create empty bins dict
        bins = {}
        for i in np.arange(xs_len):
            x_i = xs[i]
            # print(x_i)
            # print xs
            if print_step>0 and bool((i+1)%print_step): 
                print(('{:'+str(len(str(xs_len)))+'.0f}/{} {}').format(i, xs_len, x_i), flush=True)
            # woebining on one variable
            bins[x_i] = woebin2(
              dtm = pd.DataFrame({'y':dt[y], 'variable':x_i, 'value':dt[x_i]}),
              breaks=breaks_list[x_i] if (breaks_list is not None) and (x_i in breaks_list.keys()) else None,
              spl_val=special_values[x_i] if (special_values is not None) and (x_i in special_values.keys()) else None,
              init_count_distr=init_count_distr,
              count_distr_limit=count_distr_limit,
              stop_limit=stop_limit, 
              bin_num_limit=bin_num_limit,
              method=method
            )
            # try catch:
            # "The variable '{}' caused the error: '{}'".format(x_i, error-info)
    else:
        pool = mp.Pool(processes=no_cores)
        # arguments
        args = zip(
          [pd.DataFrame({'y':dt[y], 'variable':x_i, 'value':dt[x_i]}) for x_i in xs], 
          [breaks_list[i] if (breaks_list is not None) and (i in list(breaks_list.keys())) else None for i in xs],
          [special_values[i] if (special_values is not None) and (i in list(special_values.keys())) else None for i in xs],
          [init_count_distr]*xs_len, [count_distr_limit]*xs_len, 
          [stop_limit]*xs_len, [bin_num_limit]*xs_len, [method]*xs_len
        )
        # bins in dictionary
        bins = dict(zip(xs, pool.starmap(woebin2, args)))
        pool.close()
    
    # runingtime
    runingtime = time.time() - start_time
    if runingtime >= 10 and print_info:
        # print(time.strftime("%H:%M:%S", time.gmtime(runingtime)))
        print('Binning on {} rows and {} columns in {}'.format(dt.shape[0], dt.shape[1], time.strftime("%H:%M:%S", time.gmtime(runingtime))))
    if save_breaks_list is not None:
        bins_to_breaks(bins, dt, to_string=True, save_string=save_breaks_list)
    # return
    return bins



#' @import data.table
def woepoints_ply1(dtx, binx, x_i, woe_points):
    '''
    Transform original values into woe or porints for one variable.
    
    Params
    ------
    
    Returns
    ------
    
    '''
    # woe_points: "woe" "points"
    # binx = bins.loc[lambda x: x.variable == x_i] 
    # https://stackoverflow.com/questions/12680754/split-explode-pandas-dataframe-string-entry-to-separate-rows
    binx = pd.merge(
      binx[['bin']].assign(v1=binx['bin'].str.split('%,%')).explode('v1'),
      binx[['bin', woe_points]],
      how='left', on='bin'
    ).rename(columns={'v1':'V1',woe_points:'V2'})
    
    # dtx
    ## cut numeric variable
    if is_numeric_dtype(dtx[x_i]):
        is_sv = pd.Series(not bool(re.search(r'\[', str(i))) for i in binx.V1)
        binx_sv = binx.loc[is_sv]
        binx_other = binx.loc[~is_sv]
        # create bin column
        breaks_binx_other = np.unique(list(map(float, ['-inf']+[re.match(r'.*\[(.*),.+\).*', str(i)).group(1) for i in binx_other['bin']]+['inf'])))
        labels = ['[{},{})'.format(breaks_binx_other[i], breaks_binx_other[i+1]) for i in range(len(breaks_binx_other)-1)]
        
        dtx = dtx.assign(xi_bin = lambda x: pd.cut(x[x_i], breaks_binx_other, right=False, labels=labels))\
          .assign(xi_bin = lambda x: [i if (i != i) else str(i) for i in x['xi_bin']])
        # dtx.loc[:,'xi_bin'] = pd.cut(dtx[x_i], breaks_binx_other, right=False, labels=labels)
        # dtx.loc[:,'xi_bin'] = np.where(pd.isnull(dtx['xi_bin']), dtx['xi_bin'], dtx['xi_bin'].astype(str))
        #
        # mask = dtx[x_i].isin(binx_sv['V1'])
        try:
            # when binx_sv['V1'] is not int then cast type
            mask = dtx[x_i].isin(binx_sv['V1'].astype('int'))
        except:
            # if can not cast then use as is
            mask = dtx[x_i].isin(binx_sv['V1'])

        dtx.loc[mask,'xi_bin'] = dtx.loc[mask, x_i].astype(str)
        dtx = dtx[['xi_bin']].rename(columns={'xi_bin':x_i})
    ## to charcarter, na to missing
    if not is_string_dtype(dtx[x_i]):
        dtx.loc[:,x_i] = dtx.loc[:,x_i].astype(str).replace('nan', 'missing')
    # dtx.loc[:,x_i] = np.where(pd.isnull(dtx[x_i]), dtx[x_i], dtx[x_i].astype(str))
    dtx = dtx.replace(np.nan, 'missing').assign(rowid = dtx.index).sort_values('rowid')
    # rename binx
    binx.columns = ['bin', x_i, '_'.join([x_i,woe_points])]
    # merge
    dtx_suffix = pd.merge(dtx, binx, how='left', on=x_i).sort_values('rowid')\
      .set_index(dtx.index)[['_'.join([x_i,woe_points])]]
    return dtx_suffix
    
    
def woebin_ply(dt, bins, no_cores=None, print_step=0, replace_blank=True, **kwargs):
    '''
    WOE Transformation
    ------
    `woebin_ply` converts original input data into woe values 
    based on the binning information generated from `woebin`.
    
    Params
    ------
    dt: A data frame.
    bins: Binning information generated from `woebin`.
    no_cores: Number of CPU cores for parallel computation. 
      Defaults None. If no_cores is None, the no_cores will 
      set as 1 if length of x variables less than 10, and will 
      set as the number of all CPU cores if the length of x 
      variables greater than or equal to 10.
    print_step: A non-negative integer. Default is 1. If 
      print_step>0, print variable names by each print_step-th 
      iteration. If print_step=0 or no_cores>1, no message is print.
    replace_blank: Logical. Replace blank values with None. Defaults to True.
    
    Returns
    -------
    DataFrame
        a dataframe of woe values for each variables 
    
    Examples
    -------
    import scorecardpy as sc
    import pandas as pd
    
    # load data
    dat = sc.germancredit()
    
    # Example I
    dt = dat[["creditability", "credit.amount", "purpose"]]
    # binning for dt
    bins = sc.woebin(dt, y = "creditability")
    
    # converting original value to woe
    dt_woe = sc.woebin_ply(dt, bins=bins)
    
    # Example II
    # binning for germancredit dataset
    bins_germancredit = sc.woebin(dat, y="creditability")
    
    # converting the values in germancredit to woe
    ## bins is a dict
    germancredit_woe = sc.woebin_ply(dat, bins=bins_germancredit) 
    ## bins is a dataframe
    germancredit_woe = sc.woebin_ply(dat, bins=pd.concat(bins_germancredit))
    '''
    # start time
    start_time = time.time()
    ## print_info
    print_info = kwargs.get('print_info', True)
    if print_info: print('[INFO] converting into woe values ...')
    
    # remove date/time col
    # dt = rmcol_datetime_unique1(dt)
    # replace "" by NA
    if replace_blank: dt = rep_blank_na(dt)
    # ncol of dt
    # if len(dt.index) <= 1: raise Exception("Incorrect inputs; dt should have at least two columns.")
    # print_step
    print_step = check_print_step(print_step)
    
    # bins # if (is.list(bins)) rbindlist(bins)
    if isinstance(bins, dict):
        bins = pd.concat(bins, ignore_index=True)
    # x variables
    xs_bin = bins['variable'].unique()
    xs_dt = list(dt.columns)
    xs = list(set(xs_bin).intersection(xs_dt))
    # length of x variables
    xs_len = len(xs)
    # initial data set
    dat = dt.loc[:,list(set(xs_dt) - set(xs))]
    
    # loop on xs
    if (no_cores is None) or (no_cores < 1):
        all_cores = mp.cpu_count() - 1
        no_cores = int(np.ceil(xs_len/5 if xs_len/5 < all_cores else all_cores*0.9))
    if platform.system() == 'Windows': 
        no_cores = 1 
            
    # 
    if no_cores == 1:
        for i in np.arange(xs_len):
            x_i = xs[i]
            # print xs
            # print(x_i)
            if print_step>0 and bool((i+1) % print_step): 
                print(('{:'+str(len(str(xs_len)))+'.0f}/{} {}').format(i, xs_len, x_i), flush=True)
            #
            binx = bins[bins['variable'] == x_i].reset_index()
                 # bins.loc[lambda x: x.variable == x_i] 
                 # bins.loc[bins['variable'] == x_i] # 
                 # bins.query('variable == \'{}\''.format(x_i))
            dtx = dt[[x_i]]
            dat = pd.concat([dat, woepoints_ply1(dtx, binx, x_i, woe_points="woe")], axis=1)
    else:
        pool = mp.Pool(processes=no_cores)
        # arguments
        args = zip(
          [dt[[i]] for i in xs], 
          [bins[bins['variable'] == i] for i in xs], 
          [i for i in xs], 
          ["woe"]*xs_len
        )
        # bins in dictionary
        dat_suffix = pool.starmap(woepoints_ply1, args)
        dat = pd.concat([dat]+dat_suffix, axis=1)
        pool.close()
    # runingtime
    runingtime = time.time() - start_time
    if runingtime >= 10 and print_info:
        # print(time.strftime("%H:%M:%S", time.gmtime(runingtime)))
        print('Woe transformating on {} rows and {} columns in {}'.format(dt.shape[0], xs_len, time.strftime("%H:%M:%S", time.gmtime(runingtime))))
    return dat



# required in woebin_plot
#' @import data.table ggplot2
def plot_bin(binx, title, show_iv):
    '''
    plot binning of one variable
    
    Params
    ------
    binx:
    title:
    show_iv:
    
    Returns
    ------
    matplotlib fig object
    '''
    # y_right_max
    y_right_max = np.ceil(binx['badprob'].max()*10)
    if y_right_max % 2 == 1: y_right_max=y_right_max+1
    if y_right_max - binx['badprob'].max()*10 <= 0.3: y_right_max = y_right_max+2
    y_right_max = y_right_max/10
    if y_right_max>1 or y_right_max<=0 or y_right_max is np.nan or y_right_max is None: y_right_max=1
    ## y_left_max
    y_left_max = np.ceil(binx['count_distr'].max()*10)/10
    if y_left_max>1 or y_left_max<=0 or y_left_max is np.nan or y_left_max is None: y_left_max=1
    # title
    title_string = binx.loc[0,'variable']+"  (iv:"+str(round(binx.loc[0,'total_iv'],4))+")" if show_iv else binx.loc[0,'variable']
    title_string = title+'-'+title_string if title is not None else title_string
    # param
    ind = np.arange(len(binx.index))    # the x locations for the groups
    width = 0.35       # the width of the bars: can also be len(x) sequence
    ###### plot ###### 
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    # ax1
    p1 = ax1.bar(ind, binx['good_distr'], width, color=(24/254, 192/254, 196/254))
    p2 = ax1.bar(ind, binx['bad_distr'], width, bottom=binx['good_distr'], color=(246/254, 115/254, 109/254))
    for i in ind:
        ax1.text(i, binx.loc[i,'count_distr']*1.02, str(round(binx.loc[i,'count_distr']*100,1))+'%, '+str(binx.loc[i,'count']), ha='center')
    # ax2
    ax2.plot(ind, binx['badprob'], marker='o', color='blue')
    for i in ind:
        ax2.text(i, binx.loc[i,'badprob']*1.02, str(round(binx.loc[i,'badprob']*100,1))+'%', color='blue', ha='center')
    # settings
    ax1.set_ylabel('Bin count distribution')
    ax2.set_ylabel('Bad probability', color='blue')
    ax1.set_yticks(np.arange(0, y_left_max+0.2, 0.2))
    ax2.set_yticks(np.arange(0, y_right_max+0.2, 0.2))
    ax2.tick_params(axis='y', colors='blue')
    plt.xticks(ind, binx['bin'])
    plt.title(title_string, loc='left')
    plt.legend((p2[0], p1[0]), ('bad', 'good'), loc='upper right')
    # show plot 
    # plt.show()
    return fig


def woebin_plot(bins, x=None, title=None, show_iv=True):
    '''
    WOE Binning Visualization
    ------
    `woebin_plot` create plots of count distribution and bad probability 
    for each bin. The binning informations are generates by `woebin`.
    
    Params
    ------
    bins: A list or data frame. Binning information generated by `woebin`.
    x: Name of x variables. Default is None. If x is None, then all 
      variables except y are counted as x variables.
    title: String added to the plot title. Default is None.
    show_iv: Logical. Default is True, which means show information value 
      in the plot title.
    
    Returns
    ------
    dict
        a dict of matplotlib figure objests
        
    Examples
    ------
    import scorecardpy as sc
    import matplotlib.pyplot as plt
    
    # load data
    dat = sc.germancredit()
    
    # Example I
    dt1 = dat[["creditability", "credit.amount"]]
    
    bins1 = sc.woebin(dt1, y="creditability")
    p1 = sc.woebin_plot(bins1)
    plt.show(p1)
    
    # Example II
    bins = sc.woebin(dat, y="creditability")
    plotlist = sc.woebin_plot(bins)
    
    # # save binning plot
    # for key,i in plotlist.items():
    #     plt.show(i)
    #     plt.savefig(str(key)+'.png')
    '''
    xs = x
    # bins concat 
    if isinstance(bins, dict):
        bins = pd.concat(bins, ignore_index=True)
    # good bad distr
    def gb_distr(binx):
        binx['good_distr'] = binx['good']/sum(binx['count'])
        binx['bad_distr'] = binx['bad']/sum(binx['count'])
        return binx
    bins = bins.groupby('variable', group_keys=False).apply(gb_distr)
    # x variable names
    if xs is None: xs = bins['variable'].unique()
    # plot export
    plotlist = {}
    for i in xs:
        binx = bins[bins['variable'] == i].reset_index(drop=True)
        plotlist[i] = plot_bin(binx, title, show_iv)
    return plotlist 



# print basic information in woebin_adj
def woebin_adj_print_basic_info(i, xs, bins, dt, bins_breakslist):
    '''
    print basic information of woebinnig in adjusting process
    
    Params
    ------
    
    Returns
    ------
    
    '''
    x_i = xs[i-1]
    xs_len = len(xs)
    binx = bins.loc[bins['variable']==x_i]
    print("--------", str(i)+"/"+str(xs_len), x_i, "--------")
    # print(">>> dt["+x_i+"].dtypes: ")
    # print(str(dt[x_i].dtypes), '\n')
    # 
    print(">>> dt["+x_i+"].describe(): ")
    print(dt[x_i].describe(), '\n')
    
    if len(dt[x_i].unique()) < 10 or not is_numeric_dtype(dt[x_i]):
        print(">>> dt["+x_i+"].value_counts(): ")
        print(dt[x_i].value_counts(), '\n')
    else:
        dt[x_i].hist()
        plt.title(x_i)
        plt.show()
        
    ## current breaks
    print(">>> Current breaks:")
    print(bins_breakslist[x_i], '\n')
    ## woebin plotting
    plt.show(woebin_plot(binx)[x_i])
    
    
# plot adjusted binning in woebin_adj
def woebin_adj_break_plot(dt, y, x_i, breaks, stop_limit, sv_i, method):
    '''
    update breaks and provies a binning plot
    
    Params
    ------
    
    Returns
    ------
    
    '''
    if breaks == '':
        breaks = None
    breaks_list = None if breaks is None else {x_i: eval('['+breaks+']')}
    special_values = None if sv_i is None else {x_i: sv_i}
    # binx update
    bins_adj = woebin(dt[[x_i,y]], y, breaks_list=breaks_list, special_values=special_values, stop_limit = stop_limit, method=method)
    
    ## print adjust breaks
    breaks_bin = set(bins_adj[x_i]['breaks']) - set(["-inf","inf","missing"])
    breaks_bin = ', '.join(breaks_bin) if is_numeric_dtype(dt[x_i]) else ', '.join(['\''+ i+'\'' for i in breaks_bin])
    print(">>> Current breaks:")
    print(breaks_bin, '\n')
    # print bin_adj
    plt.show(woebin_plot(bins_adj))
    # return breaks 
    if breaks == '' or breaks is None: breaks = breaks_bin
    return breaks
    
    
def woebin_adj(dt, y, bins, adj_all_var=False, special_values=None, method="tree", save_breaks_list=None, count_distr_limit=0.05):
    '''
    WOE Binning Adjustment
    ------
    `woebin_adj` interactively adjust the binning breaks.
    
    Params
    ------
    dt: A data frame.
    y: Name of y variable.
    bins: A list or data frame. Binning information generated from woebin.
    adj_all_var: Logical, whether to show monotonic woe variables. Default
      is True
    special_values: the values specified in special_values will in separate 
      bins. Default is None.
    method: optimal binning method, it should be "tree" or "chimerge". 
      Default is "tree".
    save_breaks_list: The file name to save breaks_list. Default is None.
    count_distr_limit: The minimum percentage of final binning 
      class number over total. Accepted range: 0.01-0.2; default 
      is 0.05.

    
    Returns
    ------
    dict
        dictionary of breaks
        
    Examples
    ------
    import scorecardpy as sc
    
    # load data
    dat = sc.germancredit()
    
    # Example I
    dt = dat[["creditability", "age.in.years", "credit.amount"]]
    
    bins = sc.woebin(dt, y="creditability")
    breaks_adj = sc.woebin_adj(dt, y="creditability", bins=bins)
    bins_final = sc.woebin(dt, y="creditability", breaks_list=breaks_adj)
    
    # Example II
    binsII = sc.woebin(dat, y="creditability")
    breaks_adjII = sc.woebin_adj(dat, "creditability", binsII)
    bins_finalII = sc.woebin(dat, y="creditability", breaks_list=breaks_adjII)
    '''
    # bins concat 
    if isinstance(bins, dict):
        bins = pd.concat(bins, ignore_index=True)
    # x variables
    xs_all = bins['variable'].unique()
    # adjust all variables
    if not adj_all_var:
        bins2 = bins.loc[~((bins['bin'] == 'missing') & (bins['count_distr'] >= count_distr_limit))].reset_index(drop=True)
        bins2['badprob2'] = bins2.groupby('variable', group_keys=False).apply(lambda x: x['badprob'].shift(1)).reset_index(drop=True)
        bins2 = bins2.dropna(subset=['badprob2']).reset_index(drop=True)
        bins2 = bins2.assign(badprob_trend = lambda x: x.badprob >= x.badprob2)
        xs_adj = bins2.groupby('variable', group_keys=False)['badprob_trend'].nunique()
        xs_adj = xs_adj[xs_adj>1].index
    else:
        xs_adj = xs_all
    # length of adjusting variables
    xs_len = len(xs_adj)
    # special_values
    special_values = check_special_values(special_values, xs_adj)
    
    # breakslist of bins
    bins_breakslist = bins_to_breaks(bins,dt)
    # loop on adjusting variables
    if xs_len == 0:
        warnings.warn('The binning breaks of all variables are perfect according to default settings.')
        breaks_list = "{"+', '.join('\''+bins_breakslist.index[i]+'\': ['+bins_breakslist[i]+']' for i in np.arange(len(bins_breakslist)))+"}"
        return breaks_list
    # else 
    def menu(i, xs_len, x_i):
        print('>>> Adjust breaks for ({}/{}) {}?'.format(i, xs_len, x_i))
        print('1: next \n2: yes \n3: back')
        adj_brk = input("Selection: ")
        while isinstance(adj_brk,str):
            if str(adj_brk).isdigit():
                adj_brk = int(adj_brk)
                if adj_brk not in [0,1,2,3]:
                    warnings.warn('Enter an item from the menu, or 0 to exit.')               
                    adj_brk = input("Selection: ")  
            else: 
                print('Input could not be converted to digit.')
                adj_brk = input("Selection: ") #update by ZK 
        return adj_brk
        
    # init param
    i = 1
    breaks_list = None
    while i <= xs_len:
        breaks = stop_limit = None
        # x_i
        x_i = xs_adj[i-1]
        sv_i = special_values[x_i] if (special_values is not None) and (x_i in special_values.keys()) else None
        # if sv_i is not None:
        #     sv_i = ','.join('\'')
        # basic information of x_i variable ------
        woebin_adj_print_basic_info(i, xs_adj, bins, dt, bins_breakslist)
        # adjusting breaks ------
        adj_brk = menu(i, xs_len, x_i)
        if adj_brk == 0: 
            return 
        while adj_brk == 2:
            # modify breaks adj_brk == 2
            breaks = input(">>> Enter modified breaks: ")
            breaks = re.sub("^[,\.]+|[,\.]+$", "", breaks)
            if breaks == 'N':
                stop_limit = 'N'
                breaks = None
            else:
                stop_limit = 0.1
            try:
                breaks = woebin_adj_break_plot(dt, y, x_i, breaks, stop_limit, sv_i, method=method)
            except:
                pass
            # adj breaks again
            adj_brk = menu(i, xs_len, x_i)
        if adj_brk == 3:
            # go back adj_brk == 3
            i = i-1 if i>1 else i
        else:
            # go next adj_brk == 1
            if breaks is not None and breaks != '': 
                bins_breakslist[x_i] = breaks
            i += 1
    # return 
    breaks_list = "{"+', '.join('\''+bins_breakslist.index[i]+'\': ['+bins_breakslist[i]+']' for i in np.arange(len(bins_breakslist)))+"}"
    if save_breaks_list is not None:
        bins_adj = woebin(dt, y, x=bins_breakslist.index, breaks_list=breaks_list)
        bins_to_breaks(bins_adj, dt, to_string=True, save_string=save_breaks_list)
    return breaks_list
    
