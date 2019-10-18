# -*- coding: utf-8 -*-
"""
连续变量分箱(单调性检验)
"""
import warnings
import copy
from .woebin import woebin
from .monotonous.merge import monotonous_bin
from .condition_fun import (check_y,check_monotonic_variables,check_dat,check_breaks_list)

def woebin_num(dt, y, x=None, breaks_list=None, special_values=None, monotonic_variables = None,
               min_perc_fine_bin=0.02, min_perc_coarse_bin=0.05,stop_limit=0.1, max_num_bin=8,
               positive="bad|1", no_cores=None, print_step=0, method="tree"):

    """
    WOE Binning for number features, support badrate monotonic test
    ------
    `woebin_num` only optimal binning for numerical variables, support variables badrate monotonic test.
    using methods including tree-like segmentation or chi-square merge.
    `woebin_num` can also customizing breakpoints if the breaks_list or special_values was provided.

    Params
    ------
    dt: A data frame with both x (predictor/feature) and y (response/label) variables.
    y: Name of y variable.
    x: Name of x variables. Default is NULL. If x is NULL,
      then all variables except y are counted as x variables.
    breaks_list: List of break points, default is NULL.
      If it is not NULL, variable binning will based on the
      provided breaks.
    special_values: the values specified in special_values
      will be in separate bins. Default is NULL.
    monotonic_variables:list of monotonic testing variables.
    min_perc_fine_bin: The minimum percentage of initial binning
      class number over total. Accepted range: 0.01-0.2; default
      is 0.02, which means initial binning into 50 fine bins for
      continuous variables.
    min_perc_coarse_bin: The minimum percentage of final binning
      class number over total. Accepted range: 0.01-0.2; default
      is 0.05.
    stop_limit: Stop binning segmentation when information value
      gain ratio less than the stop_limit, or stop binning merge
      when the minimum of chi-square less than 'qchisq(1-stoplimit, 1)'.
      Accepted range: 0-0.5; default is 0.1.
    max_num_bin: Integer. The maximum number of binning.
    positive: Value of positive class, default "bad|1".
    no_cores: Number of CPU cores for parallel computation.
      Defaults NULL. If no_cores is NULL, the no_cores will
      set as 1 if length of x variables less than 10, and will
      set as the number of all CPU cores if the length of x variables
      greater than or equal to 10.
    print_step: A non-negative integer. Default is 1. If print_step>0,
      print variable names by each print_step-th iteration.
      If print_step=0 or no_cores>1, no message is print.
    method: Optimal binning method, it should be "tree" or "chimerge".
      Default is "tree".

    Returns
    ------
    dictionary
        Optimal or customized binning dataframe.
    """
    dt = copy.deepcopy(dt)
    dt = check_dat(dt)
    dt = check_y(dat=dt,y=y,positive=positive)
    breaks_list = check_breaks_list(breaks_list)
    monotonic_variables = check_monotonic_variables(dat=dt,y=y,monotonic_variables=monotonic_variables)
    if not breaks_list:
        breaks_list = dict()

    if monotonic_variables:
        for col in monotonic_variables:
            # print("There check {} monotonic testing..............".format(col))
            try:
                cutOffPoints = woebin(dt=dt[[col,y]],y=y,breaks_list=breaks_list,special_values=special_values,min_perc_fine_bin=min_perc_fine_bin,
                                      min_perc_coarse_bin=min_perc_coarse_bin,stop_limit=stop_limit,max_num_bin=max_num_bin,positive=positive,
                                      no_cores=no_cores,print_step=print_step,method=method)[col]["breaks"].tolist()
                cutOffPoints = [float(i) for i in cutOffPoints if str(i) not in ['inf', '-inf']]
                # 单调检验合并方案结果
                mono_cutOffPoints = monotonous_bin(df=dt[[col,y]],col=col,cutOffPoints=cutOffPoints,target=y,special_values=special_values)
                breaks_list.update(mono_cutOffPoints)
            except:
                warnings.warn("The {} have {} unique values, Fail monotonic testing".format(col,len(dt[col].unique())))

    bins = woebin(dt=dt,y=y,x=x,breaks_list=breaks_list,special_values=special_values,min_perc_fine_bin=min_perc_fine_bin,
                 min_perc_coarse_bin=min_perc_coarse_bin,stop_limit=stop_limit,max_num_bin=max_num_bin,positive=positive,
                 no_cores=no_cores,print_step=print_step,method=method)
    return bins