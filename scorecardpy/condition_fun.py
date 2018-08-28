# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import warnings
import re
from pandas.api.types import is_numeric_dtype

# remove date time columns # rm_datetime_col
# remove columns if len(x.unique()) == 1
def rmcol_datetime_unique1(dat): # add more datatime types later
    # character columns with too many unique values
    char_cols = [i for i in list(dat) if not is_numeric_dtype(dat[i])]
    char_cols_too_many_unique = [i for i in char_cols if len(dat[i].unique()) >= 50]

    if len(char_cols_too_many_unique) > 0:
        print('>>> There are {} variables have too many unique non-numberic values, which might cause the binning process slow. Please double check the following variables: \n{}'.format(len(char_cols_too_many_unique), ', '.join(char_cols_too_many_unique)))
        print('>>> Continue the binning process?')
        print('1: yes \n2: no \n')
        cont = int(input("Selection: "))
        while cont not in [1, 2]:
            cont = int(input("Selection: "))
        if cont == 2:
            raise SystemExit(0)

    # remove only 1 unique vlaues variable 
    unique1_cols = [i for i in list(dat) if len(dat[i].unique())==1]
    if len(unique1_cols) > 0:
        warnings.warn("There are {} columns have only one unique values, which are removed from input dataset. \n (ColumnNames: {})".format(len(unique1_cols), ', '.join(unique1_cols)))
        dat=dat.drop(unique1_cols, axis=1)
    
    # remove date time variable
    datetime_cols = dat.dtypes[dat.dtypes == 'datetime64[ns]'].index.tolist()
    if len(datetime_cols) > 0:
        warnings.warn("There are {} date/time type columns are removed from input dataset. \n (ColumnNames: {})".format(len(datetime_cols), ', '.join(datetime_cols)))
        dat=dat.drop(datetime_cols, axis=1)
    # return dat
    return dat


# replace blank by NA
#' @import data.table
#'
def rep_blank_na(dat): # cant replace blank string in categorical value with nan
    blank_cols = [index for index, x in dat.isin(['', ' ']).sum().iteritems() if x > 0]
    if len(blank_cols) > 0:
        warnings.warn('There are blank strings in {} columns, which are replaced with NaN. \n (ColumnNames: {})'.format(len(blank_cols), ', '.join(blank_cols)))
#        dat[dat == [' ','']] = np.nan
#        dat2 = dat.apply(lambda x: x.str.strip()).replace(r'^\s*$', np.nan, regex=True)
        dat.replace(r'^\s*$', np.nan, regex=True)
    
    return dat


# check y
#' @import data.table
#'
def check_y(dat, y, positive):
    # ncol of dt
    if isinstance(dat, pd.DataFrame) & (dat.shape[1] <= 1): 
        raise Exception("Incorrect inputs; dat should be a DataFrame with at least two columns.")
    
    # y ------
    if isinstance(y, str):
        y = [y]
    # length of y == 1
    if len(y) != 1:
        raise Exception("Incorrect inputs; the length of y should be one")
    
    y = y[0]
    # y not in dat.columns
    if y not in dat.columns: 
        raise Exception("Incorrect inputs; there is no \'{}\' column in dat.".format(y))
    
    # remove na in y
    if pd.isna(dat[y]).any():
        warnings.warn("There are NaNs in \'{}\' column. The rows with NaN in \'{}\' were removed from dat.".format(y,y))
        dat = dat.dropna(subset=[y])
        # dat = dat[pd.notna(dat[y])]
    
    # length of unique values in y
    unique_y = np.unique(dat[y].values)
    if len(unique_y) == 2:
        if [v not in [0,1] for v in unique_y] == [True, True]:
            if True in [bool(re.search(positive, str(v))) for v in unique_y]:
                warnings.warn("The positive value in \"{}\" was replaced by 1 and negative value by 0.".format(y))
                dat[y] = dat[y].apply(lambda x: 1 if str(x) in re.split('\|', positive) else 0)
            else:
                raise Exception("Incorrect inputs; the positive value in \"{}\" is not specified".format(y))
    else:
        raise Exception("Incorrect inputs; the length of unique values in y column \'{}\' != 2.".format(y))
    
    return dat


# check print_step
#' @import data.table
#'
def check_print_step(print_step):
    if not isinstance(print_step, (int, float)) or print_step<0:
        warnings.warn("Incorrect inputs; print_step should be a non-negative integer. It was set to 1.")
        print_step = 1
    return print_step


# x variable
def x_variable(dat, y, x):
    x_all = set(list(dat)).difference(set([y]))
    
    if x is None:
        x = x_all
    else:
        if isinstance(x, str):
            x = [x]
            
        if any([i in list(x_all) for i in x]) is False:
            x = x_all
        else:
            x_notin_xall = set(x).difference(x_all)
            if len(x_notin_xall) > 0:
                warnings.warn("Incorrect inputs; there are {} x variables are not exist in input data, which are removed from x. \n({})".format(len(x_notin_xall), ', '.join(x_notin_xall)))
                x = set(x).intersection(x_all)
            
    return list(x)


# check breaks_list
def check_breaks_list(breaks_list, xs):
    if breaks_list is not None:
        # is string
        if isinstance(breaks_list, str):
            breaks_list = eval(breaks_list)
        # is not dict
        if not isinstance(breaks_list, dict):
            raise Exception("Incorrect inputs; breaks_list should be a dict.")
    return breaks_list


# check special_values
def check_special_values(special_values, xs):
    if special_values is not None:
        # # is string
        # if isinstance(special_values, str):
        #     special_values = eval(special_values)
        if isinstance(special_values, list):
            warnings.warn("The special_values should be a dict. Make sure special values are exactly the same in all variables if special_values is a list.")
            sv_dict = {}
            for i in xs:
                sv_dict[i] = special_values
            special_values = sv_dict
        elif not isinstance(special_values, dict): 
            raise Exception("Incorrect inputs; special_values should be a list or dict.")
    return special_values

