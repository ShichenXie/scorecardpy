# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import warnings
import re
from pandas.api.types import is_numeric_dtype
import statsmodels.api as sm

def lr(dt, y, x):
    # dty
    dty = dt.loc[:,y] 
    # dtx
    dtx = dt.loc[:,x] 
    dtx = sm.add_constant(dtx)
    # logistic regression
    lrfit = sm.GLM(
        dty.astype(float), 
        dtx.astype(float), 
        family=sm.families.Binomial()
    ).fit()
    return lrfit

def str_to_list(x):
    if x is not None and isinstance(x, str):
        x = [x]
    return x
    
# remove constant columns
def check_const_cols(dat):
    # remove only 1 unique vlaues variable 
    unique1_cols = [i for i in list(dat) if len(dat[i].unique())==1]
    if len(unique1_cols) > 0:
        warnings.warn("There are {} columns have only one unique values, which are removed from input dataset. \n (ColumnNames: {})".format(len(unique1_cols), ', '.join(unique1_cols)))
        dat=dat.drop(unique1_cols, axis=1)
    return dat

# remove date time columns
def check_datetime_cols(dat):
    datetime_cols = dat.apply(pd.to_numeric,errors='ignore').select_dtypes(object).apply(pd.to_datetime,errors='ignore').select_dtypes('datetime64').columns.tolist()
    #datetime_cols = dat_time.dtypes[dat_time.dtypes == 'datetime64[ns]'].index.tolist()
    if len(datetime_cols) > 0:
        warnings.warn("There are {} date/time type columns are removed from input dataset. \n (ColumnNames: {})".format(len(datetime_cols), ', '.join(datetime_cols)))
        dat=dat.drop(datetime_cols, axis=1)
    return dat

# check categorical columns' unique values
def check_cateCols_uniqueValues(dat, var_skip = None):
    # character columns with too many unique values
    char_cols = [i for i in list(dat) if not is_numeric_dtype(dat[i])]
    if var_skip is not None: 
        char_cols = list(set(char_cols) - set(str_to_list(var_skip)))
    char_cols_too_many_unique = [i for i in char_cols if len(dat[i].unique()) >= 50]
    if len(char_cols_too_many_unique) > 0:
        print('>>> There are {} variables have too many unique non-numberic values, which might cause the binning process slow. Please double check the following variables: \n{}'.format(len(char_cols_too_many_unique), ', '.join(char_cols_too_many_unique)))
        print('>>> Continue the binning process?')
        print('1: yes \n2: no')
        cont = int(input("Selection: "))
        while cont not in [1, 2]:
            cont = int(input("Selection: "))
        if cont == 2:
            raise SystemExit(0)
    return None


# replace blank by NA
#' @import data.table
#'
def rep_blank_na(dat): # cant replace blank string in categorical value with nan
    # remove duplicated index
    if dat.index.duplicated().any():
        dat = dat.reset_index(drop = True)
        warnings.warn('There are duplicated index in dataset. The index has been reseted.')

    # replace "" with NaN
    blank_cols = [col for col in list(dat) if
                  dat[col].astype(str).str.findall(r'^\s*$').apply(lambda x: 0 if len(x) == 0 else 1).sum() > 0]
    if len(blank_cols) > 0:
        warnings.warn('There are blank strings in {} columns, which are replaced with NaN. \n (ColumnNames: {})'.format(
            len(blank_cols), ', '.join(blank_cols)))
        #        dat[dat == [' ','']] = np.nan
        #        dat2 = dat.apply(lambda x: x.str.strip()).replace(r'^\s*$', np.nan, regex=True)
        for col in blank_cols:
            dat.loc[dat[col] == "", col] = np.nan

    # replace inf with -999
    cols_num = [col for col in list(dat) if col not in blank_cols]
    if len(cols_num) > 0:
        cols_inf = [col for col in cols_num if
                        any(dat[col] == np.inf) | any(dat[col] == -np.inf)]
        if len(cols_inf) > 0:
            warnings.warn(
                'There are infinite or NaN values in {} columns, which are replaced with -999.\n (ColumnNames: {})'.format(
                    len(cols_inf), ', '.join(cols_inf)))
            for col in cols_inf:
                dat.loc[(dat[col] == np.inf) | (dat[col] == -np.inf), col] = -999
    
    return dat


# check y
#' @import data.table
#'
def check_y(dat, y, positive):
    positive = str(positive)
    # ncol of dt
    if not isinstance(dat, pd.DataFrame):
        raise Exception("Incorrect inputs; dat should be a DataFrame.")
    elif dat.shape[1] <= 1:
        raise Exception("Incorrect inputs; dat should be a DataFrame with at least two columns.")

    # y ------
    y = str_to_list(y)
    # length of y == 1
    if len(y) != 1:
        raise Exception("Incorrect inputs; the length of y should be one")
    
    y = y[0]
    # y not in dat.columns
    if y not in dat.columns: 
        raise Exception("Incorrect inputs; there is no \'{}\' column in dat.".format(y))
    
    # remove na in y
    if dat[y].isnull().any():
        warnings.warn("There are NaNs in \'{}\' column. The rows with NaN in \'{}\' were removed from dat.".format(y,y))
        dat = dat.dropna(subset=[y])
        # dat = dat[pd.notna(dat[y])]
    
    # numeric y to int
    if is_numeric_dtype(dat[y]):
        dat.loc[:,y] = dat[y].apply(lambda x: x if pd.isnull(x) else int(x)) #dat[y].astype(int)
    # length of unique values in y
    unique_y = np.unique(dat[y].values)
    if len(unique_y) == 2:
        # if [v not in [0,1] for v in unique_y] == [True, True]:
        if True in [bool(re.search(positive, str(v))) for v in unique_y]:
            y1 = dat[y]
            y2 = dat[y].apply(lambda x: 1 if str(x) in re.split('\|', positive) else 0)
            if (y1 != y2).any():
                dat.loc[:,y] = y2#dat[y] = y2
                warnings.warn("The positive value in \"{}\" was replaced by 1 and negative value by 0.".format(y))
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
def x_variable(dat, y, x, var_skip=None):
    y = str_to_list(y)
    if var_skip is not None: y = y + str_to_list(var_skip)
    x_all = list(set(dat.columns) - set(y))
    
    if x is None:
        x = x_all
    else:
        x = str_to_list(x)
            
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

