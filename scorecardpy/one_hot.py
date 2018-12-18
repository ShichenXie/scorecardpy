import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from .condition_fun import *

def one_hot(dt, cols_skip = None, cols_encode = None, nacol_rm = False, 
            replace_na = -1, category_to_integer = False):
    '''
    One Hot Encoding
    ------
    One-hot encoding on categorical variables. It is not needed when creating 
    a standard scorecard model, but required in models that without doing woe 
    transformation.
    
    Params
    ------
    dt: A data frame.
    cols_skip: Name of categorical variables that will skip and without doing 
        one-hot encoding. Default is None.
    cols_encode: Name of categorical variables to be one-hot encoded, default 
        is None. If it is None, then all categorical variables except in 
        cols_skip are counted.
    nacol_rm: Logical. One-hot encoding on categorical variable contains missing 
        values, whether to remove the column generated to indicate the presence 
        of NAs. Default is False.
    replace_na: Replace missing values with a specified value such as -1 by 
        default, or the mean/median value of the variable in which they occur. 
        If it is None, then no missing values will be replaced.
    factor_to_integer: Logical. Converting categorical variables to integer. 
        Default is False.
    
    Returns
    ------
    A one-hot encoded data frame.
    
    Examples
    ------
    import scorecardpy as sc
    import pandas as pd
    
    # load data
    dat1 = sc.germancredit()
    dat2 = pd.DataFrame({'creditability':['good','bad']}).sample(50, replace=True)
    dat = pd.concat([dat1, dat2], ignore_index=True)
    
    dt_oh0 = sc.one_hot(dat, cols_skip = 'creditability', nacol_rm = False) # default
    dt_oh1 = sc.one_hot(dat, cols_skip = 'creditability', nacol_rm = True)
    
    dt_oh2 = sc.one_hot(dat, cols_skip = 'creditability', replace_na = -1) # default
    dt_oh3 = sc.one_hot(dat, cols_skip = 'creditability', replace_na = 'median')
    dt_oh4 = sc.one_hot(dat, cols_skip = 'creditability', replace_na = None)
    '''
    
    # if it is str, converting to list
    cols_skip, cols_encode = str_to_list(cols_skip), str_to_list(cols_encode)
    # category columns into integer
    if category_to_integer:
        cols_cate = dt.dtypes[dt.dtypes == 'category'].index.tolist()
        if cols_skip is not None:
            cols_cate = list(set(cols_cate) - set(cols_skip))
        dt[cols_cate] = dt[cols_cate].apply(lambda x: pd.factorize(x, sort=True)[0])
    # columns encoding
    if cols_encode is None:
        cols_encode = char_cols = [i for i in list(dt) if not is_numeric_dtype(dt[i]) 
            and dt[i].dtypes != 'datetime64[ns]']
    else:
        cols_encode = x_variable(dt, y=cols_skip, x=cols_encode)
    # columns skip
    if cols_skip is not None:
        cols_encode = list(set(cols_encode) - set(cols_skip))
    # one hot encoding
    if cols_encode is None or len(cols_encode) == 0:
        dt_new = dt
    else:
        temp_dt = pd.get_dummies(dt[cols_encode], dummy_na = not nacol_rm)
        # remove cols that unique len == 1 and has _nan
        rm_cols_nan1 = [i for i in list(temp_dt) if len(temp_dt[i].unique())==1 and '_nan' in i]
        dt_new = pd.concat([dt.drop(cols_encode, axis=1), temp_dt.drop(rm_cols_nan1, axis=1)], axis=1)
    # replace missing values with fillna
    def rep_na(x, repalce_na):
        if x.isna().values.any():
            # dtype is numeric
            xisnum = is_numeric_dtype(x)
            if isinstance(repalce_na, (int, float)):
                fill_na = repalce_na
            elif replace_na in ['mean', 'median'] and xisnum:
                fill_na = getattr(np, 'mean')(x)
            else:
                fill_na = -1
            # set fill_na as str if x is not num
            if not xisnum:
                fill_na = str(fill_na)
            x = x.fillna(fill_na)
        return x
    if replace_na is not None:
        names_fillna = list(dt_new) 
        if cols_skip is not None: names_fillna = list(set(names_fillna)-set(cols_skip))
        dt_new[names_fillna] = dt_new[names_fillna].apply(lambda x: rep_na(x, replace_na))
    # return
    return dt_new
  
