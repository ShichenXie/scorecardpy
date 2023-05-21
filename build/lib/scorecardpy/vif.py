import pandas as pd
import statsmodels.api as sm
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
from .condition_fun import *

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
  
def vif(dt, y, x = None, merge_coef = False, positive = "bad|1"):
    '''
    Variance Inflation Factors
    ------
    vif calculates variance-inflation factors for logistic regression.
    
    Params
    ------
    dt: A data frame with both x (predictor/feature) and y (response/label) variables.
    y: Name of y variable.
    x: Name of x variables. Default is None. If x is None, 
      then all variables except y are counted as x variables.
    merge_coef: Logical, whether to merge with coefficients of model summary matrix. Defaults to FALSE.
    positive: Value of positive class, default "bad|1".
    
    Returns
    ------
    data frame
        A data frame with columns for variable and gvif.
    
    Examples
    ------
    import scorecardpy as sc
    
    # load data
    dat = sc.germancredit()
    
    # Example I
    sc.vif(dat, 
        y = 'creditability', 
        x=['age_in_years', 'credit_amount', 'present_residence_since'], 
        merge_coef=True)
    '''
    
    dt = dt.copy(deep=True)
    if isinstance(y, str):
        y = [y]
    if isinstance(x, str) and x is not None:
        x = [x]
    if x is not None: 
        dt = dt[y+x]
    # check y
    dt = check_y(dt, y, positive)
    # x variables
    x = x_variable(dt, y, x)

    # logistic regression
    lrfit = lr(dt, y, x)
    
    # vif
    dty, dtX = dmatrices(' ~ '.join([y[0], '+'.join(x)]), data=dt, return_type="dataframe")
    dfvif = pd.DataFrame({
        'variables': ['const'] + x, 
        'vif': [variance_inflation_factor(dtX.values, i) for i in range(dtX.shape[1])]
    })
    # merge with coef
    if merge_coef:
        dfvif = pd.merge(
            lrfit.summary2().tables[1].reset_index().rename(columns = {'index':'variables'}), 
            dfvif,
            on = 'variables', how='outer'
        )
    return dfvif

