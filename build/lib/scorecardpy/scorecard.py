# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import re
from .condition_fun import *
from .woebin import woepoints_ply1, woebin_ply


# coefficients in scorecard
def ab(points0=600, odds0=1/19, pdo=50):
    # sigmoid function
    # library(ggplot2)
    # ggplot(data.frame(x = c(-5, 5)), aes(x)) + stat_function(fun = function(x) 1/(1+exp(-x)))
  
    # log_odds function
    # ggplot(data.frame(x = c(0, 1)), aes(x)) + stat_function(fun = function(x) log(x/(1-x)))
  
    # logistic function
    # p(y=1) = 1/(1+exp(-z)),
        # z = beta0+beta1*x1+...+betar*xr = beta*x
    ##==> z = log(p/(1-p)),
        # odds = p/(1-p) # bad/good <==>
        # p = odds/1+odds
    ##==> z = log(odds)
    ##==> score = a - b*log(odds)
  
    # two hypothesis
    # points0 = a - b*log(odds0)
    # points0 - PDO = a - b*log(2*odds0)
    # if pdo > 0:
    #     b = pdo/np.log(2)
    # else:
    #     b = -pdo/np.log(2)
    b = pdo/np.log(2)
    a = points0 + b*np.log(odds0) #log(odds0/(1+odds0))
    
    return {'a':a, 'b':b}



def scorecard(bins, model, xcolumns, points0=600, odds0=1/19, pdo=50, basepoints_eq0=False, digits=0):
    '''
    Creating a Scorecard
    ------
    `scorecard` creates a scorecard based on the results from `woebin` 
    and LogisticRegression object from sklearn or statsmodels
    
    Params
    ------
    bins: Binning information generated from `woebin` function.
    model: A LogisticRegression model object.
    points0: Target points, default 600.
    odds0: Target odds, default 1/19. Odds = p/(1-p).
    pdo: Points to Double the Odds, default 50.
    basepoints_eq0: Logical, default is FALSE. If it is TRUE, the 
      basepoints will equally distribute to each variable.
    digits: The number of digits after the decimal point for points 
      calculation. Default 0.
    
    
    Returns
    ------
    DataFrame
        scorecard dataframe
    
    Examples
    ------
    import scorecardpy as sc
    
    # load data
    dat = sc.germancredit()
    
    # filter variable via missing rate, iv, identical value rate
    dt_sel = sc.var_filter(dat, "creditability")
    
    # woe binning ------
    bins = sc.woebin(dt_sel, "creditability")
    dt_woe = sc.woebin_ply(dt_sel, bins)
    
    y = dt_woe.loc[:,'creditability']
    X = dt_woe.loc[:,dt_woe.columns != 'creditability']
    
    # logistic regression ------
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(penalty='l1', C=0.9, solver='saga')
    lr.fit(X, y)
    
    # # predicted proability
    # dt_pred = lr.predict_proba(X)[:,1]
    # # performace
    # # ks & roc plot
    # sc.perf_eva(y, dt_pred)
    
    # scorecard
    # Example I # creat a scorecard
    card = sc.scorecard(bins, lr, X.columns)
    
    # credit score
    # Example I # only total score
    score1 = sc.scorecard_ply(dt_sel, card)
    # Example II # credit score for both total and each variable
    score2 = sc.scorecard_ply(dt_sel, card, only_total_score = False)
    '''
    
    # coefficients
    aabb = ab(points0, odds0, pdo)
    a = aabb['a'] 
    b = aabb['b']
    # odds = pred/(1-pred); score = a - b*log(odds)
    
    # bins # if (is.list(bins)) rbindlist(bins)
    if isinstance(bins, dict):
        bins = pd.concat(bins, ignore_index=True)
  
    # coefficients
    if str(type(model)) == "<class 'sklearn.linear_model._logistic.LogisticRegression'>":
        coef_df = pd.Series(model.coef_[0], index=np.array([re.sub('_woe$', '', i) for i in xcolumns]))\
        .loc[lambda x: x != 0]#.reset_index(drop=True)
        coef_const = model.intercept_[0]
    elif str(type(model)) == "<class 'statsmodels.genmod.generalized_linear_model.GLMResultsWrapper'>":
        coef_df = model.summary2().tables[1].loc[:,'Coef.']
        coef_const = coef_df['const']
        coef_df = coef_df.drop('const')
        coef_df.index = [re.sub('_woe$', '', i) for i in coef_df.index]
      
    # scorecard
    len_x = len(coef_df)
    basepoints = a - b*coef_const
    card = {}
    if basepoints_eq0:
        card['basepoints'] = pd.DataFrame({'variable':"basepoints", 'bin':np.nan, 'points':0}, index=np.arange(1))
        for i in coef_df.index:
            card[i] = bins.loc[bins['variable']==i, ['variable', 'bin', 'woe']]\
              .assign(points = lambda x: round(-b*x['woe']*coef_df[i] + basepoints/len_x), ndigits=digits)\
              [["variable", "bin", "points"]]
    else:
        card['basepoints'] = pd.DataFrame({'variable':"basepoints", 'bin':np.nan, 'points':round(basepoints, ndigits=digits)}, index=np.arange(1))
        for i in coef_df.index:
            card[i] = bins.loc[bins['variable']==i, ['variable', 'bin', 'woe']]\
              .assign(points = lambda x: round(-b*x['woe']*coef_df[i]), ndigits=digits)\
              [["variable", "bin", "points"]]
    return card


def scorecard2(bins, dt, y, x=None, points0=600, odds0=1/19, pdo=50, basepoints_eq0=False, digits=0, 
               return_prob = False, positive='bad|1', **kwargs):
    '''
    Creating a Scorecard
    ------
    `scorecard2` creates a scorecard based on the results from woebin. It has 
    the same function of scorecard, but without model object input.
    
    Params
    ------
    bins: Binning information generated from woebin function.
    dt: A data frame with both x (predictor/feature) and y (response/label) variables.
    y: Name of y variable.
    x: Name of x variables. If it is None, then all variables in bins are used. Defaults to None.
    points0: Target points, default 600.
    odds0: Target odds, default 1/19. Odds = p/(1-p).
    pdo: Points to Double the Odds, default 50.
    basepoints_eq0: Logical, defaults to False. If it is True, the basepoints will equally distribute to each variable.
    digits: The number of digits after the decimal point for points calculation. Default 0.
    return_prob: Logical, defaults to False. If it is True, the predict probability will also return.
    kwargs: Additional parameters.
    
    Returns
    ------
    A scorecard data frames
    
    Examples
    ------
    # load data
    import scorecardpy as sc
    dt = sc.germancredit()
    
    # filter variable via missing rate, iv, identical value rate
    dtvf = sc.var_filter(dt, "creditability")
    
    # split into train and test
    dtlst = sc.split_df(dtvf, y = 'creditability')
    # binning
    bins = sc.woebin(dtlst['train'], "creditability")
    
    # train only
    ## create scorecard
    card1 = sc.scorecard2(bins=bins, dt=dtlst['train'], y='creditability')
    ## scorecard and predicted probability
    cardprob1 = sc.scorecard2(bins=bins, dt=dtlst['train'], y='creditability', return_prob = True)
    print(cardprob1.keys())
    
    # both train and test
    ## create scorecard
    card2 = sc.scorecard2(bins=bins, dt=dtlst, y='creditability')
    ## scorecard and predicted probability
    cardprob2 = sc.scorecard2(bins=bins, dt=dtlst, y='creditability', return_prob = True)
    print(cardprob2.keys())
    print(cardprob2['prob'].keys())
    '''
        
    # data frame to list
    if isinstance(dt, pd.DataFrame):
        dt = {'dat': dt}

    # check y column
    for i in dt.keys():
        dt[i] = check_y(dt[i], y, positive)

    # dt0
    dtfstkey = list(dt.keys())[0]
    dt0 = dt[dtfstkey]

    # bind bins
    if isinstance(bins, dict):
        bins = pd.concat(bins, ignore_index=True)

    # check xs
    x_bins = bins['variable'].unique()
    if x is None: 
        x = x_bins
    x = x_variable(dt0,y,x)
    x_woe = ['_'.join([i, 'woe']) for i in x]

    # dt to woe
    dt_woe = {}
    for i in dt.keys():
        dt_woe[i] = woebin_ply(dt[i], bins=bins, print_info=False)
    dt0_woe = dt_woe[dtfstkey]

    # model
    lrmodel = lr(dt0_woe, y=y, x=x_woe)

    # scorecard
    card = scorecard(bins = bins, model = lrmodel, xcolumns=x_woe, points0 = points0, odds0 = odds0, pdo = pdo, basepoints_eq0 = basepoints_eq0, digits = digits)

    # returns
    if return_prob is True:
        probdict = {}
        for i in dt.keys():
            dtx_woe = dt_woe[i].loc[:,x_woe] 
            dtx_woe = sm.add_constant(dtx_woe)
            probdict[i] = lrmodel.predict(dtx_woe)
        rt = {'card': card, 
              'prob': probdict}
    else:
        rt = card

    return rt


def scorecard_ply(dt, card, only_total_score=True, print_step=0, replace_blank_na=True, var_kp = None):
    '''
    Score Transformation
    ------
    `scorecard_ply` calculates credit score using the results from `scorecard`.
    
    Params
    ------
    dt: Original data
    card: Scorecard generated from `scorecard`.
    only_total_score: Logical, default is TRUE. If it is TRUE, then 
      the output includes only total credit score; Otherwise, if it 
      is FALSE, the output includes both total and each variable's 
      credit score.
    print_step: A non-negative integer. Default is 1. If print_step>0, 
      print variable names by each print_step-th iteration. If 
      print_step=0, no message is print.
    replace_blank_na: Logical. Replace blank values with NA. Defaults to True. 
      This parameter should be the same with woebin's.
    var_kp: Name of force kept variables, such as id column. Defaults to None.
    
    Return
    ------
    DataFrame
        Credit score
    
    Examples
    ------
    import scorecardpy as sc
    
    # load data
    dat = sc.germancredit()
    
    # filter variable via missing rate, iv, identical value rate
    dt_sel = sc.var_filter(dat, "creditability")
    
    # woe binning ------
    bins = sc.woebin(dt_sel, "creditability")
    dt_woe = sc.woebin_ply(dt_sel, bins)
    
    y = dt_woe.loc[:,'creditability']
    X = dt_woe.loc[:,dt_woe.columns != 'creditability']
    
    # logistic regression ------
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(penalty='l1', C=0.9, solver='saga')
    lr.fit(X, y)
    
    # # predicted proability
    # dt_pred = lr.predict_proba(X)[:,1]
    # # performace
    # # ks & roc plot
    # sc.perf_eva(y, dt_pred)
    
    # scorecard
    # Example I # creat a scorecard
    card = sc.scorecard(bins, lr, X.columns)
    
    # credit score
    # Example I # only total score
    score1 = sc.scorecard_ply(dt_sel, card)
    # Example II # credit score for both total and each variable
    score2 = sc.scorecard_ply(dt_sel, card, only_total_score = False)
    '''
  
    dt = dt.copy(deep=True)
    # remove date/time col
    # dt = rmcol_datetime_unique1(dt)
    # replace "" by NA
    if replace_blank_na: dt = rep_blank_na(dt)
    # print_step
    print_step = check_print_step(print_step)
    # card # if (is.list(card)) rbindlist(card)
    if isinstance(card, dict):
        card_df = pd.concat(card, ignore_index=True)
    elif isinstance(card, pd.DataFrame):
        card_df = card.copy(deep=True)
    # x variables
    xs = card_df.loc[card_df.variable != 'basepoints', 'variable'].unique()
    # length of x variables
    xs_len = len(xs)
    # initial datasets
    dat = dt.loc[:,list(set(dt.columns)-set(xs))]
    
    # loop on x variables
    for i in np.arange(xs_len):
        x_i = xs[i]
        # print xs
        if print_step>0 and bool((i+1)%print_step): 
            print(('{:'+str(len(str(xs_len)))+'.0f}/{} {}').format(i, xs_len, x_i))
        
        cardx = card_df.loc[card_df['variable']==x_i]
        dtx = dt[[x_i]]
        # score transformation
        dtx_points = woepoints_ply1(dtx, cardx, x_i, woe_points="points")
        dat = pd.concat([dat, dtx_points], axis=1)
    
    # set basepoints
    card_basepoints = list(card_df.loc[card_df['variable']=='basepoints','points'])[0] if 'basepoints' in card_df['variable'].unique() else 0
    # total score
    dat_score = dat[xs+'_points']
    dat_score.loc[:,'score'] = card_basepoints + dat_score.sum(axis=1)
    # dat_score = dat_score.assign(score = lambda x: card_basepoints + dat_score.sum(axis=1))
    # return
    if only_total_score: dat_score = dat_score[['score']]
    
    # check force kept variables
    if var_kp is not None:
        if isinstance(var_kp, str):
            var_kp = [var_kp]
        var_kp2 = list(set(var_kp) & set(list(dt)))
        len_diff_var_kp = len(var_kp) - len(var_kp2)
        if len_diff_var_kp > 0:
            warnings.warn("Incorrect inputs; there are {} var_kp variables are not exist in input data, which are removed from var_kp. \n {}".format(len_diff_var_kp, list(set(var_kp)-set(var_kp2))) )
        var_kp = var_kp2 if len(var_kp2)>0 else None
    if var_kp is not None: dat_score = pd.concat([dt[var_kp], dat_score], axis = 1)
    return dat_score
    
    

