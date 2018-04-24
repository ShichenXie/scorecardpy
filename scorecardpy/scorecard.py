# -*- coding: utf-8 -*-

import pandas as pd
import re
from .condition_fun import *
from .woebin import woepoints_ply1


# coefficients in scorecard
def ab(points0=600, odds0=1/60, pdo=50):
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
    b = pdo/np.log(2)
    a = points0 + b*np.log(odds0) #log(odds0/(1+odds0))
    return {'a':a, 'b':b}



#' Creating a Scorecard
#'
#' \code{scorecard} creates a scorecard based on the results from \code{woebin} and \code{glm}.
#'
#' @param bins Binning information generated from \code{woebin} function.
#' @param model A glm model object.
#' @param points0 Target points, default 600.
#' @param odds0 Target odds, default 1/19. Odds = p/(1-p).
#' @param pdo Points to Double the Odds, default 50.
#' @param basepoints_eq0 Logical, default is FALSE. If it is TRUE, the basepoints will equally distribute to each variable.
#' @return scorecard
#'
#' @seealso \code{\link{scorecard_ply}}
#'
#' @examples
#' \dontrun{
#' # load germancredit data
#' data("germancredit")
#'
#' # filter variable via missing rate, iv, identical value rate
#' dt_sel = var_filter(germancredit, "creditability")
#'
#' # woe binning ------
#' bins = woebin(dt_sel, "creditability")
#' dt_woe = woebin_ply(dt_sel, bins)
#'
#' # glm ------
#' m = glm(creditability ~ ., family = binomial(), data = dt_woe)
#' # summary(m)
#'
#' # Select a formula-based model by AIC
#' m_step = step(m, direction="both", trace=FALSE)
#' m = eval(m_step$call)
#' # summary(m)
#'
#' # predicted proability
#' # dt_pred = predict(m, type='response', dt_woe)
#'
#' # performace
#' # ks & roc plot
#' # perf_eva(dt_woe$creditability, dt_pred)
#'
#' # scorecard
#' # Example I # creat a scorecard
#' card = scorecard(bins, m)
#'
#' # credit score
#' # Example I # only total score
#' score1 = scorecard_ply(dt, card)
#'
#' # Example II # credit score for both total and each variable
#' score2 = scorecard_ply(dt, card, only_total_score = F)
#' }
#' @import data.table
#' @export
#'
def scorecard(bins, model, xcolumns, points0=600, odds0=1/19, pdo=50, basepoints_eq0=False):
    # coefficients
    aabb = ab(points0, odds0, pdo)
    a, b = aabb.values()
    # odds = pred/(1-pred); score = a - b*log(odds)
    
    # bins # if (is.list(bins)) rbindlist(bins)
    if isinstance(bins, dict):
        bins = pd.concat(bins, ignore_index=True)
    xs = [re.sub('_woe$', '', i) for i in xcolumns]
    # coefficients
    coef_df = pd.Series(model.coef_[0], index=np.array(xs))\
      .loc[lambda x: x != 0]#.reset_index(drop=True)
    
    # scorecard
    len_x = len(coef_df)
    basepoints = a - b*model.intercept_[0]
    card = {}
    if basepoints_eq0:
        card['basepoints'] = pd.DataFrame({'variable':"basepoints", 'bin':np.nan, 'points':0}, index=np.arange(1))
        for i in coef_df.index:
            card[i] = bins.loc[bins['variable']==i,['variable', 'bin', 'woe']]\
              .assign(points = lambda x: round(-b*x['woe']*coef_df[i] + basepoints/len_x))\
              [["variable", "bin", "points"]]
    else:
        card['basepoints'] = pd.DataFrame({'variable':"basepoints", 'bin':np.nan, 'points':round(basepoints)}, index=np.arange(1))
        for i in coef_df.index:
            card[i] = bins.loc[bins['variable']==i,['variable', 'bin', 'woe']]\
              .assign(points = lambda x: round(-b*x['woe']*coef_df[i]))\
              [["variable", "bin", "points"]]
    return card



#' Score Transformation
#'
#' \code{scorecard_ply} calculates credit score using the results from \code{scorecard}.
#'
#' @param dt Original data
#' @param card Scorecard generated from \code{scorecard}.
#' @param only_total_score  Logical, default is TRUE. If it is TRUE, then the output includes only total credit score; Otherwise, if it is FALSE, the output includes both total and each variable's credit score.
#' @param print_step A non-negative integer. Default is 1. If print_step>0, print variable names by each print_step-th iteration. If print_step=0, no message is print.
#' @return Credit score
#'
#' @seealso \code{\link{scorecard}}
#'
#' @examples
#' \dontrun{
#' # load germancredit data
#' data("germancredit")
#'
#' # filter variable via missing rate, iv, identical value rate
#' dt_sel = var_filter(germancredit, "creditability")
#'
#' # woe binning ------
#' bins = woebin(dt_sel, "creditability")
#' dt_woe = woebin_ply(dt_sel, bins)
#'
#' # glm ------
#' m = glm(creditability ~ ., family = binomial(), data = dt_woe)
#' # summary(m)
#'
#' # Select a formula-based model by AIC
#' m_step = step(m, direction="both", trace=FALSE)
#' m = eval(m_step$call)
#' # summary(m)
#'
#' # predicted proability
#' # dt_pred = predict(m, type='response', dt_woe)
#'
#' # performace
#' # ks & roc plot
#' # perf_eva(dt_woe$creditability, dt_pred)
#'
#' # scorecard
#' # Example I # creat a scorecard
#' card = scorecard(bins, m)
#'
#' # credit score
#' # Example I # only total score
#' score1 = scorecard_ply(dt, card)
#'
#' # Example II # credit score for both total and each variable
#' score2 = scorecard_ply(dt, card, only_total_score = F)
#' }
#' @import data.table
#' @export
#'
def scorecard_ply(dt, card, only_total_score=True, print_step=0):
    # remove date/time col
    dt = rm_datetime_col(dt)
    # replace "" by NA
    dt = rep_blank_na(dt)
    # print_step
    print_step = check_print_step(print_step)
    # card # if (is.list(card)) rbindlist(card)
    if isinstance(card, dict):
        card_df = pd.concat(card, ignore_index=True)
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
    # return
    if only_total_score: dat_score = dat_score[['score']]
    return dat_score
    
    

