# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from .condition_fun import *


def eva_dfkslift(df, groupnum=None):
    if groupnum is None: groupnum=len(df.index)
    # good bad func
    def n0(x): return sum(x==0)
    def n1(x): return sum(x==1)
    df_kslift = df.sort_values('pred', ascending=False).reset_index(drop=True)\
      .assign(group=lambda x: np.ceil((x.index+1)/(len(x.index)/groupnum)))\
      .groupby('group')['label'].agg([n0,n1])\
      .reset_index().rename(columns={'n0':'good','n1':'bad'})\
      .assign(
        group=lambda x: (x.index+1)/len(x.index),
        good_distri=lambda x: x.good/sum(x.good), 
        bad_distri=lambda x: x.bad/sum(x.bad), 
        badrate=lambda x: x.bad/(x.good+x.bad),
        cumbadrate=lambda x: np.cumsum(x.bad)/np.cumsum(x.good+x.bad),
        lift=lambda x: (np.cumsum(x.bad)/np.cumsum(x.good+x.bad))/(sum(x.bad)/sum(x.good+x.bad)),
        cumgood=lambda x: np.cumsum(x.good)/sum(x.good), 
        cumbad=lambda x: np.cumsum(x.bad)/sum(x.bad)
      ).assign(ks=lambda x:abs(x.cumbad-x.cumgood))
    # bind 0
    df_kslift=pd.concat([
      pd.DataFrame({'group':0, 'good':0, 'bad':0, 'good_distri':0, 'bad_distri':0, 'badrate':0, 'cumbadrate':np.nan, 'cumgood':0, 'cumbad':0, 'ks':0, 'lift':np.nan}, index=np.arange(1)),
      df_kslift
    ], ignore_index=True)
    # return
    return df_kslift
# plot ks    
def eva_pks(dfkslift, title):
    dfks = dfkslift.loc[lambda x: x.ks==max(x.ks)].sort_values('group').iloc[0]
    ###### plot ###### 
    # fig, ax = plt.subplots()
    # ks, cumbad, cumgood
    plt.plot(dfkslift.group, dfkslift.ks, 'b-', 
      dfkslift.group, dfkslift.cumgood, 'k-', 
      dfkslift.group, dfkslift.cumbad, 'k-')
    # ks vline
    plt.plot([dfks['group'], dfks['group']], [0, dfks['ks']], 'r--')
    # set xylabel
    plt.gca().set(title=(''if title is None else title+': ')+'K-S', 
      xlabel='% of population', ylabel='% of total Good/Bad', 
      xlim=[0,1], ylim=[0,1], aspect='equal')
    # text
    # plt.text(0.5,0.96,'K-S', fontsize=15,horizontalalignment='center')
    plt.text(0.2,0.8,'Bad',horizontalalignment='center')
    plt.text(0.8,0.55,'Good',horizontalalignment='center')
    plt.text(dfks['group'], dfks['ks'], 'KS:'+ str(round(dfks['ks'],4)), horizontalalignment='center',color='b')
    # plt.grid()
    # plt.show()
    # return fig
# plot lift
def eva_plift(dfkslift, title):
    badrate_avg = sum(dfkslift.bad)/sum(dfkslift.good+dfkslift.bad)
    ###### plot ###### 
    # fig, ax = plt.subplots()
    # ks, cumbad, cumgood
    plt.plot(dfkslift.group, dfkslift.cumbadrate, 'k-')
    # ks vline
    plt.plot([0, 1], [badrate_avg, badrate_avg], 'r--')
    # set xylabel
    plt.gca().set(title=(''if title is None else title+': ')+'Lift', 
      xlabel='% of population', ylabel='% of Bad', 
      xlim=[0,1], ylim=[0,1], aspect='equal')
    # text
    # plt.text(0.5,0.96,'Lift', fontsize=15,horizontalalignment='center')
    plt.text(0.7,np.mean(dfkslift.cumbadrate),'cumulate badrate',horizontalalignment='center')
    plt.text(0.7,badrate_avg,'average badrate',horizontalalignment='center')
    # plt.grid()
    # plt.show()
    # return fig

def eva_dfrocpr(df):
    def n0(x): return sum(x==0)
    def n1(x): return sum(x==1)
    dfrocpr = df.sort_values('pred')\
      .groupby('pred')['label'].agg([n0,n1,len])\
      .reset_index().rename(columns={'n0':'countN','n1':'countP','len':'countpred'})\
      .assign(
        FN = lambda x: np.cumsum(x.countP), 
        TN = lambda x: np.cumsum(x.countN) 
      ).assign(
        TP = lambda x: sum(x.countP) - x.FN, 
        FP = lambda x: sum(x.countN) - x.TN
      ).assign(
        TPR = lambda x: x.TP/(x.TP+x.FN), 
        FPR = lambda x: x.FP/(x.TN+x.FP), 
        precision = lambda x: x.TP/(x.TP+x.FP), 
        recall = lambda x: x.TP/(x.TP+x.FN)
      ).assign(
        F1 = lambda x: 2*x.precision*x.recall/(x.precision+x.recall)
      )
    return dfrocpr
# plot roc
def eva_proc(dfrocpr, title):
    auc = dfrocpr.sort_values(['FPR','TPR'])\
          .assign(
            TPR_lag=lambda x: x['TPR'].shift(1), FPR_lag=lambda x: x['FPR'].shift(1)
          ).assign(
            auc=lambda x: (x.TPR+x.TPR_lag)*(x.FPR-x.FPR_lag)/2
          )['auc'].sum()
    ###### plot ###### 
    # fig, ax = plt.subplots()
    # ks, cumbad, cumgood
    plt.plot(dfrocpr.FPR, dfrocpr.TPR, 'k-')
    # ks vline
    x=np.array(np.arange(0,1.1,0.1))
    plt.plot(x, x, 'r--')
    # fill 
    plt.fill_between(dfrocpr.FPR, 0, dfrocpr.TPR, color='blue', alpha=0.1)
    # set xylabel
    plt.gca().set(title=(''if title is None else title+': ')+'ROC',
      xlabel='FPR', ylabel='TPR', 
      xlim=[0,1], ylim=[0,1], aspect='equal')
    # text
    # plt.text(0.5,0.96, 'ROC', fontsize=15, horizontalalignment='center')
    plt.text(0.55,0.45, 'AUC:'+str(round(auc,4)), horizontalalignment='center', color='b')
    # plt.grid()
    # plt.show()
    # return fig
# plot ppr
def eva_ppr(dfrocpr, title):
    ###### plot ###### 
    # fig, ax = plt.subplots()
    # ks, cumbad, cumgood
    plt.plot(dfrocpr.recall, dfrocpr.precision, 'k-')
    # ks vline
    x=np.array(np.arange(0,1.1,0.1))
    plt.plot(x, x, 'r--')
    # set xylabel
    plt.gca().set(title=(''if title is None else title+': ')+'P-R', 
      xlabel='Recall', ylabel='Precision', 
      xlim=[0,1], ylim=[0,1], aspect='equal')
    # text
    # plt.text(0.5,0.96, 'P-R', fontsize=15, horizontalalignment='center')
    # plt.grid()
    # plt.show()
    # return fig
# plot f1
def eva_pf1(dfrocpr, title):
    dfrocpr=dfrocpr.assign(pop=lambda x: np.cumsum(x.countpred)/sum(x.countpred))
    ###### plot ###### 
    # fig, ax = plt.subplots()
    # ks, cumbad, cumgood
    plt.plot(dfrocpr['pop'], dfrocpr['F1'], 'k-')
    # ks vline
    F1max_pop = dfrocpr.loc[dfrocpr['F1'].idxmax(),'pop']
    F1max_F1 = dfrocpr.loc[dfrocpr['F1'].idxmax(),'F1']
    plt.plot([F1max_pop,F1max_pop], [0,F1max_F1], 'r--')
    # set xylabel
    plt.gca().set(title=(''if title is None else title+': ')+'F1', 
      xlabel='% of population', ylabel='F1', 
      xlim=[0,1], ylim=[0,1], aspect='equal')
    # pred text
    pred_0=dfrocpr.loc[dfrocpr['pred'].idxmin(),'pred']
    pred_F1max=dfrocpr.loc[dfrocpr['F1'].idxmax(),'pred']
    pred_1=dfrocpr.loc[dfrocpr['pred'].idxmax(),'pred']
    if np.mean(dfrocpr.pred) < 0 or np.mean(dfrocpr.pred) > 1: 
        pred_0 = -pred_0
        pred_F1max = -pred_F1max
        pred_1 = -pred_1
    plt.text(0, 0, 'pred \n'+str(round(pred_0,4)), horizontalalignment='left',color='b')
    plt.text(F1max_pop, 0, 'pred \n'+str(round(pred_F1max,4)), horizontalalignment='center',color='b')
    plt.text(1, 0, 'pred \n'+str(round(pred_1,4)), horizontalalignment='right',color='b')
    # title F1
    plt.text(F1max_pop, F1max_F1, 'F1 max: \n'+ str(round(F1max_F1,4)), horizontalalignment='center',color='b')
    # plt.grid()
    # plt.show()
    # return fig
    
#' KS, ROC, Lift, PR
#'
#' \code{perf_eva} provides performance evaluations, such as kolmogorov-smirnow(ks), ROC, lift and precision-recall curves, based on provided label and predicted probability values.
#'
#' @name perf_eva
#' @param label Label values, such as 0s and 1s, 0 represent for good and 1 for bad.
#' @param pred Predicted probability or score.
#' @param title Title of plot, default is "performance".
#' @param groupnum The group number when calculating KS.  Default NULL, which means the number of sample size.
#' @param type Types of performance plot, such as "ks", "lift", "roc", "pr". Default c("ks", "roc").
#' @param show_plot Logical value, default is TRUE. It means whether to show plot.
#' @param positive Value of positive class, default is "bad|1".
#' @param seed Integer, default is 186. The specify seed is used for random sorting data.
#' @return ks, roc, lift, pr
#'
#' @details
#' Accuracy = true positive and true negative/total cases
#'
#' Error rate = false positive and false negative/total cases
#'
#' TPR, True Positive Rate(Recall or Sensitivity) = true positive/total actual positive
#'
#' PPV, Positive Predicted Value(Precision) = true positive/total predicted positive
#'
#' TNR, True Negative Rate(Specificity) = true negative/total actual negative
#'
#' NPV, Negative Predicted Value = true negative/total predicted negative
#'
#'
#'
#'
#'
# https://en.wikipedia.org/wiki/Positive_and_negative_predictive_values
# ROC curve: Sensitivity ~ 1-Specificity with different threshold
# Lift chart: Lift(PV+/p1) ~ Depth with different threshold
# Gains chart: PV + ~ Depth with different threshold
#'
#' @seealso \code{\link{perf_psi}}
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
#' m1 = glm( creditability ~ ., family = binomial(), data = dt_woe)
#' # summary(m1)
#'
#' # Select a formula-based model by AIC
#' m_step = step(m1, direction="both", trace=FALSE)
#' m2 = eval(m_step$call)
#' # summary(m2)
#'
#' # predicted proability
#' dt_pred = predict(m2, type='response', dt_woe)
#'
#' # performance ------
#' # Example I # only ks & auc values
#' perf_eva(dt_woe$creditability, dt_pred, show_plot=FALSE)
#'
#' # Example II # ks & roc plot
#' perf_eva(dt_woe$creditability, dt_pred)
#'
#' # Example III # ks, lift, roc & pr plot
#' perf_eva(dt_woe$creditability, dt_pred, type = c("ks","lift","roc","pr"))
#' }
#' @import data.table ggplot2 gridExtra
#' @export
#'
def perf_eva(label, pred, title=None, groupnum=None, plot_type=["ks", "roc"], show_plot=True, positive="bad|1", seed=186):
    # inputs checking
    if len(label) != len(pred):
        warnings.warn('Incorrect inputs; label and pred should be list with the same length.')
    # if pred is score
    if np.mean(pred) < 0 or np.mean(pred) > 1:
        warnings.warn('Since the average of pred is not in [0,1], it is treated as predicted score but not probability.')
        pred = -pred
    # random sort datatable
    df = pd.DataFrame({'label':label, 'pred':pred}).sample(frac=1, random_state=seed)
    # remove NAs
    if any(np.unique(df.isna())):
        warnings.warn('The NANs in \'label\' or \'pred\' were removed.')
        df = df.dropna()
    # check label
    df = check_y(df, 'label', positive)
    # title
    if title is not None: title=title+': '
    
    ### data ###
    # dfkslift ------
    if any([i in plot_type for i in ['ks', 'lift']]):
        dfkslift = eva_dfkslift(df, groupnum)
        if 'ks' in plot_type: df_ks = dfkslift
        if 'lift' in plot_type: df_lift = dfkslift
    # dfrocpr ------
    if any([i in plot_type for i in ["roc","pr",'f1']]):
        dfrocpr = eva_dfrocpr(df)
        if 'roc' in plot_type: df_roc = dfrocpr
        if 'pr' in plot_type: df_pr = dfrocpr
        if 'f1' in plot_type: df_f1 = dfrocpr
    ### return list ### 
    rt = {}
    # plot, KS ------
    if 'ks' in plot_type:
        rt['KS'] = round(dfkslift.loc[lambda x: x.ks==max(x.ks),'ks'].iloc[0],4)
    # plot, ROC ------
    if 'roc' in plot_type:
        auc = dfrocpr.sort_values(['FPR','TPR'])\
          .assign(
            TPR_lag=lambda x: x['TPR'].shift(1), FPR_lag=lambda x: x['FPR'].shift(1)
          ).assign(
            auc=lambda x: (x.TPR+x.TPR_lag)*(x.FPR-x.FPR_lag)/2
          )['auc'].sum()
        ### 
        rt['AUC'] = round(auc, 4)
        rt['Gini'] = round(2*auc-1, 4)
    
    ### export plot ### 
    if show_plot:
        plist = ["eva_p"+i+'(df_'+i+',title)' for i in plot_type]
        subplot_nrows = np.floor(len(plist)/2)
        subplot_ncols = np.ceil(len(plist)/subplot_nrows)
        
        fig = plt.figure()
        for i in np.arange(len(plist)):
            plt.subplot(subplot_nrows,subplot_ncols,i+1)
            eval(plist[i])
        plt.show()
        rt['pic'] = fig
    # return 
    return rt
    

#' PSI
#'
#' \code{perf_psi} calculates population stability index (PSI) and provides credit score distribution based on credit score datasets.
#'
#' @param score A list of credit score for actual and expected data samples. For example, score = list(actual = score_A, expect = score_E), both score_A and score_E are dataframes with the same column names.
#' @param label A list of label value for actual and expected data samples. The default is NULL. For example, label = list(actual = label_A, expect = label_E), both label_A and label_E are vectors or dataframes. The label values should be 0s and 1s, 0 represent for good and 1 for bad.
#' @param title Title of plot, default is NULL.
#' @param x_limits x-axis limits, default is c(100, 800).
#' @param x_tick_break x-axis ticker break, default is 50.
#' @param show_plot Logical, default is TRUE. It means whether to show plot.
#' @param return_distr_dat Logical, default is FALSE.
#' @param seed Integer, default is 186. The specify seed is used for random sorting data.
#'
#' @return a dataframe of psi & plots of credit score distribution
#'
#' @details The population stability index (PSI) formula is displayed below: \deqn{PSI = \sum((Actual\% - Expected\%)*(\ln(\frac{Actual\%}{Expected\%}))).} The rule of thumb for the PSI is as follows: Less than 0.1 inference insignificant change, no action required; 0.1 - 0.25 inference some minor change, check other scorecard monitoring metrics; Greater than 0.25 inference major shift in population, need to delve deeper.
#'
#' @seealso \code{\link{perf_eva}}
#'
#' @examples
#' \dontrun{
#' # load germancredit data
#' data("germancredit")
#'
#' # filter variable via missing rate, iv, identical value rate
#' dt_sel = var_filter(germancredit, "creditability")
#'
#' # breaking dt into train and test ------
#' dt_list = split_df(dt_sel, "creditability", ratio = 0.6, seed=21)
#' dt_train = dt_list$train; dt_test = dt_list$test
#'
#' # woe binning ------
#' bins = woebin(dt_train, "creditability")
#'
#' # converting train and test into woe values
#' train = woebin_ply(dt_train, bins)
#' test = woebin_ply(dt_test, bins)
#'
#' # glm ------
#' m1 = glm(creditability ~ ., family = binomial(), data = train)
#' # summary(m1)
#'
#' # Select a formula-based model by AIC
#' m_step = step(m1, direction="both", trace=FALSE)
#' m2 = eval(m_step$call)
#' # summary(m2)
#'
#' # predicted proability
#' train_pred = predict(m2, type='response', train)
#' test_pred = predict(m2, type='response', test)
#'
#' # # ks & roc plot
#' # perf_eva(train$creditability, train_pred, title = "train")
#' # perf_eva(test$creditability, test_pred, title = "test")
#'
#' #' # scorecard
#' card = scorecard(bins, m2)
#'
#' # credit score, only_total_score = TRUE
#' train_score = scorecard_ply(dt_train, card)
#' test_score = scorecard_ply(dt_test, card)
#'
#' # Example I # psi
#' psi = perf_psi(
#'   score = list(train = train_score, test = test_score),
#'   label = list(train = train$creditability, test = test$creditability)
#' )
#' # psi$psi  # psi dataframe
#' # psi$pic  # pic of score distribution
#'
#' # Example II # specifying score range
#' psi_s = perf_psi(
#'   score = list(train = train_score, test = test_score),
#'   label = list(train = train$creditability, test = test$creditability),
#'   x_limits = c(200, 750),
#'   x_tick_break = 50
#'   )
#'
#' # Example III # credit score, only_total_score = FALSE
#' train_score2 = scorecard_ply(dt_train, card, only_total_score=FALSE)
#' test_score2 = scorecard_ply(dt_test, card, only_total_score=FALSE)
#'
#' # psi
#' psi2 = perf_psi(
#'   score = list(train = train_score2, test = test_score2),
#'   label = list(train = train$creditability, test = test$creditability)
#' )
#' # psi2$psi  # psi dataframe
#' # psi2$pic  # pic of score distribution
#' }
#' @import data.table ggplot2 gridExtra
#' @export
#'
def perf_psi(score, label=None, title=None, x_limits=[100,800], x_tick_break=50, 
             show_plot=True, seed=186, return_distr_dat=False):
    # inputs checking
    ## score
    if not isinstance(score, dict) and len(score) != 2:
        raise Exception("Incorrect inputs; score should be a dictionary with two elements.")
    else:
        if any([not isinstance(i, pd.DataFrame) for i in score.values()]):
            raise Exception("Incorrect inputs; score is a dictionary of two dataframes.")
        if len(np.unique([list(i.columns) for i in score.values()])) != 1:
            raise Exception("Incorrect inputs; the column names of two dataframes in score should be the same.")
    ## label
    if label is not None:
        if not isinstance(label, dict) and len(label) != 2:
            raise Exception("Incorrect inputs; label should be a dictionary with two elements.")
        else:
            if set(score.keys()) != set(label.keys()):
                raise Exception("Incorrect inputs; the keys of score and label should be the same. ")
            for i in label.keys():
                if isinstance(label[i], pd.DataFrame):
                    if len(label[i].columns) == 1:
                        label[i] = label[i].iloc[:,0]
                    else:
                        raise Exception("Incorrect inputs; the number of columns in label should be 1.")
    # score dataframe column names
    score_names = score[list(score.keys())[0]].columns
    # merge label with score
    for i in score.keys():
        score[i] = score[i].copy(deep=True)
        if label is not None:
            score[i].loc[:,'y'] = label[i]
        else:
            score[i].copy(deep=True).loc[:,'y'] = np.nan
    # dateset of score and label
    dt_sl = pd.concat(score, names=['ae', 'rowid']).reset_index()\
      .sample(frac=1, random_state=seed)
      # ae refers to 'Actual & Expected'
    
    # PSI function
    def psi(dat):
        dt_bae = dat.groupby(['ae','bin']).size().reset_index(name='N')\
          .pivot_table(values='N', index='bin', columns='ae').fillna(0.9)\
          .agg(lambda x: x/sum(x))
        dt_bae.columns = ['A','E']
        psi_dt = dt_bae.assign(
          AE = lambda x: x.A-x.E,
          logAE = lambda x: np.log(x.A/x.E)
        ).assign(
          bin_PSI=lambda x: x.AE*x.logAE
        )['bin_PSI'].sum()
        return psi_dt
    
    # return psi and pic
    rt_psi = {}
    rt_pic = {}
    rt_dat = {}
    rt = {}
    for sn in score_names:
        # dataframe with columns of ae y sn
        dat = dt_sl[['ae', 'y', sn]]
        if len(dt_sl[sn].unique()) > 10:
            # breakpoints
            brkp = np.unique([np.floor(min(dt_sl[sn])/x_tick_break)*x_tick_break]+\
              list(np.arange(x_limits[0]+x_tick_break, x_limits[1]-x_tick_break, x_tick_break))+\
              [np.ceil(max(dt_sl[sn])/x_tick_break)*x_tick_break])
            # cut
            labels = ['[{},{})'.format(int(brkp[i]), int(brkp[i+1])) for i in range(len(brkp)-1)]
            dat.loc[:,'bin'] = pd.cut(dat[sn], brkp, right=False, labels=labels)
        # psi ------
        rt_psi[sn] = pd.DataFrame({'PSI':psi(dat)},index=np.arange(1)) 
    
        # distribution of scorecard probability
        def good(x): return sum(x==0)
        def bad(x): return sum(x==1)
        distr_prob = dat.groupby(['ae', 'bin'])\
          ['y'].agg([good, bad])\
          .assign(N=lambda x: x.good+x.bad,
            badprob=lambda x: x.bad/(x.good+x.bad)
          ).reset_index()
        distr_prob.loc[:,'distr'] = distr_prob.groupby('ae')['N'].transform(lambda x:x/sum(x))
        # pivot table
        distr_prob = distr_prob.pivot_table(values=['N','badprob', 'distr'], index='bin', columns='ae')
            
        # plot ------
        if show_plot:
            ###### param ######
            ind = np.arange(len(distr_prob.index))    # the x locations for the groups
            width = 0.35       # the width of the bars: can also be len(x) sequence
            ###### plot ###### 
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            title_string = sn+'_PSI: '+str(round(psi(dat),4))
            title_string = title_string if title is None else str(title)+' '+title_string
            # ax1
            p1 = ax1.bar(ind, distr_prob.distr.iloc[:,0], width, color=(24/254, 192/254, 196/254), alpha=0.6)
            p2 = ax1.bar(ind+width, distr_prob.distr.iloc[:,1], width, color=(246/254, 115/254, 109/254), alpha=0.6)
            # ax2
            p3 = ax2.plot(ind+width/2, distr_prob.badprob.iloc[:,0], color=(24/254, 192/254, 196/254))
            ax2.scatter(ind+width/2, distr_prob.badprob.iloc[:,0], facecolors='w', edgecolors=(24/254, 192/254, 196/254))
            p4 = ax2.plot(ind+width/2, distr_prob.badprob.iloc[:,1], color=(246/254, 115/254, 109/254))
            ax2.scatter(ind+width/2, distr_prob.badprob.iloc[:,1], facecolors='w', edgecolors=(246/254, 115/254, 109/254))
            # settings
            ax1.set_ylabel('Score distribution')
            ax2.set_ylabel('Bad probability')#, color='blue')
            # ax2.tick_params(axis='y', colors='blue')
            # ax1.set_yticks(np.arange(0, np.nanmax(distr_prob['distr'].values), 0.2))
            # ax2.set_yticks(np.arange(0, 1+0.2, 0.2))
            ax1.set_ylim([0,np.ceil(np.nanmax(distr_prob['distr'].values)*10)/10])
            ax2.set_ylim([0,1])
            plt.xticks(ind, distr_prob.index)
            plt.title(title_string, loc='left')
            ax1.legend((p1[0], p2[0]), list(distr_prob.columns.levels[1]), loc='upper left')
            ax2.legend((p3[0], p4[0]), list(distr_prob.columns.levels[1]), loc='upper right')
            # show plot 
            plt.show()
            
            # return of pic
            rt_pic[sn] = fig
        
        # return distr_dat ------
        if return_distr_dat:
            rt_dat[sn] = distr_prob[['N','badprob']].reset_index()
    # return rt
    rt['psi'] = pd.concat(rt_psi).reset_index().rename(columns={'level_0':'variable'})[['variable', 'PSI']]
    rt['pic'] = rt_pic
    if return_distr_dat: rt['dat'] = rt_dat
    return rt
    
