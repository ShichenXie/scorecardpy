# -*- coding: utf-8 -*-

import pandas as pd


def _AssignBin(x, cutOffPoints, special_attribute=None):

    '''
    :param x: 某个变量的某个取值
    :param cutOffPoints: list 上述变量的分箱结果，用切分点表示
    :param special_attribute:list  不参与分箱的特殊取值
    :return: 分箱后的对应的第几个箱，从0开始
    '''
    cutOffPoints2 = [i for i in cutOffPoints if i not in special_attribute]
    numBin = len(cutOffPoints2)
    if x in special_attribute:
        i = special_attribute.index(x)+1
        return 'Bin {}'.format(0-i)
    if x<=cutOffPoints2[0]:
        return 'Bin 0'
    elif x > cutOffPoints2[-1]:
        return 'Bin {}'.format(numBin)
    else:
        for i in range(0,numBin):
            if cutOffPoints2[i] < x <=  cutOffPoints2[i+1]:
                return 'Bin {}'.format(i+1)


def _FeatureMonotone(x):
    '''
    Param x: list cut off list
    :return: 返回序列x中有几个元素不满足单调性，以及这些元素的位置
    '''
    monotone = [x[i]<x[i+1] and x[i] < x[i-1] or x[i]>x[i+1] and x[i] > x[i-1] for i in range(1,len(x)-1)]
    index_of_nonmonotone = [i+1 for i in range(len(monotone)) if monotone[i]]
    return {'count_of_nonmonotone':monotone.count(True), 'index_of_nonmonotone':index_of_nonmonotone}


def _BinBadRate(df, col, target, grantRateIndicator=0):
    '''
    :param df: 需要计算好坏比率的数据集
    :param col: 需要计算好坏比率的特征
    :param target: 好坏标签
    :param grantRateIndicator: 1返回总体的坏样本率，0不返回
    :return: 每箱的坏样本率，以及总体的坏样本率（当grantRateIndicator＝＝1时）
    '''
    total = df.groupby([col])[target].count()
    total = pd.DataFrame({'total': total})
    bad = df.groupby([col])[target].sum()
    bad = pd.DataFrame({'bad': bad})
    regroup = total.merge(bad, left_index=True, right_index=True, how='left')
    regroup.reset_index(drop=False, inplace=True)
    regroup['bad_rate'] = regroup.apply(lambda x: x.bad / x.total, axis=1)
    dicts = dict(zip(regroup[col],regroup['bad_rate']))
    if grantRateIndicator==0:
        return (dicts, regroup)
    N = sum(regroup['total'])
    B = sum(regroup['bad'])
    overallRate = B * 1.0 / N
    return (dicts, regroup, overallRate)

def _BadRateMonotone(df, sortByVar, target,special_attribute = []):
    '''
    :param df: 包含检验坏样本率的变量,和目标变量
    :param sortByVar: 需要检验坏样本率的变量
    :param target: 目标变量，0、1表示好、坏
    :param special_attribute: 不参与检验的特殊值
    :return: 坏样本率单调与否
    '''
    df2 = df.loc[~df[sortByVar].isin(special_attribute)]
    if len(set(df2[sortByVar])) <= 2:
        return True
    regroup = _BinBadRate(df2, sortByVar, target)[1]
    combined = zip(regroup['total'],regroup['bad'])
    badRate = [x[1]*1.0/x[0] for x in combined]
    badRateNotMonotone = _FeatureMonotone(badRate)['count_of_nonmonotone']
    if badRateNotMonotone > 0:
        return False
    else:
        return True