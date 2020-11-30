# -*- coding: utf-8 -*-

import pandas as pd
from pandas.api.types import CategoricalDtype
import pkg_resources


def germancredit():
    '''
    German Credit Data
    ------
    Credit data that classifies debtors described by a set of 
    attributes as good or bad credit risks. See source link 
    below for detailed information.
    [source](https://archive.ics.uci.edu/ml/datasets/Statlog+(German+Credit+Data))
    
    Params
    ------
    
    Returns
    ------
    DataFrame
    
    Examples
    ------
    import scorecardpy as sc
    
    # load data
    dat = sc.germancredit()
    
    # # data structure
    # dat.shape
    # dat.dtypes
    '''
    DATA_FILE = pkg_resources.resource_filename('scorecardpy', 'data/germancredit.csv')
    
    dat = pd.read_csv(DATA_FILE)
    # categorical levels
    cate_levels = {
            "status_of_existing_checking_account": ['... < 0 DM', '0 <= ... < 200 DM', '... >= 200 DM / salary assignments for at least 1 year', 'no checking account'], 
            "credit_history": ["no credits taken/ all credits paid back duly", "all credits at this bank paid back duly", "existing credits paid back duly till now", "delay in paying off in the past", "critical account/ other credits existing (not at this bank)"], 
            "savings_account_and_bonds": ["... < 100 DM", "100 <= ... < 500 DM", "500 <= ... < 1000 DM", "... >= 1000 DM", "unknown/ no savings account"],
            "present_employment_since": ["unemployed", "... < 1 year", "1 <= ... < 4 years", "4 <= ... < 7 years", "... >= 7 years"], 
            "personal_status_and_sex": ["male : divorced/separated", "female : divorced/separated/married", "male : single", "male : married/widowed", "female : single"], 
            "other_debtors_or_guarantors": ["none", "co-applicant", "guarantor"], 
            "property": ["real estate",  "building society savings agreement/ life insurance",  "car or other, not in attribute Savings account/bonds",  "unknown / no property"],
            "other_installment_plans": ["bank", "stores", "none"],
            "housing": ["rent", "own", "for free"], 
            "job": ["unemployed/ unskilled - non-resident", "unskilled - resident", "skilled employee / official", "management/ self-employed/ highly qualified employee/ officer"],
            "telephone": ["none", "yes, registered under the customers name"], 
            "foreign_worker": ["yes", "no"]}
    # func of cate
    def cate_type(levels):
        return CategoricalDtype(categories=levels, ordered=True)
    # to cate
    for i in cate_levels.keys():
        dat[i] = dat[i].astype(cate_type(cate_levels[i]))
    # return
    return dat


'''
# datasets
import scorecardpy as sc
dat1 = sc.germancredit()
dat1 = check_y(dat1, 'creditability', 'bad|1')
dat2 = pd.DataFrame({'creditability':[0,1]}).sample(50, replace=True)
# dat2 = pd.DataFrame({'creditability':np.random.choice([0,1], 50)})
dat = pd.concat([dat2, dat1], ignore_index=True)


y = "creditability"
x_i = "duration.in.month"
dtm = pd.DataFrame({'y':dat[y], 'variable':x_i, 'value':dt[x_i]})

###### dtm ######
# y
y = dat['creditability']

# x
# numerical data
xvar =  "credit_amount" # "foreign_worker # 'age_in_years' #'number_of_existing_credits_at_this_bank' # 
x= dat1[xvar]
spl_val = [2600, 9960, "6850%,%missing"]
breaks = [2000, 4000, 6000]
breaks = ['26%,%missing', 28, 35, 37]

# categorical data
xvar= 'purpose'#'housing' # "job" # "credit_amount"; #
x= dat[xvar] # pd.Categorical(dat[xvar], categories=['rent', 'own','for free']) 
breaks = ["own", "for free%,%rent%,%missing"]
breaks = ["own", "for free%,%rent"]


dtm = pd.DataFrame({'y':y, 'variable':xvar, 'value':x})
# dtm.value = None
'''

