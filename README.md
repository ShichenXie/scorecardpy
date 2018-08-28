# scorecardpy

[![PyPI release](https://img.shields.io/pypi/v/scorecardpy.svg)](https://pypi.python.org/pypi/scorecardpy)
[![Downloads](http://pepy.tech/badge/scorecardpy)](http://pepy.tech/project/scorecardpy)
[![PyPI version](https://img.shields.io/pypi/pyversions/scorecardpy.svg)](https://pypi.python.org/pypi/scorecardpy)


This package is python version of R package [scorecard](https://github.com/ShichenXie/scorecard). 
Its goal is to make the development of traditional credit risk scorecard model easier and efficient by providing functions for some common tasks. 
- data partition (`split_df`)
- variable selection (`iv`, `var_filter`)
- weight of evidence (woe) binning (`woebin`, `woebin_plot`, `woebin_adj`, `woebin_ply`)
- scorecard scaling (`scorecard`, `scorecard_ply`)
- performance evaluation (`perf_eva`, `perf_psi`)

## Installation

- Install the release version of `scorecardpy` from [PYPI](https://pypi.org/project/scorecardpy/) with:
```
pip install scorecardpy
```

- Install the latest version of `scorecardpy` from [github](https://github.com/shichenxie/scorecardpy) with:
```
pip install git+git://github.com/shichenxie/scorecardpy.git
```

## Example

This is a basic example which shows you how to develop a common credit risk scorecard:

``` python
# Traditional Credit Scoring Using Logistic Regression
import scorecardpy as sc

# data prepare ------
# load germancredit data
dat = sc.germancredit()

# filter variable via missing rate, iv, identical value rate
dt_s = sc.var_filter(dat, y="creditability")

# breaking dt into train and test
train, test = sc.split_df(dt_s, 'creditability').values()

# woe binning ------
bins = sc.woebin(dt_s, y="creditability")
# sc.woebin_plot(bins)

# binning adjustment
# # adjust breaks interactively
# breaks_adj = sc.woebin_adj(dt_s, "creditability", bins) 
# # or specify breaks manually
breaks_adj = {
    'age.in.years': [26, 35, 40],
    'other.debtors.or.guarantors': ["none", "co-applicant%,%guarantor"]
}
bins_adj = sc.woebin(dt_s, y="creditability", breaks_list=breaks_adj)

# converting train and test into woe values
train_woe = sc.woebin_ply(train, bins_adj)
test_woe = sc.woebin_ply(test, bins_adj)

y_train = train_woe.loc[:,'creditability']
X_train = train_woe.loc[:,train_woe.columns != 'creditability']
y_test = test_woe.loc[:,'creditability']
X_test = test_woe.loc[:,train_woe.columns != 'creditability']

# logistic regression ------
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty='l1', C=0.9, solver='saga', n_jobs=-1)
lr.fit(X_train, y_train)
# lr.coef_
# lr.intercept_

# predicted proability
train_pred = lr.predict_proba(X_train)[:,1]
test_pred = lr.predict_proba(X_test)[:,1]

# performance ks & roc ------
train_perf = sc.perf_eva(y_train, train_pred, title = "train")
test_perf = sc.perf_eva(y_test, test_pred, title = "test")

# score ------
card = sc.scorecard(bins_adj, lr, X_train.columns)
# credit score
train_score = sc.scorecard_ply(train, card, print_step=0)
test_score = sc.scorecard_ply(test, card, print_step=0)

# psi
sc.perf_psi(
  score = {'train':train_score, 'test':test_score},
  label = {'train':y_train, 'test':y_test}
)
```
