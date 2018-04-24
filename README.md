# scorecardpy

This package is python version of R package scorecard. It makes the development of credit risk scorecard easily and efficiently by providing functions as follows: 
- information value (iv), 
- variable filter (var_filter), 
- optimal woe binning (woebin, woebin_ply, woebin_plot, woebin_adj), 
- scorecard scaling (scorecard, scorecard_ply) 
- and performace evaluation (perf_eva, perf_psi).

## Installation


## Example

This is a basic example which shows you how to develop a common credit risk scorecard:

``` python
# Traditional Credit Scoring Using Logistic Regression
from scorecardpy import *

# data prepare ------
# load germancredit data
dat = germancredit()

# filter variable via missing rate, iv, identical value rate
dt_s = var_filter(dat, y="creditability")

# breaking dt into train and test
train, test = split_df(dt_s, 'creditability').values()

# woe binning ------
bins = woebin(dt_s, y="creditability")
# woebin_plot(bins)

# binning adjustment
# # adjust breaks interactively
# breaks_adj = woebin_adj(dt_s, "creditability", bins) 
# # or specify breaks manually
breaks_adj = {
    'age.in.years': [26, 35, 40],
    'other.debtors.or.guarantors': ["none", "co-applicant%,%guarantor"]
}
bins_adj = woebin(dt_s, y="creditability", breaks_list=breaks_adj)

# converting train and test into woe values
train_woe = woebin_ply(train, bins_adj)
test_woe = woebin_ply(test, bins_adj)

y_train = train_woe[['creditability']].loc[:,'creditability']
X_train = train_woe.loc[:,train_woe.columns != 'creditability']
y_test = test_woe[['creditability']].loc[:,'creditability']
X_test = test_woe.loc[:,train_woe.columns != 'creditability']

# logistic regression ------
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty='l1', C=0.9, solver='saga')
lr.fit(X_train, y_train)
# lr.coef_
# lr.intercept_

# predicted proability
pred_train = lr.predict_proba(X_train)[:,1]
pred_test = lr.predict_proba(X_test)[:,1]

# performance ks & roc ------
perf_train = perf_eva(y_train, pred_train, title = "train")
perf_test = perf_eva(y_test, pred_test, title = "test")

# score ------
card = scorecard(bins_adj, lr, X_train.columns)
# credit score
train_score = scorecard_ply(train, card, print_step=0)
test_score = scorecard_ply(test, card, print_step=0)

# psi
psirt = perf_psi(
  score = {'train':train_score, 'test':test_score},
  label = {'train':y_train, 'test':y_test},
  x_limits = [250, 750],
  x_tick_break = 50
)
```
