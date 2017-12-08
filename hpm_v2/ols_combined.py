# imports
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import gc
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
import random
import datetime as dt
import sys
import pickle

# Parameters
catboost_weight = 0.85
OLS_WEIGHT = 1 - catboost_weight

################
################
##    OLS     ##
################
################

# This section is derived from the1owl's notebook:
#    https://www.kaggle.com/the1owl/primer-for-the-zillow-pred-approach
# which I (Andy Harless) updated and made into a script:
#    https://www.kaggle.com/aharless/updated-script-version-of-the1owl-s-basic-ols
np.random.seed(17)
random.seed(17)

train2016 = pd.read_csv("./input/train_2016_v2.csv", parse_dates=["transactiondate"])
train2017 = pd.read_csv("./input/train_2017.csv", parse_dates=["transactiondate"])
properties2016 = pd.read_csv("./input/properties_2016.csv")
properties2017 = pd.read_csv("./input/properties_2017.csv")
submission = pd.read_csv("./input/sample_submission.csv")

def get_features(df):
    df["transactiondate"] = pd.to_datetime(df["transactiondate"])
    df["transactiondate_year"] = df["transactiondate"].dt.year
    df["transactiondate_month"] = df["transactiondate"].dt.month
    df['transactiondate'] = df['transactiondate'].dt.quarter
    df = df.fillna(-1.0)
    return df

def MAE(y, ypred):
    return np.sum([abs(y[i]-ypred[i]) for i in range(len(y))]) / len(y)

print('Merge Train with Properties ...')
train2016 = pd.merge(train2016, properties2016, how = 'left', on = 'parcelid')
train2017 = pd.merge(train2017, properties2017, how = 'left', on = 'parcelid')

print('Tax Features 2017  ...')
train2017.iloc[:, train2017.columns.str.startswith('tax')] = np.nan
train2017['structuretaxvaluedollarcnt'] = np.nan
train2017['landtaxvaluedollarcnt'] = np.nan

print('Concat Train 2016 & 2017 ...')
train = pd.concat([train2016, train2017], axis = 0)
test = pd.merge(submission[['ParcelId']], properties2016.rename(columns = {'parcelid': 'ParcelId'}), how = 'left', on = 'ParcelId')

print ("Replacing NaN values by -1 !!")
train.fillna(-1., inplace=True)
test.fillna(-1., inplace=True)

y = train['logerror'].values

properties2016 = [] #memory
properties2017 = [] #memory

exc = [train.columns[c] for c in range(len(train.columns)) if train.dtypes[c] == 'O'] + ['logerror','parcelid']
exc_test = [train.columns[c] for c in range(len(train.columns)) if train.dtypes[c] == 'O'] + ['logerror','ParcelId','parcelid']
col = [c for c in train.columns if c not in exc]
col_test = [c for c in train.columns if c not in exc_test]

train = get_features(train[col])
test['transactiondate'] = '2016-01-01' #should use the most common training date
test = get_features(test[col_test])

reg = LinearRegression(n_jobs=-1)
reg.fit(train, y); print('fit...')
print(MAE(y, reg.predict(train)))
train = [];  y = [] #memory

test_dates = ['2016-10-01','2016-11-01','2016-12-01','2017-10-01','2017-11-01','2017-12-01']
test_columns = ['201610','201611','201612','201710','201711','201712']


########################
########################
##  Combine and Save  ##
########################
########################

##### COMBINE PREDICTIONS
print( "\nCombining  predicitons ..." )

ctbst = pd.read_csv("Only_CatBoost.csv")
pred0 = ctbst['201610'].values

print( "\nCombined XGB/LGB/baseline predictions:" )
print( pd.DataFrame(pred0).head() )

print( "\nPredicting with OLS and combining with XGB/LGB/baseline predicitons: ..." )
for i in range(len(test_dates)):
    test['transactiondate'] = test_dates[i]
    pred = OLS_WEIGHT*reg.predict(get_features(test)) + (1-OLS_WEIGHT)*pred0
    submission[test_columns[i]] = [float(format(x, '.4f')) for x in pred]
    print('predict...', i)

print( "\nCombined XGB/LGB/baseline/OLS predictions:" )
print( submission.head() )

##### WRITE THE RESULTS
from datetime import datetime
print( "\nWriting results to disk ..." )
submission.to_csv('sub_ols.csv', index=False)
print( "\nFinished ...")
