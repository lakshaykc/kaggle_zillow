# imports
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
import gc
from sklearn.linear_model import LinearRegression
import random
import datetime as dt
import sys
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

################
################
##  LightGBM changes ##
# V42 - sub_feature: 0.3 -> 0.35 : LB = 0.0643759
# V34 - sub_feature: 0.5 -> 0.42
# V33 - sub_feature: 0.5 -> 0.45 : LB = 0.0643866
# - sub_feature: 0.45 -> 0.3 : LB = 0.0643811 / 0.0643814
################
################

##### READ IN RAW DATA
print( "\nReading data from disk ...")
train = pd.read_csv("../data/train_2016_v2.csv")


################
################
##  XGBoost   ##
################
################

##### RE-READ PROPERTIES FILE
##### (I tried keeping a copy, but the program crashed.)

print( "\nRe-reading properties file ...")
properties = pd.read_csv('../data/properties_2016.csv')

##### PROCESS DATA FOR XGBOOST
print( "\nProcessing data for XGBoost ...")
for c in properties.columns:
    properties[c]=properties[c].fillna(-999)
    if properties[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(properties[c].values))
        properties[c] = lbl.transform(list(properties[c].values))

train_df = train.merge(properties, how='left', on='parcelid')
x_train = train_df.drop(['parcelid', 'logerror','transactiondate'], axis=1)
x_test = properties.drop(['parcelid'], axis=1)

# shape
print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))

# drop out ouliers
train_df=train_df[ train_df.logerror > -0.4 ]
train_df=train_df[ train_df.logerror < 0.419 ]
x_train=train_df.drop(['parcelid', 'logerror','transactiondate'], axis=1)
y_train = train_df["logerror"].values.astype(np.float32)
y_mean = np.mean(y_train)

print('After removing outliers:')
print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))

##### RUN XGBOOST
print("\nSetting up data for XGBoost ...")
# xgboost params
xgb_params = {
    'eta': 0.037,
    'max_depth': 5,
    'subsample': 0.80,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'lambda': 0.8,
    'alpha': 0.4,
    'base_score': y_mean,
    'silent': 1
}

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)

model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=284)

"""
cvresult = xgb.cv(xgb_params, dtrain, num_boost_round=450, nfold=5,
                    metrics=['mae'],early_stopping_rounds=None,as_pandas=True,
                    stratified=True, seed=2,
                    callbacks=[xgb.callback.print_evaluation(show_stdv=False)])

x = cvresult.index.values
y1 = cvresult['test-mae-mean'].values
y2 = cvresult['train-mae-mean'].values

plt.xlabel('Num_boost_rounds')
plt.ylabel('MAE Mean')
plt.title('CV')
plt.plot(x,y1,label='Test')
plt.plot(x,y2,label='train')
plt.legend()
plt.savefig("cv.png")
"""
