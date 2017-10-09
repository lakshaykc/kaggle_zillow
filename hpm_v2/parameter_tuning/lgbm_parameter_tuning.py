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
from sklearn.metrics import mean_absolute_error

##### READ IN RAW DATA
print( "\nReading data from disk ...")
prop = pd.read_csv('../../data/properties_2016.csv')
train = pd.read_csv("../../data/train_2016_v2.csv")


################
################
##  LightGBM  ##
################
################

##### PROCESS DATA FOR LIGHTGBM
print( "\nProcessing data for LightGBM ..." )
for c, dtype in zip(prop.columns, prop.dtypes):
    if dtype == np.float64:
        prop[c] = prop[c].astype(np.float32)

df_train = train.merge(prop, how='left', on='parcelid')
df_train.fillna(df_train.median(),inplace = True)

x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc',
                         'propertycountylandusecode', 'fireplacecnt', 'fireplaceflag'], axis=1)

y_train = df_train['logerror'].values
print(x_train.shape, y_train.shape)

train_columns = x_train.columns

for c in x_train.dtypes[x_train.dtypes == object].index.values:
    x_train[c] = (x_train[c] == True)

del df_train; gc.collect()

x_train = x_train.values.astype(np.float32, copy=False)
d_train = lgb.Dataset(x_train, label=y_train)

##### RUN LIGHTGBM
params = {}
params['max_bin'] = 10
params['learning_rate'] = 0.0021 # shrinkage_rate
params['boosting_type'] = 'gbdt'
params['objective'] = 'regression'
params['metric'] = 'l1'          # or 'mae'
params['sub_feature'] = 0.345
params['bagging_fraction'] = 0.85 # sub_row
params['bagging_freq'] = 40
params['num_leaves'] = 512        # num_leaf
params['min_data'] = 500         # min_data_in_leaf
params['min_hessian'] = 0.05     # min_sum_hessian_in_leaf
params['verbose'] = 0
params['feature_fraction_seed'] = 2
params['bagging_seed'] = 3

#print("\nFitting LightGBM model ...")
#clf = lgb.train(params, d_train, 430)

cvresult = lgb.cv(params, d_train, num_boost_round=10, nfold=5,
                    metrics='mae',early_stopping_rounds=None,
                    stratified=True, verbose_eval=50,seed=2)

df = pd.DataFrame.from_dict(cvresult)
x = df.index.values
y1 = df['l1-mean'].values
y2 = df['l1-stdv'].values

plt.xlabel('Num_boost_rounds')
plt.ylabel('MAE')
plt.title('CV')
plt.plot(x,y1,label='Test-MAE Mean')
plt.plot(x,y2,label='Test-MAE STDV')
plt.legend()
plt.savefig("lgbm_cv_5_fold.png")

"""
model  = lbg.XGBRegressor()

clf = GridSearchCV(model, xgb_params, cv=3, n_jobs = -1,
                        scoring='neg_mean_absolute_error',verbose = 4)


clf.fit(dtrain_x,dtrain_y)

print clf.best_score_
print clf.best_params_
df = pd.DataFrame.from_dict(clf.cv_results_)
df.to_pickle("xgb_gridcv_4.pkl")
"""
