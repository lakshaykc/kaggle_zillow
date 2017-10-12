import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from tqdm import tqdm
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
import random
import datetime as dt
from sklearn.preprocessing import LabelEncoder
import gc
from sklearn.linear_model import LinearRegression

s1 = pd.read_pickle("s1.pkl")

X_train = s1[['catb']]
y_train = s1['y_test']
print(X_train.shape, y_train.shape)

test = pd.read_pickle("test_s1.pkl")
X_test = test[["catb"]]

print(X_test.shape)

num_ensembles = 5
y_pred = 0.0
for i in tqdm(range(num_ensembles)):
    # TODO(you): Use CV, tune hyperparameters
    model = CatBoostRegressor(
        iterations=330, learning_rate=0.025,
        depth=6, l2_leaf_reg=3,
        loss_function='MAE',
        eval_metric='MAE',
        random_seed=i)
    model.fit(
        X_train, y_train)
    y_pred += model.predict(X_test)
y_pred /= num_ensembles

catb_pred = y_pred

## ------------------------XGB------------------------------------------

################
################
##  XGBoost   ##
################
################

y_mean = np.mean(y_train)

x_train = X_train
x_test = X_test

##### RUN XGBOOST
print("\nSetting up data for XGBoost ...")
# xgboost params
xgb_params = {
    'eta': 0.03,
    'max_depth': 5,
    'subsample': 0.60,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'lambda': 0.8,
    'alpha': 0.6,
    'base_score': y_mean,
    'silent': 1
}

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)

num_boost_rounds = 200
print("num_boost_rounds="+str(num_boost_rounds))

# train model
print( "\nTraining XGBoost ...")
model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)

print( "\nPredicting with XGBoost ...")
xgb_pred1 = model.predict(dtest)


## ---------------OLS------------------------------------
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

train = pd.read_csv("../data/train_2016_v2.csv", parse_dates=["transactiondate"])
properties = pd.read_csv("../data/properties_2016_updated.csv")
submission = pd.read_csv("../data/sample_submission.csv")
print(len(train),len(properties),len(submission))

def get_features(df):
    df["transactiondate"] = pd.to_datetime(df["transactiondate"])
    df["transactiondate_year"] = df["transactiondate"].dt.year
    df["transactiondate_month"] = df["transactiondate"].dt.month
    df['transactiondate'] = df['transactiondate'].dt.quarter
    df = df.fillna(-1.0)
    return df

def MAE(y, ypred):
    return np.sum([abs(y[i]-ypred[i]) for i in range(len(y))]) / len(y)

train = pd.merge(train, properties, how='left', on='parcelid')
y = train['logerror'].values
test = pd.merge(submission, properties, how='left', left_on='ParcelId', right_on='parcelid')
properties = [] #memory

exc = [train.columns[c] for c in range(len(train.columns)) if train.dtypes[c] == 'O'] + ['logerror','parcelid']
col = [c for c in train.columns if c not in exc]

train = get_features(train[col])
test['transactiondate'] = '2016-01-01' #should use the most common training date
test = get_features(test[col])

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

pred = [catb_pred,xgb_pred1]

for pred0 in pred:

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
    submission.to_csv('sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)
    print( "\nFinished ...")
