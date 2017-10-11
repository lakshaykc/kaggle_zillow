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

train_df = pd.read_csv('../data/train_2016_v2.csv', parse_dates=['transactiondate'], low_memory=False)
test_df = pd.read_csv('../data/sample_submission.csv', low_memory=False)
properties = pd.read_csv('../data/properties_2016.csv', low_memory=False)
# field is named differently in submission
test_df['parcelid'] = test_df['ParcelId']

#-----K-fold----------------------
k = 10
kf = KFold(n_splits=k,shuffle=True,random_state = 2)


####------------------CATBOOST---------------------------------------------

# similar to the1owl
def add_date_features(df):
    try:
        df["transaction_year"] = df["transactiondate"].dt.year
        df["transaction_month"] = df["transactiondate"].dt.month
        df["transaction_day"] = df["transactiondate"].dt.day
        df["transaction_quarter"] = df["transactiondate"].dt.quarter
        df.drop(["transactiondate"], inplace=True, axis=1)

    except:
        pass

    return df

train_df = add_date_features(train_df)
train_df = train_df.merge(properties, how='left', on='parcelid')
test_df = test_df.merge(properties, how='left', on='parcelid')
print("Train: ", train_df.shape)
print("Test: ", test_df.shape)

missing_perc_thresh = 0.98
exclude_missing = []
num_rows = train_df.shape[0]
for c in train_df.columns:
    num_missing = train_df[c].isnull().sum()
    if num_missing == 0:
        continue
    missing_frac = num_missing / float(num_rows)
    if missing_frac > missing_perc_thresh:
        exclude_missing.append(c)
print("We exclude: %s" % exclude_missing)
print(len(exclude_missing))

# exclude where we only have one unique value :D
exclude_unique = []
for c in train_df.columns:
    num_uniques = len(train_df[c].unique())
    if train_df[c].isnull().sum() != 0:
        num_uniques -= 1
    if num_uniques == 1:
        exclude_unique.append(c)
print("We exclude: %s" % exclude_unique)
print(len(exclude_unique))

exclude_other = ['parcelid', 'logerror']  # for indexing/training only
# do not know what this is LARS, 'SHCG' 'COR2YY' 'LNR2RPD-R3' ?!?
exclude_other.append('propertyzoningdesc')
train_features = []
for c in train_df.columns:
    if c not in exclude_missing \
       and c not in exclude_other and c not in exclude_unique:
        train_features.append(c)
print("We use these for training: %s" % train_features)
print(len(train_features))

cat_feature_inds = []
cat_unique_thresh = 1000
for i, c in enumerate(train_features):
    num_uniques = len(train_df[c].unique())
    if num_uniques < cat_unique_thresh \
       and not 'sqft' in c \
       and not 'cnt' in c \
       and not 'nbr' in c \
       and not 'number' in c:
        cat_feature_inds.append(i)

print("Cat features are: %s" % [train_features[ind] for ind in cat_feature_inds])

# some out of range int is a good choice
train_df_lgbm = train_df.copy()
test_df_lgbm = test_df.copy()
train_df.fillna(-999, inplace=True)
test_df.fillna(-999, inplace=True)

#---------------BEGIN Folding--------------------------------------------------
# Create a empty dataframe to store outputs in s1
cols_s = ['catb','xgb1','xgb2','lgbm','y_test']
s1 = pd.DataFrame(columns=cols_s)
s1.to_pickle("s1_backup.pkl")

count = 1

for train_index,test_index in kf.split(train_df):

    X_train = train_df[train_features].iloc[train_index]
    y_train = train_df['logerror'].iloc[train_index]
    print(X_train.shape, y_train.shape)

    X_test = train_df[train_features].iloc[test_index]
    y_test = train_df["logerror"].iloc[test_index]
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
            X_train, y_train,
            cat_features=cat_feature_inds)
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

    for c in x_train.columns:
        if x_train[c].dtype == 'object':
            lbl = LabelEncoder()
            lbl.fit(list(x_train[c].values))
            x_train[c] = lbl.transform(list(x_train[c].values))

    for c in x_test.columns:
        if x_test[c].dtype == 'object':
            lbl = LabelEncoder()
            lbl.fit(list(x_test[c].values))
            x_test[c] = lbl.transform(list(x_test[c].values))

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

    num_boost_rounds = 235
    print("num_boost_rounds="+str(num_boost_rounds))

    # train model
    print( "\nTraining XGBoost ...")
    model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)

    print( "\nPredicting with XGBoost ...")
    xgb_pred1 = model.predict(dtest)

    print( "\nFirst XGBoost predictions:" )

    ##### RUN XGBOOST AGAIN
    print("\nSetting up data for XGBoost ...")
    # xgboost params
    xgb_params = {
        'eta': 0.033,
        'max_depth': 6,
        'subsample': 0.80,
        'objective': 'reg:linear',
        'lambda': 0.8,
        'alpha': 0.2,
        'eval_metric': 'mae',
        'base_score': y_mean,
        'silent': 1
    }

    num_boost_rounds = 170
    print("num_boost_rounds="+str(num_boost_rounds))

    print( "\nTraining XGBoost again ...")
    model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)

    print( "\nPredicting with XGBoost again ...")
    xgb_pred2 = model.predict(dtest)

    print( "\nSecond XGBoost predictions:" )
    print( pd.DataFrame(xgb_pred2).head() )

    try:
        del x_train
        del x_test
        del dtest
        del dtrain
        gc.collect()
        del properties
        gc.collect()

    except:
        pass

    ###------------------LGBM--------------------------------------------------
    train_df_lgbm.fillna(train_df_lgbm.median(), inplace=True)
    test_df_lgbm.fillna(test_df_lgbm.median(), inplace=True)

    x_train = train_df_lgbm[train_features].iloc[train_index]
    y_train = train_df_lgbm["logerror"].iloc[train_index]
    print(x_train.shape, y_train.shape)

    x_test = train_df_lgbm[train_features].iloc[test_index]

    to_remove = ['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc',
                'propertycountylandusecode', 'fireplacecnt', 'fireplaceflag']

    for c in to_remove:
        print c
        try:
            x_train.drop([c],inplace=True,axis=1)
            x_test.drop([c],inplace=True,axis=1)
            print("removed %s" %c)
        except:
            pass

    print x_train.columns.tolist()

    x_train = x_train.values.astype(np.float32, copy=False)
    d_train = lgb.Dataset(x_train, label=y_train)

    ##### RUN LIGHTGBM
    params = {}
    params['max_bin'] = 7
    params['learning_rate'] = 0.0021 # shrinkage_rate
    params['boosting_type'] = 'gbdt'
    params['objective'] = 'regression'
    params['metric'] = 'l1'          # or 'mae'
    params['sub_feature'] = 0.345
    params['bagging_fraction'] = 0.7 # sub_row
    params['bagging_freq'] = 20
    params['num_leaves'] = 512        # num_leaf
    params['min_data'] = 300         # min_data_in_leaf
    params['min_hessian'] = 0.05     # min_sum_hessian_in_leaf
    params['verbose'] = 0
    params['feature_fraction_seed'] = 2
    params['bagging_seed'] = 3

    print("\nFitting LightGBM model ...")
    clf = lgb.train(params, d_train, 342)

    del d_train; gc.collect()
    del x_train; gc.collect()

    x_test = x_test.values.astype(np.float32, copy=False)
    print("Test shape :", x_test.shape)

    print("\nStart LightGBM prediction ...")
    lgbm_pred = clf.predict(x_test)

    del x_test; gc.collect()

    #-------------------- Combine Output -----------------------------------

    print("Writing dataframes %i" %count)
    L1_out = np.array([catb_pred,xgb_pred1,xgb_pred2,lgbm_pred,y_test])
    L1_out = np.transpose(L1_out)
    L1_out_df = pd.DataFrame(L1_out,columns=cols_s)
    s1 = s1.append(L1_out_df,ignore_index=True)
    print s1.head()
    s1_backup = pd.read_pickle("s1_backup.pkl")
    s1_backup = s1_backup.append(L1_out_df,ignore_index=True)
    print s1_backup.head()
    s1_backup.to_pickle("s1_backup.pkl")

    count += 1
s1.to_pickle("s1.pkl")
