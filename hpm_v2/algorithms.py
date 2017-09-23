# Algorithms

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import gc
from sklearn.linear_model import LinearRegression
import random
import datetime as dt

class algorithms(object):

    def __init__(self,prop,train,sample):
        self.prop = prop
        self.train = train
        self.sample = sample

    def run_light_gbm(self,light_gbm_params):
        params = light_gbm_params
        print( "\nProcessing data for LightGBM ..." )
        for c, dtype in zip(self.prop.columns, self.prop.dtypes):
            if dtype == np.float64:
                self.prop[c] = self.prop[c].astype(np.float32)

        df_train = self.train

        f_unused = ['parcelid', 'logerror', 'transactiondate',
                    'propertyzoningdesc','propertycountylandusecode',
                    'fireplacecnt', 'fireplaceflag']

        f_to_remove = [x for x in f_unused if x in df_train.columns.tolist()]
        x_train = df_train.drop(f_to_remove, axis=1)
        y_train = df_train['logerror'].values
        print(x_train.shape, y_train.shape)

        train_columns = x_train.columns

        for c in x_train.dtypes[x_train.dtypes == object].index.values:
            x_train[c] = (x_train[c] == True)

        del df_train; gc.collect()

        x_train = x_train.values.astype(np.float32, copy=False)
        d_train = lgb.Dataset(x_train, label=y_train)

        # Run LightGBM
        print("\nFitting LightGBM model ...")
        clf = lgb.train(params, d_train, 430)

        del d_train; gc.collect()
        del x_train; gc.collect()

        print("\nPrepare for LightGBM prediction ...")
        print("   ...")
        self.sample['parcelid'] = self.sample['ParcelId']
        print("   Merge with property data ...")
        df_test = self.sample.merge(self.prop, on='parcelid', how='left')

        print("   ...")

        print("   ...")
        x_test = df_test[train_columns]
        print("   ...")
        del df_test; gc.collect()
        print("   Preparing x_test...")
        for c in x_test.dtypes[x_test.dtypes == object].index.values:
            x_test[c] = (x_test[c] == True)
        print("   ...")
        x_test = x_test.values.astype(np.float32, copy=False)
        print("Test shape :", x_test.shape)

        print("\nStart LightGBM prediction ...")
        p_test = clf.predict(x_test)

        del x_test; gc.collect()

        print( "\nUnadjusted LightGBM predictions:" )
        print( pd.DataFrame(p_test).head() )

        return p_test

    def run_xgboost(self,xgb_params,num_boost_rounds):

        properties = self.prop
        train_df = self.train

        print( "\nProcessing data for XGBoost ...")
        for c in properties.columns:
            properties[c]=properties[c].fillna(-1)
            if properties[c].dtype == 'object':
                lbl = LabelEncoder()
                lbl.fit(list(properties[c].values))
                properties[c] = lbl.transform(list(properties[c].values))

        for c in train_df.columns:
            train_df[c]=train_df[c].fillna(-1)
            if train_df[c].dtype == 'object' and c != "transactiondate":
                lbl = LabelEncoder()
                lbl.fit(list(train_df[c].values))
                train_df[c] = lbl.transform(list(train_df[c].values))

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
        # XGBoost Parameters
        xgb_params['base_score'] = y_mean

        print('After removing outliers:')
        print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))

        # Run XGBoost
        dtrain = xgb.DMatrix(x_train, y_train)
        dtest = xgb.DMatrix(x_test)

        print("num_boost_rounds="+str(num_boost_rounds))

        # train model
        print( "\nTraining XGBoost ...")
        model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)

        print( "\nPredicting with XGBoost ...")
        xgb_pred = model.predict(dtest)

        print( "\nFirst XGBoost predictions:" )
        print( pd.DataFrame(xgb_pred).head() )

        return xgb_pred

        del df_train
        del x_train
        del x_test
        del properties
        del dtest
        del dtrain
        gc.collect()




    def run_OLS(self,test_date):

        properties = self.prop
        submission = self.sample
        train_tmp = self.train.copy()

        print(len(train_tmp),len(properties),len(submission))

        y = train_tmp["logerror"].values
        test = pd.merge(submission, properties, how='left', left_on='ParcelId', right_on='parcelid')
        properties = [] #memory

        exc = [train_tmp.columns[c] for c in range(len(train_tmp.columns)) if train_tmp.dtypes[c] == 'O'] + ['logerror','parcelid']
        col = [c for c in train_tmp.columns if c not in exc] + ['transactiondate']

        train_tmp = self.get_features(train_tmp[col])
        test['transactiondate'] = '2016-01-01' #should use the most common training date
        test = self.get_features(test[col])

        reg = LinearRegression(n_jobs=-1)
        reg.fit(train_tmp, y); print('fit...')
        pred_fit = reg.predict(train_tmp)
        print(self.MAE(y, pred_fit))
        train_tmp = []; y = [] #memory

        test["transactiondate"] = test_date
        ols_pred = reg.predict(self.get_features(test))

        return ols_pred

    def get_features(self,df):
        df["transactiondate"] = pd.to_datetime(df["transactiondate"])
        df["transactiondate_year"] = df["transactiondate"].dt.year
        df["transactiondate_month"] = df["transactiondate"].dt.month
        df['transactiondate'] = df['transactiondate'].dt.quarter
        df = df.fillna(-1.0)
        return df

    def MAE(self,y, ypred):
        # Mean absolute error
        print("Quality of fit (MAE):")
        return np.sum([abs(y[i]-ypred[i]) for i in range(len(y))]) / len(y)
