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
import sys
import pickle

class algorithms(object):

    def __init__(self,prop,train,train_path=None):
        self.prop = prop
        self.train = train
        self.train_path = train_path # only required for OLS

    def train_light_gbm(self,light_gbm_params):
        # Returns the trained model

        # Process data to save memory
        for c, dtype in zip(self.prop.columns, self.prop.dtypes):
            if dtype == np.float64:
                self.prop[c] = self.prop[c].astype(np.float32)

        params = light_gbm_params
        print( "\nProcessing data for LightGBM ..." )
        df_train = self.train.merge(self.prop, how='left', on='parcelid')
        df_train.fillna(df_train.median(),inplace = True)

        f_unused = ['parcelid', 'logerror', 'transactiondate',
                    'propertyzoningdesc','propertycountylandusecode',
                    'fireplacecnt', 'fireplaceflag']

        f_to_remove = [x for x in f_unused if x in df_train.columns.tolist()]
        x_train = df_train.drop(f_to_remove, axis=1)
        y_train = df_train['logerror'].values
        print(x_train.shape, y_train.shape)

        self.train_columns = x_train.columns

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

        return clf

    def predict_light_gbm(self,clf,sample):

        print("\nPrepare for LightGBM prediction ...")
        print("   ...")
        try:
            sample['parcelid'] = sample['ParcelId']
        except:
            print("Not using submission sample file")
            pass

        print(" LGBM:  Merge with property data ...")
        df_test = sample.merge(self.prop, on='parcelid', how='left')
        print("   ...")

        x_test = df_test[self.train_columns]

        del df_test; gc.collect()
        print("LGBM:   Preparing x_test...")
        for c in x_test.dtypes[x_test.dtypes == object].index.values:
            x_test[c] = (x_test[c] == True)
        print("   ...")
        x_test = x_test.values.astype(np.float32, copy=False)
        print("Test shape :", x_test.shape)

        print("\n LGBM prediction ...")
        p_test = clf.predict(x_test)

        del x_test; gc.collect()

        return p_test

    def train_xgboost(self,xgb_params,num_boost_rounds):

        properties = self.prop
        train = self.train

        print( "\nProcessing data for XGBoost ...")
        for c in properties.columns:
            properties[c]=properties[c].fillna(-1)
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

        del x_train
        del dtrain
        del dtest
        gc.collect()

        return model

    def predict_xgb(self,model,test_parcelid_df):

        properties = self.prop
        for c in properties.columns:
            properties[c]=properties[c].fillna(-1)
            if properties[c].dtype == 'object':
                lbl = LabelEncoder()
                lbl.fit(list(properties[c].values))
                properties[c] = lbl.transform(list(properties[c].values))

        x_test = properties.drop(['parcelid'], axis=1)
        dtest = xgb.DMatrix(x_test)

        print( "\nPredicting with XGBoost ...")
        xgb_pred = model.predict(dtest)

        del x_test
        del properties
        del dtest
        gc.collect()

        return xgb_pred


    def train_OLS(self,properties):

        train_tmp = pd.read_csv(self.train_path, parse_dates=["transactiondate"])

        print(len(train_tmp),len(properties))

        train_tmp = pd.merge(train_tmp, properties, how='left', on='parcelid')
        y = train_tmp['logerror'].values
        properties = [] #memory

        exc = [train_tmp.columns[c] for c in range(len(train_tmp.columns)) if train_tmp.dtypes[c] == 'O'] + ['logerror','parcelid']
        col = [c for c in train_tmp.columns if c not in exc]
        if "transactiondate" not in col:
            col.append("transactiondate")

        train_tmp = self.get_features(train_tmp[col])

        reg = LinearRegression(n_jobs=-1)
        reg.fit(train_tmp, y);
        pred_fit = reg.predict(train_tmp)
        #print(self.MAE(y, pred_fit))
        train_tmp = []; y = [] #memory

        return reg

    def predict_OLS(self,reg,test_date,sample,properties):
        submission = sample
        train_tmp = pd.read_csv(self.train_path, parse_dates=["transactiondate"])
        train_tmp = pd.merge(train_tmp, properties, how='left', on='parcelid')

        try:
            test = pd.merge(submission, properties, how='left', left_on='ParcelId', right_on='parcelid')
        except:
            test = pd.merge(submission, properties, how='left', left_on='parcelid', right_on='parcelid')

        properties = [] #memory

        exc = [train_tmp.columns[c] for c in range(len(train_tmp.columns)) if train_tmp.dtypes[c] == 'O'] + ['logerror','parcelid']
        col = [c for c in train_tmp.columns if c not in exc]
        if "transactiondate" not in col:
            col.append("transactiondate")

        test['transactiondate'] = '2016-01-01' #should use the most common training date
        test = self.get_features(test[col])

        test["transactiondate"] = test_date
        ols_pred = reg.predict(self.get_features(test))
        train_tmp = [] # memory
        return ols_pred

    def get_features(self,df):
        #print df["transactiondate"]
        df["transactiondate"] = pd.to_datetime(df["transactiondate"])
        df["transactiondate_year"] = df["transactiondate"].dt.year
        df["transactiondate_month"] = df["transactiondate"].dt.month
        df['transactiondate'] = df['transactiondate'].dt.quarter
        df = df.fillna(-1.0)
        return df

    def MAE(self,y, ypred):
        # Mean absolute error
        return np.sum([abs(y[i]-ypred[i]) for i in range(len(y))]) / len(y)
