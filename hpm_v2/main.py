# Main script
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import gc
from sklearn.linear_model import LinearRegression
import random
import datetime as dt
from datetime import datetime
from data_setup import data_set
from algorithms import algorithms
import time
import sys

def house_pred():

    # Start time
    t1 = time.time()

    # Set data files
    # prop_file = "properties_2016.csv"
    # train_file = "train_2016_v2.csv"
    # df_pkl = "full_df_v2.pkl"
    # sample_file = "sample_submission.csv"

    # Set data files - Testing
    prop_file = "./data_for_testing/prop_sample.csv"
    train_file = "./data_for_testing/train_sample.csv"
    df_pkl = "./data_for_testing/df_full_sample.pkl"
    sample_file = "./data_for_testing/sample_submission_for_testing.csv"

    # Weights of different models
    XGB_WEIGHT = 0.6415
    BASELINE_WEIGHT = 0.0056
    OLS_WEIGHT = 0.0828
    XGB1_WEIGHT = 0.8083  # Weight of first in combination of two XGB models
    BASELINE_PRED = 0.0115   # Baseline based on mean of training data, per Oleg

    # LigthGBM Parameters
    light_gbm_params = {}
    light_gbm_params['max_bin'] = 10
    light_gbm_params['learning_rate'] = 0.0021 # shrinkage_rate
    light_gbm_params['boosting_type'] = 'gbdt'
    light_gbm_params['objective'] = 'regression'
    light_gbm_params['metric'] = 'l1'          # or 'mae'
    light_gbm_params['sub_feature'] = 0.345
    light_gbm_params['bagging_fraction'] = 0.85 # sub_row
    light_gbm_params['bagging_freq'] = 40
    light_gbm_params['num_leaves'] = 512        # num_leaf
    light_gbm_params['min_data'] = 500         # min_data_in_leaf
    light_gbm_params['min_hessian'] = 0.05     # min_sum_hessian_in_leaf
    light_gbm_params['verbose'] = 0
    light_gbm_params['feature_fraction_seed'] = 2
    light_gbm_params['bagging_seed'] = 3

    # XGBoost Parameters
    xgb_params_1 = {
        'eta': 0.037,
        'max_depth': 5,
        'subsample': 0.80,
        'objective': 'reg:linear',
        'eval_metric': 'mae',
        'lambda': 0.8,
        'alpha': 0.4,
        'base_score': 0.,
        'silent': 1
    }

    xgb_params_2 = {
        'eta': 0.033,
        'max_depth': 6,
        'subsample': 0.80,
        'objective': 'reg:linear',
        'eval_metric': 'mae',
        'base_score': 0.,
        'silent': 1
    }

    data = data_set(prop_file,train_file,df_pkl,sample_file)
    # Options
    # 1. threshold removed w knn filled
    # 2. threshold removed w meadian filled
    # 3. threshold removed w -1 filled
    prop = data.data_to_use("prop","threshold removed w -1 filled")
    train = data.data_to_use("train","threshold removed w -1 filled")
    sample = data.data_to_use("sample")

    # algorithms
    # LGB
    algo_lgb = algorithms(prop,train,sample)
    lgb_pred = algo_lgb.run_light_gbm(light_gbm_params)
    # XGB
    num_boost_rounds_1 = 200
    num_boost_rounds_2 = 150
    algo_xgb = algorithms(prop,train,sample)
    xgb_pred1 = algo_xgb.run_xgboost(xgb_params_1,num_boost_rounds_1)
    xgb_pred2 = algo_xgb.run_xgboost(xgb_params_2,num_boost_rounds_2)
    # OLS
    algo_ols = algorithms(prop,train,sample)

    # Test dates
    test_dates = ['2016-10-01','2016-11-01','2016-12-01','2017-10-01','2017-11-01','2017-12-01']
    test_columns = ['201610','201611','201612','201710','201711','201712']

    ##### COMBINE RESULTS

    xgb_pred = XGB1_WEIGHT*xgb_pred1 + (1-XGB1_WEIGHT)*xgb_pred2
    print( "\nCombining XGBoost, LightGBM, and baseline predicitons ..." )
    lgb_weight = (1 - XGB_WEIGHT - BASELINE_WEIGHT) / (1 - OLS_WEIGHT)
    xgb_weight0 = XGB_WEIGHT / (1 - OLS_WEIGHT)
    baseline_weight0 =  BASELINE_WEIGHT / (1 - OLS_WEIGHT)
    pred0 = xgb_weight0*xgb_pred + baseline_weight0*BASELINE_PRED + lgb_weight*lgb_pred

    print( "\nCombined XGB/LGB/baseline predictions:" )
    print( pd.DataFrame(pred0).head() )

    print( "\nPredicting with OLS and combining with XGB/LGB/baseline predicitons: ..." )
    for i in range(len(test_dates)):
        ols_pred = algo_ols.run_OLS(test_dates[i])
        pred = OLS_WEIGHT*ols_pred + (1-OLS_WEIGHT)*pred0
        submission = sample
        submission[test_columns[i]] = [float(format(x, '.4f')) for x in pred]
        print('predict...', i)

    print( "\nCombined XGB/LGB/baseline/OLS predictions:" )
    print( submission.head() )

    # End time
    exec_time = time.time() - t1
    print("Execution Time: %g seconds" %exec_time)

    return submission

def write_results(submission):
    # Write the results to the file
    print( "\nWriting results to disk ..." )
    submission.to_csv('sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)
    print( "\nFinished ...")

if __name__ == '__main__':
    submission = house_pred()
    write_results(submission)
