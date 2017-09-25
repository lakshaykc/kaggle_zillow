# Main script
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import gc
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import KFold, train_test_split
import random
import datetime as dt
from datetime import datetime
from data_setup import data_set
from algorithms import algorithms
import time
import sys

class house_model(object):

    def __init__(self,prop_file1,train_file1,df_pkl1,sample_file1,columns_df1,log_entry,prop_option,train_option):

        # Note: The redudancy in variables names for using self is because
        # of laziness to make self. changes all the way down. In the interest of
        # time, I will make the change later.

        # Insatantiate files
        self.prop_file = prop_file1
        prop_file = self.prop_file
        self.train_file = train_file1
        train_file = self.train_file
        self.df_pkl = df_pkl1
        df_pkl = self.df_pkl
        self.sample_file = sample_file1
        sample_file = self.sample_file
        self.columns_df = columns_df1
        columns_df = self.columns_df
        self.columns = columns_df["columns"].values.tolist()
        self.log_entry = log_entry
        self.prop_option = prop_option
        self.train_option = train_option

    def house_train(self,weights,light_gbm_params,xgb_params_1,xgb_params_2,type = "test"):
        # Start time
        t1 = time.time()

        # Weights of different models
        self.XGB_WEIGHT = weights[0]
        self.BASELINE_WEIGHT = weights[1]
        self.OLS_WEIGHT = weights[2]
        self.XGB1_WEIGHT = weights[3]  # Weight of first in combination of two XGB models
        self.BASELINE_PRED = weights[4]   # Baseline based on mean of training data, per Oleg

        data = data_set(prop_file,train_file,df_pkl,sample_file,self.columns)

        self.log_entry[2] = self.prop_option
        self.log_entry[3] = self.train_option

        self.prop = data.data_to_use("prop",prop_option)
        prop = self.prop
        self.train_set_org = data.data_to_use("train",train_option)
        train_set_org = self.train_set_org

        if type == "submission":
            test_set = data.data_to_use("sample")
            train_set = self.train_set_org

        elif type == "test":
            X_train,X_test = train_test_split(train_set_org.index,test_size=0.1)
            train_set = train_set_org.iloc[X_train]
            test_set = train_set_org[["parcelid","transactiondate","logerror"]].iloc[X_test]

        # KFold Cross Validation
        [m,n]  = train_set.shape
        k = 5
        section_size = m//k
        kf = KFold(n_splits=k)
        result_metrics = np.zeros((k,3))
        count = 0
        models = []

        for train_index, test_index in kf.split(train_set):
            train = train_set.iloc[train_index]
            sample = train_set[["parcelid","transactiondate"]].iloc[test_index]
            y = train_set["logerror"].iloc[test_index].values

            # algorithms
            # LGB
            self.algo_lgb = algorithms(prop,train)
            lgb_model = self.algo_lgb.train_light_gbm(light_gbm_params)
            lgb_pred = self.algo_lgb.predict_light_gbm(lgb_model,sample)

            # XGB
            num_boost_rounds_1 = 200
            num_boost_rounds_2 = 150
            self.algo_xgb = algorithms(prop,train)
            test_parcelid_df = sample[["parcelid"]]

            xgb_model1 = self.algo_xgb.train_xgboost(xgb_params_1,num_boost_rounds_1)
            xgb_pred1 = self.algo_xgb.predict_xgb(xgb_model1,test_parcelid_df)

            xgb_model2 = self.algo_xgb.train_xgboost(xgb_params_2,num_boost_rounds_2)
            xgb_pred2 = self.algo_xgb.predict_xgb(xgb_model2,test_parcelid_df)

            # OLS
            self.algo_ols = algorithms(prop,train)

            ##### COMBINE RESULTS
            xgb_pred = self.XGB1_WEIGHT*xgb_pred1 + (1-self.XGB1_WEIGHT)*xgb_pred2
            print( "\nCombining XGBoost, LightGBM, and baseline predicitons ..." )
            lgb_weight = (1 - self.XGB_WEIGHT - self.BASELINE_WEIGHT) / (1 - self.OLS_WEIGHT)
            xgb_weight0 = self.XGB_WEIGHT / (1 - self.OLS_WEIGHT)
            baseline_weight0 =  self.BASELINE_WEIGHT / (1 - self.OLS_WEIGHT)
            pred0 = xgb_weight0*xgb_pred + baseline_weight0*self.BASELINE_PRED + lgb_weight*lgb_pred


            # Predict with OLS
            print( "\nPredicting with OLS and combining with XGB/LGB/baseline predicitons: ..." )
            test_date = sample["transactiondate"].values
            ols_model = self.algo_ols.train_OLS()
            ols_pred = self.algo_ols.predict_OLS(ols_model,test_date,sample)
            pred = self.OLS_WEIGHT*ols_pred + (1-self.OLS_WEIGHT)*pred0

            models.append([lgb_model,xgb_model1,xgb_model2,ols_model])

            # Calculate metrics
            model_mae = mae(y,pred)
            model_mse = mse(y,pred)
            model_r2 = r2_score(y,pred)

            result_metrics[count,:] = np.array([model_mae,model_mse,model_r2])
            count += 1

        score = np.mean(result_metrics,axis=0)
        print("MAE: %g, MSE: %g, R2: %g" %(score[0],score[1],score[2]))

        # End time
        exec_time = time.time() - t1
        print("Execution Time: %g seconds" %exec_time)

        self.log_entry[4] = models
        self.log_entry[5] = score[0]
        self.log_entry[6] = score[1]
        self.log_entry[7] = score[2]
        self.log_entry[-2] = exec_time

        return score, models, test_set, self.log_entry

    def house_pred(self,models,sample,type = "test"):

        t2 = time.time()

        [m,n] = sample.shape

        if type == "test":
            y = sample["logerror"].values
            sample = sample.drop(["logerror"],axis=1)
            test_parcelid_df = sample[["parcelid"]]

        elif type == "submission":
            y = np.zeros(m)
            sample["parcelid"] = sample["ParcelId"]
            test_parcelid_df = sample[["parcelid"]]

            # Test dates
            test_dates = ['2016-10-01','2016-11-01','2016-12-01','2017-10-01','2017-11-01','2017-12-01']
            test_columns = ['201610','201611','201612','201710','201711','201712']

        n = len(models)
        count = 0
        for i in range(n):
            lgb_model = models[i][0]
            xgb_model1 = models[i][1]
            xgb_model2 = models[i][2]
            ols_model = models[i][3]

            lgb_pred = self.algo_lgb.predict_light_gbm(lgb_model,sample)

            xgb_pred1 = self.algo_xgb.predict_xgb(xgb_model1,test_parcelid_df)
            xgb_pred2 = self.algo_xgb.predict_xgb(xgb_model2,test_parcelid_df)

            ##### COMBINE RESULTS
            xgb_pred = self.XGB1_WEIGHT*xgb_pred1 + (1-self.XGB1_WEIGHT)*xgb_pred2
            print( "\nCombining XGBoost, LightGBM, and baseline predicitons ..." )
            lgb_weight = (1 - self.XGB_WEIGHT - self.BASELINE_WEIGHT) / (1 - self.OLS_WEIGHT)
            xgb_weight0 = self.XGB_WEIGHT / (1 - self.OLS_WEIGHT)
            baseline_weight0 =  self.BASELINE_WEIGHT / (1 - self.OLS_WEIGHT)
            pred0 = xgb_weight0*xgb_pred + baseline_weight0*self.BASELINE_PRED + lgb_weight*lgb_pred

            print( "\nCombined XGB/LGB/baseline predictions:" )
            print( pd.DataFrame(pred0).head() )

            if type == "test":
                print( "\nPredicting with OLS and combining with XGB/LGB/baseline predicitons: ..." )
                test_date = sample["transactiondate"].values
                ols_pred = self.algo_ols.predict_OLS(ols_model,test_date,sample)
                pred = self.OLS_WEIGHT*ols_pred + (1-self.OLS_WEIGHT)*pred0

                # Calculate metrics
                model_mae = mae(y,pred)
                model_mse = mse(y,pred)
                model_r2 = r2_score(y,pred)

                print("MAE: %g, MSE: %g, R2: %g" %(model_mae,model_mse,model_r2))

                return [model_mae,model_mse,model_r2]

            if type == "submission":
                print( "\nPredicting with OLS and combining with XGB/LGB/baseline predicitons: ..." )
                for i in range(len(test_dates)):
                    ols_model = self.algo_ols.train_OLS()
                    ols_pred = self.algo_ols.predict_OLS(ols_model,test_dates[i],sample)
                    pred = self.OLS_WEIGHT*ols_pred + (1-self.OLS_WEIGHT)*pred0
                    submission = sample
                    submission[test_columns[i]] = [float(format(x, '.4f')) for x in pred]
                    print('predict...', i)

                    print( "\nCombined XGB/LGB/baseline/OLS predictions:" )
                    print( submission.head() )

                submission  = submission.drop(["parcelid"],axis=1)
                return submission


    def write_results(self,submission):
        # Write the results to the file
        print( "\nWriting results to disk ..." )
        submission.to_csv('sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)
        print( "\nFinished ...")

if __name__ == '__main__':

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
    columns_df = pd.read_csv("columns.csv")

    # Options
    option_1 = "threshold removed w knn filled"
    option_2 =  "threshold removed w median filled"
    option_3 = "threshold removed w -1 filled"

    # INPUTS-------------------------------------------------------------------
    prop_option = option_1
    train_option = option_1

    # Run type
    type = "test"

    # Weights of different models
    XGB_WEIGHT = 0.6415
    BASELINE_WEIGHT = 0.0056
    OLS_WEIGHT = 0.0828
    XGB1_WEIGHT = 0.8083  # Weight of first in combination of two XGB models
    BASELINE_PRED = 0.0115   # Baseline based on mean of training data, per Oleg

    weights = [XGB_WEIGHT,BASELINE_WEIGHT,OLS_WEIGHT,XGB1_WEIGHT,BASELINE_PRED]

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
    light_gbm_params['min_data'] = 10         # min_data_in_leaf
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
        'lambda': 1.0,
        'alpha': 0,
        'base_score': 0.,
        'silent': 1
    }


    # END INPUTS---------------------------------------------------------------

    # Initialize dataframes

    # col_names = ["ID","Run_type","Prop_option","train_option","Models",...
    # "CV_score: MAE,MSE,R2","Test_score","Status"]
    # log_df = pd.DataFrame(columns = col_names)
    log_df = pd.read_pickle("log_df.pkl")

    # Set unique ID for each run
    id = int(datetime.now().strftime('%Y%m%d%H%M%S'))

    log_entry = [0]*len(log_df.columns.tolist())
    log_entry[0] = id
    log_entry[-3] = 0. # Status

    # Note about the data and the run_xgboost
    note = "Test on sample data"
    log_entry[-1] = note

    # Traing the models
    log_entry[1] = type
    house = house_model(prop_file,train_file,df_pkl,sample_file,columns_df,
                        log_entry,prop_option,train_option)

    score, models, test_set, log_entry = house.house_train(weights,
                                        light_gbm_params,xgb_params_1,
                                        xgb_params_2,type=type)

    print("Validation Score:")
    print("MAE            MSE           R2")
    print score

    # Final Prediction
    out = house.house_pred(models,test_set,type = type)

    if type == "test":
        print("Score:")
        print out
        log_entry[8] = out[0]
        log_entry[9] = out[1]
        log_entry[10] = out[2]

    if type == "submission":
        house.write_results(out)
        log_entry[8] = 0.
        log_entry[9] = 0.
        log_entry[10] = 0.

    log_entry[-3] = 1. # Status = 1, if run is successful

    # Update logging dataframes

    print("Writing log dataframes")
    # Run Log
    log_df =  log_df.append(pd.DataFrame(np.array([log_entry]), columns=log_df.columns.tolist()), ignore_index=True)
    log_df.to_pickle("log_df.pkl")

    # Parameters log
    lgbm_params_df = pd.read_pickle("lgbm_params_df.pkl")
    lgbm_params_df = lgbm_params_df.append(pd.DataFrame([light_gbm_params]),ignore_index=True)
    lgbm_params_df["ID"].iloc[-1] = id
    lgbm_params_df.to_pickle("lgbm_params_df.pkl")

    xgb_params_1_df = pd.read_pickle("xgb_params_1_df.pkl")
    xgb_params_1_df = xgb_params_1_df.append(pd.DataFrame([xgb_params_1]),ignore_index=True)
    xgb_params_1_df["ID"].iloc[-1] = id
    xgb_params_1_df.to_pickle("xgb_params_1_df.pkl")

    xgb_params_2_df = pd.read_pickle("xgb_params_2_df.pkl")
    xgb_params_2_df = xgb_params_2_df.append(pd.DataFrame([xgb_params_2]),ignore_index=True)
    xgb_params_2_df["ID"].iloc[-1] = id
    xgb_params_2_df.to_pickle("xgb_params_2_df.pkl")

    #backup_pkl_path = ""
    log_df.to_pickle("./back_up_logs/backup_log" + str(id) + ".pkl")

    print("FINISHED---------------------------------------------")
