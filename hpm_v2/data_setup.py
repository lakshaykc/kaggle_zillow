# Data Module

import numpy as np
import pandas as pd
import random
import datetime as dt
import gc

class data_set(object):

    def __init__(self,prop_file,train_file,df_pkl,sample_file,columns):

        print("Reading properties file.......\n")
        self.prop_raw = pd.read_csv(prop_file)
        self.prop_raw = self.prop_raw[columns]
        print("Reading train file.......\n")
        self.train_raw = pd.read_csv(train_file)
        print("Reading full df pkl.......\n")
        self.df_full = pd.read_pickle(df_pkl)
        print("Reading sample file.......\n")
        self.sample = pd.read_csv(sample_file)

        self.n_rows,self.n_columns = self.df_full.shape
        self.train_w_prop_filled = self.df_full[0:90275]

        # Prop df with filled data without the last 11437 empty rows
        self.prop_filled_A = self.df_full.loc[90275::].drop(["logerror","transactiondate"],axis=1).reset_index()
        self.prop_filled_A = self.prop_filled_A.drop(["index"],axis=1)

        # prop df including 11437 empty rows
        self.prop_filled_B = pd.concat([self.prop_filled_A,self.prop_raw.loc[2973780::]],ignore_index=True)

    # Features with values missing more than threshold percentage will be removed or replaced with -1 or median
    # Options
    # 1. threshold removed w knn filled
    # 2. threshold removed w meadian filled
    # 3. threshold removed w -1 filled

    def null_check(self,df):
        if df.isnull().sum().any() == True:
            print("NANs present")

    def data_to_use(self,data,option = "threshold removed w -1 filled",threshold=0.9):
        # Returns data -> prop, train, sample  (choose 1)
        na_perct = self.df_full.isnull().sum().values/(self.n_rows*1.0)
        df_na_summary = pd.DataFrame(self.df_full.columns.tolist(),columns=["Feature"])
        df_na_summary["% NA"] = na_perct
        active_features = df_na_summary["Feature"][df_na_summary["% NA"] <= threshold].values

        if data == "prop":
            active_features = np.delete(active_features,[1,2]) #removes logerror, transactiondate
            if option == "threshold removed w knn filled":
                df_tmp = self.prop_filled_B[active_features]
                df_tmp = df_tmp.fillna(self.prop_filled_B.median()) # filling the last empty rows
                self.null_check(df_tmp)
                print("Selected %s with %s" %(data,option))

            if option == "threshold removed w median filled":
                df_tmp = self.prop_raw[active_features]
                df_tmp2 = self.prop_filled_B.copy()
                df_tmp.update(df_tmp2,join='left')
                #df_tmp = df_tmp.fillna(self.prop_filled_B.median()) # filling the last empty rows
                df_tmp = df_tmp.fillna(df_tmp.median())
                self.null_check(df_tmp)
                print("Selected %s with %s" %(data,option))

            if option == "threshold removed w -1 filled":
                df_tmp = self.prop_raw[active_features]
                df_tmp2 = self.prop_filled_B.copy()
                df_tmp.update(df_tmp2,join='left')
                df_tmp = df_tmp.fillna(-1) # filling the last empty rows
                self.null_check(df_tmp)
                print("Selected %s with %s" %(data,option))

        if data == "train":
            if option == "threshold removed w knn filled":
                df_tmp = self.train_w_prop_filled[active_features]
                self.null_check(df_tmp)
                print("Selected %s with %s" %(data,option))

            if option == "threshold removed w median filled":
                df_tmp2 = self.train_raw
                df_tmp3 = self.train_raw.merge(self.prop_raw, how='left',on='parcelid')
                df_tmp = df_tmp3[active_features]
                df_tmp = df_tmp.fillna(df_tmp.median())
                self.null_check(df_tmp)
                print("Selected %s with %s" %(data,option))

            if option == "threshold removed w -1 filled":
                df_tmp2 = self.train_raw
                df_tmp3 = self.train_raw.merge(self.prop_raw, how='left',on='parcelid')
                df_tmp = df_tmp3[active_features]
                df_tmp = df_tmp.fillna(-1)
                self.null_check(df_tmp)
                print("Selected %s with %s" %(data,option))

        if data == "sample":
            df_tmp = self.sample
            print("Selected %s" %(data))

        check = 0
        if check == 1:
            # Delete unused dataframes for memory
            del self.prop_raw
            del self.train_raw
            del self.df_full
            gc.collect()

        return df_tmp
