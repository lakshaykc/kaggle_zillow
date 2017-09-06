
# coding: utf-8

# # Handling Missing Data

# 1. Mean/ Mode
# 2. KNN with top 15 features
# 3. KNN with all the features

# In[27]:

import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sets import Set
from scipy.stats import norm
from scipy import spatial
from sklearn.neighbors import KDTree
import sys

sys.setrecursionlimit(1000000)


# Importing Data

# In[2]:

df = pd.read_pickle("train_df_partial_filled.pkl")
df.shape



# # 2. KNN with top 15 features
# 

# Import feature importance df to use 15 most important features.
df_f_imp = pd.read_pickle("feature_importance.pkl")

df_f_imp = df_f_imp.sort_values(by = "SUM",ascending=False).reset_index()


# ## Using KDTree to find nearest neighbors

# ### Fill the first 15 


class knn(object):
    
    def __init__(self,feature,tree=tree):
        self.feature = feature

    def mean(self,X,df,k):
        dist,ind = tree.query(X,k=k)
        feature_mean = np.mean(df[self.feature].iloc[ind[0]].values)

        try:
            return feature_mean
        
        except ValueError:
            pass

			
top_15_f = list(df_f_imp["index"][0:15].values)
df_top_15 = df[top_15_f]
df_top_15_nonull = df_top_15.dropna()
df_top_15_nonull = df_top_15_nonull.reset_index()

for feature in top_15_f:
    print feature
    
    tmp_df = df_top_15_nonull.drop(['index'],1)
    df_to_be_filled = df_top_15[df_top_15[feature].isnull()]
    [m,n]  = df_to_be_filled.shape
    data_top_15 = tmp_df.as_matrix()
    
    for i in range(m):
        tmp_df_2 = df_to_be_filled.iloc[i].drop(feature)
        tmp_df_2 = tmp_df_2.dropna()
        local_feat = tmp_df_2.index.tolist()
        training_df = df_top_15_nonull[local_feat].as_matrix()
        
        tree = KDTree(training_df)
        
        k = 50
        X = df_to_be_filled[local_feat].iloc[i].as_matrix()
        X = np.reshape(X,(1,-1))
        f_knn = knn(feature)
        df_to_be_filled[feature].iloc[i] = f_knn.mean(X,tmp_df,k)
        
    df_top_15.update(df_to_be_filled,join='left')
    print("saving after completing %f" %feature)
    df_top_15.to_pickle("df.pkl")

