
# coding: utf-8

# # Feature Importance and Regression Models

# 11 features for which feataure importance has been calculated using Random Forest
# 1. finishedsquarefeet13
# 2. finishedsquarefeet15
# 3. finishedsquarefeet6
# 4. garagecarcnt
# 5. garagetotalsqft
# 6. heatingorsystemtypeid
# 7. landtaxvaluedollarcnt
# 8. numberofstories
# 9. taxamount
# 10. taxvaluedollarcnt
# 11. unitcnt
# 
# The random forest training was run on AWS and the results have been imported back as csv. 
# 
# In this notebook, each target feature, a polynomial regression model is fit using the most important features. The number of features for each training is subjective to individual feature

# In[11]:

import pandas as pd
import numpy as np
import time
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sets import Set
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor as mlp



color = sns.color_palette()
#get_ipython().magic(u'matplotlib inline')


# Importing Data


df = pd.read_pickle("train_df_partial_filled.pkl")
df.shape

# Creating feature importance dataframes for each target feature

path = "./feature_importance/"
feature_list = []
filenames = []
importance_dfs = []
for filename in glob(path + "*.csv"):
    filenames.append(filename)
    # WINDOWS
    feature = filename.split(".")[-2].split("\\")[-1]
    # LINUX
    #feature = filename.split(".")[-2].split("/")[-1]    
    feature_list.append(feature)
    df2 = pd.read_csv(filename)
    importance_dfs.append(df2)

# # 1. Plotting feature importance for each target variable

imp_features = []
filtered_importance_dfs = []
features_to_keep = ["Unnamed: 0","MeanDecreaseAccuracy","MeanDecreaseGini"]

for i in importance_dfs:
    if len(i.columns) > 3:
        j = i[features_to_keep]
    else:
        j = i
    j.columns = ["Unnamed: 0", "%IncMSE", "IncNodePurity"]
    filtered_importance_dfs.append(j)
        
for df1 in filtered_importance_dfs:
    #df1 = df1.sort_values(by = ["%IncMSE"])
    df1 = df1.sort_values(by = ["%IncMSE"],ascending=False).reset_index()

    # Plot importance
    a = df1["Unnamed: 0"].values
    #b = df1["IncNodePurity"].values
    b = df1["%IncMSE"].values
    names = np.ndarray.tolist(a)
    imp = np.ndarray.tolist(b)
    y_pos = np.arange(len(names))
    imp_features.append(list(df1["Unnamed: 0"].ix[0:14].values))



# removing non-numeric features
for i in range(len(imp_features)):
    if 'propertycountylandusecode' in imp_features[i]:
        imp_features[i].remove('propertycountylandusecode')    


# # 2. Selection of variables based on importance and %NAN  

# create sets for features
a = Set()

for i in imp_features:
    for element in i:
        a.update([element])
    
a = list(a)    
b = {}

for i in range(0, len(a)):
    b[a[i]] = i

confusion_matrix = []    
    
for i in imp_features:
    temp = [0]*len(a)
    for j in range(0, len(i)):
        temp[b[i[j]]] = 16 - (j + 1)
    confusion_matrix.append(temp)
    
final_sum = list(np.sum(confusion_matrix, axis=0))

df_cm = pd.DataFrame(confusion_matrix, index = [i for i in feature_list],
                  columns = [i for i in a])




temp_frame_laks = df.isnull().sum(axis=0)
feature_list_laks = temp_frame_laks.index.values
null_percentage = temp_frame_laks.values/3064055.

feature_list_laks = list(feature_list_laks)
null_percentage = list(null_percentage)

nan_values = []
for i in a:
    temp_index = feature_list_laks.index(i)
    nan_values.append(int(100.0 - round(null_percentage[temp_index]*100, 1)))
    
transpose = map(list, zip(*[nan_values, final_sum]))    
    
df_nan = pd.DataFrame(transpose, index = [i for i in a],
                  columns = ["Not Missing %", "SUM"])



# The above plot shows the sum of importance for each feature and % of filled valules (opposite of %NANs) in the data set. We would start filling the fields which are dark on both scales. 
#That would mean the most important features with least amount of NANs first and so on. 


nan_dict = dict(zip(a, nan_values))


filtered_imp_features = []

###############################################
NAN_THRESHOLD = 99  
###############################################

for i in imp_features:
    tmp = []
    for j in i:
        if nan_dict[j] > NAN_THRESHOLD:
            tmp.append(j)
    filtered_imp_features.append(tmp)

    
actual_names = [i.split("_")[0] for i in feature_list]
    
final_dict = dict(zip(actual_names, filtered_imp_features))

print final_dict


# # 3. Building prediction models for target variables

# # Order of fill
# 1. Fill group B using no_nan feature and update global data after every prediction
# 2. Fill group A using the feature selection from previous section

# Grouping of data as described above

df_no_nan = df_nan[df_nan["Not Missing %"]==100].sort_values(by=["SUM","Not Missing %"],ascending=False).reset_index()
df_to_be_filled_all = df_nan[df_nan["Not Missing %"]!=100].sort_values(by=["SUM","Not Missing %"],ascending=False).reset_index()

# Removing 16,17,19,20 rows from all data. 
# 16 - finishedsquarefeet15
# 17 - finishedfloor1squarefeet
# 19 - finishedfloor1squarefeet
# 20 - finishedsqaurefeet50.
# These are useless features as they are 100 % or 94 % empty 

df_to_be_filled_all = df_to_be_filled_all.drop([16,17,19,20])

df_to_be_filled_A = df_to_be_filled_all[df_to_be_filled_all["index"].isin(actual_names)].reset_index()
df_to_be_filled_B = df_to_be_filled_all[~df_to_be_filled_all["index"].isin(actual_names)].reset_index()


# # Filling Group B


list_no_nan = list(df_no_nan["index"].values)
list_B = list(df_to_be_filled_B["index"].values)

for target_var in list_B:
    features = list_no_nan
    col_names = features +  [target_var]
    df_local = df[col_names]
    df_local = df_local.dropna(axis=0)
    X = df_local[features].as_matrix()
    y = df_local[target_var].values
    sc = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    t1 = time.time()
    
    #clf = SVR(kernel='rbf', degree=3, gamma='auto', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, \
    #         shrinking=True, cache_size=200, verbose=False, max_iter=-1)
    
    clf = mlp(hidden_layer_sizes=(100, ), activation='relu', solver='adam', alpha=0.0001,batch_size='auto',
				learning_rate='constant', learning_rate_init=0.001, power_t=0.5, 
				max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False,
				warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False,
				validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    
    clf.fit(X_train, y_train) 
    
    # Score
    score = clf.score(X_test,y_test)
    
    # Execution Time
    exec_time = time.time() - t1
    print("Execution Time: %g"%exec_time)
    print("score: %g"%score)
    break




