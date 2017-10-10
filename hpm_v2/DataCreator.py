import pandas as pd
import numpy as np
import pickle as pk
import matplotlib as plt
from sklearn import feature_selection as fs
from sklearn import preprocessing as ps
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


print("## ~~~~~~~~~~~~~~~~ Reading in files ~~~~~~~~~~~~~~~~~~~##")
#Read in files from local directory
prop16 = pd.read_csv('../data/properties_2016.csv')
train16 = pd.read_csv('../data/train_2016_v2.csv')


print("## ~~~~~~~~~~~~~~~~ Find features with too many missing values ~~~~~~~~~~~~~~~~~~~##")
#Add count of missing values before any manipulation
prop16['nf_missingcount'] = prop16.isnull().sum(axis=1)
#Add rounded/absolute lat and long as new features, along with lat/long concatenated together
prop16['nf_latrounded'] = (prop16.latitude / 100000).round()*10000
prop16['nf_lonrounded'] = np.abs((prop16.longitude / 100000).round()*10000)
prop16['nf_latlong'] = (prop16.nf_latrounded/1000).map(str).str.slice(0,4) + (prop16.nf_lonrounded/1000).map(str).str.slice(0,4)

"""
print("#Counting missing features and focus on only those with more than 10% populated")
missing = prop16.isnull().sum(axis=0).reset_index()
missing.columns = ['feature', 'count']
missing = missing.sort_values(by='count',ascending=False)
missing['prop'] = missing['count']/prop16.shape[0]
#Features with less than 90% missing, the rest are not very useful
feat90 = list(missing[missing['prop']<.9].feature)
emptyfeatures = list(missing[missing['prop']>=.9].feature)


print("## ~~~~~~~~~~~~~~~~ Make sure Zipcode is completely imputed ~~~~~~~~~~~~~~~~~~~##")
#First select the columns to use to impute the zip code
columnsx = ['basementsqft','bathroomcnt','bedroomcnt','calculatedbathnbr','threequarterbathnbr','finishedfloor1squarefeet',
'calculatedfinishedsquarefeet','finishedsquarefeet6','finishedsquarefeet12','finishedsquarefeet13','finishedsquarefeet15',
'finishedsquarefeet50','fireplacecnt','fullbathcnt','garagecarcnt','garagetotalsqft','latitude','longitude','lotsizesquarefeet',
'numberofstories','poolcnt','poolsizesum','regionidcounty','regionidcity','regionidneighborhood','roomcnt','unitcnt','yardbuildingsqft17',
'yardbuildingsqft26','taxvaluedollarcnt','structuretaxvaluedollarcnt','landtaxvaluedollarcnt','taxamount']
columnsy = 'regionidzip'

print("#Impute all the x quantitative variables with median")
x_na = prop16[columnsx]
imp = ps.Imputer(missing_values='NaN',strategy='median')
x_imp = imp.fit_transform(x_na)
x_imp = np.absolute(x_imp)

print("#Append the zipcode to the imputed dataset")
datfin = pd.DataFrame(x_imp)
datfin['zipcode'] = list(prop16.regionidzip)

print("#Split the data set into NA and filled, for test and train respectively")
xtst = datfin[datfin.zipcode.isnull()==True].iloc[:,0:33]
xtrn = datfin[datfin.zipcode.isnull()==False].iloc[:,0:33]
ytrn = datfin[datfin.zipcode.isnull()==False].iloc[:,33]

print("#Train on the filled values to predict zipcode")
clf = DecisionTreeClassifier()
clf.fit(xtrn,ytrn)
ynew = list(clf.predict(xtst))

print("#Use the calculated list to replace the missing values")
prop16.regionidzip = prop16.regionidzip.fillna(value=1)
prop16.loc[prop16['regionidzip']==1,'regionidzip'] = ynew


print("## ~~~~~~~~~~~~~~~~ Imputing with simple methods ~~~~~~~~~~~~~~~~~~~##")
#Making function that takes a dataframe, column list of fields to use, list of column to group by, method to use (e.g. median/mean/mode)
def simpleimp(dat,col,group=None,method='median'):
    if group == None:
        imp = ps.Imputer(missing_values='NaN',strategy=method,axis=0)
        data = imp.fit_transform(dat[col])
        data = pd.DataFrame(data)
        data.columns = col
        return data
    else:
        data = dat[col+group]
        grp = data.groupby(group)
        for i in col:
            if method=='median':
                data[i] = grp[i].transform(lambda x: x.fillna(x.median()))
            else:
                data[i] = grp[i].transform(lambda x: x.fillna(x.mean()))
        return data[col]

print("#Pick out fields to impute")
feattoimp = list(feat90)
feattoimp = [x for x in feattoimp if x not in ['nf_latlong','regionidzip','parcelid','propertycountylandusecode','propertyzoningdesc']]

print("#Impute fields based on the median by zipcode of each column")
prop16[feattoimp] = simpleimp(prop16,feattoimp,['regionidzip'],'median')
#prop16[feattoimp] = simpleimp(prop16,feattoimp,['regionidneighborhood'],'median')

print("## ~~~~~~~~~~~~~~~~ Adding new features ~~~~~~~~~~~~~~~~~~~##")
#Creating a function to take a dataframe and a list of categorical features and modify the dataframe to add new features
#Using some features from @aharless and @nikunjm88

#May want to look into these - Standard deviations by the logerror
# citystd = train_df[~select_qtr4].groupby('regionidcity')['logerror'].aggregate("std").to_dict()
# zipstd = train_df[~select_qtr4].groupby('regionidzip')['logerror'].aggregate("std").to_dict()
# hoodstd = train_df[~select_qtr4].groupby('regionidneighborhood')['logerror'].aggregate("std").to_dict()
# df['zip_std'] = df['regionidzip'].map(zipstd)
# df['city_std'] = df['regionidcity'].map(citystd)
# df['hood_std'] = df['regionidneighborhood'].map(hoodstd)
"""

def gimmefeatures(df, cat):
    #Number of properties in each zip
    zip_count = df['regionidzip'].value_counts().to_dict()
    # Number of properties in the city
    city_count = df['regionidcity'].value_counts().to_dict()
    # Median year of construction by neighborhood
    medyear = df.groupby('regionidneighborhood')['yearbuilt'].aggregate('median').to_dict()
    # Mean square feet by neighborhood
    meanarea = df.groupby('regionidneighborhood')['calculatedfinishedsquarefeet'].aggregate('mean').to_dict()
    # Neighborhood latitude and longitude
    medlat = df.groupby('regionidneighborhood')['latitude'].aggregate('median').to_dict()
    medlong = df.groupby('regionidneighborhood')['longitude'].aggregate('median').to_dict()
    df['nf_zipCount'] = df['regionidzip'].map(zip_count)
    # Number of properties in the city
    df['nf_cityCount'] = df['regionidcity'].map(city_count)
    # Does property have a garage, pool or hot tub and AC?
    df['nf_PoolACGarageFl'] = ((df['garagecarcnt']>0) & \
                         (df['pooltypeid10']>0) & \
                         (df['airconditioningtypeid']!=5))*1
    # Mean square feet of neighborhood properties
    df['nf_meanNeighSqft'] = df['regionidneighborhood'].map(meanarea)
    # Median year of construction of neighborhood properties
    df['nf_medianNeighYear'] = df['regionidneighborhood'].map(medyear)
    # Neighborhood latitude and longitude
    df['nf_medianNeighLat'] = df['regionidneighborhood'].map(medlat)
    df['nf_medianNeighLong'] = df['regionidneighborhood'].map(medlong)
    for i in cat:
        nm = 'nf_'+str(i)
        grp = df[i].value_counts().to_dict()
        df[nm] = df[i].map(grp)

print("## ~~~~~~~~~~~~~~~~ Creating and outputting new data files ~~~~~~~~~~~~~~~~~~~##")
#Creating first data set with minor imputation and -1 for nan (categor are the categorical features)
categor = ['nf_latlong','airconditioningtypeid','architecturalstyletypeid','buildingqualitytypeid','buildingclasstypeid','decktypeid','fips','fireplaceflag','hashottuborspa','heatingorsystemtypeid','pooltypeid10','pooltypeid2','pooltypeid7','propertycountylandusecode','propertylandusetypeid','propertyzoningdesc','rawcensustractandblock','censustractandblock','storytypeid','typeconstructiontypeid','taxdelinquencyflag']
gimmefeatures(prop16,categor)
#prop16 = prop16.fillna(value = -1)
prop16.to_csv('../data/NewData_MedianbyZipNegforNA.csv')

#Modifying prop16 to now filter out fields that have over 90% missing
#prop16_filtered = prop16.drop(emptyfeatures, axis=1, inplace = False)

