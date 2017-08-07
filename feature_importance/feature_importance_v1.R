# Import libraries
library(readr)
library(randomForest)
library(caret)
library(foreach)
library(doParallel)
registerDoParallel(cores=16)

# Import data
#train_df_4 <- readRDS("C://Users//lakshay//Documents//zes//zestims//data//R//train_df_partial_filled.rds")
load(file = "train_df_3.Rda")
train_df_4 <- train_df_3

# Add pseudo randomness
set.seed(400)

# Change class type
train_df_4$airconditioningtypeid <- as.factor(train_df_4$airconditioningtypeid)
train_df_4$architecturalstyletypeid <- as.factor(train_df_4$architecturalstyletypeid)
train_df_4$bathroomcnt <- as.factor(train_df_4$bathroomcnt)
train_df_4$bedroomcnt <- as.factor(train_df_4$bedroomcnt)
train_df_4$buildingclasstypeid <- as.factor(train_df_4$buildingclasstypeid)
train_df_4$buildingqualitytypeid <- as.factor(train_df_4$buildingqualitytypeid)
train_df_4$calculatedbathnbr <- as.factor(train_df_4$calculatedbathnbr)
train_df_4$decktypeid <- as.factor(train_df_4$decktypeid)
train_df_4$fips <- as.factor(train_df_4$fips)
train_df_4$fireplacecnt <- as.factor(train_df_4$fireplacecnt)
train_df_4$fullbathcnt <- as.factor(train_df_4$fullbathcnt)
train_df_4$garagecarcnt <- as.factor(train_df_4$garagecarcnt)
train_df_4$heatingorsystemtypeid <- as.factor(train_df_4$heatingorsystemtypeid)
train_df_4$poolcnt <- as.factor(train_df_4$poolcnt)
train_df_4$poolsizesum <- as.integer(train_df_4$poolsizesum)
train_df_4$pooltypeid10 <- as.factor(train_df_4$pooltypeid10)
train_df_4$pooltypeid2 <- as.factor(train_df_4$pooltypeid2)
train_df_4$pooltypeid7 <- as.factor(train_df_4$pooltypeid7)
train_df_4$propertycountylandusecode <- as.integer(train_df_4$propertycountylandusecode)
train_df_4$propertylandusetypeid <- as.factor(train_df_4$propertylandusetypeid)
train_df_4$rawcensustractandblock <- as.integer(train_df_4$rawcensustractandblock)
train_df_4$regionidcity <- as.integer(train_df_4$regionidcity)
train_df_4$regionidcounty <- as.factor(train_df_4$regionidcounty)
train_df_4$regionidneighborhood <- as.integer(train_df_4$regionidneighborhood)
train_df_4$regionidzip <- as.integer(train_df_4$regionidzip)
train_df_4$roomcnt <- as.factor(train_df_4$roomcnt)
train_df_4$storytypeid <- as.factor(train_df_4$storytypeid)
train_df_4$threequarterbathnbr <- as.factor(train_df_4$threequarterbathnbr)
train_df_4$typeconstructiontypeid <- as.factor(train_df_4$typeconstructiontypeid)
train_df_4$unitcnt <- as.factor(train_df_4$unitcnt)
train_df_4$yearbuilt <- as.integer(train_df_4$yearbuilt)
train_df_4$numberofstories <- as.integer(train_df_4$numberofstories)
train_df_4$assessmentyear <- as.factor(train_df_4$assessmentyear)
train_df_4$taxdelinquencyyear <- as.factor(train_df_4$taxdelinquencyyear)

#train_df_4 <- train_df_4[,!names(train_df_4) %in% c("X1","parcelid","transactiondate","airconditioningtypeid",
#                                                "architecturalstyletypeid","basementsqft","buildingclasstypeid",
#                                                "decktypeid","finishedfloor1squarefeet","finishedsquarefeet13",
#                                                "finishedsquarefeet15","finishedsquarefeet50","finishedsquarefeet6",
#                                                "fireplacecnt","garagecarcnt","garagetotalsqft","hashottuborspa",
#                                                "heatingorsystemtypeid","poolcnt","pooltypeid10","pooltypeid2",
#                                                "pooltypeid7","propertyzoningdesc","storytypeid","threequarterbathnbr",
#                                                "typeconstructiontypeid","unitcnt","yardbuildingsqft17","yardbuildingsqft26",
#                                                "numberofstories","fireplaceflag","taxdelinquencyflag","taxdelinquencyyear","poolsizesum")]

train_df_4 <- train_df_4[,!names(train_df_4) %in% c("X1","parcelid","transactiondate","logerror")]



train_df_4 <- na.omit(train_df_4)

# Randomize the df
train_df_4 <- train_df_4[sample(nrow(train_df_4)),]


## Build random froest models for each of the following features to get important features
# 1. taxamount
# 2. taxvaluedollarcnt
# 3. landtaxvaluedollarcnt
# 4. finishedsquarefeet13
# 5. finishedsquarefeet6
# 6. finishedsquarefeet15
# 7. numberofstories
# 8. garagetotalsqft
# 9. garagecarcnt
# 10. heatingorsystemtypeid
# 11. unitcnt

#taxamount
#taxvaluedollarcnt
#all_finished_feet
#numberofstories

missing_features <- c("garagetotalsqft" , 
                     "garagecarcnt" ,
                     "heatingorsystemtypeid", 
                     "unitcnt")



for (feature in missing_features) {
  print(feature)
  ## Build random forest model
  #var_names <- names(train_df_4)
  # Exclude response variables and dependent variables
  #var_names <- var_names[!var_names %in% c(feature)]
  
  # build the formula for random forest
  #var_names1 <- paste(var_names, collapse='+')
  #var_names2 <- paste(feature,var_names1, sep='~')
  
  #rf.form_1 <- as.formula(var_names2)
  #fit <- randomForest(rf.form_1,mtry=2, data=train_df_4, ntree=4, importance = T)
  #imp <- importance(fit)
  
  ##################################################
  # Using parallel random forest  

  #x <- subset(train_df_4,select=-c(taxamount))
  x <- train_df_4[,!names(train_df_4) %in% c(feature)]
  y <- train_df_4[[feature]]
  yy <- data.frame(y)

  rf <- foreach(ntree=rep(10,16), .combine=combine, .packages='randomForest') %dopar% randomForest(x, yy[,1], ntree=ntree, mtry=10, importance=TRUE)

	##################################################
  
  imp = importance(rf)
  # write csv file for importance
  write.csv(imp,file = paste(feature,"importance_v1.csv",sep="_"))
  save(rf,file=paste(feature, "rf_v1.RData",sep="_"))
  
}





