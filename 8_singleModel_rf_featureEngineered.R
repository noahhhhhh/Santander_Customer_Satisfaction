setwd("/Volumes/Data Science/Google Drive/data_science_competition/kaggle/Santander_Customer_Satisfaction/")
rm(list = ls()); gc();
require(data.table)
require(purrr)
require(caret)
require(xgboost)
require(ranger)
require(Ckmeans.1d.dp)
require(Metrics)
require(ggplot2)
require(combinat)
source("utilities/preprocess.R")
source("utilities/cv.R")
load("../data/Santander_Customer_Satisfaction/RData/dt_featureEngineered.RData")

#######################################################################################
## 1.0 train, valid, test #############################################################
#######################################################################################
cat("prepare train, valid, and test data set...\n")
set.seed(888)
ind.train <- createDataPartition(dt.featureEngineered[TARGET >= 0]$TARGET, p = .7, list = F) # remember to change it to .66
dt.train <- dt.featureEngineered[TARGET >= 0][ind.train]
dt.valid <- dt.featureEngineered[TARGET >= 0][-ind.train]
dt.test <- dt.featureEngineered[TARGET == -1]
dim(dt.train); dim(dt.valid); dim(dt.test)

table(dt.train$TARGET)
table(dt.valid$TARGET)

#######################################################################################
## 2.0 train ##########################################################################
#######################################################################################
md.rf <- ranger(TARGET ~.
                , data = dt.train[, !c("ID"), with = F]
                , num.trees = 2000 # 500: 0.8259111; 1000: 0.8273999; 2000: 0.8282475; 5000: 0.8282747
                , mtry = 50
                , importance = "impurity"
                , write.forest = T
                , replace = T
                , num.threads = 8
                , seed = 888
                , verbose = T)

pred.valid.rf <- predict(md.rf, dt.valid[, !c("ID"), with = F])$predictions
auc(dt.valid$TARGET, pred.valid.rf)

pred.test.rf <- predict(md.rf, dt.test)$predictions
vec.meta.rf.test[[i]] <- pred.test.rf # save to meta feature, valid
