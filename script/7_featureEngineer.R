setwd("/Volumes/Data Science/Google Drive/data_science_competition/kaggle/Santander_Customer_Satisfaction/")
rm(list = ls()); gc();
require(data.table)
require(purrr)
require(caret)
require(xgboost)
require(Ckmeans.1d.dp)
require(Metrics)
require(ggplot2)
require(combinat)
source("utilities/preprocess.R")
source("utilities/cv.R")
load("../data/Santander_Customer_Satisfaction/RData/dt_cleansed.RData")
#######################################################################################
## count 0 ############################################################################
#######################################################################################
cnt0 <- apply(dt.cleansed[, !c("ID", "TARGET"), with = F], 1, function(x)sum(x == 0))


#######################################################################################
## count 1 ############################################################################
#######################################################################################
cnt1 <- apply(dt.cleansed[, !c("ID", "TARGET"), with = F], 1, function(x)sum(x == 1))

#######################################################################################
## save ###############################################################################
#######################################################################################
dt.cleansed[, cnt0 := cnt0]
dt.cleansed[, cnt1 := cnt1]

dt.featureEngineered <- dt.cleansed
save(dt.featureEngineered, file = "../data/Santander_Customer_Satisfaction/RData/dt_featureEngineered.RData")
