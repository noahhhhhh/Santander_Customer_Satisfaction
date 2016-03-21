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
## 1.0 train, valid, test #############################################################
#######################################################################################
cat("prepare train, valid, and test data set...\n")
set.seed(888)
ind.train <- createDataPartition(dt.cleansed[TARGET >= 0]$TARGET, p = .8, list = F) # remember to change it to .66
dt.train <- dt.cleansed[TARGET >= 0][ind.train]
dt.valid <- dt.cleansed[TARGET >= 0][-ind.train]
dt.test <- dt.cleansed[TARGET == -1]
dim(dt.train); dim(dt.valid); dim(dt.test)

table(dt.train$TARGET)
table(dt.valid$TARGET)
dmx.train <- xgb.DMatrix(data = data.matrix(dt.train[, !c("ID", "TARGET"), with = F]), label = dt.train$TARGET)
dmx.valid <- xgb.DMatrix(data = data.matrix(dt.valid[, !c("ID", "TARGET"), with = F]), label = dt.valid$TARGET)
x.test <- data.matrix(dt.test[, !c("ID", "TARGET", cols.int64), with = F])

#######################################################################################
## 2.0 cv #############################################################################
#######################################################################################
params <- list(booster = "gbtree"
               , nthread = 8
               , objective = "binary:logistic"
               , eval_metric = "auc"
               , md = 9
               , ss = .9
               , mcw = 1
               , csb = .5
               , eta = .025)

df.summary <- myCV_xgb(dt.train
                       , setdiff(names(dt.train), c("ID", "TARGET"))
                       , dt.valid
                       , k = 5
                       , params)

df.summary
# benchmark - k = 10
# round   eta mcw md  ss csb mean.dval  max.dval  min.dval   sd.dval mean.valid max.valid min.vaild    sd.valid
# 1     1 0.025   1  9 0.9 0.5 0.8364064 0.8522113 0.8140117 0.0104776  0.8443013 0.8460127 0.8406825 0.001822651
# benckmark - k = 5
# round   eta mcw md  ss csb mean.dval  max.dval  min.dval    sd.dval mean.valid max.valid min.vaild    sd.valid
# 1     1 0.025   1  9 0.9 0.5 0.8355522 0.8502162 0.8199273 0.01114614  0.8434327 0.8455512 0.8414356 0.001604345

