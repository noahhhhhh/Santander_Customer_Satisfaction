setwd("/Volumes/Data Science/Google Drive/data_science_competition/kaggle/Santander_Customer_Satisfaction/")
rm(list = ls()); gc();
require(data.table)
require(purrr)
require(caret)
require(xgboost)
require(Ckmeans.1d.dp)
require(Metrics)
require(ggplot2)
require(DMwR)
source("utilities/preprocess.R")
source("utilities/cv.R")
load("../data/Santander_Customer_Satisfaction/RData/dt_cleansed.RData")
#######################################################################################
## 1.0 train, valid, test #############################################################
#######################################################################################
cat("prepare train, valid, and test data set...\n")
set.seed(888)
ind.train <- createDataPartition(dt.cleansed[TARGET >= 0]$TARGET, p = .9, list = F) # remember to change it to .66
dt.train <- dt.cleansed[TARGET >= 0][ind.train]
dt.valid <- dt.cleansed[TARGET >= 0][-ind.train]
dt.test <- dt.cleansed[TARGET == -1]
dim(dt.train); dim(dt.valid); dim(dt.test)
# [1] 60816   310
# [1] 15204   310
# [1] 75818   310

table(dt.train$TARGET)
# 0     1 
# 58378  2438
table(dt.valid$TARGET)
# 0     1 
# 14634   570

## SMOTE minor cases ----- not useful
# dt.train[, TARGET := as.factor(dt.train$TARGET)]
# df.minor <- SMOTE(TARGET ~ .
#                   , dt.train
#                   , k = 3
#                   , perc.over = 20
#                   , perc.under = 0)
# table(df.minor$TARGET)
# dt.train <- rbind(dt.train, df.minor)
# dt.train[, TARGET := as.numeric(dt.train$TARGET) - 1]
# dim(dt.train)
# table(dt.train$TARGET)

## UNDER SAMPLE
nrow(dt.train[TARGET == 0])
sp <- sample(nrow(dt.train[TARGET == 0]), nrow(dt.train[TARGET == 0]) * .7)
length(sp)
dt.train <- rbind(dt.train[TARGET == 0][sp], dt.train[TARGET == 1])
table(dt.train$TARGET)
# 0     1 
# 31624  5363 
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
                       , k = 10
                       , params)

df.summary
# UNDER SAMPLE 0.5, dt.valid .2
# round   eta mcw md  ss csb mean.dval  max.dval  min.dval     sd.dval mean.valid max.valid min.vaild
# 1     1 0.025   1  9 0.9 0.5 0.8364912 0.8500371 0.8268205 0.009596537  0.8430839 0.8445712 0.8406413
# sd.valid
# 1 0.001746582
# UNDER SAMPLE 0.3, dt.valid .2
# round   eta mcw md  ss csb mean.dval  max.dval  min.dval    sd.dval mean.valid max.valid min.vaild
# 1     1 0.025   1  9 0.9 0.5 0.8366125 0.8548091 0.8101786 0.01768117  0.8422296 0.8436109 0.8399217
# sd.valid
# 1 0.001430463
# UNDER SAMPLE 0.7, dt.valid .2
# round   eta mcw md  ss csb mean.dval  max.dval  min.dval     sd.dval mean.valid max.valid min.vaild
# 1     1 0.025   1  9 0.9 0.5 0.8366834 0.8448043 0.8295128 0.005722074  0.8440323 0.8454439 0.8432108
# sd.valid
# 1 0.0009144937
# UNDER SAMPLE 0.7, kfold 10, dt.valid .2
# round   eta mcw md  ss csb mean.dval  max.dval  min.dval    sd.dval mean.valid max.valid min.vaild
# 1     1 0.025   1  9 0.9 0.5 0.8380746 0.8540579 0.8314056 0.00663848  0.8445303 0.8463393 0.8431303
# sd.valid
# 1 0.001091325
# UNDER SAMPLE 0.7, kfold 10, dt.valid .1