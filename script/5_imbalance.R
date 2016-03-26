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
ind.train <- createDataPartition(dt.cleansed[TARGET >= 0]$TARGET, p = .8, list = F) # remember to change it to .66
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

# UNDER SAMPLE ----- UNDER SAMPLE 0.7, kfold 10, dt.valid .2
# UNDER SAMPLE 0.7, kfold 10, dt.valid .2
# round   eta mcw md  ss csb mean.dval  max.dval  min.dval    sd.dval mean.valid max.valid min.vaild
# 1     1 0.025   1  9 0.9 0.5 0.8380746 0.8540579 0.8314056 0.00663848  0.8445303 0.8463393 0.8431303
# sd.valid
# 1 0.001091325
# nrow(dt.train[TARGET == 0])
# set.seed(888)
# sp <- sample(nrow(dt.train[TARGET == 0]), nrow(dt.train[TARGET == 0]) * .7)
# length(sp)
# dt.train <- rbind(dt.train[TARGET == 0][sp], dt.train[TARGET == 1])
# table(dt.train$TARGET)
# 0     1
# 31624  5363

# OVER SAMPLE 
# OVER SAMPLE .2 + org, dt.valid .1, kfold = 10
# round   eta mcw md  ss csb mean.dval max.dval  min.dval    sd.dval mean.valid max.valid min.vaild
# 1     1 0.025   1  9 0.9 0.5 0.8465469 0.869957 0.8333904 0.01111857  0.8443441 0.8455045 0.8424316
# sd.valid
# 1 0.0009773344
# nrow(dt.train[TARGET == 1])
# set.seed(888)
# sp <- sample(nrow(dt.train[TARGET == 1]), nrow(dt.train[TARGET == 1]) * .1, replace = T)
# length(sp)
# dt.train <- rbind(dt.train[TARGET == 1][sp], dt.train[TARGET == 1], dt.train[TARGET == 0])
# table(dt.train$TARGET)
# 0     1 
# 25536  3060

# UNDER AND OVER SAMPLE
# UNDER AND OVER SAMPLE, UNDER .7, OVER .2, k = 10
# round   eta mcw md  ss csb mean.dval  max.dval  min.dval    sd.dval mean.valid max.valid min.vaild
# 1     1 0.025   1  9 0.9 0.5  0.852985 0.8745056 0.8325113 0.01284746  0.8441758 0.8462107 0.8395824
# sd.valid
# 1 0.002092373
nrow(dt.train[TARGET == 0])
set.seed(888)
sp <- sample(nrow(dt.train[TARGET == 0]), nrow(dt.train[TARGET == 0]) * .5)
length(sp)
dt.train <- rbind(dt.train[TARGET == 0][sp], dt.train[TARGET == 1])
table(dt.train$TARGET)

nrow(dt.train[TARGET == 1])
set.seed(888)
sp <- sample(nrow(dt.train[TARGET == 1]), nrow(dt.train[TARGET == 1]) * .2, replace = T)
length(sp)
dt.train <- rbind(dt.train[TARGET == 1][sp], dt.train[TARGET == 1], dt.train[TARGET == 0])
table(dt.train$TARGET)

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
