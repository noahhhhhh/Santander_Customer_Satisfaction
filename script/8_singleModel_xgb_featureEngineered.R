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
dmx.train <- xgb.DMatrix(data = data.matrix(dt.train[, !c("ID", "TARGET"), with = F]), label = dt.train$TARGET)
dmx.valid <- xgb.DMatrix(data = data.matrix(dt.valid[, !c("ID", "TARGET"), with = F]), label = dt.valid$TARGET)
x.test <- data.matrix(dt.test[, !c("ID", "TARGET"), with = F])

#######################################################################################
## 2.0 train ##########################################################################
#######################################################################################
watchlist <- list(val = dmx.valid, train = dmx.train) # change to dval
params <- list(booster = "gbtree"
               , nthread = 8
               , objective = "binary:logistic"
               , eval_metric = "auc"
               , max_depth = 9
               , subsample = .9
               , min_child_weight = 1
               , colsample_bytree = .5
               , eta = .025)
set.seed(888)
md.xgb <- xgb.train(params = params
                    , data = dmx.train
                    , nrounds = 100000 
                    , early.stop.round = 50
                    , watchlist = watchlist
                    , print.every.n = 200
                    , verbose = F
)
pred.valid <- predict(md.xgb, dmx.valid)
auc(dt.valid$TARGET, pred.valid)
# 0.8447578
# 0.8449555 with cnt0
# 0.8458973 with cnt0, tuned(incorrect)
# 0.8475985 with cnt0, tuned(correct)
# 0.8466288 with cnt0, cnt1

## importance
importance <- xgb.importance(setdiff(names(dt.train), c("ID", "TARGET")), model = md.xgb)
importance[Feature == "cnt0"]
# Feature       Gain     Cover  Frequence
# 1:    cnt0 0.03573754 0.0179006 0.03331433
importance[Feature == "cnt1"]
as.data.frame(importance) # cnt1 top 4, cnt0 top 8
#######################################################################################
## submit #############################################################################
#######################################################################################
pred.test <- predict(md.xgb, x.test)
submit <- data.table(ID = dt.test$ID, TARGET = pred.test)
write.csv(submit, file = "submission/6_single_xgb_cnt0_cnt1.csv", row.names = F)
# 0.836426
# 0.836738 with cnt0
# 0.837194 with cnt0, tuned(incorrect)

