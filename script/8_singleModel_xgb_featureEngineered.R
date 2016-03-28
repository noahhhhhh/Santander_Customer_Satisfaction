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
               , max_depth = 5 # 9
               , subsample = .68 #.9
               # , min_child_weight = 1
               , colsample_bytree = .7 #.5
               , eta = .02 #.025
               )
round <- 10
# ls.pred.valid <- list()
# ls.pred.test <- list()
# for(i in 1:round){
#     set.seed(1234 * i)
#     md.xgb <- xgb.train(params = params
#                         , data = dmx.train
#                         , nrounds = 100000 
#                         , early.stop.round = 50
#                         , watchlist = watchlist
#                         , print.every.n = 50
#                         , verbose = F
#     )
#     # valid
#     pred.valid <- predict(md.xgb, dmx.valid)
#     ls.pred.valid[[i]] <- pred.valid
#     print(paste("round:", i, "valid auc:", auc(dt.valid$TARGET, pred.valid)))
#     
#     # test
#     pred.test <- predict(md.xgb, x.test)
#     ls.pred.test[[i]] <- pred.test
# }
# pred.valid.mean <- apply(as.data.table(sapply(ls.pred.valid, print)), 1, mean)
# auc(dt.valid$TARGET, pred.valid.mean)
set.seed(1234)
md.xgb <- xgb.train(params = params
                    , data = dmx.train
                    , nrounds = 100000
                    , early.stop.round = 50
                    , watchlist = watchlist
                    , print.every.n = 50
                    , verbose = F
)
# valid
pred.valid <- predict(md.xgb, dmx.valid)
ls.pred.valid[[i]] <- pred.valid
auc(dt.valid$TARGET, pred.valid)
# 0.8447578
# 0.8449555 with cnt0
# 0.8458973 with cnt0, tuned(incorrect)
# 0.8475985 with cnt0, tuned(correct)
# 0.8466288 with cnt0, cnt1
# 0.8498562 with cnt0, cnt1 with benchmark tuning
# 0.8492806 with cnt0 with benchmark tuning
# 0.8502181 with 10 rounds of mean of xgb, with cnt 0, cnt1 with benchmark tuning
# 0.8498649 with cnt0, cnt1, kmeans with benchmark tuning

## importance
importance <- xgb.importance(setdiff(names(dt.train), c("ID", "TARGET")), model = md.xgb)
importance[Feature == "cnt0"]
# Feature       Gain     Cover  Frequence
# 1:    cnt0 0.03573754 0.0179006 0.03331433
importance[Feature == "cnt1"]
# Feature       Gain      Cover  Frequence
# 1:    cnt1 0.05092287 0.03767977 0.01651278
importance[Feature == "kmeans"]
# Feature         Gain        Cover   Frequence
# 1:  kmeans 0.0001917778 0.0002788239 0.000185271
as.data.frame(importance) # cnt1 top 4, cnt0 top 8, kmeans top 106
#######################################################################################
## submit #############################################################################
#######################################################################################
pred.test <- predict(md.xgb, x.test)
# pred.test.mean <- apply(as.data.table(sapply(ls.pred.test, print)), 1, mean)
submit <- data.table(ID = dt.test$ID, TARGET = pred.test)
# submit <- data.table(ID = dt.test$ID, TARGET = pred.test.mean)
write.csv(submit, file = "submission/10_single_xgb_cnt0_cnt1_kmeans_benchmark_tuning.csv", row.names = F)
# 0.836426
# 0.836738 with cnt0
# 0.837194 with cnt0, tuned(incorrect)
# 0.836437 with cnt0, tuned(corrected)
# 0.836331 with cnt0 and cnt1, tuned
# 0.839958 with cnt0, cnt1 with benchmark tuning
# 0.839282 with cnt0 with benchmark tuning
# 0.840131 with 10 rounds of mean of xgb, with cnt 0, cnt1 with benchmark tuning

