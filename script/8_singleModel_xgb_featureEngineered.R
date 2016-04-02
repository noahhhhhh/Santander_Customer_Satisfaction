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
# dt.featureEngineered[, knn := .4 * knn_2 + .3 * knn_4 + .2 * knn_8 + .1 * knn_16]
# dt.featureEngineered[, knn_2 := NULL]
# dt.featureEngineered[, knn_4 := NULL]
# dt.featureEngineered[, knn_8 := NULL]
# dt.featureEngineered[, knn_16 := NULL]
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
## 2.0 imbalance ######################################################################
#######################################################################################
# UNDER AND OVER SAMPLE
# nrow(dt.train[TARGET == 0])
# set.seed(888)
# sp <- sample(nrow(dt.train[TARGET == 0]), nrow(dt.train[TARGET == 0]) * .5)
# length(sp)
# dt.train <- rbind(dt.train[TARGET == 0][sp], dt.train[TARGET == 1])
# table(dt.train$TARGET)
# 
# nrow(dt.train[TARGET == 1])
# set.seed(888)
# sp <- sample(nrow(dt.train[TARGET == 1]), nrow(dt.train[TARGET == 1]) * .2, replace = T)
# length(sp)
# dt.train <- rbind(dt.train[TARGET == 1][sp], dt.train[TARGET == 1], dt.train[TARGET == 0])
# table(dt.train$TARGET)

# UNDER SAMPLE
# nrow(dt.train[TARGET == 0])
# set.seed(888)
# sp <- sample(nrow(dt.train[TARGET == 0]), nrow(dt.train[TARGET == 0]) * .7)
# length(sp)
# dt.train <- rbind(dt.train[TARGET == 0][sp], dt.train[TARGET == 1])
# table(dt.train$TARGET)

# OVER SAMPLE
# nrow(dt.train[TARGET == 1])
# set.seed(888)
# sp <- sample(nrow(dt.train[TARGET == 1]), nrow(dt.train[TARGET == 1]) * .1, replace = T)
# length(sp)
# dt.train <- rbind(dt.train[TARGET == 1][sp], dt.train[TARGET == 1], dt.train[TARGET == 0])
# table(dt.train$TARGET)

#######################################################################################
## 2.0 train ##########################################################################
#######################################################################################
dmx.train <- xgb.DMatrix(data = data.matrix(dt.train[, !c("ID", "TARGET"), with = F]), label = dt.train$TARGET)
dmx.valid <- xgb.DMatrix(data = data.matrix(dt.valid[, !c("ID", "TARGET"), with = F]), label = dt.valid$TARGET)
x.test <- data.matrix(dt.test[, !c("ID", "TARGET"), with = F])

watchlist <- list(val = dmx.valid, train = dmx.train) # change to dval
params <- list(booster = "gbtree"
               , nthread = 8
               , objective = "binary:logistic"
               , eval_metric = "auc"
               , max_depth = 6 # 5; 4:0.850155 5:0.8504593 6:0.8507164 7:0.850147l 9:0.8444527 
               , subsample = .74 #.74; .65:0.85016 .70: 0.8502957; .73:0.8511474 .74:0.8514069 .75:0.8508004 .8: 0.850362
               , min_child_weight = 1 # 1; 2:0.8428779
               , colsample_bytree = .6 #.7; .5:0.8494549 .62:0.8504506 .6:0.8514069* .65:0.8504949 .8: 0.8506108
               , eta = .022 #.02; .021:0.8510815 .022:0.8513952* .023:0.8512173 .025:0.850257
               )
round <- 10
ls.pred.valid <- list()
ls.pred.test <- list()
for(i in 1:round){
    set.seed(1234 * i)
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
    print(paste("round:", i, "valid auc:", auc(dt.valid$TARGET, pred.valid)))

    # test
    pred.test <- predict(md.xgb, x.test)
    ls.pred.test[[i]] <- pred.test
}
pred.valid.mean <- apply(as.data.table(sapply(ls.pred.valid, print)), 1, mean)
auc(dt.valid$TARGET, pred.valid.mean)
# set.seed(1234)
# md.xgb <- xgb.train(params = params
#                     , data = dmx.train
#                     , nrounds = 100000
#                     , early.stop.round = 50
#                     , watchlist = watchlist
#                     , print.every.n = 50
#                     , verbose = F
# )
# # valid
# pred.valid <- predict(md.xgb, dmx.valid)
# auc(dt.valid$TARGET, pred.valid)
# 0.8447578 73 train vs valid 
# 0.8449555 73 train vs valid with cnt0
# 0.8458973 73 train vs valid with cnt0, tuned(incorrect)
# 0.8475985 73 train vs valid with cnt0, tuned(correct)
# 0.8466288 73 train vs valid with cnt0, cnt1
# 0.8498562 73 train vs valid with cnt0, cnt1 with benchmark tuning
# 0.8492806 73 train vs valid with cnt0 with benchmark tuning
# 0.8502181 73 train vs valid with 10 rounds of mean of xgb, with cnt 0, cnt1 with benchmark tuning
# 0.8498649 73 train vs valid with cnt0, cnt1, kmeans with benchmark tuning
# 0.8503481 73 train vs valid with 10 rounds of mean of xgb, with cnt0, cnt1, kmeans, with bench tuning
# 0.8481537 82 train vs valid with 10 rounds of mean of xgb, with cnt0, cnt1, kmeans, with bench tuning
# 0.8497099 65:35 train vs valid with 10 rounds of mean of xgb, with cnt0, cnt1, kmeans, with bench tuning
# 0.8498508 73 train vs valid with 10 rounds of mean of xgb, with cnt0, cnt1, kmeans, with bench tuning and .78 ss
# 0.849999 73 train vs valid with 10 rounds of mean of xgb, with cnt0, cnt1, kmeans, with bench tuning and .74 ss
# 0.8491979 imbalance under and over sampling 73 train vs valid with 10 rounds of mean of xgb, with cnt0, cnt1, kmeans, with bench tuning and .74 ss
# 0.8497037 imbalance under sampling 73 train vs valid with 10 rounds of mean of xgb, with cnt0, cnt1, kmeans, with bench tuning and .74 ss
# 0.8501797 imbalance over sampling 73 train vs valid with 10 rounds of mean of xgb, with cnt0, cnt1, kmeans, with bench tuning and .74 ss
# 0.8492684 73 train vs valid with 10 rounds of mean of xgb, with cnt0, cnt1, kmeans, knn, with bench tuning and .74 ss
# 0.8490421 73 train vs valid with 10 rounds of mean of xgb, with cnt0, cnt1, kmeans, knncombined, with bench tuning and .74 ss
# 0.8503138 73 train vs valid with 10 rounds of mean of xgb, with cnt0, cnt1, kmeans, vars, with bench tuning and .74 ss
# 0.8514069 73 train vs valid with single xgb with cnt0, cnt1, kmeans, vars with my tuning
# 0.8509671 73 train vs valid with 10 xgb with cnt0, cnt1, kmeans, vars with my tuning

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
# pred.test <- predict(md.xgb, x.test)
pred.test.mean <- apply(as.data.table(sapply(ls.pred.test, print)), 1, mean)
# submit <- data.table(ID = dt.test$ID, TARGET = pred.test)
submit <- data.table(ID = dt.test$ID, TARGET = pred.test.mean)
write.csv(submit, file = "submission/23_10_xgb_73_train_valid_cnt0_cnt1_kmeans_vars_my_tuning.csv", row.names = F)
# 0.836426 73 train vs valid 
# 0.836738 73 train vs valid with cnt0
# 0.837194 73 train vs valid with cnt0, tuned(incorrect)
# 0.836437 73 train vs valid with cnt0, tuned(corrected)
# 0.836331 73 train vs valid with cnt0 and cnt1, tuned
# 0.839958 73 train vs valid with cnt0, cnt1 with benchmark tuning
# 0.839282 73 train vs valid with cnt0 with benchmark tuning
# 0.840131 73 train vs valid with 10 rounds of mean of xgb, with cnt 0, cnt1 with benchmark tuning
# 0.840231 73 train vs valid with cnt0, cnt1, kmeans with benchmark tuning
# 0.840358 73 train vs valid with 10 rounds of mean of xgb, with cnt0, cnt1, kmeans, with bench tuning
# 0.839938 82 train vs valid with 10 rounds of mean of xgb, with cnt0, cnt1, kmeans, with bench tuning 82 train vs valid with 10 rounds of mean of xgb, with cnt0, cnt1, kmeans, with bench tuning
# 0.839585 65:35 train vs valid with 10 rounds of mean of xgb, with cnt0, cnt1, kmeans, with bench tuning
# 0.840300 73 train vs valid with 10 rounds of mean of xgb, with cnt0, cnt1, kmeans, with bench tuning and .78 ss
# 0.840367 73 train vs valid with 10 rounds of mean of xgb, with cnt0, cnt1, kmeans, with bench tuning and .74 ss
# 0.839623 imbalance under and over sampling 73 train vs valid with 10 rounds of mean of xgb, with cnt0, cnt1, kmeans, with bench tuning and .74 ss
# 0.840010 imbalance under sampling 73 train vs valid with 10 rounds of mean of xgb, with cnt0, cnt1, kmeans, with bench tuning and .74 ss
# 0.840197 imbalance over sampling 73 train vs valid with 10 rounds of mean of xgb, with cnt0, cnt1, kmeans, with bench tuning and .74 ss
# 0.839139 73 train vs valid with 10 rounds of mean of xgb, with cnt0, cnt1, kmeans, knn, with bench tuning and .74 ss
# 0.838047 73 train vs valid with 10 rounds of mean of xgb, with cnt0, cnt1, kmeans, knncombined, with bench tuning and .74 ss
# 0.839522 73 train vs valid with 10 rounds of mean of xgb, with cnt0, cnt1, kmeans, vars, with bench tuning and .74 ss
# 0.839537 73 train vs valid with single xgb with cnt0, cnt1, kmeans, vars with my tuning
# 0.839343 73 train vs valid with 10 xgb with cnt0, cnt1, kmeans, vars with my tuning