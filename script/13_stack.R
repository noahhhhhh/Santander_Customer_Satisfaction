setwd("/Volumes/Data Science/Google Drive/data_science_competition/kaggle/Santander_Customer_Satisfaction/")
rm(list = ls()); gc();
require(data.table)
require(caret)
require(xgboost)
require(glmnet)
load("../data/Santander_Customer_Satisfaction/RData/dt_featureEngineered.RData")
load("../data/Santander_Customer_Satisfaction/RData/dt_meta_1_all.RData")
#######################################################################################
## 1.0 train, valid, test #############################################################
#######################################################################################
cat("prepare train, valid, and test data set...\n")
dt.featureEngineered <- cbind(dt.featureEngineered[, !c("TARGET"), with = F], rbind(dt.pred.train, dt.pred.test))
set.seed(888)
ind.train <- createDataPartition(dt.featureEngineered[TARGET >= 0]$TARGET, p = .8, list = F) # remember to change it to .66
dt.train <- dt.featureEngineered[TARGET >= 0][ind.train]
dt.valid <- dt.featureEngineered[TARGET >= 0][-ind.train]
dt.test <- dt.featureEngineered[TARGET == -1]
dim(dt.train); dim(dt.valid); dim(dt.test)

table(dt.train$TARGET)
table(dt.valid$TARGET)

#######################################################################################
## 1.0 train ##########################################################################
#######################################################################################
dmx.valid <- xgb.DMatrix(data = sparse.model.matrix(TARGET ~., data = dt.valid[, setdiff(names(dt.train), c("ID")), with = F]), label = dt.valid$TARGET)
dmx.train <- xgb.DMatrix(data = sparse.model.matrix(TARGET ~., data = dt.train[, setdiff(names(dt.train), c("ID")), with = F]), label = dt.train$TARGET)
dmx.test <- xgb.DMatrix(data = sparse.model.matrix(TARGET ~., data = dt.test[, setdiff(names(dt.train), c("ID")), with = F]), label = dt.test$TARGET)

watchlist <- list(val = dmx.valid, train = dmx.train) # change to dval
params <- list(booster = "gbtree"
               , nthread = 8
               , objective = "binary:logistic"
               , eval_metric = "auc"
               , max_depth = 5
               , subsample = .74
               , min_child_weight = 1
               , colsample_bylevel = .2
               , eta = .022)
set.seed(888)
md.xgb <- xgb.train(params = params
                    , data = dmx.train
                    , nrounds = 100000 
                    , early.stop.round = 50
                    , watchlist = watchlist
                    , print.every.n = 200
                    , verbose = F
)
as.data.frame(xgb.importance(feature_names = setdiff(names(dt.train), c("ID", "TARGET")), model = md.xgb))
xgb.plot.importance(xgb.importance(feature_names = setdiff(names(dt.train), c("ID", "TARGET")), model = md.xgb))
pred.dval <- predict(md.xgb, dmx.valid)
auc(getinfo(dmx.valid, "label"), pred.dval)
# 0.8462411
pred.test <- predict(md.xgb, dmx.test)
#######################################################################################
## submit #############################################################################
#######################################################################################
submit <- data.table(ID = dt.test$ID, TARGET = pred.test)
write.csv(submit, file = "submission/36_ensemble_with_single_xgb.csv", row.names = F)
















