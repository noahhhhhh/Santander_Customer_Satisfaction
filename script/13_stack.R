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
dt.train <- dt.featureEngineered[TARGET >= 0]
dt.test <- dt.featureEngineered[TARGET == -1]
dim(dt.train); dim(dt.test)

table(dt.train$TARGET)


#######################################################################################
## 1.0 train ##########################################################################
#######################################################################################
## folds
cat("folds ...\n")
k = 7
set.seed(888)
folds <- createFolds(dt.train$TARGET, k = k, list = F)

dmx.test <- xgb.DMatrix(data = sparse.model.matrix(TARGET ~., data = dt.test[, setdiff(names(dt.train), c("ID")), with = F]), label = dt.test$TARGET)

pred.dval <- rep(0, nrow(dt.valid))
pred.test <- rep(0, nrow(dt.test))

cat("init meta features...\n")
vec.meta.2.xgb.train <- rep(0, nrow(dt.train))
vec.meta.2.xgb.test <- rep(0, nrow(dt.test))

for(i in 1:k){
    f <- folds == i
    dmx.valid <- xgb.DMatrix(data = sparse.model.matrix(TARGET ~., data = dt.train[f, setdiff(names(dt.train), c("ID")), with = F]), label = dt.train$TARGET[f])
    dmx.train <- xgb.DMatrix(data = sparse.model.matrix(TARGET ~., data = dt.train[!f, setdiff(names(dt.train), c("ID")), with = F]), label = dt.train$TARGET[!f])
    
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
    set.seed(888 * i)
    md.xgb <- xgb.train(params = params
                        , data = dmx.train
                        , nrounds = 100000 
                        , early.stop.round = 50
                        , watchlist = watchlist
                        , print.every.n = 50
                        , verbose = F
    )
    # as.data.frame(xgb.importance(feature_names = setdiff(names(dt.train), c("ID", "TARGET")), model = md.xgb))
    # xgb.plot.importance(xgb.importance(feature_names = setdiff(names(dt.train), c("ID", "TARGET")), model = md.xgb))
    pred.dval <- predict(md.xgb, dmx.valid)
    vec.meta.2.xgb.train[f] <- pred.dval
    print(paste("fold", i, "auc:", auc(getinfo(dmx.valid, "label"), pred.dval)))
    # 0.8462411
    
    vec.meta.2.xgb.test <- vec.meta.2.xgb.test + predict(md.xgb, dmx.test) / k
}
auc(dt.train$TARGET, vec.meta.2.xgb.train)
# 0.8462269
# 0.8462269 20 xgb with randome cols
# 0.8463264 20 xgb

#######################################################################################
## submit #############################################################################
#######################################################################################
submit <- data.table(ID = dt.test$ID, TARGET = pred.test)
write.csv(submit, file = "submission/38_ensemble_with_20_xgb_not_random_cols.csv", row.names = F)
# 0.840580
# 0.840556 20 xgb with randome cols
# 0.840373 20 xgb 















