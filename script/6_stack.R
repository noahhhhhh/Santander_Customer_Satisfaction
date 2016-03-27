setwd("/Volumes/Data Science/Google Drive/data_science_competition/kaggle/Santander_Customer_Satisfaction/")
rm(list = ls()); gc();
require(data.table)
require(Matrix)
require(purrr)
require(caret)
require(xgboost)
require(Ckmeans.1d.dp)
require(Metrics)
require(ggplot2)
require(combinat)
require(ranger)
require(h2o)
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

x.test <- data.matrix(dt.test[, !c("ID", "TARGET"), with = F])

h2o.init(ip = 'localhost', port = 54321, max_mem_size = '6g')
h2o.valid <- as.h2o(dt.valid[, TARGET := as.factor(dt.valid$TARGET)])
h2o.test <- as.h2o(dt.test[, !c("ID", "TARGET"), with = F])
dt.valid[, TARGET := as.integer(dt.valid$TARGET) - 1]
#######################################################################################
## 2.0 stacking #######################################################################
#######################################################################################
## folds
cat("folds ...\n")
k = 10
set.seed(888)
folds <- createFolds(dt.train$TARGET, k = k, list = F)
cat("init meta features...\n")
dt.meta.train <- data.table(meta_xgb = rep(0, nrow(dt.train)))
dt.meta.valid <- data.table(meta_xgb = rep(0, nrow(dt.valid)))
dt.meta.test <- data.table(meta_xgb = rep(0, nrow(dt.test)))
## xgb
vec.meta.xgb.train <- rep(0, nrow(dt.train))
vec.meta.xgb.valid <- list()
vec.meta.xgb.test <- list()
## rf
vec.meta.rf.train <- rep(0, nrow(dt.train))
vec.meta.rf.valid <- list()
vec.meta.rf.test <- list()
## h2o
vec.meta.h2o.train <- rep(0, nrow(dt.train))
vec.meta.h2o.valid <- list()
vec.meta.h2o.test <- list()

## 1 level stacking
for(i in 1:k){
    f <- folds == i
    ## xgb #################################################################
    cat("xgb...\n")
    dmx.valid <- xgb.DMatrix(data = sparse.model.matrix(TARGET ~., data = dt.valid[, setdiff(names(dt.train), c("ID")), with = F]), label = dt.valid$TARGET)
    dval <- xgb.DMatrix(data = sparse.model.matrix(TARGET ~., data = dt.train[f, setdiff(names(dt.train), c("ID")), with = F]), label = dt.train[f]$TARGET)
    dtrain <- xgb.DMatrix(data = sparse.model.matrix(TARGET ~., data = dt.train[!f, setdiff(names(dt.train), c("ID")), with = F]), label = dt.train[!f]$TARGET)
    watchlist <- list(val = dmx.valid, train = dtrain) # change to dval
    params <- list(booster = "gbtree"
                   , nthread = 8
                   , objective = "binary:logistic"
                   , eval_metric = "auc"
                   , md = 9
                   , ss = .9
                   , mcw = 1
                   , csb = .5
                   , eta = .025)
    set.seed(888)
    print(paste("cv:", i, "-------"))
    md.xgb <- xgb.train(params = params
                     , data = dtrain
                     , nrounds = 100000 
                     , early.stop.round = 50
                     , watchlist = watchlist
                     , print.every.n = 200
                     , verbose = F
    )
    
    pred.dval <- predict(md.xgb, dval)
    vec.meta.xgb.train[f] <- pred.dval # save to meta feature, train
    result.dval <- auc(getinfo(dval, "label"), pred.dval)
    
    pred.valid <- predict(md.xgb, dmx.valid)
    vec.meta.xgb.valid[[i]] <- pred.valid # save to meta feature, valid
    result.valid <- auc(getinfo(dmx.valid, "label"), pred.valid)
    
    pred.test <- predict(md.xgb, x.test)
    vec.meta.xgb.test[[i]] <- pred.test # save to meta feature, test
    
    print(paste("cv:", i, "; xgb - oof:", result.dval, "; valid:", result.valid))
    
    ## rf  ##################################################################
    cat("rf...\n")
    dtrain <- dt.train[!f]
    dval <- dt.train[f]
    md.rf <- ranger(TARGET ~.
                    , data = dtrain
                    , importance = "impurity"
                    , write.forest = T
                    , replace = T
                    , num.threads = 8
                    , seed = 888
                    , verbose = F)
    pred.dval.rf <- predict(md.rf, dval)$predictions
    vec.meta.rf.train[f] <- pred.dval.rf # save to meta feature
    result.dval <- auc(dval$TARGET, pred.dval.rf)
    
    pred.valid.rf <- predict(md.rf, dt.valid)$predictions
    vec.meta.rf.valid[[i]] <- pred.valid.rf # save to meta feature, valid
    result.valid <- auc(dt.valid$TARGET, pred.valid.rf)
    
    pred.test.rf <- predict(md.rf, dt.test)$predictions
    vec.meta.rf.test[[i]] <- pred.test.rf # save to meta feature, valid
    
    print(paste("cv:", i, "; rf - oof:", result.dval, "; valid:", result.valid))
    
    ## h2o  #################################################################
    h2o.train <- as.h2o(dt.train[, TARGET := as.factor(dt.train$TARGET)][!f])
    h2o.val <- as.h2o(dt.train[, TARGET := as.factor(dt.train$TARGET)][f])
    
    
    md.h2o <- h2o.deeplearning(x = setdiff(names(dt.train), c("ID", "TARGET")),
                               y = "TARGET",
                               training_frame = h2o.train,
                               stopping_rounds = 3,
                               epochs = 20,
                               overwrite_with_best_model = TRUE,
                               activation = "RectifierWithDropout",
                               input_dropout_ratio = 0.2,
                               hidden = c(256,128),
                               l1 = 1e-4,
                               loss = "CrossEntropy",
                               distribution = "bernoulli",
                               stopping_metric = "AUC"
    )
    pred.val <- as.data.frame(h2o.predict(object = md.h2o, newdata = h2o.val))
    result.dval <- auc(dt.train[f]$TARGET, pred.val$p1)
    vec.meta.h2o.train[f] <- pred.val$p1 # save to meta feature, train
    
    pred.valid <- as.data.frame(h2o.predict(object = md.h2o, newdata = h2o.valid))
    vec.meta.h2o.valid[[i]] <- pred.valid$p1 # save to meta feature, valid
    result.valid <- auc(dt.valid$TARGET, pred.valid$p1)
    
    pred.test <- as.data.frame(h2o.predict(object = md.h2o, newdata = h2o.test))
    vec.meta.h2o.test[[i]] <- pred.test$p1 # save to meta feature, test
    
    print(paste("cv:", i, "; h2o - oof:", result.dval, "; valid:", result.valid))
    dt.train[, TARGET := as.integer(dt.train$TARGET) - 1]
}

## dt.meta
cat("add to dt.meta...\n")
## xgb
dt.meta.train[, meta_xgb := vec.meta.xgb.train]
dt.meta.valid[, meta_xgb := apply(as.data.table(sapply(vec.meta.xgb.valid, print)) , 1, function(x) mean(x))]
dt.meta.test[, meta_xgb := apply(as.data.table(sapply(vec.meta.xgb.test, print)) , 1, function(x) mean(x))]
## rf
dt.meta.train[, meta_rf := vec.meta.rf.train]
dt.meta.valid[, meta_rf := apply(as.data.table(sapply(vec.meta.rf.valid, print)) , 1, function(x) mean(x))]
dt.meta.test[, meta_rf := apply(as.data.table(sapply(vec.meta.rf.test, print)) , 1, function(x) mean(x))]
## h2o
dt.meta.train[, meta_h2o := vec.meta.h2o.train]
dt.meta.valid[, meta_h2o := apply(as.data.table(sapply(vec.meta.h2o.valid, print)) , 1, function(x) mean(x))]
dt.meta.test[, meta_h2o := apply(as.data.table(sapply(vec.meta.h2o.test, print)) , 1, function(x) mean(x))]

## save
save(dt.meta.train, dt.meta.valid, dt.meta.test, file = "../data/Santander_Customer_Satisfaction/RData/dt_1_level_meta_feature.RData")

## 2 level stacking
load("../data/Santander_Customer_Satisfaction/RData/dt_1_level_meta_feature.RData")
dt.train <- cbind(dt.train, dt.meta.train)
dt.valid <- cbind(dt.valid, dt.meta.valid)
dt.test <- cbind(dt.test, dt.meta.test)
## folds
cat("folds ...\n")
k = 5
set.seed(888)
folds <- createFolds(dt.train$TARGET, k = k, list = F)
cat("init meta features...\n")
dt.meta.train <- data.table(meta_xgb = rep(0, nrow(dt.train)))
dt.meta.valid <- data.table(meta_xgb = rep(0, nrow(dt.valid)))
dt.meta.test <- data.table(meta_xgb = rep(0, nrow(dt.test)))
## xgb
vec.meta.xgb.train <- rep(0, nrow(dt.train))
vec.meta.xgb.valid <- list()
vec.meta.xgb.test <- list()
## rf
vec.meta.rf.train <- rep(0, nrow(dt.train))
vec.meta.rf.valid <- list()
vec.meta.rf.test <- list()
## h2o
vec.meta.h2o.train <- rep(0, nrow(dt.train))
vec.meta.h2o.valid <- list()
vec.meta.h2o.test <- list()

for(i in 1:k){
    f <- folds == i
    ## xgb #################################################################
    cat("xgb...\n")
    dmx.valid <- xgb.DMatrix(data = sparse.model.matrix(TARGET ~., data = dt.valid[, setdiff(names(dt.train), c("ID")), with = F]), label = dt.valid$TARGET)
    dval <- xgb.DMatrix(data = sparse.model.matrix(TARGET ~., data = dt.train[f, setdiff(names(dt.train), c("ID")), with = F]), label = dt.train[f]$TARGET)
    dtrain <- xgb.DMatrix(data = sparse.model.matrix(TARGET ~., data = dt.train[!f, setdiff(names(dt.train), c("ID")), with = F]), label = dt.train[!f]$TARGET)
    watchlist <- list(val = dmx.valid, train = dtrain) # change to dval
    params <- list(booster = "gbtree"
                   , nthread = 8
                   , objective = "binary:logistic"
                   , eval_metric = "auc"
                   , md = 9
                   , ss = .9
                   , mcw = 1
                   , csb = .5
                   , eta = .025)
    set.seed(888)
    print(paste("cv:", i, "-------"))
    md.xgb <- xgb.train(params = params
                        , data = dtrain
                        , nrounds = 100000 
                        , early.stop.round = 50
                        , watchlist = watchlist
                        , print.every.n = 200
                        , verbose = F
    )
    
    pred.dval <- predict(md.xgb, dval)
    vec.meta.xgb.train[f] <- pred.dval # save to meta feature, train
    result.dval <- auc(getinfo(dval, "label"), pred.dval)
    
    pred.valid <- predict(md.xgb, dmx.valid)
    vec.meta.xgb.valid[[i]] <- pred.valid # save to meta feature, valid
    result.valid <- auc(getinfo(dmx.valid, "label"), pred.valid)
    
    pred.test <- predict(md.xgb, x.test)
    vec.meta.xgb.test[[i]] <- pred.test # save to meta feature, test
    
    print(paste("cv:", i, "; xgb - oof:", result.dval, "; valid:", result.valid))
    
    ## rf  ##################################################################
    cat("rf...\n")
    dtrain <- dt.train[!f]
    dval <- dt.train[f]
    md.rf <- ranger(TARGET ~.
                    , data = dtrain
                    , importance = "impurity"
                    , write.forest = T
                    , replace = T
                    , num.threads = 8
                    , seed = 888
                    , verbose = F)
    pred.dval.rf <- predict(md.rf, dval)$predictions
    vec.meta.rf.train[f] <- pred.dval.rf # save to meta feature
    result.dval <- auc(dval$TARGET, pred.dval.rf)
    
    pred.valid.rf <- predict(md.rf, dt.valid)$predictions
    vec.meta.rf.valid[[i]] <- pred.valid.rf # save to meta feature, valid
    result.valid <- auc(dt.valid$TARGET, pred.valid.rf)
    
    pred.test.rf <- predict(md.rf, dt.test)$predictions
    vec.meta.rf.test[[i]] <- pred.test.rf # save to meta feature, valid
    
    print(paste("cv:", i, "; rf - oof:", result.dval, "; valid:", result.valid))
    
}

## dt.meta
cat("add to dt.meta...\n")
## xgb
dt.meta.train[, meta_xgb := vec.meta.xgb.train]
dt.meta.valid[, meta_xgb := apply(as.data.table(sapply(vec.meta.xgb.valid, print)) , 1, function(x) mean(x))]
dt.meta.test[, meta_xgb := apply(as.data.table(sapply(vec.meta.xgb.test, print)) , 1, function(x) mean(x))]
## rf
dt.meta.train[, meta_rf := vec.meta.rf.train]
dt.meta.valid[, meta_rf := apply(as.data.table(sapply(vec.meta.rf.valid, print)) , 1, function(x) mean(x))]
dt.meta.test[, meta_rf := apply(as.data.table(sapply(vec.meta.rf.test, print)) , 1, function(x) mean(x))]

## 3 level stacking/blending
# train
result.meta.train <- (dt.meta.train$meta_xgb * .8
                      + dt.meta.train$meta_rf * .2)
auc(dt.train$TARGET, result.meta.train)
# 0.8054994
# valid
result.meta.valid <- (dt.meta.valid$meta_xgb * .8
                      + dt.meta.valid$meta_rf * .2)
auc(dt.valid$TARGET, result.meta.valid)
# 0.8469115
# test
result.meta.test <- (dt.meta.test$meta_xgb * .8
                     + dt.meta.test$meta_rf * .2)
#######################################################################################
## submit #############################################################################
#######################################################################################
submit <- data.table(ID = dt.test$ID, TARGET = result.meta.test)
write.csv(submit, file = "submission/1_raw_features_stacking_1_xgb_rf_h2o_2_xgb_rf_3_08_02.csv", row.names = F)
