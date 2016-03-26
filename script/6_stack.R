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

#######################################################################################
## 2.0 stacking #######################################################################
#######################################################################################
## folds
cat("folds ...\n")
set.seed(888)
folds <- createFolds(dt.train$TARGET, k = k, list = F)
cat("init meta features...\n")
dt.meta.train <- data.table(meta_xgb = rep(0, nrow(dt.train)))
dt.meta.valid <- data.table(meta_xgb = rep(0, nrow(dt.valid)))
dt.meta.test <- data.table(meta_xgb = rep(0, nrow(dt.test)))
## xgb
vec.meta.xgb.train <- rep(0, nrow(dt.train))
vec.meta.xgb.valid <- rep(0, nrow(dt.valid))
vec.meta.xgb.test <- rep(0, nrow(dt.train))
## rf
vec.meta.rf.train <- rep(0, nrow(dt.train))
vec.meta.rf.valid <- rep(0, nrow(dt.valid))
vec.meta.rf.test <- rep(0, nrow(dt.test))
## h2o
vec.meta.h2o.train <- rep(0, nrow(dt.train))
vec.meta.h2o.valid <- rep(0, nrow(dt.valid))
vec.meta.h2o.test <- rep(0, nrow(dt.test))

## stacking
k = 10
set.seed(888)
folds <- createFolds(dt.train$TARGET, k = k, list = F)
for(i in 1:k){
    f <- folds == i
    ## xgb
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
    vec.meta.xgb.train[f] <- pred.dval # save to meta feature
    result.dval <- auc(getinfo(dval, "label"), pred.dval)
    
    pred.valid <- predict(md.xgb, dmx.valid)
    result.valid <- auc(getinfo(dmx.valid, "label"), pred.valid)
    
    print(paste("cv:", i, "; xgb - oof:", result.dval, "; valid:", result.valid))
    
    ## rf
    cat("rf...\n")
    dtrain <- dt.train[!f]
    dval <- dt.train[f]
    md.rf <- ranger(TARGET ~.
                    , data = dt.train
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
    result.valid <- auc(dt.valid$TARGET, pred.valid.rf)
    
    print(paste("cv:", i, "; rf - oof:", result.dval, "; valid:", result.valid))
    
    ## h2o
    h2o.init(ip = 'localhost', port = 54321, max_mem_size = '6g')
    h2o.train <- as.h2o(dt.train[, TARGET := as.factor(dt.train$TARGET)][!f])
    h2o.val <- as.h2o(dt.train[, TARGET := as.factor(dt.train$TARGET)][f])
    h2o.valid <- as.h2o(dt.valid[, TARGET := as.factor(dt.valid$TARGET)])
    
    md.h2o <- h2o.deeplearning(x = setdiff(names(dt.train), c("ID", "TARGET")),
                               y = "TARGET",
                               training_frame = h2o.train,
                               nfolds = 3,
                               stopping_rounds = 3,
                               epochs = 20,
                               overwrite_with_best_model = TRUE,
                               activation = "RectifierWithDropout",
                               input_dropout_ratio = 0.2,
                               hidden = c(100,100),
                               l1 = 1e-4,
                               loss = "CrossEntropy",
                               distribution = "bernoulli",
                               stopping_metric = "AUC"
    )
    pred.val <- as.data.frame(h2o.predict(object = md.h2o, newdata = h2o.val))
    result.dval <- auc(dt.train[f]$TARGET, pred.val$p1)
    vec.meta.h2o.train[f] <- pred.val$p1 # save to meta feature
    
    pred.valid <- as.data.frame(h2o.predict(object = md.h2o, newdata = h2o.valid))
    result.valid <- auc(dt.valid$TARGET, pred.valid$p1)
    
    print(paste("cv:", i, "; h2o - oof:", result.dval, "; valid:", result.valid))
}
# dmx.train <- xgb.DMatrix(data = data.matrix(dt.train[, !c("ID", "TARGET"), with = F]), label = dt.train$TARGET)
# dmx.valid <- xgb.DMatrix(data = data.matrix(dt.valid[, !c("ID", "TARGET"), with = F]), label = dt.valid$TARGET)
# x.test <- data.matrix(dt.test[, !c("ID", "TARGET"), with = F])