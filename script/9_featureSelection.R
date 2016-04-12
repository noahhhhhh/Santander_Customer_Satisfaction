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
## 1.0 xgb importance #################################################################
#######################################################################################
cat("prepare train, valid, and test data set...\n")
set.seed(888)
dt.train <- dt.featureEngineered[TARGET >= 0]
dt.test <- dt.featureEngineered[TARGET == -1]
dim(dt.train); dim(dt.test)

table(dt.train$TARGET)

## init preds
vec.xgb.pred.train <- rep(0, nrow(dt.train))
## init ls.importance
ls.importance <- list()
## init vec.auc
## folds
cat("folds ...\n")
k <- 5
rounds <- 100
vec.auc <- rep(0, rounds)
watchlist <- list(val = dmx.valid, train = dmx.train) # change to dval
## params
params <- list(booster = "gbtree"
               , nthread = 8
               , objective = "binary:logistic"
               , eval_metric = "auc"
               , max_depth = 6 # 5
               , subsample = .74 #.74
               , min_child_weight = 1 # 1
               , colsample_bytree = .6 #.7
               , eta = 0.022 #.0201
)
## oof train
j <- 1
for(r in 1:rounds){
    for(i in 1:k){
        set.seed(1234 * i * r)
        folds <- createFolds(dt.train$TARGET, k = k, list = F)
        f <- folds == i
        ## dmx
        dmx.train <- xgb.DMatrix(data = sparse.model.matrix(~., data = dt.train[!f, setdiff(names(dt.train), c("ID", "TARGET")), with = F]), label = dt.train[!f]$TARGET)
        dmx.valid <- xgb.DMatrix(data = sparse.model.matrix(~., data = dt.train[f, setdiff(names(dt.train), c("ID", "TARGET")), with = F]), label = dt.train[f]$TARGET)
        watchlist <- list(val = dmx.valid, train = dmx.train) # change to dval
        set.seed(1234 * i * r)
        md.xgb <- xgb.train(params = params
                            , data = dmx.train
                            , nrounds = 100000
                            , early.stop.round = 50
                            , watchlist = watchlist
                            , print.every.n = 50
                            , verbose = T
        )
        # valid
        pred.valid <- predict(md.xgb, dmx.valid)
        vec.xgb.pred.train[f] <- pred.valid
        print(paste("fold:", i, "val auc:", auc(dt.train$TARGET[f], pred.valid)))
        # importance
        importance <- xgb.importance(model = md.xgb)
        ls.importance[[j]] <- importance[[1]]
        j <- j + 1
    }
    print(paste("round:", r, "train auc:", auc(dt.train$TARGET, vec.xgb.pred.train))) 
    vec.auc[r] <- auc(dt.train$TARGET, vec.xgb.pred.train)
}
## save
# save(ls.importance, vec.auc, file = "../data/Santander_Customer_Satisfaction/RData/ls_importance.RData")

#######################################################################################
## 2.0 feature selection for xgb ######################################################
#######################################################################################
## load
load("../data/Santander_Customer_Satisfaction/RData/ls_importance.RData")
## 
vec.importance <- as.character()
for(p in 1:(k * rounds)){
    vec.importance <- c(vec.importance, ls.importance[[p]])
}
vec.importance <- as.numeric(vec.importance)

vec.importance <- table(vec.importance)[order(-table(vec.importance))]
cols.importance <- as.numeric(names(vec.importance))
cols.importance <- setdiff(cols.importance, 312)
## train, test
cat("prepare train, valid, and test data set...\n")
set.seed(888)
ind.train <- createDataPartition(dt.featureEngineered[TARGET >= 0]$TARGET, p = .7, list = F) # remember to change it to .66
dt.train <- dt.featureEngineered[TARGET >= 0][ind.train]
dt.valid <- dt.featureEngineered[TARGET >= 0][-ind.train]
dt.test <- dt.featureEngineered[TARGET == -1]
dim(dt.train); dim(dt.valid); dim(dt.test)
## dmx
x.test <- data.matrix(dt.test[, !c("ID", "TARGET"), with = F])

## params
params <- list(booster = "gbtree"
               , nthread = 8
               , objective = "binary:logistic"
               , eval_metric = "auc"
               , max_depth = 5 # 5
               , subsample = .74 #.74
               , min_child_weight = 1 # 1
               , colsample_bytree = 1 #.7
               , eta = 0.022 #.0201
)

## init tb for graph
tb <- data.table(top = 0, auc = 0)
score <- 0

## feature selection
start <- 10
end <- 300
for(q in start:end){
    ## dmx
    dmx.train <- xgb.DMatrix(data = sparse.model.matrix(~., data = dt.train[, setdiff(names(dt.train), c("ID", "TARGET")), with = F][, cols.importance[1:q], with = F]), label = dt.train$TARGET)
    dmx.valid <- xgb.DMatrix(data = sparse.model.matrix(~., data = dt.valid[, setdiff(names(dt.train), c("ID", "TARGET")), with = F][, cols.importance[1:q], with = F]), label = dt.valid$TARGET)
    ## watchlist
    watchlist <- list(val = dmx.valid, train = dmx.train) # change to dval
    set.seed(1234)
    md.xgb <- xgb.train(params = params
                        , data = dmx.train
                        , nrounds = 100000
                        , early.stop.round = 50
                        , watchlist = watchlist
                        , print.every.n = 50
                        , verbose = T
    )
    # valid
    pred.valid <- predict(md.xgb, dmx.valid)
    print(paste("top:", q, "val auc:", auc(dt.valid$TARGET, pred.valid)))
    
    score <- auc(dt.valid$TARGET, pred.valid)
    tb <- rbind(tb, data.table(top = q, auc = score))
}

tb <- tb[2:(end - start) + 2]
plot(x = tb$top, y = tb$auc, type = "l")
tb[which.max(tb$auc)]
# top       auc
# 1: 119 0.8509882

#######################################################################################
## save ###############################################################################
#######################################################################################
cols.importance[1:119]
cols.selected <- setdiff(names(dt.train), c("ID", "TARGET"))[setdiff(names(dt.train), c("ID", "TARGET")) != "lr"][1:119]
cols.full <- setdiff(names(dt.train), c("ID", "TARGET"))[setdiff(names(dt.train), c("ID", "TARGET")) != "lr"][cols.importance]

save(cols.selected, cols.full, file = "../data/Santander_Customer_Satisfaction/RData/cols_selected.RData")


