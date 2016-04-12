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
load("../data/Santander_Customer_Satisfaction/RData/cols_selected.RData")
#######################################################################################
## 1.0 train with oof #################################################################
#######################################################################################
# dt.featureSelected <- dt.featureEngineered[, c(cols.selected, "ID", "TARGET"), with = F]
cat("prepare train, valid, and test data set...\n")
set.seed(888)
dt.train <- dt.featureEngineered[TARGET >= 0]
dt.test <- dt.featureEngineered[TARGET == -1]
dim(dt.train); dim(dt.test)

table(dt.train$TARGET)

## folds
cat("folds ...\n")
k = 5 # change to 5
set.seed(888)
folds <- createFolds(dt.train$TARGET, k = k, list = F)
## init preds
vec.xgb.pred.train <- rep(0, nrow(dt.train))
vec.xgb.pred.test <- rep(0, nrow(dt.test))
## init x.test
x.test <- sparse.model.matrix(~., data = dt.test[, !c("ID", "TARGET"), with = F])

watchlist <- list(val = dmx.valid, train = dmx.train) # change to dval
cbl <- seq(.1, 1, .1)
score <- as.numeric()
for (s in 1:length(cbl)){
    ## params
    params <- list(booster = "gbtree"
                   , nthread = 8
                   , objective = "binary:logistic"
                   , eval_metric = "auc"
                   , max_depth = 6 # 5
                   , subsample = .74 #.74
                   , min_child_weight = 1 # 1
                   # , colsample_bytree = 1 #.7
                   # , scale_pos_weight = spws[s]
                   , colsample_bylevel = cbl[s]
                   , eta = 0.022 #.0201
    )
    
    ## oof train
    for(i in 1:k){
        f <- folds == i
        dmx.train <- xgb.DMatrix(data = sparse.model.matrix(TARGET ~., data = dt.train[!f, setdiff(names(dt.train), c("ID")), with = F]), label = dt.train[!f]$TARGET)
        dmx.valid <- xgb.DMatrix(data = sparse.model.matrix(TARGET ~., data = dt.train[f, setdiff(names(dt.train), c("ID")), with = F]), label = dt.train[f]$TARGET)
        watchlist <- list(val = dmx.valid, train = dmx.train) # change to dval
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
        vec.xgb.pred.train[f] <- pred.valid
        print(paste("fold:", i, "valid auc:", auc(dt.train$TARGET[f], pred.valid)))
        
        # test
        pred.test <- predict(md.xgb, x.test)
        vec.xgb.pred.test <- vec.xgb.pred.test + pred.test / k
    }
    auc(dt.train$TARGET, vec.xgb.pred.train)
    score[s] <- auc(dt.train$TARGET, vec.xgb.pred.train)
}
tb <- data.table(cbl = cbl, score = score)
plot(tb$cbl, tb$score, type = "l")
# scale_pos_weight .9: 0.7986057
# scale_pos_weight .1: 0.7112151
# scale_pos_weight .5: 0.8102958