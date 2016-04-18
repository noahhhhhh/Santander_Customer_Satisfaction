setwd("/Volumes/Data Science/Google Drive/data_science_competition/kaggle/Santander_Customer_Satisfaction/")
rm(list = ls()); gc();
require(data.table)
require(purrr)
require(caret)
require(xgboost)
require(Matrix)
require(Ckmeans.1d.dp)
require(Metrics)
require(ggplot2)
require(combinat)
source("utilities/preprocess.R")
source("utilities/cv.R")
load("../data/Santander_Customer_Satisfaction/RData/dt_featureEngineered.RData")
# load("../data/Santander_Customer_Satisfaction/RData/dt_featureEngineered_combine.RData")
load("../data/Santander_Customer_Satisfaction/RData/cols_selected.RData")
#######################################################################################
## 1.0 train and test #################################################################
#######################################################################################
dt.featureEngineered <- dt.featureEngineered[, !c("lr"), with = F]
cat("prepare train, valid, and test data set...\n")
set.seed(888)
dt.train <- dt.featureEngineered[TARGET >= 0]
dt.test <- dt.featureEngineered[TARGET == -1]
dim(dt.train); dim(dt.test)

table(dt.train$TARGET)

#######################################################################################
## 2.0 train oof ######################################################################
#######################################################################################
## folds
cat("folds ...\n")
k = 5 # change to 5
set.seed(888)
folds <- createFolds(dt.train$TARGET, k = k, list = F)
## init preds
vec.xgb.reg.pred.train <- rep(0, nrow(dt.train))
vec.xgb.reg.pred.test <- rep(0, nrow(dt.test))
## init x.test
x.test <- sparse.model.matrix(~., data = dt.test[, !c("ID", "TARGET"), with = F])
## init some params
# watchlist <- list(val = dmx.valid, train = dmx.train) # change to dval
md <- 5
gamma <- 0
cbl <- .5 # .3* | .4 | .5 ; .2
ss <- .4 # .4; .9
spw <- 1
j <- 1
score <- numeric()

params <- list(booster = "gbtree"
               , nthread = 8
               , objective = "reg:linear"
               , eval_metric = "rmse"
               , max_depth = 5 # 5
               , subsample = ss #.74
               , min_child_weight = 1 # 1
               , gamma = gamma
               , colsample_bylevel = cbl
               , eta = 0.022 #.0201
               , scale_pos_weight = spw
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
    vec.xgb.reg.pred.train[f] <- pred.valid
    print(paste("fold:", i, "valid auc:", auc(dt.train$TARGET[f], pred.valid)))
    score[i] <- auc(dt.train$TARGET[f], pred.valid)
    
    # test
    pred.test <- predict(md.xgb, x.test)
    vec.xgb.reg.pred.test <- vec.xgb.reg.pred.test + pred.test / k
}

mean(score)
sd(score)

auc(dt.train$TARGET, vec.xgb.reg.pred.train)
# 0.8410063 oof k = 5 
#######################################################################################
## submit #############################################################################
#######################################################################################
save(vec.xgb.reg.pred.train, vec.xgb.reg.pred.test, file = "../data/Santander_Customer_Satisfaction/RData/dt_meta_1_xgb_reg.RData")
