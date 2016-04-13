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
## init some params
# watchlist <- list(val = dmx.valid, train = dmx.train) # change to dval
md <- 5
gamma <- 0
cbl <- seq(.1, 1, .1)
ss <- seq(.1, 1, .1)
j <- 1
score <- as.numeric()
tb <- data.table(gamma = 0, md = 0, ss = 0, cbl = 0, score = 0)

## oof cv
for(g in length(gamma):length(gamma)){
    for(m in length(md):length(md)){
        for(s in 1:length(ss)){
            for (c in 1:length(cbl)){
                ## params
                params <- list(booster = "gbtree"
                               , nthread = 8
                               , objective = "binary:logistic"
                               , eval_metric = "auc"
                               , max_depth = 5 # 5
                               , subsample = ss[s] #.74
                               , min_child_weight = 1 # 1
                               , gamma = gamma[g]
                               # , colsample_bytree = 1 #.7
                               # , scale_pos_weight = spws[s]
                               , colsample_bylevel = cbl[c]
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
                score[j] <- auc(dt.train$TARGET, vec.xgb.pred.train)
                
                tb <- rbind(tb, data.table(gamma = gamma[g], md = md[m], ss = ss[s], cbl = cbl[c], score = score[j]))
                
                j <- j + 1
            }
        }
    }
}


# plot(tb$cbl, tb$score, type = "l")
tb
# subsample = .74
# cbl     score
# 1: 0.1 0.8414968
# 2: 0.2 0.8406441
# 3: 0.3 0.8411211
# 4: 0.4 0.8411251
# 5: 0.5 0.8412948
# 6: 0.6 0.8411529
# 7: 0.7 0.8408487
# 8: 0.8 0.8404011
# 9: 0.9 0.8407047
# 10: 1.0 0.8404477


