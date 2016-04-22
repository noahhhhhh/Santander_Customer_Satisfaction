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
require(class)
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
vec.xgb.knn.pred.train <- rep(0, nrow(dt.train))
vec.xgb.knn.pred.test <- rep(0, nrow(dt.test))

score <- numeric()

## oof train
knn <- 4
for(i in 1:k){
    f <- folds == i
    dtrain <- dt.train[!f]
    dval <- dt.train[f]
    
    vec.xgb.knn.pred.train[f] <- attributes(knn(train = dtrain[, !c("ID", "TARGET"), with = F]
                                   , test = dval[, !c("ID", "TARGET"), with = F]
                                   , cl = dtrain$TARGET
                                   , k = knn
                                   , prob = T
                                   , use.all = F))$prob
    print(paste("fold:", i, "valid auc:", auc(dval$TARGET, vec.xgb.knn.pred.train[f])))
    score[[i]] <- auc(dval$TARGET, vec.xgb.knn.pred.train[f])
    vec.xgb.knn.pred.test <- vec.xgb.knn.pred.test + attributes(knn(train = dtrain[, !c("ID", "TARGET"), with = F]
                                                , test = dt.test[, !c("ID"), with = F]
                                                , cl = dtrain$TARGET
                                                , k = knn
                                                , prob = T
                                                , use.all = F))$prob / k
    
}

mean(score)
sd(score)

auc(dt.train$TARGET, vec.xgb.knn.pred.train)
# 0.83.. oof k = 5 
#######################################################################################
## submit #############################################################################
#######################################################################################
save(vec.xgb.knn.pred.train, vec.xgb.knn.pred.test, file = "../data/Santander_Customer_Satisfaction/RData/dt_meta_1_xgb_knn.RData")
