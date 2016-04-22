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
detach("package:extraTrees", unload = T)
options( java.parameters = "-Xmx6g" )
require(extraTrees)
source("utilities/preprocess.R")
source("utilities/cv.R")
load("../data/Santander_Customer_Satisfaction/RData/dt_featureEngineered.RData")
# load("../data/Santander_Customer_Satisfaction/RData/dt_featureEngineered_combine.RData")
load("../data/Santander_Customer_Satisfaction/RData/cols_selected.RData")
#######################################################################################
## 1.0 train and test #################################################################
#######################################################################################
dt.featureEngineered <- dt.featureEngineered[, !c("lr"), with = F]
dt.featureEngineered <- dt.featureEngineered[, !names(dt.featureEngineered)[grepl("new", names(dt.featureEngineered))], with = F]

cat("prepare train, valid, and test data set...\n")
set.seed(888)
dt.train <- dt.featureEngineered[TARGET >= 0]
# dt.train <- dt.train[createFolds(dt.train$TARGET, k = 2, list = F) == 1]
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
vec.et.pred.train <- rep(0, nrow(dt.train))
vec.et.pred.test <- rep(0, nrow(dt.test))

score <- numeric()

## oof train
for(i in 1:k){
    f <- folds == i
    dtrain <- dt.train[!f]
    dval <- dt.train[f]
    
    md.et <- extraTrees(x = model.matrix(TARGET ~., data = dtrain[, !c("ID"), with = F])
                        , y = as.factor(dtrain$TARGET)
                        , mtry = 15
                        , nodesize = 10
                        , numThreads = 8
                        , ntree = 2000
                        , numRandomCuts = 2)
    # valid
    pred.valid <- predict(md.et, newdata = model.matrix(TARGET ~., data = dval[, !c("ID"), with = F]), probability = T)[, 1]
    vec.et.pred.train[f] <- pred.valid
    print(paste("fold:", i, "valid auc:", auc(dval$TARGET, pred.valid)))
    score[i] <- auc(dval$TARGET, pred.valid)
    
    # test
    pred.test <- predict(md.et, newdata = model.matrix(TARGET ~., data = dt.test[, !c("ID"), with = F]), probability = T)[, 1]
    vec.et.pred.test <- vec.et.pred.test + pred.test / k
}

mean(score)
sd(score)

auc(dt.train$TARGET, vec.et.pred.train)
# 0.7770218 oof k = 5
#######################################################################################
## submit #############################################################################
#######################################################################################
save(vec.et.pred.train, vec.et.pred.test, file = "../data/Santander_Customer_Satisfaction/RData/dt_meta_1_et.RData")
vec.et.fac.pred.train <- 1 - vec.et.pred.train
vec.et.fac.pred.test <- 1- vec.et.pred.test
save(vec.et.fac.pred.train, vec.et.fac.pred.test, file = "../data/Santander_Customer_Satisfaction/RData/dt_meta_1_et_factor.RData")
