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
require(RSofia)
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
vec.svm.pred.train <- rep(0, nrow(dt.train))
vec.svm.pred.test <- rep(0, nrow(dt.test))
## init score
score <- numeric()

## oof train
for(i in 1:k){
    f <- folds == i
    dtrain <- dt.train[!f]
    dval <- dt.train[f]
    md.sofia <- sofia(TARGET ~ .
                      , data = dtrain[, !c("ID"), with = F]
                      # , lambda = 1e-3
                      # , iiterations = 1e+18
                      # , random_seed = 1234
                      , learner_type = 'logreg-pegasos'
                      # , eta_type = 'pegasos'
                      # , loop_type = 'roc'
                      # , rank_step_probability = 0.5
                      # , passive_aggressive_c = 1e+07
                      # , passive_aggressive_lambda = 1e+3
                      # # , dimensionality = 49
                      # , perceptron_margin_size = 1
                      # , training_objective = F
                      # , hash_mask_bits = 0
                      # , verbose = T
                      # , reserve = 1
    )
    # valid
    pred.valid <- predict(md.sofia, newdata = dval[, !c("ID"), with = F], prediction_type = "logistic")
    vec.svm.pred.train[f] <- pred.valid
    print(paste("fold:", i, "valid auc:", auc(dtrain$TARGET, pred.valid)))
    score[i] <- auc(dt.train$TARGET[f], pred.valid)
    
    # test
    pred.test <- predict(md.sofia, dt.test[, !c("ID"), with = F], "logistic")
    vec.svm.pred.test <- vec.svm.pred.test + pred.test / k
}

mean(score)
sd(score)

auc(dt.train$TARGET, vec.svm.pred.train)
# 0.83.. oof k = 5 
#######################################################################################
## submit #############################################################################
#######################################################################################
save(vec.svm.pred.train, vec.svm.pred.test, file = "../data/Santander_Customer_Satisfaction/RData/dt_meta_1_svm.RData")
