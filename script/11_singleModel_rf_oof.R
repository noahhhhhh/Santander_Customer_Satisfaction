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
require(h2o)
source("utilities/preprocess.R")
source("utilities/cv.R")
load("../data/Santander_Customer_Satisfaction/RData/dt_featureEngineered.RData")
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
h2o.init(ip = 'localhost', port = 54321, max_mem_size = '8g')
h2o.test <- as.h2o(dt.test[, !c("ID", "TARGET"), with = F])
#######################################################################################
## 2.0 train oof ######################################################################
#######################################################################################
## folds
cat("folds ...\n")
k = 5 # change to 5
set.seed(888)
folds <- createFolds(dt.train$TARGET, k = k, list = F)
## init preds
vec.meta.rf.train <- rep(0, nrow(dt.train))
vec.meta.rf.test <- rep(0, nrow(dt.test))
result.dval <- rep(0, 5)
## oof train
for(i in 1:k){
    f <- folds == i
    dtrain <- dt.train[!f]
    dval <- dt.train[f]
    h2o.init(ip = 'localhost', port = 54321, max_mem_size = '8g')
    h2o.train <- as.h2o(dtrain)
    h2o.val <- as.h2o(dval)
    
    md.rf <- h2o.randomForest(x = setdiff(names(dt.train), c("ID", "TARGET"))
                              , y = "TARGET"
                              , training_frame = h2o.train
                              # , model_id = NULL
                              # , validation_frame = h2o.val
                              , ignore_const_cols = T
                              # , checkpoint = NULL
                              , mtries = ceiling(sqrt(ncol(dt.train)))
                              , sample_rate = .8
                              , col_sample_rate_per_tree = .8
                              # , build_tree_one_node = NULL
                              , ntrees = 5000
                              # , max_depth = 5
                              # , min_rows = 5
                              # , nbins = NULL
                              # , nbins_top_level = NULL
                              # , nbins_cats = NULL
                              , binomial_double_trees = F
                              , balance_classes = T
                              # , max_after_balance_size = .1
                              # , offset_column = NULL
                              # , weights_column = NULL
                              # , fold_column = NULL
                              # , fold_assignment = NULL
                              , keep_cross_validation_predictions = F
                              # , score_each_iteration = F
                              # , score_tree_interval = 0
                              # , stopping_rounds = NULL
                              # , stopping_tolerance = NULL
                              # , max_runtime_secs = NULL
                              , seed = 1024 * i
                              # , nfolds = 5
                              , stopping_metric = "AUC"
                              
    )
    
    pred.val <- as.data.frame(h2o.predict(object = md.rf, newdata = h2o.val))
    result.dval[i] <- auc(dval$TARGET, pred.val$predict)
    vec.meta.rf.train[f] <- pred.val$predict # save to meta feature, train
    
    pred.test <- as.data.frame(h2o.predict(object = md.rf, newdata = h2o.test))
    vec.meta.rf.test <- vec.meta.rf.test + pred.test$predict / k # save to meta feature, test
    
    print(paste("cv:", i, "; rf - oof:", result.dval[i]))
}

# mean(result.dval)
# sd(result.dval)

auc(dt.train$TARGET, vec.meta.rf.train)
## 0.806
save(vec.meta.rf.train, vec.meta.rf.test, file = "../data/Santander_Customer_Satisfaction/RData/dt_meta_1_rf.RData")
