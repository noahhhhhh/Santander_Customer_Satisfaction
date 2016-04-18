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
vec.meta.h2o.train <- rep(0, nrow(dt.train))
vec.meta.h2o.test <- rep(0, nrow(dt.test))
result.dval <- rep(0, 5)
## oof train
for(i in 1:k){
    f <- folds == i
    dtrain <- dt.train[!f]
    dval <- dt.train[f]
    h2o.init(ip = 'localhost', port = 54321, max_mem_size = '8g')
    h2o.train <- as.h2o(dtrain[, TARGET := as.factor(dtrain$TARGET)])
    h2o.val <- as.h2o(dval[, TARGET := as.factor(dval$TARGET)])
    
    
    md.h2o <- h2o.deeplearning(x = setdiff(names(dt.train), c("ID", "TARGET")),
                               y = "TARGET",
                               training_frame = h2o.train,
                               stopping_rounds = 5,
                               epochs = 20,
                               overwrite_with_best_model = TRUE,
                               activation = "RectifierWithDropout",
                               input_dropout_ratio = 0.2,
                               hidden = c(32, 64, 32),
                               rate = 0.001,
                               balance_classes = T,
                               l1 = 1e-4,
                               loss = "CrossEntropy",
                               distribution = "bernoulli",
                               stopping_metric = "AUC"
    )
    pred.val <- as.data.frame(h2o.predict(object = md.h2o, newdata = h2o.val))
    result.dval <- auc(dt.train[f]$TARGET, pred.val$p1)
    vec.meta.h2o.train[f] <- pred.val$p1 # save to meta feature, train
    
    pred.test <- as.data.frame(h2o.predict(object = md.h2o, newdata = h2o.test))
    vec.meta.h2o.test <- vec.meta.h2o.test + pred.test$p1 / k # save to meta feature, test
    
    # print(paste("cv:", i, "; h2o - oof:", auc(dval$TARGET, vec.meta.h2o.train[f])))
    # dt.train[, TARGET := as.integer(dt.train$TARGET) - 1]
}

# mean(result.dval)
# sd(result.dval)

auc(dt.train$TARGET, vec.meta.h2o.train)
## 0.8215519
save(vec.meta.h2o.train, vec.meta.h2o.test, file = "../data/Santander_Customer_Satisfaction/RData/dt_meta_1_h20.RData")
