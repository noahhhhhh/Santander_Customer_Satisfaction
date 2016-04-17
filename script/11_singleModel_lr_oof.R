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
## test
x.test <- sparse.model.matrix(as.factor(TARGET) ~ ., data = dt.test)
#######################################################################################
## 2.0 train oof ######################################################################
#######################################################################################
## folds
cat("folds ...\n")
k = 5 # change to 5
set.seed(888)
folds <- createFolds(dt.train$TARGET, k = k, list = F)
## init preds
vec.meta.glm.train <- rep(0, nrow(dt.train))
vec.meta.glm.test <- rep(0, nrow(dt.test))
result.dval <- rep(0, 5)
## init grids
grid <- 10^seq(10, -2, length = 100)

## oof train
for(i in 1:k){
    f <- folds == i
    x.train <- sparse.model.matrix(as.factor(TARGET) ~ ., data = dt.train[!f])
    y.train <- as.factor(dt.train$TARGET[!f])
    x.valid <- sparse.model.matrix(as.factor(TARGET) ~ ., data = dt.train[f])
    y.valid <- as.factor(dt.train$TARGET[f])
    
    # train
    md.lasso <- glmnet(x.train, y.train, alpha = .5, lambda = grid, family = "binomial")
    plot(md.lasso)
    
    # cv to choose λ
    set.seed(888)
    cv.out <- cv.glmnet(x.train, y.train, alpha = .5, type.measure = "auc", family = "binomial")
    # plot(cv.out)
    bestlam <- cv.out$lambda.min
    # bestlam
    
    # valid
    pred.lasso <- predict(md.lasso , s = bestlam, type = "response", newx = x.valid)
    vec.meta.glm.test <- vec.meta.glm.test + predict(md.lasso , s = bestlam, type = "response", newx = x.test) / k
    
    vec.meta.glm.train[f] <- pred.lasso
    print(paste("k:", k, "auch:", auc(as.numeric(y.valid) - 1, vec.meta.glm.train[f])))
}
auc(dt.train$TARGET, vec.meta.glm.train)
# 0.7736692

save(vec.meta.glm.train, vec.meta.glm.test, file = "../data/Santander_Customer_Satisfaction/RData/dt_meta_1_glm.RData")
