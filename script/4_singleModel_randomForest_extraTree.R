setwd("/Volumes/Data Science/Google Drive/data_science_competition/kaggle/Santander_Customer_Satisfaction/")
rm(list = ls()); gc();
require(data.table)
require(purrr)
require(caret)
require(Metrics)
require(ggplot2)
require(ranger)
require(caTools)
source("utilities/preprocess.R")
source("utilities/cv.R")
load("../data/Santander_Customer_Satisfaction/RData/dt_cleansed.RData")
#######################################################################################
## 1.0 train, valid, test #############################################################
#######################################################################################
cat("prepare train, valid, and test data set...\n")
set.seed(888)
ind.train <- createDataPartition(dt.cleansed[TARGET >= 0]$TARGET, p = .8, list = F) # remember to change it to .66
dt.train <- dt.cleansed[TARGET >= 0][ind.train]
dt.valid <- dt.cleansed[TARGET >= 0][-ind.train]
dt.test <- dt.cleansed[TARGET == -1]
dim(dt.train); dim(dt.valid); dim(dt.test)

table(dt.train$TARGET)
table(dt.valid$TARGET)

#######################################################################################
## 2.0 randomForest cv ################################################################
#######################################################################################
cat("folds ...\n")
k <- 5
set.seed(888)
folds <- createFolds(dt.train$TARGET, k = k, list = F)
cat("cv ...\n")
vec.result.dval <- rep(0, k)
vec.result.valid <- rep(0, k)
for(i in 1:k){
    f <- folds == i
    dtrain <- dt.train[!f]
    dval <- dt.train[f]
    md.rf <- ranger(TARGET ~.
                    , data = dt.train
                    , importance = "impurity"
                    , write.forest = T
                    , replace = T
                    , num.threads = 8
                    , seed = 888
                    , verbose = T)
    pred.dval.rf <- predict(md.rf, dval)$predictions
    vec.result.dval[i] <- auc(dval$TARGET, pred.dval.rf)
    
    pred.valid.rf <- predict(md.rf, dt.valid)$predictions
    vec.result.valid[i] <- auc(dt.valid$TARGET, pred.valid.rf)
}
df.summary <- as.data.frame(mean.dval = mean(vec.result.dval)
                            , max.dval = max(vec.result.dval)
                            , min.dval = min(vec.result.dval)
                            , sd.dval = sd(vec.result.dval)
                            , mean.valid = mean(vec.result.valid)
                            , max.valid = max(vec.result.valid)
                            , min.valid = min(vec.result.valid)
                            , sd.valid = sd(vec.result.valid)
                            )
df.summary
# benchmark
# round   eta mcw md  ss csb mean.dval  max.dval  min.dval    sd.dval mean.valid max.valid min.vaild    sd.valid
# 1     1 0.025   1  9 0.9 0.5 0.8355522 0.8502162 0.8199273 0.01114614  0.8434327 0.8455512 0.8414356 0.001604345

#######################################################################################
## 3.0 extraTree cv ###################################################################
#######################################################################################
options(java.parameters = "-Xmx8g" )
require(extraTrees)
cat("folds ...\n")
k <- 5
set.seed(888)
folds <- createFolds(dt.train$TARGET, k = k, list = F)
cat("cv ...\n")
vec.result.dval <- rep(0, k)
vec.result.valid <- rep(0, k)
for(i in 1:k){
    f <- folds == i
    dtrain <- dt.train[!f]
    dval <- dt.train[f]
    md.et <- extraTrees(dtrain[, !c("ID", "TARGET"), with = F]
                        , dtrain$TARGET
                        , numThreads = 8
                        )
    pred.dval.et <- predict(md.et, dval[, !c("ID", "TARGET"), with = F])
    vec.result.dval[i] <- auc(dval$TARGET, pred.dval.et)
    
    pred.valid.et <- predict(md.et, dt.valid[, !c("ID", "TARGET"), with = F])
    vec.result.valid[i] <- auc(dt.valid$TARGET, pred.valid.et)
}
df.summary <- data.table(mean.dval = mean(vec.result.dval)
                            , max.dval = max(vec.result.dval)
                            , min.dval = min(vec.result.dval)
                            , sd.dval = sd(vec.result.dval)
                            , mean.valid = mean(vec.result.valid)
                            , max.valid = max(vec.result.valid)
                            , min.valid = min(vec.result.valid)
                            , sd.valid = sd(vec.result.valid)
)
df.summary
# benchmark
# mean.dval max.dval  min.dval      sd.dval mean.valid max.valid min.valid    sd.valid
# 1: 0.9965538 0.997758 0.9952437 0.0009579658  0.7736706 0.7752181 0.7720287 0.001177851