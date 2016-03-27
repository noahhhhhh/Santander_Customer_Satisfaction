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
dmx.train <- xgb.DMatrix(data = data.matrix(dt.train[, !c("ID", "TARGET"), with = F]), label = dt.train$TARGET)
dmx.valid <- xgb.DMatrix(data = data.matrix(dt.valid[, !c("ID", "TARGET"), with = F]), label = dt.valid$TARGET)
x.test <- data.matrix(dt.test[, !c("ID", "TARGET"), with = F])

#######################################################################################
## 2.0 cv #############################################################################
#######################################################################################
params <- list(booster = "gbtree"
               , nthread = 8
               , objective = "binary:logistic"
               , eval_metric = "auc"
               , md = 9
               , ss = .9
               , mcw = 1
               , csb = .5
               , eta = .025)

df.summary <- myCV_xgb(dt.train
                       , setdiff(names(dt.train), c("ID", "TARGET"))
                       , dt.valid
                       , k = 5
                       , params)

df.summary
# benchmark - k = 10
# round   eta mcw md  ss csb mean.dval  max.dval  min.dval   sd.dval mean.valid max.valid min.vaild    sd.valid
# 1     1 0.025   1  9 0.9 0.5 0.8364064 0.8522113 0.8140117 0.0104776  0.8443013 0.8460127 0.8406825 0.001822651
# benckmark - k = 5
# round   eta mcw md  ss csb mean.dval  max.dval  min.dval    sd.dval mean.valid max.valid min.vaild    sd.valid
# 1     1 0.025   1  9 0.9 0.5 0.8355522 0.8502162 0.8199273 0.01114614  0.8434327 0.8455512 0.8414356 0.001604345

#######################################################################################
## 3.0 train ##########################################################################
#######################################################################################
watchlist <- list(val = dmx.valid, train = dmx.train) # change to dval
params <- list(booster = "gbtree"
               , nthread = 8
               , objective = "binary:logistic"
               , eval_metric = "auc"
               , max_depth = 9
               , subsample = .9
               , min_child_weight = 1
               , colsample_bytree = .5
               , eta = .025)
set.seed(888)
md.xgb <- xgb.train(params = params
                    , data = dmx.train
                    , nrounds = 100000 
                    , early.stop.round = 50
                    , watchlist = watchlist
                    , print.every.n = 200
                    , verbose = F
)
pred.valid <- predict(md.xgb, dmx.valid)
auc(dt.valid$TARGET, pred.valid)
# 0.8447578

## importance
print(xgb.importance(names(dt.train), model = md.xgb))
# Feature         Gain        Cover    Frequence
# 1:                   var3 3.205132e-01 2.341219e-01 9.246247e-02
# 2:            saldo_var25 2.193412e-01 1.883859e-01 4.519323e-02
# 3: saldo_medio_var44_ult3 7.751085e-02 9.630832e-02 1.454807e-01
# 4: saldo_medio_var5_hace2 3.344543e-02 3.760834e-02 4.175982e-02
# 5:         num_var45_ult3 3.172609e-02 3.838697e-02 5.094219e-02
# ---                                                              
# 110:     num_op_var39_hace2 3.157688e-05 9.253813e-06 3.992335e-04
# 111:            ind_var34_0 2.447589e-05 8.918021e-07 2.395401e-04
# 112:               ind_var1 1.630982e-05 1.117807e-04 4.790802e-04
# 113:              num_var44 5.476317e-06 3.039403e-06 7.984669e-05
# 114:   imp_amort_var34_ult1 4.120415e-06 4.850502e-07 7.984669e-05
#######################################################################################
## submit #############################################################################
#######################################################################################
pred.test <- predict(md.xgb, x.test)
submit <- data.table(ID = dt.test$ID, TARGET = pred.test)
write.csv(submit, file = "submission/2_single_xgb.csv", row.names = F)
# 0.836426

