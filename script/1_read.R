setwd("/Volumes/Data Science/Google Drive/data_science_competition/kaggle/Santander_Customer_Satisfaction/")
m(list = ls()); gc();
require(data.table)
require(bit64)
#######################################################################################
## 1.0 read ###########################################################################
#######################################################################################
dt.train.raw <- fread("../data/Santander_Customer_Satisfaction/train.csv", integer64 = "numeric")
dt.test.raw <- fread("../data/Santander_Customer_Satisfaction/test.csv", integer64 = "numeric")
dim(dt.train.raw); dim(dt.test.raw)
# [1] 76020   371
# [1] 75818   370

## check the balance of target of dt.train.raw
table(dt.train.raw$TARGET)
# 0     1 
# 73012  3008 : more than 20:1

#######################################################################################
## 2.0 combine ########################################################################
#######################################################################################
## set -1 to target to dt.test.raw
dt.test.raw[, TARGET := -1]
dim(dt.train.raw); dim(dt.test.raw)
# [1] 76020   371
# [1] 75818   371

## rearrange the column names of dt.test.raw
dt.test.raw <- dt.test.raw[, names(dt.train.raw), with = F]

## check if the column names are identical
identical(names(dt.train.raw), names(dt.test.raw))
# [1] TRUE

## combine
dt.all <- rbind(dt.train.raw, dt.test.raw)
dim(dt.all)
# [1] 151838    371

## check the number of dt.test.raw
dim(dt.all[dt.all$TARGET == -1])
# [1] 75818   371
dim(dt.test.raw)
# [1] 75818   371

#######################################################################################
## save ###############################################################################
#######################################################################################
save(dt.all, file = "../data/Santander_Customer_Satisfaction/RData/dt_all.RData")
