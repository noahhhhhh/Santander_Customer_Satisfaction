setwd("/Volumes/Data Science/Google Drive/data_science_competition/kaggle/Santander_Customer_Satisfaction/")
rm(list = ls()); gc();
require(data.table)
require(purrr)
source("utilities/preprocess.R")

load("../data/Santander_Customer_Satisfaction/RData/dt_all.RData")
#######################################################################################
## 1.0 explore ########################################################################
#######################################################################################
## check NAs
ColNAs(dt.all, method = "mean")
# 0

## check class
unlist(lapply(dt.all, class))


## summarise column class
class <- unlist(lapply(dt.all, class))
table((class)[!names(class) %in% c("ID", "target")])
# integer integer64   numeric 
# 238        16       116 

## check unique values
ColUnique(dt.all)
# v3   v22   v24   v30   v31   v47   v52   v56   v66   v71   v74   v75   v79   v91  v107  v110  v112  v113  v125 
# 4 23420     5     8     4    10    13   131     3    12     3     4    18     8     8     3    23    38    91 

# check tabular summary of all factor columns
lapply(dt.all[, cols.factor, with = F], table)

#######################################################################################
## 2.0 save ###########################################################################
#######################################################################################
dt.explored <- dt.all
save(dt.explored, cols.factor, cols.integer, cols.numeric, file = "../data/BNP_Paribas_Cardif_Claims_Management/RData/dt_explored.RData")





