setwd("/Volumes/Data Science/Google Drive/data_science_competition/kaggle/Santander_Customer_Satisfaction/")
rm(list = ls()); gc();
require(data.table)
require(caret)
require(purrr)
source("utilities/preprocess.R")

load("../data/Santander_Customer_Satisfaction/RData/dt_explored.RData")
#######################################################################################
## 1.0 remove zero vars ###############################################################
#######################################################################################
dim(dt.explored)
# [1] 151838    371
nzv <- nearZeroVar(dt.explored[, !c("ID", "TARGET"), with = F], saveMetrics= T)
nzv.train <- nearZeroVar(dt.explored[TARGET >= 0, ][, !c("ID", "TARGET"), with = F], saveMetrics= T)
nzv.test <- nearZeroVar(dt.explored[TARGET == -1, ][, !c("ID", "TARGET"), with = F], saveMetrics= T)

cols.zeroVar.all <- rownames(nzv[nzv$zeroVar == T, ])
cols.zeroVar.train <- rownames(nzv.train[nzv.train$zeroVar == T, ])
cols.zeroVar.test <- rownames(nzv.test[nzv.test$zeroVar == T, ])

## all, train, and test difference on zeroVar
setdiff(cols.zeroVar.all, cols.zeroVar.train)
# character(0)
setdiff(cols.zeroVar.train, cols.zeroVar.all)
# character(0)

dt.explored <- dt.explored[, !cols.zeroVar.all, with = F]
dim(dt.explored)
# [1] 151838    337

#######################################################################################
## 2.0 remove duplicates ##############################################################
#######################################################################################
features_pair <- combn(names(dt.explored), 2, simplify = F)
toRemove <- c()
for(pair in features_pair) {
    f1 <- pair[1]
    f2 <- pair[2]
    
    if (!(f1 %in% toRemove) & !(f2 %in% toRemove)) {
        if (all(dt.explored[[f1]] == dt.explored[[f2]])) {
            cat(f1, "and", f2, "are equals.\n")
            toRemove <- c(toRemove, f2)
        }
    }
}
dt.explored <- dt.explored[, !toRemove, with = F]
dim(dt.explored)
# [1] 151838    310

#######################################################################################
## save ###############################################################################
#######################################################################################
dt.cleansed <- dt.explored
save(dt.cleansed, file = "../data/Santander_Customer_Satisfaction/RData/dt_cleansed.RData")


