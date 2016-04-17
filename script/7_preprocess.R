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
load("../data/Santander_Customer_Satisfaction/RData/dt_featureEngineered.RData")
load("../data/Santander_Customer_Satisfaction/RData/cols_selected.RData")
#######################################################################################
## one hot encoding ###################################################################
#######################################################################################
apply(dt.featureEngineered[, names(dt.featureEngineered)[grepl("^num_", names(dt.featureEngineered))], with = F]
      , 2
      , table)
# 0, 3, 6, 9, 12, other

cols.num <- names(dt.featureEngineered)[grepl("^num_", names(dt.featureEngineered))]
dt.num <- dt.featureEngineered[, cols.num, with = F]
dt.new <- data.table(matrix(nrow = nrow(dt.num), ncol = ncol(dt.num)))
setnames(dt.new, names(dt.new), paste0("new_factor", cols.num))
for(i in 1:length(cols.num)){
    new_num <- ifelse(dt.num[[i]] %in% c(0, 3, 6, 9, 12)
                      , dt.num[[i]]
                      , -1
    )
    dt.new[[i]] <- as.factor(new_num)
}

dummies <- dummyVars(~ ., data = dt.new)
dt.new.factor <- predict(dummies, newdata = dt.new)

#######################################################################################
## preprocess #########################################################################
#######################################################################################
preProcValues <- preProcess(dt.featureEngineered[, !c(cols.num, "ID", "TARGET"), with = F], method = c("center", "scale"))
dt.featureEngineered.scale <- predict(preProcValues, dt.featureEngineered[, !c(cols.num, "ID", "TARGET"), with = F])
dt.featureEngineered.combine <- cbind(ID = dt.featureEngineered$ID, TARGET = dt.featureEngineered$TARGET, dt.featureEngineered.scale, dt.new.factor)

#######################################################################################
## 1.0 remove zero vars ###############################################################
#######################################################################################
dim(dt.featureEngineered.combine)
# [1] 151838    371
nzv <- nearZeroVar(dt.featureEngineered.combine[, !c("ID", "TARGET"), with = F], saveMetrics= T)
nzv.train <- nearZeroVar(dt.featureEngineered.combine[TARGET >= 0, ][, !c("ID", "TARGET"), with = F], saveMetrics= T)
nzv.test <- nearZeroVar(dt.featureEngineered.combine[TARGET == -1, ][, !c("ID", "TARGET"), with = F], saveMetrics= T)

cols.zeroVar.all <- rownames(nzv[nzv$zeroVar == T, ])
cols.zeroVar.train <- rownames(nzv.train[nzv.train$zeroVar == T, ])
cols.zeroVar.test <- rownames(nzv.test[nzv.test$zeroVar == T, ])

## all, train, and test difference on zeroVar
setdiff(cols.zeroVar.all, cols.zeroVar.train)
# character(0)
setdiff(cols.zeroVar.train, cols.zeroVar.all)
# character(0)

dt.featureEngineered.combine <- dt.featureEngineered.combine[, !cols.zeroVar.all, with = F]
dim(dt.featureEngineered.combine)
# [1] 151838    337

#######################################################################################
## 2.0 remove duplicates ##############################################################
#######################################################################################
features_pair <- combn(names(dt.featureEngineered.combine), 2, simplify = F)
toRemove <- c()
for(pair in features_pair) {
    f1 <- pair[1]
    f2 <- pair[2]
    
    if (!(f1 %in% toRemove) & !(f2 %in% toRemove)) {
        if (all(dt.featureEngineered.combine[[f1]] == dt.featureEngineered.combine[[f2]])) {
            cat(f1, "and", f2, "are equals.\n")
            toRemove <- c(toRemove, f2)
        }
    }
}
dt.featureEngineered.combine <- dt.featureEngineered.combine[, !toRemove, with = F]
dim(dt.featureEngineered.combine)
# [1] 151838    310

#######################################################################################
## save ###############################################################################
#######################################################################################
newnames <- gsub(".", "_", names(dt.featureEngineered.combine))
setnames(dt.featureEngineered.combine, names(dt.featureEngineered.combine), newnames)
save(dt.featureEngineered.combine, file = "../data/Santander_Customer_Satisfaction/RData/dt_featureEngineered_combine.RData")
