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
require(Matrix)
require(glmnet)
source("utilities/preprocess.R")
source("utilities/cv.R")
load("../data/Santander_Customer_Satisfaction/RData/dt_cleansed.RData")
#######################################################################################
## count 0 ############################################################################
#######################################################################################
cnt0 <- apply(dt.cleansed[, !c("ID", "TARGET"), with = F], 1, function(x)sum(x == 0))


#######################################################################################
## count 1 ############################################################################
#######################################################################################
cnt1 <- apply(dt.cleansed[, !c("ID", "TARGET"), with = F], 1, function(x)sum(x == 1))

#######################################################################################
## kmeans #############################################################################
#######################################################################################
## scale
prep <- preProcess(dt.cleansed[, !c("ID", "TARGET"), with = F]
                                       # , method = c("range")
                                       , method = c("center", "scale")
                                       , verbose = T)
dt.cleansed.scale <- predict(prep, dt.cleansed)

set.seed(888)
md.keams <- kmeans(dt.cleansed.scale[, !c("ID", "TARGET"), with = F]
                                    , centers = 3
                                    , nstart = 20)
kmeans <- md.keams$cluster
temp <- data.table(kmeans = kmeans, target = dt.cleansed$TARGET)

ggplot(temp[dt.cleansed$TARGET >= 0, ], aes(x = as.factor(kmeans), fill = as.factor(target))) +
    geom_histogram(binwidth = 500)

#######################################################################################
## linear #############################################################################
#######################################################################################
cat("prepare train, valid, and test data set...\n")
set.seed(888)
dt.train <- dt.cleansed.scale[TARGET >= 0]
dt.test <- dt.cleansed.scale[TARGET == -1]
dim(dt.train); dim(dt.test)

table(dt.train$TARGET)

## folds
cat("folds ...\n")
k = 5 # change to 5
set.seed(888)
folds <- createFolds(dt.train$TARGET, k = k, list = F)
## init preds
vec.meta.glm.train <- rep(0, nrow(dt.train))
vec.meta.glm.test <- rep(0, nrow(dt.test))
## init grids
grid <- 10^seq(10, -2, length = 100)
## test
x.test <- sparse.model.matrix(as.factor(TARGET) ~ ., data = dt.test)

## oof prediction
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
    print(paste("k:", k, "done"))
}
auc(dt.train$TARGET, vec.meta.glm.train)
# 0.7736529 k = 10
# 0.7728757 k = 5

# save(vec.meta.glm.train, vec.meta.glm.test, file = "../data/Santander_Customer_Satisfaction/RData/glm_train.RData")

# coef
coef.lasso <- predict(md.lasso, type = "coefficients", s = bestlam)
coef.lasso
coef.lasso[coef.lasso != 0]
#######################################################################################
## var based  #########################################################################
#######################################################################################
## alike: var5, var8, var12
## alike: var1, var6, var14, var20, var 24, var30, var31, var32
## alike: var9, var10
## alike: v17, v33
## alike: v39, v40, v41
## alike: v7, v43
## var1
dt.cleansed[, names(dt.cleansed)[grepl("var1[^[:digit:]]|var1$+", names(dt.cleansed))], with = F]
new_var1_cnt0 <- apply(dt.cleansed[, names(dt.cleansed)[grepl("var1[^[:digit:]]|var1$+", names(dt.cleansed))], with = F], 1, function(x)sum(x == 0))
new_var1_ind_diff <- dt.cleansed$ind_var1_0 - dt.cleansed$ind_var1
new_var1_ind_sum <- dt.cleansed$ind_var1_0 + dt.cleansed$ind_var1
new_var1_num_diff <- dt.cleansed$num_var1_0 - dt.cleansed$num_var1
new_var1_num_sum <- dt.cleansed$num_var1_0 + dt.cleansed$num_var1

## var5
dt.cleansed[, names(dt.cleansed)[grepl("var5[^[:digit:]]|var5$+", names(dt.cleansed))], with = F]
new_var5_cnt0 <- apply(dt.cleansed[, names(dt.cleansed)[grepl("var5[^[:digit:]]|var5$+", names(dt.cleansed))], with = F], 1, function(x)sum(x == 0))
new_var5_ind_diff <- dt.cleansed$ind_var5_0 - dt.cleansed$ind_var5
new_var5_ind_sum <- dt.cleansed$ind_var5_0 + dt.cleansed$ind_var5
new_var5_num_diff <- dt.cleansed$num_var5_0 - dt.cleansed$num_var5
new_var5_num_sum <- dt.cleansed$num_var5_0 + dt.cleansed$num_var5
new_var5_saldo_diff_1_2 <- dt.cleansed$saldo_var5 - dt.cleansed$saldo_medio_var5_hace2
new_var5_saldo_sum_1_2 <- dt.cleansed$saldo_var5 + dt.cleansed$saldo_medio_var5_hace2
new_var5_saldo_diff_1_3 <- dt.cleansed$saldo_var5 - dt.cleansed$saldo_medio_var5_hace3
new_var5_saldo_sum_1_3 <- dt.cleansed$saldo_var5 + dt.cleansed$saldo_medio_var5_hace3
new_var5_saldo_diff_2_3 <- dt.cleansed$saldo_medio_var5_hace2 - dt.cleansed$saldo_medio_var5_hace3
new_var5_saldo_sum_2_3 <- dt.cleansed$saldo_medio_var5_hace2 + dt.cleansed$saldo_medio_var5_hace3
new_var5_saldo_ult_diff_1_3 <- dt.cleansed$saldo_medio_var5_ult1 - dt.cleansed$saldo_medio_var5_ult3
new_var5_saldo_ult_sum_1_3 <- dt.cleansed$saldo_medio_var5_ult1 + dt.cleansed$saldo_medio_var5_ult3

## var6
dt.cleansed[, names(dt.cleansed)[grepl("var6[^[:digit:]]|var6$+", names(dt.cleansed))], with = F]
new_var6_cnt0 <- apply(dt.cleansed[, names(dt.cleansed)[grepl("var6[^[:digit:]]|var6$+", names(dt.cleansed))], with = F], 1, function(x)sum(x == 0))
new_var6_ind_diff <- dt.cleansed$ind_var6_0 - dt.cleansed$ind_var6
new_var6_ind_sum <- dt.cleansed$ind_var6_0 + dt.cleansed$ind_var6
new_var6_num_diff <- dt.cleansed$num_var6_0 - dt.cleansed$num_var6
new_var6_num_sum <- dt.cleansed$num_var6_0 + dt.cleansed$num_var6

## var7
dt.cleansed[, names(dt.cleansed)[grepl("var7[^[:digit:]]|var7$+", names(dt.cleansed))], with = F]
new_var7_cnt0 <- apply(dt.cleansed[, names(dt.cleansed)[grepl("var7[^[:digit:]]|var7$+", names(dt.cleansed))], with = F], 1, function(x)sum(x == 0))
new_var7_imp_diff <- dt.cleansed$imp_var7_emit_ult1 - dt.cleansed$imp_var7_recib_ult1
new_var7_imp_sum <- dt.cleansed$imp_var7_emit_ult1 + dt.cleansed$imp_var7_recib_ult1
new_var7_ind_diff <- dt.cleansed$ind_var7_emit_ult1 - dt.cleansed$ind_var7_recib_ult1
new_var7_ind_sum <- dt.cleansed$ind_var7_emit_ult1 + dt.cleansed$ind_var7_recib_ult1
new_var7_num_diff <- dt.cleansed$num_var7_emit_ult1 - dt.cleansed$num_var7_recib_ult1
new_var7_num_sum <- dt.cleansed$num_var7_emit_ult1 + dt.cleansed$num_var7_recib_ult1

## var8
dt.cleansed[, names(dt.cleansed)[grepl("var8[^[:digit:]]|var8$+", names(dt.cleansed))], with = F]
new_var8_cnt0 <- apply(dt.cleansed[, names(dt.cleansed)[grepl("var8[^[:digit:]]|var8$+", names(dt.cleansed))], with = F], 1, function(x)sum(x == 0))
new_var8_ind_diff <- dt.cleansed$ind_var8_0 - dt.cleansed$ind_var8
new_var8_ind_sum <- dt.cleansed$ind_var8_0 + dt.cleansed$ind_var8
new_var8_num_diff <- dt.cleansed$num_var8_0 - dt.cleansed$num_var8
new_var8_num_sum <- dt.cleansed$num_var8_0 + dt.cleansed$num_var8
new_var8_saldo_diff_1_2 <- dt.cleansed$saldo_var8 - dt.cleansed$saldo_medio_var8_hace2
new_var8_saldo_sum_1_2 <- dt.cleansed$saldo_var8 + dt.cleansed$saldo_medio_var8_hace2
new_var8_saldo_diff_1_3 <- dt.cleansed$saldo_var8 - dt.cleansed$saldo_medio_var8_hace3
new_var8_saldo_sum_1_3 <- dt.cleansed$saldo_var8 + dt.cleansed$saldo_medio_var8_hace3
new_var8_saldo_diff_2_3 <- dt.cleansed$saldo_medio_var8_hace2 - dt.cleansed$saldo_medio_var8_hace3
new_var8_saldo_sum_2_3 <- dt.cleansed$saldo_medio_var8_hace2 + dt.cleansed$saldo_medio_var8_hace3
new_var8_saldo_ult_diff_1_3 <- dt.cleansed$saldo_medio_var8_ult1 - dt.cleansed$saldo_medio_var8_ult3
new_var8_saldo_ult_sum_1_3 <- dt.cleansed$saldo_medio_var8_ult1 + dt.cleansed$saldo_medio_var8_ult3

## var9
dt.cleansed[, names(dt.cleansed)[grepl("var9[^[:digit:]]|var9$+", names(dt.cleansed))], with = F]
new_var9_cnt0 <- apply(dt.cleansed[, names(dt.cleansed)[grepl("var9[^[:digit:]]|var9$+", names(dt.cleansed))], with = F], 1, function(x)sum(x == 0))
new_var9_ind_diff <- dt.cleansed$ind_var9_ult1 - dt.cleansed$ind_var9_cte_ult1
new_var9_ind_sum <- dt.cleansed$ind_var9_ult1 + dt.cleansed$ind_var9_cte_ult1

## var10
dt.cleansed[, names(dt.cleansed)[grepl("var10[^[:digit:]]|var10$+", names(dt.cleansed))], with = F]
new_var10_cnt0 <- apply(dt.cleansed[, names(dt.cleansed)[grepl("var10[^[:digit:]]|var10$+", names(dt.cleansed))], with = F], 1, function(x)sum(x == 0))
new_var10_ind_diff <- dt.cleansed$ind_var10_ult1 - dt.cleansed$ind_var10cte_ult1
new_var10_ind_sum <- dt.cleansed$ind_var10_ult1 + dt.cleansed$ind_var10cte_ult1

## var12
dt.cleansed[, names(dt.cleansed)[grepl("var12[^[:digit:]]|var12$+", names(dt.cleansed))], with = F]
new_var12_cnt0 <- apply(dt.cleansed[, names(dt.cleansed)[grepl("var12[^[:digit:]]|var12$+", names(dt.cleansed))], with = F], 1, function(x)sum(x == 0))
new_var12_ind_diff <- dt.cleansed$ind_var12_0 - dt.cleansed$ind_var12
new_var12_ind_sum <- dt.cleansed$ind_var12_0 + dt.cleansed$ind_var12
new_var12_num_diff <- dt.cleansed$num_var12_0 - dt.cleansed$num_var12
new_var12_num_sum <- dt.cleansed$num_var12_0 + dt.cleansed$num_var12
new_var12_saldo_diff_1_2 <- dt.cleansed$saldo_var12 - dt.cleansed$saldo_medio_var12_hace2
new_var12_saldo_sum_1_2 <- dt.cleansed$saldo_var12 + dt.cleansed$saldo_medio_var12_hace2
new_var12_saldo_diff_1_3 <- dt.cleansed$saldo_var12 - dt.cleansed$saldo_medio_var12_hace3
new_var12_saldo_sum_1_3 <- dt.cleansed$saldo_var12 + dt.cleansed$saldo_medio_var12_hace3
new_var12_saldo_diff_2_3 <- dt.cleansed$saldo_medio_var12_hace2 - dt.cleansed$saldo_medio_var12_hace3
new_var12_saldo_sum_2_3 <- dt.cleansed$saldo_medio_var12_hace2 + dt.cleansed$saldo_medio_var12_hace3
new_var12_saldo_ult_diff_1_3 <- dt.cleansed$saldo_medio_var12_ult1 - dt.cleansed$saldo_medio_var12_ult3
new_var12_saldo_ult_sum_1_3 <- dt.cleansed$saldo_medio_var12_ult1 + dt.cleansed$saldo_medio_var12_ult3

## var13
dt.cleansed[, names(dt.cleansed)[grepl("var13[^[:digit:]]|var13$", names(dt.cleansed))], with = F]
new_var13_cnt0 <- apply(dt.cleansed[, names(dt.cleansed)[grepl("var13[^[:digit:]]|var13$+", names(dt.cleansed))], with = F], 1, function(x)sum(x == 0))
new_var13_delta_imp_diff <- dt.cleansed$delta_imp_aport_var13_1y3 - dt.cleansed$delta_imp_reemb_var13_1y3
new_var13_ind_diff <- dt.cleansed$ind_var13_0 - dt.cleansed$ind_var13
new_var13_ind_sum <- dt.cleansed$ind_var13_0 + dt.cleansed$ind_var13
new_var13_ind_diff_corto <- dt.cleansed$ind_var13_corto_0 - dt.cleansed$ind_var13_corto
new_var13_ind_sum_corto <- dt.cleansed$ind_var13_corto_0 + dt.cleansed$ind_var13_corto
new_var13_ind_diff_largo <- dt.cleansed$ind_var13_largo_0 - dt.cleansed$ind_var13_largo
new_var13_ind_sum_largo <- dt.cleansed$ind_var13_largo_0 + dt.cleansed$ind_var13_largo
new_var13_ind_diff_0_largo_corto <- dt.cleansed$ind_var13_largo_0 - dt.cleansed$ind_var13_corto_0
new_var13_ind_sum_0_largo_corto <- dt.cleansed$ind_var13_largo_0 + dt.cleansed$ind_var13_corto_0
new_var13_ind_diff_0_largo_medio <- dt.cleansed$ind_var13_largo_0 - dt.cleansed$ind_var13_medio_0
new_var13_ind_sum_0_largo_medio <- dt.cleansed$ind_var13_largo_0 + dt.cleansed$ind_var13_medio_0
new_var13_ind_diff_0_medio_corto <- dt.cleansed$ind_var13_medio_0 - dt.cleansed$ind_var13_corto_0
new_var13_ind_sum_0_medio_corto <- dt.cleansed$ind_var13_medio_0 + dt.cleansed$ind_var13_corto_0
new_var13_ind_diff_largo_corto <- dt.cleansed$ind_var13_largo - dt.cleansed$ind_var13_corto
new_var13_ind_sum_largo_corto <- dt.cleansed$ind_var13_largo + dt.cleansed$ind_var13_corto
new_var13_num_meses_diff_largo_corto <- dt.cleansed$num_meses_var13_largo_ult3 - dt.cleansed$num_meses_var13_corto_ult3
new_var13_num_meses_sum_largo_corto <- dt.cleansed$num_meses_var13_largo_ult3 + dt.cleansed$num_meses_var13_corto_ult3
new_var13_num_meses_diff_largo_medio <- dt.cleansed$num_meses_var13_largo_ult3 - dt.cleansed$num_meses_var13_medio_ult3
new_var13_num_meses_sum_largo_medio <- dt.cleansed$num_meses_var13_largo_ult3 + dt.cleansed$num_meses_var13_medio_ult3
new_var13_num_meses_diff_medio_corto <- dt.cleansed$num_meses_var13_medio_ult3 - dt.cleansed$num_meses_var13_corto_ult3
new_var13_num_meses_sum_medio_corto <- dt.cleansed$num_meses_var13_medio_ult3 + dt.cleansed$num_meses_var13_corto_ult3
new_var13_num_diff_aport_resemb <- dt.cleansed$num_aport_var13_ult1 - dt.cleansed$num_reemb_var13_ult1
new_var13_num_sum_aport_resemb <- dt.cleansed$num_aport_var13_ult1 + dt.cleansed$num_reemb_var13_ult1
new_var13_num_diff <- dt.cleansed$num_var13_0 - dt.cleansed$num_var13
new_var13_num_sum <- dt.cleansed$num_var13_0 + dt.cleansed$num_var13
new_var13_num_diff_corto <- dt.cleansed$num_var13_corto_0 - dt.cleansed$num_var13_corto
new_var13_num_sum_corto <- dt.cleansed$num_var13_corto_0 + dt.cleansed$num_var13_corto
new_var13_num_diff_medio <- dt.cleansed$num_var13_medio_0 - dt.cleansed$num_var13_medio
new_var13_num_sum_medio <- dt.cleansed$num_var13_medio_0 + dt.cleansed$num_var13_medio
new_var13_num_diff_largo <- dt.cleansed$num_var13_largo_0 - dt.cleansed$num_var13_largo
new_var13_num_sum_largo <- dt.cleansed$num_var13_largo_0 + dt.cleansed$num_var13_largo
new_var13_num_diff_0_largo_corto <- dt.cleansed$num_var13_largo_0 - dt.cleansed$num_var13_corto_0
new_var13_num_sum_0_largo_corto <- dt.cleansed$num_var13_largo_0 + dt.cleansed$num_var13_corto_0
new_var13_num_diff_0_largo_medio <- dt.cleansed$num_var13_largo_0 - dt.cleansed$num_var13_medio_0
new_var13_num_sum_0_largo_medio <- dt.cleansed$num_var13_largo_0 + dt.cleansed$num_var13_medio_0
new_var13_num_diff_0_medio_corto <- dt.cleansed$num_var13_medio_0 - dt.cleansed$num_var13_corto_0
new_var13_num_sum_0_medio_corto <- dt.cleansed$num_var13_medio_0 + dt.cleansed$num_var13_corto_0
new_var13_saldo_diff_corto_hace_2_3 <- dt.cleansed$saldo_medio_var13_corto_hace2 - dt.cleansed$saldo_medio_var13_corto_hace3
new_var13_saldo_sum_corto_hace_2_3 <- dt.cleansed$saldo_medio_var13_corto_hace2 + dt.cleansed$saldo_medio_var13_corto_hace3
new_var13_saldo_diff_largo_hace_2_3 <- dt.cleansed$saldo_medio_var13_largo_hace2 - dt.cleansed$saldo_medio_var13_largo_hace3
new_var13_saldo_sum_largo_hace_2_3 <- dt.cleansed$saldo_medio_var13_largo_hace2 + dt.cleansed$saldo_medio_var13_largo_hace3
new_var13_saldo_diff_corto_ult_1_3 <- dt.cleansed$saldo_medio_var13_corto_ult1 - dt.cleansed$saldo_medio_var13_corto_ult3
new_var13_saldo_sum_corto_ult_1_3 <- dt.cleansed$saldo_medio_var13_corto_ult1 + dt.cleansed$saldo_medio_var13_corto_ult3
new_var13_saldo_diff_medio_ult_1_3 <- dt.cleansed$saldo_medio_var13_medio_ult1 - dt.cleansed$saldo_medio_var13_medio_ult3
new_var13_saldo_sum_medio_ult_1_3 <- dt.cleansed$saldo_medio_var13_medio_ult1 + dt.cleansed$saldo_medio_var13_medio_ult3
new_var13_saldo_diff_largo_ult_1_3 <- dt.cleansed$saldo_medio_var13_largo_ult1 - dt.cleansed$saldo_medio_var13_largo_ult3
new_var13_saldo_sum_largo_ult_1_3 <- dt.cleansed$saldo_medio_var13_largo_ult1 + dt.cleansed$saldo_medio_var13_largo_ult3
new_var13_saldo_diff_hace_2_largo_corto <- dt.cleansed$saldo_medio_var13_largo_hace2 - dt.cleansed$saldo_medio_var13_corto_hace2
new_var13_saldo_sum_hace_2_largo_corto <- dt.cleansed$saldo_medio_var13_largo_hace2 + dt.cleansed$saldo_medio_var13_corto_hace2
new_var13_saldo_diff_hace_2_largo_medio <- dt.cleansed$saldo_medio_var13_largo_hace2 - dt.cleansed$saldo_medio_var13_medio_hace2
new_var13_saldo_sum_hace_2_largo_medio <- dt.cleansed$saldo_medio_var13_largo_hace2 + dt.cleansed$saldo_medio_var13_medio_hace2
new_var13_saldo_diff_hace_2_medio_corto <- dt.cleansed$saldo_medio_var13_medio_hace2 - dt.cleansed$saldo_medio_var13_corto_hace2
new_var13_saldo_sum_hace_2_medio_corto <- dt.cleansed$saldo_medio_var13_medio_hace2 + dt.cleansed$saldo_medio_var13_corto_hace2
new_var13_saldo_diff_hace_3_largo_corto <- dt.cleansed$saldo_medio_var13_largo_hace3 - dt.cleansed$saldo_medio_var13_corto_hace3
new_var13_saldo_sum_hace_3_largo_corto <- dt.cleansed$saldo_medio_var13_largo_hace3 + dt.cleansed$saldo_medio_var13_corto_hace3
new_var13_saldo_diff_ult1_largo_corto <- dt.cleansed$saldo_medio_var13_largo_ult1 - dt.cleansed$saldo_medio_var13_corto_ult1
new_var13_saldo_sum_ult1_largo_corto <- dt.cleansed$saldo_medio_var13_largo_ult1 + dt.cleansed$saldo_medio_var13_corto_ult1
new_var13_saldo_diff_ult1_largo_medio <- dt.cleansed$saldo_medio_var13_largo_ult1 - dt.cleansed$saldo_medio_var13_medio_ult1
new_var13_saldo_sum_ult1_largo_medio <- dt.cleansed$saldo_medio_var13_largo_ult1 + dt.cleansed$saldo_medio_var13_medio_ult1
new_var13_saldo_diff_ult1_medio_corto <- dt.cleansed$saldo_medio_var13_medio_ult1 - dt.cleansed$saldo_medio_var13_corto_ult1
new_var13_saldo_sum_ult1_medio_corto <- dt.cleansed$saldo_medio_var13_medio_ult1 + dt.cleansed$saldo_medio_var13_corto_ult1
new_var13_saldo_diff_ult3_largo_corto <- dt.cleansed$saldo_medio_var13_largo_ult3 - dt.cleansed$saldo_medio_var13_corto_ult3
new_var13_saldo_sum_ult3_largo_corto <- dt.cleansed$saldo_medio_var13_largo_ult3 + dt.cleansed$saldo_medio_var13_corto_ult3
new_var13_saldo_diff_ult3_largo_medio <- dt.cleansed$saldo_medio_var13_largo_ult3 - dt.cleansed$saldo_medio_var13_medio_ult3
new_var13_saldo_sum_ult3_largo_medio <- dt.cleansed$saldo_medio_var13_largo_ult3 + dt.cleansed$saldo_medio_var13_medio_ult3
new_var13_saldo_diff_ult3_medio_corto <- dt.cleansed$saldo_medio_var13_medio_ult3 - dt.cleansed$saldo_medio_var13_corto_ult3
new_var13_saldo_sum_ult3_medio_corto <- dt.cleansed$saldo_medio_var13_medio_ult3 + dt.cleansed$saldo_medio_var13_corto_ult3
new_var13_saldo_diff_largo_medio <- dt.cleansed$saldo_var13_largo - dt.cleansed$saldo_var13_medio
new_var13_saldo_diff_largo_corto <- dt.cleansed$saldo_var13_largo - dt.cleansed$saldo_var13_corto
new_var13_saldo_diff_medio_corto <- dt.cleansed$saldo_var13_medio - dt.cleansed$saldo_var13_corto

## var14
dt.cleansed[, names(dt.cleansed)[grepl("var14[^[:digit:]]|var14$+", names(dt.cleansed))], with = F]
new_var14_cnt0 <- apply(dt.cleansed[, names(dt.cleansed)[grepl("var14[^[:digit:]]|var14$+", names(dt.cleansed))], with = F], 1, function(x)sum(x == 0))
new_var14_ind_diff <- dt.cleansed$ind_var14_0 - dt.cleansed$ind_var14
new_var14_ind_sum <- dt.cleansed$ind_var14_0 + dt.cleansed$ind_var14
new_var14_num_diff <- dt.cleansed$num_var14_0 - dt.cleansed$num_var14
new_var14_num_sum <- dt.cleansed$num_var14_0 + dt.cleansed$num_var14

## var15 (age?)
dt.cleansed[, names(dt.cleansed)[grepl("var15[^[:digit:]]|var15$+", names(dt.cleansed))], with = F]

## var16
dt.cleansed[, names(dt.cleansed)[grepl("var16[^[:digit:]]|var16$+", names(dt.cleansed))], with = F]
new_var16_cnt0 <- apply(dt.cleansed[, names(dt.cleansed)[grepl("var16[^[:digit:]]|var16$+", names(dt.cleansed))], with = F], 1, function(x)sum(x == 0))

## var17
dt.cleansed[, names(dt.cleansed)[grepl("var17[^[:digit:]]|var17$+", names(dt.cleansed))], with = F]
new_var17_cnt0 <- apply(dt.cleansed[, names(dt.cleansed)[grepl("var17[^[:digit:]]|var17$+", names(dt.cleansed))], with = F], 1, function(x)sum(x == 0))
new_var17_ind_diff <- dt.cleansed$ind_var17_0 - dt.cleansed$ind_var17
new_var17_ind_sum <- dt.cleansed$ind_var17_0 + dt.cleansed$ind_var17
new_var17_num_diff <- dt.cleansed$num_var17_0 - dt.cleansed$num_var17
new_var17_num_sum <- dt.cleansed$num_var17_0 + dt.cleansed$num_var17
new_var17_num_diff_hace3_aport_reemb <- dt.cleansed$num_aport_var17_hace3 - dt.cleansed$num_reemb_var17_hace3
new_var17_num_sum_hace3_aport_reemb <- dt.cleansed$num_aport_var17_hace3 + dt.cleansed$num_reemb_var17_hace3
new_var17_num_diff_ult1_aport_reemb <- dt.cleansed$num_aport_var17_ult1 - dt.cleansed$num_reemb_var17_ult1
new_var17_num_sum_ult1_aport_reemb <- dt.cleansed$num_aport_var17_ult1 + dt.cleansed$num_reemb_var17_ult1
new_var17_num_diff_ult1_trasp_in_out <- dt.cleansed$num_trasp_var17_in_ult1 - dt.cleansed$num_trasp_var33_in_ult1
new_var17_num_sum_ult1_trasp_in_out <- dt.cleansed$num_trasp_var17_in_ult1 + dt.cleansed$num_trasp_var33_in_ult1
new_var17_delta_diff_imp_aport_reemb <- dt.cleansed$delta_imp_aport_var17_1y3 - dt.cleansed$delta_imp_reemb_var17_1y3
new_var17_delta_sum_imp_aport_reemb <- dt.cleansed$delta_imp_aport_var17_1y3 + dt.cleansed$delta_imp_reemb_var17_1y3
new_var17_delta_diff_imp_trasp <- dt.cleansed$delta_imp_trasp_var17_in_1y3 - dt.cleansed$delta_imp_trasp_var17_out_1y3
new_var17_delta_sum_imp_trasp <- dt.cleansed$delta_imp_trasp_var17_in_1y3 + dt.cleansed$delta_imp_trasp_var17_out_1y3
new_var17_imp_diff_hace3_aport_reemb <- dt.cleansed$imp_aport_var13_hace3 - dt.cleansed$imp_reemb_var17_hace3
new_var17_imp_sum_hace3_aport_reemb <- dt.cleansed$imp_aport_var13_hace3 + dt.cleansed$imp_reemb_var17_hace3
new_var17_imp_diff_hace3_trasp_in_out <- dt.cleansed$imp_trasp_var17_in_hace3 - dt.cleansed$imp_trasp_var33_in_hace3
new_var17_imp_sum_hace3_trasp_in_out <- dt.cleansed$imp_trasp_var17_in_hace3 + dt.cleansed$imp_trasp_var33_in_hace3
new_var17_imp_diff_ult1_aport_reemb <- dt.cleansed$imp_aport_var13_ult1 - dt.cleansed$imp_reemb_var17_ult1
new_var17_imp_sum_ult1_aport_reemb <- dt.cleansed$imp_aport_var13_ult1 + dt.cleansed$imp_reemb_var17_ult1
new_var17_imp_diff_ult1_trasp_in_out <- dt.cleansed$imp_trasp_var17_in_ult1 - dt.cleansed$imp_trasp_var33_in_ult1
new_var17_imp_sum_ult1_trasp_in_out <- dt.cleansed$imp_trasp_var17_in_ult1 + dt.cleansed$imp_trasp_var33_in_ult1
new_var17_saldo_diff_medio_hace <- dt.cleansed$saldo_medio_var17_hace2 - dt.cleansed$saldo_medio_var17_hace3
new_var17_saldo_sum_medio_hace <- dt.cleansed$saldo_medio_var17_hace2 - dt.cleansed$saldo_medio_var17_hace3
new_var17_saldo_diff_medio_ult <- dt.cleansed$saldo_medio_var17_ult1 - dt.cleansed$saldo_medio_var17_ult3
new_var17_saldo_sum_medio_ult <- dt.cleansed$saldo_medio_var17_ult1 - dt.cleansed$saldo_medio_var17_ult3

## var18
dt.cleansed[, names(dt.cleansed)[grepl("var18[^[:digit:]]|var18$+", names(dt.cleansed))], with = F]
new_var18_cnt0 <- apply(dt.cleansed[, names(dt.cleansed)[grepl("var18[^[:digit:]]|var18$+", names(dt.cleansed))], with = F], 1, function(x)sum(x == 0))

## var20
dt.cleansed[, names(dt.cleansed)[grepl("var20[^[:digit:]]|var20$+", names(dt.cleansed))], with = F]
new_var20_cnt0 <- apply(dt.cleansed[, names(dt.cleansed)[grepl("var20[^[:digit:]]|var20$+", names(dt.cleansed))], with = F], 1, function(x)sum(x == 0))
new_var20_ind_diff <- dt.cleansed$ind_var20_0 - dt.cleansed$ind_var20
new_var20_ind_sum <- dt.cleansed$ind_var20_0 + dt.cleansed$ind_var20
new_var20_num_diff <- dt.cleansed$num_var20_0 - dt.cleansed$num_var20
new_var20_num_sum <- dt.cleansed$num_var20_0 + dt.cleansed$num_var20

## var22
dt.cleansed[, names(dt.cleansed)[grepl("var22[^[:digit:]]|var22$+", names(dt.cleansed))], with = F]
new_var22_cnt0 <- apply(dt.cleansed[, names(dt.cleansed)[grepl("var22[^[:digit:]]|var22$+", names(dt.cleansed))], with = F], 1, function(x)sum(x == 0))
new_var22_num_diff_hace <- dt.cleansed$num_var22_hace2 - dt.cleansed$num_var22_hace3
new_var22_num_sum_hace <- dt.cleansed$num_var22_hace2 + dt.cleansed$num_var22_hace3
new_var22_num_diff_ult <- dt.cleansed$num_var22_ult1 - dt.cleansed$num_var22_ult3
new_var22_num_sum_ult <- dt.cleansed$num_var22_ult1 + dt.cleansed$num_var22_ult3

## var24
dt.cleansed[, names(dt.cleansed)[grepl("var24[^[:digit:]]|var24$+", names(dt.cleansed))], with = F]
new_var24_cnt0 <- apply(dt.cleansed[, names(dt.cleansed)[grepl("var24[^[:digit:]]|var24$+", names(dt.cleansed))], with = F], 1, function(x)sum(x == 0))
new_var24_ind_diff <- dt.cleansed$ind_var24_0 - dt.cleansed$ind_var24
new_var24_ind_sum <- dt.cleansed$ind_var24_0 + dt.cleansed$ind_var24
new_var24_num_diff <- dt.cleansed$num_var24_0 - dt.cleansed$num_var24
new_var24_num_sum <- dt.cleansed$num_var24_0 + dt.cleansed$num_var24

## var25
dt.cleansed[, names(dt.cleansed)[grepl("var25[^[:digit:]]|var25$+", names(dt.cleansed))], with = F]
new_var25_cnt0 <- apply(dt.cleansed[, names(dt.cleansed)[grepl("var25[^[:digit:]]|var25$+", names(dt.cleansed))], with = F], 1, function(x)sum(x == 0))
new_var25_ind_diff <- dt.cleansed$ind_var25_0 - dt.cleansed$ind_var25_cte
new_var25_ind_sum <- dt.cleansed$ind_var25_0 + dt.cleansed$ind_var25_cte

## var26
dt.cleansed[, names(dt.cleansed)[grepl("var26[^[:digit:]]|var26$+", names(dt.cleansed))], with = F]
new_var26_cnt0 <- apply(dt.cleansed[, names(dt.cleansed)[grepl("var26[^[:digit:]]|var26$+", names(dt.cleansed))], with = F], 1, function(x)sum(x == 0))
new_var26_ind_diff <- dt.cleansed$ind_var26_0 - dt.cleansed$ind_var26_cte
new_var26_ind_sum <- dt.cleansed$ind_var26_0 + dt.cleansed$ind_var26_cte
new_var26_num_diff <- dt.cleansed$num_var26_0 - dt.cleansed$num_var26
new_var26_num_sum <- dt.cleansed$num_var26_0 + dt.cleansed$num_var26

## var29
dt.cleansed[, names(dt.cleansed)[grepl("var29[^[:digit:]]|var29$+", names(dt.cleansed))], with = F]
new_var29_cnt0 <- apply(dt.cleansed[, names(dt.cleansed)[grepl("var29[^[:digit:]]|var29$+", names(dt.cleansed))], with = F], 1, function(x)sum(x == 0))
new_var29_saldo_diff_hace <- dt.cleansed$saldo_medio_var29_hace2 - dt.cleansed$saldo_medio_var29_hace3
new_var29_saldo_sum_hace <- dt.cleansed$saldo_medio_var29_hace2 + dt.cleansed$saldo_medio_var29_hace3
new_var29_saldo_diff_ult <- dt.cleansed$saldo_medio_var29_ult1 - dt.cleansed$saldo_medio_var29_ult3
new_var29_saldo_sum_ult <- dt.cleansed$saldo_medio_var29_ult1 + dt.cleansed$saldo_medio_var29_ult3

## var30
dt.cleansed[, names(dt.cleansed)[grepl("var30[^[:digit:]]|var30$+", names(dt.cleansed))], with = F]
new_var30_cnt0 <- apply(dt.cleansed[, names(dt.cleansed)[grepl("var30[^[:digit:]]|var30$+", names(dt.cleansed))], with = F], 1, function(x)sum(x == 0))
new_var30_ind_diff <- dt.cleansed$ind_var30_0 - dt.cleansed$ind_var30
new_var30_ind_sum <- dt.cleansed$ind_var30_0 + dt.cleansed$ind_var30
new_var30_num_diff <- dt.cleansed$num_var30_0 - dt.cleansed$num_var30
new_var30_num_sum <- dt.cleansed$num_var30_0 + dt.cleansed$num_var30

## var31
dt.cleansed[, names(dt.cleansed)[grepl("var31[^[:digit:]]|var31$+", names(dt.cleansed))], with = F]
new_var31_cnt0 <- apply(dt.cleansed[, names(dt.cleansed)[grepl("var31[^[:digit:]]|var31$+", names(dt.cleansed))], with = F], 1, function(x)sum(x == 0))
new_var31_ind_diff <- dt.cleansed$ind_var31_0 - dt.cleansed$ind_var31
new_var31_ind_sum <- dt.cleansed$ind_var31_0 + dt.cleansed$ind_var31
new_var31_num_diff <- dt.cleansed$num_var31_0 - dt.cleansed$num_var31
new_var31_num_sum <- dt.cleansed$num_var31_0 + dt.cleansed$num_var31

## var32
dt.cleansed[, names(dt.cleansed)[grepl("var32[^[:digit:]]|var32$+", names(dt.cleansed))], with = F]
new_var32_cnt0 <- apply(dt.cleansed[, names(dt.cleansed)[grepl("var32[^[:digit:]]|var32$+", names(dt.cleansed))], with = F], 1, function(x)sum(x == 0))
new_var32_ind_diff <- dt.cleansed$ind_var32_0 - dt.cleansed$ind_var32_cte
new_var32_ind_sum <- dt.cleansed$ind_var32_0 + dt.cleansed$ind_var32_cte
new_var32_num_diff <- dt.cleansed$num_var32_0 - dt.cleansed$num_var32
new_var32_num_sum <- dt.cleansed$num_var32_0 + dt.cleansed$num_var32

## var33
dt.cleansed[, names(dt.cleansed)[grepl("var33[^[:digit:]]|var33$+", names(dt.cleansed))], with = F]
new_var33_cnt0 <- apply(dt.cleansed[, names(dt.cleansed)[grepl("var33[^[:digit:]]|var33$+", names(dt.cleansed))], with = F], 1, function(x)sum(x == 0))
new_var33_ind_diff <- dt.cleansed$ind_var33_0 - dt.cleansed$ind_var33
new_var33_ind_sum <- dt.cleansed$ind_var33_0 + dt.cleansed$ind_var33
new_var33_num_diff <- dt.cleansed$num_var33_0 - dt.cleansed$num_var33
new_var33_num_sum <- dt.cleansed$num_var33_0 + dt.cleansed$num_var33
new_var33_num_diff_ult1_trasp_in_out <- dt.cleansed$num_trasp_var33_in_ult1 - dt.cleansed$num_trasp_var33_in_ult1
new_var33_num_sum_ult1_trasp_in_out <- dt.cleansed$num_trasp_var33_in_ult1 + dt.cleansed$num_trasp_var33_in_ult1
new_var33_delta_diff_imp_aport_reemb <- dt.cleansed$delta_imp_aport_var33_1y3 - dt.cleansed$delta_imp_reemb_var33_1y3
new_var33_delta_sum_imp_aport_reemb <- dt.cleansed$delta_imp_aport_var33_1y3 + dt.cleansed$delta_imp_reemb_var33_1y3
new_var33_delta_diff_imp_trasp <- dt.cleansed$delta_imp_trasp_var33_in_1y3 - dt.cleansed$delta_imp_trasp_var33_out_1y3
new_var33_delta_sum_imp_trasp <- dt.cleansed$delta_imp_trasp_var33_in_1y3 + dt.cleansed$delta_imp_trasp_var33_out_1y3
new_var33_imp_diff_hace3_trasp_in_out <- dt.cleansed$imp_trasp_var33_in_hace3 - dt.cleansed$imp_trasp_var33_in_hace3
new_var33_imp_sum_hace3_trasp_in_out <- dt.cleansed$imp_trasp_var33_in_hace3 + dt.cleansed$imp_trasp_var33_in_hace3
new_var33_imp_diff_ult1_aport_reemb <- dt.cleansed$imp_aport_var13_ult1 - dt.cleansed$imp_reemb_var33_ult1
new_var33_imp_sum_ult1_aport_reemb <- dt.cleansed$imp_aport_var13_ult1 + dt.cleansed$imp_reemb_var33_ult1
new_var33_imp_diff_ult1_trasp_in_out <- dt.cleansed$imp_trasp_var33_in_ult1 - dt.cleansed$imp_trasp_var33_in_ult1
new_var33_imp_sum_ult1_trasp_in_out <- dt.cleansed$imp_trasp_var33_in_ult1 + dt.cleansed$imp_trasp_var33_in_ult1
new_var33_saldo_diff_medio_hace <- dt.cleansed$saldo_medio_var33_hace2 - dt.cleansed$saldo_medio_var33_hace3
new_var33_saldo_sum_medio_hace <- dt.cleansed$saldo_medio_var33_hace2 - dt.cleansed$saldo_medio_var33_hace3
new_var33_saldo_diff_medio_ult <- dt.cleansed$saldo_medio_var33_ult1 - dt.cleansed$saldo_medio_var33_ult3
new_var33_saldo_sum_medio_ult <- dt.cleansed$saldo_medio_var33_ult1 - dt.cleansed$saldo_medio_var33_ult3

## var34
dt.cleansed[, names(dt.cleansed)[grepl("var34[^[:digit:]]|var34$+", names(dt.cleansed))], with = F]
new_var34_cnt0 <- apply(dt.cleansed[, names(dt.cleansed)[grepl("var34[^[:digit:]]|var34$+", names(dt.cleansed))], with = F], 1, function(x)sum(x == 0))

## var37
dt.cleansed[, names(dt.cleansed)[grepl("var37[^[:digit:]]|var37$+", names(dt.cleansed))], with = F]
new_var37_cnt0 <- apply(dt.cleansed[, names(dt.cleansed)[grepl("var37[^[:digit:]]|var37$+", names(dt.cleansed))], with = F], 1, function(x)sum(x == 0))
new_var37_ind_diff <- dt.cleansed$ind_var37_0 - dt.cleansed$ind_var37_cte
new_var37_ind_sum <- dt.cleansed$ind_var37_0 + dt.cleansed$ind_var37_cte

## var39
dt.cleansed[, names(dt.cleansed)[grepl("var39[^[:digit:]]|var39$+", names(dt.cleansed))], with = F]
new_var39_cnt0 <- apply(dt.cleansed[, names(dt.cleansed)[grepl("var39[^[:digit:]]|var39$+", names(dt.cleansed))], with = F], 1, function(x)sum(x == 0))
new_var39_imp_op_diff_comer <- dt.cleansed$imp_op_var39_comer_ult3 - dt.cleansed$imp_op_var39_comer_ult1
new_var39_imp_op_sum_comer <- dt.cleansed$imp_op_var39_comer_ult3 + dt.cleansed$imp_op_var39_comer_ult1
new_var39_imp_op_diff_efect <- dt.cleansed$imp_op_var39_efect_ult3 - dt.cleansed$imp_op_var39_efect_ult1
new_var39_imp_op_sum_efect <- dt.cleansed$imp_op_var39_efect_ult3 + dt.cleansed$imp_op_var39_efect_ult1
new_var39_num_op_diff_comer <- dt.cleansed$num_op_var39_comer_ult3 - dt.cleansed$num_op_var39_comer_ult1
new_var39_num_op_sum_comer <- dt.cleansed$num_op_var39_comer_ult3 + dt.cleansed$num_op_var39_comer_ult1
new_var39_num_op_diff_efect <- dt.cleansed$num_op_var39_efect_ult3 - dt.cleansed$num_op_var39_efect_ult1
new_var39_num_op_sum_efect <- dt.cleansed$num_op_var39_efect_ult3 + dt.cleansed$num_op_var39_efect_ult1
new_var39_num_op_diff_ult <- dt.cleansed$num_op_var39_ult3 - dt.cleansed$num_op_var39_ult1
new_var39_num_op_sum_ult <- dt.cleansed$num_op_var39_ult3 + dt.cleansed$num_op_var39_ult1
new_var39_num_op_diff_hace <- dt.cleansed$num_op_var39_hace2 - dt.cleansed$num_op_var39_hace3
new_var39_num_op_sum_hace <- dt.cleansed$num_op_var39_hace2 + dt.cleansed$num_op_var39_hace3

## var40
dt.cleansed[, names(dt.cleansed)[grepl("var40[^[:digit:]]|var40$+", names(dt.cleansed))], with = F]
new_var40_cnt0 <- apply(dt.cleansed[, names(dt.cleansed)[grepl("var40[^[:digit:]]|var40$+", names(dt.cleansed))], with = F], 1, function(x)sum(x == 0))
new_var40_imp_op_diff_comer <- dt.cleansed$imp_op_var40_comer_ult3 - dt.cleansed$imp_op_var40_comer_ult1
new_var40_imp_op_sum_comer <- dt.cleansed$imp_op_var40_comer_ult3 + dt.cleansed$imp_op_var40_comer_ult1
new_var40_imp_op_diff_efect <- dt.cleansed$imp_op_var40_efect_ult3 - dt.cleansed$imp_op_var40_efect_ult1
new_var40_imp_op_sum_efect <- dt.cleansed$imp_op_var40_efect_ult3 + dt.cleansed$imp_op_var40_efect_ult1
new_var40_ind_diff <- dt.cleansed$ind_var40_0 - dt.cleansed$ind_var40
new_var40_ind_sum <- dt.cleansed$ind_var40_0 + dt.cleansed$ind_var40
new_var40_num_op_diff_comer <- dt.cleansed$num_op_var40_comer_ult3 - dt.cleansed$num_op_var40_comer_ult1
new_var40_num_op_sum_comer <- dt.cleansed$num_op_var40_comer_ult3 + dt.cleansed$num_op_var40_comer_ult1
new_var40_num_op_diff_efect <- dt.cleansed$num_op_var40_efect_ult3 - dt.cleansed$num_op_var40_efect_ult1
new_var40_num_op_sum_efect <- dt.cleansed$num_op_var40_efect_ult3 + dt.cleansed$num_op_var40_efect_ult1
new_var40_num_op_diff_ult <- dt.cleansed$num_op_var40_ult3 - dt.cleansed$num_op_var40_ult1
new_var40_num_op_sum_ult <- dt.cleansed$num_op_var40_ult3 + dt.cleansed$num_op_var40_ult1
new_var40_num_op_diff_hace <- dt.cleansed$num_op_var40_hace2 - dt.cleansed$num_op_var40_hace3
new_var40_num_op_sum_hace <- dt.cleansed$num_op_var40_hace2 + dt.cleansed$num_op_var40_hace3
new_var40_num_diff <- dt.cleansed$num_var40_0 - dt.cleansed$num_var40
new_var40_num_sum <- dt.cleansed$num_var40_0 + dt.cleansed$num_var40

## var41
dt.cleansed[, names(dt.cleansed)[grepl("var41[^[:digit:]]|var41$+", names(dt.cleansed))], with = F]
new_var41_cnt0 <- apply(dt.cleansed[, names(dt.cleansed)[grepl("var41[^[:digit:]]|var41$+", names(dt.cleansed))], with = F], 1, function(x)sum(x == 0))
new_var41_imp_op_diff_comer <- dt.cleansed$imp_op_var41_comer_ult3 - dt.cleansed$imp_op_var41_comer_ult1
new_var41_imp_op_sum_comer <- dt.cleansed$imp_op_var41_comer_ult3 + dt.cleansed$imp_op_var41_comer_ult1
new_var41_imp_op_diff_efect <- dt.cleansed$imp_op_var41_efect_ult3 - dt.cleansed$imp_op_var41_efect_ult1
new_var41_imp_op_sum_efect <- dt.cleansed$imp_op_var41_efect_ult3 + dt.cleansed$imp_op_var41_efect_ult1
new_var41_num_op_diff_comer <- dt.cleansed$num_op_var41_comer_ult3 - dt.cleansed$num_op_var41_comer_ult1
new_var41_num_op_sum_comer <- dt.cleansed$num_op_var41_comer_ult3 + dt.cleansed$num_op_var41_comer_ult1
new_var41_num_op_diff_efect <- dt.cleansed$num_op_var41_efect_ult3 - dt.cleansed$num_op_var41_efect_ult1
new_var41_num_op_sum_efect <- dt.cleansed$num_op_var41_efect_ult3 + dt.cleansed$num_op_var41_efect_ult1
new_var41_num_op_diff_ult <- dt.cleansed$num_op_var41_ult3 - dt.cleansed$num_op_var41_ult1
new_var41_num_op_sum_ult <- dt.cleansed$num_op_var41_ult3 + dt.cleansed$num_op_var41_ult1
new_var41_num_op_diff_hace <- dt.cleansed$num_op_var41_hace2 - dt.cleansed$num_op_var41_hace3
new_var41_num_op_sum_hace <- dt.cleansed$num_op_var41_hace2 + dt.cleansed$num_op_var41_hace3

## var42
dt.cleansed[, names(dt.cleansed)[grepl("var42[^[:digit:]]|var42$+", names(dt.cleansed))], with = F]
new_var42_cnt0 <- apply(dt.cleansed[, names(dt.cleansed)[grepl("var42[^[:digit:]]|var42$+", names(dt.cleansed))], with = F], 1, function(x)sum(x == 0))
new_var42_num_diff <- dt.cleansed$num_var42_0 - dt.cleansed$num_var42
new_var42_num_sum <- dt.cleansed$num_var42_0 + dt.cleansed$num_var42

## var43
dt.cleansed[, names(dt.cleansed)[grepl("var43[^[:digit:]]|var43$+", names(dt.cleansed))], with = F]
new_var43_cnt0 <- apply(dt.cleansed[, names(dt.cleansed)[grepl("var43[^[:digit:]]|var43$+", names(dt.cleansed))], with = F], 1, function(x)sum(x == 0))
new_var43_ind_diff <- dt.cleansed$ind_var43_emit_ult1 - dt.cleansed$ind_var43_recib_ult1
new_var43_ind_sum <- dt.cleansed$ind_var43_emit_ult1 + dt.cleansed$ind_var43_recib_ult1
new_var43_num_diff <- dt.cleansed$num_var43_emit_ult1 - dt.cleansed$num_var43_recib_ult1
new_var43_num_sum <- dt.cleansed$num_var43_emit_ult1 + dt.cleansed$num_var43_recib_ult1

## var44
dt.cleansed[, names(dt.cleansed)[grepl("var44[^[:digit:]]|var44$+", names(dt.cleansed))], with = F]
new_var44_cnt0 <- apply(dt.cleansed[, names(dt.cleansed)[grepl("var44[^[:digit:]]|var44$+", names(dt.cleansed))], with = F], 1, function(x)sum(x == 0))
new_var44_delta_imp_diff_1y3 <- dt.cleansed$delta_imp_compra_var44_1y3 - dt.cleansed$delta_imp_venta_var44_1y3
new_var44_delta_imp_sum_1y3 <- dt.cleansed$delta_imp_compra_var44_1y3 + dt.cleansed$delta_imp_venta_var44_1y3
new_var44_delta_num_diff_1y3 <- dt.cleansed$delta_num_compra_var44_1y3 - dt.cleansed$delta_num_venta_var44_1y3
new_var44_delta_num_sum_1y3 <- dt.cleansed$delta_num_compra_var44_1y3 + dt.cleansed$delta_num_venta_var44_1y3
new_var44_imp_diff_hace_3 <- dt.cleansed$imp_compra_var44_hace3 - dt.cleansed$imp_venta_var44_hace3
new_var44_imp_sum_hace_3 <- dt.cleansed$imp_compra_var44_hace3 + dt.cleansed$imp_venta_var44_hace3
new_var44_imp_diff_ult_1 <- dt.cleansed$imp_compra_var44_ult1 - dt.cleansed$imp_venta_var44_ult1
new_var44_imp_sum_ult_1 <- dt.cleansed$imp_compra_var44_ult1 + dt.cleansed$imp_venta_var44_ult1
new_var44_ind_diff <- dt.cleansed$ind_var44_0 - dt.cleansed$ind_var44
new_var44_ind_sum <- dt.cleansed$ind_var44_0 + dt.cleansed$ind_var44
new_var44_num_diff_hace_3 <- dt.cleansed$num_compra_var44_hace3 - dt.cleansed$num_venta_var44_hace3
new_var44_num_sum_hace_3 <- dt.cleansed$num_compra_var44_hace3 + dt.cleansed$num_venta_var44_hace3
new_var44_num_diff_ult_1 <- dt.cleansed$num_compra_var44_ult1 - dt.cleansed$num_venta_var44_ult1
new_var44_num_sum_ult_1 <- dt.cleansed$num_compra_var44_ult1 + dt.cleansed$num_venta_var44_ult1
new_var44_num_diff <- dt.cleansed$num_var44_0 - dt.cleansed$num_var44
new_var44_num_sum <- dt.cleansed$num_var44_0 + dt.cleansed$num_var44
new_var44_saldo_diff_hace <- dt.cleansed$saldo_medio_var44_hace2 - dt.cleansed$saldo_medio_var44_hace3
new_var44_saldo_sum_hace <- dt.cleansed$saldo_medio_var44_hace2 + dt.cleansed$saldo_medio_var44_hace3
new_var44_saldo_diff_ult <- dt.cleansed$saldo_medio_var44_ult1 - dt.cleansed$saldo_medio_var44_ult3
new_var44_saldo_sum_ult <- dt.cleansed$saldo_medio_var44_ult1 + dt.cleansed$saldo_medio_var44_ult3

## var45
dt.cleansed[, names(dt.cleansed)[grepl("var45[^[:digit:]]|var45$+", names(dt.cleansed))], with = F]
new_var45_cnt0 <- apply(dt.cleansed[, names(dt.cleansed)[grepl("var45[^[:digit:]]|var45$+", names(dt.cleansed))], with = F], 1, function(x)sum(x == 0))
new_var45_num_diff_hace <- dt.cleansed$num_var45_hace2 - dt.cleansed$num_var45_hace3
new_var45_num_sum_hace <- dt.cleansed$num_var45_hace2 + dt.cleansed$num_var45_hace3
new_var45_num_diff_ult <- dt.cleansed$num_var45_ult1 - dt.cleansed$num_var45_ult3
new_var45_num_sum_ult <- dt.cleansed$num_var45_ult1 + dt.cleansed$num_var45_ult3

#######################################################################################
## knn ################################################################################
#######################################################################################
require(class)
dt.train <- dt.cleansed[TARGET >= 0]
dt.test <- dt.cleansed[TARGET == -1]
## folds
cat("folds ...\n")
k = 4
set.seed(888)
folds <- createFolds(dt.train$TARGET, k = k, list = F)
# oof knn
ls.knn.train <- list()
ls.knn.test <- list()
knn.train <- rep(0, nrow(dt.train))
knn.test <- rep(0, nrow(dt.test))

## number of nn
nnn <- c(2, 4, 8, 16, 32, 64, 128)
for(n in 1:length(nnn)){
    for(i in 1:k){
        print(paste("n:", nnn[n], "; k:", k, "start:", Sys.time()))
        f <- folds == i
        dtrain <- dt.train[!f]
        dval <- dt.train[f]
        dtest <- dt.test
        
        knn.train[f] <- attributes(knn(train = dtrain[, !c("ID", "TARGET"), with = F]
                            , test = dval[, !c("ID", "TARGET"), with = F]
                            , cl = dtrain$TARGET
                            , k = nnn[n]
                            , prob = T
                            , use.all = F))$prob
        print(paste("n:", nnn[n], "; k:", k, "end:", Sys.time()))
    }
    ls.knn.train[[n]] <- knn.train
    ls.knn.test[[n]] <- knn.test
}

ls.knn.test[[4]] <- knn.train
save(ls.knn.train, file = "../data/Santander_Customer_Satisfaction/RData/knn_train.RData")
#######################################################################################
## pca ################################################################################
#######################################################################################
pca <- prcomp(dt.cleansed[, !c("ID", "TARGET"), with = F]
              , center = T
              , scale. = T) 

pca.all <- pca$x
pca.var <- pca$sdev^2
pve <- pca.var/sum(pca.var)

plot(pve[1:100] , xlab =" Principal Component ", ylab=" Proportion of
Variance Explained ", ylim=c(0,1) ,type = 'b')

plot(cumsum(pve[1:100]), xlab=" Principal Component ", ylab ="Cumulative Proportion of
     Variance Explained ", ylim=c(0,1) ,type = 'b')
plot(pca.all[, 1][dt.cleansed$TARGET >= 0], pca.all[, 2][dt.cleansed$TARGET >= 0], cols = as.factor(dt.cleansed$TARGET[dt.cleansed$TARGET >= 0]))

#######################################################################################
## tnse ###############################################################################
#######################################################################################
## scale
prep <- preProcess(dt.cleansed[, !c("ID", "TARGET"), with = F]
                   , method = c("center", "scale")
                   , verbose = T)
dt.cleaned.scaled <- predict(prep, dt.cleansed)
## t-sne on all
require(Rtsne)
mx.cleaned.scaled <- data.matrix(dt.cleaned.scaled[, !c("ID", "TARGET"), with = F])
tsne.out <- Rtsne(mx.cleaned.scaled
                  , check_duplicates = F
                  , pca = F
                  , verbose = T
                  , perplexity = 30
                  , theta = .5
                  , dims = 2)
embedding <- as.data.frame(tsne.out$Y)[dt.cleansed$TARGET >= 0, ]
embedding$Class <- as.factor(sub("Class_", "", dt.cleansed$TARGET[dt.cleansed$TARGET >= 0]))

p <- ggplot(embedding, aes(x=V1, y=V2, color=Class)) +
    geom_point(size=1.25) +
    guides(colour = guide_legend(override.aes = list(size=6))) +
    xlab("") + ylab("") +
    ggtitle("t-SNE 2D Embedding of Betting Data") +
    theme_light(base_size=20) +
    theme(strip.background = element_blank(),
          strip.text.x     = element_blank(),
          axis.text.x      = element_blank(),
          axis.text.y      = element_blank(),
          axis.ticks       = element_blank(),
          axis.line        = element_blank(),
          panel.border     = element_blank())
p
# seems it does not seperate at all
mx.tsne.out <- tsne.out$Y
save(mx.tsne.out, file = "../data/Santander_Customer_Satisfaction/RData/dt_tsne_all.RData")

# try different variable sets
## t-sne on vars
require(Rtsne)
mx.cleaned.scaled <- data.matrix(dt.cleaned.scaled[, names(dt.cleansed)[grep("^var", names(dt.cleaned.scaled))], with = F])
tsne.out <- Rtsne(mx.cleaned.scaled
                  , check_duplicates = F
                  , pca = F
                  , verbose = T
                  , perplexity = 30
                  , theta = .5
                  , dims = 2)
embedding <- as.data.frame(tsne.out$Y)[dt.cleansed$TARGET >= 0, ]
embedding$Class <- as.factor(sub("Class_", "", dt.cleansed$TARGET[dt.cleansed$TARGET >= 0]))

p <- ggplot(embedding, aes(x=V1, y=V2, color=Class)) +
    geom_point(size=1.25) +
    guides(colour = guide_legend(override.aes = list(size=6))) +
    xlab("") + ylab("") +
    ggtitle("t-SNE 2D Embedding of Betting Data") +
    theme_light(base_size=20) +
    theme(strip.background = element_blank(),
          strip.text.x     = element_blank(),
          axis.text.x      = element_blank(),
          axis.text.y      = element_blank(),
          axis.ticks       = element_blank(),
          axis.line        = element_blank(),
          panel.border     = element_blank())
p

# try different variable sets
## t-sne on imp
require(Rtsne)
mx.cleaned.scaled <- data.matrix(dt.cleaned.scaled[, names(dt.cleansed)[grep("^imp", names(dt.cleaned.scaled))], with = F])
tsne.out <- Rtsne(mx.cleaned.scaled
                  , check_duplicates = F
                  , pca = F
                  , verbose = T
                  , perplexity = 30
                  , theta = .5
                  , dims = 2)
embedding <- as.data.frame(tsne.out$Y)[dt.cleansed$TARGET >= 0, ]
embedding$Class <- as.factor(sub("Class_", "", dt.cleansed$TARGET[dt.cleansed$TARGET >= 0]))

p <- ggplot(embedding, aes(x=V1, y=V2, color=Class)) +
    geom_point(size=1.25) +
    guides(colour = guide_legend(override.aes = list(size=6))) +
    xlab("") + ylab("") +
    ggtitle("t-SNE 2D Embedding of Betting Data") +
    theme_light(base_size=20) +
    theme(strip.background = element_blank(),
          strip.text.x     = element_blank(),
          axis.text.x      = element_blank(),
          axis.text.y      = element_blank(),
          axis.ticks       = element_blank(),
          axis.line        = element_blank(),
          panel.border     = element_blank())
p

# try different variable sets
## t-sne on num
require(Rtsne)
mx.cleaned.scaled <- data.matrix(dt.cleaned.scaled[, names(dt.cleansed)[grep("^num", names(dt.cleaned.scaled))], with = F])
tsne.out <- Rtsne(mx.cleaned.scaled
                  , check_duplicates = F
                  , pca = F
                  , verbose = T
                  , perplexity = 30
                  , theta = .5
                  , dims = 2)
embedding <- as.data.frame(tsne.out$Y)[dt.cleansed$TARGET >= 0, ]
embedding$Class <- as.factor(sub("Class_", "", dt.cleansed$TARGET[dt.cleansed$TARGET >= 0]))

p <- ggplot(embedding, aes(x=V1, y=V2, color=Class)) +
    geom_point(size=1.25) +
    guides(colour = guide_legend(override.aes = list(size=6))) +
    xlab("") + ylab("") +
    ggtitle("t-SNE 2D Embedding of Betting Data") +
    theme_light(base_size=20) +
    theme(strip.background = element_blank(),
          strip.text.x     = element_blank(),
          axis.text.x      = element_blank(),
          axis.text.y      = element_blank(),
          axis.ticks       = element_blank(),
          axis.line        = element_blank(),
          panel.border     = element_blank())
p

# try different variable sets
## t-sne on saldo
require(Rtsne)
mx.cleaned.scaled <- data.matrix(dt.cleaned.scaled[, names(dt.cleansed)[grep("^saldo", names(dt.cleaned.scaled))], with = F])
tsne.out <- Rtsne(mx.cleaned.scaled
                  , check_duplicates = F
                  , pca = F
                  , verbose = T
                  , perplexity = 30
                  , theta = .5
                  , dims = 2)
embedding <- as.data.frame(tsne.out$Y)[dt.cleansed$TARGET >= 0, ]
embedding$Class <- as.factor(sub("Class_", "", dt.cleansed$TARGET[dt.cleansed$TARGET >= 0]))

p <- ggplot(embedding, aes(x=V1, y=V2, color=Class)) +
    geom_point(size=1.25) +
    guides(colour = guide_legend(override.aes = list(size=6))) +
    xlab("") + ylab("") +
    ggtitle("t-SNE 2D Embedding of Betting Data") +
    theme_light(base_size=20) +
    theme(strip.background = element_blank(),
          strip.text.x     = element_blank(),
          axis.text.x      = element_blank(),
          axis.text.y      = element_blank(),
          axis.ticks       = element_blank(),
          axis.line        = element_blank(),
          panel.border     = element_blank())
p
#######################################################################################
## save ###############################################################################
#######################################################################################
dt.cleansed[, cnt0 := cnt0]
dt.cleansed[, cnt1 := cnt1]
dt.cleansed[, kmeans := kmeans]
load("../data/Santander_Customer_Satisfaction/RData/glm_train.RData")
dt.cleansed[, lr := c(vec.meta.glm.train, vec.meta.glm.test)]
## var1
dt.cleansed[, new_var1_cnt0 := new_var1_cnt0]
dt.cleansed[, new_var1_ind_diff := new_var1_ind_diff]
dt.cleansed[, new_var1_ind_sum := new_var1_ind_sum]
dt.cleansed[, new_var1_num_diff := new_var1_num_diff]
dt.cleansed[, new_var1_num_sum := new_var1_num_sum]
## var5
dt.cleansed[, new_var5_cnt0 := new_var5_cnt0]
dt.cleansed[, new_var5_ind_diff := new_var5_ind_diff]
dt.cleansed[, new_var5_ind_sum := new_var5_ind_sum]
dt.cleansed[, new_var5_num_diff := new_var5_num_diff]
dt.cleansed[, new_var5_num_sum := new_var5_num_sum]
dt.cleansed[, new_var5_saldo_diff_1_2 := new_var5_saldo_diff_1_2]
dt.cleansed[, new_var5_saldo_sum_1_2 := new_var5_saldo_sum_1_2]
dt.cleansed[, new_var5_saldo_diff_1_3 := new_var5_saldo_diff_1_3]
dt.cleansed[, new_var5_saldo_sum_1_3 := new_var5_saldo_sum_1_3]
dt.cleansed[, new_var5_saldo_diff_2_3 := new_var5_saldo_diff_2_3]
dt.cleansed[, new_var5_saldo_sum_2_3 := new_var5_saldo_sum_2_3]
dt.cleansed[, new_var5_saldo_ult_diff_1_3 := new_var5_saldo_ult_diff_1_3]
dt.cleansed[, new_var5_saldo_ult_sum_1_3 := new_var5_saldo_ult_sum_1_3]
## var6
dt.cleansed[, new_var6_cnt0 := new_var6_cnt0]
dt.cleansed[, new_var6_ind_diff := new_var6_ind_diff]
dt.cleansed[, new_var6_ind_sum := new_var6_ind_sum]
dt.cleansed[, new_var6_num_diff := new_var6_num_diff]
dt.cleansed[, new_var6_num_sum := new_var6_num_sum]
## var7
dt.cleansed[, new_var7_cnt0 := new_var7_cnt0]
dt.cleansed[, new_var7_imp_diff := new_var7_imp_diff]
dt.cleansed[, new_var7_imp_sum := new_var7_imp_sum]
dt.cleansed[, new_var7_ind_diff := new_var7_ind_diff]
dt.cleansed[, new_var7_ind_sum := new_var7_ind_sum]
dt.cleansed[, new_var7_num_diff := new_var7_num_diff]
dt.cleansed[, new_var7_num_sum := new_var7_num_sum]
## var8
dt.cleansed[, new_var8_cnt0 := new_var8_cnt0]
dt.cleansed[, new_var8_ind_diff := new_var8_ind_diff]
dt.cleansed[, new_var8_ind_sum := new_var8_ind_sum]
dt.cleansed[, new_var8_num_diff := new_var8_num_diff]
dt.cleansed[, new_var8_num_sum := new_var8_num_sum]
dt.cleansed[, new_var8_saldo_diff_1_2 := new_var8_saldo_diff_1_2]
dt.cleansed[, new_var8_saldo_sum_1_2 := new_var8_saldo_sum_1_2]
dt.cleansed[, new_var8_saldo_diff_1_3 := new_var8_saldo_diff_1_3]
dt.cleansed[, new_var8_saldo_sum_1_3 := new_var8_saldo_sum_1_3]
dt.cleansed[, new_var8_saldo_diff_2_3 := new_var8_saldo_diff_2_3]
dt.cleansed[, new_var8_saldo_sum_2_3 := new_var8_saldo_sum_2_3]
dt.cleansed[, new_var8_saldo_ult_diff_1_3 := new_var8_saldo_ult_diff_1_3]
dt.cleansed[, new_var8_saldo_ult_sum_1_3 := new_var8_saldo_ult_sum_1_3]
## var9
dt.cleansed[, new_var9_cnt0 := new_var9_cnt0]
dt.cleansed[, new_var9_ind_diff := new_var9_ind_diff]
dt.cleansed[, new_var9_ind_sum := new_var9_ind_sum]
## var10
dt.cleansed[, new_var10_cnt0 := new_var10_cnt0]
dt.cleansed[, new_var10_ind_diff := new_var10_ind_diff]
dt.cleansed[, new_var10_ind_sum := new_var10_ind_sum]
## var12
dt.cleansed[, new_var12_cnt0 := new_var12_cnt0]
dt.cleansed[, new_var12_ind_diff := new_var12_ind_diff]
dt.cleansed[, new_var12_ind_sum := new_var12_ind_sum]
dt.cleansed[, new_var12_num_diff := new_var12_num_diff]
dt.cleansed[, new_var12_num_sum := new_var12_num_sum]
dt.cleansed[, new_var12_saldo_diff_1_2 := new_var12_saldo_diff_1_2]
dt.cleansed[, new_var12_saldo_sum_1_2 := new_var12_saldo_sum_1_2]
dt.cleansed[, new_var12_saldo_diff_1_3 := new_var12_saldo_diff_1_3]
dt.cleansed[, new_var12_saldo_sum_1_3 := new_var12_saldo_sum_1_3]
dt.cleansed[, new_var12_saldo_diff_2_3 := new_var12_saldo_diff_2_3]
dt.cleansed[, new_var12_saldo_sum_2_3 := new_var12_saldo_sum_2_3]
dt.cleansed[, new_var12_saldo_ult_diff_1_3 := new_var12_saldo_ult_diff_1_3]
dt.cleansed[, new_var12_saldo_ult_sum_1_3 := new_var12_saldo_ult_sum_1_3]
## var13
dt.cleansed[, new_var13_cnt0 := new_var13_cnt0]
dt.cleansed[, new_var13_delta_imp_diff := new_var13_delta_imp_diff]
dt.cleansed[, new_var13_ind_diff := new_var13_ind_diff]
dt.cleansed[, new_var13_ind_sum := new_var13_ind_sum]
dt.cleansed[, new_var13_ind_diff_corto := new_var13_ind_diff_corto]
dt.cleansed[, new_var13_ind_sum_corto := new_var13_ind_sum_corto]
dt.cleansed[, new_var13_ind_diff_largo := new_var13_ind_diff_largo]
dt.cleansed[, new_var13_ind_sum_largo := new_var13_ind_sum_largo]
dt.cleansed[, new_var13_ind_diff_0_largo_corto := new_var13_ind_diff_0_largo_corto]
dt.cleansed[, new_var13_ind_sum_0_largo_corto := new_var13_ind_sum_0_largo_corto]
dt.cleansed[, new_var13_ind_diff_0_largo_medio := new_var13_ind_diff_0_largo_medio]
dt.cleansed[, new_var13_ind_sum_0_largo_medio := new_var13_ind_sum_0_largo_medio]
dt.cleansed[, new_var13_ind_diff_0_medio_corto := new_var13_ind_diff_0_medio_corto]
dt.cleansed[, new_var13_ind_sum_0_medio_corto := new_var13_ind_sum_0_medio_corto]
dt.cleansed[, new_var13_ind_diff_largo_corto := new_var13_ind_diff_largo_corto]
dt.cleansed[, new_var13_ind_sum_largo_corto := new_var13_ind_sum_largo_corto]
dt.cleansed[, new_var13_num_meses_diff_largo_corto := new_var13_num_meses_diff_largo_corto]
dt.cleansed[, new_var13_num_meses_sum_largo_corto := new_var13_num_meses_sum_largo_corto]
dt.cleansed[, new_var13_num_meses_diff_largo_medio := new_var13_num_meses_diff_largo_medio]
dt.cleansed[, new_var13_num_meses_sum_largo_medio := new_var13_num_meses_sum_largo_medio]
dt.cleansed[, new_var13_num_meses_diff_medio_corto := new_var13_num_meses_diff_medio_corto]
dt.cleansed[, new_var13_num_meses_sum_medio_corto := new_var13_num_meses_sum_medio_corto]
dt.cleansed[, new_var13_num_diff_aport_resemb := new_var13_num_diff_aport_resemb]
dt.cleansed[, new_var13_num_sum_aport_resemb := new_var13_num_sum_aport_resemb]
dt.cleansed[, new_var13_num_diff := new_var13_num_diff]
dt.cleansed[, new_var13_num_sum := new_var13_num_sum]
dt.cleansed[, new_var13_num_diff_corto := new_var13_num_diff_corto]
dt.cleansed[, new_var13_num_sum_corto := new_var13_num_sum_corto]
dt.cleansed[, new_var13_num_diff_medio := new_var13_num_diff_medio]
dt.cleansed[, new_var13_num_sum_medio := new_var13_num_sum_medio]
dt.cleansed[, new_var13_num_diff_largo := new_var13_num_diff_largo]
dt.cleansed[, new_var13_num_sum_largo := new_var13_num_sum_largo]
dt.cleansed[, new_var13_num_diff_0_largo_corto := new_var13_num_diff_0_largo_corto]
dt.cleansed[, new_var13_num_sum_0_largo_corto := new_var13_num_sum_0_largo_corto]
dt.cleansed[, new_var13_num_diff_0_largo_medio := new_var13_num_diff_0_largo_medio]
dt.cleansed[, new_var13_num_sum_0_largo_medio := new_var13_num_sum_0_largo_medio]
dt.cleansed[, new_var13_num_diff_0_medio_corto := new_var13_num_diff_0_medio_corto]
dt.cleansed[, new_var13_num_sum_0_medio_corto := new_var13_num_sum_0_medio_corto]
dt.cleansed[, new_var13_saldo_diff_corto_hace_2_3 := new_var13_saldo_diff_corto_hace_2_3]
dt.cleansed[, new_var13_saldo_sum_corto_hace_2_3 := new_var13_saldo_sum_corto_hace_2_3]
dt.cleansed[, new_var13_saldo_diff_largo_hace_2_3 := new_var13_saldo_diff_largo_hace_2_3]
dt.cleansed[, new_var13_saldo_sum_largo_hace_2_3 := new_var13_saldo_sum_largo_hace_2_3]
dt.cleansed[, new_var13_saldo_diff_corto_ult_1_3 := new_var13_saldo_diff_corto_ult_1_3]
dt.cleansed[, new_var13_saldo_sum_corto_ult_1_3 := new_var13_saldo_sum_corto_ult_1_3]
dt.cleansed[, new_var13_saldo_diff_medio_ult_1_3 := new_var13_saldo_diff_medio_ult_1_3]
dt.cleansed[, new_var13_saldo_sum_medio_ult_1_3 := new_var13_saldo_sum_medio_ult_1_3]
dt.cleansed[, new_var13_saldo_diff_largo_ult_1_3 := new_var13_saldo_diff_largo_ult_1_3]
dt.cleansed[, new_var13_saldo_sum_largo_ult_1_3 := new_var13_saldo_sum_largo_ult_1_3]
dt.cleansed[, new_var13_saldo_diff_hace_2_largo_corto := new_var13_saldo_diff_hace_2_largo_corto]
dt.cleansed[, new_var13_saldo_sum_hace_2_largo_corto := new_var13_saldo_sum_hace_2_largo_corto]
dt.cleansed[, new_var13_saldo_diff_hace_2_largo_medio := new_var13_saldo_diff_hace_2_largo_medio]
dt.cleansed[, new_var13_saldo_sum_hace_2_largo_medio := new_var13_saldo_sum_hace_2_largo_medio]
dt.cleansed[, new_var13_saldo_diff_hace_2_medio_corto := new_var13_saldo_diff_hace_2_medio_corto]
dt.cleansed[, new_var13_saldo_sum_hace_2_medio_corto := new_var13_saldo_sum_hace_2_medio_corto]
dt.cleansed[, new_var13_saldo_diff_hace_3_largo_corto := new_var13_saldo_diff_hace_3_largo_corto]
dt.cleansed[, new_var13_saldo_sum_hace_3_largo_corto := new_var13_saldo_sum_hace_3_largo_corto]
dt.cleansed[, new_var13_saldo_diff_ult1_largo_corto := new_var13_saldo_diff_ult1_largo_corto]
dt.cleansed[, new_var13_saldo_sum_ult1_largo_corto := new_var13_saldo_sum_ult1_largo_corto]
dt.cleansed[, new_var13_saldo_diff_ult1_largo_medio := new_var13_saldo_diff_ult1_largo_medio]
dt.cleansed[, new_var13_saldo_sum_ult1_largo_medio := new_var13_saldo_sum_ult1_largo_medio]
dt.cleansed[, new_var13_saldo_diff_ult1_medio_corto := new_var13_saldo_diff_ult1_medio_corto]
dt.cleansed[, new_var13_saldo_sum_ult1_medio_corto := new_var13_saldo_sum_ult1_medio_corto]
dt.cleansed[, new_var13_saldo_diff_ult3_largo_corto := new_var13_saldo_diff_ult3_largo_corto]
dt.cleansed[, new_var13_saldo_sum_ult3_largo_corto := new_var13_saldo_sum_ult3_largo_corto]
dt.cleansed[, new_var13_saldo_diff_ult3_largo_medio := new_var13_saldo_diff_ult3_largo_medio]
dt.cleansed[, new_var13_saldo_sum_ult3_largo_medio := new_var13_saldo_sum_ult3_largo_medio]
dt.cleansed[, new_var13_saldo_diff_ult3_medio_corto := new_var13_saldo_diff_ult3_medio_corto]
dt.cleansed[, new_var13_saldo_sum_ult3_medio_corto := new_var13_saldo_sum_ult3_medio_corto]
dt.cleansed[, new_var13_saldo_diff_largo_medio := new_var13_saldo_diff_largo_medio]
dt.cleansed[, new_var13_saldo_diff_largo_corto := new_var13_saldo_diff_largo_corto]
dt.cleansed[, new_var13_saldo_diff_medio_corto := new_var13_saldo_diff_medio_corto]
## var14
dt.cleansed[, new_var14_cnt0 := new_var14_cnt0]
dt.cleansed[, new_var14_ind_diff := new_var14_ind_diff]
dt.cleansed[, new_var14_ind_sum := new_var14_ind_sum]
dt.cleansed[, new_var14_num_diff := new_var14_num_diff]
dt.cleansed[, new_var14_num_sum := new_var14_num_sum]
## var16
dt.cleansed[, new_var16_cnt0 := new_var16_cnt0]
## var17
dt.cleansed[, new_var17_cnt0 := new_var17_cnt0]
dt.cleansed[, new_var17_ind_diff := new_var17_ind_diff]
dt.cleansed[, new_var17_ind_sum := new_var17_ind_sum]
dt.cleansed[, new_var17_num_diff := new_var17_num_diff]
dt.cleansed[, new_var17_num_sum := new_var17_num_sum]
dt.cleansed[, new_var17_num_diff_hace3_aport_reemb := new_var17_num_diff_hace3_aport_reemb]
dt.cleansed[, new_var17_num_sum_hace3_aport_reemb := new_var17_num_sum_hace3_aport_reemb]
dt.cleansed[, new_var17_num_diff_ult1_aport_reemb := new_var17_num_diff_ult1_aport_reemb]
dt.cleansed[, new_var17_num_sum_ult1_aport_reemb := new_var17_num_sum_ult1_aport_reemb]
dt.cleansed[, new_var17_num_diff_ult1_trasp_in_out := new_var17_num_diff_ult1_trasp_in_out]
dt.cleansed[, new_var17_num_sum_ult1_trasp_in_out := new_var17_num_sum_ult1_trasp_in_out]
dt.cleansed[, new_var17_delta_diff_imp_aport_reemb := new_var17_delta_diff_imp_aport_reemb]
dt.cleansed[, new_var17_delta_sum_imp_aport_reemb := new_var17_delta_sum_imp_aport_reemb]
dt.cleansed[, new_var17_delta_diff_imp_trasp := new_var17_delta_diff_imp_trasp]
dt.cleansed[, new_var17_delta_sum_imp_trasp := new_var17_delta_sum_imp_trasp]
dt.cleansed[, new_var17_imp_diff_hace3_aport_reemb := new_var17_imp_diff_hace3_aport_reemb]
dt.cleansed[, new_var17_imp_sum_hace3_aport_reemb := new_var17_imp_sum_hace3_aport_reemb]
dt.cleansed[, new_var17_imp_diff_hace3_trasp_in_out := new_var17_imp_diff_hace3_trasp_in_out]
dt.cleansed[, new_var17_imp_sum_hace3_trasp_in_out := new_var17_imp_sum_hace3_trasp_in_out]
dt.cleansed[, new_var17_imp_diff_ult1_aport_reemb := new_var17_imp_diff_ult1_aport_reemb]
dt.cleansed[, new_var17_imp_sum_ult1_aport_reemb := new_var17_imp_sum_ult1_aport_reemb]
dt.cleansed[, new_var17_imp_diff_ult1_trasp_in_out := new_var17_imp_diff_ult1_trasp_in_out]
dt.cleansed[, new_var17_imp_sum_ult1_trasp_in_out := new_var17_imp_sum_ult1_trasp_in_out]
dt.cleansed[, new_var17_saldo_diff_medio_hace := new_var17_saldo_diff_medio_hace]
dt.cleansed[, new_var17_saldo_sum_medio_hace := new_var17_saldo_sum_medio_hace]
dt.cleansed[, new_var17_saldo_diff_medio_ult := new_var17_saldo_diff_medio_ult]
dt.cleansed[, new_var17_saldo_sum_medio_ult := new_var17_saldo_sum_medio_ult]
## var18
dt.cleansed[, new_var18_cnt0 := new_var18_cnt0]
## var20
dt.cleansed[, new_var20_cnt0 := new_var20_cnt0]
dt.cleansed[, new_var20_ind_diff := new_var20_ind_diff]
dt.cleansed[, new_var20_ind_sum := new_var20_ind_sum]
dt.cleansed[, new_var20_num_diff := new_var20_num_diff]
dt.cleansed[, new_var20_num_sum := new_var20_num_sum]
## var22
dt.cleansed[, new_var22_cnt0 := new_var22_cnt0]
dt.cleansed[, new_var22_num_diff_hace := new_var22_num_diff_hace]
dt.cleansed[, new_var22_num_sum_hace := new_var22_num_sum_hace]
dt.cleansed[, new_var22_num_diff_ult := new_var22_num_diff_ult]
dt.cleansed[, new_var22_num_sum_ult := new_var22_num_sum_ult]
## var24
dt.cleansed[, new_var24_cnt0 := new_var24_cnt0]
dt.cleansed[, new_var24_ind_diff := new_var24_ind_diff]
dt.cleansed[, new_var24_ind_sum := new_var24_ind_sum]
dt.cleansed[, new_var24_num_diff := new_var24_num_diff]
dt.cleansed[, new_var24_num_sum := new_var24_num_sum]
## var25
dt.cleansed[, new_var25_cnt0 := new_var25_cnt0]
dt.cleansed[, new_var25_ind_diff := new_var25_ind_diff]
dt.cleansed[, new_var25_ind_sum := new_var25_ind_sum]
dt.cleansed[, new_var25_num_diff := new_var25_num_diff]
dt.cleansed[, new_var25_num_sum := new_var25_num_sum]
## var26
dt.cleansed[, new_var26_cnt0 := new_var26_cnt0]
dt.cleansed[, new_var26_ind_diff := new_var26_ind_diff]
dt.cleansed[, new_var26_ind_sum := new_var26_ind_sum]
dt.cleansed[, new_var26_num_diff := new_var26_num_diff]
dt.cleansed[, new_var26_num_sum := new_var26_num_sum]
## var29
dt.cleansed[, new_var29_cnt0 := new_var29_cnt0]
dt.cleansed[, new_var29_saldo_diff_hace := new_var29_saldo_diff_hace]
dt.cleansed[, new_var29_saldo_sum_hace := new_var29_saldo_sum_hace]
dt.cleansed[, new_var29_saldo_diff_ult := new_var29_saldo_diff_ult]
dt.cleansed[, new_var29_saldo_sum_ult := new_var29_saldo_sum_ult]
## var30
dt.cleansed[, new_var30_cnt0 := new_var30_cnt0]
dt.cleansed[, new_var30_ind_diff := new_var30_ind_diff]
dt.cleansed[, new_var30_ind_sum := new_var30_ind_sum]
dt.cleansed[, new_var30_num_diff := new_var30_num_diff]
dt.cleansed[, new_var30_num_sum := new_var30_num_sum]
## var31
dt.cleansed[, new_var31_cnt0 := new_var31_cnt0]
dt.cleansed[, new_var31_ind_diff := new_var31_ind_diff]
dt.cleansed[, new_var31_ind_sum := new_var31_ind_sum]
dt.cleansed[, new_var31_num_diff := new_var31_num_diff]
dt.cleansed[, new_var31_num_sum := new_var31_num_sum]
## var32
dt.cleansed[, new_var32_cnt0 := new_var32_cnt0]
dt.cleansed[, new_var32_ind_diff := new_var32_ind_diff]
dt.cleansed[, new_var32_ind_sum := new_var32_ind_sum]
dt.cleansed[, new_var32_num_diff := new_var32_num_diff]
dt.cleansed[, new_var32_num_sum := new_var32_num_sum]
## var33
dt.cleansed[, new_var33_cnt0 := new_var33_cnt0]
dt.cleansed[, new_var33_ind_diff := new_var33_ind_diff]
dt.cleansed[, new_var33_ind_sum := new_var33_ind_sum]
dt.cleansed[, new_var33_num_diff := new_var33_num_diff]
dt.cleansed[, new_var33_num_sum := new_var33_num_sum]
dt.cleansed[, new_var33_num_diff_ult1_trasp_in_out := new_var33_num_diff_ult1_trasp_in_out]
dt.cleansed[, new_var33_num_sum_ult1_trasp_in_out := new_var33_num_sum_ult1_trasp_in_out]
dt.cleansed[, new_var33_delta_diff_imp_aport_reemb := new_var33_delta_diff_imp_aport_reemb]
dt.cleansed[, new_var33_delta_sum_imp_aport_reemb := new_var33_delta_sum_imp_aport_reemb]
dt.cleansed[, new_var33_delta_diff_imp_trasp := new_var33_delta_diff_imp_trasp]
dt.cleansed[, new_var33_delta_sum_imp_trasp := new_var33_delta_sum_imp_trasp]
dt.cleansed[, new_var33_imp_diff_hace3_trasp_in_out := new_var33_imp_diff_hace3_trasp_in_out]
dt.cleansed[, new_var33_imp_sum_hace3_trasp_in_out := new_var33_imp_sum_hace3_trasp_in_out]
dt.cleansed[, new_var33_imp_diff_ult1_aport_reemb := new_var33_imp_diff_ult1_aport_reemb]
dt.cleansed[, new_var33_imp_sum_ult1_aport_reemb := new_var33_imp_sum_ult1_aport_reemb]
dt.cleansed[, new_var33_imp_diff_ult1_trasp_in_out := new_var33_imp_diff_ult1_trasp_in_out]
dt.cleansed[, new_var33_imp_sum_ult1_trasp_in_out := new_var33_imp_sum_ult1_trasp_in_out]
dt.cleansed[, new_var33_saldo_diff_medio_hace := new_var33_saldo_diff_medio_hace]
dt.cleansed[, new_var33_saldo_sum_medio_hace := new_var33_saldo_sum_medio_hace]
dt.cleansed[, new_var33_saldo_diff_medio_ult := new_var33_saldo_diff_medio_ult]
dt.cleansed[, new_var33_saldo_sum_medio_ult := new_var33_saldo_sum_medio_ult]
## var34
dt.cleansed[, new_var34_cnt0 := new_var34_cnt0]
## var37
dt.cleansed[, new_var37_cnt0 := new_var37_cnt0]
dt.cleansed[, new_var37_ind_diff := new_var37_ind_diff]
dt.cleansed[, new_var37_ind_sum := new_var37_ind_sum]
## var39
dt.cleansed[, new_var39_cnt0 := new_var39_cnt0]
dt.cleansed[, new_var39_imp_op_diff_comer := new_var39_imp_op_diff_comer]
dt.cleansed[, new_var39_imp_op_sum_comer := new_var39_imp_op_sum_comer]
dt.cleansed[, new_var39_imp_op_diff_efect := new_var39_imp_op_diff_efect]
dt.cleansed[, new_var39_imp_op_sum_efect := new_var39_imp_op_sum_efect]
dt.cleansed[, new_var39_num_op_diff_comer := new_var39_num_op_diff_comer]
dt.cleansed[, new_var39_num_op_sum_comer := new_var39_num_op_sum_comer]
dt.cleansed[, new_var39_num_op_diff_efect := new_var39_num_op_diff_efect]
dt.cleansed[, new_var39_num_op_sum_efect := new_var39_num_op_sum_efect]
dt.cleansed[, new_var39_num_op_diff_ult := new_var39_num_op_diff_ult]
dt.cleansed[, new_var39_num_op_sum_ult := new_var39_num_op_sum_ult]
dt.cleansed[, new_var39_num_op_diff_hace := new_var39_num_op_diff_hace]
dt.cleansed[, new_var39_num_op_sum_hace := new_var39_num_op_sum_hace]
## var40
dt.cleansed[, new_var40_cnt0 := new_var40_cnt0]
dt.cleansed[, new_var40_imp_op_diff_comer := new_var40_imp_op_diff_comer]
dt.cleansed[, new_var40_imp_op_sum_comer := new_var40_imp_op_sum_comer]
dt.cleansed[, new_var40_imp_op_diff_efect := new_var40_imp_op_diff_efect]
dt.cleansed[, new_var40_imp_op_sum_efect := new_var40_imp_op_sum_efect]
dt.cleansed[, new_var40_ind_diff := new_var40_ind_diff]
dt.cleansed[, new_var40_ind_sum := new_var40_ind_sum]
dt.cleansed[, new_var40_num_op_diff_comer := new_var40_num_op_diff_comer]
dt.cleansed[, new_var40_num_op_sum_comer := new_var40_num_op_sum_comer]
dt.cleansed[, new_var40_num_op_diff_efect := new_var40_num_op_diff_efect]
dt.cleansed[, new_var40_num_op_sum_efect := new_var40_num_op_sum_efect]
dt.cleansed[, new_var40_num_op_diff_ult := new_var40_num_op_diff_ult]
dt.cleansed[, new_var40_num_op_sum_ult := new_var40_num_op_sum_ult]
dt.cleansed[, new_var40_num_op_diff_hace := new_var40_num_op_diff_hace]
dt.cleansed[, new_var40_num_op_sum_hace := new_var40_num_op_sum_hace]
dt.cleansed[, new_var40_num_diff := new_var40_num_diff]
dt.cleansed[, new_var40_num_sum := new_var40_num_sum]
## var41
dt.cleansed[, new_var41_cnt0 := new_var41_cnt0]
dt.cleansed[, new_var41_imp_op_diff_comer := new_var41_imp_op_diff_comer]
dt.cleansed[, new_var41_imp_op_sum_comer := new_var41_imp_op_sum_comer]
dt.cleansed[, new_var41_imp_op_diff_efect := new_var41_imp_op_diff_efect]
dt.cleansed[, new_var41_imp_op_sum_efect := new_var41_imp_op_sum_efect]
dt.cleansed[, new_var41_num_op_diff_comer := new_var41_num_op_diff_comer]
dt.cleansed[, new_var41_num_op_sum_comer := new_var41_num_op_sum_comer]
dt.cleansed[, new_var41_num_op_diff_efect := new_var41_num_op_diff_efect]
dt.cleansed[, new_var41_num_op_sum_efect := new_var41_num_op_sum_efect]
dt.cleansed[, new_var41_num_op_diff_ult := new_var41_num_op_diff_ult]
dt.cleansed[, new_var41_num_op_sum_ult := new_var41_num_op_sum_ult]
dt.cleansed[, new_var41_num_op_diff_hace := new_var41_num_op_diff_hace]
dt.cleansed[, new_var41_num_op_sum_hace := new_var41_num_op_sum_hace]
## var42
dt.cleansed[, new_var42_cnt0 := new_var42_cnt0]
dt.cleansed[, new_var42_num_diff := new_var42_num_diff]
dt.cleansed[, new_var42_num_sum := new_var42_num_sum]
## var43
dt.cleansed[, new_var43_cnt0 := new_var43_cnt0]
dt.cleansed[, new_var43_ind_diff := new_var43_ind_diff]
dt.cleansed[, new_var43_ind_sum := new_var43_ind_sum]
dt.cleansed[, new_var43_num_diff := new_var43_num_diff]
dt.cleansed[, new_var43_num_sum := new_var43_num_sum]
## var44
dt.cleansed[, new_var44_cnt0 := new_var44_cnt0]
dt.cleansed[, new_var44_delta_imp_diff_1y3 := new_var44_delta_imp_diff_1y3]
dt.cleansed[, new_var44_delta_imp_sum_1y3 := new_var44_delta_imp_sum_1y3]
dt.cleansed[, new_var44_delta_num_diff_1y3 := new_var44_delta_num_diff_1y3]
dt.cleansed[, new_var44_delta_num_sum_1y3 := new_var44_delta_num_sum_1y3]
dt.cleansed[, new_var44_imp_diff_hace_3 := new_var44_imp_diff_hace_3]
dt.cleansed[, new_var44_imp_sum_hace_3 := new_var44_imp_sum_hace_3]
dt.cleansed[, new_var44_imp_diff_ult_1 := new_var44_imp_diff_ult_1]
dt.cleansed[, new_var44_imp_sum_ult_1 := new_var44_imp_sum_ult_1]
dt.cleansed[, new_var44_ind_diff := new_var44_ind_diff]
dt.cleansed[, new_var44_ind_sum := new_var44_ind_sum]
dt.cleansed[, new_var44_num_diff_hace_3 := new_var44_num_diff_hace_3]
dt.cleansed[, new_var44_num_sum_hace_3 := new_var44_num_sum_hace_3]
dt.cleansed[, new_var44_num_diff_ult_1 := new_var44_num_diff_ult_1]
dt.cleansed[, new_var44_num_sum_ult_1 := new_var44_num_sum_ult_1]
dt.cleansed[, new_var44_num_diff := new_var44_num_diff]
dt.cleansed[, new_var44_num_sum := new_var44_num_sum]
dt.cleansed[, new_var44_saldo_diff_hace := new_var44_saldo_diff_hace]
dt.cleansed[, new_var44_saldo_sum_hace := new_var44_saldo_sum_hace]
dt.cleansed[, new_var44_saldo_diff_ult := new_var44_saldo_diff_ult]
dt.cleansed[, new_var44_saldo_sum_ult := new_var44_saldo_sum_ult]
## var45
dt.cleansed[, new_var45_cnt0 := new_var45_cnt0]
dt.cleansed[, new_var45_num_diff_hace := new_var45_num_diff_hace]
dt.cleansed[, new_var45_num_sum_hace := new_var45_num_sum_hace]
dt.cleansed[, new_var45_num_diff_ult := new_var45_num_diff_ult]
dt.cleansed[, new_var45_num_sum_ult := new_var45_num_sum_ult]

# dt.cleansed[, knn_2 := c(ls.knn.train[[1]], rep(-1, nrow(dt.test)))]
# dt.cleansed[, knn_4 := c(ls.knn.train[[2]], rep(-1, nrow(dt.test)))]
# dt.cleansed[, knn_8 := c(ls.knn.train[[3]], rep(-1, nrow(dt.test)))]
# dt.cleansed[, knn_16 := c(ls.knn.test[[4]], rep(-1, nrow(dt.test)))]
dt.featureEngineered <- dt.cleansed
save(dt.featureEngineered, file = "../data/Santander_Customer_Satisfaction/RData/dt_featureEngineered.RData")
