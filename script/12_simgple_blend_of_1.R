setwd("/Volumes/Data Science/Google Drive/data_science_competition/kaggle/Santander_Customer_Satisfaction/")
rm(list = ls()); gc();
require(data.table)
require(caret)
require(glmnet)
load("../data/Santander_Customer_Satisfaction/RData/dt_featureEngineered.RData")

#######################################################################################
## load ###############################################################################
#######################################################################################
load("../data/Santander_Customer_Satisfaction/RData/dt_meta_1_et.RData")
load("../data/Santander_Customer_Satisfaction/RData/dt_meta_1_et_factor.RData")
load("../data/Santander_Customer_Satisfaction/RData/dt_meta_1_glm.RData")
load("../data/Santander_Customer_Satisfaction/RData/dt_meta_1_rf.RData")
load("../data/Santander_Customer_Satisfaction/RData/dt_meta_1_h20.RData")
load("../data/Santander_Customer_Satisfaction/RData/dt_meta_1_xgb_linear.RData")
load("../data/Santander_Customer_Satisfaction/RData/dt_meta_1_xgb_pairwise.RData")
load("../data/Santander_Customer_Satisfaction/RData/dt_meta_1_xgb_poisson.RData")
load("../data/Santander_Customer_Satisfaction/RData/dt_meta_1_xgb.RData")

# range to 0 and 1
range01 <- function(x){(x-min(x))/(max(x)-min(x))}
#######################################################################################
## train and valid ####################################################################
#######################################################################################
dt.pred.train <- data.table(et = vec.et.pred.train # 0.7770218
                            , et_fac = vec.et.fac.pred.train # 0.7
                            , glm = as.numeric(vec.meta.glm.train) # 0.7736692
                            , rf = vec.meta.rf.train # 0.8
                            , h2o = vec.meta.h2o.train # 0.8215519
                            , xgb_linear = vec.xgb.linear.pred.train # 0.7927233
                            , xgb_pariwise = vec.xgb.pairwise.pred.train # 0.8390535
                            , xgb_poisson = vec.xgb.poisson.pred.train # 0.7767584
                            , xgb = vec.xgb.pred.train # 0.842384
                            , TARGET = dt.featureEngineered$TARGET[dt.featureEngineered$TARGET >= 0]
                            )
r <- as.data.table(apply(dt.pred.train[, !c("TARGET"), with = F], 2, rank))
x <- as.data.table(apply(r, 2, range01))
dt.pred.train <- cbind(x, TARGET = dt.pred.train$TARGET)

## partition
ind.train <- createDataPartition(dt.pred.train$TARGET, p = .7, list = F)
dt.predtrain <- dt.pred.train[ind.train]
dt.predval <- dt.pred.train[-ind.train]

# test set
dt.pred.test <- data.table(et = vec.et.pred.test # 0.7770218
                           , et_fac = vec.et.fac.pred.test # 0.7
                           , glm = as.numeric(vec.meta.glm.test) # 0.7736692
                           , rf = vec.meta.rf.test # 0.8
                           , h2o = vec.meta.h2o.test # 0.8215519
                           , xgb_linear = vec.xgb.linear.pred.test # 0.7927233
                           , xgb_pariwise = vec.xgb.pairwise.pred.test # 0.8390535
                           , xgb_poisson = vec.xgb.poisson.pred.test # 0.7767584
                           , xgb = vec.xgb.pred.test # 0.842384
                           , TARGET = dt.featureEngineered$TARGET[dt.featureEngineered$TARGET < 0]
)
r <- as.data.table(apply(dt.pred.test[, !c("TARGET"), with = F], 2, rank))
x <- as.data.table(apply(r, 2, range01))
dt.pred.test <- cbind(x, TARGET = dt.pred.test$TARGET)

save(dt.pred.train, dt.pred.test, file = "../data/Santander_Customer_Satisfaction/RData/dt_meta_1_all.RData")
#######################################################################################
## blend ##############################################################################
#######################################################################################
######################
## weighted average ##
######################
pred.avg.test <- with(dt.pred.test
                      , (et * .1
                         + et_fac * .05
                      + glm * .1
                      + rf * .15
                      + xgb_linear * .15
                      + xgb_pariwise * .15
                      + xgb * .3)
)
submit <- data.table(ID = dt.featureEngineered$ID[dt.featureEngineered$TARGET < 0]
                     , TARGET = pred.avg.test)
write.csv(submit, file = "submission/34_blend_with_avg_no_rf_model_output.csv", row.names = F)
# 0.835321 with no et_fac, with no rf

####################
## blend with glm ##
####################
x.train <- model.matrix(TARGET ~., data = dt.predtrain)[, -1]
y.train <- dt.predtrain$TARGET
x.valid <- model.matrix(TARGET ~., data = dt.predval)[, -1]
y.valid <- dt.predval$TARGET
x.test <- model.matrix(TARGET ~., data = dt.pred.test)[, -1]
# init grids
grid <- 10^seq(10, -2, length = 100)

# train
md.lasso <- glmnet(x.train, y.train, alpha = .5, lambda = grid, family = "binomial")
plot(md.lasso)

# cv to choose λ
set.seed(888)
cv.out <- cv.glmnet(x.train, y.train, alpha = .5, type.measure = "auc", family = "binomial")
# plot(cv.out)
bestlam <- cv.out$lambda.min
plot(cv.out)
# bestlam

# valid
pred.lasso <- predict(md.lasso , s = bestlam, type = "response", newx = x.valid)
auc(y.valid, pred.lasso)
# 0.8440793
# 0.8393489 with et_fac and rf

# submit
pred.lasso.test <- predict(md.lasso, s = bestlam, type = "response", newx = x.test)
submit <- data.table(ID = dt.featureEngineered$ID[dt.featureEngineered$TARGET < 0]
                     , TARGET = as.numeric(pred.lasso.test))
write.csv(submit, file = "submission/35_blend_with_glm.csv", row.names = F)
# 0.838885
# 0.839985 with et_fac and rf
