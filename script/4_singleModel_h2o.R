setwd("/Volumes/Data Science/Google Drive/data_science_competition/kaggle/Santander_Customer_Satisfaction/")
rm(list = ls()); gc();
require(data.table)
require(purrr)
require(caret)
require(Metrics)
require(ggplot2)
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

dt.train[, TARGET := as.factor(dt.train$TARGET)]
dt.valid[, TARGET := as.factor(dt.valid$TARGET)]

#######################################################################################
## 2.0 h2o cv #########################################################################
#######################################################################################
require(h2o)
h2o.init(ip = 'localhost', port = 54321, max_mem_size = '6g')
h2o.train <- as.h2o(dt.train)
h2o.valid <- as.h2o(dt.valid)

md.h2o <- h2o.deeplearning(x = setdiff(names(dt.train), c("ID", "TARGET")),
                        y = "TARGET",
                        training_frame = h2o.train,
                        nfolds = 3,
                        stopping_rounds = 3,
                        epochs = 20,
                        overwrite_with_best_model = TRUE,
                        activation = "RectifierWithDropout",
                        input_dropout_ratio = 0.2,
                        hidden = c(100,100),
                        l1 = 1e-4,
                        loss = "CrossEntropy",
                        distribution = "bernoulli",
                        stopping_metric = "AUC"
)
pred.valid <- as.data.frame(h2o.predict(object = md.h2o, newdata = h2o.valid))
auc(dt.valid$TARGET, pred.valid$p1)
# benchmark
# 0.790934