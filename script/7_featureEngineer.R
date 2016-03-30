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
nnn <- c(1, 2, 4, 8, 16, 32, 64, 128)
for(n in 1:length(nnn)){
    for(i in 1:k){
        print(paste("n:", nnn[n], "; k:", k, "start:", Sys.time()))
        f <- folds == i
        dtrain <- dt.train[!f]
        dval <- dt.train[f]
        dtest <- dt.test
        
        knn.train[f] <- knn(train = dtrain[, !c("ID", "TARGET"), with = F]
                            , test = dval[, !c("ID", "TARGET"), with = F]
                            , cl = as.factor(dtrain$TARGET)
                            , k = nnn[n]
                            , prob = T)
        print(paste("n:", nnn[n], "; k:", k, "end:", Sys.time()))
    }
    ls.knn.train[[n]] <- knn.train
    ls.knn.test[[n]] <- knn.test
}
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
dt.featureEngineered <- dt.cleansed
save(dt.featureEngineered, file = "../data/Santander_Customer_Satisfaction/RData/dt_featureEngineered.RData")
