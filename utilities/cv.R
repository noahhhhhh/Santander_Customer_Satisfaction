myTrainValidTest <- function(dt){
    cat("prepare train, valid, and test data set...\n")
    set.seed(888)
    ind.train <- createDataPartition(dt[TARGET >= 0]$TARGET, p = .9, list = F) # remember to change it to .66
    dt.train <- dt[TARGET >= 0][ind.train]
    dt.valid <- dt[TARGET >= 0][-ind.train]
    dt.test <- dt[TARGET == -1]
    dim(dt.train); dim(dt.valid); dim(dt.test)
    
    table(dt.train$TARGET)
    table(dt.valid$TARGET)
    
    dmx.train <- xgb.DMatrix(data = data.matrix(dt.train[, !c("ID", "TARGET"), with = F]), label = dt.train$TARGET)
    dmx.valid <- xgb.DMatrix(data = data.matrix(dt.valid[, !c("ID", "TARGET"), with = F]), label = dt.valid$TARGET)
    x.test <- data.matrix(dt.test[, !c("ID", "TARGET"), with = F])
    return(list(dt.train, dt.valid, dt.test
                , dmx.train, dmx.valid, x.test))
}

myCV_xgb <- function(dt.train, cols.features, dt.valid, k = 10, params){
    ## store the result
    cat("init dt.result and dt.summary ...\n")
    dt.result <- as.data.table(matrix(rep(0, 9), 1))
    setnames(dt.result, c("round", "eta", "mcw", "md", "ss", "csb", "cv_num", "result.dval", "resul.valid"))
    df.result <- as.data.frame(dt.result) # df is easier to insert rows
    
    dt.summary <- as.data.table(matrix(rep(0, 14), 1))
    setnames(dt.summary, c("round", "eta", "mcw", "md", "ss", "csb", "mean.dval", "max.dval", "min.dval", "sd.dval", "mean.valid", "max.valid", "min.vaild", "sd.valid"))
    df.summary <- as.data.frame(dt.summary) # df is easier to insert rows
    
    m <- 1 # round
    
    if(k == 0){
        dval <- xgb.DMatrix(data = data.matrix(dt.valid[, cols.features, with = F]), label = dt.valid$TARGET)
        dtrain <- xgb.DMatrix(data = data.matrix(dt.train[, cols.features, with = F]), label = dt.train$TARGET)
        watchlist <- list(val = dval, train = dtrain)
        
        set.seed(888)
        clf <- xgb.train(params = params
                         , data = dtrain
                         , nrounds = 100000 
                         , early.stop.round = 50
                         , watchlist = watchlist
                         , print.every.n = 10
        )
        
        pred.dval <- predict(clf, dval)
        result.dval <- auc(getinfo(dval, "label"), pred.dval)
        vec.result.dval <- result.dval
        
        df.result[m, ] <- c(m
                                          , params$eta
                                          , params$mcw
                                          , params$md
                                          , params$ss
                                          , params$csb
                                          , m, result.dval, result.dval)
        df.summary[m, ] <- c(m
                             , params$eta
                             , params$mcw
                             , params$md
                             , params$ss
                             , params$csb
                             , mean(vec.result.dval), max(vec.result.dval), min(vec.result.dval), sd(vec.result.dval)
                             , mean(result.dval), max(result.dval), min(result.dval), sd(result.dval))
    } else {
        ## folds
        cat("folds ...\n")
        set.seed(888)
        folds <- createFolds(dt.train$TARGET, k = k, list = F)
        cat("cv ...\n")
        vec.result.dval <- rep(0, k)
        vec.result.valid <- rep(0, k)
        for(i in 1:k){
            f <- folds == i
            dval <- xgb.DMatrix(data = data.matrix(dt.train[f, cols.features, with = F]), label = dt.train[f]$TARGET)
            dtrain <- xgb.DMatrix(data = data.matrix(dt.train[!f, cols.features, with = F]), label = dt.train[!f]$TARGET)
            watchlist <- list(val = dval, train = dtrain)
            set.seed(888)
            print(paste("cv:", i, "-------"))
            clf <- xgb.train(params = params
                             , data = dtrain
                             , nrounds = 100000 
                             , early.stop.round = 50
                             , watchlist = watchlist
                             , print.every.n = 10
            )
            
            pred.dval <- predict(clf, dval)
            result.dval <- auc(getinfo(dval, "label"), pred.dval)
            vec.result.dval[i] <- result.dval
            
            dmx.valid <- xgb.DMatrix(data = data.matrix(dt.valid[, cols.features, with = F]), label = dt.valid$TARGET)
            pred.valid <- predict(clf, dmx.valid)
            result.valid <- auc(getinfo(dmx.valid, "label"), pred.valid)
            vec.result.valid[i] <- result.valid
            
            df.result[(m - 1) * k + i, ] <- c(m
                                              , params$eta
                                              , params$mcw
                                              , params$md
                                              , params$ss
                                              , params$csb
                                              , i, result.dval, result.valid)
        }
        df.summary[m, ] <- c(m
                             , params$eta
                             , params$mcw
                             , params$md
                             , params$ss
                             , params$csb
                             , mean(vec.result.dval), max(vec.result.dval), min(vec.result.dval), sd(vec.result.dval)
                             , mean(vec.result.valid), max(vec.result.valid), min(vec.result.valid), sd(vec.result.valid))
        print(df.summary)
        m <- m + 1
    }
    
    return(df.summary)
}