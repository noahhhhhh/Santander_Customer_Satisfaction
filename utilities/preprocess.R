require(data.table)
############################################################################################
## 1. ColNAs ###############################################################################
############################################################################################
## Intro: summarise the stats of NAs of columns in a data table
## Args:
##  dt(data.table): a data table
##  method(character): "sum" - number of NAs; "mean" - NAs proportion of total
##  output(character): "all" - return all the output; "nonZero" - return only non-zero output
## Return(numeric): output of NA stats
ColNAs <- function(dt, method = "sum", output = "all"){
    out <- as.numeric()
    if (method == "sum"){
        out <- apply(dt, 2, function(x){sum(is.na(x))})
    } else if (method == "mean"){
        out <- apply(dt, 2, function(x){round(mean(is.na(x)), 2)})
    } else {
        return(F)
    }
    
    if (output == "all"){
        return(out)
    } else if (output == "nonZero"){
        return(out[out != 0])
    } else {
        return(F)
    }
}

############################################################################################
## 2. ColUnique ############################################################################
############################################################################################
## Intro: summarise the number of unique values of columns in a data table
## Args:
##  dt(data.table): a data table
## Return(numeric): output of unique value stats
ColUnique <- function(dt){
    out <- apply(dt, 2, function(x){length(unique(x))})
    return(out)
}

############################################################################################
## 3. BinaryEncode #########################################################################
############################################################################################
## Intro: Binary Encoding for categorial features
## Args:
##  dt(data.table): a data table
##  cols(a vector of characters): names of targeted columns
## Return(data.table): output of a data table with the additional binary columns
BinaryEncode <- function(dt, cols){
    require(miscFuncs)
    require(stringr)
    for(col in cols){
        # unique values dict
        dict.uniq <- data.table(ID = rownames(unique(dt[, col, with = F]))
                                , unique(dt[, col, with = F]))
        # decimal to binary vector, e.g. 19 --> 10011
        vec.dec <- bin(dim(dict.uniq)[1])
        # length of the binary vector, e.g. 19 --> 10011 --> 5
        num.len <- length(vec.dec)
        # binary encoding
        # ID corresponding to col
        dt.col <- merge(dt[, col, with = F], dict.uniq, by = col, all.x = T)
        dt.col[, ID := as.integer(ID)]
        # encoded vector
        vec.bin <- unlist(apply(dt.col[, "ID", with = F], 1, function(x)(str_pad(paste(bin(x), collapse = ""), num.len, side = "left", pad = "0"))))
        # set up the col names and binary values
        vec.col  <- as.character()
        for (i in 1:num.len){
            vec.col[i] <- paste(col, "_bin_", i, sep = "")
            dt[, vec.col[i]:= as.integer(substr(vec.bin, i, i))]
        }
        dt <- dt[, !col, with = F]
    }
    return(dt)
}

############################################################################################
## 4. ConvertNonNumFactorToNumFactor #######################################################
############################################################################################
## Intro: Convert non-numeric factor to a numeric factor
## Args:
##  dt(data.table): a data table
##  cols(a vector of characters): name of targeted columns
## Return(data.table): output of a data table with the additional binary columns
ConvertNonNumFactorToNumFactor <- function(dt, cols){
    # unique values dict
    dict.uniq <- data.table(ID = rownames(unique(dt[, cols, with = F]))
                            , unique(dt[, cols, with = F]))
    # ID corresponding to cols
    dt.col <- merge(dt, dict.uniq, by = cols, all.x = T)
    # set the name of the new cols
    colname <- paste(cols, "_toNum", sep = "")
    setnames(dt.col, names(dt.col), c(names(dt.col)[-length(names(dt.col))], colname))
    
    return(dt.col)
}

############################################################################################
## 4. ConvertNonNumFactorToOrderedNum ######################################################
############################################################################################
## Intro: Convert non-numeric factor to a ordered numeric values
## e.g.: AA(100), BB(21), CC(33) --> map AA, BB, CC to 3, 1, 2
## Args:
##  dt(data.table): a data table
##  cols(a vector of characters): name of targeted columns
## Return(data.table): output of a data table with the ordered numeric values
ConvertNonNumFactorToOrderedNum <- function(dt.train, dt.test, cols){
    require(plyr)
    # get the list of tabular summary 
    ls.table <- lapply(dt.train[, cols, with = F], table)
    # get the list of names of the tablular summary
    ls.names <- lapply(ls.table, function(x) names(x))
    colnames <- paste(cols, "_toOrderedNum", sep = "")
    # get the list of the ordred numeirc representing the tabluar summary
    ls.ordered <- lapply(ls.table, function(x)(frank(as.vector(x), ties.method = "dense")))
    
    ls.replaced.train <- list()
    ls.replaced.test <- list()
    for (col in cols){
        n <- ls.names[[col]]
        o <- ls.ordered[[col]]
        ls.replaced.train[[paste(col, "toOrderedNum", sep = "")]] <- mapvalues(dt.train[[col]], from = n, to = o)
        ls.replaced.test[[paste(col, "toOrderedNum", sep = "")]] <- mapvalues(dt.test[[col]], from = n, to = o)
        # new level to be come 0 in dt.test
        ls.replaced.test[[paste(col, "toOrderedNum", sep = "")]][grepl("[[:alpha:]]", ls.replaced.test[[paste(col, "toOrderedNum", sep = "")]])] <- 0
    }
    dt.replaced.train <- as.data.table(lapply(ls.replaced.train, print))
    dt.replaced.test <- as.data.table(lapply(ls.replaced.test, print))
    return(list(dt.replaced.train, dt.replaced.test))
}

############################################################################################
## 6. Noise ################################################################################
############################################################################################
## Intro: add noise into a data table. This function references Ivan Liu.
## Args:
##  dt(data.table): a data table
##  noise_l(numeric): lower noise
##  noise_u(numeric): upper noise
##  col_excl(a vector of characters): names of columns not included which do not apply the noise
## Return(data.table): output of a data table after adding noise
Noise <- function(dt, noise_l = -.00001, noise_u = .00001, col_excl){
    dim(dt)
    dt_noise <- apply(dt[, !col_excl, with = F], 2, function(x){
        runif(n = length(x), min = noise_l * diff(range(x)), max = noise_u * diff(range(x)))
    })
    dt_noise <- dt[, !col_excl, with = F] + dt_noise
    
    dt_noise <- data.table(dt_noise, dt[, col_excl, with = F])
    dt <- rbind(dt, dt_noise)
    dim(dt)
    dt <- dt[sample(1:nrow(dt), size = nrow(dt))]
    return(dt)
}

############################################################################################
## 7. MyImpute #############################################################################
############################################################################################
## Intro: Impute with median, -1, and amelia
## Args:
##  dt(data.table): a data table
##  cols(a vector of characters): name of targeted columns
##  imputed_type(a single vector of characters): imputation type
##  m(integer): amelia only - number of imputation sets
##  idvars(a vector of characters): column names of identification vars
##  noms(a vector of characters): column names of nominal vars
##  ords(a vector of characters): column names of ordinal vars
## Return(list): output of a list of m imputation sets
MyImpute <- function(dt, cols, impute_type = c("median", "amelia", num), m, idvars, noms, ords){
    ls.imputed <- list()
    if(impute_type == "median"){ # median impute
        require(randomForest)
        ls.imputed[[1]] <- na.roughfix(dt[, cols, with = F])
        # set the new names
        setnames(ls.imputed[[1]], cols, paste(cols, "toImputed", sep = ""))
    } else if(impute_type == "amelia"){ # amelia impute
        require(Amelia)
        a.out <- amelia(x = dt
                        , m = m
                        , p2s = 1
                        , idvars = idvars
                        , noms = noms
                        , ords = ords
                        , parallel = "multicore"
                        , ncpus = 8)
        ls.imputed <- a.out$imputations
    } else { # -1 impute
        # class <- unlist(lapply(dt[, cols, with = F], class))
        # cols.factor <- names(class)[class == "factor"]
        # cols.others <- names(class)[class != "factor"]
        df.imputed <- as.data.frame(dt[, cols, with = F])
        # as.data.frame(apply(df.imputed[, cols.factor], 2, as.character))[is.na(df.imputed[, cols.factor])] <- "-1"
        df.imputed[is.na(df.imputed)] <- as.numeric(impute_type)
        ls.imputed[[1]] <- as.data.table(df.imputed)
        # set the new names
        setnames(ls.imputed[[1]], cols, paste(cols, "toImputed", sep = ""))
    } 
    
    return(ls.imputed)
}

############################################################################################
## 8. VisNAs ###############################################################################
############################################################################################
# visualise NAs
VisNAs <- function(dt){
    dt <- train
    library(readr)
    for (f in names(train)) {
        if (class(train[[f]])=="character") {
            levels <- unique(train[[f]])
            train[[f]] <- as.integer(factor(train[[f]], levels=levels))
        }
    }
    
    # make a table of missing values
    library(mice)
    missers <- md.pattern(train[, -c(1:2)])
    View(missers)
    
    # plot missing values
    library(VIM)
    miceplot <- aggr(train[, -c(1:2)], col=c("dodgerblue","dimgray"),
                     numbers=TRUE, combined=TRUE, border="gray50",
                     sortVars=TRUE, ylabs=c("Missing Data Pattern"),
                     labels=names(train[-c(1:2)]), cex.axis=.7,
                     gap=3)
}

############################################################################################
## 9. ConvertNonNumFactorToSumOfTargets ####################################################
############################################################################################
ConvertNonNumFactorToSumOfTargets <- function(dt, cols, ind.train){
    for(col in cols){
        colname1 <- paste(col, "_Sum1", sep = "")
        colname2 <- paste(col, "_Sum0", sep = "")
        dt[, temp1 := sum(as.numeric(target[ind.train] == 1)), by = col]
        dt[, temp0 := sum(as.numeric(target[ind.train] == 0)), by = col]
        setnames(dt, names(dt), c(names(dt)[!names(dt) %in% c("temp1", "temp0")], c(colname1, colname2)))
    }
    return(dt)
}
















