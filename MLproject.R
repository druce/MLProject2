############################################################
# load some libraries
############################################################

## mylibraries = c("lattice","ggplot2","caret","arm","MASS",
##     "Matrix","lme4","Rcpp","caTools","randomForest")
## invisible(lapply(mylibraries, require, character.only=T))

library(lattice)
library(ggplot2)
library(caret)
library(arm)
library(MASS)
library(Matrix)
library(lme4)
library(Rcpp)
library(caTools)
library(randomForest)

############################################################
# import data
############################################################

setwd("C:/Users/druce/R/MLproject2")

# download data
# download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", "pml-training.csv")
# download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", "pml-testing.csv")

training<-read.csv('pml-training.csv',na.strings = c("NA","#DIV/0!", ""))
testing<-read.csv('pml-training.csv',na.strings = c("NA","#DIV/0!", ""))

# set random seed for reproducibility
# sample(1:10000, 1)
set.seed(1105)

############################################################
# exploratory data analysis, data cleanup
############################################################

# check out classes
sapply(training, class)

# check out histogram of classification prediction variable
table(training$classe)
histtable <- table(training$classe)
min(histtable)/sum(histtable)

#plot histogram
qplot(training$classe, geom="histogram")

# count NAs
countMissing <- function(mycol) {
    return (sum(is.na(training[, mycol]))/ nrow(training))
}
countNAs <- data.frame(countNA=sapply(colnames(training), countMissing))
subset(countNAs, countNAs$countNA > 0.5)

# We see a number of summary columns with 97% of data NA
# we can safely delete them since we want to predict on individual reps
colsToDeleteNA <- countNAs$countNA > 0.9
training <- training[, !colsToDeleteNA]

# identify near zero values
nearzero <- nearZeroVar(training, saveMetrics=TRUE)
nearzero[nearzero$nzv==TRUE, ]

# lowest frequency of prediction var is ~16%
# theoretically you could have a column with 84% 0s
# 16% uniquely distributed nonzeros, perfectly predicting that class
# gives like 16406 ratio of must frequent to 2nd most frequent
# so, not totally obvious we need to take them out

# if we wanted to delete
# delete columns with > 1000 ratio of most frequent to 2nd most frequent
## nearzero$colsToDelete <- FALSE
## nearzero$colsToDelete[nearzero$freqRatio> 1000] <- TRUE
## nearzero[nearzero$colsToDelete,]
## training <- training[, !nearzero$colsToDelete]

# first 7 columns are descriptive, no predictive value and can be omitted
colsToDeleteLabels <- names(training)[1:7]
training <- training[,8:60]

# correlation analysis
mycorr = cor(training[-53])
hicorr = findCorrelation(mycorr, cutoff=0.8, verbose=FALSE)
colnames(training)[hicorr]
corrmatrix = matrix(mycorr[hicorr,hicorr], nrow=12, ncol=12, dimnames=list(colnames(training)[hicorr]))
colnames(corrmatrix)<- colnames(training)[hicorr]
corrmatrix[(corrmatrix > -0.8 & corrmatrix < 0.8) ] <- NA
corrmatrix

# 12 rows flagged, can try to predict with PCA and without

# PCA analysis
pcaframe<-training[-53]
preProc1 <- preProcess(pcaframe, method = c("center", "scale","pca"))
# look at contributors to 1st PCA component
sort(preProc1$rotation[,1])
# accel_belt_y
# accel_belt_z
# roll_belt
# accel_arm_y

# plot these top contributors vs. predicted variable
plotcolor <- data.frame(classe=training$classe)
plotcolor$classe2 <- as.integer(0)
plotcolor[plotcolor$classe=="A", "classe2"] <- 1
plotcolor[plotcolor$classe=="B", "classe2"] <- 2
plotcolor[plotcolor$classe=="C", "classe2"] <- 3
plotcolor[plotcolor$classe=="D", "classe2"] <- 4
plotcolor[plotcolor$classe=="E", "classe2"] <- 5

plotshape <- data.frame(classe=training$classe)
plotshape$classe2 <- as.integer(0)
plotshape[plotshape$classe=="A", "classe2"] <- 16
plotshape[plotshape$classe=="B", "classe2"] <- 17
plotshape[plotshape$classe=="C", "classe2"] <- 18
plotshape[plotshape$classe=="D", "classe2"] <- 19
plotshape[plotshape$classe=="E", "classe2"] <- 15

plot(training$accel_belt_y, training$accel_arm_y, col=plotcolor[, "classe2"], pch=20, cex=0.5)
plot(training$roll_arm, training$roll_belt, col=plotcolor[, "classe2"], pch=20, cex=0.5)

plotcolor <- data.frame(classe=training$classe)
plotcolor$classe2 <- ""
plotcolor[plotcolor$classe=="A", "classe2"] <- "red"
plotcolor[plotcolor$classe=="B", "classe2"] <- "green"
plotcolor[plotcolor$classe=="C", "classe2"] <- "blue"
plotcolor[plotcolor$classe=="D", "classe2"] <- "yellow"
plotcolor[plotcolor$classe=="E", "classe2"] <- "black"

qplot(accel_belt_y, accel_arm_y, data=training, color=plotcolor[, "classe2"])
qplot(accel_belt_y, roll_arm, data=training, color=plotcolor[, "classe2"])
qplot(roll_arm, roll_belt, data=training, color=plotcolor[, "classe2"])
qplot(magnet_dumbbell_z, roll_belt, data=training, color=plotcolor[, "classe2"])
qplot(pitch_forearm, yaw_belt, data=training, color=plotcolor[, "classe2"])

# seems promising. try PCA
PCAtraining <- predict(preProc1, pcaframe)
qplot(PC1, PC2, data=PCAtraining, col=plotcolor[, "classe2"])

# now... this looks like it separates almost cleanly into 5 zones, and yet they don't match up with colors
# I don't see any error that could make the colors not line up with PCA components
# PCA needs 25 components to capture 95 percent of variance v. 52 raw data components
# good dimensionality reduction could speed up and get better results from some algorithms
# would appear PCAs yields the amount of dimensionality reduction we want but not the classification we need from PCA1 and PCA2 anyway
# like it's separating individual people, not good form/bad form

# train control <- cross-validation (use defaults = 10 k-folds)

# this is if you want to train explicitly from PCA components
#PCAtraining$classe <- training$classe

#myMethods <- c("rf", "glm", "LogitBoost", "nnet", "gbm", "svmLinear", "svmRadial")
#myMethods <- c("LogitBoost", "nnet", "rf", "gbm", "svmLinear", "svmRadial")
myMethods <- c("rf")

runModel <- function(mxpar) {
    return (train(classe ~ ., data=training, method=mxpar, trControl=trainControl(method="cv"), verbose=FALSE))
}

runModelPCA <- function(mxpar) {
    return (train(classe ~ ., data=training, method=mxpar, preProcess="pca", trControl=trainControl(method="cv"), verbose=FALSE))
}

retvals <- list()
modelLabels <- list()
ACC <- list()
KPP <- list()

mycount <- 0
for (mx in myMethods) {
    print(sprintf ("%s : %s" , Sys.time(), mx))

    # set.seed(1105)     # if you want to always use same folds
    # run model, store result

    mycount <- mycount+1
    models[[mycount]] <- runModel(mx)
    modelLabels[[mycount]] <- retvals[[mycount]]$modelInfo$label,
    ACC[[mycount]] <- retvals[[mycount]]$results$Accuracy,
    KPP[[mycount]] <- retvals[[mycount]]$modelInfo$Kappa,
    print(retvals[[mycount]])

    ## mycount <- mycount+1
    ## set.seed(1105)     
    ## retvals[[mycount]] <- runModelPCA(mx)
    ## print(retvals[[mycount]])
   
}

performance <- cbind(modelLabels,ACC,KPP)

performance

# random forest works best
RFclassifier <- retvals[[1]]
# RFclassifier <- train(classe ~ ., data=training, method="rf", trControl=trainControl(method="cv"), verbose=FALSE)
varImp(RFclassifier)

qplot(roll_belt, yaw_belt, data=training, col=plotcolor[, "classe2"])
qplot(roll_belt, magnet_dumbbell_z, data=training, col=plotcolor[, "classe2"])
qplot(yaw_belt, magnet_dumbbell_z, data=training, col=plotcolor[, "classe2"])

# Table of results from training set

myPredict <- data.frame(predict(RFclassifier, training))
myPredict$classe<-training$classe
table(myPredict)

# run v. test and plot results
testPrediction=predict(RFclassifier, testing)
testPrediction
