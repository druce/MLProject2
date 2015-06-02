# Classify Fitbit data for correct form in bicep curls
# Load libraries

library(lattice)
library(ggplot2)
library(caret)
library(MASS)
library(Matrix)
library(Rcpp)
library(lme4)
library(arm)
library(caTools)
library(randomForest)
library(parallel)
library(splines)
library(survival)
library(gbm)
library(nnet)
library(kernlab)
library(plyr)

# Load data

setwd("C:/Users/druce/R/MLproject2")

# download data
# download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", "pml-training.csv")
# download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", "pml-testing.csv")

training<-read.csv('pml-training.csv',na.strings = c("NA","#DIV/0!", ""))
testing<-read.csv('pml-testing.csv',na.strings = c("NA","#DIV/0!", ""))

# set random seed for reproducibility
# sample(1:10000, 1)
set.seed(1105)

# Explore data

table(training$classe)
# Histogram

qplot(training$classe, geom="histogram")

# Clean Data

countMissing <- function(mycol) {
    return (sum(is.na(training[, mycol]))/ nrow(training))
}
countNAs <- data.frame(countNA=sapply(colnames(training), countMissing))
subset(countNAs, countNAs$countNA > 0.5)

# Omit columns with mostly NAs

colsToDeleteNA <- countNAs$countNA > 0.9
training <- training[, !colsToDeleteNA]

# Identify low-information columns

nearzero <- nearZeroVar(training, saveMetrics=TRUE)
nearzero[nearzero$nzv==TRUE, ]

# Delete descriptive columns
colsToDeleteLabels <- names(training)[1:7]
training <- training[,8:60]

# Correlation analysis

mycorr <- cor(training[-53])
hicorr <- findCorrelation(mycorr, cutoff=0.8, verbose=FALSE)
colnames(training)[hicorr]
corrmatrix <- matrix(mycorr[hicorr,hicorr], nrow=12, ncol=12, dimnames=list(colnames(training)[hicorr]))
colnames(corrmatrix)<- colnames(training)[hicorr]
corrmatrix[(corrmatrix > -0.8 & corrmatrix < 0.8) ] <- NA

# correlation analysis
corrmatrix

# Explore PCA decomposition

# PCA analysis
pcaframe<-training[-53]
preProc1 <- preProcess(pcaframe, method = c("center", "scale","pca"))
preProc1
# look at contributors to 1st PCA component
head(sort(preProc1$rotation[,1]))
head(sort(-preProc1$rotation[,1]))

# Plot some significant variables v. each other

qplot(magnet_dumbbell_z, roll_belt, data=training, color=classe)

qplot(accel_belt_y, roll_arm, data=training, color=classe)

# Plot top 2 PCA components v. each other

PCAtraining <- predict(preProc1, pcaframe)
qplot(PC1, PC2, data=PCAtraining, color=training$classe)

# plot v. user_name
training2<-read.csv('pml-training.csv',na.strings = c("NA","#DIV/0!", ""))
PCAtraining$user_name = training2$user_name
qplot(PC1, PC2, data=PCAtraining, col=user_name)

# Variable selection - discussion

qplot(classe, num_window, data=training2, color = user_name)


# add back user_name
training$user_name <- training2$user_name

# Train classifiers - set up a loop with various methods


myMethods <- c("rf", "gbm", "LogitBoost", "nnet", "svmLinear", "svmRadial")
#myMethods <- c("rf")
trc_cv = trainControl(method="cv")
# center and scale for better performance on some methods
runModel <- function(mxpar) {
    return (train(classe ~ ., data=training, method=mxpar, preProcess=c("center", "scale"), trControl=trc_cv, verbose=FALSE))
}

runModelPCA <- function(mxpar) {
    return (train(classe ~ ., data=training, method=mxpar, preProcess=c("center", "scale","pca"), trControl=trc_cv, verbose=FALSE))
}

models <- list()
modelLabels <- list()
ACC <- list()
KPP <- list()
modelsStartTime <- list()

mycount <- 0
for (mx in myMethods) {
    # set.seed(1105)     # if you want to always use same folds
    # run model, store result

    mycount <- mycount+1
    modelsStartTime[[mycount]] <- Sys.time()         
    print(sprintf ("Start %s : %s" , Sys.time(), mx))
    models[[mycount]] <- runModel(mx)
    modelLabels[[mycount]] <- models[[mycount]]$modelInfo$label
    ACC[[mycount]] <- max(models[[mycount]]$results$Accuracy)
    KPP[[mycount]] <- max(max(models[[mycount]]$results$Kappa))
    print(models[[mycount]])

    # PCA
    mycount <- mycount+1
    # set.seed(1105)     
    modelsStartTime[[mycount]] <- Sys.time()    	   
    print(sprintf ("Start %s : %s (PCA)" , Sys.time(), mx))
    models[[mycount]] <- runModelPCA(mx)
    modelLabels[[mycount]] <- sprintf("%s (PCA)", models[[mycount]]$modelInfo$label)
    ACC[[mycount]] <- max(models[[mycount]]$results$Accuracy)
    KPP[[mycount]] <- max(max(models[[mycount]]$results$Kappa))
    print(models[[mycount]])
   
}
modelsEndTime <- Sys.time()    	   
modelsEndTime

# Compare performance
performance <- cbind(modelLabels,ACC,KPP)
performance

# Variable importance

RFclassifier <- models[[1]]
varImp(RFclassifier)

# Plot top 3 random forest variables v. each other

qplot(roll_belt, yaw_belt, data=training, col=classe)

qplot(roll_belt, magnet_dumbbell_z, data=training, col=classe)

qplot(yaw_belt, magnet_dumbbell_z, data=training, col=classe)

# Table of results from training set

myPredict <- data.frame(prediction=predict(RFclassifier, training))
myPredict$classe<-training$classe
#table(myPredict)
confusionMatrix(myPredict$prediction, myPredict$classe)

# Predict vs. test data
testPrediction=predict(RFclassifier, newdata = testing)
testPrediction

# Generate test output
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(testPrediction)

