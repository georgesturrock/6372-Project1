source("helper_functions.R")
library(dplyr)
library(olsrr)
library(car)
library(caret)
library(mice)
library(reshape)
library(glmnet)
library(plotmo)

setwd("C:/Users/Sturrock/Documents/SMU Data Science/Stats2/Project 1")

df.train2 <- read.csv("train.csv")
#df.test <- read.csv('test.csv')
df.test <- read.csv("true_test.csv")
## clean data for scatter plots
df.train2.scatters <- cleanData(df.train2)

fit.full2 <- lm(df.train2.scatters$SalePrice ~ ., data = df.train2, na.action = na.exclude)
summary(fit.full2)

# look at the scatter plots and SalePrice histogram to assess assumptions
# df.train2.numeric <- select_if(df.train2, is.numeric)

# df.plots <- melt(df.train2.numeric, "SalePrice")

#ggplot(df.plots, aes(value, df.plots$SalePrice)) + 
# geom_point() + 
# facet_wrap(~variable, scales = "free")

# hist(df.train2.numeric$SalePrice)

#Save outliers for analysis
df.Q2outliers <- read.csv("train.csv")
Q2meanSalePrice <- mean(df.Q2outliers$SalePrice)
Q2meanGrLivArea <- mean(df.Q2outliers$GrLivArea)
Q2PPSF <- Q2meanSalePrice / Q2meanGrLivArea
df.Q2outliers <- df.Q2outliers[(df.Q2outliers$Id %in% c(1299, 524, 3, 463, 633, 89, 589, 496, 682, 813, 969)), ]
df.Q2outliers$PPSF <- df.Q2outliers$SalePrice / df.Q2outliers$GrLivArea

######################################################################

# for generating imputed training data
#writeImputeData <- T
writeImputeData <- F
if (writeImputeData == T) {
  # clean data
  df.train2 <- cleanData(df.train2)
  # transform values
  df.train2 <- transformData(df.train2)
  # encode variables
  df.train2 <- encodeData(df.train2)
  df.train2 <- mice(df.train2[, names(df.train2)], method="rf")
  df.train2 <- complete(df.train2)
  write.csv(x = df.train2, file = "true_train.csv", row.names = F)
}

df.train2 <- read.csv("true_train.csv")

# get internal train and test
set.seed(101) 
train.size <- 0.8
train.index <- sample.int(length(df.train2$SalePrice), round(length(df.train2$SalePrice) * train.size))

train = df.train2[train.index, ]
test  = df.train2[-train.index, ]

#--------------------------------------------------------#

## glmnet lasso, ridge and elastic net variable selection
cvalpha <- c(0,0.5,1)
mdlmtx.train2 <- model.matrix(SalePrice ~ ., data = df.train2)
#df.delete <- df.train2
#df.delete <- df.delete[, !(names(df.delete) %in% 'SalePrice')]
#mtx.train2 <- data.matrix(df.delete[, !(names(df.delete) %in% 'SalePrice')])
mtx.train2 <- data.matrix(select(df.train2, -SalePrice))

for(i in cvalpha) {
#  assign(paste('fit', i, sep = ''), cv.glmnet(mdlmtx.train2, df.train2$SalePrice, alpha=i, family='gaussian'))
  assign(paste('fit', i, sep = ''), cv.glmnet(mtx.train2, df.train2$SalePrice, alpha=i, family='gaussian'))
}

# choose model 
#CV from model.matrix
cat('Ridge Lambda Min =', fit0$lambda.min, 'Ridge Lambda 1SE =', fit0$lambda.1se)
cat('Elastic Net Lambda Min =', fit0.5$lambda.min, 'Ridge Lambda 1SE =', fit0.5$lambda.1se)
cat('Lasso Lambda Min =', fit1$lambda.min, 'Lasso Lambda 1SE =', fit1$lambda.1se)

## Fit from model.matrix
fit.lasso <- glmnet(mdlmtx.train2, df.train2$SalePrice, alpha = 1, family='gaussian')
fit.ridge <- glmnet(mdlmtx.train2, df.train2$SalePrice, alpha = 0, family='gaussian')
fit.enet <- glmnet(mdlmtx.train2, df.train2$SalePrice, alpha = 0.5, family='gaussian')

##fit from data.matrix
fit.lassob <- glmnet(mtx.train2, df.train2$SalePrice, alpha = 1, family='gaussian')
fit.ridgeb <- glmnet(mtx.train2, df.train2$SalePrice, alpha = 0, family='gaussian')
fit.enetb <- glmnet(mtx.train2, df.train2$SalePrice, alpha = 0.5, family='gaussian')

### variable selection stats and plots for glmnet
#Plots for reqularized fits
par(mfrow=c(3,2))
plot(fit.lasso, xvar = 'lambda', labels = T)
plot(fit1, main = 'Lasso', labels = T)
plot(fit.ridge, xvar = 'lambda', labels = T)
plot(fit0, main = 'Ridge', labels = T)
plot(fit.enet, xvar = 'lambda', labels = T)
plot(fit1, main = 'Elastic Net', labels = T)

plotres(fit1, which = 1:4)
plotres(fit0.5, which = 1:4)
plotres(fit0, which = 1:4)

#Print residual and other SAS type plots to help analyze the different models

#Print AIC, ASE and other required stats for each model.

#Print coefficients for each model
coef.cv.glmnet(fit0)
coef.cv.glmnet(fit0.5)
coef.cv.glmnet(fit1)

# clean, transform, encode test data
df.test$SalePrice <- rep(0, 1459)

# for creating an imputed test set with test.csv
if (writeImputeData == T) {
  df.test <- cleanData(df.test, isTrain = F)
  df.test <- transformData(df.test, isTest = T)
  df.test <- encodeData(df.test)
  df.test <- mice(df.test[, names(df.test)], method="rf")
  df.test <- complete(df.test)
  write.csv(x = df.test, file = "true_test.csv", row.names = F)
}

# Create matrix from df.test
#Prep df.test for conversion to mtx.test by creating individual atomic vectors for factors.
#mdlmtx.test <- model.matrix(SalePrice ~ ., data = df.test)
#df.mdlmtx.test <- data.frame(mdlmtx.test)
##df.mdlmtx.test <- subset(df.mdlmtx.test, select = -c(X.Intercept., BsmtQualTA, BsmtExposureGd))
#df.mdlmtx.test$HouseStyle2.5Fin <- 0
#df.mdlmtx.test$Exterior2ndOther <- 0
#df.mdlmtx.test$SalePrice <- 0


mtx.test <- data.matrix(select(df.test, -BsmtFinSF2, -SalePrice))

# generate predictions for ridge, lasso and elastic net
pred.lasso <- predict(fit.lasso, type = 'link', newx = mtx.test, s=fit1$lambda.min, terms = 'Id')
pred.ridge <- predict(fit.ridge, type = 'link', newx = mtx.test, s=fit1$lambda.min, terms = 'Id')
pred.enet <- predict(fit.enet, type = 'link', newx = mtx.test, s=fit1$lambda.min, terms = 'Id')

# generate predictions for ridge, lasso and elastic net (NO MODEL.MATRIX!!!)
pred.lassob <- predict(fit.lassob, type = 'link', newx = mtx.test, s=fit1$lambda.min, terms = 'Id')
pred.ridgeb <- predict(fit.ridgeb, type = 'link', newx = mtx.test, s=fit1$lambda.min, terms = 'Id')
pred.enetb <- predict(fit.enetb, type = 'link', newx = mtx.test, s=fit1$lambda.min, terms = 'Id')

#Convert glmnet prediction matrices to data frames and rename probability columns
pred.lasso <- as.data.frame(pred.lasso)
names(pred.lasso)[names(pred.lasso) == "1"] <- "LassoPredict"

pred.ridge <- as.data.frame(pred.ridge)
names(pred.ridge)[names(pred.ridge) == "1"] <- "RidgePredict"

pred.enet <- as.data.frame(pred.enet)
names(pred.enet)[names(pred.enet) == "1"] <- "ENetPredict"



##create submission files for lasso, ridge and elastic net predictions
df.compTitanicTest <- tibble::rowid_to_column(df.compTitanicTest, 'RowId')

#df.test$PredPrice <- predict.lm(object = fit.both, newdata = df.test)
#df.test$SalePrice <- exp(df.test$PredPrice)

# create kaggle data frames
kaggleColumns <- c("Id", "SalePrice")
df.kaggle <- df.test[kaggleColumns]
write.csv(x = df.kaggle, file = "kaggle_predictions.csv", row.names = F)

# RMSE = 0.08283118 fit.both
# Kaggle = 0.12127 - fit.both