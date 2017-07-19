setwd("H:/Analytics Vidhya/AV_Datafest_Rampaging")
# Load data and libraries -------------------------------------------------
library(e1071)
library(data.table)
library(h2o)
library(caret)
library(stringr)
library(xgboost)
library(readr)

train <- read_csv("H:/Analytics Vidhya/AV_Datafest_Rampaging/train.csv")
test <- read_csv("H:/Analytics Vidhya/AV_Datafest_Rampaging/test.csv")
# Data Preprocessing ---------------------------------------


# Check correlation and remove correlated variables -----------------------

num_col <- colnames(train)[sapply(train, is.numeric)]
num_col <- num_col[!(num_col %in% c("ID","Stock_ID"))]
library(corrplot)
corrplot::corrplot(cor(train[,num_col]),method = "number")

# Machine Learning with XGB -----------------------------------------------
train[is.na(train)] <- -1
test[is.na(test)] <- -1
train$pos_neg <- (train$Positive_Directional_Movement+train$Negative_Directional_Movement)/2
test$pos_neg <- (test$Positive_Directional_Movement+test$Negative_Directional_Movement)/2
train$ID <- NULL
test$ID <- NULL
train$Stock_ID <- NULL
test$Stock_ID <- NULL
train$timestamp <- NULL
test$timestamp <- NULL
test$Outcome <- NA
library(caret)
x <- createDataPartition(y = train$Outcome,p = 0.70,list = F)
dtrain <- train[x,]
dval <- train[-x,]
t1 <- train$Outcome
t2 <- dval$Outcome
t3 <- test$Outcome
train$Outcome<-NULL
dval$Outcome=NULL
test$Outcome=NULL
dtrain <- xgb.DMatrix(data = data.matrix(train),label=data.matrix(t1))
dval <- xgb.DMatrix(data = data.matrix(dval),label=data.matrix(t2))
dtest <- xgb.DMatrix(data = data.matrix(test),label=data.matrix(t3))


#default parameters
params <- list(
  booster = "gbtree",
  objective = "binary:logistic",
  eta=0.05,
  gamma=0,
  max_depth=8,
  subsample=1,
  colsample_bytree=1
)

#first default - model training
xgb1 <- xgb.train(
  params = params
  ,data = dtrain
  ,nrounds = 1000
  ,watchlist = list(val=dtest,train=dtrain)
  ,print.every.n = 10
  #,early.stop.round = 10
  ,maximize = F
  ,eval_metric = "logloss"
)

#model prediction
pred <- predict(xgb1,dtest)
#pred <- ifelse(pred > 0.5,1,0)
mysolution = data.frame( ID = ID, Outcome = pred, stringsAsFactors = FALSE)
submission_xgb = mysolution
write.csv(submission_xgb, file = "xgb9.csv", row.names = FALSE)
