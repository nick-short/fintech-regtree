          ## Clear workspace

rm(list = ls())

          ## Load Libraries


library(caret)
library(rpart)
library(tidyverse)
library(doParallel)


       ## Run the model on the training set


# Implement 10-fold cross-validation (CV) ; the index command in
# trainControl effectively specifies stratified CV

# The first path is for using the remote computing environment (RCE)
#train_final <- readRDS(file = 'shared_space/nis130/data/train_final.RDS')
train_final <- readRDS(file = 'data/train_final.RDS')

# Initialize parallel processing
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)

cv_folds <- createMultiFolds(train_final$Label, k = 10, times = 1)
#cv_cntrl <- trainControl(method = "cv", number = 10, index = cv_folds)
cv_cntrl <- trainControl(method = "cv", number = 10, index = cv_folds, allowParallel = TRUE)

# Run the model
start_time <- Sys.time()
rpart_cv1 <- train(Label ~ ., data = train_final, method = "rpart", 
                   trControl = cv_cntrl, tuneLength = 7) # should be method = "repeatedcv" and repeats = 3 for repeated CV
end_time <- Sys.time()
end_time - start_time

# Save output
#saveRDS(rpart_cv1, file = "shared_space/nis130/data/rpart1.RDS")
saveRDS(rpart_cv1, file = "data/rpart1.RDS")

# Stop parallel processing
stopCluster(cluster)
registerDoSEQ()

# Clear up some memory
rm(train_final)


          ## Predict outcomes on the test set


#test_final <- readRDS(file = 'shared_space/nis130/data/test_final.RDS')
test_final <- readRDS(file = 'data/test_final.RDS')

# Initialize parallel processing
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)

predictions <- predict(rpart_cv1, test_final)
#Save the results.  Note: train_final is stored in rpart_cv1$trainingData
saveRDS(cv_folds, cv_cntrl, predictions, file = "shared_space/nis130/predictions.RData")

# Stop parallel processing
stopCluster(cluster)
registerDoSEQ()










