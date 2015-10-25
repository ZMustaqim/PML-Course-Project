library(caret)
library(ggplot2)
library(dplyr)
library(rattle)
library(rpart)
library(rpart.plot)

set.seed(33)

trainingDataRaw = read.csv("pml-training.csv")
testingDataRaw = read.csv("pml-testing.csv")


trainingDataClean = trainingDataRaw[,colSums(is.na(trainingDataRaw)) == 0]

irrelevant_indices = 1:7
relevant_indices = - irrelevant_indices
trainingDataClean = trainingDataClean[,relevant_indices]

testingDataClean = testingDataRaw[,colSums(is.na(testingDataRaw)) == 0]

irrelevant_indices = 1:7
relevant_indices = - irrelevant_indices
testingDataClean = testingDataClean[,relevant_indices]
rm(trainingDataRaw, testingDataRaw, irrelevant_indices, relevant_indices)

inTrain = createDataPartition(y=trainingDataClean$classe, p=0.75, list= FALSE)
training = trainingDataClean[inTrain,]
testing = trainingDataClean[-inTrain,]

dim(training); dim(testing)

trainingStep = training # Data Alternative A

trainingStepB = training # Data Alternative B
rm(training, inTrain)

nzv = nearZeroVar(trainingStep, saveMetrics = TRUE)
trainingNZV = trainingStep[, !nzv$zeroVar & !nzv$nzv ]
dim(trainingNZV)

trainingStep = trainingNZV
rm(nzv, trainingNZV)

nzvB = nearZeroVar(trainingStepB, saveMetrics = TRUE)
trainingNZVB = trainingStepB[, !nzvB$zeroVar & !nzvB$nzv ]
dim(trainingNZVB)

trainingStepB = trainingNZVB
rm(nzvB, trainingNZVB)

trainingTemp = trainingStep[,-53]
descrCor <- cor(trainingTemp)
highlyCorrIndices = findCorrelation(descrCor, cutoff = 0.75)
trainingCor = trainingTemp[,-highlyCorrIndices]
trainingCor$classe = trainingStep$classe
trainingStep = trainingCor

trainingTempB = trainingStepB[,-53]
descrCorB <- cor(trainingTempB)
highlyCorrIndicesB = findCorrelation(descrCorB, cutoff = 0.75)
trainingCorB = trainingTempB[,-highlyCorrIndicesB]
trainingCorB$classe = trainingStepB$classe
trainingStepB = trainingCorB

rm(trainingCor, trainingCorB, trainingTemp, trainingTempB, descrCor, descrCorB, highlyCorrIndices, highlyCorrIndicesB)

trainingSemifinal = trainingStep

trainingSemifinalB = trainingStepB

trainingSemifinalC = trainingStepB

rm(trainingStep, trainingStepB)


modelFit <- train(classe ~ .,method="rpart",data=trainingSemifinal)
fancyRpartPlot(modelFit$finalModel, sub = "decision trees with no cross-validation")   

predicted = predict(modelFit, newdata=testing)
confusionMatrix(predicted, testing$classe)

ctrlB = trainControl(method="repeatedcv", number = 10, repeats = 2)
modelFitB <- train(classe ~., data = trainingSemifinalB, method="rpart", tuneLength = 15, trControl=ctrlB)
fancyRpartPlot(modelFitB$finalModel, sub = "decision trees with with 10-fold cross-validation repeated 2 times")

predictedB = predict(modelFitB, newdata=testing)
confusionMatrix(predictedB, testing$classe)

ctrlC = trainControl(method="repeatedcv", number = 3, repeats = 2)
modelFitC <- train(classe ~., data = trainingSemifinalC, method="rf", tuneLength = 2, trControl=ctrlC)

predictedC = predict(modelFitC, newdata=testing)
confusionMatrix(predictedC, testing$classe)

answers = predict(modelFitC, newdata=testingDataClean)
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(answers)



