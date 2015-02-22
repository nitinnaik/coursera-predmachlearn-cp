# Practical Machine Learning - Course Project

Writeup for Coursera's Practical Machine Learning from Johns Hopkins University course project.


GitHub repo: http://github.com/nitinnaik/coursera-predmachlearn-cp

RPubs: http://rpubs.com/nitinnaik/coursera-predmachlearn-cp



##Introduction

"Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement ??? a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). "

The goal of your project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases. 

1. Your submission should consist of a link to a Github repo with your R markdown and compiled HTML file describing your analysis. Please constrain the text of the writeup to < 2000 words and the number of figures to be less than 5. It will make it easier for the graders if you submit a repo with a gh-pages branch so the HTML page can be viewed online (and you always want to make it easy on graders :-).
2. You should also apply your machine learning algorithm to the 20 test cases available in the test data above. Please submit your predictions in appropriate format to the programming assignment for automated grading. See the programming assignment for additional details. 

##Reproduciblity

In order to reproduce the same results,  

1. Required packages: caret, rpart, rpart.plot, RColorBrewer, rattle, randomForest
*Note:To install use install.packages. e.g. to install caret package type install.packages("caret")
and
2. setting a pseudo random seed same as here.


```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(rpart)
library(rpart.plot)
library(RColorBrewer)
library(rattle)
```

```
## Rattle: A free graphical interface for data mining with R.
## Version 3.4.1 Copyright (c) 2006-2014 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```

```r
library(randomForest)
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

Finally, load the same seed with the following line of code:

```r
set.seed(54321)
```

##Getting the data


```r
filesDirectory = "./data"
testUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
trainUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
trainFile <- "pml-training.csv"
testFile <- "pml-testing.csv"
trainFilePath <- paste(filesDirectory, trainFile, sep = "/")
testFilePath <- paste(filesDirectory, testFile, sep = "/")
if (!file.exists(filesDirectory)) {
   dir.create(filesDirectory)
}
if (!file.exists(trainFilePath)) {
   download.file(trainUrl, destfile = trainFilePath, method="curl")
}
if (!file.exists(testFilePath)) {
   download.file(testUrl, destfile = testFilePath, method="curl")
}
training <- read.csv(trainFilePath, na.strings=c("NA","#DIV/0!",""))
testing <- read.csv(testFilePath, na.strings=c("NA","#DIV/0!",""))
```

##Partioning the training set into two

Partioning Training data set into two data sets, 60% for myTraining, 40% for myTesting:


```r
inTrain <- createDataPartition(y=training$classe, p=0.6, list=FALSE)
myTraining <- training[inTrain, ]
myTesting <- training[-inTrain, ]
dim(myTraining)
```

```
## [1] 11776   160
```

```r
dim(myTesting)
```

```
## [1] 7846  160
```


##Cleaning/Transforming the data

Transformation 1: Cleaning NearZeroVariance Variables


```r
myDataNZV <- nearZeroVar(myTraining, saveMetrics=TRUE)

myNZVvars <- names(myTraining) %in% c("new_window", "kurtosis_roll_belt", "kurtosis_picth_belt",
"kurtosis_yaw_belt", "skewness_roll_belt", "skewness_roll_belt.1", "skewness_yaw_belt",
"max_yaw_belt", "min_yaw_belt", "amplitude_yaw_belt", "avg_roll_arm", "stddev_roll_arm",
"var_roll_arm", "avg_pitch_arm", "stddev_pitch_arm", "var_pitch_arm", "avg_yaw_arm",
"stddev_yaw_arm", "var_yaw_arm", "kurtosis_roll_arm", "kurtosis_picth_arm",
"kurtosis_yaw_arm", "skewness_roll_arm", "skewness_pitch_arm", "skewness_yaw_arm",
"max_roll_arm", "min_roll_arm", "min_pitch_arm", "amplitude_roll_arm", "amplitude_pitch_arm",
"kurtosis_roll_dumbbell", "kurtosis_picth_dumbbell", "kurtosis_yaw_dumbbell", "skewness_roll_dumbbell",
"skewness_pitch_dumbbell", "skewness_yaw_dumbbell", "max_yaw_dumbbell", "min_yaw_dumbbell",
"amplitude_yaw_dumbbell", "kurtosis_roll_forearm", "kurtosis_picth_forearm", "kurtosis_yaw_forearm",
"skewness_roll_forearm", "skewness_pitch_forearm", "skewness_yaw_forearm", "max_roll_forearm",
"max_yaw_forearm", "min_roll_forearm", "min_yaw_forearm", "amplitude_roll_forearm",
"amplitude_yaw_forearm", "avg_roll_forearm", "stddev_roll_forearm", "var_roll_forearm",
"avg_pitch_forearm", "stddev_pitch_forearm", "var_pitch_forearm", "avg_yaw_forearm",
"stddev_yaw_forearm", "var_yaw_forearm")

myTraining <- myTraining[!myNZVvars]

dim(myTraining)
```

```
## [1] 11776   100
```

Transformation 2: Remove first ID variable so that it does not interfer with ML Algorithms:


```r
myTraining <- myTraining[c(-1)]
```

Transformation 3: Removing Variables with too many (i.e. more that 60% NAs) NAs.


```r
trainingV3 <- myTraining 
for(i in 1:length(myTraining)) { 
        if( sum( is.na( myTraining[, i] ) ) /nrow(myTraining) >= .6 ) { 
		for(j in 1:length(trainingV3)) {
			if( length( grep(names(myTraining[i]), names(trainingV3)[j]) ) ==1)  {
				trainingV3 <- trainingV3[ , -j]
			}	
		} 
	}
}

dim(trainingV3)
```

```
## [1] 11776    58
```

```r
myTraining <- trainingV3
rm(trainingV3)
```

Perform above 3 transformations, myTesting and testing datasets.


```r
clean1 <- colnames(myTraining)
clean2 <- colnames(myTraining[, -58]) #already with classe column removed
myTesting <- myTesting[clean1]
testing <- testing[clean2]

dim(myTesting)
```

```
## [1] 7846   58
```

```r
dim(testing)
```

```
## [1] 20 57
```

In order to ensure proper functioning of Decision Trees and RandomForest with the Test data set (data set provided), we need to make the data into the same type.


```r
for (i in 1:length(testing) ) {
        for(j in 1:length(myTraining)) {
		if( length( grep(names(myTraining[i]), names(testing)[j]) ) ==1)  {
			class(testing[j]) <- class(myTraining[i])
		}      
	}      
}
testing <- rbind(myTraining[2, -58] , testing)
testing <- testing[-1,]
```
##Methodolody
Prediction evaluations will be based on maximizing the accuracy and minimizing the out-of-sample error. Decision tree and random forest algorithms will be used. The algorithm with  the highest accuracy will be chosen as our final model.
Cross-validation will be performed by subsampling our training data set randomly without replacement into 2 subsamples: myTraining data (60% of the original Training data set) and myTesting data (40%). Our models will be fitted on the myTraining dataset, and tested on the myTesting dataset. Once the most accurate model is choosen, it will be tested on the original Testing data set.

##Expected out-of-sample error

The expected out-of-sample error will correspond to the quantity: 1-accuracy in the cross-validation data. Accuracy is the proportion of correct classified observation over the total sample in the subTesting data set. Expected accuracy is the expected accuracy in the out-of-sample data set (i.e. original testing data set). Thus, the expected value of the out-of-sample error will correspond to the expected number of missclassified observations/total observations in the Test data set, which is the quantity: 1-accuracy found from the cross-validation data set.

##Using Decision Trees for prediction 


```r
modFitA1 <- rpart(classe ~ ., data=myTraining, method="class")
```

Note: to view the decision tree with fancy run this command:


```r
fancyRpartPlot(modFitA1)
```

![](course_project_report_files/figure-html/unnamed-chunk-11-1.png) 

Predicting:


```r
predictionsA1 <- predict(modFitA1, myTesting, type = "class")
```

Using confusion Matrix to test results:

```r
confusionMatrix(predictionsA1, myTesting$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2154   60    7    3    0
##          B   75 1378  110    7    0
##          C    3   73 1226  118   57
##          D    0    7   15  970   77
##          E    0    0   10  188 1308
## 
## Overall Statistics
##                                           
##                Accuracy : 0.8968          
##                  95% CI : (0.8898, 0.9034)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.8694          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9651   0.9078   0.8962   0.7543   0.9071
## Specificity            0.9875   0.9697   0.9613   0.9849   0.9691
## Pos Pred Value         0.9685   0.8777   0.8301   0.9074   0.8685
## Neg Pred Value         0.9861   0.9777   0.9777   0.9534   0.9789
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2745   0.1756   0.1563   0.1236   0.1667
## Detection Prevalence   0.2835   0.2001   0.1882   0.1362   0.1919
## Balanced Accuracy      0.9763   0.9387   0.9287   0.8696   0.9381
```

## Using Random Forests for prediction


```r
modFitB1 <- randomForest(classe ~. , data=myTraining)
```

Predicting:

```r
predictionsB1 <- predict(modFitB1, myTesting, type = "class")
```
Using confusion Matrix to test results:

```r
confusionMatrix(predictionsB1, myTesting$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2229    0    0    0    0
##          B    3 1518    1    0    0
##          C    0    0 1366    4    0
##          D    0    0    1 1282    1
##          E    0    0    0    0 1441
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9987          
##                  95% CI : (0.9977, 0.9994)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9984          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9987   1.0000   0.9985   0.9969   0.9993
## Specificity            1.0000   0.9994   0.9994   0.9997   1.0000
## Pos Pred Value         1.0000   0.9974   0.9971   0.9984   1.0000
## Neg Pred Value         0.9995   1.0000   0.9997   0.9994   0.9998
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2841   0.1935   0.1741   0.1634   0.1837
## Detection Prevalence   0.2841   0.1940   0.1746   0.1637   0.1837
## Balanced Accuracy      0.9993   0.9997   0.9990   0.9983   0.9997
```

## Choosing algorithm to predict test data

As you can see from above, random Forests yielded better prediction than decision Trees.
Accuracy for Random Forest model was 0.9987 (95% CI : (0.9977, 0.9994)) compared to 0.8968 (95% CI : (0.8898, 0.9034)) for Decision Tree model. The random Forest model is choosen. The accuracy of the model is 0.995. The expected out-of-sample error is estimated at 0.0013, or 0.13%. The expected out-of-sample error is calculated as 1 - accuracy for predictions made against the cross-validation set. Our Test data set comprises 20 cases.

## Generating Files to submit as answers for the Assignment:

Using random forests on testing dataset for generarting files.

```r
predictionsB2 <- predict(modFitB1, testing, type = "class")
```

Function to generate files with predictions to submit for assignment

```r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(predictionsB2)
```
