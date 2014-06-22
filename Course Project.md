# Course Project - Practical Machine Learning 

## Introduction
Nowadays, using some devices such as *Jawbone Up*, *Nike FuelBand* and *Fitbit*, it is now possible to record a large amount of data about personal activity relatively inexpensively. Those records can quantify how much of a particular activity people do but also how well they do it. Moreover they may help researchers and sport professionals to understand better which features matter for a movement execution. 

In this course project, we'll try to use machine learning tools seen in Practical Machine Learning lectures. The data for this project come from [Groupware](http://groupware.les.inf.puc-rio.br/har). It's a collection of records of accelerometers posed on the belt, forearm, arm and dumbell of 6 participants while they're performing barbell lifts correctly and incorrectly in 5 different ways.


```r
library(caret)
```

## Data cleaning and processing

In this study, we'll focus on the training set since we'll evaluate the efficiency by cross validation. Nonetheless, we'll mention the results of the prediction on the test set at the end of our study. 

First of all, we might download (if necessary) the two data sets given on the course project page:


```r
# Create a folder where data will be downloaded
if(!file.exists("./data")){
        dir.create("./data")
}
# Download training set
if (!file.exists("./data/trainingset.csv")){
        url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
        download.file(url,"./data/trainingset.csv",method="curl")
}
# Download test set
if (!file.exists("./data/testset.csv")){
        url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
        download.file(url, "./data/testset.csv",method="curl")
}
```

Then, we use **read.csv()** function to create two data sets corresponding to the train and test sets in csv format:


```r
train <- read.csv("./data/trainingset.csv")
test <- read.csv("./data/testset.csv")
```

By looking at the training set, we find out that a lot of columns are empty or filled with NA values.The following code shows that there are only 406 totally complete observations and especially that **100 columns are filled with less than 90% of values**.


```r
# Number of complete observations
sum(complete.cases(train))
```

```
## [1] 406
```

```r
# Number of column filled with less than 90% of values
emptyColumn <- 0
for(name in names(train)){
        index <- (is.na(train[,name])) | (train[,name]=="")
        if(sum(index) > 0.1*nrow(train)){
              emptyColumn <- emptyColumn +1  
        }
}
emptyColumn
```

```
## [1] 100
```

So now we are facing a choice, whether to keep only the 406 complete observations or the columns which have 100% of values. After several simulation, I decided *a posteriori* to keep only the columns which have 100% of values.


```r
cleanedTrain <- train
for(name in names(train)){
        index <- (is.na(train[,name])) | (train[,name]=="")
        if(sum(index)!=0){
                cleanedTrain[,name] <- NULL
        }
}
```

It can also be noticed, that 7th first columns of our new data set don't give any information in the analysis: **X** is just an increment variable, **user_name** corresponds to the participant's first name, the **raw_timestamp_part_1**, **raw_timestamp_part_2**, **cvtd_timestamp**, **new_window** and **num_window** are related to time variables which shouldn't have a link with the experiment results. Consequently we should get rid of them.


```r
cleanedTrain <- cleanedTrain[,8:60]
```

Finally, we ends up with a cleaned data set with **19622 observations** and **53 variables** (52 features and 1 outcome variable, the **classe**). And e can now go to the machine learning part of our study. 

## Feature selection

Even if our data set has been cleaned, 19622 observations by 53 variables is still a pretty big data set. Some features may contain more information than others. It's possible to evaluate the features that contain a certain ratio of variance thank to the correlation matrix.


```r
corrMatrix <- abs(cor(cleanedTrain[,-53]))
diag(corrMatrix) <- 0
## Features with a correlation of more than 80%
features <- c(names(which(corrMatrix > 0.8, arr.ind=T)[,1]), "classe")
length(features)
```

```
## [1] 39
```

We then create the two train and test data sets of **39 variables** that will be used by the machine learning algorithms.


```r
training <-cleanedTrain[, features]
testing <- test[, c(features[-length(features)],names(test)[length(test)])]
```

## Number of observations retained

The algorithms we may use can take quite some time in terms of computing (particularly there because we opted for the Random Forest algorithm). As said before, we still have a huge quantity of data and we probably should consider to remove certain observations of our data set. Here is a plot of the learning curve with the Random Forest algorithm in order to evaluate how much data we'll keep.

<img src="figure/learning curve.png" title="plot of chunk learning curve" alt="plot of chunk learning curve" style="display: block; margin: auto;" />

## Algorithm used and my model

We'll use the Random Forest algortihm which is well known for giving the best accuracies.

Then we randomly select 10000 of observations. 


```r
randomSample <- sample(nrow(training), 10000)
training <- training[randomSample,]
```

The model is then evaluated with the **train** function of the caret package. Since the computation takes some time, the following line of code won't be evaluated. I kept the **fit** variable in a .RData file that will be loaded in a hidden line of code readable in the source file.


```r
fit <- train(training$classe~.,data=training, method="rf")
save(fit,file="./models/fit10000.RData")
```




```r
fit$results
```

```
##   mtry Accuracy  Kappa AccuracySD  KappaSD
## 1    2   0.9603 0.9498   0.003733 0.004721
## 2   20   0.9536 0.9413   0.004240 0.005360
## 3   38   0.9449 0.9303   0.005165 0.006547
```

## Estimation of the error with cross-validation

To estimate the accuracy and thus the error rate, we can make a 5-folds cross validation.


```r
control <- trainControl(method = "cv", number = 5)
fitCV <- train(training$classe~.,data=training, method="rf",trControl=control)
save(fitCV,file="./models/fitCV.RData")
```




```r
fitCV$resample
```

```
##   Accuracy  Kappa Resample
## 1   0.9725 0.9652    Fold1
## 2   0.9640 0.9545    Fold2
## 3   0.9650 0.9558    Fold5
## 4   0.9740 0.9672    Fold4
## 5   0.9635 0.9539    Fold3
```

```r
fitCV$results
```

```
##   mtry Accuracy  Kappa AccuracySD  KappaSD
## 1    2   0.9678 0.9593   0.005025 0.006350
## 2   20   0.9637 0.9541   0.005211 0.006591
## 3   38   0.9594 0.9487   0.007141 0.009033
```

The we obtain an accuracy of 96.6% which seems pretty good for a 5 classes classification problem. Thus, we can use this model to make some predictions on our test set.

The we obtain **an accuracy of 96.6%** which seems pretty good for a 5 classes classification problem. Consequently, we can use this model to make some predictions on our test set.

## Preprocessing and PCA

In this study, the data was not normalized and Principal Component Analysis was not used. Indeed, if we keep 2500 observations and use PCA with the Random Forest algorithm, it seems that we get worse results:


```r
fit2500 <- train(training$classe~.,data=training, methode='rf')
fit2500PCA <- train(training$classe~.,data=training, methode='rf', preProcess="pca")
save(fit2500, file="./models/fit2500.RData")
save(fit2500PCA, file="./models/fit2500PCA.RData")
```




```r
fit$results 
```

```
##   mtry Accuracy  Kappa AccuracySD KappaSD
## 1    2   0.8910 0.8621   0.009729 0.01238
## 2   20   0.8819 0.8508   0.009522 0.01209
## 3   38   0.8723 0.8387   0.010286 0.01305
```

```r
fit2500PCA$results
```

```
##   mtry Accuracy  Kappa AccuracySD KappaSD
## 1    2   0.7174 0.6419    0.02028 0.02573
## 2   20   0.6949 0.6135    0.02222 0.02812
## 3   38   0.6939 0.6124    0.02296 0.02893
```

## Test set

To conclude this analysis, I briefly present the results I got from the test set. To predict the outcome for our testing set, we simply use the command:


```r
predict(fit, newdata=testing)
```

After submission to Coursera's server, I got one false prediction over 20, which represents an accuracy of **95%**.

----
