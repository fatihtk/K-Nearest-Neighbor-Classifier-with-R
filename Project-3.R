# I�ll read in the training and testing datasets; the split into train and test should be the same as that used by most other sources:

set.seed(1)
train <- read.csv("mnist_train.psv", sep="|", as.is=TRUE, header=FALSE)
test <- read.csv("mnist_test.psv", sep="|", as.is=TRUE, header=FALSE)

#Looking at the data, we see that it has 257 columns: a first column giving the true digit class and the others giving the pixel intensity (in a scale from -1 to 1) of the 16x16 pixel image.


dim(train)
train[1:10,1:10]

#We can plot what the image actually looks like in R using the rasterImage function:

y <- matrix(as.matrix(train[3400,-1]),16,16,byrow=TRUE)
y <- 1 - (y + 1)*0.5

plot(0,0)
rasterImage(y,-1,-1,1,1)

#With a minimal amount of work, we can build a much better visualization of what these digits actually look like. Here is a grid of 35 observations. 

iset <- sample(1:nrow(train),5*7)
par(mar=c(0,0,0,0))
par(mfrow=c(5,7))
for (j in iset) {
  y <- matrix(as.matrix(train[j,-1]),16,16,byrow=TRUE)
  y <- 1 - (y + 1)*0.5
  
  plot(0,0,xlab="",ylab="",axes=FALSE)
  rasterImage(y,-1,-1,1,1)
  box()
  text(-0.8,-0.7, train[j,1], cex=3, col="red")
}

# Now, let�s extract out the matrices and classes:

Xtrain <- as.matrix(train[,-1])
Xtest <- as.matrix(test[,-1])
ytrain <- train[,1]
ytest <- test[,1]

# I now want to apply a suite of techniques that we have studied to trying to predict the correct class for each handwritten digit.
# As a simple model, we can use k-nearest neighbors. I set k equal to three, which in a multi-class model says to use the closest point unless the next two closest points agree on the class label.

library(FNN)
predKnn <- knn(Xtrain,Xtest,ytrain,k=3)

# For ridge regression, I�ll directly use the multinomial loss function and let the R function do cross validation for me. The glmnet package is my preferred package for doing ridge regressions, just remember to set alpha to 0.

library(glmnet)
outLm <- cv.glmnet(Xtrain, ytrain, alpha=0, nfolds=3,
                   family="multinomial")
predLm <- apply(predict(outLm, Xtest, s=outLm$lambda.min,
                        type="response"), 1, which.max) - 1L

# We can also run a random forest model. It will run significantly faster if I restrict the maximum number of nodes somewhat.

library(randomForest)
outRf <- randomForest(Xtrain,  factor(ytrain), maxnodes=10)
predRf <- predict(outRf, Xtest)

# Gradient boosted trees also run directly on the multiclass labels. The model performs much better if I increase the interaction depth slightly. Increasing it past 2-3 is beneficial in large models, but rarely useful with smaller cases like this. I could also play with the learning rate, but won�t fiddle with that here for now.

library(gbm)
outGbm <- gbm.fit(Xtrain,  factor(ytrain), distribution="multinomial",
                  n.trees=500, interaction.depth=2)

predGbm <- apply(predict(outGbm, Xtest, n.trees=outGbm$n.trees),1,which.max) - 1L

# Finally, we will also fit a support vector machine. We can give the multiclass problem directly to the support vector machine, and one-vs-one prediction is done on all combinations of the classes. I found the radial kernel performed the best and the default cost also worked well:

library(e1071)
outSvm <- svm(Xtrain,  factor(ytrain), kernel="radial", cost=1)
predSvm <- predict(outSvm, Xtest)

# We see that the methods differ substantially in how predictive they are on the test dataset:

mean(predKnn != ytest)
mean(predLm != ytest)
mean(predRf != ytest)
mean(predGbm != ytest)
mean(predSvm != ytest)

# The tree-models perform far worse than the others. The ridge regression seems to work quite well given that it is constrained to only linear separating boundaries. The support vector machine does about twice as well but utilizing the kernel trick to easily fit higher dimensional models. The k-nearest neighbors performs just slightly better than the support vector machine.

# You may have noticed that some of the techniques (when verbose is set to True) spit out a mis-classification rate by class. This is useful to assess the model when there is more than two categories. For example look at where the ridge regression and support vector machines make the majority of their errors:

tapply(predLm != ytest, ytest, mean)

tapply(predSvm != ytest, ytest, mean)

# We see that 8 and 3 are particularly difficult, with 1 being quite easy to predict.

# We might think that a lot 8�s and 3�s are being mis-classified as one another (they do look similar in some ways). Looking at the confusion matricies we see that this is not quite the case:

table(predLm,ytest)

table(predSvm,ytest)

# We also see that the points where these two models make mistakes do not have too great of an overlap.

table(predSvm != ytest, predLm != ytest)

# this gives evidence that stacking could be beneficial.

# We can pick out a large sample of images that are 3�s and see if these are particularly difficult to detect.

iset <- sample(which(train[,1] == 3),5*7)
par(mar=c(0,0,0,0))
par(mfrow=c(5,7))
for (j in iset) {
  y <- matrix(as.matrix(train[j,-1]),16,16,byrow=TRUE)
  y <- 1 - (y + 1)*0.5
  
  plot(0,0,xlab="",ylab="",axes=FALSE)
  rasterImage(y,-1,-1,1,1)
  box()
  text(-0.8,-0.7, train[j,1], cex=3, col="red")
}

# For the most part though, these do not seem difficult for a human to classify.

# What if we look at the actual mis-classified points. Here are the ones from the support vector machine:

iset <- sample(which(predSvm != ytest),7*7)
par(mar=c(0,0,0,0))
par(mfrow=c(7,7))
for (j in iset) {
  y <- matrix(as.matrix(test[j,-1]),16,16,byrow=TRUE)
  y <- 1 - (y + 1)*0.5
  
  plot(0,0,xlab="",ylab="",axes=FALSE)
  rasterImage(y,-1,-1,1,1)
  box()
  text(-0.8,-0.7, test[j,1], cex=3, col="red")
  text(0.8,-0.7, predSvm[j], cex=3, col="blue")
}

# And the ridge regression:

iset <- sample(which(predLm != ytest),7*7)
par(mar=c(0,0,0,0))
par(mfrow=c(7,7))
for (j in iset) {
  y <- matrix(as.matrix(test[j,-1]),16,16,byrow=TRUE)
  y <- 1 - (y + 1)*0.5
  
  plot(0,0,xlab="",ylab="",axes=FALSE)
  rasterImage(y,-1,-1,1,1)
  box()
  text(-0.8,-0.7, test[j,1], cex=3, col="red")
  text(0.8,-0.7, predLm[j], cex=3, col="blue")
}


