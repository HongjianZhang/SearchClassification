library(ISLR)
library(class)
library(boot)
training <- read.csv("Downloads/training.csv", header=TRUE)
# VARIABLE SELECTION

#logistic regression
train = sample(dim(training)[1], dim(training)[1] / 2)
training.train= training[train,]
training.test = training[-train,]
glm.fit = glm(relevance~., data=training, family=binomial)
summary(glm.fit)
# Cross Validation
set.seed(17)
cv.error.10=rep(0,10)
for (i in 1:10) {
  glm.fit=glm(relevance~poly(query_length+is_homepage+sqrt(sig1)+sig2+log(sig6+0.01)+sig7+sig8, i), data=training)
  cv.error.10[i]=cv.glm(training, glm.fit, K=10)$delta[1]
}
cv.error.10
#
set.seed(1)
glm.fit=glm(relevance~query_length+is_homepage+sqrt(sig1)+sig2+log(sig6+0.01)+sig7*sig8, data=training.train, family=binomial)
glm.probs=predict(glm.fit, training.test, type="response")
glm.pred=rep("0", 40023)
glm.pred[glm.probs>.5]="1"
table(glm.pred, relevance[-train])
mean(glm.pred!=relevance[-train])


#lasso, 
library(glmnet)
#split data
set.seed(1)
train=sample(c(TRUE, FALSE), nrow(training), rep=TRUE)
test=(!train)
#set x and y, and grid for glmnet
x=model.matrix(relevance~.,training)[,-1]
y=training$relevance
grid=10^seq(10,-2,length=100)
#lasso
lasso.mod=glmnet(x[train,], y[train], alpha=1, lambda=grid)
plot(lasso.mod)
#cross validation for lasso
set.seed(1)
cv.lasso=cv.glmnet(x[train,], y[train], alpha=1)
plot(cv.lasso)
bestlam=cv.lasso$lambda.min
lasso.pred=predict(lasso.mod,s=bestlam,newx=x[test,])
out=glmnet(x, y, alpha=1, lambda=grid)
lasso.coef=predict(out, type="coefficients", s=bestlam)
lasso.coef

# Knn
train.X=cbind(query_length,is_homepage,sqrt(sig1),sig2,log(sig6+0.01),sig7,sig8)[train,]
test.X=cbind(query_length,is_homepage,sqrt(sig1),sig2,log(sig6+0.01),sig7,sig8)[-train,]
train.relevance=relevance[train]
set.seed(1)
knn.error = rep(0,10)
for (i in 51:60) {
knn.pred=knn(train.X, test.X, train.relevance, k=i)
knn.error[i-50] = mean(knn.pred!=relevance[-train])
}
table(knn.pred, relevance[-train])

# Decision Tree, in order to avoid taking log of 0, we add 0.01 to sig2 and sig6
library(tree)
relevant=ifelse(relevance==0, "No", "Yes")
training.train=data.frame(training.train, relevant)
tree.training=tree(relevant~query_length+is_homepage+sqrt(sig1)+sig2+log(sig6+0.01)+sig7+sig8, training)
summary(tree.training)
plot(tree.training)
text(tree.training, pretty=0)
# CV on Decision Tree
set.seed(2)
train=sample(c(TRUE, FALSE), nrow(training), rep=TRUE)
training.test=training[-train,]
relevant.test=relevant[-train]
tree.training=tree(relevant~query_length+is_homepage+sqrt(sig1)+sig2+log(sig6+0.01)+sig7+sig8, training, subset=train)
tree.pred=predict(tree.training, relevant.test, type="class")
table(tree.pred, relevant.test)
set.seed(3)
cv.dtree=cv.tree(tree.training, FUN=prune.misclass)
names(cv.dtree)
cv.dtree
# best terminal node is 3 in this case
prune.dtree=prune.misclass(tree.training, best=3)
plot(prune.dtree)
text(prune.dtree,pretty=0)
dtree.predict=predict(prune.dtree,training.test, type="class")
table(dtree.predict, relevant.test)


# Random Forest with Bagging
library(randomForest)
train=sample(1:nrow(training), nrow(training)/2)
bag.test=training[-train, "relevance"]
set.seed(1)
bag.training=randomForest(relevance~query_length+is_homepage+sig1+sig2+sig3+sig4+sig5+sig6+sig7+sig8, data=training, subset=train, mtry=sqrt(7), importance=TRUE)
bag.training
yhat.bag=predict(bag.training, newdata=training[-train,])
plot(yhat.bag, bag.test)
abline(0, 1)
# mse
mean((yhat.bag-bag.test)^2)
# error rate
bag.pred=rep("0", 40023)
bag.pred[yhat.bag>.5]="1"
mean(bag.pred!=bag.test)


# boost
train = sample(dim(training)[1], dim(training)[1] / 2)
training.train= training[train,]
training.test = training[-train,]
y<-training.train[,13]
# change y's 0 to -1
i=1
while (i<=40023) {
  if (y[i]==0) {
    y[i]=-1
  }
  i<-i+1
}
x<-training.train[,3:12]
y_test<-training.test[,13]
#change y_test's 0 to -1
i=1
while (i<=40023) {
  if (y_test[i]==0) {
    y_test[i]=-1
  }
  i<-i+1
}
x_test<-training.test[,3:12]
train_error<-rep(0, 500)
test_error<-rep(0,500)
f<-rep(0,40023)
f_test<-rep(0,40022)
i<-1

library(rpart)
while(i<=500){
  w<-exp(-y*f)
  w<-w/sum(w)
  fit<-rpart(y~.,x,w,method="class")
  g <- -1 + 2*(predict(fit,x)[,2]>.5)
  g_test<--1+2*(predict(fit,x_test)[,2]>.5)
  e<-sum(w*(y*g<0))
  alpha<-.5*log ( (1-e) / e )
  f<-f+alpha*g
  f_test<-f_test+alpha*g_test
  train_error[i]<-sum(1*f*y<0)/40023
  test_error[i]<-sum(1*f_test*y_test<0)/40022
  i<-i+1
}
plot(seq(1,500),test_error,type="l",
     ylim=c(0,.5),
     ylab="Error Rate",xlab="Iterations",lwd=2)
lines(train_error,lwd=2,col="purple")
legend(4,.5,c("Training Error","Test Error"),   
       col=c("purple","black"),lwd=2)
mean(test_error)


#svm
library(e1071)
svm.radial=svm(relevance~query_length+is_homepage+sqrt(sig1)+sig2+log(sig6+0.01)+sig7+sig8, data=training.train, kernel="radial", cost=5)
train.pred=predict(svm.radial,training.test)
train.pred=ifelse(train.pred>0.5, 1.0)
table(train.pred, training.test$relevance)
mean(train.pred!=train$relevance)
#

x_svm=cbind(query_length,is_homepage,sqrt(sig1),sig2,log(sig6+0.01),sig7,sig8)[train,]
y_svm<-as.factor(training.train[,13])
svm.fit<-svm(x,y_svm)
1-sum(y==predict(svm.fit,x))/length(y_svm)
y_test_svm<-as.factor(training.test[,13])
1-sum(y_test_svm==predict(svm.fit,x_test))/length(y_test_svm)


