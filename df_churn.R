set.seed(2000)

#load the packages

library(plyr)

library(ggplot2)

library(caret)

library(MASS)

library(party)

library(RColorBrewer)

library(ROCR)

library(rpart)

library(rattle)

library(rpart.plot)

library(caret)

library(e1071)

library(readr)

#load the dataset after setting the working dir

df_churn <- read_csv("Imarticus/Project3ChurnAnalysisinTelecomIndustry/Dataset/churn.csv")
View(df_churn)

head(df_churn)

str(df_churn)

#missing value check

colSums(is.na(df_churn))

length(df_churn$customerID)#count of customers 

#visualization

#gender
count(df_churn$gender)
plot(table(df_churn$Churn,df_churn$gender),col=c("red","green"),xlab="churned Yes or No",ylab="Gender")

#senior citizen
table(df_churn$SeniorCitizen==TRUE)

#partner
plot(table(df_churn$Churn,df_churn$Partner),col=c("red","yellow"))

#dependents
plot(table(df_churn$Churn,df_churn$Dependents),col=c("red","blue"),xlab = "dependents",ylab = "churned yes os no")

#tenure
summary(churn$tenure)
hist(churn$tenure,col = ("blue"),main = "tenure") #right tail observation

#callservice
table(df_churn$CallService)
plot(table(df_churn$Churn,df_churn$CallService),col=c("orange","green"))

#multipleconnection
table(df_churn$MultipleConnections)
plot(table(df_churn$Churn,churn$MultipleConnections),col=c("red","blue","pink")) #no phone service are max 

#internetconnection
table(df_churn$InternetConnection)
plot(table(churn$Churn,churn$InternetConnection),col=c("red","blue","pink")) #people opting for fiber has highest churn ,dsl have less churn


#online security
table(df_churn$OnlineSecurity)
plot(table(df_churn$Churn,df_churn$OnlineSecurity),col=c("red","blue","pink")) #people having no online security churn is high,shows high significance

#onlinebackup
table(df_churn$OnlineBackup)
plot(table(df_churn$Churn,df_churn$OnlineBackup),col=c("red","blue","yellow"))

#device protection
table(df_churn$DeviceProtectionService)
plot(table(df_churn$Churn,df_churn$DeviceProtectionService),col=c("red","blue","yellow"))

#technical help
table(df_churn$TechnicalHelp)
plot(table(df_churn$Churn,df_churn$TechnicalHelp),col=c("red","blue","yellow"))

#onlinetv
table(df_churn$OnlineTV)
plot(table(df_churn$Churn,df_churn$OnlineTV),col=c("red","blue","yellow")) #less significance

#onlinemovies
table(df_churn$OnlineMovies)
plot(table(df_churn$Churn,df_churn$OnlineMovies),col=c("red","blue","yellow"))

#agreement
table(df_churn$Agreement)
plot(table(df_churn$Churn,df_churn$Agreement),col=c("red","blue","yellow")) #high significance

#billing method
table(df_churn$BillingMethod)
plot(table(df_churn$Churn,df_churn$BillingMethod),col=c("red","blue"))

#payment method
table(df_churn$PaymentMethod)
plot(table(df_churn$Churn,df_churn$PaymentMethod),col=c("red","blue","yellow","brown")) #high significance

#service charge
hist(df_churn$MonthlyServiceCharges,col=("pink"))

#churn
table(df_churn$Churn)

#churn amount
hist(churn$TotalAmount,col = "yellow")

view(df_churn)
colnames(df_churn[10:15])

df_churn$OnlineTV<-as.factor(ifelse(df_churn$OnlineTV=="No internet service","No",df_churn$OnlineTV))
df_churn$OnlineMovies<-as.factor(ifelse(df_churn$OnlineMovies=="No internet service","No",df_churn$OnlineMovies))
df_churn$OnlineSecurity<-as.factor(ifelse(df_churn$OnlineSecurity=="No internet service","No",df_churn$OnlineSecurity))
df_churn$OnlineBackup<-as.factor(ifelse(df_churn$OnlineBackup=="No internet service","No",df_churn$OnlineBackup))
df_churn$DeviceProtectionService<-as.factor(ifelse(df_churn$DeviceProtectionService=="No internet service","No",df_churn$DeviceProtectionService))
df_churn$TechnicalHelp<-as.factor(ifelse(df_churn$TechnicalHelp=="No internet service","No",df_churn$TechnicalHelp))
df_churn$MultipleConnections<-as.factor(ifelse(df_churn$MultipleConnections=="No phone service","No",df_churn$MultipleConnections))


view(df_churn)
df_churn$Dependents<-NULL
df_churn$SeniorCitizen<-NULL
df_churn$customerID<-NULL
df_churn$gender<-NULL

view(df_churn)

library(caTools)
names(df_churn)
spl=sample.split(df_churn$Churn,SplitRatio = .70)
train<-df_churn[spl==TRUE,]
test<-df_churn[spl==FALSE,]

#building models

#simple model without cp cart model  

model_tree<-rpart(Churn ~ .,train,method = "class",minbucket=10) #minbucket is used to control the tree if 0 complex tree will be made
plotcp(model_tree)
prp(model_tree)
rpart.plot(model_tree,type = 2,cex=0.6)
fancyRpartPlot(model_tree)
names(train)
dim(train)
dim(test)
rpart.plot(model_tree,type = 5,cex = 0.8) #cex is used for zoomin plot
view(df_churn)
str(df_churn)
names(df_churn)

#testing
pred_tree<-predict(model_tree,newdata = test,type ="class")
table(test$Churn,pred_tree)

#accuracy confusion matrix
(1602+1231)/(1618+1206+451+416) #0.7675427 accuracy of my model

#fully grown tree gives very complex model
final_tree<-rpart(Churn~.,data = train,method = "class",cp=0,minsplit=0) #full grown tree may or maynot improve model
plot(final_tree)
plotcp(final_tree)


#final pred
final_pred<-predict(final_tree,newdata = test,type = "class")
table(final_pred,test$Churn)

#accuracy confusion matrix
(1772+1298)/(1772+1298+246+384) #our model has improved 0.8297297 if cp=0


library(caret)
library(e1071)
#advanced implementation to get best cp(complexity paramter)
numfolds<-trainControl(method = "cv",number = 10)#cross validation number 10 is kfold
cpgrid<-expand.grid(.cp=seq(0.0001,0.5,0.01))

#cross validation for getting best cp
train(Churn~.,data=train,method="rpart",trControl=numfolds,tuneGrid=cpgrid)
plotcp(final_tree)  #our assumption we can take a point from between because if we choose from start underfiiting issues at end over fiiting and complex
#expand.grid will give us best ,cp 1e-04 ie0.0001
#final predictiom

cp_tree<-rpart(Churn~.,data = train,method = "class",cp=0.0014)
plotcp(cp_tree)
prp(cp_tree)

#test
cp_test<-predict(cp_tree,newdata = test,type = "class")
#confusion matrix
final_matrix<-table(cp_test,test$Churn)
final_matrix

#accuracy
sum(diag(final_matrix))/sum(final_matrix)
#accuracy of our model is 0.8010811