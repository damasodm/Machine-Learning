library(h2o)
library(data.table)
library(dplyr)
library(caret)
library(dplyr)
library(klaR)
library(pROC)
library(ROCR)
library(e1071)
library(ggplot2)
library(corrplot)

localH2O = h2o.init(nthreads=-1)

# Leemos los datos
setwd("/Users/damasodiaz/home/usuario/")
credit <- readRDS('credit-training.RDa')
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

credit$Rating <- as.character(credit$Rating)
credit[Rating=='A',Rating:=5]
credit[Rating=='B',Rating:=4]
credit[Rating=='C',Rating:=3]
credit[Rating=='D',Rating:=2]


credit$SeriousDlqin2yrs <- as.character(credit$SeriousDlqin2yrs)
credit[SeriousDlqin2yrs=='S',SeriousDlqin2yrs:=1]
credit[SeriousDlqin2yrs=='N_S',SeriousDlqin2yrs:=0]

credit$SeriousDlqin2yrs <- as.numeric(credit$SeriousDlqin2yrs)
credit$Rating <- as.numeric(credit$Rating)


sapply(credit, function(x)any(is.numeric(x)))
sapply(credit, function(x)any(is.na(x)))
ncol(credit)
credit[,Id:=NULL]
credit <- as.data.table(credit)
credit <- credit[,c(3,2,1,4,5,6,7,8,9,10,11,12,13)]
str(credit)


# Vemos independencia
res <- cor(credit)
round(res,2)

corrplot(res, type = "upper", order = "hclust", 
         tl.col = "black", tl.srt = 45)


credit <- credit[Isoutlier==0]
credit[,Isoutlier:=NULL]
credit[,N30to59DaysLate:=NULL]
credit <- as.data.frame(lapply(credit, normalize))

set.seed(101) 
# Seleccionamos el 67% de los datos
sample <- sample.int(n = nrow(credit), size = floor(.67*nrow(credit)), replace = F)
credit.train <- credit[sample, ]
credit.test  <- credit[-sample, ]

str(credit.train)

train.hex <- as.h2o(credit.train)
test.hex <- as.h2o(credit.test)


h2o.credit <- h2o.deeplearning(x = 2:ncol(credit), y = 1, training_frame = train.hex,
                               #hidden = c(10,20,30,20,10),
                               hidden = c(10,30,30,30,10), seed = 1234,
                               activation = "Tanh",
                               stopping_rounds = 20,
                               validation_frame = test.hex,
                               epochs = 50
)

plot(h2o.credit)

predictions <- as.vector(h2o.predict(h2o.credit, test.hex))
cor(predictions, credit.test$MonthlyIncome)
RMSE <- sqrt(mean((credit.test$MonthlyIncome - predictions)^2))
RMSE
par(mfrow=c(1,2))
plot(credit.test$MonthlyIncome,predictions,col='red',main='Real vs predicho NN 2',pch=18,cex=0.7)
abline(0,1,lwd=2)

# Lo comparamos con una regresión lineal
lm.model <- lm(MonthlyIncome ~ age + SeriousDlqin2yrs + DebtRatio
               + NumberOfOpenCreditLinesAndLoans + NumberRealEstateLoansOrLines + NumberOfDependents
               + RevolvingUtilizationOfUnsecuredLines + DefaulterGrade+Scoringtotal+Rating,
               data = credit.train)
pred.lm <- predict(lm.model, credit.test[, 1:11])
cor(pred.lm, credit.test$MonthlyIncome)
RMSE <- sqrt(mean((credit.test$MonthlyIncome - pred.lm)^2))
RMSE


plot(credit.test$MonthlyIncome,pred.lm,col='blue',main='Real vs predicho NN 2',pch=18,cex=0.7)
abline(0,1,lwd=2)

