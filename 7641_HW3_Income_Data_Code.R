#code for assignment 3 - unsupervised learning - Adult Income Dataset

#loading packages
library(tidyverse)
library(caret)
library(ClusterR)
library(fastICA)
library(cluster)
library(mclust)
library(factoextra)

#loading data
income_data <- read.csv('adult.csv', header = T)

#preprocessing  -----------------------------------
#replacing question marks with NA
income_data[income_data == "?"] <- NA

#removing education number because it's basically a duplicate of education and removing native country because it has too many levels
income_data <- income_data %>%
  select(-education, -native.country)

#making the y variable a factor
income_data$income <- as.character(income_data$income)  #converting factor to characters
income_data$income <- ifelse(income_data$income == ">50K", "greater50K", "less50K") 
income_y <- data.frame(income = as.factor(income_data$income))

#creating dummy variables
dummies <- dummyVars(income ~ ., data = income_data)
income_data <- predict(dummies, newdata = income_data)
income_data <- data.frame(income_data)

#getting rid of missing data and scaling
missing <- preProcess(income_data, "knnImpute")
income_data <- predict(missing, income_data)


#adding the y variable back to the training set so everything in one data set for ease of use
income_data <- cbind(income_data, income_y)
rm(income_y)  #no longer needed


#kmeans clustering  --------------------------------------------

#setting the seed
set.seed(191)

#calculating within sum of squares for different number of clusters
fviz_nbclust(income_data, kmeans, method = 'wss') +
  labs(subtitle = 'Census Income Data')

#calculating within sum of squares for different number of clusters
wss <- sapply(1:k_max, 
              function(k){kmeans(income_data[,1:50], k, nstart=50, algorithm = 'Lloyd')$tot.withinss})

#plotting kmeans - elbow plot
plot(1:k_max, wss, type = 'b', pch = 19, col= 'blue',
     ylab = "Total within cluster sum of squares",
     xlab = 'Number of clusters',
     main = "Eblow diagram for Adult Income Kmeans")

#final kmeans clustering with 20 clusters
pima_kmeans <- kmeans(pima_clust, 20, algorithm = 'Lloyd', nstart = 50)