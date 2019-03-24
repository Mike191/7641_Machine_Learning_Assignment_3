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

#cutting down the data 
train_index <- createDataPartition(income_data$income, p = .5, list = FALSE, times = 1)
income_data <- income_data[train_index,]

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

#using wss method to determine optimal number of clusters
fviz_nbclust(income_data, kmeans, method = 'wss', nstart = 20) +
  labs(subtitle = 'Census Income Data')


#final kmeans clustering with 9 clusters
income_kmeans <- kmeans(income_data, 9, algorithm = 'Lloyd', nstart = 20)

#attaching clusters to data
income_kclusters <- cbind(income_y, income_kmeans$cluster)

#creating a table to compare to actual labels
table(income_kclusters)


#expectation maximization --------------------------------------

#em model
income_em <- Mclust(income_data, G = 1:50, modelNames = 'VII')

#plotting to determine number of clusters
plot(income_em, what = 'BIC', main = TRUE, col = 'blue')
title(main = 'BIC and Clusters for Census Income Data')
#optimal clusters using VII = 50

#em model
income_em_final <- Mclust(income_data, G = 50, modelNames = 'VII')

#attaching clusters to labels
income_emclusters <- cbind(income_y, income_em_final$classification)

#table
table(income_emclusters)


#PCA  ----------------------------------------------------------
#pca model
income_pca <- prcomp(income_data)

#scree plot
fviz_eig(income_pca, addlabels = TRUE, ggtheme = theme_hc(), ncp = 30, linecolor = "red", main = "Scree plot of Census Income PCA model")
#28 components = 80% of variance

#subsetting the first 28 pcas for clustering
income_pcas <- income_pca$x[,1:28]

#using wss method to determine optimal number of clusters for kmeans
fviz_nbclust(income_pcas, kmeans, method = 'wss', nstart = 20, k.max = 30) +
  labs(subtitle = 'Census Income Data - PCA')


#em pca model
income_pca_em <- Mclust(income_pcas, G = 1:50, modelNames = 'VII')

#plotting to determine number of clusters
plot(income_pca_em, what = 'BIC', main = TRUE, col = 'blue')
title(main = 'BIC and Clusters for Census Income Data - PCA')
#optimal clusters using VII = 50


#ICA  ----------------------------------------------------------

#building model
income_ica <- fastICA(income_data, 8)

#plotting
pairs(income_ica$S, col=rainbow(2)[income_y[,1]], main = 'Census Income ICA Variables')
