#code for assignment 3 - unsupervised learning - Adult Income Dataset

#loading packages
library(tidyverse)
library(caret)
library(ClusterR)
library(fastICA)
library(cluster)
library(mclust)
library(factoextra)
library(psych)
library(GPArotation)

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

#kmeans with ica
fviz_nbclust(income_ica$S, kmeans, method = 'wss', k.max = 40) +
  labs(subtitle = 'Income Data - ICA')


#em ica model
income_ica_em <- Mclust(income_ica$S, G = 1:50, modelNames = 'VII')

#plotting to determine number of clusters
plot(income_ica_em, what = 'BIC', main = TRUE, col = 'blue')
title(main = 'BIC and Clusters for Census Income Data - ICA')
#optimal clusters using VII = 


#RCA  ----------------------------------------------------------

# Functions
maxfactor <- function(x) {
  return(which(x == max(x)))
}

vecnorm <- function (x) {
  p <- 2
  if (!is.numeric(x) && !is.complex(x))
    stop("mode of x must be either numeric or complex")
  if (is.numeric(x))
    x <- abs(x)
  else x <- Mod(x)
  return(.Fortran("d2norm", as.integer(length(x)), as.double(x),
                  as.integer(1), double(1), PACKAGE = "mclust")[[4]])
}

rca <- function(data, p = 2) {
  n <- ncol(data)
  u <- rnorm(n)
  u <- u/vecnorm(u)
  v <- rnorm(n)
  v <- v/vecnorm(v)
  Q <- cbind(u, v - sum(u * v) * u)
  dimnames(Q) <- NULL
  Data <- as.matrix(data) %*% Q
  Data
}

set.seed(20)

#building model
income_rca_model <- rca(income_data)

#plotting first two components
plot(income_rca_model, main = 'Income Data First Two Random Projections')

#RCA kmeans
#calculating within sum of squares for different number of clusters
fviz_nbclust(income_rca_model, kmeans, method = 'wss', k.max = 40) +
  labs(subtitle = 'Income Data - Randomized Projections')
#optimal number of clusters = 17


#em rca model
income_rca_em <- Mclust(income_rca_model, G = 1:40, modelNames = 'VII')

#plotting to determine number of clusters
plot(income_rca_em, what = 'BIC', main = TRUE, col = 'blue')
title(main = 'BIC and Clusters for Income Data - Randomized Projections')
#optimal number of clusters = 32




#EFA  ----------------------------------------------------------

income_efa_data <- read.csv('adult.csv', header = T)

#replacing question marks with NA
income_efa_data[income_efa_data == "?"] <- NA

#removing education number because it's basically a duplicate of education and removing native country because it has too many levels
income_efa_data <- income_efa_data[,-4]
income_efa_data <- income_efa_data[,-13]

#getting rid of missing data
income_efa_data <- income_efa_data[complete.cases(income_efa_data),]

#cutting down the data 
train_index <- createDataPartition(income_efa_data$income, p = .5, list = FALSE, times = 1)
income_efa_data <- income_efa_data[train_index,]

#making the y variable a factor
income_efa_data$income <- as.character(income_efa_data$income)  #converting factor to characters
income_efa_data$income <- ifelse(income_efa_data$income == ">50K", 1, 0) 
income_efa_y <- data.frame(income = income_efa_data$income)

#creating dummy variables
dummies <- dummyVars(income ~ ., data = income_efa_data)
income_efa_data <- predict(dummies, newdata = income_efa_data)
income_efa_data <- data.frame(income_efa_data)
income_efa_data <- cbind(income_efa_data, income_efa_y)


#scaling the data
income_efa_data <- scale(income_efa_data)

#getting rid of NaNs
income_efa_data <- income_efa_data[,-c(2,5,20)]

ev <- eigen(cor(income_efa_data))
ap <- parallel(subject=nrow(income_efa_data),
               var=ncol(income_efa_data),
               rep=100, cent=.05)
nS <- nScree(x=ev$values, aparallel=ap$eigen$qevpea)

plotnScree(nS, main='EFA Income Data')

#income_efa_result <- factanal(income_efa_data, 3, scores="regression")

income_efa_result <- fa(r = cor(income_efa_data), nfactors = 7, rotate = 'varimax', SMC = F, fm = 'minres')

#EFA Kmeans
#calculating within sum of squares for different number of clusters
fviz_nbclust(income_efa_result$weights, kmeans, method = 'wss', k.max = 40) +
  labs(subtitle = 'Income Data - EFA')
#optimal number of clusters = 10


#em efa model
income_efa_em <- Mclust(income_efa_result$weights, G = 1:10, modelNames = 'VII')

#plotting to determine number of clusters
plot(income_efa_em, what = 'BIC', main = TRUE, col = 'blue')
title(main = 'BIC and Clusters for Income Data - EFA')
#optimal number of clusters = 7

#em efa model - final
income_efa_em_final <- Mclust(income_efa_result$weights, G = 7, modelNames = 'VII')
















