#code for assignment 3 - unsupervised learning - Pima Indians Dataset

#loading packages
library(tidyverse)
library(caret)
library(fastICA)
library(cluster)
library(mclust)
library(ggthemes)
library(neuralnet)
library(nFactors)

#loading pima indians data
pima <- read.csv('pima-indians-diabetes.csv', header = F)

#creating column names
colnames(pima) <- c("Pregnancies",
                        "Glucose",
                        "DiastolicBP",
                        "TSFT",
                        "Serum2Hr",
                        "BMI",
                        "DPF",
                        "Age",
                        "Diagnosis")

#removing Observations with Missing Data
pima <- pima[pima[2] != 0 &
        pima[3] != 0 &
        pima[4] != 0 &
        pima[5] != 0 &
        pima[6] != 0 &
        pima[7] != 0 &
        pima[8] != 0, ]

#scaling data
pima_scaled <- scale(pima)



#kmeans clustering  --------------------------------------------

#setting the seed
set.seed(191)

#calculating within sum of squares for different number of clusters
fviz_nbclust(pima_scaled, kmeans, method = 'wss') +
labs(subtitle = 'Pima Indians Data')

#final kmeans clustering with 9 clusters
pima_kmeans <- kmeans(pima_scaled, 9, algorithm = 'Lloyd', nstart = 20)

#attaching clusters to data
pima_kclusters <- cbind(pima, pima_kmeans$cluster)

#creating a table to compare to actual labels
table(pima_kclusters[,9:10])


#expectation maximization --------------------------------------

#em model
pima_em <- Mclust(pima_scaled, G = 1:20, modelNames = 'VII')

#plotting to determine number of clusters
plot(pima_em, what = 'BIC', main = TRUE, col = 'blue')
title(main = 'BIC and Clusters for Pima Indian Data')
#optimal clusters using VII = 9.

#final model
pima_em_final <- Mclust(pima_scaled, G = 9, modelNames = 'VII')

#adding clusters to pima data to analyze
pima_emclusters <- cbind(pima, pima_em_final$classification)

#creating a table to compare to actuals
table(pima_emclusters[,9:10])


#PCA  ----------------------------------------------------------
#pca model
pima_pca_model <- prcomp(pima_scaled)

#scree plot
fviz_eig(pima_pca_model, addlabels = TRUE, ggtheme = theme_hc(), linecolor = "red", main = "Scree plot of Pima Indians PCA model")
#6 components = 90% of variance

#subsetting the first 6 components for clustering
pima_pca <- pima_pca_model$x[,1:6]

#kmeans with pca
#calculating within sum of squares for different number of clusters
fviz_nbclust(pima_pca, kmeans, method = 'wss', k.max = 40) +
  labs(subtitle = 'Pima Indians Data - PCA')
#optimal number of clusters = 30

#final pca kmenans model
pima_pca_final_kmeans <-  kmeans(pima_scaled, 30, algorithm = 'Lloyd', nstart = 20)

#em pca model
pima_pca_em <- Mclust(pima_pca, G = 1:20, modelNames = 'VII')

#plotting to determine number of clusters
plot(pima_pca_em, what = 'BIC', main = TRUE, col = 'blue')
title(main = 'BIC and Clusters for Pima Indian Data - PCA')
#optimal number of clusters = 6

#final em pca model
pima_pca_em_final <- Mclust(pima_pca, G = 6, modelNames = 'VII')

#ICA  ----------------------------------------------------------

#building model
pima_ica <- fastICA(pima_scaled, 4)

#plotting
pairs(pima_ica$S, col=rainbow(2)[pima[,1]], main = 'Pima ICA Variables')


#kmeans with ica
#calculating within sum of squares for different number of clusters
fviz_nbclust(pima_ica$S, kmeans, method = 'wss', k.max = 40) +
  labs(subtitle = 'Pima Indians Data - ICA')
#optimal number of clusters = 28

#final ica kmenans model
pima_ica_final_kmeans <-  kmeans(pima_ica$S, 28, algorithm = 'Lloyd', nstart = 20)

#em ica model
pima_ica_em <- Mclust(pima_ica$S, G = 1:40, modelNames = 'VII')

#plotting to determine number of clusters
plot(pima_pca_em, what = 'BIC', main = TRUE, col = 'blue')
title(main = 'BIC and Clusters for Pima Indian Data - ICA')
#optimal number of clusters = 6

#final em ica model
pima_ica_em_final <- Mclust(pima_ica$S, G = 6, modelNames = 'VII')





#Randomized Projections  ----------------------------------------------------------

set.seed(20)
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

#building model
pima_rca_model <- rca(pima_scaled[,1:8])

#plotting first two components
plot(pima_rca_model, main = 'Pima Data First Two Random Projections')

#RCA kmeans
#calculating within sum of squares for different number of clusters
fviz_nbclust(pima_rca_model, kmeans, method = 'wss', k.max = 40) +
  labs(subtitle = 'Pima Indians Data - Randomized Projections')
#optimal number of clusters = 15

#final rca kmenans model
pima_rca_final_kmeans <-  kmeans(pima_rca_model, 15, algorithm = 'Lloyd', nstart = 20)

#em rca model
pima_rca_em <- Mclust(pima_rca_model, G = 1:40, modelNames = 'VII')

#plotting to determine number of clusters
plot(pima_rca_em, what = 'BIC', main = TRUE, col = 'blue')
title(main = 'BIC and Clusters for Pima Indian Data - Randomized Projections')
#optimal number of clusters = 2

#final em rca model
pima_rca_em_final <- Mclust(pima_rca_model, G = 2, modelNames = 'VII')


#adding clusters to pima data to analyze
pima_rca_emclusters <- cbind(pima, pima_rca_em_final$classification)

#creating a table to compare to actuals
table(pima_rca_emclusters[,9:10])


#EFA  ----------------------------------------------------------


#building model
ev <- eigen(cor(pima_scaled[,1:8]))
ap <- parallel(subject=nrow(pima_scaled[,1:8]),
               var=ncol(pima_scaled[,1:8]),
               rep=100, cent=.05)
nS <- nScree(x=ev$values, aparallel=ap$eigen$qevpea)

plotnScree(nS, main='EFA Pima Data')

pima_efa_result <- factanal(pima_scaled[,1:8], 3, scores="regression")


#EFA Kmeans
#calculating within sum of squares for different number of clusters
fviz_nbclust(pima_efa_result$scores, kmeans, method = 'wss', k.max = 40) +
  labs(subtitle = 'Pima Indians Data - EFA')
#optimal number of clusters = 20

#final efa kmeans model
pima_efa_final_kmeans <-  kmeans(pima_efa_result$scores, 20, algorithm = 'Lloyd', nstart = 20)

#em efa model
pima_efa_em <- Mclust(pima_efa_result$scores, G = 1:40, modelNames = 'VII')

#plotting to determine number of clusters
plot(pima_efa_em, what = 'BIC', main = TRUE, col = 'blue')
title(main = 'BIC and Clusters for Pima Indian Data - EFA')
#optimal number of clusters = 5

#final em efa model
pima_efa_em_final <- Mclust(pima_efa_result$scores, G = 5, modelNames = 'VII')


#adding clusters to pima data to analyze
pima_efa_emclusters <- cbind(pima, pima_efa_em_final$classification)

#creating a table to compare to actuals
table(pima_efa_emclusters[,9:10])

#Neural Network --------------------------------------------------

#PCA NN 
#attaching actual labels
pima_pca_nn_data <- data.frame(cbind(pima_pca, Diagnosis = as.factor(pima[,9])))
pima_pca_nn_data$Diagnosis <- as.factor(pima_pca_nn_data$Diagnosis)

#splitting pca data into train/test
trainIndex <- createDataPartition(pima_pca_nn_data$Diagnosis, p = .6, list = F, times = 1)
pima_pca_nn_train <- pima_pca_nn_data[trainIndex,]
pima_pca_nn_test <- pima_pca_nn_data[-trainIndex,]


#building model
ctrl <- trainControl(method = 'repeatedcv', repeats = 10)
pima_pca_nn_model <- train(Diagnosis ~ ., data = pima_pca_nn_train, method = 'nnet', trControl = ctrl)

#plotting model complexity curve
plot(pima_pca_nn_model, main = 'Principle Component Analysis NN')


#making predictions on test data
pima_pca_nn_pred <- predict(pima_pca_nn_model, newdata = pima_pca_nn_test[,1:6], type = 'raw')


#creating a confusion matrix
pima_pca_nn_cm <- confusionMatrix(pima_pca_nn_pred, pima_pca_nn_test$Diagnosis, mode = 'prec_recall')
pima_pca_nn_cm$table


#ICA NN

#attaching actual labels
pima_ica_nn_data <- data.frame(cbind(pima_ica$S, Diagnosis = as.factor(pima[,9])))
pima_ica_nn_data$Diagnosis <- as.factor(pima_ica_nn_data$Diagnosis)

#splitting pca data into train/test
trainIndex <- createDataPartition(pima_ica_nn_data$Diagnosis, p = .8, list = F, times = 1)
pima_ica_nn_train <- pima_ica_nn_data[trainIndex,]
pima_ica_nn_test <- pima_ica_nn_data[-trainIndex,]


#building model
ctrl <- trainControl(method = 'repeatedcv', repeats = 5)
pima_ica_nn_model <- train(Diagnosis ~ ., data = pima_ica_nn_train, method = 'nnet', trControl = ctrl)


#plotting model complexity curve
plot(pima_ica_nn_model, main = 'Independent Component Analysis NN')

#making predictions on test data
pima_ica_nn_pred <- predict(pima_ica_nn_model, newdata = pima_ica_nn_test, type = 'raw')


#creating a confusion matrix
pima_ica_nn_cm <- confusionMatrix(pima_ica_nn_pred, pima_ica_nn_test$Diagnosis, mode = 'prec_recall')
pima_ica_nn_cm$table



#RP NN

#attaching actual labels
pima_rca_nn_data <- data.frame(cbind(pima_rca_model, Diagnosis = as.factor(pima[,9])))
pima_rca_nn_data$Diagnosis <- as.factor(pima_rca_nn_data$Diagnosis)

#splitting pca data into train/test
trainIndex <- createDataPartition(pima_rca_nn_data$Diagnosis, p = .8, list = F, times = 1)
pima_rca_nn_train <- pima_rca_nn_data[trainIndex,]
pima_rca_nn_test <- pima_rca_nn_data[-trainIndex,]


#building model
ctrl <- trainControl(method = 'repeatedcv', repeats = 5)
pima_rca_nn_model <- train(Diagnosis ~ ., data = pima_rca_nn_train, method = 'nnet', trControl = ctrl)


#plotting model complexity curve
plot(pima_rca_nn_model, main = 'Randomized Projections NN')

#making predictions on test data
pima_rca_nn_pred <- predict(pima_rca_nn_model, newdata = pima_rca_nn_test, type = 'raw')


#creating a confusion matrix
pima_rca_nn_cm <- confusionMatrix(pima_rca_nn_pred, pima_rca_nn_test$Diagnosis, mode = 'prec_recall')
pima_rca_nn_cm$table


#EFA NN

#attaching actual labels
pima_efa_nn_data <- data.frame(cbind(pima_efa_result$scores, Diagnosis = as.factor(pima[,9])))
pima_efa_nn_data$Diagnosis <- as.factor(pima_efa_nn_data$Diagnosis)

#splitting pca data into train/test
trainIndex <- createDataPartition(pima_efa_nn_data$Diagnosis, p = .8, list = F, times = 1)
pima_efa_nn_train <- pima_efa_nn_data[trainIndex,]
pima_efa_nn_test <- pima_efa_nn_data[-trainIndex,]


#building model
ctrl <- trainControl(method = 'repeatedcv', repeats = 5)
pima_efa_nn_model <- train(Diagnosis ~ ., data = pima_efa_nn_train, method = 'nnet', trControl = ctrl)


#plotting model complexity curve
plot(pima_efa_nn_model, main = 'Exploratory Factor Analysis NN')

#making predictions on test data
pima_efa_nn_pred <- predict(pima_efa_nn_model, newdata = pima_efa_nn_test, type = 'raw')


#creating a confusion matrix
pima_efa_nn_cm <- confusionMatrix(pima_efa_nn_pred, pima_efa_nn_test$Diagnosis, mode = 'prec_recall')
pima_efa_nn_cm$table






#Clustered NN  ----------------------------------------------
#creating dataframe of clusters
clustered_nn_data <- data.frame(pca_kmeans = pima_pca_final_kmeans$cluster,
                                pca_em = pima_em_final$classification,
                                ica_kmeans = pima_ica_final_kmeans$cluster,
                                ica_em = pima_ica_em_final$classification,
                                rca_kmeans = pima_rca_final_kmeans$cluster,
                                rca_em = pima_rca_em_final$classification,
                                efa_kmeans = pima_efa_final_kmeans$cluster,
                                efa_em = pima_efa_em_final$classification,
                                Diagnosis = as.factor(pima[,9]))


#splitting pca data into train/test
trainIndex <- createDataPartition(clustered_nn_data$Diagnosis, p = .8, list = F, times = 1)
clustered_nn_train <- clustered_nn_data[trainIndex,]
clustered_nn_test <- clustered_nn_data[-trainIndex,]


#building model
ctrl <- trainControl(method = 'repeatedcv', repeats = 5)
clustered_nn_model <- train(Diagnosis ~ ., data = clustered_nn_train, method = 'nnet', trControl = ctrl)


#plotting model complexity curve
plot(clustered_nn_model, main = 'Clustered NN')

#making predictions on test data
clustered_nn_pred <- predict(clustered_nn_model, newdata = clustered_nn_test, type = 'raw')


#creating a confusion matrix
clustered_nn_cm <- confusionMatrix(clustered_nn_pred, clustered_nn_test$Diagnosis, mode = 'prec_recall')
clustered_nn_cm$table
