#code for assignment 3 - unsupervised learning - Pima Indians Dataset

#loading packages
library(tidyverse)
library(caret)
library(ClusterR)
library(fastICA)
library(cluster)
library(mclust)
library(ggthemes)
library(ica)

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

#em pca model
pima_pca_em <- Mclust(pima_pca, G = 1:20, modelNames = 'VII')

#plotting to determine number of clusters
plot(pima_pca_em, what = 'BIC', main = TRUE, col = 'blue')
title(main = 'BIC and Clusters for Pima Indian Data - PCA')
#optimal number of clusters = 6


#ICA  ----------------------------------------------------------

#building model
pima_ica <- fastICA(pima_scaled, 4)

#plotting
pairs(pima_ica$S, col=rainbow(2)[pima[,1]], main = 'Pima ICA Variables')



