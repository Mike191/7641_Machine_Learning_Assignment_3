#code for assignment 3 - unsupervised learning - Pima Indians Dataset

#loading packages
library(tidyverse)
library(caret)
library(ClusterR)
library(fastICA)
library(cluster)
library(mclust)

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
fviz_nbclust(pima_scaled, kmeans, nstart = 25, method = 'wss') +
labs(subtitle = 'Pima Indians Data')

#final kmeans clustering with 20 clusters
pima_kmeans <- kmeans(pima_clust, 20, algorithm = 'Lloyd', nstart = 50)

#attaching clusters to data
pima_kclusters <- cbind(pima, pima_clusters$cluster)

#creating a table to compare to actual labels
table(pima_kclusters[,9:10])


#expectation maximization --------------------------------------

pima_em <- Mclust(pima_scaled)

