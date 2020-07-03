{
  graphics.off()
  rm(list=ls())
  setwd("C:\\Users\\victo\\source\\repos\\Silly_Python_Stuff\\knapsack")
}

data1 <- read.csv(file="test_data_dense_knapsac.csv")
data2 <- read.csv(file="test_data_sparse_knapsac.csv")

boxplot(data1[, c(1, 2, 3, 4)])
boxplot(data1[, -c(1, 2, 3, 4)])

boxplot(data2[, c(1, 2, 3, 4)])
boxplot(data2[, -c(1, 2, 3, 4)])
