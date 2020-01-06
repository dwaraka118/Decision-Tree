# What are Decision Trees?
# Decision trees are versatile Machine Learning algorithm that can perform both classification and regression tasks. 
# They are very powerful algorithms, capable of fitting complex datasets. 
# Besides, decision trees are fundamental components of random forests,
# which are among the most potent Machine Learning algorithms available today.
# Training and Visualizing a decision trees
# To build your first decision trees, we will proceed as follow:
#   
# Step 1: Import the data
# Step 2: Clean the dataset
# Step 3: Create train/test set
# Step 4: Build the model
# Step 5: Make prediction
# Step 6: Measure performance
# Step 7: Tune the hyper-parameters


# Step 1) Import the data
set.seed(678)
path <- 'https://raw.githubusercontent.com/guru99-edu/R-Programming/master/titanic_data.csv'
titanic <-read.csv(path)
head(titanic)
tail(titanic)
# From the head and tail output, you can notice the data is not shuffled. 
# This is a big issue! When you will split your data between a train set and test set, you will select only the passenger from class 1 and 2 (No passenger from class 3 are in the top 80 percent of the observations),
# which means the algorithm will never see the features of passenger of class 3. 
# This mistake will lead to poor prediction.

# To overcome this issue, you can use the function sample().

shuffle<- sample(1:nrow(titanic))
head(shuffle)

# Code Explanation

# sample(1:nrow(titanic)):
# Generate a random list of index from 1 to 1309 (i.e. the maximum number of rows).
titanic <- titanic[shuffle, ]
head(titanic)

# Step 2) Clean the dataset
# The structure of the data shows some variables have NA's. Data clean up to be done as follows
# 
# Drop variables home.dest,cabin, name, X and ticket
# Create factor variables for pclass and survived
# Drop the NA

library(dplyr)
# Drop variables
clean_titanic <- titanic %>%
  select(-c("home.dest", "cabin", "name", "x", "ticket")) %>%
  #Convert to factor level
  mutate(pclass = factor(pclass, levels = c(1, 2, 3), labels = c('Upper', 'Middle', 'Lower')),
         survived = factor(survived, levels = c(0, 1), labels = c('No', 'Yes'))) %>%
  na.omit()
glimpse(clean_titanic)
# Code Explanation
# 
# select(-c("home.dest", "cabin", "name", "x", "ticket")): Drop unnecessary variables
# pclass = factor(pclass, levels = c(1,2,3), labels= c('Upper', 'Middle', 'Lower')): Add label to the variable pclass. 1 becomes Upper, 2 becomes MIddle and 3 becomes lower
# factor(survived, levels = c(0,1), labels = c('No', 'Yes')): Add label to the variable survived. 1 Becomes No and 2 becomes Yes
# na.omit(): Remove the NA observations

#Step 3) Create train/test set
# Before you train your model, you need to perform two steps:
#   Create a train and test set: You train the model on the train set and test the prediction on the test set (i.e. unseen data)
# Install rpart.plot from the console
# The common practice is to split the data 80/20, 80 percent of the data serves to train the model, and 20 percent to make predictions. 
# You need to create two separate data frames.
# You don't want to touch the test set until you finish building your model. 
# You can create a function name create_train_test() that takes three arguments.
# create_train_test(df, size = 0.8, train = TRUE)

# arguments:
#   -df: Dataset used to train the model.
# -size: Size of the split. By default, 0.8. Numerical value
# -train: If set to `TRUE`, the function creates the train set, otherwise the test set.
#         Default value sets to `TRUE`. Boolean value.
#         You need to add a Boolean parameter because R does not allow to return two data frames simultaneously.

create_train_test<- function(data, size = 0.8, train = TRUE) {
  n_row = nrow(data)
  total_row = size * n_row
  train_sample <- 1: total_row
  if (train == TRUE) {
    return (data[train_sample, ])
  } else {
    return (data[-train_sample, ])
  }
}

# Code Explanation
# 
# function(data, size=0.8, train = TRUE): Add the arguments in the function
# n_row = nrow(data): Count number of rows in the dataset
# total_row = size*n_row: Return the nth row to construct the train set
# train_sample <- 1:total_row: Select the first row to the nth rows
# if (train ==TRUE){ } else { }: If condition sets to true, return the train set, else the test set.

data_train <- create_train_test(clean_titanic, 0.8, train = TRUE)
data_test <- create_train_test(clean_titanic, 0.8, train = FALSE)
dim(data_train)

# You use the function prop.table() combined with table() to verify if the randomization process is correct.
prop.table(table(data_train$survived))
prop.table(table(data_test$survived))

# Install rpart.plot
# rpart.plot is not available from conda libraries. You can install it from the console:
#   
install.packages("rpart.plot")	


# Step 4) Build the model
# You are ready to build the model. The syntax for Rpart() function is:
#   
#   rpart(formula, data=, method='')
# arguments:			
#   - formula: The function to predict
#   - data: Specifies the data frame
# method: 			
#   - "class" for a classification tree 			
#   - "anova" for a regression tree	

# You use the class method because you predict a class.

library(rpart)
library(rpart.plot)
fit <- rpart(survived~., data = data_train, method = 'class')
rpart.plot(fit, extra = 106)

# Code Explanation
# 
# rpart(): Function to fit the model. The arguments are:
#   survived ~.: Formula of the Decision Trees
# data = data_train: Dataset
# method = 'class': Fit a binary model
# rpart.plot(fit, extra= 106): Plot the tree. 
# The extra features are set to 101 to display the probability of the 2nd class (useful for binary responses).
# You can refer to the vignette for more information about the other choices.
           
# You start at the root node (depth 0 over 3, the top of the graph):
#   
#   At the top, it is the overall probability of survival. It shows the proportion of passenger that survived the crash. 41 percent of passenger survived.
# This node asks whether the gender of the passenger is male. If yes, then you go down to the root's left child node (depth 2). 63 percent are males with a survival probability of 21 percent.
# In the second node, you ask if the male passenger is above 3.5 years old. If yes, then the chance of survival is 19 percent.
# You keep on going like that to understand what features impact the likelihood of survival.
# Note that, one of the many qualities of Decision Trees is that they require very little data preparation. In particular, they don't require feature scaling or centering.
# 
# By default, rpart() function uses the Gini impurity measure to split the note. The higher the Gini coefficient, the more different instances within the node.

# Step 5) Make a prediction
# You can predict your test dataset. To make a prediction, you can use the predict() function. 
# The basic syntax of predict for decision trees is:
# predict(fitted_model, df, type = 'class')
# arguments:
#   - fitted_model: This is the object stored after model estimation. 
# - df: Data frame used to make the prediction
# - type: Type of prediction			
# - 'class': for classification			
# - 'prob': to compute the probability of each class			
# - 'vector': Predict the mean response at the node level	

# You want to predict which passengers are more likely to survive after the collision from the test set. 
# It means, you will know among those 209 passengers, which one will survive or not.

predict_unseen <-predict(fit, data_test, type = 'class')

# Code Explanation
# 
# predict(fit, data_test, type = 'class'): Predict the class (0/1) of the test set
# Testing the passenger who didn't make it and those who did.
table_mat <- table(data_test$survived, predict_unseen)
table_mat
# Code Explanation:
# table(data_test$survived, predict_unseen): Create a table to count how many passengers are classified as survivors and passed away compare to the correct classification
# The model correctly predicted 106 dead passengers but classified 15 survivors as dead.
# By analogy, the model misclassified 30 passengers as survivors while they turned out to be dead.

# Step 6) Measure performance
# You can compute an accuracy measure for classification task with the confusion matrix:
#   
#   The confusion matrix is a better choice to evaluate the classification performance.
# The general idea is to count the number of times True instances are classified are False
accuracy_Test <- sum(diag(table_mat)) / sum(table_mat)
# Code Explanation
# 
# sum(diag(table_mat)): Sum of the diagonal
# sum(table_mat): Sum of the matrix.

# You can print the accuracy of the test set:
print(paste('Accuracy for test', accuracy_Test))


# Step 7) Tune the hyper-parameters
# Decision tree has various parameters that control aspects of the fit. 
# In rpart library, you can control the parameters using the rpart.control() function.
# rpart.control(minsplit = 20, minbucket = round(minsplit/3), maxdepth = 30)
# Arguments:
#   -minsplit: Set the minimum number of observations in the node before the algorithm perform a split
# -minbucket:  Set the minimum number of observations in the final note i.e. the leaf
# -maxdepth: Set the maximum depth of any node of the final tree. The root node is treated a depth 0

# We will proceed as follow:
#   
#   Construct function to return accuracy
# Tune the maximum depth
# Tune the minimum number of sample a node must have before it can split
# Tune the minimum number of sample a leaf node must have
# You can write a function to display the accuracy. You simply wrap the code you used before:
#   
#   predict: predict_unseen <- predict(fit, data_test, type = 'class')
# Produce table: table_mat <- table(data_test$survived, predict_unseen)
# Compute accuracy: accuracy_Test <- sum(diag(table_mat))/sum(table_mat)

accuracy_tune <- function(fit) {
  predict_unseen <- predict(fit, data_test, type = 'class')
  table_mat <- table(data_test$survived, predict_unseen)
  accuracy_Test <- sum(diag(table_mat)) / sum(table_mat)
  accuracy_Test
}

# You can try to tune the parameters and see if you can improve the model over the default value.
# As a reminder, you need to get an accuracy higher than 0.78

control <- rpart.control(minsplit = 4,
                         minbucket = round(5 / 3),
                         maxdepth = 3,
                         cp = 0)
tune_fit <- rpart(survived~., data = data_train, method = 'class', control = control)
accuracy_tune(tune_fit)

# With the following parameter:
#   
#   minsplit = 4
# minbucket= round(5/3)
# maxdepth = 3cp=0	
