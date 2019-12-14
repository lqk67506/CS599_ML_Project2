# Load required package
library(penaltyLearning)
library(future)
library(future.apply)
library(directlabels)
# Read in target and feature of abs
abs_features <- read.csv("abs_features.csv")
abs_target <- read.csv("abs_targets.csv")
abs_features_mat = data.matrix(abs_features)
abs_targets_mat = data.matrix(abs_target)
# Generate abs predictor with IntervalRegressionCV function in penaltyLearining package
abs_predictor <- penaltyLearning::IntervalRegressionCV(abs_features_mat, abs_targets_mat)
# Read in targe and feature of sin
sin_features <- read.csv("sin_features.csv")
sin_target <- read.csv("sin_targets.csv")
sin_features_mat = data.matrix(sin_features)
sin_targets_mat = data.matrix(sin_target)
# Generate sin  with IntervalRegressionCV function in penaltyLearining package
sin_predictor <- penaltyLearning::IntervalRegressionCV(sin_features_mat, sin_targets_mat)
# Read in targe and feature of linear
linear_features <- read.csv("linear_features.csv")
linear_target <- read.csv("linear_targets.csv")
linear_features_mat = data.matrix(linear_features)
linear_targets_mat = data.matrix(linear_target)
# Generate sin  with IntervalRegressionCV function in penaltyLearining package
linear_predictor <- penaltyLearning::IntervalRegressionCV(linear_features_mat, linear_targets_mat)

# Change parameters to dataframe
linear_data <- data.frame(linear_predictor$param.mat)
# Rename colomn
colnames(linear_data) <- c( "parameters")
# Write in a csv file
write.csv(linear_data, file = "linear_data.csv")
# Change parameters to dataframe
sin_data <- data.frame(sin_predictor$param.mat)
# Rename colomn
colnames(sin_data) <- c( "parameters")
# Change parameters to dataframe
write.csv(sin_data, file = "sin_data.csv")
# Change parameters to dataframe
abs_data <- data.frame(abs_predictor$param.mat)
# Rename colomn
colnames(abs_data) <- c( "parameters")
# Change parameters to dataframe
write.csv(abs_data, file = "abs_data.csv")
