close all; clear; clc;

load carbig.mat
%scatter(Weight, Horsepower,'x');






%% Closed form
%	(training data)
%x = input x values (matrix)
%t = input y values (matrix)
%n = num datapoints (scalar)

%y = predicted values

%X = design matrix (2 columns, n rows)
%
%W = ( (X')*(X) )^(-1) * (X') * t;
%solution: y = W_1 * X + W_0

%first, to fix NaN problems in dataset, calculate the mean of the dataset
%and replace NaN with the mean
horsepower_sum = nansum(Horsepower)
Horsepower(isnan(Horsepower)) = 



%% Gradient descent
%gradient = 2*(W')*(X')*X - 2*(t')*X;