close all; clear; clc;

load carbig.mat
scatter(Weight, Horsepower,'x');






%% Closed form
%y = predicted values
%solution = W_1 * n + W_0

%W = ( (X')*(X) )^(-1) * (X') * t;

%% Gradient descent
%gradient = 2*(W')*(X')*X - 2*(t')*X;