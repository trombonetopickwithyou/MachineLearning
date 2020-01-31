close all; clear; clc;

load carbig.mat

%calculate the mean of the dataset and replace NaN's with the mean
horsepower_mean = nanmean(Horsepower);
Horsepower(isnan(Horsepower)) = horsepower_mean;
horsepower_std_deviation = std(Horsepower);

weight_mean = mean(Weight);
weight_std_deviation = std(Weight);
%% Closed form
%	(training data)
%x = input x values (matrix N x D+1)
%t = input y values (matrix N x 1)
%n = num datapoints (scalar)

%X = design matrix (2 columns, n rows)
%W = ( (X')*(X) )^(-1) * (X') * t;
%solution: y = W_0 * x + W_1

%X is the design matrix
A = ones(length(Weight),1);
X = [Weight A];

%t is a vector of "y" values corresponding to the values in X
t = Horsepower;

%Solution (2 x 1 matrix) [W_0  W_1]
W = ( (X')*(X) )^(-1) * (X') * t;

%scatter(Weight, Horsepower,'x');
%hold on;

%plot the solution
x = min(Weight):50:max(Weight);
y = W(1)*x + W(2);
%plot(x,y);

%clear the answer so it can be solved again using gradient descent
clearvars W y t X x

%% Gradient descent

%normalize data
Horsepower(:) = (Horsepower(:) - horsepower_mean)./(horsepower_std_deviation);
Weight(:) = (Weight(:) - weight_mean)./(weight_std_deviation);

t = Horsepower;
W = [0.05 0.05];		%guess initial W
rho = 0.0001;			%learning rate
A = ones(length(Weight),1);
X = [Weight A];

x = min(Weight):0.01:max(Weight);

hold on;
for index = 1:1000
	%gradient descent algorithm
	gradient = 2*(W)*((X')*X) - 2*(t')*X;
	
	W = W - rho*gradient
	
% 	y = W(1).*x + W(2);
% 	scatter(Weight, Horsepower, 'x');
% 	plot(x,y);
	%clf;
end

y = W(1).*x + W(2);
scatter(Weight, Horsepower, 'x');
plot(x,y);
