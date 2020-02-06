close all; clear; clc;

%% Generate training set
seed = 1;

rng(seed);
X_train = rand(1,10);

normal_dist_train = normrnd(0,0.3,[1 10]);
t_train(:) = sin(2.*pi.*X_train(:)) + normal_dist_train(:);

scatter(X_train,t_train);
hold on;
%% Generate test set
X_test = rand(1,100);

normal_dist_test = normrnd(0,0.3,[1 100]);
t_test(:) = sin(2.*pi.*X_test(:)) + normal_dist_test(:);

%scatter(X_test,t_test);

%% find line of best fit using linear regression with non-linear models (with training set)

for M=0:9 %degree of polynomial

    %map basis function matrix
    for N=1:length(X_train)             %N=rows (10 datapoints)
        for k=1:(M+1)                   %k=columns (M+1 columns)
            phi(N,k) = X_train(N).^(k-1);
        end
    end

    %compute analytical solution
    W = inv(phi'*phi)*phi'*t_train';

    %plot solution
    x=0:0.01:max(X_train);
    plot(x,polyval(flip(W),x));

end


