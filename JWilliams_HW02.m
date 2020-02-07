%Joshua Williams
%2/7/2020
%ECE 5332-002: Machine Learning

close all; clear; clc;

seed = 3;
N_train = 10;
N_test = 100;


%% Generate training set
rng(seed);

X_train = rand(1,N_train);	%randomly sampled x-values from U(0,1)

epsilon = normrnd(0,0.3,[1 N_train]);				%"noise" to be added N(mu = 0, sigma = 0.3).
t_train(:) = sin(2.*pi.*X_train(:)) + epsilon(:);	%y-values corresponding to X_train samples. (sine function with "noise")

subplot(3,1,1);
scatter(X_train,t_train);	%(debug)
title('Scatter Plot of Training Data');
hold on;
%% Generate test set
X_test = rand(1,N_test);	%more randomly sampled x-values (for testing after model is made)

normal_dist_test = normrnd(0,0.3,[1 100]);	%N(0,0.3) again, but for more samples
t_test(:) = sin(2.*pi.*X_test(:)) + normal_dist_test(:);

subplot(3,1,2);
scatter(X_test,t_test);	%(debug)
title('Scatter Plot of Test Data');
hold on;
%% find line of best fit using linear regression with non-linear models (with training set)

for M=0:9 %degree of polynomial

    %map basis function matrix
	for N=1:length(X_train)             %N=rows (10 datapoints)
        for k=1:(M+1)                   %k=columns (M+1 columns)
            phi(N,k) = X_train(N).^(k-1);
        end
	end

    %compute closed-form analytical solution
    %W = inv(phi'*phi)*phi'*t_train'
	W = (phi'*phi)\phi'*t_train';
	W = flip(W);
	
	%plot solution (debug)
	x=0:0.001:1;
	y = polyval(W,x);
	subplot(3,1,1);
	plot(x,y);
	
	x=0:0.001:1;
	y = polyval(W,x);
	subplot(3,1,2);
	plot(x,y);	%(debug)
	
	%best fit line evaluated at X_train - actual datapoint evaluated at X_train
	J_train = sum( (polyval(W,X_train(:))-t_train(:)).^2 );
	E_rms_training(M+1) = sqrt(J_train/N_train);
	
	J_test = sum( (polyval(W,X_test(:)) - t_test(:)).^2 );
	E_rms_test(M+1) = sqrt(J_test/N_test);
    
end

x = 0:1:9;
subplot(3,1,3);
hold on;
plot(x, E_rms_training);
plot(x, E_rms_test);
ylim([0 1]);

legend('Training','Test');
title('Error (Model created with N train = 10)');


