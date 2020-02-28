%Joshua Williams
%ECE 5332
%HW03: Test Error with Regularization Parameter

close all; clear; clc;

seed = 0;

%% Generate Dataset

rng(seed);

L = 100;
N = 25;

X = rand(1,N);		%uniform distribution U(0,1)

epsilon = normrnd(0,0.3,[1 N]);
t(:) = sin(2.*pi.*X(:)) + epsilon(:);

%% Create Gaussian Basis Function
%phi_j(x) = exp( -(x-u_j).^2/(2s^2) )

lambda = 0;

mu = 0.5;
s = 0.1;

phi = [ones(1,N); exp(-(X-mu).^2/(2*s.^2))]';

%compute closed-form analytical solution
W = (phi'*phi + lambda*eye(2))\phi'*t';

y = W'.*phi';

scatter(X,y);
