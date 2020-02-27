%Joshua Williams
%ECE 5332
%HW03: Test Error with Regularization Parameter

close all; clear; clc;

seed = 0;
%% Generate Dataset

rng(seed);
L = 100;
N = 25;

X = rand(1,N);

epsilon = normrnd(0,0.3,[1 N]);
t(:) = sin(2.*pi.*X(:)) + epsilon(:);

lambda = 0;%:0.01:7;  %regularization parameter

%% Create Gaussian Basis Function
%phi_j(x) = exp( -(x-u_j).^2/(2s^2) )

mu = 0.5;
s=0.1;

for i=1:N
    phi(i,:)= [1,exp( (-(X(i)-mu).^2)./(2.*s.^2) )];
end

%compute closed-form analytical solution
W = (phi'*phi + lambda*eye(2))\phi'*t';

y=W(1)*X + W(2);

scatter(X,t);
hold on;
plot(X,y);
