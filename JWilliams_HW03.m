%Joshua Williams
%ECE 5332
%HW03: Test Error with Regularization Parameter

close all; clear; clc;

seed = 0;
M = 2; %--??

%% Generate Dataset
rng(seed);

L = 100;
N = 25;
X = rand(N,1);		%uniform distribution U(0,1)

epsilon = normrnd(0,0.3,[1 N]);
t(:) = sin(2.*pi.*X(:)) + epsilon(:);

lambda = 0:0.0001:7;