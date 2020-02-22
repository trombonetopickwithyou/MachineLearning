%Joshua Williams
%ECE 5332
%HW03: Test Error with Regularization Parameter

close all; clear; clc;

seed = 0;
%% Generate Dataset
rng(seed);
L = 100;
N = 25;
X = rand(N,1);