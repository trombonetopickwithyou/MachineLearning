close all; clear; clc;

% Set up dataset
t = [0.15 0.3 0.5]';
X = [0.08 0.19 0.27]';

hold all;
plot(X, t, 'x', 'MarkerSize', 10, 'LineWidth', 2);


% take Nx1 inputs and return NxD basis function values
design_matrix = [ones(3,1), phi(X,X(1)) phi(X,X(2)) phi(X,X(3))];        % Dx1

w = (design_matrix'*design_matrix)\design_matrix'*t;


X_grid = (0:0.01:4)'; % Nx1
f_grid = [ones(length(X_grid),1), phi(X_grid, X(1)) phi(X_grid, X(2)) phi(X_grid, X(3))]*w;
plot(X_grid, f_grid, 'LineWidth', 2);

function out = phi(x,mu)
    
    s = 0.1;
    out = exp(-(x-mu).^2 / (2.*(s.^2)) );

end