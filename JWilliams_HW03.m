%Joshua Williams
%2/29/2020
%ECE 5332
%HW03: Regularization with Linear Regression and Gaussian basis functions

close all; clear; clc;

seed = 0;
L = 100;
N = 25;
lambda = 0.001;

while(log(lambda) < 2.2)     
    %% Generate dataset
    rng(seed)
    X = rand(1,N);		%uniform distribution U(0,1)
    
    epsilon = normrnd(0,0.3,[1 N]);
    t(:) = sin(2.*pi.*X(:)) + epsilon(:);
    
    t = t';
    X = X';
    
    clf
    hold all;
    plot(X, t, 'x', 'MarkerSize', 10, 'LineWidth', 2);
    
    %% Create model
    % take Nx1 inputs and return NxD basis function values
    design_matrix = design_rbf_matrix(X,X,N);
    
    % compute analytical solution
    w = (design_matrix'*design_matrix + lambda*eye(N+1))\design_matrix'*t;
    
    %% Plot solution
    x_plot = (0:0.01:1)';       % Nx1
    y = design_rbf_matrix(x_plot,X,N)*w;
    
    plot(x_plot, y, 'LineWidth', 2);
    title(['ln(\lambda)= ', num2str(log(lambda))]);
    pause(1);
    
    lambda = lambda.*2;
    
    clearvars -except lambda N L seed
    
end


%% functions
function phi_out = phi(x,mu)
    s = 0.1;
    phi_out = exp(-(x-mu).^2 / (2.*(s.^2)) );
end

function [design_matrix_out] = design_rbf_matrix(X,Xi,N)

    design_matrix_out = ones(length(X),1);
    for i=1:N
        design_matrix_out(:,i+1) = [phi(X,Xi(i))];
    end

end