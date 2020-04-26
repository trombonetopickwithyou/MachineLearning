close all; clear; clc;

%% input parameters
rng(0);

train_input = [-1, -1; 1, 1; -1, 1; 1, -1];
train_t = [0; 0; 1; 1];

rho = 0.1;
epochs = 3000;

% network architecture
num_inputs = 2;
num_hidden_units = 2;
num_hidden_layers = 1;
num_outputs = 1;    %implement later with two classes?

%% ml stuff
% initialize neural network object
network = neuralnetwork(num_inputs, num_hidden_units, num_outputs);

% train

% figure
% hold on;
for i = 1:epochs
    network = nn_train(network, train_input, train_t, rho);

%     disp( mean(abs(train_t - network.output)) );
     
%      plot(i, -sum(train_t.*log10(network.output)), '.r' );
%       
%      set(gca,'yscale','log');
%      xlim([0 epochs]);
%      pause(0);
     
%     clc;
%     disp(epochs);
%     disp(network.input);
%     disp(train_t);
%     disp(network.output);
end

%% output results
fprintf('results:\n\n');
fprintf('epochs: %d\n\n', epochs);
fprintf('  input \tnetwork output\n');

for index=1:4
    fprintf('%3d\t', train_input(index,:));
    network = feedforward(train_input(index,:), network);
    disp(network.output);
end


%% functions

%train the nn using gradient descent
function network = nn_train(network, input, t, rho)
    network = feedforward(input, network);
    
    network.error_out = t - network.output;
    %network.error_out = -sum(t.*log10(network.output));
    
    d_predicted_output = network.error_out.*(network.output.*(1-network.output));
    
    error_h = d_predicted_output*(network.W{2}');
    d_hidden = error_h.*(network.h_out.*(1-network.h_out));
    
    % update equations
    network.W{2} = network.W{2} + ((network.h_out')*d_predicted_output).*rho;
    network.bias{2} = network.bias{2} + sum(d_predicted_output).*rho;
    
    network.W{1} = network.W{1} + ((network.input')*d_hidden).*rho;
    network.bias{1} = network.bias{1} + sum(d_hidden).*rho;    
end


%use pre-initialized neural network architecture, generates network output(s) (or "guess)
function network = feedforward(inputs, network)
    
    %column vector (length = num inputs)
    network.input = inputs;

    %column vector (length = num hidden units)
    network.h_out = sigmoid((network.input*network.W{1}) + network.bias{1});   %({1} is input-to-hidden layer)
    
    %column vector (length = num outputs)
    network.output = sigmoid((network.h_out*network.W{2}) + network.bias{2});

end


%initialize neural network object
function network_obj = neuralnetwork(num_inputs, num_hidden_units, num_outputs)
    
    % neural network architecture %
    % (currently only creates architecture described in hw05) %
    
    % input
    network_obj.input = [];
    
    % input-to-hidden layer
    network_obj.W{1} = rand(num_inputs, num_hidden_units); %W{1} is weights between input and hidden layer
    network_obj.bias{1} = rand(1,num_hidden_units);
    
    % hidden-to-output layer
    network_obj.W{2} = rand(num_hidden_units, num_outputs); %W{2} is weights between hidden layer and output
    network_obj.bias{2} = rand(1, num_outputs);
    
    network_obj.h_out = [];         %output of hidden layer
    network_obj.h_out_error = [];   %error associated with output of hidden layer
    
    % output
    network_obj.output = [];        %output of nn
    network_obj.error_out = [];  %error associated with output of nn
    
end

%activation function
function out = sigmoid(x)
    out = 1./(1+exp(-x));
end