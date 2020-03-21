%Joshua Williams
%Machine Learning HW04
%Test code with mnist handwriting data
%Base code uses lawrenceglewis' code

close all; clear; clc;

% read in training set of mnist data (28x28 x 60,000)
[imgs, labels] = mnist_parse('train-images.idx3-ubyte', 'train-labels.idx1-ubyte');
%displayMNISTimage(imgs,labels,1);

% unroll each greyscale image so X is 784 x 60,000 (transposed)
X = reshape(double(imgs), [784 60000])';
t = labels;

[N,~] = size(X);
% add column of ones to X
X = [ones(N,1),X];

[N,m] = size(X);

% normalize data
X = X/255;

%% ML stuff
rho = 0.0000001;

% first let's train with just the 0's!
val = 0;    % if it's the value we're training for, put a 10 there, otherwise put a zero

t(t==val) = 10;
t(t~=10) = 0;
t(t==10) = 1;

% fill W vector with random W's [ W1 W2 W3 ... ]
W = zeros(1,m);
% map X inputs to phi (just using X at the moment)

% use gradient descent to find actual W's
for k=1:200
    %(prediction(X,W) should be values between 1 and 0 (because sigmoid function))
   
    pred = prediction(W,X');    
    gradient = (pred - t')*X;
    gradient = sum(gradient);
    
    disp(gradient);
    W = W - rho.*gradient;
    
end
% compute answer using sigmoid function

y = prediction(W,X');





%% functions
% read in mnist data
function [images, labels] = mnist_parse(path_to_digits, path_to_labels)

% Open digits files
fid_digits = fopen(path_to_digits, 'r');

% Open labels files
fid_labels = fopen(path_to_labels, 'r');

% Read in magic number for digits
A = fread(fid_digits, 1, 'uint32');
magicNumber1 = swapbytes(uint32(A)); % Should be 2051
fprintf('Magic Number - Images: %d\n', magicNumber1);

% Read in magic number for labels
A = fread(fid_labels, 1, 'uint32');
magicNumber2 = swapbytes(uint32(A)); % Should be 2049
fprintf('Magic Number - Labels: %d\n', magicNumber2);

% Read in total number of images
A = fread(fid_digits, 1, 'uint32');
totalImages = swapbytes(uint32(A));

A = fread(fid_labels, 1, 'uint32');

% Ensure that this number matches with the labels file
if totalImages ~= swapbytes(uint32(A))
    error('Total number of images read from images and labels files are not the same');
end

fprintf('Total number of images: %d\n', totalImages);

% Read in number of rows
A = fread(fid_digits, 1, 'uint32');
numRows = swapbytes(uint32(A));

% Read in number of columns
A = fread(fid_digits, 1, 'uint32');
numCols = swapbytes(uint32(A));

fprintf('Dimensions of each digit: %d x %d\n', numRows, numCols);

% For each image, store into an individual slice
images = zeros(numRows, numCols, totalImages, 'uint8');
for k = 1 : totalImages
    % Read in numRows*numCols pixels at a time
    A = fread(fid_digits, numRows*numCols, 'uint8');

    % Reshape so that it becomes a matrix
    % We are actually reading this in column major format
    % so we need to transpose this at the end
    images(:,:,k) = reshape(uint8(A), numCols, numRows).';
end

% Read in the labels
labels = fread(fid_labels, totalImages, 'uint8');

% Close the files
fclose(fid_digits);
fclose(fid_labels);

end

%for debug
function displayMNISTimage(imgs,labels,k)
    imshow(imgs(:,:,k));
    title(num2str(labels(k)));
end

% logistic sigmoid function
function out = sigmoid(x)
     out = 1./(1+exp(-x));
end

% hypothesis function
function out = prediction(W,X)
     out = sigmoid(W*X);
end