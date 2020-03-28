%Joshua Williams
%Machine Learning HW04
%Test code with mnist handwriting data
%Base code uses lawrenceglewis' image input code

close all; clear; clc;

% read in training set of mnist data (28x28 x 60,000)
[imgs, labels] = mnist_parse('train-images.idx3-ubyte', 'train-labels.idx1-ubyte');
%displayMNISTimage(imgs,labels,1);

% unroll each greyscale image so X is 784 x 60,000 (transposed)
X = reshape(double(imgs), [784 60000])';
t = labels;

% normalize data
X = X/255;

% add column of ones to X (call this the design matrix "phi")
phi = [ones(60000,1),X];

[N,m] = size(phi);

%% ML stuff
rho = 0.0001;

% first let's train with just the digit 0
val = 0;    

% if it's the value we're training for, put a 1 there, otherwise put a zero
t(t==val) = 10;
t(t~=10) = 0;
t(t==10) = 1;

% fill W vector with random W's [ W1 W2 W3 ... ]
W = 0.05*ones(1,m);

% use gradient descent to find actual W's
for iter=1:100
    
    %prediction(X,W) should be values between 1 and 0 (because sigmoid function) 
    
    %convert this stuff to sum() function
    gradient = 0;
    for i=1:N   %summing per sample at the moment
        
        temp_pred = sigmoid(W*phi(i,:)');
        temp_difference = temp_pred - t(i);
        
        gradient = gradient + (temp_difference).*phi(i,:);
    end
    
    %update W using found gradient
    W = W - rho*gradient;
    
    %console debug
    disp(iter);
end

% compute answer using sigmoid function
y = sigmoid(W*phi');

% y(y>0.7) = 1;
% y(y<1) = 0;

for i=1:N
   if y(i) > 0.95
       displayMNISTimage(imgs,labels,i,y);
       pause;
   end   
end

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

% for debug
function displayMNISTimage(imgs,labels,k,y)
    imshow(imgs(:,:,k));
    title({
        [num2str(y(k)*100) '% confident it''s a zero']
        ['Actual: ' num2str(labels(k))]
        });
end

% logistic sigmoid function
function out = sigmoid(x)
     out = 1./(1+exp(-x));
end
