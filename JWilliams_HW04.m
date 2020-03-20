%Joshua Williams
%Machine Learning HW04
%Test code with mnist handwriting data
%Base code uses lawrenceglewis' code

clear; clc;

%new learn mnist
[imgs, labels] = mnist_parse('train-images.idx3-ubyte', 'train-labels.idx1-ubyte');

f1 = figure;
for i=1:length(labels)
    % display the image and label associated
    imshow(imgs(:,:,i))
    title(num2str(labels(i)))
    figure(f1)
    pause(1);
end





%% functions
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