function images = getImages(imageDir, side)
% getImages(imageDir)
% Return all the images as one matrix 
% Assume params.folder indicate a folder of patches
% ------------------------
%
fileList = dir(imageDir);
fileList = fileList(3:end); %Remove . and ..
% Display dir and file name
display(imageDir)
fprintf('Images : ')
fprintf(fileList(1).name)
fprintf('\n')
% Read the first image
temp = imread(fullfile(imageDir, fileList(1).name));
images = reshape(temp, [1, side*side]);
fmax = size(fileList);
fmax = fmax(1);
% Read and append the others as line vectors
for f=2:fmax
   temp = imread(fullfile(imageDir, fileList(f).name));
   temp = reshape(temp, [1, side*side]); % will crash if wrong size
   images = [images; temp];
end
fprintf('\n')