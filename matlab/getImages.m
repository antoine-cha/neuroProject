function images = getImages(params)
%{
Return a batch of images 
Assume params.folder indicate a folder of patches
------------------------
%}
fileList = dir(params.imagefolder);
fileList = fileList(3:end); %Remove . and ..
% Display dir and file name
display(imageDir)
fprintf('Images : ')
fprintf(fileList(1).name)
fprintf(', ')
% Read the first image
temp = imread(fullfile(imageDir, fileList(1).name));
temp = temp(:, :, 1)/3 + temp(:, :, 2)/3 + temp(:, :, 3)/3;
images = reshape(temp, [1, params.I]);
fmax = size(fileList);
fmax = fmax(1);
% Read and append the others as line vectors
for f=2:fmax
   fprintf(fileList(f).name)
   fprintf(', ')
   temp = imread(fullfile(imageDir, fileList(f).name));
   % Get BW image if colour
   s = size(temp);
   if length(s)>2
        temp = temp(:, :, 1)/3 + temp(:, :, 2)/3 + temp(:, :, 3)/3;
   end
   temp = reshape(temp, [1, params.I]); % will crash if wrong size
   images = [images; temp];
end
fprintf('\n')