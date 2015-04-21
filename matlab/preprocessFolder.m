function preprocessFolder(source, target, vanhateren)
    % PREPROCESSFOLDER(source, target)
    % Preprocess a folder of images (source)
    % Write the new images at target
    %
    % -----------------------------------
    % source : string
    %     absolute path to the source folder
    % target : string
    %     absolute path to the target folder
    % vanhateren : bool
    %     indicates wether the imagefile belongs to Van Hateren dataset
    %     if TRUE, triggers another way to read the image
    fileList = dir(source);
    fileList = fileList(3:end);
    fmax = size(fileList);
    fmax = fmax(1);
    % Low pass gaussian filter
    %h = fspecial('gaussian', 20, 0.9);
    for f=1:fmax
        fprintf(fileList(f).name);
        fprintf('\n');
        filename =  fullfile(source, fileList(f).name);
        image = preprocessBig(filename, vanhateren);
        [~, name, ~] = fileparts(filename);
        targetname = fullfile(target, strcat(name,'.png'));
        imwrite(image, targetname);
    end
end





function image = preprocessBig(filename, vanhateren)
    % Preprocess the image at filename
    % 1. Log transform
    % 2. Low pass filtering (not done anymore)
    % -----------------------------------
    % filename : string
    %     absolute path to the image file
    % vanhateren : bool
    %     indicates wether the imagefile belongs to Van Hateren dataset
    %     if TRUE, triggers another way to read the image
    if vanhateren
        temp = double(imageLoad(filename));
    else
        temp = double(imread(filename));
    end
    sz = size(temp);
    % It will be 4 with tiff, we dont need the last dimension
    % Also deal with the RGB case
    if length(sz)>2
        temp = temp(:, :, 1:3);
    end
    R = max(temp(:));
    % 1. Log transform
    temp = log(1+temp)*255/log(1+R);
    % 2. Low pass filtering
    %temp = imfilter(temp, h);
    % We don't need too much accuracy
    image = uint8(temp);
end