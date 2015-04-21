function image = imageLoad(filename)
    % IMAGELOAD(filename)
    %   load one image from van Hateren's Natural Image Dataset
    % -----------------------------------------
    % filename : string
    %   absolute path to the image file 
    f1 = fopen(filename, 'rb', 'ieee-be');
    w = 1536; h = 1024;
    image = fread(f1, [w, h], 'uint16');
    image = image';
    fclose(f1);
end