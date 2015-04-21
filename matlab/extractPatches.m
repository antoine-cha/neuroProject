function extractPatches(source, target)
    %EXTRACTPATCHES(source, target)
    %   extract random patches from images in source folder
    %   Write the new images at target
    %-----------------------------------
    %source : string
    %    absolute path to the source folder
    %target : string
    %    absolute path to the target folder
    fileList = dir(source);
    fileList = fileList(3:end);
    fmax = size(fileList);
    fmax = fmax(1);
    % Define parameters :
    nb_patches = 50;    % patches per image
    side = 20;          % side of a patch in pixels
    for f=1:fmax
        fprintf(fileList(f).name);
        fprintf('\n');
        filename =  fullfile(source, fileList(f).name);
        image = preprocessPatches(filename, nb_patches, side);
        [~, name, ~] = fileparts(filename);
        targetname = fullfile(target, name);
        for v=1:nb_patches
            num = strcat(num2str(v), '.png');
            num = strcat('-', num);
            temp_name = strcat(targetname, num);
            imwrite(image(1+(v-1)*side:v*side, :), temp_name);
        end
    end
end






function patches = preprocessPatches(filename, nb_patches, side)
    % PREPROCESSPATCHES(filename, nb_patches)
    %   extract and preprocess patches of a given image file
    %----------------------------------------------
    % filename : string
    %   absolute path to the image
    % nb_patches : int
    %   number of patches to be extracted
    temp = imread(filename);
    [n, m, ~]= size(temp);
    % Instantiate patches
    if length(size(temp))>2
        patch = temp(randi(n-side+1)+(0:side-1), ...
            randi(m-side+1)+(0:side-1), :);
    else
        patch = temp(randi(n-side+1)+(0:side-1), ...
            randi(m-side+1)+(0:side-1));
    end
    % Compute the mean luminance and substract it
    L = mean(mean(mean(patch)));
    patch = patch - L;
    % Whiten the patch
    fX = fft(fft(patch,[],2),[],3); %fourier transform of the images
    spectr = sqrt(mean(abs(fX).^2)); %Mean spectrum
    patches = ifft(ifft(bsxfun(@times,fX,1./spectr),[],2),[],3); %whitened X
    
    for t=2:nb_patches
        % Extract patch
        if length(size(temp))>2
            patch = temp(randi(n-side+1)+(0:side-1), ...
                randi(m-side+1)+(0:side-1), :);
        else
            patch = temp(randi(n-side+1)+(0:side-1), ...
                randi(m-side+1)+(0:side-1));
        end
        % Compute the mean luminance then substract it
        %L  = 0.299*mean(patch(:,:,1)) + 0.587*mean(patch(:,:,2)) +...
        %     0.114*mean(patch(:,:,3));
        L = mean(mean(mean(patch)));
        patch = patch - L;
        % Whiten the patch
        fX = fft(fft(patch,[],2),[],3); %fourier transform of the images
        spectr = sqrt(mean(abs(fX).^2)); %Mean spectrum
        wX = ifft(ifft(bsxfun(@times,fX,1./spectr),[],2),[],3); %whitened X       
        patches = [patches; wX];
    end
end