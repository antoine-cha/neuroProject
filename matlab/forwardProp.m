function y = forwardProp(data, model, params)
%forwardProp(data, model, params)
%   perform the forward prop on a batch of images
% --------------------------------
    y = model.b' * data.x;
    y = model.w' * y;
end