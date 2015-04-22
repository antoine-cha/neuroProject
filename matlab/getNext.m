function batch = getNext(dataset, nb_batch, nb_examples)
% GETNEXT(images, nb_batch, nb_examples)
%   Returns a batch of nb_examples from the dataset
% -------------------------------------
% dataset  : nD-array
%   array representing the dataset
% nb_batch : int
%   index of the batch of nb_examples
% nb_examples : int
%   number of examples in the batch
    s = size(dataset);
    index = mod(nb_batch, s(1)/nb_examples);
    if (index+1)*nb_examples < s(1)
        start = index*nb_examples + 1;
        stop  = (index+1)*nb_examples;
    else
        % Border case
        % Reuse the last ones
        start = (index-1)*nb_examples - (index*nb_examples - s(1)) + 1;
        stop = s(1);
    end
    batch = dataset(start:stop,:);

end