function visu = drawFeatures(Model, params)
    %{
    Draw the feature vectors as images
    ------------------------
    
    %}
    n_cols = 10;
    maxk = min(100, params.K);
    n_rows = floor((maxk - 1) / n_cols) + 1;  % Number of vectors in B
    fig = figure;
    set(fig, 'visible','off');
    colormap(gray);
    for i=1:maxk
        subplot(n_rows, n_cols, i);
        imagesc(reshape(Model.b(:, i),[params.side, params.side]));
        axis square;
        set(gca,'xtick',[],'ytick',[]);
    end
    hold off
    % Get the filename
    filename = strcat('features', num2str(params.nb_draw));
    filename = fullfile(params.featfolder, filename);
    % Save plot to file
    print(fig, filename, '-dpng')
    close(fig)
end