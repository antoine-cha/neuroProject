function Hist = drawDisplay(trueModel, Model, Data, Hist, params, ax, t)
% function Hist = drawDisplay(trueModel, Model, Data, Hist, params, ax, t)

Hist.T = [Hist.T t];
Hist.L = [Hist.L calcL(Model, Data, params)];

saxes(ax(1)); 
plot(Hist.T, Hist.L,'.k'); 
title(sprintf('[iter %d] log likelihood %.2f',Hist.T(end),Hist.L(end)));

drawnow; 


