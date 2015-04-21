function learn()
% function learn()
%
%  main function
%
%  yan karklin.  jan 2010.



params.epsy = 0.05;    % step size for inference of y's
params.epsb = 0.05;    % step size for gradient ascent on dL/dB
params.epsw = 0.05;     % step size for gradient ascent on dL/dw
params.decayw = 0.1;   % decay amount for weights w
params.yiters = 50;    % number of steps in MAP inference of y
params.yvar	 = 1;       % variance of y

params.side = 20;   
params.I = params.side * params.side;  % dimensionality of input
params.J = 10;  % number of units (y's) in the model 
params.K = 10;  % number of directions in input space used to stretch the covariance matrix
               %   i.e. number of vectors in B
params.N = 100; % data batch size

% Folder params
params.imagefolder = fullfile(pwd, 'images');
params.featfolder = fullfile(pwd,'features');

params.debug    = 0;
params.nb_draw = 0;
params.dispfreq = 20;
params.dispfeatfreq = 1;


% Function has changed, only displays the log likelihood now
ax = setUpFig(params);

[Model trueModel Hist] = initModel(params);

% Function to display the filters
drawFeatures(Model, params);

% main learning loop
for t=1:10000,
    fprintf(num2str(t));
    fprintf(', ');
  % Change the getData interface to give back images as vectors
  Data = getData(params, trueModel);

  % infer MAP estimate for v
  Data = inferLatents(Model, Data, params);

  % use this to take a gradient step in b, w
  [db dw] = calcDeltaBw(Model, Data, params); 
  db = cliprange(db,2); dw = cliprange(dw,2);
  
  % weight decay to w
  dw = dw - params.decayw*Model.w; 

  Model.b  = Model.b  + params.epsb  * db;
  Model.w  = Model.w  + params.epsw  * dw;

  % keep vectors in B unit-lenglfth
  Model.b = Model.b * diag(1./sqrt(sum(Model.b.^2)));
  
  if ~rem(t,params.dispfreq),
    Hist = drawDisplay(trueModel, Model, Data, Hist, params, ax, t);
  end;
  params.nb_draw = t;
  if ~rem(t,params.dispfeatfreq),
      drawFeatures(Model, params)
  end
  
end;


function ax = setUpFig(params)
figure(1); clf; colormap gray;
set(gcf,'defaultlinelinewidth',2);
ax(1) = subplot(1,1,1);









function [Model trueModel Hist] = initModel(params)
[I J K] = deal(params.I, params.J, params.K);


bk = zeros(I,1); bk(1:7) = fir1(6,.5,'high')';
for k=1:K,
  trueModel.b(:,k) = shift(bk,[2*(k-1) 0]);
end;
for j=1:J,
  trueModel.w(:,j) = sin(j*(pi*(1:K)/K) + j/pi)';
end;

% start with small random values
Model.b = 0.1*randn(I,K);
Model.w = 0.1*randn(K,J); 

% but make sure vectors in B unit-length
Model.b     = Model.b * diag(1./sqrt(sum(Model.b.^2)));
trueModel.b = trueModel.b * diag(1./sqrt(sum(trueModel.b.^2)));

% normalize w
Model.w     = 2*Model.w     * diag(1./sqrt(sum(Model.w.^2)));
trueModel.w = 2*trueModel.w * diag(1./sqrt(sum(trueModel.w.^2)));

Hist.T = [];
Hist.L = [];
Hist.Db = [];
Hist.Dw = [];






function Data = getData(params, trueModel)
% function Data = getData(params, trueModel)

I = params.I;
J = params.J;
N = params.N;

% draw random coefficients from a laplacian distribution
Data.truey = ggaussrnd(0,params.yvar,1, J,N);  
% (make sure generating y's are not too large -- otherwise exponential is unstable)
Data.truey = cliprange(Data.truey,3*sqrt(params.yvar)*[-1 1]);

% generate data:
% 1. start with Gaussian white noise
Data.x     = randn(I,N);
% 2. apply covariance matrices
for n=1:N,
  C12 = expm(trueModel.b*diag(trueModel.w*Data.truey(:,n))*trueModel.b'/2);
  Data.x(:,n) = C12*Data.x(:,n);
end;

% initialize latent variables
Data.y = 0.1*randn(J,N);







function [dB  dw] = calcDeltaBw(Model, Data, params)
% function [dB dw] = calcDeltaBw(Model, Data, params)
%
% uses series approximation to matrix exponential 
N = params.N;

BB = Model.b'*Model.b;  
wy = Model.w*Data.y;

C0 = Model.b'*Data.x; 
C1 = BB * (C0 .* wy); 
C2 = BB * (C1 .* wy);  
C3 = BB * (C2 .* wy); 
C4 = BB * (C3 .* wy);  
C5 = BB * (C4 .* wy);

C = C0.*(C0 - C1 + C2/3 - C3/12 + C4/60 -C5/360) + ...
    C1.*(C1/6 - C2/12  + C3/60 - C4/360 + C5/2520) + ...
    C2.*(C2/120 - C3/360 + C4/2520);

dw = 0.5 * (C - 1) * Data.y';

C0 = C0.*wy; C1 = C1.*wy; C2 = C2.*wy; C3 = C3.*wy; C4 = C4.*wy; C5 = C5.*wy;

dB =   Data.x * (C0 - C1/2 + C2/6  - C3/24  + C4/120)'+...
       + Model.b*(- C0*(C0/2 - C1/6 + C2/24 - C3/120 + C4/720)'+...
                  + C1*(C0/6 - C1/24 + C2/120 - C3/720 + C4/5040)' +...
                  - C2*(C0/24 - C1/120 + C2/720 - C3/5040)' +...
                  + C3*(C0/120 - C1/720 + C2/5040)' - diag(sum(wy,2)));

  
dw  = dw/N;
dB  = dB/N;







function h = saxes(h)
% function h = saxes(h)
%
%  silent axes function
%    modified from sfigure.m by Daniel Eaton, 2005


if nargin>=1,
  if ishandle(h), 
    figh = get(h,'parent');
    set(figh, 'CurrentAxes', h);
    set(0,'currentfigure',figh);
  else,              
    h = axes(h);
  end
else,
  h = axes;
end


function x = cliprange(x, maxv)
% function x = cliprange(x, maxv)

if length(maxv)==1, maxv = [-abs(maxv) abs(maxv)]; end;

x(find(x<maxv(1))) = maxv(1);
x(find(x>maxv(2))) = maxv(2);



    

