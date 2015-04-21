function L = calcL(Model, Data, params)
% function L = calcL(Model, Data, params)
%
% yan karklin. jan 2010.

N = params.N;

exact = 1;
% Data Likelihood = log p(x|B,w) 
%                 = \int log p(x|B,w,y)p(y)  dy
%             \approx  log p(x|B,w,y_MAP)p(y_MAP)
%             \approx  log p(x|B,w,y_MAP) + log p(y_MAP)

L = 0;

if exact,

  % here's the exact likelihood using matlab's expm function

  wy = Model.w*Data.y;

  for n=1:N,
    
    logiC = -Model.b*diag(wy(:,n))*Model.b';
    L = L + .5*trace(logiC);
    L = L - .5 * Data.x(:,n)' * expm(logiC) * Data.x(:,n);
    
  end;
else,

  % this uses series approximation to expm (much faster)
  BB = Model.b'*Model.b;
  
  wy = Model.w * Data.y / 2;

  C0 = (Model.b' * Data.x).* wy;
  C1 = (BB * C0).*wy;  C2 = (BB * C1).*wy;
  C3 = (BB * C2).*wy;  C4 = (BB * C3).*wy;
  C =  Data.x +  Model.b * (-C0 + C1 / 2 - C2/6 + C3/24 - C4/120);

  L = -.5 * sum(sum(C.^2)) - sum(sum(wy));
  
end;

% lalpacian prior on y
L = L - sqrt(2)*sum(sum(abs(Data.y)))/sqrt(params.yvar);
% gaussian prior on y
%L = L - sum(sum(Data.y.^2));

L = L/N;


