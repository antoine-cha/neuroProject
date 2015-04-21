function Data = inferLatents(Model, Data, params)
% function Data = inferLatents(Model, Data, params)
% 
% yan karklin. jan 2010.

N = params.N;


if params.debug, Hist.L = calcL(Model, Data, params); Hist.y = Data.y(1:20);end;


% inference loop to compute \hat{y}
for i = 1:params.yiters,

  % start with small steps (makes things more stable)
  if     i<5,                 epsy = params.epsy/10;
  elseif i>(params.yiters-5), epsy = params.epsy/10; 
  else,                       epsy = params.epsy; end;
  
  % compute gradient
  dy = calcDeltay(Model, Data, params);
  
  % take step
  dy = cliprange(dy,2);
  Data.y = Data.y + epsy * dy;
  % for stability, keep within small range
  Data.y = cliprange(Data.y,2);
  
  if params.debug, 
    L =  calcL(Model, Data, params);
    Hist.L = [Hist.L L]; 
    Hist.y = [Hist.y; Data.y(1:20)]; 
  end;


  if params.debug & i==params.yiters, 
    figure(2); clf; 
    subplot(2,2,1); plot(Hist.L); title('log likelihood'); xlabel('iters');
    subplot(2,2,2); plot(Hist.y); title('first 20 y''s');  xlabel('iters');
    subplot(2,2,3); plot(Data.truey(:), Data.y(:),'.'); xlabel('generating y'); ylabel('estimated y');
    drawnow;
   
  end;
end;








function dy = calcDeltay(Model, Data, params)
% uses series taylor expansion for the exponential
BB = Model.b'*Model.b;  wy = -Model.w*Data.y;


C0 = Model.b'*Data.x;  
C1 = BB * (C0 .* wy); 
C2 = BB * (C1 .* wy);  C3 = BB * (C2 .* wy); 
C4 = BB * (C3 .* wy);  C5 = BB * (C4 .* wy);

C = C0.*(C0 + C1 + C2/3 + C3/12 + C4/60  + C5/360) + ...
    C1.*(          C1/6 + C2/12 + C3/60  + C4/360 + C5/2520) + ...
    C2.*(                         C2/120 + C3/360 + C4/2520);

C2 = zeros(size(C0));

% log posterior = log likelihood + log prior = log p(x|B,w,y) + log p(y)
% (likelihood) d p(x|B,w,y)/dy
dy = .5 * Model.w' * (C - 1);

% sparse (Laplacian) prior on y, d p(y)/dy
dy = dy - sqrt(2)*tanh(10*Data.y)/sqrt(params.yvar);




function x = cliprange(x, maxv)
% function x = cliprange(x, maxv)

if length(maxv)==1, maxv = [-abs(maxv) abs(maxv)]; end;

x(find(x<maxv(1))) = maxv(1);
x(find(x>maxv(2))) = maxv(2);
