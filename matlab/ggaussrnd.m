function r = ggaussrnd(mu,sigma,q,m,n)
% GGAUSRND Random matrices from an generalized Gaussain distribution.
%   R = GGAUSSRRND(MU,SIGMA,Q,M,N) returns an M by N matrix of random
%   numbers chosen from the generalized Gaussian distribution with
%   parameters MU, SIGMA, and Q.  All inputs are scalar.
%
%   See also EXPWRPDF.

% Written by Mike Lewicki 3/99
%
% Copyright (c) 1999 Michael S. Lewicki and CMU
%
% consolidated (to stand alone) by yan karklin. feb 2005.
%
% Permission to use, copy, modify, and distribute this software and its
% documentation for any purpose and without fee is hereby granted,
% provided that the above copyright notice and this paragraph appear in
% all copies.  Copyright holder(s) make no representation about the
% suitability of this software for any purpose. It is provided "as is"
% without express or implied warranty.

% Convert to a generalized exponential distribution


c = (gamma(3/q) / gamma(1/q))^(q/2);
s = sigma * c^(-1/q);

if (q == 1)
  % there seems to be a bug in gamrnd for this case where the
  % returned distribution has twice the variance
  r = exprnd(s,m,n);
else
  a = 1/q;
  b = s^q;
  r = gamrnd(a,b,m,n);
  r = r.^(1/q);
end

b = 2*unidrnd(2,m,n) - 3;

r = r .* b + mu;
