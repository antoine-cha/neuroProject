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