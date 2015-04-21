function D = calcAmariDist(A, B)
% function D = calcAmariDist(A, B)

J = size(A,2);

M = abs(pinv(A) * B);

D = sum(sum(M./repmat(max(M,[],1),[J 1]) + M./repmat(max(M,[],2),[1 J]))) - 2*J;
