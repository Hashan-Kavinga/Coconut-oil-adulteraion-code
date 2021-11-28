function [ alpha ] = b_distance( X1,X2 )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
m=size(X1,2);
% check dimension 
% assert(isequal(size(X2),[n m]),'Dimension of X1 and X2 mismatch.');
assert(size(X2,2)==m,'Dimension of X1 and X2 mismatch.');
m1=mean(X1);
C1=cov(X1);
m2=mean(X2);
C2=cov(X2);

alpha_p1 = (1/8)*(m1-m2)*inv((C1+C2)./2)*(m1-m2)';
alpha_p2 = (1/2)*log(det((C1+C2)./2)/sqrt(det(C1)*det(C2)));

alpha = alpha_p1+alpha_p2;

end