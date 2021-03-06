syms x1 x2 y1 y2 A11 A12 A13 A21 A22 A23 A31 A32 A33
%syms K11 K12 K13 K21 K22 K23 K31 K32 K33

K=rand(3,3);

eq13='-A12*(x1+y2)==1';
eq12='A12*(-x1+y2)+A11*(-x1+y1)==2';
eq11='-A11*(x1+y1)==3';
eq21='A11*(x1-y1)+A21*(x2-y1)==4';
eq31='-A21*(x2+y1)==5';

eqns=[-A12*(x1+y2)==1,A12*(-x1+y2)+A11*(-x1+y1)==2,-A11*(x1+y1)==3,A11*(x1-y1)+A21*(x2-y1)==4,-A21*(x2+y1)==5];


B=solve(eqns,x1,y1)

