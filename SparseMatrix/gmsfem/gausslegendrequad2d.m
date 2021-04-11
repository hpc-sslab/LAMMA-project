function [x2,c2]=gausslegendrequad2d(n)
P=zeros(n+1,n+1);
P([1,2],1)=1;
for k=1:n-1
    P(k+2,1:k+2)=((2*k+1)*[P(k+1,1:k+1),0]-k*[0,0,P(k,1:k)])/(k+1);
end
x=sort(roots(P(n+1,1:n+1)));
A=zeros(n,n);
for i=1:n
    A(i,:)=polyval(P(i,1:i),x)';
end
c=A\[2;zeros(n-1,1)];
%%
%  7-8-9
%  4-5-6
%  1-2-3
x2=zeros(n^2,2);
for jy=1:n
    for jx=1:n
        %fprintf('xi=%d, yi=%d, ind=%d \n',jx,jy,(jy-1)*n+jx)
        x2((jy-1)*n+jx,1)=x(jx);
        x2((jy-1)*n+jx,2)=x(jy);
    end
end
c2=zeros(n^2,1);
for jy=1:n
    for jx=1:n
        %fprintf('xi=%d, yi=%d, ind=%d \n',jx,jy,(jy-1)*n+jx)
        c2((jy-1)*n+jx,1)=c(jx)*c(jy);
    end
end