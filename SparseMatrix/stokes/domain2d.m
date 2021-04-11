function [domain,node,elem,edge,gquad]=domain2d(domx,domy,h,nquad)
%% Problem
% The steady state Stokes equation,
% computational domain setup, rectangle
% uniform discretization, element: hxh square
% domx=[a,b], domy=[c,d], Omega = [a,b]x[c,d]
% h=1/2^l;
% nquad: # of quadrature points
%%
%% Discretization
%%
%% Domain, square
dim=2;
nx=(domx(2)-domx(1))/h;
ny=(domy(2)-domy(1))/h;
% structure
domain=struct('x',domx,'y',domy,'dim',dim,'nx',nx,'ny',ny,'h',h);
%% node
x=domx(1):h:domx(2);
y=domy(1):h:domy(2);
[X,Y]=meshgrid(x,y);
xi=reshape(X',(nx+1)*(ny+1),1);
eta=reshape(Y',(nx+1)*(ny+1),1);
xb=ones(size(x)); yb=ones(size(y));
xb(1)=0; xb(end)=0; yb(1)=0; yb(end)=0;
nbd=reshape(xb'*yb,(nx+1)*(ny+1),1); % boundary index
%node=[xi(:),eta(:)]; % x coord, y coord
nnd=(nx+1)*(ny+1); % # of nodes
%structure
node=struct('num',nnd,'coord',[xi(:),eta(:)],'boundary',nbd);
%% elem
% 4---3
% |   |
% 1---2
eln=zeros(nx*ny,4); % element to node
for jy=1:ny
    for jx=1:nx
        id=(jy-1)*nx+jx;
        eln(id,1)=(jy-1)*(nx+1)+jx;
        eln(id,2)=(jy-1)*(nx+1)+jx+1;
        eln(id,3)=jy*(nx+1)+jx+1;
        eln(id,4)=jy*(nx+1)+jx;
    end
end
nel=nx*ny; % # of elements
erb=zeros(nx*ny,1); % element, red-black index
for jy=1:2:ny
    for jx=1:2:nx
        id=(jy-1)*nx+jx;
        erb(id)=0; erb(id+1+nx)=0; % red elements
        erb(id+1)=1; erb(id+nx)=1; % black elements
    end
end
%% edge
%     3
%   4---3
% 4 |   | 2
%   1---2
%     1
edges=zeros(4*nel,2);
for j=1:nel
    edges((j-1)*4+1,:)=eln(j,[1,2]);
    edges((j-1)*4+2,:)=eln(j,[2,3]);
    edges((j-1)*4+3,:)=eln(j,[3,4]);
    edges((j-1)*4+4,:)=eln(j,[4,1]);
end
%edge=[elem(:,[1,2]),elem(:,[2,3]),elem(:,[3,4]),elem(:,[4,1])];
dd=sort(edges,2);
[egn,~,ic]=unique(dd,'rows'); % edge=dd(ia,:); dd=edge(ic,:);
egb=nbd(egn(:,1))+nbd(egn(:,2));
egb( egb == 2 )=1;
elg=zeros(nel,4); % elem to edge
bde=zeros(nel,1);
for j=1:nel
    elg(j,:)=ic((j-1)*4+1:(j-1)*4+4)';
    bde(j)=sum(egb(elg(j,:)));
end
neg=size(egn,1);
bdel=zeros(nel,1);
bdel( bde == 4)=1;
%%
%% element and edge structure
%%
elem=struct('num',nel,'node',eln,'edge',elg,'boundary',bdel,'rb',erb);
edge=struct('num',neg,'node',egn,'boundary',egb);
%%
%% Integration
%%
%% Gauss-Legengre Quadrature
[gp,gw]=gausslegendrequad2d(nquad);
nquad2=nquad^dim;
jac=[0.5*h,0;0,0.5*h];
detk=det(jac);
gquad=struct('num',nquad,'dim',nquad2,'gp',gp,'gw',gw,'Jacobian',jac,'det',detk);