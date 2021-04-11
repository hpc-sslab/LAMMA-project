clear; close all; clc;
% Stokes problem
addpath('./solvers')
% 064x064 grid : p1nc6.mat
% 128x128 grid : p1nc7.mat
% 256x256 grid : p1nc7.mat
load('p1nc8.mat')
nu=1; % viscosity
fprintf('Reynolds number=%d, mesh size=%f, N=%d \n',1/nu,hx,nx)
% DSSY
[~,bub,gbub]=ncelements(gp,'DSSY');
bubs=zeros(nvb,1);
for ig=1:nquad2
    for j=1:nvb
        bubs(j,1)=bubs(j,1) + ...
            gw(ig).*detk.*((invJ*gbub(ig,:,3)')'*(invJ*gphi(ig,:,j)'));
    end
end
% Pressure mass matrix
qm=4.*detk.*gallery('tridiag',npb/2);
qm=kron([1,0;0,1],qm);
dofv=size(dof,1);
%% Case 1
ux=zeros(nx+1,ny+1); uy=zeros(nx+1,ny+1);
ux(2:nx,ny+1)=0.5; % P1NC boundary condition
ux=reshape(ux,(nx+1)*(ny+1),1); uy=reshape(uy,(nx+1)*(ny+1),1);
% bubs = Int (Grad DSSY) : (Grad phi_j)
bubs=zeros(nvb,1);
for ig=1:nquad2
    for j=1:nvb
        bubs(j,1)=bubs(j,1) + ...
            gw(ig).*detk.*((invJ*gbub(ig,:,3)')'*(invJ*gphi(ig,:,j)'));
    end
end
% p, piecewise constant and integral int (d DSSY2)/(d x) = 0;
% P1NC part
fx=-(nu.*S)*ux; fx=fx(dof);
fy=-(nu.*S)*uy; fy=fy(dof);
g=-[Bx,By]*[ux;uy];
% DSSY part
fx((ny-1-1)*(nx-1)+1)=fx((ny-1-1)*(nx-1)+1)-0.5*nu*bubs(2);
fx((ny-1-1)*(nx-1)+nx-1)=fx((ny-1-1)*(nx-1)+nx-1)-0.5*nu*bubs(1);
% IFISS solver
[wx,wy,p]=stokesolver(nu*S(dof,dof),dofv,Bx(:,dof),By(:,dof),npb, ...
    [fx;fy;g],qm,1e-12,dofv);
ux(dof)=wx(1:dofv); uy(dof)=wy(1:dofv);
[~,tab1,figd1]=rtable(ux,uy,p,nx,ny,hx,hy,loc,S,dof,node,elem, ...
        nquad,nquad2,gp,gw,detk,invJ,nvb,npb,phi,gphi,psi,'P1NC+DSSY');
%% Case 2
ux=zeros(nx+1,ny+1); uy=zeros(nx+1,ny+1);
ux(2:2:nx,ny+1)=1; % P1NC boundary condition
ux=reshape(ux,(nx+1)*(ny+1),1); uy=reshape(uy,(nx+1)*(ny+1),1);
% P1NC part
fx=-(nu.*S)*ux; fx=fx(dof);
fy=-(nu.*S)*uy; fy=fy(dof);
g=-[Bx,By]*[ux;uy];
% IFISS solver
[wx,wy,p]=stokesolver(nu*S(dof,dof),dofv,Bx(:,dof),By(:,dof),npb, ...
    [fx;fy;g],qm,1e-12,dofv);
ux(dof)=wx(1:dofv); uy(dof)=wy(1:dofv);
[~,tab2,figd2]=rtable(ux,uy,p,nx,ny,hx,hy,loc,S,dof,node,elem, ...
        nquad,nquad2,gp,gw,detk,invJ,nvb,npb,phi,gphi,psi,'P1NC');
%% Case 3
ux=zeros(nx+1,ny+1); uy=zeros(nx+1,ny+1);
ux(3:2:nx-1,ny+1)=1; % P1NC boundary condition, top
ux(3:2:nx-1,1)=1; % P1NC boundary condition, bottom
ux(2:2:nx,1)=-1; % P1NC boundary condition, bottom
ux(1,1:2:ny+1)=1; % P1NC boundary condition, left
ux(1,2:2:ny)=-1; % P1NC boundary condition, left
ux(nx+1,1:2:ny+1)=1; % P1NC boundary condition, right
ux(nx+1,2:2:ny)=-1; % P1NC boundary condition, right
ux=reshape(ux,(nx+1)*(ny+1),1); uy=reshape(uy,(nx+1)*(ny+1),1);
% P1NC part
fx=-(nu.*S)*ux; fx=fx(dof);
fy=-(nu.*S)*uy; fy=fy(dof);
g=-[Bx,By]*[ux;uy];
% IFISS solver
[wx,wy,p]=stokesolver(nu*S(dof,dof),dofv,Bx(:,dof),By(:,dof),npb, ...
    [fx;fy;g],qm,1e-12,dofv);
ux(dof)=wx(1:dofv); uy(dof)=wy(1:dofv);
[~,tab3,figd3]=rtable(ux,uy,p,nx,ny,hx,hy,loc,S,dof,node,elem, ...
        nquad,nquad2,gp,gw,detk,invJ,nvb,npb,phi,gphi,psi,'P1NC');
%%
figure(1)
plot(tab1.vertical(:,1),tab1.vertical(:,2)), hold on
plot(tab2.vertical(:,1),tab2.vertical(:,2),'-s')
plot(tab3.vertical(:,1),tab3.vertical(:,2),'-*')
h=legend('P1NC-DSSY','P1NC-T1','P1NC-T2','location','best');
set(h,'fontsize',14);

figure(2)
plot(tab1.horizontal(:,1),tab1.horizontal(:,2)), hold on
plot(tab2.horizontal(:,1),tab2.horizontal(:,2),'-s')
plot(tab3.horizontal(:,1),tab3.horizontal(:,2),'-*')
h=legend('P1NC-DSSY','P1NC-T1','P1NC-T2','location','best');
set(h,'fontsize',14);