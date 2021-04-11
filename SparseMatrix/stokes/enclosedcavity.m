clear; close all; clc;

h=1/2^6;
nquad=5; %DSSY ow use 3
[domain,node,elem,edge,gquad]=domain2d([-1,1],[0,1],h,nquad);
nquad2=gquad.dim;
gp=gquad.gp; gw=gquad.gw; jac=gquad.Jacobian; detk=gquad.det;
nx=domain.nx; ny=domain.ny;
neg=edge.num;
nquad=gquad.num;
domx=domain.x;
domy=domain.y;
fprintf('mesh size=%f, nx=%d ny=%d\n',h,nx,ny)

%DSSY     
[nvb,phi,gphi]=ncelements(gp,'DSSY');
loc=elem.edge;

elemi=find(elem.boundary == 1);
%% MASS and STIFFNESS matrix
%%
%% mass matrix
mass=zeros(nvb,nvb,elem.num);
for ig=1:nquad2
    for j=1:nvb
        for k=j:nvb
            mass(j,k,:)=squeeze(mass(j,k,:))+ ...
                gw(ig).*detk.*(phi(ig,:,k).*phi(ig,:,j));
        end
    end
end
for j=2:nvb
    for k=1:j-1
        mass(j,k,:)=mass(k,j,:);
    end
end
mj=reshape(repmat(loc',nvb,1),nvb,nvb,elem.num);
mk=permute(mj,[2,1,3]); % xid transpose
M=sparse(mj(:),mk(:),mass(:));
%% stiffness matrix
invJ=inv(jac)';
stif=zeros(nvb,nvb,elem.num);
for ig=1:nquad2
    for j=1:nvb
        for k=j:nvb
            stif(j,k,:)=squeeze(stif(j,k,:))+ ...
                gw(ig).*detk.*((invJ*gphi(ig,:,k)')'*(invJ*gphi(ig,:,j)'));
        end
    end
end
for j=2:nvb
    for k=1:j-1
        stif(j,k,:)=stif(k,j,:);
    end
end
T=sparse(mj(:),mk(:),stif(:));


% Potential stiffness matrix
[~,pphi,gpphi]=ncelements(gp,'DSSY');
pstiff=zeros(nvb,1);
for ig=1:nquad2
    for j=1:nvb
         pstiff(j,1)=pstiff(j,1) + ...
            gw(ig).*detk.*((invJ*gpphi(ig,:,3)')'*(invJ*gphi(ig,:,j)'));
    end
end
for j=2:nvb
    for k=1:j-1
        pstiff(j,k,:)=pstiff(k,j,:);
    end
end
PT=sparse(mj(:),mk(:),pstiff(:));

dof=find(edge.boundary);
t_f = 10; t_i = 0; dt = 0.1;
ntime = (t_f - t_i)/dt;
con1=zeros(ntime,neg); cony=zeros(ntime,neg); 
pot=zeros(ntime,neg))

% boundary condition
con1(1:ntime,bdryedge)=0.0; , con2(1:ntime,)=0.0; , pot(1:ntime,)=0.0; , 
% bubs = Int (Grad DSSY) : (Grad phi_j)

%%unsteady time dependent part

% p, piecewise constant and integral int (d DSSY2)/(d x) = 0;
% P1NC part
A=S+M;
fx=-A*ux; fx=fx(dof);
fy=-A*uy; fy=fy(dof);
g=-[Bx,By]*[ux;uy];
% DSSY part
fx((ny-1-1)*(nx-1)+1)=fx((ny-1-1)*(nx-1)+1)-0.5*1*bubs(2);
fx((ny-1-1)*(nx-1)+nx-1)=fx((ny-1-1)*(nx-1)+nx-1)-0.5*1*bubs(1);
% IFISS solver
[wx,wy,p]=stokesolver(A(dof,dof),dofv,Bx(:,dof),By(:,dof),npb, ...
    [fx;fy;g],qm,1e-10,dofv);
ux(dof)=wx(1:dofv); uy(dof)=wy(1:dofv);
[tab,figd]=rpost(ux,uy,p,nx,ny,h,domx,domy,loc,T,dof,node,elem, ...
        nquad,nquad2,gp,gw,detk,invJ,nvb,npb,phi,gphi,psi,'P1NC+DSSY');
%% post process
obi=find(nu ~= 1);
obn=elem.node(obi,:);
obx=(node.coord(obn(:,1),1)+node.coord(obn(:,2),1))./2;
oby=(node.coord(obn(:,2),2)+node.coord(obn(:,3),2))./2;
%
cnum=[1,0.7,0.4,0.1,1.0*10^-3,1.0*10^-4,1.0*10^-5];
contour(figd.gridx,figd.gridy,figd.u,cnum,'k','Showtext','on'), hold on
plot(obx,oby,'s')
