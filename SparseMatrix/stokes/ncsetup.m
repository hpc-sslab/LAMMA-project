%function
clear; close all; clc;
%% Problem
% The steady state Stokes equation,
% nc elements, mass, stiffness matrix
% if nc == DSSY, nquad=5;
h=1/2^6;
nquad=3;
[domain,node,elem,edge,gquad]=domain2d([-1,1],[0,1],h,nquad);
nquad2=gquad.dim;
gp=gquad.gp; gw=gquad.gw; jac=gquad.Jacobian; detk=gquad.det;
nc='P1NC';
switch nc
    case 'P1NC'
        [nvb,phi,gphi]=ncelements(gp,'P1NC');
        loc=elem.node;
    case 'rotQ1'
        [nvb,phi,gphi]=ncelements(gp,'rotQ1');
        loc=elem.edge;
    case 'DSSY'
        [nvb,phi,gphi]=ncelements(gp,'DSSY');
        loc=elem.edge;
    otherwise
        disp('error')
        return
end
%% Visocity:nu=1, sigma:sig=0
% random obstacles, 19
elemi=find( elem.boundary == 1);
obs=datasample(elemi,19);
nu=ones(elem.num,1);
nu(obs)=1/h;
sig=zeros(elem.num,1);
sig(obs)=1/h^3;
%%
%% MASS and STIFFNESS matrix
%%
%% mass matrix
mass=zeros(nvb,nvb,elem.num);
for ig=1:nquad2
    for j=1:nvb
        for k=j:nvb
            mass(j,k,:)=squeeze(mass(j,k,:))+ ...
                gw(ig).*detk.*(phi(ig,:,k).*phi(ig,:,j)).*sig;
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
%% stiffness matrix, nu=1
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
%% stiffness matrix
invJ=inv(jac)';
stif=zeros(nvb,nvb,elem.num);
for ig=1:nquad2
    for j=1:nvb
        for k=j:nvb
            stif(j,k,:)=squeeze(stif(j,k,:))+ ...
                gw(ig).*detk.*(...
                (invJ*gphi(ig,:,k)')'*(invJ*gphi(ig,:,j)') ).*nu;
        end
    end
end
for j=2:nvb
    for k=1:j-1
        stif(j,k,:)=stif(k,j,:);
    end
end
S=sparse(mj(:),mk(:),stif(:));
%% Gradient or Divergence matrix
if ( strcmp(nc,'P1NC') )
    npb=elem.num-2;
    psi=zeros(npb,2);
    rid=find(elem.rb == 0); bid=find(elem.rb);
    for j=1:size(rid,1)-1
        psi(j,:)=[rid(j),rid(j+1)];
        psi(elem.num/2-1+j,:)=[bid(j),bid(j+1)];
    end
else
    npb=elem.num-1;
    psi=zeros(npb,2);
    psi(:,1)=1:1:elem.num-1;
    psi(:,2)=psi(:,1)+1;
end
% 1 point is enough
nbq=1; nbq2=nbq^2;
[gp1,gw1]=gausslegendrequad2d(nbq);
[~,~,gphi1]=ncelements(gp1,nc);
bx=zeros(npb,2*nvb);
by=zeros(npb,2*nvb);
for ig=1:nbq2
    for k=1:nvb
        bx(:,k)=bx(:,k)+gw1(ig).*detk.*1.*invJ(1,1).*gphi1(ig,1,k);
        bx(:,nvb+k)=bx(:,nvb+k)+gw1(ig).*detk.*(-1).*invJ(1,1).*gphi1(ig,1,k);
        by(:,k)=by(:,k)+gw1(ig).*detk.*1.*invJ(2,2).*gphi1(ig,2,k);
        by(:,nvb+k)=by(:,nvb+k)+gw1(ig).*detk.*(-1).*invJ(2,2).*gphi1(ig,2,k);
    end
end
lj=repmat(1:npb,2*nvb,1); lj=lj';
lk=[loc(psi(:,1),:),loc(psi(:,2),:)];
Bx=-sparse(lj(:),lk(:),bx(:));
By=-sparse(lj(:),lk(:),by(:));
%%
%% Degrees of freedom
%%
if ( strcmp(nc,'P1NC') )
    dof=find(node.boundary);
else
    dof=find(edge.boundary);
end
%%
nx=domain.nx; ny=domain.ny;
neg=edge.num;
nquad=gquad.num;
domx=domain.x;
domy=domain.y;
save p1nc.mat h nx ny domx domy nquad nquad2 gp gw detk jac invJ...
    node edge elem phi gphi nu sig M S T mj mk nvb Bx By npb ...
    phi gphi psi dof loc