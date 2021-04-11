function [tab,fig]=rpost(ux,uy,ps,nx,ny,h,domx,domy,loc,S,dofs, ...
    node,elem,nquad,nquad2,gp,gw,detk,invJ,nvb,npb,phi,gphi,psi,nc)
%%
%% vorticity and divergence of solution
%%
uh1dx=zeros(nx*ny,nquad2); uh1dy=zeros(nx*ny,nquad2);
uh2dx=zeros(nx*ny,nquad2); uh2dy=zeros(nx*ny,nquad2);
for ig=1:nquad2
    uh1dx(:,ig)=ux(loc(:,1)).*gphi(ig,1,1)+ux(loc(:,2)).*gphi(ig,1,2) ...
        +ux(loc(:,3)).*gphi(ig,1,3)+ux(loc(:,4)).*gphi(ig,1,4);
    uh1dy(:,ig)=ux(loc(:,1)).*gphi(ig,2,1)+ux(loc(:,2)).*gphi(ig,2,2) ...
        +ux(loc(:,3)).*gphi(ig,2,3)+ux(loc(:,4)).*gphi(ig,2,4);
    uh2dx(:,ig)=uy(loc(:,1)).*gphi(ig,1,1)+uy(loc(:,2)).*gphi(ig,1,2) ...
        +uy(loc(:,3)).*gphi(ig,1,3)+uy(loc(:,4)).*gphi(ig,1,4);
    uh2dy(:,ig)=uy(loc(:,1)).*gphi(ig,2,1)+uy(loc(:,2)).*gphi(ig,2,2) ...
        +uy(loc(:,3)).*gphi(ig,2,3)+uy(loc(:,4)).*gphi(ig,2,4);
end
uxe=ux(loc(:,1))+ux(loc(:,2))+ux(loc(:,3))+ux(loc(:,4));
uye=uy(loc(:,1))+uy(loc(:,2))+uy(loc(:,3))+uy(loc(:,4));
uxe=uxe./2;
uye=uye./2;
if ( strcmp(nc,'P1NC+DSSY') )
    [~,~,gdssy]=ncelements(gp,'DSSY');
    uh1dx((ny-1)*nx+1,1:nquad2)=uh1dx((ny-1)*nx+1,1:nquad2) ...
        +(0.5.*gdssy(1:nquad2,1,3))';
    uh1dx((ny-1)*nx+nx,1:nquad2)=uh1dx((ny-1)*nx+nx,1:nquad2) ...
        +(0.5.*gdssy(1:nquad2,1,3))';
    uh1dy((ny-1)*nx+1,1:nquad2)=uh1dy((ny-1)*nx+1,1:nquad2) ...
        +(0.5.*gdssy(1:nquad2,2,3))';
    uh1dy((ny-1)*nx+nx,1:nquad2)=uh1dy((ny-1)*nx+nx,1:nquad2) ...
        +(0.5.*gdssy(1:nquad2,2,3))';
    uxe((ny-1)*nx+1)=uxe((ny-1)*nx+1)+0.5*0.25;
    uxe((ny-1)*nx+nx)=uxe((ny-1)*nx+nx)+0.5*0.25;
end
uabs=sqrt(uxe.^2+uye.^2);
%
div=zeros(nx*ny,1);
vor=zeros(nx*ny,1);
vorc=zeros(nx*ny,1);
for ig=1:nquad2
    div(:)=div(:) ...
        +gw(ig).*detk.*(invJ(1,1).*uh1dx(:,ig)+invJ(2,2).*uh2dy(:,ig));
    vor(:)=vor(:) ...
        +gw(ig).*detk.*(-invJ(2,2).*uh1dy(:,ig)+invJ(1,1).*uh2dx(:,ig));
end
div=abs(div);
ig=((nquad+1)/2-1)*nquad+(nquad+1)/2;
vorc(:)=vorc(:) - invJ(2,2).*uh1dy(:,ig) + invJ(1,1).*uh2dx(:,ig);
%%
%% Pressure
%%
pressure=zeros(nx*ny,1);
for j=1:npb
    id=psi(j,1);
    pressure(id)=pressure(id)+ps(j);
    id=psi(j,2);
    pressure(id)=pressure(id)-ps(j);
end
%%
%% ux velocity along vertical line
%%
uxv=zeros(ny,2);
id=(0:1:ny-1).*nx+nx/2;
uxv(:,1)=(node.coord(elem.node(id,2),2)+node.coord(elem.node(id,3),2))./2;
if ( strcmp(nc,'P1NC') )||( strcmp(nc,'P1NC+DSSY') )
    uxv(:,2)=ux(loc(id,2))+ux(loc(id,3));
else
    uxv(:,2)=ux(loc(id,2));
end
%%
%% uy velocity along horizontal line
%%
uyv=zeros(nx,2);
id=(1:1:nx)+(ny/2-1)*nx;
uyv(:,1)=(node.coord(elem.node(id,3),1)+node.coord(elem.node(id,4),1))./2;
if ( strcmp(nc,'P1NC') )||( strcmp(nc,'P1NC+DSSY') )
    uyv(:,2)=uy(loc(id,3))+uy(loc(id,4));
else
    uyv(:,2)=uy(loc(id,3));
end
%%
%% streamline
%%
srhs=zeros(nx*ny,nvb);
for ig=1:nquad2
    for j=1:nvb
        avb=-invJ(2,2).*uh1dy(:,ig)+invJ(1,1).*uh2dx(:,ig);
        srhs(:,j)=srhs(:,j)+gw(ig).*detk.*avb.*phi(ig,1,j);
    end
end
srhs=sparse(loc,1,srhs);
sfn=mldivide(S(dofs,dofs),srhs(dofs));
sols=zeros(size(S,1),1); sols(dofs)=sfn;
ig=((nquad+1)/2-1)*nquad+(nquad+1)/2;
spsi=zeros(nx*ny,1);
spsi(:)=sols(loc(:,1)).*phi(ig,1,1)+sols(loc(:,2)).*phi(ig,1,2) ...
    +sols(loc(:,3)).*phi(ig,1,3)+sols(loc(:,4)).*phi(ig,1,4);
%%
%% Flow rate
%%
Qu=2.*(0.5.*h).*uxv(:,2);
Qv=2.*(0.5.*h).*uyv(:,2);
%%
%% Table
%%
xy=zeros(nx*ny,2);
xy(:,1)=(node.coord(elem.node(:,1),1)+node.coord(elem.node(:,2),1))./2;
xy(:,2)=(node.coord(elem.node(:,1),2)+node.coord(elem.node(:,4),2))./2;
% vortex table
% primary vortex
[mpsi,lpsi]=min(spsi);
prim=struct('psi',mpsi,'vorticity',vorc(lpsi),'coord',xy(lpsi,:));
%
tab=struct('primary',prim,'maxdiv',max(div),'intomega',sum(vor),...
    'Qu',abs(sum(Qu)),'Qv',abs(sum(Qv)),...
    'vertical',uxv,'horizontal',uyv);
%%
[xq,yq]=meshgrid(domx(1):h:domx(2),domy(1):h:domy(2));
sfig=griddata(xy(:,1),xy(:,2),spsi,xq,yq,'natural');
pfig=griddata(xy(:,1),xy(:,2),pressure,xq,yq,'natural');
vfig=griddata(xy(:,1),xy(:,2),vorc,xq,yq,'natural');
uxfig=griddata(xy(:,1),xy(:,2),uxe,xq,yq,'natural');
uyfig=griddata(xy(:,1),xy(:,2),uye,xq,yq,'natural');
ufig=griddata(xy(:,1),xy(:,2),uabs,xq,yq,'natural');
fig=struct('gridx',xq,'gridy',yq, ...
    'sf',sfig,'vorticity',vfig,'pressure',pfig,...
    'ux',uxfig,'uy',uyfig,'u',ufig);