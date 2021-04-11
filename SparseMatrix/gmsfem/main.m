% Example of nonconforming GMsFEM for 2nd scalar elliptic equation
%
% 
% 
close all
%%
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%       BASIC VALUES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
global zero one two hf qt pi eps
zero = 0.0 ; one = 1.0 ;  two=2.0 ;  hf=0.5;  qt=0.25;
eps = exp(one); pi=acos(-one);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%       FETCH DATA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fid = fopen("./data-dssy", "r");
global x_beg x_end y_beg y_end homo epsil nx ny ndx ndy
global beta_fac method max_iter tol ncons extend extmom omega robin
global xsize ysize hx hy hdx hdy micnx micny noffbasis ntot nnum lnum
x_beg = fscanf(fid, '%*s %*s %*s %*s\n%f', 1);
x_end = fscanf(fid, '%f', 1);
y_beg = fscanf(fid, '%f', 1);
y_end = fscanf(fid, '%f\n', 1);
homo = fscanf(fid, '%*s %*s %*s\n%s\n', 1);
epsil = fscanf(fid, '%*s\n%f', 1);
nx = fscanf(fid, '%*s %*s\n%d', 1);
ny = fscanf(fid, '%d\n', 1);
ndx = fscanf(fid, '%*s %*s\n%d', 1);
ndy = fscanf(fid, '%d\n', 1);
beta_fac = fscanf(fid, '%*s\n%f', 1);
method = fscanf(fid, '%*s\n%s', 1);
max_iter = fscanf(fid, '%*s\n%d', 1);
tol = fscanf(fid, '%*s\n%f', 1);
ncons = fscanf(fid, '%*s\n%d', 1);
extend = fscanf(fid, '%*s\n%d', 1);
extmom = fscanf(fid, '%*s\n%d', 1);
omega = fscanf(fid, '%*s\n%f', 1);

% bdry condition
robin = zeros(4,1);

% domain information
xsize = x_end-x_beg;      ysize = y_end-y_beg;
hx = xsize/real(nx);      hy = ysize/real(ny);

% added
hdx = xsize/real(ndx);    hdy = ysize/real(ndy);
micnx = nx/ndx;    micny = ny/ndy;
noffbasis = 4*ncons;
nnum = (micnx+1)*(micny+1);
lnum = micnx*(micny+1)+(micnx+1)*micny;
ntot = ncons*( (ndx-1)*ndy+ndx*(ndy-1) );


%% SOURCE TERM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       EXACT SOL
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  u_ex = sin(3.0*pi*x).*y.*(one-y)...
%           +epsil*sin(pi.*x./epsil).*sin(pi.*y./epsil);
exSolCase = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       KAPPA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     kap
%     case homo
%         kap = ones(size(x)); % identity
%     case inho
%         kap = one + (one+x)*(one+y) +...
%                epsil*sin(10.0*pi*x)*sin(5.0*pi*y);
permTensorCase=3;

% gradient ftn(exactGrad.m) generator
syms a b
f=gradient(exactSol(a,b,exSolCase));
gradf=strrep(char(f),'*','.*');
gradf=strrep(gradf,'^','.^');
gradf=strrep(gradf,'matrix','');
fileID=fopen('exactGrad.m','w');
fprintf(fileID, 'function result=exactGrad(a,b)\n');
fprintf(fileID, 'result=');
fprintf(fileID, gradf);
fprintf(fileID, ';');
fclose(fileID);

% src function(src.m) generator
f=permTensor(a,b)*f;
f=-divergence(f);
vecf=strrep(char(f),'*','.*');
vecf=strrep(vecf,'^','.^');
fileID = fopen('src.m','w');
fprintf(fileID, 'function result=src(a,b)\n');
fprintf(fileID, 'result=');
fprintf(fileID, vecf);
fprintf(fileID, ';');
fclose(fileID);

%% CONTROL PARTS

% isSt : 0=not execute 1=generate microscale basis
% isMs : 0=not execute 1=generate multiscale basis
% isPlot : 0=no plot 1=plot
% isMid : 0=plot vertex value(onprogress) 1=plot midpoint value
% isOver : 0=no oversampling n=oversampling using n-width tube
% isEstiPerm : 0=exact perm coeffi 1=estimated perm coeffi from mic stiff
% assmRel : "avg3"= corn coef is avg of  o
%                                        o o
%           "avg4"= corn coef is avg of  o o
%                                        o o
%           "avg6"= corn coef is avg of  o
%                                        o o
%                                        o o o
%           "avg9"= corn coef is avg of  o o o
%                                        o o o
%                                        o o o
%           "half"= divide corner rel a21+a11+a12 to a21+a11/2 and
%           a11/2+a12
isSt=1;
isMs=1;
isPlot=0;
isMid=1;
isOver=0;
isEstiPerm=0;
isSavePerm=1;
assmRel="avg9";
GMsFEM=0;

FEType='p1nc';
% FEType='q1-c';
% FEType='DSSY';

%% BODY

MsStart=6;
MsEnd=10;
% for topp=MsStart:MsEnd
%% GRID INFORMATION
topp = 2;
%     nx=2^topp; ny=2^topp;
%     hx = xsize/real(nx);      hy = ysize/real(ny);

totNum=nx*ny;
%% GENERATE MESH
% number of Gauss quad pts
if (strcmp(FEType,'p1nc'))
    num_quad=2;
elseif (strcmp(FEType,'DSSY'))
    num_quad=5;
end

% Generate 2D Mesh
% whole domain
[domain,node,elem,edge,gaussquad]=domain2d([x_beg,x_end],[y_beg,y_end],hx,hy,num_quad);

if (strcmp(FEType,'p1nc'))
    loc=elem.node;
elseif (strcmp(FEType,'DSSY'))
    loc=elem.edge;
end

% number of quad pts
nquad2=gaussquad.dim;
% gaussquad point; weight; jacobian; determinent
gp=gaussquad.gp; gw=gaussquad.gw; jac=gaussquad.Jacobian; detk=gaussquad.det;

% [ num of vertices on bdry, phi, grad_phi ]
[nvb,phi,gphi]=elemGen(gp,FEType);

%% PROBLEM INFORMATION
    % Permeability tensor
    % perm : permeability coeff on each microscale grids
    perm=permTensor(elem.mid(:,1),elem.mid(:,2));
    perm=reshape(perm,elem.num,1);

    %%%%% perm save %%%%%
    if (isSavePerm==1)
    perm=perm';
    filename=strcat('exactPerm',int2str(permTensorCase),'.mat');
    save(filename, 'perm');
    perm=perm';
    end
    %%%%% perm save %%%%%

%% STIFFNESS MATRIX (MICRO-SCALE)
% local stiffness matrix
% invJ : inverse of jacobian matrix
invJ=inv(jac)';
stif=zeros(nvb,nvb,elem.num);
tmpGquad=zeros(size(elem.mid));
for ig=1:nquad2
    % tmpGquad : ig TH gquadpt on each micro-scale grid
    tmpGquad(:,1)=elem.mid(:,1)+0.5*hx*gp(ig,1);
    tmpGquad(:,2)=elem.mid(:,2)+0.5*hy*gp(ig,2);
    kapval=permTensor(tmpGquad(:,1),tmpGquad(:,2));
    for j=1:nvb
        for k=j:nvb
            % integrate grad phi * grad phi
            stif(j,k,:)=squeeze(stif(j,k,:))+...
                gw(ig).*detk.*(...
                (invJ*gphi(ig,:,k)')'*(invJ*gphi(ig,:,j)'));
        end
    end
end

% estimate kappa in p.w. constant
% (to 
for jelem=1:elem.num
    stif(:,:,jelem)=stif(:,:,jelem)*perm(jelem);
end

% symmetric
for j=2:nvb
    for k=1:j-1
        stif(j,k,:)=stif(k,j,:);
    end
end
% Assemble global stiffness matrix
% mj : node number for j
% mk : node number for k
mj=reshape(repmat(loc',nvb,1),nvb,nvb,elem.num);
mk=permute(mj,[2,1,3]); % xid transpose
S=sparse(mj(:),mk(:),stif(:));
%% MASS MATRIX (MICRO-SCALE)
mass=zeros(nvb,nvb,elem.num);
for ig=1:nquad2
    % tmpGquad : ig TH gquadpt on each micro-scale grid
    tmpGquad(:,1)=elem.mid(:,1)+0.5*hx*gp(ig,1);
    tmpGquad(:,2)=elem.mid(:,2)+0.5*hy*gp(ig,2);
    kapval=permTensor(tmpGquad(:,1),tmpGquad(:,2));
    for j=1:nvb
        for k=j:nvb
            mass(j,k,:)=squeeze(mass(j,k,:))+ ...
                kapval.*gw(ig).*detk.*(phi(ig,:,k).*phi(ig,:,j));
        end
    end
end
for j=2:nvb
    for k=1:j-1
        mass(j,k,:)=mass(k,j,:);
    end
end
M=sparse(mj(:),mk(:),mass(:));


%% LOAD VECTOR (MICRO-SCALE)
% rhs : load vector
rhs=zeros(nvb,elem.num);
for ig=1:nquad2
    % tmpGquad : ig TH gquadpt on each micro-scale grid
    tmpGquad(:,1)=elem.mid(:,1)+0.5*hx*gp(ig,1);
    tmpGquad(:,2)=elem.mid(:,2)+0.5*hy*gp(ig,2);
    % fv = f value at each gquad pts
    fv=src(tmpGquad(:,1),tmpGquad(:,2));
    fv=squeeze(fv);
    for j=1:nvb
        % intetrate f*phi
        rhs(j,:)=rhs(j,:)+gw(ig).*detk.*(fv'.*phi(ig,j));
    end
end
% sum up the results
rhsf=sparse(loc,1,rhs');
%% MICRO-SCALE SOLUTION
% pick up dofs
if (strcmp(FEType,'p1nc'))
    dof=find(node.boundary);
    eta=zeros(node.num,1);
    
elseif (strcmp(FEType,'DSSY'))
    dof=find(edge.boundary);
    eta=zeros(edge.num,1);
    
    x=(domain.x(1)+hf*hx):hx:(domain.x(2)-hf*hx);
    y=(domain.y(1)+hf*hy):hy:(domain.y(2)-hf*hy);
    
    eta(edge.trace(:,1))=exactSol(x',y_beg,exSolCase);
    eta(edge.trace(:,3))=exactSol(x',y_end,exSolCase);
    eta(edge.trace(:,2))=exactSol(x_end,y',exSolCase);
    eta(edge.trace(:,4))=exactSol(x_beg,y',exSolCase);
end
% save dof only
Sdof=S(dof,dof);
fdof=rhsf(dof);
eta(dof)=Sdof\fdof;
%% ERROR ANAYLYSIS (MICRO-SCALE)
% l2 err
% l2Int : l2err integrand results
% enInt : Enerr integrand results
l2Int=zeros(elem.num,nquad2);
exL2Int=zeros(elem.num,nquad2);
enInt=zeros(elem.num,nquad2);
exEnInt=zeros(elem.num,nquad2);
% dofEta : eta value at each dof
if (strcmp(FEType,'p1nc'))
    dofEta=eta(elem.node);
elseif (strcmp(FEType,'DSSY'))
    dofEta=eta(elem.edge);
end
for ig=1:nquad2
    tmpGquad(:,1)=elem.mid(:,1)+0.5*hx*gp(ig,1);
    tmpGquad(:,2)=elem.mid(:,2)+0.5*hy*gp(ig,2);
    
    exSol=exactSol(tmpGquad(:,1),tmpGquad(:,2),exSolCase);
    exSol=squeeze(exSol);
    l2tmp=exSol;
    
    exGrd=exactGrad(tmpGquad(:,1),tmpGquad(:,2));
    exGrd=squeeze(exGrd);
    entmp=exGrd;
    for j=1:nvb
        l2tmp=l2tmp-dofEta(:,j).*phi(ig,j);
        entmp=entmp-dofEta(:,j).*squeeze(gphi(ig,:,j))*invJ;
    end
    l2Int(:,ig)=gw(ig).*detk.*l2tmp.*l2tmp;
    exL2Int(:,ig)=gw(ig).*detk.*exSol.*exSol;
    entmp=entmp.*entmp; exEntmp=exGrd.*exGrd;
    enInt(:,ig)=gw(ig).*detk.*(entmp(:,1)+entmp(:,2));
    exEnInt(:,ig)=gw(ig).*detk.*(exEntmp(:,1)+exEntmp(:,2));
end
L2err=sqrt(sum(sum(l2Int)));
Enerr=sqrt(sum(sum(enInt)));
exL2=sqrt(sum(sum(exL2Int)));
exEn=sqrt(sum(sum(exEnInt)));
L2ratio=1.0;
Enratio=1.0;
if(isMs==0)
    if(topp>MsStart)
        L2ratio=L2errold./L2err;
        Enratio=Enerrold./Enerr;
    end
end
L2errold=L2err;
Enerrold=Enerr;
fprintf("Exact L2 : %f Exact En : %f \n", exL2, exEn);
fprintf("l2err is %f order is %f \n",L2err,log2(L2ratio));
fprintf("Enerr is %f order is %f \n",Enerr,log2(Enratio));
% end

%figure check; Micro_al & Micro_be

figure
if (strcmp(FEType,'p1nc'))
    x=1:(domain.nx+1); y=1:domain.ny; [X,Y]=ndgrid(x,y);
    tmp=reshape(eta,domain.nx+1,domain.ny+1);
    mesh(X,Y,0.5*(tmp(:,1:end-1)+tmp(:,2:end)));
elseif (strcmp(FEType,'DSSY'))
    x=1:(size(edge.al,1));y=1:(size(edge.al,2));[X,Y]=ndgrid(x,y);
    mesh(X,Y,reshape(eta(edge.al),size(edge.al,1),size(edge.al,2)));
end
%
%         figure
%         x=1:(size(edge.be,1));y=1:(size(edge.be,2));[X,Y]=ndgrid(x,y);
%         mesh(X,Y,reshape(eta(edge.be),size(edge.be,1),size(edge.be,2)));



%% MULTISCALE BASIS
fprintf('%1s%9s%14s%18s%14s%20s\n','N_H','N_h', 'Rel.err.H^1', 'order_energy', 'Rel.err_L^2', 'order_l2norm');
formatSpec = '%d \t %d \t %8.4g \t %8.3g \t %8.4g \t %8.3g\n';

%% LOAD MICRO-SCALE STIFFNESS MATRIX
%     stifFile=['Stif',int2str(permTensorCase),'.mat'];
%     load(stifFile);
%     permFile=['perm',int2str(permTensorCase),'.mat'];
%     load(permFile);
%     domainFile=['domain.mat'];
%     load(domainFile);


%% SNAPSHOT AND OFFLINE
if (strcmp(FEType,'p1nc'))
    offEta=zeros(nnum, noffbasis, ndx, ndy);
    ovoffEta=zeros(nnum, noffbasis+4*extend, ndx, ndy);
elseif (strcmp(FEType,'DSSY'))
    offEta=zeros(lnum, noffbasis, ndx, ndy);
    ovoffEta=zeros(lnum, noffbasis+4*extend, ndx, ndy);
end
snapEdgEta=zeros(2*(micnx+micny),2*(micnx+micny)+8*extend,ndx,ndy);
for jd=1:ndx
    for kd=1:ndy
        ndxbeg=(jd-1)*micnx;    ndybeg=(kd-1)*micnx;
        ndxend=jd*micnx;        ndyend=kd*micny;
        adxbeg=0; adxend=0; adybeg=0; adyend=0;
        if (jd ~= 1)
            ndxbeg=ndxbeg-extend;
            adxbeg=extend;
        end
        if (jd ~= ndx)
            ndxend=ndxend+extend;
            adxend=extend;
        end
        if (kd ~= 1)
            ndybeg=ndybeg-extend;
            adybeg=extend;
        end
        if (kd ~= ndy)
            ndyend=ndyend+extend;
            adyend=extend;
        end
        
        % generate 2d domain
        [locDom,locNod,locElm,locEdg,locGqd]=...
            domain2d(hx*[ndxbeg, ndxend],hy*[ndybeg, ndyend],hx,hy,num_quad);
        bdrydof=2*(locDom.nx+locDom.ny);
        
        if (strcmp(FEType,'p1nc'))
            locLoc=locElm.node;
        elseif (strcmp(FEType,'DSSY'))
            locLoc=locElm.edge;
        end
        
        % fetch loc_stif matrix from glb stif
        [X,Y]=meshgrid((ndxbeg+1):ndxend,(ndybeg+1):ndyend);
        locElmList=sort(reshape(X+nx*(Y-1),[],1));
        
        locStif=stif(:,:,locElmList);
        locMass=mass(:,:,locElmList);
        % assemble loc stiffness matrix
        locMj=reshape(repmat(locLoc',nvb,1),nvb,nvb,locElm.num);
        locMk=permute(locMj,[2,1,3]); % xid transpose
        locS=sparse(locMj(:),locMk(:),locStif(:));
        locM=sparse(locMj(:),locMk(:),locMass(:));
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%
        % construct snapshot space
        % snapEta is snapshot ftn.
        % Only save the traces of snap ftns on macro edges as snapEdgeta
        % snapEdgEta(2*(micnx+micny),2*(micnx+micny)+8*extend,ndx,ndy);
        
        if (strcmp(FEType,'p1nc'))
            locDof=find(locNod.boundary);
            locBdry=find(1-locNod.boundary);
            snapEta=zeros(locNod.num,bdrydof);
        elseif (strcmp(FEType,'DSSY'))
            locDof=find(locEdg.boundary);
            snapEta=zeros(locEdg.num,bdrydof);
        end
        
          
        % apply bdry condition (dirac-delta)
        if (strcmp(FEType, 'p1nc'))
            for jelem=1:bdrydof
                snapEta(locBdry(jelem),jelem)=1;
            end
            
        elseif (strcmp(FEType, 'DSSY'))
            for jelem=1:locDom.nx
                snapEta(locEdg.trace(jelem,1),jelem)=1;
                snapEta(locEdg.trace(jelem,3),jelem+locDom.nx+locDom.ny)=1;
            end
            for jelem=1:locDom.ny
                snapEta(locEdg.trace(jelem,2),jelem+locDom.nx)=1;
                snapEta(locEdg.trace(jelem,4),jelem+2*locDom.nx+locDom.ny)=1;
            end
        end
        
        % build locRhs %To solve without bdrydof
        if (strcmp(FEType, 'p1nc'))
            locRhs=zeros(locNod.num,bdrydof);
            for jelem=1:bdrydof
                ind=locBdry(jelem);
                [row, col]=find(locElm.node == ind);
                if (size(row,1)==1)
                    nod1=locElm.node(row(1),:);
                    locRhs(nod1,jelem)=-locStif(col(1),:,row(1));
                else
                    nod1=locElm.node(row(1),:);
                    nod2=locElm.node(row(2),:);
                    locRhs(nod1,jelem)=-locStif(col(1),:,row(1))';
                    locRhs(nod2,jelem)=locRhs(nod2,jelem)-locStif(col(2),:,row(2))';
                end
            end
        elseif (strcmp(FEType, 'DSSY'))
            locRhs=zeros(locEdg.num,bdrydof);
            for jelem=1:locDom.nx
                nod1=locElm.trace(jelem,1);
                nod3=locElm.trace(jelem,3);
                edg1=locElm.edge(nod1,:);
                edg3=locElm.edge(nod3,:);
                locRhs(edg1,jelem)=-locStif(1,:,nod1);
                locRhs(edg3,jelem+locDom.nx+locDom.ny)=-locStif(3,:,nod3);
            end
            for jelem=1:locDom.ny
                nod2=locElm.trace(jelem,2);
                nod4=locElm.trace(jelem,4);
                edg2=locElm.edge(nod2,:);
                edg4=locElm.edge(nod4,:);
                locRhs(edg2,jelem+locDom.nx)=-locStif(2,:,nod2);
                locRhs(edg4,jelem+2*locDom.nx+locDom.ny)=-locStif(4,:,nod4);
            end
        end
        
        % solve local problem
        locSdof=locS(locDof,locDof);
        locFdof=locRhs(locDof,:);
        snapEta(locDof,:)=locSdof\locFdof;
        
        % figure check for Snapshot ftn. alpha
        %         figure
        %         if (strcmp(FEType, 'p1nc'))
        %             for jelem = 1:bdrydof
        %                 x=1:(size(locEdg.al,1));y=1:(size(locEdg.al,2));[X,Y]=ndgrid(x,y);
        %                 tmp=reshape(snapEta(:,jelem),locDom.nx+1,locDom.ny+1);
        %                 tmp=(tmp(:,1:end-1)+tmp(:,2:end))/2;
        %                 mesh(X,Y,tmp)
        %             end
        %         elseif (strcmp(FEType, 'DSSY'))
        %             for jelem = 1:bdrydof
        %                 x=1:(size(locEdg.al,1));y=1:(size(locEdg.al,2));[X,Y]=ndgrid(x,y);
        %                 mesh(X,Y,reshape(snapEta(locEdg.al,jelem),size(locEdg.al,1),size(locEdg.al,2)))
        %             end
        %             snap_al=cell(1,bdrydof);
        %             for jelem=1:bdrydof
        %                 snap_al{jelem} = reshape(snapEta(locEdg.al,jelem),size(locEdg.al,1),size(locEdg.al,2));
        %             end
        %         end
        
        if (strcmp(FEType, 'p1nc'))
            % original mac domain
            tmp_dom=1:locNod.num;
            tmp_dom=reshape(tmp_dom, locDom.nx+1, locDom.ny+1);
            mac_dom=tmp_dom(1+adxbeg:end-adxend, 1+adybeg:end-adyend);
            
            % save the traces of the snap ftns on macro edges.
            
            snapEdgEta(1:micnx,1:bdrydof,jd,kd)=...
                snapEta(mac_dom(1:end-1,1),:)+snapEta(mac_dom(2:end,1),:);
            snapEdgEta(micnx+1:micnx+micny,1:bdrydof,jd,kd)=...
                snapEta(mac_dom(end,1:end-1),:)+snapEta(mac_dom(end,2:end),:);
            snapEdgEta(micnx+micny+1:2*micnx+micny,1:bdrydof,jd,kd)=...
                snapEta(mac_dom(1:end-1,end),:)+snapEta(mac_dom(2:end,end),:);
            snapEdgEta(2*micnx+micny+1:2*micnx+2*micny,1:bdrydof,jd,kd)=...
                snapEta(mac_dom(1,1:end-1),:)+snapEta(mac_dom(1,2:end),:);
        elseif (strcmp(FEType, 'DSSY'))
            % original mac domain
            mac_al=locEdg.al(1+adxbeg:end-adxend, 1+adybeg:end-adyend);
            mac_be=locEdg.be(1+adxbeg:end-adxend, 1+adybeg:end-adyend);
            
            % save the traces of the snap ftns on macro edges.
            snapEdgEta(1:micnx,1:bdrydof,jd,kd)=snapEta(mac_be(:,1),:);
            snapEdgEta(micnx+1:micnx+micny,1:bdrydof,jd,kd)=snapEta(mac_al(end,:),:);
            snapEdgEta(micnx+micny+1:2*micnx+micny,1:bdrydof,jd,kd)=snapEta(mac_be(:,end),:);
            snapEdgEta(2*micnx+micny+1:2*micnx+2*micny,1:bdrydof,jd,kd)=snapEta(mac_al(1,:),:);
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%
        % construct offline space
        % (offEta(lnum, noffbasis, jd, kd))
        
        % construct snapS snapM
        snapS=snapEta'*locS*snapEta;
        snapM=snapEta'*locM*snapEta;
        
        if (strcmp(FEType, 'DSSY'))
            % eig val decomp
            [V,lam_i]=eig(full(snapS),full(snapM));
            [lam_i,I]=sort(diag(lam_i),'ascend');
            % find offline
            offTmp=snapEta*V(:,I(1:noffbasis));
            ovoffTmp=snapEta*V(:,I(1:noffbasis+4*extend));
        elseif (strcmp(FEType, 'p1nc'))
            % eig val decomp
            [V,lam_i]=eig(full(snapS(1:end-1,1:end-1)),full(snapM(1:end-1,1:end-1)));
            [lam_i,I]=sort(diag(lam_i),'ascend');
            % find offline
            offTmp=snapEta(:,1:end-1)*V(:,I(1:noffbasis));
            ovoffTmp=snapEta(:,1:end-1)*V(:,I(1:noffbasis+4*extend));
        end
        
        % restrict off to mac dom
        if (strcmp(FEType, 'p1nc'))
            stddof=reshape(mac_dom,[],1);
        elseif (strcmp(FEType, 'DSSY'))
            stddof=[reshape(locEdg.al(1+adxbeg:end-adxend, 1+adybeg:end-adyend),[],1);...
                reshape(locEdg.be(1+adxbeg:end-adxend, 1+adybeg:end-adyend),[],1)];
        end
        offEta(:,:,jd,kd)=offTmp(sort(stddof),:);
        ovoffEta(:,:,jd,kd)=ovoffTmp(sort(stddof),:);
        
        %        % figure check for Offline ftn.
%         figure
%         if (strcmp(FEType, 'p1nc'))
%             for jelem = 1:bdrydof
%                 x=1:(size(locEdg.al,1));y=1:(size(locEdg.al,2));[X,Y]=ndgrid(x,y);
%                 tmp=reshape(offTmp(tmp_dom,jelem),locDom.nx+1,locDom.ny+1);
%                 tmp=(tmp(:,1:end-1)+tmp(:,2:end))/2;
%                 mesh(X,Y,tmp)
%             end
%         elseif (strcmp(FEType, 'DSSY'))
%             for jelem = 1:noffbasis
%                 x=1:(size(locEdg.al,1)-1);y=1:(size(locEdg.al,2)-1);[X,Y]=ndgrid(x,y);
%                 mesh(X,Y,reshape(offTmp(locEdg.al(1:end-1, 1:end-1),jelem),size(locEdg.al,1)-1,size(locEdg.al,2)-1))
%             end
%         end
    end
end

%% MOMENT
% snapEdgEta=zeros(2*(micnx+micny),2*(micnx+micny)+8*extend,ndx,ndy);
% generate mom domain
momEta=zeros(2*(micnx+micny),noffbasis,ndx,ndy);
ed1=micnx; ed2=micnx+micny; ed3=2*micnx+micny; ed4=2*micnx+2*micny;
st1=1;     st2=ed1+1;       st3=ed2+1;         st4=ed3+1;
for jd=1:ndx
    for kd=1:ndy
        if (kd == 1) % bottom
            tmp1=snapEdgEta(st1:ed1,:,jd,kd);
        else
            tmp1=[snapEdgEta(st1:ed1,:,jd,kd) ...
                snapEdgEta(st3:ed3,:,jd,kd-1)];
        end
        if (kd == ndy) % top
            tmp3=snapEdgEta(st3:ed3,:,jd,kd);
        else
            tmp3=[snapEdgEta(st1:ed1,:,jd,kd+1) ...
                snapEdgEta(st3:ed3,:,jd,kd)];
        end
        if (jd == 1) % leftmost
            tmp4=snapEdgEta(st4:ed4,:,jd,kd);
        else
            tmp4=[snapEdgEta(st2:ed2,:,jd-1,kd) ...
                snapEdgEta(st4:ed4,:,jd,kd)];
        end
        if (jd == ndx) % rightmost
            tmp2=snapEdgEta(st2:ed2,:,jd,kd);
        else
            tmp2=[snapEdgEta(st2:ed2,:,jd,kd) ...
                snapEdgEta(st4:ed4,:,jd+1,kd)];
        end
        [V1, lam1]=eig(tmp1'*tmp1);
        [V2, lam2]=eig(tmp2'*tmp2);
        [V3, lam3]=eig(tmp3'*tmp3);
        [V4, lam4]=eig(tmp4'*tmp4);
        [lam1, I1]=sort(diag(lam1),'descend');
        [lam2, I2]=sort(diag(lam2),'descend');
        [lam3, I3]=sort(diag(lam3),'descend');
        [lam4, I4]=sort(diag(lam4),'descend');
        V1 = V1(:,I1(1:ncons));
        V2 = V2(:,I2(1:ncons));
        V3 = V3(:,I3(1:ncons));
        V4 = V4(:,I4(1:ncons));
        tmp1=tmp1*V1;
        tmp2=tmp2*V2;
        tmp3=tmp3*V3;
        tmp4=tmp4*V4;
        momEta(st1:ed1,1:ncons,jd,kd)=tmp1;
        momEta(st2:ed2,ncons+1:2*ncons,jd,kd)=tmp2;
        momEta(st3:ed3,2*ncons+1:3*ncons,jd,kd)=tmp3;
        momEta(st4:ed4,3*ncons+1:4*ncons,jd,kd)=tmp4;
    end
end

%% PATCH TEST
[macDom,macNod,macElm,macEdg,macGqd]=...
    domain2d([0, hdx],[0, hdy],hx,hy,num_quad);
tmp_dom=reshape(1:macNod.num,macDom.nx+1,macDom.ny+1);
for jd=1:ndx
    for kd=1:ndy %Modified from ndx!
        if (strcmp(FEType, 'p1nc'))
            bnd_trace=zeros(2*(micnx+micny),noffbasis);
            bnd_trace(1:micnx,:)=...
                offEta(tmp_dom(1:end-1,1),:,jd,kd)+offEta(tmp_dom(2:end,1),:,jd,kd);
            bnd_trace(micnx+1:micnx+micny,:)=...
                offEta(tmp_dom(end,1:end-1),:,jd,kd)+offEta(tmp_dom(end,2:end),:,jd,kd);
            bnd_trace(micnx+micny+1:2*micnx+micny,:)=...
                offEta(tmp_dom(1:end-1,end),:,jd,kd)+offEta(tmp_dom(2:end,end),:,jd,kd);
            bnd_trace(2*micnx+micny+1:2*micnx+2*micny,:)=...
                offEta(tmp_dom(1,1:end-1),:,jd,kd)+offEta(tmp_dom(1,2:end),:,jd,kd);
        elseif (strcmp(FEType, 'DSSY'))
            bnd_trace=offEta(macEdg.trace,:,jd,kd);
        end
        tmp_mom=momEta(:,:,jd,kd);
        alpha=(bnd_trace'*tmp_mom)\eye(noffbasis);
        if rank(bnd_trace'*tmp_mom)<noffbasis
            fprintf('Constraint problem in coarse element %d %d is rank deficient in !!!',jd, kd);
        end
        offEta(:,:,jd,kd)=offEta(:,:,jd,kd)*alpha';
        
        
        if (strcmp(FEType, 'p1nc'))
            bnd_trace=zeros(2*(micnx+micny),noffbasis+4*extend);
            bnd_trace(1:micnx,:)=...
                ovoffEta(tmp_dom(1:end-1,1),:,jd,kd)+ovoffEta(tmp_dom(2:end,1),:,jd,kd);
            bnd_trace(micnx+1:micnx+micny,:)=...
                ovoffEta(tmp_dom(end,1:end-1),:,jd,kd)+ovoffEta(tmp_dom(end,2:end),:,jd,kd);
            bnd_trace(micnx+micny+1:2*micnx+micny,:)=...
                ovoffEta(tmp_dom(1:end-1,end),:,jd,kd)+ovoffEta(tmp_dom(2:end,end),:,jd,kd);
            bnd_trace(2*micnx+micny+1:2*micnx+2*micny,:)=...
                ovoffEta(tmp_dom(1,1:end-1),:,jd,kd)+ovoffEta(tmp_dom(1,2:end),:,jd,kd);
        elseif (strcmp(FEType, 'DSSY'))
            bnd_trace=ovoffEta(macEdg.trace,:,jd,kd);
        end
        beta=pinv(bnd_trace'*tmp_mom);
%         If we take more snapshot ftn;
%        offEta(:,:,jd,kd)=ovoffEta(:,:,jd,kd)*beta';
        
        % % %         test1
        %         offEta(macEdg.trace,:,jd,kd)*tmp_mom
        
              % figure check
%                 figure
%         if (strcmp(FEType, 'p1nc'))
%                 for jelem = 1:noffbasis
%                     x=1:(size(macEdg.al,1));y=1:(size(macEdg.al,2));[X,Y]=ndgrid(x,y);
%                     tmp=reshape(offEta(tmp_dom,jelem,jd,kd),macDom.nx+1,macDom.ny+1);
%                     tmp=(tmp(:,1:end-1)+tmp(:,2:end))/2;
%                     mesh(X,Y,tmp)
%                 end
%         elseif (strcmp(FEType, 'DSSY'))
%                 for jelem = 1:noffbasis
%                     x=1:(size(macEdg.al,1));y=1:(size(macEdg.al,2));[X,Y]=ndgrid(x,y);
%                     mesh(X,Y,reshape(offEta(macEdg.al,jelem,jd,kd),size(macEdg.al,1),size(macEdg.al,2)))
%                 end
%         end
    end
end

%% LINK BASIS
% % % % % % % % % % % % % % %
% % % WORKING
% % % % % % % % % % % % % % %
[msDom,msNod,msElm,msEdg,msGqd]=...
    domain2d([x_beg, x_end],[y_beg, y_end],hdx,hdy,num_quad);

if (strcmp(FEType, 'p1nc'))
    msLoc=ncons*msElm.edge;
elseif (strcmp(FEType, 'DSSY'))
    msLoc=ncons*msElm.edge;
end
msLoc=repmat(msLoc,ncons,1);
msLoc=reshape(msLoc,[], ncons*4);
for jelem=1:(ncons-1)
    msLoc(:,jelem:ncons:end)=msLoc(:,jelem:ncons:end)-ncons+jelem;
end
msMj=reshape(repmat(msLoc',noffbasis,1),noffbasis,noffbasis,msElm.num);
msMk=permute(msMj,[2,1,3]);% xid transpose

ms_stif=zeros(noffbasis,noffbasis,msElm.num);
ms_rhs=zeros(noffbasis,msElm.num);

if (strcmp(FEType, 'p1nc'))
    msEta=zeros(ncons*msEdg.num,1);
    msEta2=zeros(ncons*msEdg.num,1);
    msdof=find(repelem(msEdg.boundary,ncons,1));
elseif (strcmp(FEType, 'DSSY'))
    msEta=zeros(ncons*msEdg.num,1);
    msEta2=zeros(ncons*msEdg.num,1);
    msdof=find(repelem(msEdg.boundary,ncons,1));
end


for jd=1:ndx
    for kd=1:ndy
        elmind=jd+(kd-1)*ndx;
        ndxbeg=(jd-1)*micnx;    ndybeg=(kd-1)*micnx;
        ndxend=jd*micnx;        ndyend=kd*micny;
        
        % generate 2d domain
        [locDom,locNod,locElm,locEdg,locGqd]=...
            domain2d(hx*[ndxbeg, ndxend],hy*[ndybeg, ndyend],hx,hy,num_quad);
        bdrydof=2*(locDom.nx+locDom.ny);
        
        if (strcmp(FEType, 'p1nc'))
            locLoc=locElm.node;
        elseif (strcmp(FEType, 'DSSY'))
            locLoc=locElm.edge;
        end
        
        % fetch loc_stif matrix from glb stif
        [X,Y]=meshgrid((ndxbeg+1):ndxend,(ndybeg+1):ndyend);
        locElmList=sort(reshape(X+nx*(Y-1),[],1));
        
        locStif=stif(:,:,locElmList);
        locRhsf=rhs(:,locElmList);
        % assemble loc stiffness matrix
        locMj=reshape(repmat(locLoc',nvb,1),nvb,nvb,locElm.num);
        locMk=permute(locMj,[2,1,3]); % xid transpose
        locS=sparse(locMj(:),locMk(:),locStif(:));
        locF=sparse(locLoc,1,locRhsf');
        
        msOff=offEta(:,:,jd,kd);
        ms_stif(:,:,elmind)=msOff'*locS*msOff;
        ms_rhs(:,elmind)=msOff'*locF;
    end
end

msS=sparse(msMj(:),msMk(:),ms_stif(:));
for jelem=1:size(msS,2)
%     msS(jelem+1:end,jelem)=msS(jelem,jelem+1:end)';
end
msF=sparse(msLoc,1,ms_rhs'); %Note the transpose(ms_rhs)
msSdof=msS(msdof,msdof); msFdof=msF(msdof);
msEta(msdof)=msSdof\msFdof;
 
if (strcmp(FEType,'DSSY'))
    % ORDER
    tmpal=msEdg.al;
    tmpal=tmpal';
    tmpal=tmpal(:,2:end-1);
    tmpbe=msEdg.be;
    tmpbe=tmpbe(:,2:end-1);
    % % ord : list of dofs. counts vertical first horizontal later
    ord=[reshape(tmpal,[],1);reshape(tmpbe,[],1)];
    ord=repelem(ncons*ord,ncons,1);
    for jelem=1:(ncons-1)
        ord(jelem:ncons:end)=ord(jelem:ncons:end)-ncons+jelem;
    end
    tmpSdof=msS(ord,ord);
    tmpFdof=msF(ord);
    tmpEta=msEta;
    tmpEta(ord)=tmpSdof\tmpFdof;
end

%% ERROR ANALYSIS
% l2 err
% l2Int : l2err integrand results
% enInt : Enerr integrand results
l2Int=zeros(elem.num,nquad2);
enInt=zeros(elem.num,nquad2);
% dofEta : eta value at each dof
if (strcmp(FEType,'p1nc'))
    dofMsEta=msEta(msLoc);
elseif (strcmp(FEType,'DSSY'))
    dofMsEta=msEta(msLoc);
end
L2err=zeros(ndx,ndy);
Enerr=zeros(ndx,ndy);

figure
hold on

for jd=1:ndx
    for kd=1:ndy
        elmind=jd+(kd-1)*ndx;
        ndxbeg=(jd-1)*micnx;    ndybeg=(kd-1)*micnx;
        ndxend=jd*micnx;        ndyend=kd*micny;
        
        % generate 2d domain
        [locDom,locNod,locElm,locEdg,locGqd]=...
            domain2d(hx*[ndxbeg, ndxend],hy*[ndybeg, ndyend],hx,hy,num_quad);
        bdrydof=2*(locDom.nx+locDom.ny);
        
        if (strcmp(FEType, 'p1nc'))
        locLoc=locElm.node;
        elseif (strcmp(FEType, 'DSSY'))
        locLoc=locElm.edge;
        end
        
        % locEta : phi_j
        locEta=offEta(:,:,jd,kd);
        % locMsEta : a_j
        locMsEta=dofMsEta(elmind,:);
        
        % a_j * phi_j (apply weight)
        locEta=locEta*diag(locMsEta);
        % summation
        locEta=sum(locEta,2);
        
        % figure check
        %x=1:(size(locEdg.al,1));y=1:(size(locEdg.al,2));[X,Y]=ndgrid(x,y);
        %mesh(X,Y,reshape(locEta(locEdg.al),size(locEdg.al,1),size(locEdg.al,2)))
        
        if (strcmp(FEType, 'p1nc'))
        x=locDom.x(1):hx:locDom.x(2); y=locDom.y(1)+hy/2:hy:locDom.y(2);[X,Y]=ndgrid(x,y);
        tmp=reshape(locEta,locDom.nx+1,locDom.ny+1);
        mesh(X,Y,(tmp(:,1:end-1)+tmp(:,2:end))/2);
        elseif (strcmp(FEType, 'DSSY'))
        x=locDom.x(1):hx:locDom.x(2); y=locDom.y(1)+hy/2:hy:locDom.y(2);[X,Y]=ndgrid(x,y);
        mesh(X,Y,reshape(locEta(locEdg.al),size(locEdg.al,1),size(locEdg.al,2)));
        end
        
        dofEta=locEta(locLoc);
%         if (strcmp(FEType, 'p1nc'))
%         dofEta=locEta(locElm.node);
%         elseif (strcmp(FEType, 'DSSY'))
%         dofEta=locEta(locElm.edge);
%         end
        
        l2Int=zeros(locElm.num,nquad2);
        enInt=zeros(locElm.num,nquad2);
        tmpGquad=zeros(locElm.num,2);
        for ig=1:nquad2
            tmpGquad(:,1)=locElm.mid(:,1)+0.5*hx*gp(ig,1);
            tmpGquad(:,2)=locElm.mid(:,2)+0.5*hy*gp(ig,2);
            
            exSol=exactSol(tmpGquad(:,1),tmpGquad(:,2),exSolCase);
            exSol=squeeze(exSol);
            l2tmp=exSol;
            
            exGrd=exactGrad(tmpGquad(:,1),tmpGquad(:,2));
            exGrd=squeeze(exGrd);
            entmp=exGrd;
            for j=1:nvb
                l2tmp=l2tmp-dofEta(:,j).*phi(ig,j);
                entmp=entmp-dofEta(:,j).*squeeze(gphi(ig,:,j))*invJ;
            end
            l2Int(:,ig)=gw(ig).*detk.*l2tmp.*l2tmp;
            entmp=entmp.*entmp;
            enInt(:,ig)=gw(ig).*detk.*(entmp(:,1)+entmp(:,2));
        end
        L2err(jd,kd)=sum(sum(l2Int));
        Enerr(jd,kd)=sum(sum(enInt));
        
    end
end

finL2err=sqrt(sum(sum(L2err)));
finEnerr=sqrt(sum(sum(Enerr)));
RelL2err=finL2err/exL2
RelEnerr=finEnerr/exEn
hold off
