%% SOURCE TERM
% oscillation
eps=1/60;
%     exSolCase
%     1 : x.*(1-x).*y.*(1-y) 2 : sin(pi.*x).*sin(pi.*y)
exSolCase=2; 
%     permTensorCase
%     case 1
%         k=ones(size(x)); % identity
%     case 2
%         k = exp(x.*y); % smooth & no oscillation
%     case 3
%         k = 10.*x+10.*y+10.*cos(pi.*(x+y)./eps)+10;
%     case 4
%         k=(1+x.*y).*(2./3).*(1+cos(2.*pi.*(x.*y)./eps).*(1+cos(2.*pi.*(x.*y)./eps)));
permTensorCase=4;

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
f=permTensor(a,b,eps,permTensorCase)*f;
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
isSt=1;
isMs=1; 
isPlot=1;
isMid=1;
isOver=0;
adjType=3;

%% BODY

MsStart=1;
MsEnd=5;
    %% GRID INFORMATION
    totNum=2^8; % number of fine grids on each side of whole domain
    
    totMic=totNum*totNum; % number of fine grids on whole domain

    % Size of fine and coarse grid
    h = 1/totNum;
    
if(isSt==1)
    %% GENERATE MESH
    % number of Gauss quad pts
    num_quad=3;

    % Generate 2D Mesh
    % whole domain
    [domain,node,elem,edge,gaussquad]=domain2d([0,1],[0,1],h,num_quad);
    
    loc=elem.node;
    % number of quad pts
    nquad2=gaussquad.dim;
    % gaussquad point; weight; jacobian; determinent
    gp=gaussquad.gp; gw=gaussquad.gw; jac=gaussquad.Jacobian; detk=gaussquad.det;

    % [ num of vertices on bdry, phi, grad_phi ]
    [nvb,phi,gphi]=p1nc_2d(gp);

    % Gauss quad pts location
    gquadpt=repelem(elem.mid,nquad2,1);
    % IF WE FIND BETTER WAY TO RESHAPE, PLEASE CHANGE IT SMARTER!!!!
    gquadpt_x=reshape(gquadpt(:,1),nquad2,1,elem.num);
    gquadpt_y=reshape(gquadpt(:,2),nquad2,1,elem.num);
    gquadpt=[gquadpt_x,gquadpt_y];

    clear gquadpt_x gquadpt_y;
    % gquadpt(:,:,n) includes nquad2 number of Gauss quad pts on nth element.
%    gquadpt=gquadpt+0.5*h*gp;
    %% PROBLEM INFORMATION
    % Permeability tensor
    % perm : permeability coeff on each microscale grids
    perm=permTensor(elem.mid(:,1),elem.mid(:,2),eps,permTensorCase);
%    perm=ones(domain.nx,domain.ny); % standard case ( A = 1 )
    perm=reshape(perm,elem.num,1);
    %% STIFFNESS MATRIX (MICRO-SCALE)
    % local stiffness matrix
    % invJ : inverse of jacobian matrix
    invJ=inv(jac)';
    stif=zeros(nvb,nvb,elem.num);
    for ig=1:nquad2
      for j=1:nvb
          for k=j:nvb
              % integrate grad phi * grad phi
              stif(j,k,:)=squeeze(stif(j,k,:))+...
                   gw(ig).*detk.*(...
                   (invJ*gphi(ig,:,k)')'*(invJ*gphi(ig,:,j)'));
          end
      end
    end
    % symmetric
    for j=2:nvb
        for k=1:j-1
            stif(j,k,:)=stif(k,j,:);
        end
    end

    % apply perm coeffi
    for k=1:elem.num
        stif(:,:,k)=perm(k)*stif(:,:,k);
    end

    % Assemble global stiffness matrix
    % mj : node number for j
    % mk : node number for k
    mj=reshape(repmat(loc',nvb,1),nvb,nvb,elem.num);
    mk=permute(mj,[2,1,3]); % xid transpose
    S=sparse(mj(:),mk(:),stif(:));
    
    %% LOAD VECTOR (MICRO-SCALE)
    % rhs : load vector
    rhs=zeros(nvb,elem.num);
    for ig=1:nquad2
        % fv = f value at each gquad pts
        fv=src(gquadpt(ig,1,:),gquadpt(ig,2,:));
        fv=squeeze(fv);
        for j=1:nvb
            % intetrate f*phi
           rhs(j,:)=rhs(j,:)+gw(ig).*detk.*(fv'.*phi(ig,j));
        end
    end
    % sum up the results
    rhs=sparse(loc,1,rhs');
    %% MICRO-SCALE SOLUTION
    % pick up dofs
    dof=find(node.boundary);
    % save dof only
    Sdof=S(dof,dof);
    fdof=rhs(dof);
    eta=zeros(node.num,1);
    eta(dof)=Sdof\fdof;
    %% ERROR ANAYLYSIS (MICRO-SCALE)
    % l2 err
    % l2Int : l2err integrand results
    % enInt : Enerr integrand results
    l2Int=zeros(elem.num,nquad2);
    enInt=zeros(elem.num,nquad2);
    % nodeEta : eta value at each node
    nodeEta=eta(elem.node);
    for ig=1:nquad2
        exSol=exactSol(gquadpt(ig,1,:),gquadpt(ig,2,:),exSolCase);
        exSol=squeeze(exSol);
        l2tmp=exSol;
        
        exGrd=exactGrad(gquadpt(ig,1,:),gquadpt(ig,2,:));
        exGrd=squeeze(exGrd);
        entmp=exGrd';
        for j=1:nvb
            l2tmp=l2tmp-nodeEta(:,j).*phi(ig,j);
            entmp=entmp-nodeEta(:,j).*gphi(ig,:,j)*invJ;
        end
        l2Int(:,ig)=gw(ig).*detk.*l2tmp.*l2tmp;
        entmp=perm.*entmp.*entmp;
        enInt(:,ig)=gw(ig).*detk.*(entmp(:,1)+entmp(:,2));
    end
    L2err=sqrt(sum(sum(l2Int)));
    Enerr=sqrt(sum(sum(enInt)));
    L2ratio=1.0;
    Enratio=1.0;
    if(isMs==0)
        if(hf>MsStart)
        L2ratio=L2errold./L2err;
        Enratio=Enerrold./Enerr;
        end
    end
    L2errold=L2err;
    Enerrold=Enerr;
    fprintf("l2err is %f order is %f \n",L2err,log2(L2ratio));  
    fprintf("Enerr is %f order is %f \n",Enerr,log2(Enratio));
end
    %% MULTISCALE BASIS
if (isMs==1)
    fprintf('%1s%9s%14s%18s%14s%20s\n','N_H','N_h', 'Rel.err.H^1', 'order_energy', 'Rel.err_L^2', 'order_l2norm');
formatSpec = '%d \t %d \t %8.4g \t %8.3g \t %8.4g \t %8.3g\n';
    
%     %% LOAD MICRO-SCALE STIFFNESS MATRIX
%     stifFile=['Stif',int2str(permTensorCase),'.mat'];
%     load(stifFile);
%     permFile=['perm',int2str(permTensorCase),'.mat'];
%     load(permFile);
%     domainFile=['domain.mat'];
%     load(domainFile);
    for hf=MsStart:MsEnd    
        %% GENERATE MESH
        num_quad=3;
                           
        macNum=2^hf; % number of coarse girds on each side of whole domain

        micNum=totNum/macNum; % number of fine grids on each side of coarse grid

        micInMac=micNum * micNum; % number of fine grids on each coarse grid
        macInDom=macNum * macNum; % number of fine grids on each coarse grid

        H = 1/macNum;
        
        % Oversampling parameters
        ovH=H+2*isOver*h;
        ovMicNum=ovH/h; % new micNum
        
        % Macro-scale domain
        [macDom,macNod,macElm,macEdg,macGqd]=domain2d([0,1],[0,1],H,num_quad);
        % reference domain ( 1 macro scale domain )
        [locDom,locNod,locElm,locEdg,locGqd]=domain2d([0,H],[0,H],h,num_quad);
        % reference domain ( oversampling)
        [ovLocDom,ovLocNod,ovLocElm,ovLocEdg,ovLocGqd]=domain2d([0,ovH],[0,ovH],h,num_quad);
        
        
        
        %% GQUAD PTS SETUP
        nquad2=locGqd.dim;
        % gaussquad point; weight; jacobian; determinent
        gp=locGqd.gp; gw=locGqd.gw; jac=locGqd.Jacobian; detk=locGqd.det;
        
        invJ=inv(jac)';
        
        macLoc=macElm.node;
        locLoc=locElm.node;
        
        % [ num of vertices on bdry, phi, grad_phi ]
        [nvb,phi,gphi]=p1nc_2d(gp);

        %% BUILD MULTISCALE BASIS 
        % Load Vector
        % dof marker
        locDof=find(locNod.boundary);
        ovLocDof=find(ovLocNod.boundary);

        % Generate 4 number of local fine scale load vector for each coarse grid
        % ( P1-NC / P1-NC )
        % Ref bdry weight
        refBdryW=zeros(micNum+1, micNum+1, 4);
        [refX,refY]=meshgrid(0:h:H,0:h:H);
        ovRefBdryW=zeros(ovMicNum+1, ovMicNum+1, 4);
        [ovRefX,ovRefY]=meshgrid(0:h:ovH,0:h:ovH);
        % MsBasisW : eta values of multiscale basis
        MsBasisW=zeros(locNod.num,nvb,macInDom);
        ovMsBasisW=zeros(ovLocNod.num,nvb,macInDom);
        % Bdry Weight = half of the bdry value on each bdry node.
        for i=1:4
            % Allocate boundary function on whole grid
            refBdryW(:,:,i)=...
                bdry_value(refX,refY,H,i).*0.5;
            ovRefBdryW(:,:,i)=...
                bdry_value(ovRefX,ovRefY,ovH,i).*0.5;
            % Make values 0 for interier node
            refBdryW(2:micNum,2:micNum,i)=0;
            ovRefBdryW(2:ovMicNum,2:ovMicNum,i)=0;
            % apply to MsEta by reshaping and repeleming bdry weight 
            tmp=reshape(refBdryW(:,:,i),locNod.num,1);
            ovTmp=reshape(ovRefBdryW(:,:,i),ovLocNod.num,1);
            MsBasisW(:,i,:)=repelem(tmp,1,macInDom);
            ovMsBasisW(:,i,:)=repelem(ovTmp,1,macInDom);
        end

        % Local perm coeffi ( perm coeffi at each coarse matrix )
        locPerm=mat2cell(reshape(perm,domain.nx,domain.ny),...
            micNum*ones(1,macNum),micNum*ones(1,macNum));
        locPerm=reshape(locPerm,1,[]);
        locPerm=cell2mat(locPerm);
        locPerm=reshape(locPerm,micNum,micNum,[]);
        
        ovLocPerm=zeros(ovMicNum,ovMicNum,macInDom);
        tmpPerm=reshape(perm,domain.nx,domain.ny);
        for ielem=1:macInDom            
            % Macro grid location ( 2D (0,0) to (macNum-1,macNum-1) )
            mx=mod(ielem-1,macNum);
            my=(ielem-mx-1)/macNum;
            % start point ( nw corner )
            crn=micNum*[mx,my]+[1,1];
            ovCrn=crn-[isOver,isOver];
            if (macElm.boundary(ielem))
                ovLocPerm(:,:,ielem)=tmpPerm(ovCrn(1):ovMicNum+ovCrn(1)-1,...
                    ovCrn(2):ovMicNum+ovCrn(2)-1);
            end
        end
        
        % Build the load vector for Ms basis
        % Since source term is 0, only the bdry weight and perm coeffi on
        % bdry elements will affect load vector
        MsloadVec=zeros(micNum-1,micNum-1,macInDom,4);
        ovMsloadVec=zeros(ovMicNum-1,ovMicNum-1,macInDom,4);
        for ielem=1:macInDom
            if (macElm.boundary(ielem)==0)
                for i=1:4
                    MsloadVec(1,:,ielem,i)=refBdryW(1,1:micNum-1,i)...
                        .*locPerm(1,1:micNum-1,ielem); % g_k,1 * K_k,1 
                    MsloadVec(1,:,ielem,i)=MsloadVec(1,:,ielem,i)+...
                        refBdryW(1,3:micNum+1,i)...
                        .*locPerm(1,2:micNum,ielem); % g_k+1,1 * K_k,1
                    MsloadVec(2:micNum-1,1,ielem,i)=refBdryW(2:micNum-1,1,i)...
                        .*locPerm(2:micNum-1,1,ielem); % g_1,k * K_1,k (1st comp is omitted)
                    MsloadVec(:,1,ielem,i)=MsloadVec(:,1,ielem,i)+...
                        refBdryW(3:micNum+1,1,i)...
                        .*locPerm(2:micNum,1,ielem); % g_1,k+1 * K_1,k
                    MsloadVec(micNum-1,2:micNum-1,ielem,i)=refBdryW(micNum+1,2:micNum-1,i)...
                        .*locPerm(micNum,2:micNum-1,ielem); % g_n+1,k * K_n,k (1st comp is omitted)
                    MsloadVec(micNum-1,:,ielem,i)=MsloadVec(micNum-1,:,ielem,i)+...
                        refBdryW(micNum+1,3:micNum+1,i)...
                        .*locPerm(micNum,2:micNum,ielem); % g_n+1,k+1 * K_n,k 
                    MsloadVec(2:micNum-1,micNum-1,ielem,i)=MsloadVec(2:micNum-1,micNum-1,ielem,i)+...
                        refBdryW(2:micNum-1,micNum+1,i)...
                        .*locPerm(2:micNum-1,micNum,ielem); % g_n+1,k * K_n,k (1st comp is omitted)
                    MsloadVec(1:micNum-2,micNum-1,ielem,i)=MsloadVec(1:micNum-2,micNum-1,ielem,i)+...
                        refBdryW(3:micNum,micNum+1,i)...
                        .*locPerm(2:micNum-1,micNum,ielem); % g_n+1,k+1 * K_n,k
                end
            else
                for i=1:4
                    ovMsloadVec(1,:,ielem,i)=ovRefBdryW(1,1:ovMicNum-1,i)...
                        .*ovLocPerm(1,1:ovMicNum-1,ielem); % g_k,1 * K_k,1 
                    ovMsloadVec(1,:,ielem,i)=ovMsloadVec(1,:,ielem,i)+...
                        ovRefBdryW(1,3:ovMicNum+1,i)...
                        .*ovLocPerm(1,2:ovMicNum,ielem); % g_k+1,1 * K_k,1
                    ovMsloadVec(2:ovMicNum-1,1,ielem,i)=ovRefBdryW(2:ovMicNum-1,1,i)...
                        .*ovLocPerm(2:ovMicNum-1,1,ielem); % g_1,k * K_1,k (1st comp is omitted)
                    ovMsloadVec(:,1,ielem,i)=ovMsloadVec(:,1,ielem,i)+...
                        ovRefBdryW(3:ovMicNum+1,1,i)...
                        .*ovLocPerm(2:ovMicNum,1,ielem); % g_1,k+1 * K_1,k
                    ovMsloadVec(ovMicNum-1,2:ovMicNum-1,ielem,i)=ovRefBdryW(ovMicNum+1,2:ovMicNum-1,i)...
                        .*ovLocPerm(ovMicNum,2:ovMicNum-1,ielem); % g_n+1,k * K_n,k (1st comp is omitted)
                    ovMsloadVec(ovMicNum-1,:,ielem,i)=ovMsloadVec(ovMicNum-1,:,ielem,i)+...
                        ovRefBdryW(ovMicNum+1,3:ovMicNum+1,i)...
                        .*ovLocPerm(ovMicNum,2:ovMicNum,ielem); % g_n+1,k+1 * K_n,k 
                    ovMsloadVec(2:ovMicNum-1,ovMicNum-1,ielem,i)=ovMsloadVec(2:ovMicNum-1,ovMicNum-1,ielem,i)+...
                        ovRefBdryW(2:ovMicNum-1,ovMicNum+1,i)...
                        .*ovLocPerm(2:ovMicNum-1,ovMicNum,ielem); % g_n+1,k * K_n,k (1st comp is omitted)
                    ovMsloadVec(1:ovMicNum-2,ovMicNum-1,ielem,i)=ovMsloadVec(1:ovMicNum-2,ovMicNum-1,ielem,i)+...
                        ovRefBdryW(3:ovMicNum,ovMicNum+1,i)...
                        .*ovLocPerm(2:ovMicNum-1,ovMicNum,ielem); % g_n+1,k+1 * K_n,k
                end
            end
        end
        MsloadVec=permute(MsloadVec,[1,2,4,3]);
        ovMsloadVec=permute(ovMsloadVec,[1,2,4,3]);
        MsloadVec=reshape(2*MsloadVec,(micNum-1)*(micNum-1),4,macInDom);        
        ovMsloadVec=reshape(2*ovMsloadVec,(ovMicNum-1)*(ovMicNum-1),4,macInDom);        
        
        
        
        
        % Generate local fine scale stiffness matrix for each coarse grid
        locVerNum=locNod.num;
        ovLocVerNum=ovLocNod.num;
 %       msstif=zeros(locVerNum, locVerNum, macInDom);
        for ielem=1:macInDom
            % Macro grid location ( 2D (0,0) to (macNum-1,macNum-1) )
            mx=mod(ielem-1,macNum);
            my=(ielem-mx-1)/macNum;
            % start point ( nw corner )
            crn=micNum*[mx,my]+[1,1];
            ovCrn=crn-[isOver,isOver];
            if (macElm.boundary(ielem)==0)
                % coordinates of local fine grid point ( 2D )
                [fLocX,fLocY]=meshgrid([crn(1):crn(1)+micNum],[crn(2):crn(2)+micNum]);
                % 1D expression of local fine grid point
                fLoc=fLocX+(fLocY-1)*(totNum+1);
                fLoc=reshape(fLoc',1,locVerNum);
                msstif=S(fLoc,fLoc); % local stiffness matrix
                MsBasisW(locDof,:,ielem)=...
                    msstif(locDof,locDof)\MsloadVec(:,:,ielem);
            else
                [fLocX,fLocY]=meshgrid([ovCrn(1):ovCrn(1)+ovMicNum],[ovCrn(2):ovCrn(2)+ovMicNum]);
                % 1D expression of local fine grid point
                fLoc=fLocX+(fLocY-1)*(totNum+1);
                fLoc=reshape(fLoc',1,ovLocVerNum);
                ovMsstif=S(fLoc,fLoc); % local stiffness matrix
                ovMsBasisW(ovLocDof,:,ielem)=...
                    ovMsstif(ovLocDof,ovLocDof)\ovMsloadVec(:,:,ielem);
                Tmp1=reshape(ovMsBasisW(:,:,ielem),...
                    ovMicNum+1,ovMicNum+1,4);
                Tmp2=Tmp1(1+isOver:1+micNum+isOver, 1+isOver:1+micNum+isOver,:);
                adjA=squeeze(Tmp2(micNum/2+1,1,:))-[0.5;0.5;0;0];
                adjB=squeeze(Tmp2(1,micNum/2+1,:))-[0.5;0;0;0.5];
                adjC=squeeze(Tmp2(micNum/2+1,micNum+1,:))-[0;0;0.5;0.5];
                adjD=squeeze(Tmp2(micNum+1,micNum/2+1,:))-[0;0.5;0.5;0];
                %adjC=squeeze(Tmp2(   
                for i=1:nvb
                    Tmp2(:,:,i)=Tmp2(:,:,i)-adjust(locNod.meshX',locNod.meshY',...
                        adjA(i),adjB(i),adjC(i),adjD(i),H,adjType);
                end
                MsBasisW(:,:,ielem)=...
                    reshape(Tmp2,(micNum+1)*(micNum+1),4);
            end
        
         end
        %% CACULATE SIFFNESS MATRIX & LOAD VECTOR ( MACRO-SCALE )
        % MsStif : local stiffness matrix for each macro element
        % MsEtaSq : square form of eta values of Ms basis
        % psi : Ms basis function
        % phi : Micro-scale basis fundtion
        MsStif=zeros(nvb,nvb,macElm.num);
        MsEtaSq=reshape(MsBasisW,micNum+1,micNum+1,nvb,macElm.num);
        % Along the j,k loop, we calculate grad(psi j) * grad(psi k)
        % for each jth and kth nodes of a macro-scale grid
        for j=1:nvb
            for k=1:nvb
                % lj, lk are the lj th, lk th nodes on micro-scale grid
                % First, we fix the MsEtaSq for psi j
                % And save the weight on lj th corner to tmplj;
                % The weight information is a (n-1) X (n-1) size submatrix 
                % of MsEtaSq and the starting point is derived by stpt ftn.
                for lj=1:nvb
                    [ljx,ljy]=stpt(lj);
                    tmplj=MsEtaSq(1+ljx:micNum+ljx,1+ljy:micNum+ljy,j,:);
                    tmplj=squeeze(tmplj);
                    % Since the grad(phi lj)*grad(phi lk) = 0 when 
                    % abs(lj-lk)=1, we only consider the identical case(2),
                    % and diagonal case(-2)
                    for tk=lj:2:lj+2
                        lk=mod(tk-1,nvb)+1;
                        [lkx,lky]=stpt(lk);
                        tmplk=MsEtaSq(1+lkx:micNum+lkx,1+lky:micNum+lky,k,:);
                        tmplk=squeeze(tmplk);
                        MsStif(j,k,:)=MsStif(j,k,:)+(-1)^(abs(lj-lk)/2)*2.*sum(sum(tmplj.*tmplk.*locPerm));
                    end
                end
            end
        end
        
        % Assemble Macro-scale stiffness matrix
        % macj : node number for j
        % mack : node number for k
        macj=reshape(repmat(macLoc',nvb,1),nvb,nvb,macElm.num);
        mack=permute(macj,[2,1,3]); % xid transpose
        MsS=sparse(macj(:),mack(:),MsStif(:)); % Macroscale stiffness mtx
         %% LOAD VECTOR (MACRO-SCALE)
        % MsRhs : Multiscale load vector 
        MsRhs=zeros(nvb,macElm.num);
        % locGquadPt : Midpoint of microscale grid in macroscale grid
        %
        %             
        % nwcrn : north west corner of each macroscale grid
        nwcrn=macNod.coord(macElm.node(:,1),:);
        locGquadPt=repmat(locElm.mid,macElm.num,1)+repelem(nwcrn,micInMac,1);
        for j=1:nvb
            % macEta : Eta value of each Ms basis ftn which are jth crn of
            %           Mac grid
            % tmpMsRhs : auxiliary space to store integrate value of f*phi
            macEta=reshape(MsBasisW(locElm.node,j,:),[],4,macInDom);
            tmpMsRhs=zeros(micInMac,macInDom);
            for ig=1:nquad2
                % tempGquad : coordinates of igTH gquadpt on each mic gird
                % Msfv : f value at each quad pts on microscale grid in 
                %       macroscale grid.
                tmpGquad=locGquadPt+0.5*h*gp(ig,:);
                Msfv=src(tmpGquad(:,1),tmpGquad(:,2));
                Msfv=reshape(Msfv,[],macInDom);
                for lj=1:nvb
                    tmpMsRhs=tmpMsRhs+...
                        gw(ig).*detk.*Msfv.*squeeze(macEta(:,lj,:)).*phi(ig,lj);                    
                end
            end
            MsRhs(j,:)=sum(tmpMsRhs);
        end
        % sum up the results
        MsRhs=sparse(macLoc,1,MsRhs');
        %% MULTI-SCALE SOLUTION
        % pick up dofs
        MsDof=find(macNod.boundary);
        % save dof only
        MsSdof=MsS(MsDof,MsDof);
        MsFdof=MsRhs(MsDof);
        MsEta=zeros(macNod.num,1);
        MsEta(MsDof)=MsSdof\MsFdof;
        
        
%         %% ERROR ANALYSIS ( TRY )
%         MsL2Int=zeros(macElm.num,nquad2);
%         MsL2Ext=zeros(macElm.num,nquad2);
%         Rec=zeros(macElm.num,1);
%         % MsNodeEta : Weight of Ms Basis ftns on each node of macro-scale 
%         %             grid
%         MsNodeEta=MsEta(macElm.node);
%         MsL2err=0;
%         for ielem=1:macElm.num
%             locG=nwcrn(ielem,:)+locElm.mid;
%             % loop of gquad pts
%             for ig=1:nquad2
%                 tmpGquad=locG+0.5*h*gp(ig,:);
%                 exSol=exactSol(tmpGquad(:,1),tmpGquad(:,2),exSolCase);
%                 MsL2ExtTmp=squeeze(exSol);
%                 MsL2IntTmp=MsL2ExtTmp*0;
%                 % loop of macro-scale node vertices
%                 for j=1:nvb
%                     % macMsEta : eta value of micro-scale basis ftn which
%                     %            consists of Ms basis ftn attached to
%                     %            j th corner of ielem th macro-scale element
%                     tmpVal=MsBasisW(:,j,ielem);
%                     macMsEta=reshape(tmpVal(locElm.node),[],4);
%                     % loop of micro-scale node vertices
%                     for innj=1:nvb
%                         tmpMsEta=MsNodeEta(ielem,j)*macMsEta(:,innj);
%                         MsL2IntTmp=MsL2IntTmp+tmpMsEta.*phi(ig,innj);
%                     end
%                 end
%                 MsL2err=MsL2err+sum(gw(ig).*detk.*(MsL2ExtTmp-MsL2IntTmp).*(MsL2ExtTmp-MsL2IntTmp));            
%             end
%             Rec(ielem)=MsL2err;
%         end
% %        MsL2err=sqrt(sum(sum(MsL2Int)));
%         MsL2err=sqrt(MsL2err);
%         MsL2ratio=1.0;
%         if(hf>MsStart)
%             MsL2ratio=MsL2errold./MsL2err;
% %            MsEnratio=MsEnerrold./MsEnerr;
%         end
%         MsL2errold=MsL2err;
%         %        MsEnerrold=MsEnerr;
%         fprintf("MsL2err is %f ratio is %f \n",MsL2err,MsL2ratio);
        
        
        %% ERROR ANALYSIS ( MULTI-SCALE )
        totElm=totNum*totNum;
        MsL2Int=zeros(totElm,nquad2);
        ExL2Int=zeros(totElm,1);
        MsEnInt=zeros(totElm,1);
        ExEnInt=zeros(totElm,1);
        MsNodeEta=MsEta(macElm.node);
        for ig=1:nquad2
            tmpGquad=locGquadPt+0.5*h*gp(ig,:);
            exSol=exactSol(tmpGquad(:,1),tmpGquad(:,2),exSolCase);
            MsL2ext=squeeze(exSol); 
            MsL2tmp=MsL2ext*0;
            
            exGrd=exactGrad(tmpGquad(:,1),tmpGquad(:,2));
            exGrd=squeeze(exGrd);
            MsEntmp=exGrd;
            for j=1:nvb
                macEta=reshape(MsBasisW(locElm.node,j,:),[],4,macInDom);
                for lj=1:nvb 
                    tmpNodeEta=(repelem(MsNodeEta(:,j),micInMac,1).*...
                        reshape(squeeze(macEta(:,lj,:)),[],1));
                    MsL2tmp=MsL2tmp+tmpNodeEta.*phi(ig,lj);
                    MsEntmp=MsEntmp-tmpNodeEta.*gphi(ig,:,lj)*invJ;
                end
            end
            MsL2Int(:,ig)=gw(ig).*detk.*(MsL2ext-MsL2tmp)...
                .*(MsL2ext-MsL2tmp);
            ExL2Int=ExL2Int+gw(ig).*detk.*MsL2ext.*MsL2ext;
            
            MsEntmp=MsEntmp.*MsEntmp;
            MsEnInt=MsEnInt+gw(ig).*detk.*(MsEntmp(:,1)+MsEntmp(:,2));
            exGrd=exGrd.*exGrd;
            ExEnInt=ExEnInt+perm.*gw(ig).*detk.*(exGrd(:,1)+exGrd(:,2));
        end
        
        MsL2err=sqrt(sum(sum(MsL2Int)));
%         MsL2err=norm(MsL2Int-ExL2Int,2);
        MsEnerr=sqrt(sum(MsEnInt));
        SolL2Norm=sqrt(sum(ExL2Int));
        SolEnNorm=sqrt(sum(ExEnInt));
        
        f=@(x,y) exactSol(x,y,exSolCase);
        RelMsL2err=MsL2err/squeeze(SolL2Norm);
        RelMsEnerr=MsEnerr/squeeze(SolEnNorm);
                
        MsL2ratio=1.0;
        MsEnratio=1.0;
        if(hf~=MsStart)
            MsL2ratio=MsL2errold./RelMsL2err;
            MsEnratio=MsEnerrold./RelMsEnerr;
        end
        MsL2errold=RelMsL2err;
        MsEnerrold=RelMsEnerr;
        if (hf~=MsStart)
            fprintf(formatSpec, 1/H, 1/h, RelMsEnerr,log2(MsEnratio),RelMsL2err,log2(MsL2ratio));
        end
        
        
        
%         fprintf("MsL2err is %f ratio is %f \n",MsL2err,log2(MsL2ratio));
%         fprintf("MsEnerr is %f ratio is %f \n",MsEnerr,log2(MsEnratio));
   end
       
    %% PLOTING
    if (isPlot==1)
        figure
        if(isMid==1)
            [rMx,rMy]=meshgrid(0.5*h:h:1-0.5*h,0:h:1);
            midSol=exactSol(rMx,rMy,exSolCase);
            
            eta=reshape(eta,totNum+1,totNum+1);
            colMid=eta(1:totNum,:)+eta(2:totNum+1,:);
            rowMid=eta(:,1:totNum)+eta(:,2:totNum+1);
            tmp=midSol-rowMid;
            
            subplot(2,3,1)       
            mesh(rMx,rMy,midSol);
            subplot(2,3,2)      
            mesh(rMx,rMy,rowMid);
            subplot(2,3,3)      
            mesh(rMx,rMy,tmp);

            [rMsMx,rMsMy]=meshgrid(0.5*H:H:1-0.5*H,0:H:1);
            midMsSol=exactSol(rMsMx,rMsMy,exSolCase);
             
            MsEta=reshape(MsEta,macNum+1,macNum+1);
            rowMsMid=MsEta(:,1:macNum)+MsEta(:,2:macNum+1);
            Mstmp=midMsSol-rowMsMid;
%             sum(sum(abs(Mstmp)))
            
            subplot(2,3,4)       
            mesh(rMsMx,rMsMy,midMsSol);
            subplot(2,3,5)      
            mesh(rMsMx,rMsMy,rowMsMid);
            subplot(2,3,6)      
            mesh(rMsMx,rMsMy,Mstmp);
            
        else
            sol=exactSol(node.meshX,node.meshY);
            eta=reshape(eta,totNum+1,totNum+1);
            V=zeros(totNum+3);
            V(2:totNum+2,2:totNum+2)=eta;
            T=1.5*V(2:totNum+2,2:totNum+2)+...%itself
                0.5*(V(2:totNum+2,1:totNum+1)+V(2:totNum+2,3:totNum+3)+...
                V(1:totNum+1,2:totNum+2)+V(3:totNum+3,2:totNum+2))-...%adjacent
                0.5*(V(1:totNum+1,1:totNum+1)+V(1:totNum+1,3:totNum+3)+...
                V(3:totNum+3,1:totNum+1)+V(3:totNum+3,3:totNum+3)); %diagonal

            tmp=sol-T;
            subplot(1,3,1)       
            mesh(node.meshX,node.meshY,sol);
            subplot(1,3,2)      
            mesh(node.meshX,node.meshY,T);
            subplot(1,3,3)      
            mesh(node.meshX,node.meshY,tmp);
        end
        temp=['fig',num2str(hf),'.png']; saveas(gca,temp); 
    end
end
