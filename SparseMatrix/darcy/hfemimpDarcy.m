function [err,time,solver,eqn] = hfemimpDarcy(node,elem,pde,bdFlag,option,varargin)
%% HFEMDARCY solve Poisson equation by various hybrid finite element methods
%
%   HFEMDARCY computes approximations to the Darcy equation on a
%   sequence of meshes obtained by uniform refinement of an input mesh.
% 
%   
%   
%
% See also hfemPoisson
%
% 

%% Check input arguments
if nargin >=1 && ischar(node)
    option.elemType = node;
    clear node
end
if ~exist('node','var') || ~exist('elem','var')
    [node,elem] = squaremesh([0,1,0,1],0.25);  % default mesh is a square
end
if ~exist('option','var'), option = []; end
if ~exist('pde','var')
    pde = mixBCdata;                          % default data
end
if ~exist('bdFlag','var')
    bdFlag = setboundary(node,elem,'Dirichlet') ;
end

%% Parameters
[elemType,maxIt,maxN,L0,refType,option] = mfemoption(option);

%% Generate an initial mesh 
% for k = 1:L0
%     if strcmp(refType,'red')
%         [node,elem,bdFlag] = uniformrefine(node,elem,bdFlag);
%     elseif strcmp(refType,'bisect')
%         [node,elem,bdFlag] = uniformbisect(node,elem,bdFlag);
%     end
% end

%% Initialize err and time array
erruL2 = zeros(maxIt,1);   
erruHdiv = zeros(maxIt,1); 
erruIuh = zeros(maxIt,1); 
erruhybrid= zeros(maxIt,1); 
errTime = zeros(maxIt,1);      
solverTime = zeros(maxIt,1); 
assembleTime = zeros(maxIt,1); 
meshTime = zeros(maxIt,1); 
itStep = zeros(maxIt,1);  
stopErr = zeros(maxIt,1); 
flag = zeros(maxIt,1);
N = zeros(maxIt,1); 
h = zeros(maxIt,1);

%        for exact solution graph
%        figure;
%       K={1 1 1 100] 
%        ftx = inline('(3*x.^2.*y + pi*cos(pi*x).*cos(pi*y)) + (x.^3 + 4*y.^3 - pi*sin(pi*x).*sin(pi*y))');
 %       fty = inline('(3*x.^2.*y + pi*cos(pi*x).*cos(pi*y)) + 100*(x.^3 + 4*y.^3 - pi*sin(pi*x).*sin(pi*y))');
        %K={1 0 0 1] 
        %ftx = inline('(3*x.^2.*y + pi*cos(pi*x).*cos(pi*y))');
        %fty = inline('(x.^3 + 4*y.^3 - pi*sin(pi*x).*sin(pi*y))');
%        ftp = inline('x.^3.*y + y.^4 + sin(pi*x).*cos(pi*y)-13/40');
        %K=2*(2+sin(10*pi*x).*cos(10*pi*y)); %oscilltory coefficient data
%         ftx = inline('2*(2+sin(10*pi*x).*cos(10*pi*y)).*(3*x.^2.*y + pi*cos(pi*x).*cos(pi*y))');
%         fty = inline('2*(2+sin(10*pi*x).*cos(10*pi*y)).*(x.^3 + 4*y.^3 - pi*sin(pi*x).*sin(pi*y))');
%         [pos, elm]= squaremesh([0,1,0,1],1/2^5);
%         ftu=ftx(pos(:,1),pos(:,2));
%         ftv=fty(pos(:,1),pos(:,2));
%         ftp =ftp(pos(:,1),pos(:,2));
%         subplot(1,2,1);showsolution(pos,elm,ftu);
%         subplot(1,2,2);showsolution(pos,elm,ftv);
%         figure;
%         showsolution(pos,elm,ftp);

%% Finite Element Method        
for k = 1:maxIt
    % solve the equation
    h(k) = 1./(sqrt(size(node,1))-1);
    
    switch elemType
        case 'RT0'  % RT0 hybrid FEM
           %hybrid_parameter = 1./(sqrt(size(node,1))-1)^2;%mesh size^2
           hybrid_parameter = 3.;
           [roughu,eqn,info] = DarcyRT0hybrid(node,elem,pde,bdFlag,option,hybrid_parameter);%rough-1dim coefficients
           for improve =1:5 %maximp
           [u,eqn,info] = DarcyRT0hybridimprove(roughu,node,elem,pde,bdFlag,option,hybrid_parameter);
           roughu=u;
           end
%     u=roughu;
           p=recoverp(u,node,elem,pde,bdFlag,option);

        case 'BDM1'
            %hybrid_parameter = 1./(sqrt(size(node,1))-1)^4;%mesh size^4
            hybrid_parameter = 3.;
%            [u,eqn,info] = DarcyBDM1hybrid(node,elem,pde,bdFlag,option);
           [roughu,eqn,info] = DarcyBDM1hybrid(node,elem,pde,bdFlag,option,hybrid_parameter);
           for improve =1:10 %maximp
           [u,eqn,info] = DarcyBDM1hybridimprove(roughu,node,elem,pde,bdFlag,option,hybrid_parameter);
           roughu=u;
           end
           p=recoverBDM1p(u,node,elem,pde,bdFlag,option);
    end
    % compute error
    t = cputime;
    
        
    if isfield(pde,'exactu') && isfield(pde,'f')
        if strcmp(elemType,'RT0')
            erruL2(k) = getL2errorRT0(node,elem,pde.exactu,u,[]);
            erruHdiv(k) = getHdiverrorRT0(node,elem,pde.f,-u,[]);
            uI = faceinterpolate(pde.exactu,node,elem,'RT0',pde);
        else
            erruL2(k) = getL2errorBDM1(node,elem,pde.exactu,u,[]);% Set u=Kgrad p
            erruHdiv(k) = getHdiverrorBDM1(node,elem,pde.f,-u,[]);
            uI = faceinterpolate(pde.exactu,node,elem,'BDM1',pde);
        end
	erruhybrid(k) = sqrt(erruHdiv(k).^2 + hybrid_parameter*erruL2(k).^2);
        erruIuh(k)=sqrt((u-uI)'*eqn.A*(u-uI));
    end

    if isfield(pde,'exactp')
        errpL2(k) = getL2error(node,elem,pde.exactp,p);
        area = simplexvolume(node,elem);
        pbar = sum(p.*area)/sum(area);
        p=p-pbar;
        % interpolation
        pI = Lagrangeinterpolate(pde.exactp,node,elem,'P0');
        pIbar = sum(pI.*area)/sum(area);
        pI = pI - pIbar;
        errpIphL2(k) = sqrt(dot((pI-p).^2,area));
        errpIpL2(k) = getL2error(node,elem,pde.exactp,pI);
    end

    errTime(k) = cputime - t;
   % record time
    solverTime(k) = info.solverTime;
    assembleTime(k) = info.assembleTime;
    if option.printlevel>1
        fprintf('Time to compute the error %4.2g s \n erruHdiv err %4.2g    L2err %4.2g \n',...
            errTime(k),erruHdiv(k), erruL2(k));
    end
    % record solver information
    itStep(k) = info.itStep;
    stopErr(k) = info.stopErr;
    flag(k) = info.flag;
 
   % plot 
     N(k) = length(u);
     h(k) = 1./(sqrt(size(node,1))-1);
%in hybrid current version, dont interested in finding pressure
%     if option.plotflag && N(k) > 2e3 % show mesh and solution for small size
        figure;  
        subplot(1,2,1);showsolution(node,elem,p);
        subplot(1,2,2);showsolution(node,elem,pI);

        figure;
        showsolutionRT(node,elem,u);
        figure;
        showsolutionRT(node,elem,uI);
%end
    if N(k) > maxN
        break;
    end

    % refine mesh
    t = cputime;
    if strcmp(refType,'red')
        [node,elem,bdFlag] = uniformrefine(node,elem,bdFlag);
    elseif strcmp(refType,'bisect')
        [node,elem,bdFlag] = uniformbisect(node,elem,bdFlag);
    end

    if isfield(pde,'K') && isnumeric(pde.K)
        pde.K = repmat(pde.K,4,1); % prolongate to the fine grid
    end
    meshTime(k) = cputime - t;
end

%% Plot convergence rates
if option.rateflag
    figure;
    set(gcf,'Units','normal'); 
    set(gcf,'Position',[0.25,0.25,0.55,0.4]);
    subplot(1,2,1)
    showrateh2(h(1:k),errpIphL2(1:k)',1,'-*','||p_I-p_h||',...
               h(1:k),errpL2(1:k)',1,'k-+','||p-p_h||');
    subplot(1,2,2)
    showrateh3(h(1:k),erruHdiv(1:k),1,'-X','||div(u - u_h)||',... 
               h(1:k),erruL2(1:k),1,'k-*','|| u - u_h||',...
               h(1:k),erruIuh(1:k),1,'m-O','|| uI-uh||');
               %h(1:k),erruhybrid(1:k),1,'m-O','|| div(u-u_h)|| + sq-delta* ||u- u_h||');
    
    pL2rate=0;
    pIphL2rate = 0;
    pIpL2rate = 0;
    L2rate = 0;
    uIuhrate = 0;
    Hdivrate=0;
    sumrate=0.0;           
    for k=2:maxIt
          pL2rate(k) = log2(errpL2(k-1)./errpL2(k));
          pIphL2rate(k) = log2(errpIphL2(k-1)./errpIphL2(k));
          pIpL2rate(k) = log2(errpIpL2(k-1)./errpIpL2(k));
          L2rate(k) = log2(erruL2(k-1)./erruL2(k));
          uIuhrate(k) = log2(erruIuh(k-1)./erruIuh(k));
          Hdivrate(k) = log2(erruHdiv(k-1)./erruHdiv(k));
          sumrate(k) = log2(erruhybrid(k-1)./erruhybrid(k));
    end
end

pL2rate = pL2rate';
pIphL2rate = pIphL2rate';
pIpL2rate = pIpL2rate';
L2rate = L2rate';
uIuhrate = uIuhrate';
Hdivrate=Hdivrate';
sumrate=sumrate';
%% Output
err = struct('h',h(1:k),'N',N(1:k),'pL2',errpL2(1:k)','pIphL2',errpIphL2(1:k)','pIpL2',errpIpL2(1:k)',...
             'uL2',erruL2(1:k),'uHdiv',erruHdiv(1:k),...
             'uIuL2',erruIuh(1:k),'uhybrid',erruhybrid(1:k));
time = struct('N',N,'err',errTime(1:k),'solver',solverTime(1:k), ...
              'assemble',assembleTime(1:k),'mesh',meshTime(1:k));
solver = struct('N',N(1:k),'itStep',itStep(1:k),'time',solverTime(1:k),...
                'stopErr',stopErr(1:k),'flag',flag(1:k));
            
%% Display error and CPU time
display('Table: Error')
colname = {'#Dof','h','||p-p_h||','rate','||p_I-p_h||','rate','||p_I-p||','rate','||u-u_h||','rate','||u-u_h||_{div}','rate','||uI-u_h||','rate','sqrt(||u-u_h||^2_{div} + h||u-u_h||^2)','rate'};
displaytable(colname,err.N,[],err.h,'%0.2e', err.pL2, '%0.5e',pL2rate, '%0.4f',err.pIphL2, '%0.5e',pIphL2rate,'%0.4f',err.pIpL2, '%0.5e',pIpL2rate,'%0.4f',err.uL2,'%0.5e',L2rate,'%0.4f',err.uHdiv,'%0.5e',Hdivrate, '%0.4f',err.uIuL2,'%0.5e',uIuhrate, '%0.4f',err.uhybrid,'%0.5e',sumrate, '%0.4f');
                 
display('Table: CPU time')
colname = {'#Dof','Assemble','Solve','Error','Mesh'};
disptable(colname,time.N,[],time.assemble,'%0.2e',time.solver,'%0.2e',...
                  time.err,'%0.2e',time.mesh,'%0.2e');     
end
