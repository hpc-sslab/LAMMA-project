function LshapeimphybridDarcy
%% LSHAPE hybrid Darcy Problem
%
% LSHAPE solves the Poisson equation $-\Delta u =f$ in $\Omega$ and $u =
% g_D$ on $\partial \Omega$ in a crack domain $\Omega=(-1,1)^2\backslash
% (0,1)\times (-1,0)$
%  using adaptive finite element method (AFEM). We choose f and g_D such
%  that the exact solution is $u = r^{\beta}\sin(\beta\theta), \beta = 2/3$
%  in the polar coordinate.
%
% EXAMPLE
%    Lshape 
%
close all; 
%% Parameters
maxN = 3e6;     theta = 0.5;    maxIt = 7;
N = zeros(maxIt,1);  h = zeros(maxIt,1);  energy = zeros(maxIt,1);  uIuhErrH1 = zeros(maxIt,1);
option=[];

%%  Generate an initial mesh
[node,elem] = squaremesh([-1,1,-1,1],0.5);
[node,elem] = delmesh(node,elem,'x>0 & y<0');
%bdFlag = setboundary(node,elem,'Dirichlet');
bdFlag = setboundary(node,elem,'Neumann');

elemType ='RT0'
%elemType ='BDM1'
%% Set up PDE data
pde.f = 0;
pde.g_D = @exactp;
pde.exactp = @exactp;
pde.Du=@Du;% used for recoverFlux;
pde.d=[];
pde.g_N = @g_N;
pde.exactu=@Du;

%%  Adaptive Finite Element Method
% *SOLVE* -> *ESTIMATE* -> *MARK* -> *REFINE*

%        for exact solution graph
%figure;
% pos(:,1)=node(:,1);
% pos(:,2)=node(:,2);
% r = sqrt(sum(pos.^2,2));
% theta = atan2(pos(:,2),pos(:,1));
% theta = (theta>=0).*theta + (theta<0).*(theta+2*pi);
% ftp = inline('r.^(2/3).*sin(2*theta/3)-0.5334');
% ftp =ftp(pos(:,1),pos(:,2));
% showsolution(pos,elem,ftp);
% 

for k = 1:maxIt
   %figure;
   %showmesh(node,elem);
    % Step 1: SOLVE
    h(k) = 1./(sqrt(size(node,1))-1);
    switch elemType
        case 'RT0'  % RT0 hybrid FEM 
	    hybrid_parameter = 1./(sqrt(size(node,1))-1)^(4/3);%Lshape            
        %hybrid_parameter = 3;
        [roughu,eqn,info] = DarcyRT0hybrid(node,elem,pde,bdFlag,option,hybrid_parameter);
         for improve =1:5 %maximp
        [u,eqn,info] = DarcyRT0hybridimprove(roughu,node,elem,pde,bdFlag,option,hybrid_parameter);
        roughu=u;
        end
        p=recoverp(u,node,elem,pde,bdFlag,option);
        case 'BDM1'
         hybrid_parameter = 1./(sqrt(size(node,1))-1)^(10/3);%Lshape           
        % hybrid_parameter = 3;%Lshape           
        [roughu,eqn,info] = DarcyBDM1hybrid(node,elem,pde,bdFlag,option,hybrid_parameter);
        for improve =1:10 %maximp
        [u,eqn,info] = DarcyBDM1hybridimprove(roughu,node,elem,pde,bdFlag,option,hybrid_parameter);
        roughu=u;
        end
        p=recoverBDM1p(u,node,elem,pde,bdFlag,option);
    end
    % compute error
    t = cputime;
    
    if strcmp(elemType,'RT0')
    erruL2(k) = getL2errorRT0(node,elem,pde.exactu,u,[]);% Set u=Kgrad p
    erruHdiv(k) = getHdiverrorRT0(node,elem,pde.f,-u,[]);
    uII = faceinterpolate(pde.exactu,node,elem,'RT0',pde);
    %uI=pde.exactu(node);
    else
    erruL2(k) = getL2errorBDM1(node,elem,pde.exactu,u,[]);% Set u=Kgrad p
    erruHdiv(k) = getHdiverrorBDM1(node,elem,pde.f,-u,[]);
    uII = faceinterpolate(pde.exactu,node,elem,'BDM1',pde);
    %uI=pde.exactu(node);
    end
    erruhybrid(k) = sqrt(erruHdiv(k).^2 + h(k).*erruL2(k).^2);
    erruIuh(k)=sqrt((u-uII)'*eqn.A*(u-uII));
%    erruIuh(k)=sqrt((u-uI)'*eqn.A*(u-uI));

    errpL2(k) = getL2error(node,elem,pde.exactp,p);
    % interpolation 
    pI = Lagrangeinterpolate(pde.exactp,node,elem,'P0');
    area = simplexvolume(node,elem);
    pIbar = sum(pI.*area)/sum(area);
    pI = pI - pIbar;
    errpIphL2(k) = sqrt(dot((pI-p).^2,area));
    
% Plot mesh and solution
        figure;  
        subplot(1,2,1);showsolution(node,elem,p);
        subplot(1,2,2);showsolution(node,elem,pI);

    figure;  showsolutionRT(node,elem,u,[-50,12]);    
    figure;  showsolutionRT(node,elem,uII,[-50,12]);    

    % Record error and number of vertices
    N(k) = size(node,1);
    %if (N(k)>maxN), break; end        


    %  REFINE
    [node,elem,bdFlag] = uniformrefine(node,elem,bdFlag);
end

%% Plot convergence rates

    figure;
    set(gcf,'Units','normal'); 
    set(gcf,'Position',[0.25,0.25,0.55,0.4]);
    subplot(1,2,1)
    showrateh2(h(1:k),errpIphL2(1:k)',1,'-*','||p_I-p_h||',...
               h(1:k),errpL2(1:k)',1,'k-+','||p-p_h||');
    subplot(1,2,2)
    showrateh3(h(1:k),erruHdiv(1:k)',1,'-X','||div(u - u_h)||',... 
               h(1:k),erruL2(1:k)',1,'k-*','|| u - u_h||',...
	       h(1:k),erruhybrid(1:k)',1,'m-O','|| div(u-u_h)|| + sq-delta* ||u- u_h||');                %h(1:k),erruIuh(1:k),1,'m-O','|| uI-uh||');

    
    pL2rate=0;
    pIphL2rate = 0;
    L2rate = 0;
    uIuhrate = 0;
    Hdivrate=0;
    sumrate=0.0;           
    for k=2:maxIt
          pL2rate(k) = log2(errpL2(k-1)./errpL2(k));
          pIphL2rate(k) = log2(errpIphL2(k-1)./errpIphL2(k));
          L2rate(k) = log2(erruL2(k-1)./erruL2(k));
          uIuhrate(k) = log2(erruIuh(k-1)./erruIuh(k));
          Hdivrate(k) = log2(erruHdiv(k-1)./erruHdiv(k));
          sumrate(k) = log2(erruhybrid(k-1)./erruhybrid(k));
    end


pL2rate = pL2rate';
pIphL2rate = pIphL2rate';
L2rate = L2rate';
uIuhrate = uIuhrate';
Hdivrate=Hdivrate';
sumrate=sumrate';
%% Output
err = struct('h',h(1:k),'N',N(1:k),'pL2',errpL2(1:k)','pIphL2',errpIphL2(1:k)',...
             'uL2',erruL2(1:k),'uHdiv',erruHdiv(1:k),...
             'uIuL2',erruIuh(1:k),'uhybrid',erruhybrid(1:k));
%time = struct('N',N,'err',errTime(1:k),'solver',solverTime(1:k), ...
%              'assemble',assembleTime(1:k),'mesh',meshTime(1:k));
%solver = struct('N',N(1:k),'itStep',itStep(1:k),'time',solverTime(1:k),...
%                'stopErr',stopErr(1:k),'flag',flag(1:k));
            
%% Display error and CPU time
display('Table: Error')
colname = {'#Dof','h','||p-p_h||','rate','||p_I-p_h||','rate','||u-u_h||','rate','||u-u_h||_{div}','rate','||uI-u_h||','rate','sqrt(||u-u_h||^2_{div} + h||u-u_h||^2)','rate'};
displaytable(colname,err.N,[],err.h,'%0.2e', err.pL2, '%0.5e',pL2rate, '%0.4f',err.pIphL2, '%0.5e',pIphL2rate,'%0.4f',err.uL2','%0.5e',L2rate,'%0.4f',err.uHdiv','%0.5e',Hdivrate, '%0.4f',err.uIuL2','%0.5e',uIuhrate, '%0.4f',err.uhybrid','%0.5e',sumrate, '%0.4f');
                 
%display('Table: CPU time')
%colname = {'#Dof','Assemble','Solve','Error','Mesh'};
%disptable(colname,time.N,[],time.assemble,'%0.2e',time.solver,'%0.2e',...
%                  time.err,'%0.2e',time.mesh,'%0.2e');     
end % End of function LSHAPE


function u = exactp(p) % exact solution
r = sqrt(sum(p.^2,2));
theta = atan2(p(:,2),p(:,1));
theta = (theta>=0).*theta + (theta<0).*(theta+2*pi);
u = r.^(2/3).*sin(2*theta/3)-0.5334;
end

    function s = Du(p) % exact solution
    x = p(:,1); y = p(:,2);
    r = sqrt(sum(p.^2,2));
    theta = atan2(y,x);
    theta = (theta>=0).*theta + (theta<0).*(theta+2*pi);
    s(:,1) = 2/3*r.^(-1/3).*sin(2*theta/3).*x./r ...
                - 2/3*r.^(2/3).*cos(2*theta/3).*y./r.^2;
    s(:,2) = 2/3*r.^(-1/3).*sin(2*theta/3).*y./r ...
                + 2/3*r.^(2/3).*cos(2*theta/3).*x./r.^2;
    end



function f=g_N(p,vargin)
    
        f = zeros(size(p,1),1);

        x = p(:,1); y = p(:,2);
        Du=Du(p);
        uprime = [Du(:,1),  Du(:,2)];

        leftbd = (abs(x+1)<eps);  % n = (-1,0); 
        f(leftbd) = - uprime(leftbd,1);
        rightbd = (abs(x-1)<eps);% n = (1,0); 
        f(rightbd) = uprime(rightbd,1);
        rightbd = (abs(x) < eps); % n = (1,0); 
        f(rightbd) = uprime(rightbd,1);
        topbd = (abs(y-1)<eps);   % n = (0,1)
        f(topbd) = uprime(topbd,2);
        bottombd = (abs(y+1)<eps ); % n = (0,-1)
        f(bottombd) = - uprime(bottombd,2);    
        bottombd = (abs(y) < eps);% n = (0,-1)
        f(bottombd) = - uprime(bottombd,2);    
end
