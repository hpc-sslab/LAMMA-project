function uI = faceinterpolate(u,node,elem,elemType,pde)
%% FACEINTERPOLATE interpolate to face elements (RT0 or BDM1).
%
% uI = faceinterpolate(u,node,elem,elemType) interpolates a given function u
% into the lowesr order RT0 or BDM1 finite element spaces. The coefficient
% is given by the line integral int_e u*n ds. The input elemType can be 'RT0'
% or 'BDM1'.
%
% Example
%
%    maxIt = 5;
%    node = [1,0; 1,1; 0,1; -1,1; -1,0; -1,-1; 0,-1; 0,0]; % nodes
%    elem = [1,2,8; 3,8,2; 8,3,5; 4,5,3; 7,8,6; 5,6,8];    % elements
%    bdEdge = setboundary(node,elem,'Dirichlet');
%    pde = mixBCdata;
%    err = zeros(maxIt,2); N = zeros(maxIt,1);
%    for i =1:maxIt
%      [node,elem,bdEdge] = uniformrefine(node,elem,bdEdge);
%      [u,sigma,M] = PoissonRT0(node,elem,pde,bdEdge);
%      sigmaI = faceinterpolate(pde.Du,node,elem,'RT0');
%      err(i,1) = getL2errorRT0(node,elem,pde.Du,sigmaI,pde.d);
%      err(i,2)=sqrt((sigma-sigmaI)'*M*(sigma-sigmaI));
%      N(i) = size(u,1);
%    end
%     figure;
%     showrate2(N,err(:,1),2,'r-+','||Du - \sigma_I||',...
%               N,err(:,2),2,'b-+','||\sigma^{RT_0} - \sigma_I||');
%
% See also edgeinterpolate, edgeinterpolate1, edgeinterpolate2
%
% Created by Ming Wang at Mar 29, 2011, M-lint modified at May 14, 2011.
%
% Copyright (C) Long Chen. See COPYRIGHT.txt for details.

if ~exist('elemType','var'), elemType = 'RT0'; end
%% edge 
if size(elem,2) == 2 % the input elem is edge
    edge = elem;
else
    [tempvar,edge] = dofedge(elem);
end
NE = size(edge,1);
edgeVec = node(edge(:,2),:) - node(edge(:,1),:);
nVec = zeros(NE,2);
nVec(:,1) = edgeVec(:,2); 
nVec(:,2) = -edgeVec(:,1);

%% dof for RT0
[lambda,weight] = quadpts1(4);
nQuad = size(lambda,1);
uI = zeros(NE,1);
%% Diffusion coefficient
if ~isfield(pde,'d'), pde.d = []; end
if isfield(pde,'d') && ~isempty(pde.d)
   if isnumeric(pde.d)
      K = pde.d;                   % d is an array
   else                            % d is a function
      center = (node(elem(:,1),:) + node(elem(:,2),:) + node(elem(:,3),:))/3;
      K = pde.d(center);  % take inverse sequencil.             
   end
else
    K = [];
end
%detK=K(:,1)*K(:,2)-K(:,3)^2;
for i = 1:nQuad
    pxy = lambda(i,1)*node(edge(:,1),:)+lambda(i,2)*node(edge(:,2),:);
    flux = u(pxy);
 %   flux(:,1) = K(:,2).*flux(:,1) - K(:,3).*flux(:,2);
 %   flux(:,2) = -K(:,3).*flux(:,1) + K(:,1).*flux(:,2);
 %   flux =flux/detK;
 %   flux(:,1) = K(:,2).*flux(:,1) - K(:,3).*flux(:,2);
 %   flux(:,2) = -K(:,3).*flux(:,1) + K(:,1).*flux(:,2);
    uI = uI + weight(i)*dot(flux,nVec,2);   
end

%% dof for BDM1
if strcmp(elemType,'BDM1')
   uI(NE+1:2*NE) = zeros(NE,1);
   for i = 1:nQuad
        pxy = lambda(i,1)*node(edge(:,1),:)+lambda(i,2)*node(edge(:,2),:);
        flux = u(pxy);
        uI(NE+1:2*NE) = uI(NE+1:2*NE)+ ...
                     weight(i)*3*(lambda(i,1)-lambda(i,2))*dot(flux,nVec,2); 
   end
end
