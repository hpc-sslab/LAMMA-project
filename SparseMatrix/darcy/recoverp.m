function[p] = recoverp(roughu,node,elem,pde,bdFlag,option)%1 dim uI on edges
  %%  solve u only, roughu coefficients in step 1
%
%  The velocity u = d*grad(p) is approximated using the lowest order
%  Raviart-Thomas element 
%
%
% u= grad p ==> (p, div v)  =-(u, v) 
%

% Copyright (C) Long Chen. See COPYRIGHT.txt for details.
% added by I. Kim

if ~exist('option','var'), option = []; end

%% Diffusion coefficient
if ~isfield(pde,'K'), pde.K = []; end
if isfield(pde,'K') && ~isempty(pde.K)
   if isnumeric(pde.K)
      K = pde.K;                   % K is an array
   else                            % K is a function
      center = (node(elem(:,1),:) + node(elem(:,2),:) + node(elem(:,3),:))/3;
      K = pde.K(center);  % take inverse sequencil.             
   end
else
    K = [];
end


% elemold = elem;
[elem,bdFlag] = sortelem(elem,bdFlag);  % ascend ordering
[elem2edge,edge] = dofedge(elem);
NT = size(elem,1);
NE = size(edge,1); 
Ndof=NE;
%% Compute geometric quantities and gradient of local basis
[Dlambda,area,elemSign] = gradbasis(node,elem);

% M. Mass matrix for RT0 element -->size     M = sparse(Ndof,Ndof) 
M = getmassmatvec(elem2edge,area,Dlambda,'RT0',K);
size(M);

% C. negative divergence operator
C = icdmat(double(elem2edge),elemSign*[1 -1 1]);
C = C';

%recover pressure
p = zeros(NT,1);
pF = zeros(Ndof,1);
rhsu = zeros(Ndof,1);

    %% No boundary conditions
    if ~isfield(pde,'g_D'), pde.g_D = []; end
    if ~isfield(pde,'g_N'), pde.g_N = []; end

    %% Set up bdFlag
    if isempty(bdFlag) % no bdFlag information
       if ~isempty(pde.g_N) % case: Neumann
           bdFlag = setboundary(node,elem,'Neumann');
       elseif ~isempty(pde.g_D) % case: Dirichlet
           bdFlag = setboundary(node,elem,'Dirichlet');
       end
    end

    %% Find Dirichlet and Neumann dofs 
    if ~isempty(bdFlag)
        isDirichlet(elem2edge(bdFlag(:)==1)) = true;
        isNeumann(elem2edge(bdFlag(:)==2)) = true;
        % Direction of boundary edges may not be the outwards normal
        % direction of the domain. edgeSign is introduced to record this
        % inconsistency.
        edgeSign = ones(NE,1);
        idx = (bdFlag(:,1) ~= 0) & (elemSign == -1);% first edge is on boundary
        edgeSign(elem2edge(idx,1)) = -1;
        idx = (bdFlag(:,2) ~= 0) & (elemSign == 1); % second edge is on boundary
        edgeSign(elem2edge(idx,2)) = -1;
        idx = (bdFlag(:,3) ~= 0) & (elemSign == -1);% third edge is on boundary
        edgeSign(elem2edge(idx,3)) = -1;
    end
    Dirichlet = edge(isDirichlet,:);
    Neumann = edge(isNeumann,:); 
    isBdDof = false(Ndof,1); 
    isBdDof(isNeumann) = true;   % for mixed method, Neumann edges are fixed
    freeDof = find(~isBdDof);
    isFreeEdge = true(NE,1);
    isFreeEdge(isNeumann) = false;
    freeEdge = find(isFreeEdge);

% DIRICHLET
    if ~isempty(pde.g_D) && isnumeric(pde.g_D) && (pde.g_D==0)
        pde.g_D = [];
    end
     if ~isempty(pde.g_D) && any(isDirichlet) 
         if ~isfield(option,'gNquadorder')
         option.gNquadorder = 2;   % default order exact for linear gN
         end
        [lambda,weight] = quadpts1(option.gNquadorder);
        nQuad = size(lambda,1);
        ve = node(Dirichlet(:,1),:)-node(Dirichlet(:,2),:);
        edgeLength = sqrt(sum(ve.^2,2)); 
        for ip = 1:nQuad
        	pxy = lambda(ip,1)*node(Dirichlet(:,1),:)+...
                  lambda(ip,2)*node(Dirichlet(:,2),:);               
            
              rhsu(isDirichlet) = rhsu(isDirichlet) + weight(ip)*pde.g_D(pxy);
              
        end
        rhsu(isDirichlet) = rhsu(isDirichlet).*edgeSign(isDirichlet);

        pF(isDirichlet)  = pF(isDirichlet) + rhsu(isDirichlet);
     end

    %Neumann
    if ~isempty(pde.g_N) && any(isNeumann)
        mid = 1/2*(node(Neumann(:,1),:)+node(Neumann(:,2),:));
        ve = node(Neumann(:,1),:)-node(Neumann(:,2),:);
        edgeLength = sqrt(sum(ve.^2,2)); 
        ne = [ve(:,2) -ve(:,1)]; % rotation of tangential vector
        if isnumeric(pde.g_N)
            evalg_N = pde.g_N;
        else
            evalg_N = pde.g_N(mid,1);
        end
        roughu(isNeumann) = edgeLength.*evalg_N;
        roughu(isNeumann) = roughu(isNeumann).*edgeSign(isNeumann);
        pF(isNeumann) = roughu(isNeumann);
    end

    if ~any(isNeumann) && any(isDirichlet)
        pF = pF - M*roughu;
        p=C\pF;
    end
    
    if ~any(isDirichlet) && any(isNeumann)
        pF = pF - M*roughu;
%        pF = - M*roughu;
% C*p=pF-> C dimension Ndof*NT
        p=C(freeDof,:)\pF(freeDof);
        pbar = sum(p.*area)/sum(area);
        p=p-pbar;
        %psum = sum(p);
    end
end



