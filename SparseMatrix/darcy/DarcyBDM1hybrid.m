function [u,eqn,info] = DarcyBDM1hybrid(node,elem,pde,bdFlag,option,hybrid_parameter)
%%  solve u only
%  [u,eqn,info] = DarcyBDM1hybrid(node,elem,pde,bdFlag) produces an approximation of
%  the Poisson equation 
%
%       -div(d*grad(p))=f  in \Omega, with 
%       Dirichlet boundary condition p=g_D on \Gamma_D, 
%       Neumann boundary condition   d*grad(p)*n=g_N on \Gamma_N
%
%  The velocity u = d*grad(p) is approximated using the BDM1
%
%
% (div u, div v) + hybrid_parameter*(d^{-1} u,v) = -(f, div v)
%
%  
%
% Copyright (C) Long Chen. See COPYRIGHT.txt for details.
% added by I. Kim

if ~exist('option','var'), option = []; end

time = cputime;  % record assembling time
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

%% Data structure B matrices

%% Assemble matrix 

[elem2dof,edge,elem2edgeSign] = dofedge(elem); %unsorted yet
NT = size(elem,1);
NE = size(edge,1); 
Ndof=2*NE;

% compute div phi
[Dlambda,area] = gradbasis(node,elem); % Dlambda = grad lambda
% div phi = 2*(Dlambda_i,Rot_j);
% for phi_i=lambda_j rot_k - lambda_k rot_j
% rotation matrix for computing rotLambda.
rotMat = [0 -1; 1 0]; % rotation matrix for computing rotLambda.
divPhi(:,3) = 2*dot(Dlambda(:,:,1),Dlambda(:,:,2)*rotMat,2);
divPhi(:,1) = 2*dot(Dlambda(:,:,2),Dlambda(:,:,3)*rotMat,2);
divPhi(:,2) = 2*dot(Dlambda(:,:,3),Dlambda(:,:,1)*rotMat,2);

%for psi_i=lambda_j rot_k + lambda_k rot_j
% div psi_i= 0;
divPsi=zeros(NT,3);

B = sparse(Ndof,Ndof);
%localEdge = [2 3; 1 3; 1 2]; % ascend ordering
localEdge = [2 3; 3 1; 1 2]; % ccwise ordering

for i = 1:3
    for j = i:3
        %local to global index map 
        ii = double(elem2dof(:,i)); % global edge indx of local ith edge indx
        jj = double(elem2dof(:,j));
        
        i1 = localEdge(i,1); i2 = localEdge(i,2); % [i1,i2] is the edge opposite to vertex i.
        j1 = localEdge(j,1); j2 = localEdge(j,2);
        
       
        % computation of B matrix, note that (div u, div v) 
        % (div phi_i, div phi_j)
        Bij= area.*elem2edgeSign(:,i).*divPhi(:,i).*elem2edgeSign(:,j).*divPhi(:,j);
        
        if (j==i)
            B = B + sparse(ii,jj,Bij,Ndof,Ndof);
        else
	    B = B + sparse([ii,jj],[jj,ii],[Bij; Bij],Ndof,Ndof);        
	end
        
        % (div psi_i, div psi_j)
        Bij= area.*elem2edgeSign(:,i).*divPsi(:,i).*elem2edgeSign(:,j).*divPsi(:,j);
        if (j==i)
            B = B + sparse(ii+NE,jj+NE,Bij,Ndof,Ndof);
        else
	    B = B + sparse([ii,jj]+NE,[jj,ii]+NE,[Bij; Bij],Ndof,Ndof);        
        end
    end
end

for i = 1:3
        for j = 1:3
            % local to global index map and its sign
            ii = double(elem2dof(:,i));
            jj = double(elem2dof(:,j));
            i1 = localEdge(i,1); i2 = localEdge(i,2);
            j1 = localEdge(j,1); j2 = localEdge(j,2);
         
            % (div psi_i,div phi_j)
            Bij= area.*elem2edgeSign(:,i).*divPhi(:,i).*elem2edgeSign(:,j).*divPsi(:,j);
            B = B + sparse([ii+NE;jj],[jj;ii+NE],[Bij; Bij],2*NE,2*NE);
        end
end
    
%sort is needed
elemold = elem;
[elem,bdFlag] = sortelem(elem,bdFlag);  % ascend ordering
[elem2edge,edge] = dofedge(elem);

%% Compute geometric quantities and gradient of local basis
[Dlambda,area,elemSign] = gradbasis(node,elem);

% M. Mass matrix for BDM1 element -->size     M = sparse(Ndof,Ndof) 
M = getmassmatvec(elem2edge,area,Dlambda,'BDM1',K);
size(M);

A = B+hybrid_parameter.*M;

%% Assemble right hand side. 
% ((f, div(phi_j) )  for first NE, (f, div(psi_j)) for NE for BDM1 element

%locEdge = [2,3; 1 3; 1,2]; % ascend ordering
fu = zeros(Ndof,1);
tmpsign = zeros(NT,3);
tmpsign(:,:)=elemSign*[1 -1 1];

if ~isfield(pde,'f') || (isreal(pde.f) && (pde.f==0))
    pde.f = [];
end
if ~isfield(option,'fquadorder')
    option.fquadorder = 2;   % default order
end

if ~isempty(pde.f)
    elem2dofu= [elem2edge NE+elem2edge];

    [lambda,weight] = quadpts(option.fquadorder);   

    nQuad = size(lambda,1); 

    bt = zeros(NT,6);  
    for k = 1:nQuad
	% quadrature points in the x-y coordinate
            pxy = lambda(k,1)*node(elem(:,1),:) ...
                + lambda(k,2)*node(elem(:,2),:) ...
                + lambda(k,3)*node(elem(:,3),:);

		fp = pde.f(pxy);

      %(f, divPhi)
        for p=1:3

            bt(:,p) =  bt(:,p) - weight(k)*dot(tmpsign(:,p).*divPhi(:,p),fp,2);
            bt(:,3+p) =  bt(:,3+p) - weight(k)*dot(tmpsign(:,p).*divPsi(:,p),fp,2);
            end
         end
     bt = bt.*repmat(area,1,6);
%    fu = accumarray(elem2edge(:),bt(:),[Ndof 1]);
     fu = accumarray(elem2dofu(:),bt(:),[Ndof 1]);
end
clear pxy fp bt rhs tmpsign 
F(1:Ndof,1) = fu;

if ~exist('bdFlag','var'), bdFlag = []; end
[AD,F,u,freeEdge,isPureNeumannBC] = getbdhyBDM1(F,hybrid_parameter);

eqn = struct('A',AD,'f',F,'freeEdge',freeEdge);


%% Record assembling time
assembleTime = cputime - time;
if ~isfield(option,'printlevel'), option.printlevel = 1; end
if option.printlevel >= 2
    fprintf('Time to assemble matrix equation %4.2g s\n',assembleTime);
end

option.solver = 'none';
%% Solve the system of linear equations
if isempty(freeEdge), return; end
% Set up solver type
%if isempty(option) || ~isfield(option,'solver')    % no option.solver
    if Ndof <= 2e3  % Direct solver for small size systems
        option.solver = 'direct';
    else            % MGCG  solver for large size systems
        option.solver = 'amg';
    end
%end

option.solver = 'direct';
solver = option.solver;

% solve
switch solver
    case 'direct'
     t = cputime;
     u(freeEdge) = AD(freeEdge,freeEdge)\F(freeEdge);       
     %u(freeEdge) = GaussPP(AD(freeEdge,freeEdge),F(freeEdge));       
     %nu=u(1:NE); 
    residual = norm(F - AD*u);
    info = struct('solverTime',cputime - t,'itStep',1,'error',residual,'flag',2,'stopErr',residual);
    case 'none'
        info = struct('solverTime',[],'itStep',0,'error',[],'flag',3,'stopErr',[]);
    case 'mg'
        option.x0 = u;
        option.solver = 'CG';
        [u,info] = mg(AD,F,elem,option,edge);
    case 'amg'
        option.solver = 'GMRES';
        [u(freeEdge),info] = amg(AD(freeEdge,freeEdge),F(freeEdge),option);                 
end

%% Output information
info.assembleTime = assembleTime; 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% subfunction getbdRT0 from Darcy's
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [AD,F,u,freeDof,isPureNeumannBC] = getbdhyBDM1(F,hybrid_parameter)

%% GETBDBDM1 Boundary conditions for Poisson equation: BDM1 element.
    %
    %  Created by Ming Wang. Improved the check of edgeSign by Long Chen.

    u = zeros(Ndof,1);
    
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
    idpsi = NE + (1:NE);
    isBdDof = false(Ndof,1); 
    isBdDof(isNeumann) = true;   % for mixed method, 1st Neumann edges are fixed
    isBdDof(idpsi(isNeumann)) = true;   % for mixed method, 2nd Neumann edges are fixed
    freeDof = find(~isBdDof);
    size(freeDof,1)

%    isFreeEdge = true(NE,1);
%    isFreeEdge(isNeumann) = false;
%    freeEdge = find(isFreeEdge);
    
    %% Dirichlet boundary condition (Neumann BC in mixed form)
    %   We need only modify the rhs on dof associated with Dirichlet
    %   boundary. Compute the int_e \Phi\cdot n g_D and  <\psi\cdot n,g_D>=1/|e_{i,j}|\int e_{i,j}(\lambda_j-\lambda_i)g_Dds on the boundary 
    %   using quadrature rules.
    
    if ~isempty(pde.g_D) && isnumeric(pde.g_D) && (pde.g_D==0)
        pde.g_D = [];
    end

    if ~isempty(pde.g_D) && (any(isDirichlet))
        [lambda,weight] = quadpts1(4);
         nQuad = size(lambda,1);
         % <\phi \cdot n, g_D> = 1/|e_{i,j}|\int e_{i,j} g_D ds
         % <\psi\cdot n,g_D>=1/|e_{i,j}|\int e_{i,j}(\lambda_j-\lambda_i)g_Dds
         for ip = 1:nQuad
             pxy = lambda(ip,1)*node(Dirichlet(:,1),:)+lambda(ip,2)*node(Dirichlet(:,2),:);
             u(isDirichlet) = u(isDirichlet) + weight(ip)*pde.g_D(pxy);
             u(idpsi(isDirichlet)) = u(idpsi(isDirichlet)) + ...
                                     weight(ip)*(lambda(ip,1)-lambda(ip,2))*pde.g_D(pxy);
         end
         u(isDirichlet) = hybrid_parameter*u(isDirichlet).*edgeSign(isDirichlet);
         u(idpsi(isDirichlet)) = hybrid_parameter*u(idpsi(isDirichlet)).*edgeSign(isDirichlet);
   
      % no edge length since the basis of sigma contains it.
        F = F + u;
    end

    %% Neumann boundary condition (Dirichlet BC in mixed form)
% B.  divergence operator
%B = icdmat(double(elem2edge),elemSign*[1 -1 1]);
   
    if ~isempty(pde.g_N) && any(isNeumann)
        % modify the rhs to include Dirichlet boundary condition 
        % Here we use simpson formula.
        % The dual functional for bases phi and psi are:
        % \Phi(u) = \int_e{i,j} u \cdot n_{i,j} ds
        % \Psi(u) = 3\int_e{i,j} u \cdot n_{i,j}(\lambda_i-\lambda_j) ds

        mid = 1/2*(node(Neumann(:,1),:)+node(Neumann(:,2),:));
        ve = node(Neumann(:,1),:)-node(Neumann(:,2),:);
        edgeLength = sqrt(sum(ve.^2,2)); 
        ne = [ve(:,2) -ve(:,1)]; % rotation of tangential vector
        
        if isnumeric(pde.g_N)
            evalg_N = pde.g_N;
        else
            %evalg_N = pde.g_N(mid,ne);
            evalg_N = pde.g_N(mid,1);
        end

        u(isNeumann) = 0; u(idpsi(isNeumann))=0;

        [lambda,weight] = quadpts1(4);
        nQuad = size(lambda,1);
        for ip = 1:nQuad
            pxy = lambda(ip,1)*node(Neumann(:,1),:) + lambda(ip,2)*node(Neumann(:,2),:);
            u(isNeumann) = u(isNeumann) + weight(ip)*pde.g_N(pxy,1).*edgeLength;
            u(idpsi(isNeumann)) = u(idpsi(isNeumann)) + ...
                                  weight(ip)*3*(lambda(ip,1)-lambda(ip,2))*pde.g_N(pxy,1).*edgeLength;
        end
        u(isNeumann) = u(isNeumann).*edgeSign(isNeumann);
        u(idpsi(isNeumann)) = u(idpsi(isNeumann)).*edgeSign(isNeumann);

        F = F - A*u;
        F(isNeumann) = u(isNeumann); F(idpsi(isNeumann))=u(idpsi(isNeumann));

%        if ~isempty(pde.K)
%	       mid = (node(elem(:,1),:) + node(elem(:,2),:) + node(elem(:,3),:))/3;
%          d = pde.K(mid);  % take inverse sequencil.             
%          u(isNeumann) = d.*u(isNeumann);
%        end
    end
    
    %% Pure Neumann boundary condition
    isPureNeumannBC = false;
    
    %% Modify the matrix
    %  Build Neumann boundary condition(Dirichlet BC in mixed form) into the
    %  matrix AD by enforcing  |AD(bdNode,bdNode)=I, 
    %  AD(bdNode,FreeNode)=0, AD(FreeNode,bdNode)=0|.
    if any(isBdDof)
       bdidx = zeros(Ndof,1); 
       bdidx(isBdDof) = 1;
       Tbd = spdiags(bdidx,0,Ndof,Ndof);
       T = spdiags(1-bdidx,0,Ndof,Ndof);
       AD = T*A*T + Tbd;
    else
       AD = A;
    end
  
end

end

  
    
