function [ux,uy,ps]=stokesolver(kx,nv,bx,by,np,rhs,qm,TOL,maxit)
%%
global amg_grid amg_smoother   
%%
%% step1, set up AMG structure
%%
amg_grid = amg_grids_setup(kx,1,5);
%%
%% step2, smoother
%%
% point damped Jacobi
smoother_params = amg_smoother_params(amg_grid, 'PDJ', 2);
% point Gauss-Seidel
%smoother_params = amg_smoother_params(amg_grid, 'PGS', 2);
amg_smoother = amg_smoother_setup(amg_grid, smoother_params);
%%
%% MINRES iteration
%%
x0=zeros(nv+nv+np,1);
A=[kx,sparse(nv,nv);sparse(nv,nv),kx];
B=[bx,by];
%tic
[sol,flag,~,~,~]=minres([A,B';B,sparse(np,np)],rhs, ...
    TOL,maxit,'m_st_amgz',[],x0,A,qm);
%toc
ux=sol(1:nv); uy=sol(nv+1:2*nv); ps=sol(2*nv+1:2*nv+np);
if ( flag ~= 0 )
    disp('not converged')
end