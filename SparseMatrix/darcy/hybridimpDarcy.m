%% CONVERGENCE OF hybrid method (RT0) FOR DARCY'S EQUATIONS
%
% This example is to show the convergence hybrid method
% (RT0-BDM1)  approximation of the Darcy's equations.
%

close all

%% Setting
[node,elem] = squaremesh([0,1,0,1],0.25); 
option.L0 = 1;
option.maxIt = 6; %RT0
option.printlevel = 1;
option.elemType = 'RT0';
%option.elemType = 'BDM1';
option.refType = 'red';
% option.solver = 'uzawapcg';
% option.solver = 'tripremixpoisson';
% option.solver = 'none';

%% Poisson
pde = Darcydataorg1;
%pde = Darcydataorg2;
%pde = Darcydataorg3;%k oscillating

display('hybrid Darcy: uniform grid')

bdFlag = setboundary(node,elem,'Neumann');
%bdFlag = setboundary(node,elem,'Dirichlet');
hfemimpDarcy(node,elem,pde,bdFlag,option);
