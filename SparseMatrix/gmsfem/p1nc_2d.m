function [nvb,val,grad_val]=p1nc_2d(gridpoint)

% Numbering for P1NC element
%  4--3--3
%  |     |
%  4     2
%  |     |
%  1--1--2
%
%  
%  1 = (-1,-1), 2 = (1,-1), 3 = (1,1), 4 = (-1,1)

num_quad_grids=size(gridpoint,1);
nvb=4;

dssy=zeros(num_quad_grids,1,nvb);
gdssy=zeros(num_quad_grids,2,nvb);
%
dssy(:,:,3)=0.5+0.5.*gridpoint(:,1)+0.5.*gridpoint(:,2);
dssy(:,:,4)=0.5-0.5.*gridpoint(:,1)+0.5.*gridpoint(:,2);
dssy(:,:,1)=0.5-0.5.*gridpoint(:,1)-0.5.*gridpoint(:,2);
dssy(:,:,2)=0.5+0.5.*gridpoint(:,1)-0.5.*gridpoint(:,2);
%
gradient_value=0.5.*ones(num_quad_grids,1);
gdssy(:,:,3)=[gradient_value,gradient_value]; gdssy(:,:,4)=[-gradient_value,gradient_value];
gdssy(:,:,1)=[-gradient_value,-gradient_value]; gdssy(:,:,2)=[gradient_value,-gradient_value];
val=dssy; grad_val=gdssy;

