function [nvb,val,grad_val]=q1c_2d(gridpoint)

% Assume that the vertice of each grid is stored in
%  1-----4
%  |     |
%  |     |
%  |     |
%  2-----3
%  
%  1 = (-1,-1), 2 = (1,-1), 3 = (1,1), 4 = (-1,1)


num_quad_grids=size(gridpoint,1);
nvb=4;

q1c=zeros(num_quad_grids,1,nvb);
gq1c=zeros(num_quad_grids,2,nvb);
%
q1c(:,:,1)=0.25-0.25.*gridpoint(:,1)-0.25.*gridpoint(:,2)+0.25.*gridpoint(:,1).*gridpoint(:,2);
q1c(:,:,2)=0.25+0.25.*gridpoint(:,1)-0.25.*gridpoint(:,2)-0.25.*gridpoint(:,1).*gridpoint(:,2);
q1c(:,:,3)=0.25+0.25.*gridpoint(:,1)+0.25.*gridpoint(:,2)+0.25.*gridpoint(:,1).*gridpoint(:,2);
q1c(:,:,4)=0.25-0.25.*gridpoint(:,1)+0.25.*gridpoint(:,2)-0.25.*gridpoint(:,1).*gridpoint(:,2);
%
gq1c(:,:,1)=[-0.25+0.25.*gridpoint(:,2),-0.25+0.25.*gridpoint(:,1)]; 
gq1c(:,:,2)=[0.25-0.25.*gridpoint(:,2),-0.25-0.25.*gridpoint(:,1)]; 
gq1c(:,:,3)=[0.25+0.25.*gridpoint(:,2),0.25+0.25.*gridpoint(:,1)]; 
gq1c(:,:,4)=[-0.25-0.25.*gridpoint(:,2),0.25-0.25.*gridpoint(:,1)]; 
val=q1c; grad_val=gq1c;

end