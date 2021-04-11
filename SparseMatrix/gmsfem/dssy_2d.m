function [nvb,val,gval]=dssy_2d(gp)

% Numbering for P1NC element
%  4--3--3
%  |     |
%  4     2
%  |     |
%  1--1--2
%
%
%  1 = (-1,-1), 2 = (1,-1), 3 = (1,1), 4 = (-1,1)
nvb = 4; nquad2=size(gp,1);
dssy=zeros(nquad2,1,nvb);
gdssy=zeros(nquad2,2,nvb);
%
dssy1=@(x,y) 0.25+0.5.*x-3./8.*(x.^2-5./3.*x.^4-y.^2+5./3.*y.^4);
dssy2=@(x,y) 0.25+0.5.*y+3./8.*(x.^2-5./3.*x.^4-y.^2+5./3.*y.^4);
dssy3=@(x,y) 0.25-0.5.*x-3./8.*(x.^2-5./3.*x.^4-y.^2+5./3.*y.^4);
dssy4=@(x,y) 0.25-0.5.*y+3./8.*(x.^2-5./3.*x.^4-y.^2+5./3.*y.^4);
%
gdssy1x=@(x,y) (5.*x.^3)./2 - (3.*x)./4 + 0.5;
gdssy1y=@(x,y) (3.*y)./4 - (5.*y.^3)./2;
gdssy2x=@(x,y) (3.*x)./4 - (5.*x.^3)./2;
gdssy2y=@(x,y) (5.*y.^3)./2 - (3.*y)./4 + 0.5;
gdssy3x=@(x,y) (5.*x.^3)./2 - (3.*x)./4 - 0.5;
gdssy3y=@(x,y) (3.*y)./4 - (5.*y.^3)./2;
gdssy4x=@(x,y) (3.*x)./4 - (5.*x.^3)./2;
gdssy4y=@(x,y) (5.*y.^3)./2 - (3.*y)./4 - 0.5;
%
dssy(:,:,1)=dssy4(gp(:,1),gp(:,2));
dssy(:,:,2)=dssy1(gp(:,1),gp(:,2));
dssy(:,:,3)=dssy2(gp(:,1),gp(:,2));
dssy(:,:,4)=dssy3(gp(:,1),gp(:,2));
%
gdssy(:,:,1)=[gdssy4x(gp(:,1),gp(:,2)),gdssy4y(gp(:,1),gp(:,2))];
gdssy(:,:,2)=[gdssy1x(gp(:,1),gp(:,2)),gdssy1y(gp(:,1),gp(:,2))];
gdssy(:,:,3)=[gdssy2x(gp(:,1),gp(:,2)),gdssy2y(gp(:,1),gp(:,2))];
gdssy(:,:,4)=[gdssy3x(gp(:,1),gp(:,2)),gdssy3y(gp(:,1),gp(:,2))];
val=dssy; gval=gdssy;