function [nvb,val,gval]=ncelements(gp,etype)
%     3
%   4---3
% 4 |   | 2
%   1---2
%     1
nquad2=size(gp,1);
nvb=4;
switch etype
    case 'P1NC'
%% P1NC finite elements
p1nc=zeros(nquad2,1,nvb);
gp1nc=zeros(nquad2,2,nvb);
%
p1nc(:,:,3)=0.5+0.5.*gp(:,1)+0.5.*gp(:,2);
p1nc(:,:,4)=0.5-0.5.*gp(:,1)+0.5.*gp(:,2);
p1nc(:,:,1)=0.5-0.5.*gp(:,1)-0.5.*gp(:,2);
p1nc(:,:,2)=0.5+0.5.*gp(:,1)-0.5.*gp(:,2);
%
gv=0.5.*ones(nquad2,1);
gp1nc(:,:,3)=[gv,gv]; gp1nc(:,:,4)=[-gv,gv];
gp1nc(:,:,1)=[-gv,-gv]; gp1nc(:,:,2)=[gv,-gv];
val=p1nc; gval=gp1nc;
    case 'DSSY'
%% DSSY finite elements
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
    case 'rotQ1'
%% Rotated Q1
rt=zeros(nquad2,1,nvb);
grt=zeros(nquad2,2,nvb);
%
rt1=@(x,y) 0.25+0.5.*x+0.25.*(x.^2-y.^2);
rt2=@(x,y) 0.25+0.5.*y-0.25.*(x.^2-y.^2);
rt3=@(x,y) 0.25-0.5.*x+0.25.*(x.^2-y.^2);
rt4=@(x,y) 0.25-0.5.*y-0.25.*(x.^2-y.^2);
%
grt1x=@(x,y) x./2 + 0.5;
grt1y=@(x,y) -y./2;
grt2x=@(x,y) -x./2;
grt2y=@(x,y) y./2 + 0.5;
grt3x=@(x,y) x./2 - 0.5;
grt3y=@(x,y) -y./2;
grt4x=@(x,y) -x./2;
grt4y=@(x,y) y./2 - 0.5;
%
rt(:,:,1)=rt4(gp(:,1),gp(:,2));
rt(:,:,2)=rt1(gp(:,1),gp(:,2));
rt(:,:,3)=rt2(gp(:,1),gp(:,2));
rt(:,:,4)=rt3(gp(:,1),gp(:,2));
%
grt(:,:,1)=[grt4x(gp(:,1),gp(:,2)),grt4y(gp(:,1),gp(:,2))];
grt(:,:,2)=[grt1x(gp(:,1),gp(:,2)),grt1y(gp(:,1),gp(:,2))];
grt(:,:,3)=[grt2x(gp(:,1),gp(:,2)),grt2y(gp(:,1),gp(:,2))];
grt(:,:,4)=[grt3x(gp(:,1),gp(:,2)),grt3y(gp(:,1),gp(:,2))];
val=rt; gval=grt;
    otherwise
        nvb=0; val=0; gval=0;
        disp('error, nc-elements')
end