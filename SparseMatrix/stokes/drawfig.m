clear; close all; clc;
load result512x1024.mat
%% post process
obi=find(nu ~= 1);
obn=elem.node(obi,:);
obx=(node.coord(obn(:,1),1)+node.coord(obn(:,2),1))./2;
oby=(node.coord(obn(:,2),2)+node.coord(obn(:,3),2))./2;
%
figure(1)
cnum=[1,0.7,0.4,0.1,1.0*10^-3,1.0*10^-4,1.0*10^-5];
contour(figd.gridx,figd.gridy,figd.u,cnum,'k','Showtext','on'), hold on
plot(obx,oby,'s')

figure(2)
contourf(figd.gridx,figd.gridy,figd.u,20), hold on
plot(obx,oby,'s')
colorbar