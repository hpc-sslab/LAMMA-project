function [xIndex,yIndex]=indexChanger1Dto2D(Index,nx,ny)
% start from (0,0)
xIndex=mod(Index-1,nx+1);
yIndex=(Index-xIndex-1)/(ny+1);
end
