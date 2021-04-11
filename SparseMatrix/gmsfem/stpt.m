function [x,y]=stpt(type)
switch type
    case 1
        x=0; y=0;
    case 2
        x=1; y=0;
    case 3
        x=1; y=1;
    case 4
        x=0; y=1;
end