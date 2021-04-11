function [ovX,ovY]=nwCornerGet(mx,my,xMacNum,yMacNum,isOver)
ovX=-isOver;
ovY=-isOver;

if (mx==0)
    if(my==0)
        ovX=0; ovY=0;
    elseif(my==yMacNum-1)
        ovX=0; ovY=-2*isOver;
    else
        ovX=0; ovY=-isOver;
    end
elseif (mx==xMacNum-1)
    if(my==0)
        ovX=-2*isOver; ovY=0;
    elseif(my==yMacNum-1)
        ovX=-2*isOver; ovY=-2*isOver;
    else
        ovX=-2*isOver; ovY=-isOver;
    end
end

if (my==0)
    if(mx==0)
        ovX=0; ovY=0;
    elseif(mx==xMacNum-1)
        ovX=-2*isOver; ovY=0;
    else
        ovX=-isOver; ovY=0;
    end
elseif (my==yMacNum-1)
    if(mx==0)
        ovX=0; ovY=-2*isOver;
    elseif(mx==xMacNum-1)
        ovX=-2*isOver; ovY=-2*isOver;
    else
        ovX=-isOver; ovY=-2*isOver;
    end
end
    
end