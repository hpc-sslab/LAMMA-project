function dg=diagCheck(nx,ny,ovMicNum)
    dg=zeros(4,1);
    if (nx-1 > 1 & ny-1 > 1)
        dg(1)=1;
    end
    if (nx+1 < ovMicNum+1 & ny-1 > 1)
        dg(2)=1;
    end
    if (nx+1 < ovMicNum+1 & ny+1 < ovMicNum+1)
        dg(3)=1;
    end
    if (nx-1 > 1 & ny+1 < ovMicNum+1)
        dg(4)=1;
    end
end