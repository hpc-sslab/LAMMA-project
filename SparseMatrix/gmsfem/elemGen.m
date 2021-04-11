function [nvb,val,grad_val]=elemGen(gp, type)
    switch type
        case 'p1nc'
            [nvb,val,grad_val]=p1nc_2d(gp);
        case 'q1-c'
            [nvb,val,grad_val]=q1c_2d(gp);
        case 'DSSY'
            [nvb,val,grad_val]=dssy_2d(gp);
    end
end