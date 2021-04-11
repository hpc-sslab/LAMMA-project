function val=bdry_value(x,y,domain_size,type,FEType)
switch FEType
    case 'p1-nc'
        switch type
            case 1
                %    1
                % 1     0
                %    0
                val = (-x-y+1.5*domain_size)/(domain_size);
            case 2
                %    1
                % 0     1
                %    0
                val = (-x+y+0.5*domain_size)/(domain_size);
            case 3
                %    0
                % 0     1
                %    1
                val = (+x+y-0.5*domain_size)/(domain_size);
            case 4
                %    0
                % 1     0
                %    1
                val = (+x-y+0.5*domain_size)/(domain_size);
        end
    case 'q1-c'
        switch type
            case 1
                % 1     0
                %       
                % 0     0
                val = (-x-y+x.*y/domain_size+domain_size)/(domain_size);
            case 2
                % 0     0
                %       
                % 1     0
                val = (x-x.*y/domain_size)/(domain_size);
            case 3
                % 0     0
                %       
                % 0     1
                val = (x.*y/domain_size)/(domain_size);
            case 4
             	% 0     1
                %       
                % 0     0
                val = (+y-x.*y/domain_size)/(domain_size);
        end
end