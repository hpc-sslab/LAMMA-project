function val=adjust(x,y,a,b,c,d,domain_size,type,type2)
switch type
    case 1
        A=[1/2,0,1;0,1/2,1;1/2,1,1];
        switch type2
            case 1
                rhs=[a;a;c];
            case 2
                rhs=[a;c;c];
            case 3
                rhs=[a;a;c];
            case 4
                rhs=[a;c;c];
        end
        coeffi=A\rhs;
        val=(coeffi(1)*x+coeffi(2)*y+coeffi(3)*domain_size)/domain_size;
    case 2
        A=[1/2,0,1;0,1/2,1;1/2,1,1];
        rhs=[a;b;c];
        coeffi=A\rhs;
        val=(coeffi(1)*x+coeffi(2)*y+coeffi(3)*domain_size)/domain_size;
    case 3
        A=[1/2,1/4*domain_size,0,1;0,0,1/2,1;...
            1/2,1/4*domain_size,1,1;1,domain_size,1/2,1];
        rhs=[a;b;c;d];
        coeffi=A\rhs;
        val=(coeffi(1).*x+coeffi(2).*x.*x+coeffi(3).*y+coeffi(4).*domain_size)/domain_size;
    case 4      
        H=domain_size;
        A=[1/2*H,0,1;0,1/2*H,1;1/2*H,H,1];
        switch type2
            case 1
                rhs=[a;a;c];
            case 2
                rhs=[a;c;c];
            case 3
                rhs=[a;a;c];
            case 4
                rhs=[a;c;c];
        end
        coeffi=A\rhs;
        val=(coeffi(1)*x+coeffi(2)*y+coeffi(3))/domain_size;
    case 5      
        H=domain_size;
        A=[1/2*H,0,1;0,1/2*H,1;1/2*H,H,1];
        rhs=[a;b;c];
        coeffi=A\rhs;
        val=(coeffi(1)*x+coeffi(2)*y+coeffi(3))/domain_size;
    case 6      
        H=domain_size;
        A=[1/2*H,0,1/3*H*H,0;0,1/2*H,0,1/3*H*H;1/2*H,H,1/3*H*H,H*H;H,1/2*H,H*H,1/3*H*H];
        rhs=[a;b;c;d];
        coeffi=A\rhs;
        val=(coeffi(1).*x+coeffi(2).*y+coeffi(3).*x.*x+coeffi(4).*y.*y)/domain_size;
    case 100
        val=0;
end