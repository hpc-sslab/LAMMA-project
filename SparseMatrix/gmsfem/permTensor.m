% Permeability tensor K
function k = permTensor(x,y)
    global homo
    global one pi epsil
    switch homo
        case "homo"
            k=ones(size(x)); % identity
        case "inho"
            k = one + (one+x).*(one+y) +...
                epsil.*sin(10.0.*pi.*x).*sin(5.0.*pi.*y);
    end
end