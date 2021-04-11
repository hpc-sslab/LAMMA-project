function result=exactSol(x,y,exSolCase)
    global epsil
    global zero one two hf qt pi eps
    switch exSolCase
        case 1
            result =  sin(3.0*pi*x).*y.*(one-y) + epsil*sin(pi.*x./epsil).*sin(pi.*y./epsil);
    end
end
