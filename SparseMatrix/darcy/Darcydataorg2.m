function pde = Darcydataorg2
%% 
% p = x^3y+y^4 + sin(pi*x)cos(pi*y)

  pde = struct('K', @K, 'exactp',@exactp,'exactu',@exactu,'Dp',@gradp,...
             'f', @f,'g_D',@g_D,'g_N',@g_N);

 function s = K(pt)
        x = pt(:,1); y = pt(:,2);
    s(:,1)=1;
    s(:,2)=100;
    s(:,3)=1;
  end

    function s = exactp(pt)
       x = pt(:,1); y = pt(:,2);
       s = x.^3.*y + y.^4 + sin(pi*x).*cos(pi*y)-13/40;
       tot=sum(s);
    end
    function s = gradp(pt)
       x = pt(:,1); y = pt(:,2);
       s(:,1) = 3*x.^2.*y + pi*cos(pi*x).*cos(pi*y);       
       s(:,2) = x.^3 + 4*y.^3 - pi*sin(pi*x).*sin(pi*y);

    end
    function s = exactu(pt)
      Dp = gradp(pt);  % u =  K*grad(p)
      K = K(pt);
      s(:,1) = K(:,1).*Dp(:,1) + K(:,3).*Dp(:,2);
      s(:,2) = K(:,3).*Dp(:,1)+ K(:,2).*Dp(:,2);

      % s(:,1) = Dp(:,1);
      % s(:,2) = Dp(:,2);
       
    end
    function s = f(pt)
        x = pt(:,1); y = pt(:,2);
        s = -(6.*x.*y - pi*pi.*sin(pi.*x).*cos(pi.*y) + 2* (3.*x.*x - pi*pi.*cos(pi.*x).*sin(pi.*y)) + 100*(12.*y.*y - pi*pi.*sin(pi.*x).*cos(pi.*y)));
    end
    function s = g_D(pt)
       s = exactp(pt); 
    end

function s = g_N(pt,vargin) %kgradp=g
       s = zeros(size(pt,1),1);
       x = pt(:,1); y = pt(:,2);
       K=K(pt);
       uprime = [ (3*x.^2.*y + pi*cos(pi*x).*cos(pi*y)) + (x.^3 + 4*y.^3 - pi*sin(pi*x).*sin(pi*y)), (3*x.^2.*y + pi*cos(pi*x).*cos(pi*y))+ 100*(x.^3 + 4*y.^3 - pi*sin(pi*x).*sin(pi*y)) ];

        leftbd = (abs(x)<eps);  % n = (-1,0); 
        s(leftbd) = - uprime(leftbd,1);
        rightbd = (abs(x-1)<eps); % n = (1,0); 
        s(rightbd) = uprime(rightbd,1);
        topbd = (abs(y-1)<eps);   % n = (0,1)
        s(topbd) = uprime(topbd,2);
        bottombd = (abs(y)<eps);% n = (0,-1)
        s(bottombd) = - uprime(bottombd,2);    
end

end
