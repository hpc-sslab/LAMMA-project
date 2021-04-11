function pde = Darcydataorg3
%% 
% p = x^3y+y^4 + sin(pi*x)cos(pi*y)

  pde = struct('K', @K, 'exactp',@exactp,'exactu',@exactu,'Dp',@gradp,...
             'f', @f,'g_D',@g_D,'g_N',@g_N);

 function s = K(pt)
        x = pt(:,1); y = pt(:,2);
        s=2*(2+sin(10*pi*x).*cos(10*pi*y)); %oscilltory coefficient data

	%random data
	%z = 0.63+0.02*(0.5-rand(n,1)); 	
        % s  = ones(size(pt,1),1);
        % rand('seed', sum(100*clock))
        % for k=1:size(pt,1) 
 	%  if rand < 0.2
     	%s(k)= 1000;
 	%  else 
 	%    s(k)=1;
 	% end;
        % end
         
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
      K = K(pt)

     s(:,1) = K(:).*Dp(:,1);
     s(:,2) = K(:).*Dp(:,2);
       
    end
    function s = f(pt)
        x = pt(:,1); y = pt(:,2);
        K = K(pt)  

        uxx = 6.*x.*y - pi*pi.*sin(pi.*x).*cos(pi.*y); % u_{xx}       
        uyy = 12.*y.*y - pi*pi.*sin(pi.*x).*cos(pi.*y);% u_{yy}
        s = -(K(:).*uxx +  K(:).*uyy);
    end
    function s = g_D(pt)
       s = exactp(pt); 
    end

function s = g_N(pt,vargin) %kgradp=g
        s = zeros(size(pt,1),1);
        x = pt(:,1); y = pt(:,2);
        K=K(pt);
        u=exactu(pt);
        uprime=[u(:,1),u(:,2)];

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
