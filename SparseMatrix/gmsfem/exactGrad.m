function result=exactGrad(a,b)
result=([[pi.*cos((1000.*a.*pi)/37).*sin((1000.*b.*pi)/37) - 3.*b.*pi.*cos(3.*a.*pi).*(b - 1)], [pi.*cos((1000.*b.*pi)/37).*sin((1000.*a.*pi)/37) - sin(3.*a.*pi).*(b - 1) - b.*sin(3.*a.*pi)]]);