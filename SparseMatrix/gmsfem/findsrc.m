syms a b
f=gradient(exactSol(a,b));
f=permTensor(a,b,eps)*f;
f=-divergence(f);

fileID = fopen('src.m','w');
fprintf(fileID, 'function result=src(x,y)\n');
fprintf(fileID, 'result=');
fprintf(fileID, char(f));
fprintf(fileID, ';');
fclose(fileID);
type srctest.m