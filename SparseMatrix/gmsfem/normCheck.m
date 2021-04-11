clear; close all; clc;

    %% CONTROL PART
    % isRmesh = 0 : uniform mesh 1 : random mesh
    % isRcoef = 0 : predefined coef 1 : random coef
    isRmesh=0;
    isRcoef=0;
    coefType=0;
    
    lwlim=1e-4;
    Hfstart=1;
    fprintf("N_H\t\tN_h\t\tH1err\t\tH1ord\t\tL2err\t\tL2ord\n");
    for Hf=Hfstart:8
        hf=10;
        %% MESH GENERATE
        H=1/2^Hf;
%         h=1/2^hf;
        h=1/2^hf;

    %     if (H <= h)
    %         fprintf("Invalid setting");
    %     end
        mesh=0:h:H;
        if (isRmesh==1)
            mesh=0*mesh;
            for i=2:2^(hf-Hf)+1
                mesh(i)=rand()+mesh(i-1);
            end
            mesh=mesh/mesh(2^(hf-Hf)+1)*H;
        end
        % mesh size array
        meshh=mesh(2:end)-mesh(1:end-1);
        %% COEFFICIENT
        coef=0*mesh(2:end);
        coef=exp(mesh(2:end));
        if (isRcoef==1)
            coef=rand(size(coef))+lwlim;
        end
        % weighted harmonic mean of coef;
        harmC=1/sum(meshh./coef);

        %% ERROR ESTIMATE
        l2err=0;
        h1err=0;
        l2ord=0.0;
        h1ord=0.0;
        % cOvCoef = (c/a_j)
        cOvCoef=harmC./coef;
        for i=1:2^(hf-Hf)
            l2err=l2err+1/3*(1/(cOvCoef(i)-1/H))...
                *(sum((cOvCoef(1:i)-1/H).*meshh(1:i))^3....
            -sum((cOvCoef(1:i-1)-1/H).*meshh(1:i-1))^3);
            h1err=h1err+(harmC/coef(i)-1/H)^2*meshh(i);
    %        fprintf("%f\n",1/3*(1/(cOvCoef(i)-1/H))*(sum((cOvCoef(1:i)-1/H).*meshh(1:i)))^3);
        end
        l2err=sqrt(l2err);
        h1err=sqrt(h1err);
        if(Hf~=Hfstart)
            l2ord=log2(l2errold/l2err);            
            h1ord=log2(h1errold/h1err);
        end
        l2errold=l2err;
        h1errold=h1err;
            
        fprintf("%d\t\t%d\t%0.3f\t\t%f\t\t%f\t\t%f\n",1/H,1/h,h1err,h1ord,l2err,l2ord);
    end