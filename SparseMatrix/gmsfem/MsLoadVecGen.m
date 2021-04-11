function [rhs]=MsLoadVecGen(S,locBdryNode,ovMicNum,macInDom,totNum,macNum,micNum,isOver,ovRefBdryW,ovLocPerm,type,GMsFEM)
switch GMsFEM
    case 0
    switch type
        case 'p1-nc'
            rhs=zeros(ovMicNum-1,ovMicNum-1,macInDom,4);
            for i=1:4
                rhs(1,:,:,i)=ovRefBdryW(1,1:ovMicNum-1,i)...
                    .*ovLocPerm(1,1:ovMicNum-1,:); % g_k,1 * K_k,1 
                rhs(1,:,:,i)=rhs(1,:,:,i)+...
                    ovRefBdryW(1,3:ovMicNum+1,i)...
                    .*ovLocPerm(1,2:ovMicNum,:); % g_k+1,1 * K_k,1
                rhs(2:ovMicNum-1,1,:,i)=ovRefBdryW(2:ovMicNum-1,1,i)...
                    .*ovLocPerm(2:ovMicNum-1,1,:); % g_1,k * K_1,k (1st comp is omitted)
                rhs(:,1,:,i)=rhs(:,1,:,i)+...
                    ovRefBdryW(3:ovMicNum+1,1,i)...
                    .*ovLocPerm(2:ovMicNum,1,:); % g_1,k+1 * K_1,k
                rhs(ovMicNum-1,2:ovMicNum-1,:,i)=ovRefBdryW(ovMicNum+1,2:ovMicNum-1,i)...
                    .*ovLocPerm(ovMicNum,2:ovMicNum-1,:); % g_n+1,k * K_n,k (1st comp is omitted)
                rhs(ovMicNum-1,:,:,i)=rhs(ovMicNum-1,:,:,i)+...
                    ovRefBdryW(ovMicNum+1,3:ovMicNum+1,i)...
                    .*ovLocPerm(ovMicNum,2:ovMicNum,:); % g_n+1,k+1 * K_n,k 
                rhs(2:ovMicNum-1,ovMicNum-1,:,i)=rhs(2:ovMicNum-1,ovMicNum-1,:,i)+...
                    ovRefBdryW(2:ovMicNum-1,ovMicNum+1,i)...
                    .*ovLocPerm(2:ovMicNum-1,ovMicNum,:); % g_n+1,k * K_n,k (1st comp is omitted)
                rhs(1:ovMicNum-2,ovMicNum-1,:,i)=rhs(1:ovMicNum-2,ovMicNum-1,:,i)+...
                    ovRefBdryW(3:ovMicNum,ovMicNum+1,i)...
                    .*ovLocPerm(2:ovMicNum-1,ovMicNum,:); % g_n+1,k+1 * K_n,k
            end
    end
    
    case 1
        switch type
            case 'p1-nc'
                num=ovMicNum*4;
                node=[1:(totNum+1)*(totNum+1)];
                node=reshape(node,totNum+1,totNum+1);
                glbNode=zeros(ovMicNum+1,ovMicNum+1,macInDom);
                for ielem=1:macInDom            
                    % Macro grid location ( 2D (0,0) to (macNum-1,macNum-1) )
                    mx=mod(ielem-1,macNum);
                    my=(ielem-mx-1)/macNum;
                    % start point ( nw corner )
                    crn=micNum*[mx,my]+[1,1];
                    [ovX,ovY]=nwCornerGet(mx,my,macNum,macNum,isOver);
                    ovCrn=crn+[ovX,ovY];
                    glbNode(:,:,ielem)=node(ovCrn(1):ovMicNum+ovCrn(1),...
                            ovCrn(2):ovMicNum+ovCrn(2));
                end
                rhs=zeros((ovMicNum-1)*(ovMicNum-1),macInDom,num);
                for i=1:num
                    tmpRhs=zeros(ovMicNum+1,ovMicNum+1,macInDom);
                    bdnode=locBdryNode(i);
                    nx=mod(bdnode-1,ovMicNum+1);
                    ny=(bdnode-nx-1)/(ovMicNum+1);
                    nx=nx+1;ny=ny+1;
                    currentPos=[nx,ny];
                    relPos=[-1,-1;1,-1;1,1;-1,1];
                    dg=diagCheck(nx,ny,ovMicNum);
                    dof=find(dg);
                    dofNum=size(dof);
                    for j=1:dofNum(1)
                        intPos=currentPos+relPos(dof(j),:);
                        tmpRhs(intPos(1),intPos(2),:)=...
                            diag(S(glbNode(currentPos(1),currentPos(2),:),...
                            glbNode(intPos(1),intPos(2),:)));
                    end
                    rhs(:,:,i)=reshape(tmpRhs(2:ovMicNum,2:ovMicNum,:),...
                        (ovMicNum-1)*(ovMicNum-1),macInDom);
                    %rhs(locBdryNode(i),:,i)=full(diag(S(LocGlbNode(1,1,:),LocGlbNode(2,2,:))));
                end
                
        end
    
end