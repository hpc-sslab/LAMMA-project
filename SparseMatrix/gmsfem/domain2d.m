function [domain,node,elem,edge,gquad]=domain2d(domx,domy,hx,hy,nquad)
%% Problem
% The steady state Stokes equation,
% computational domain setup, rectangle
% uniform discretization, element: hxh square
% domx=[a,b], domy=[c,d], Omega = [a,b]x[c,d]
% h=1/2^l;
% nquad: # of quadrature points
global ncons
%%
%% Discretization
%%
%% Domain, square
dim=2;
nx=round((domx(2)-domx(1))/hx);
ny=round((domy(2)-domy(1))/hy);
% structure
domain=struct('x',domx,'y',domy,'dim',dim,'nx',nx,'ny',ny,'hx',hx,'hy',hy);
%% node
x=domx(1):hx:domx(2);
y=domy(1):hy:domy(2);
[X,Y]=meshgrid(x,y);
xi=reshape(X',(nx+1)*(ny+1),1);
eta=reshape(Y',(nx+1)*(ny+1),1);
xb=ones(size(x)); yb=ones(size(y));
xb(1)=0; xb(end)=0; yb(1)=0; yb(end)=0;
% boundary indicator. 0 == boundary, 1 == interior
node_on_bdry=reshape(xb'*yb,(nx+1)*(ny+1),1); 
num_nodes=(nx+1)*(ny+1); % # of nodes
node_trace=zeros(max(nx+1,ny+1), 4);
for j=1:nx+1
    node_trace(j,1)=j;
    node_trace(nx+1+1-j,3)=num_nodes+1-j;
end
for j=1:ny+1
    node_trace(j,4)=1+(nx+1)*(j-1);
    node_trace(j,2)=(nx+1)*j;
end
%structure
node=struct('num',num_nodes,'coord',[xi(:),eta(:)],'boundary',node_on_bdry,'trace',node_trace,'meshX',X,'meshY',Y);
%% elem
% 4---3
% |   |
% 1---2
elem_to_node=zeros(nx*ny,4); 
elem_middle=zeros(nx*ny,2);
% indicates nodes on each rectangular element

for jy=1:ny
    for jx=1:nx
        id=(jy-1)*nx+jx;
        elem_middle(id,1)=x(jx)+0.5*hx;
        elem_middle(id,2)=y(jy)+0.5*hy;
        elem_to_node(id,1)=(jy-1)*(nx+1)+jx;
        elem_to_node(id,2)=(jy-1)*(nx+1)+jx+1;
        elem_to_node(id,3)=jy*(nx+1)+jx+1;
        elem_to_node(id,4)=jy*(nx+1)+jx;
    end
end
num_elem=nx*ny; % # of elements
elem_red_black=zeros(nx*ny,1); 
% To describe checkerboard pattern, attach 0(red) 1(black) indices to each
% rectangular element
for jy=1:2:ny
    for jx=1:2:nx
        id=(jy-1)*nx+jx;
        elem_red_black(id)=0; elem_red_black(id+1+nx)=0; % red elements
        elem_red_black(id+1)=1; elem_red_black(id+nx)=1; % black elements
    end
end
elem_trace = zeros(max(nx,ny),4);
for j=1:nx
    elem_trace(j,1)=j;
    elem_trace(nx+1-j,3)=num_elem+1-j;
end
for j=1:ny
    elem_trace(j,4)=1+nx*(j-1);
    elem_trace(j,2)=nx*j;
end
%% edge
%     3
%   4---3
% 4 |   | 2
%   1---2
%     1
edges=zeros(4*num_elem,2);
% For each rectanguler element, there are 4 edges.
% Store two nodes(endpoints) for each edge.
for j=1:num_elem
    edges((j-1)*4+1,:)=elem_to_node(j,[1,2]);
    edges((j-1)*4+2,:)=elem_to_node(j,[2,3]);
    edges((j-1)*4+3,:)=elem_to_node(j,[3,4]);
    edges((j-1)*4+4,:)=elem_to_node(j,[4,1]);
end
% Adjust id of nodes which compose each edge
dd=sort(edges,2);
% Remove repeated edges
[edge_node,~,ic]=unique(dd,'rows'); % edge=dd(ia,:); dd=edge(ic,:);
% edge_bdry_node indicates the edges on bdry. 0 == bdry 1 == interior
edge_bdry_node=node_on_bdry(edge_node(:,1))+node_on_bdry(edge_node(:,2));
edge_bdry_node( edge_bdry_node == 2 )=1;

% Store edge element and bdry node elements.
elem_to_edge=zeros(num_elem,4); % elem to edge
bdry_node=zeros(num_elem,1);
for j=1:num_elem
    elem_to_edge(j,:)=ic((j-1)*4+1:(j-1)*4+4)';
    bdry_node(j)=sum(edge_bdry_node(elem_to_edge(j,:)));
end
neg=size(edge_node,1);

edge_trace = zeros(max(nx,ny),4);
for j=1:nx
    edge_trace(j,1)=2*(j-1)+1;
    edge_trace(nx+1-j,3)=neg+1-j;
end
for j=1:ny
    edge_trace(j,4)=2*nx*(j-1)+j+1;
    edge_trace(j,2)=2*nx*j+j;
end
edge_be = zeros(nx,ny+1); edge_al = zeros(nx+1,ny);
for j = 1:nx
    for k= 1:ny
        edge_be(j,k)=1+2*(j-1)+(2*nx+1)*(k-1);
    end
end
edge_be(:,ny+1)=(neg-nx+1):neg;
for j = 1:nx
    for k= 1:ny
        edge_al(j,k)=2*(j)+(2*nx+1)*(k-1);
    end
end
edge_al(nx+1,:)=edge_al(nx,:)+1;

bdel=zeros(num_elem,1);
bdel( bdry_node == 4)=1;
%%
%% element and edge structure
%%
elem=struct('num',num_elem,'node',elem_to_node,'edge',elem_to_edge,'boundary',bdel,'mid',elem_middle,'rb',elem_red_black,'trace',elem_trace);
edge=struct('num',neg,'node',edge_node,'boundary',edge_bdry_node,'trace',edge_trace,'al',edge_al,'be',edge_be);
%%
%% Integration
%%
%% Gauss-Legengre Quadrature
[gp,gw]=gausslegendrequad2d(nquad);
nquad2=nquad^dim;
jac=[0.5*hx,0;0,0.5*hy];
detk=det(jac);
gquad=struct('num',nquad,'dim',nquad2,'gp',gp,'gw',gw,'Jacobian',jac,'det',detk);