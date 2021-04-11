function phi(l,x,j,k,m) ! NEW DSSY basis ftn. on K_jk
  !On quadrilateral K_jk; vertices v1=(j,k), v2=(j-1,k), v3=(j-1,k-1), v4=(j,k-1)
  ! these are defined on physical domain K_jk
  use prec
  use common_var
  use maps
  use omp_lib
  implicit none
  integer:: l,j,k,m
  real(mp):: phi,x(2), h(2)
  real(mp) :: v1(2), v2(2), v3(2), v4(2)
  real(mp) :: c(4,0:3)
  real(mp):: den, n12, n21, nh1, nh2

  call vv(j,k,v1,v2,v3,v4)
  call hh(j,k,h)
!  h(1)=-0.99672860474004477_mp
!  h(2)=-0.99934383202099730_mp
  
  !q(x1,x2,h1,h2)=x1^2 + x2^2 - (3/10)*( (1+h1)*x1 + (1+h2)*x2) + (3/20)*(h1+h2)
  !mu(x1,x2,h1,h2) = x1*x2*q(x1,x2,h1,h2)

  !phi(x1,x2,h1,h2) = ( a+ b*x1 + c*x2 + d*mu(x1,x2,h1,h2) )

  den=2._mp+h(1)+h(2)+h(1)**2+h(2)**2
  n12=h(2)*(h(1)**2 + h(1) + h(2)**2 + 1._mp);  n21=h(1)*(h(2)**2 + h(2) + h(1)**2 + 1._mp)
  nh1=h(1)**2 + h(1) + 2._mp;  nh2=h(2)**2 + h(2) + 2._mp
  !----phi1---------- ! Indices are shifted for Meng's defs for ell.
!$omp parallel
!$omp do

!$omp end do
!$omp end parallel
  c(2,0) = h(1)*h(2)
  c(2,1) = -2._mp*n12/den
  c(2,2) = -2._mp*n21/den
  c(2,3) = 40._mp/den
  !----phi2----------
  c(3,0) = -h(2)
  c(3,1) = 2._mp*n12/den
  c(3,2) = 2._mp*nh2/den
  c(3,3) = -40._mp/den
  !----phi3---------
  c(4,0) = 1._mp
  c(4,1) = -2._mp*nh1/den
  c(4,2) = -2._mp*nh2/den
  c(4,3) = 40._mp/den
  !----phi4----------
  c(1,0) = -h(1)
  c(1,1) = 2._mp*nh1/den
  c(1,2) = 2._mp*n21/den
  c(1,3) = -40._mp/den

!  print*, "v1 = ", v1
!  print*, "v2 = ", v2
!  print*, "v3 = ", v3
!  print*, "v4 = ", v4
!  print*, " h = ", h
!  
!  print*, "c(1,:) = ", c(1,:)
!  print*, "c(2,:) = ", c(2,:)
!  print*, "c(3,:) = ", c(3,:)
!  print*, "c(4,:) = ", c(4,:)
!  stop  123
  
  !c = c/( (1-h(1))*(1-h(2)))
  phi = c(l,1)*ell(2,x,j,k,m) +c(l,2)*ell(1,x,j,k,m) &
       +c(l,3)*mu(x,j,k,m) ! for gradients m=1,2 omit the const terms
  if (m==0) phi = phi + c(l,0) ! for ftn vals m=0 add the const terms
  phi = phi/( (1._mp-h(1))*(1._mp-h(2)))

end function phi

function b_phi(l,b_x,j,k,m) ! NEW DSSY basis ftn. on K_jk
  !On quadrilateral K_jk; vertices v1=(j,k), v2=(j-1,k), v3=(j-1,k-1), v4=(j,k-1)
  ! these are defined on physical domain K_jk
  use prec
  use common_var
  use maps
  use omp_lib
  implicit none
  integer:: l,j,k,m
  real(mp):: b_phi,b_x(2)
  real(mp) :: v1(2), v2(2), v3(2), v4(2)
  real(mp) :: c(4,0:3), h(2)
  real(mp):: den, n12, n21, nh1, nh2

  call vv(j,k,v1,v2,v3,v4)
  call hh(j,k,h)

  !q(x1,x2,h1,h2)=x1^2 + x2^2 - (3/10)*( (1+h1)*x1 + (1+h2)*x2) + (3/20)*(h1+h2)
  !mu(x1,x2,h1,h2) = x1*x2*q(x1,x2,h1,h2)

  !phi(x1,x2,h1,h2) = ( a+ b*x1 + c*x2 + d*mu(x1,x2,h1,h2) )

  den=2._mp+h(1)+h(2)+h(1)**2+h(2)**2
  n12=h(2)*(h(1)**2 + h(1) + h(2)**2 + 1._mp);  n21=h(1)*(h(2)**2 + h(2) + h(1)**2 + 1._mp)
  nh1=h(1)**2 + h(1) + 2._mp;  nh2=h(2)**2 + h(2) + 2._mp
  !----phi1---------- ! Indices are shifted for Meng's defs for ell.
  c(2,0) = h(1)*h(2)
  c(2,1) = -2._mp*n12/den
  c(2,2) = -2._mp*n21/den
  c(2,3) = 40._mp/den
  !----phi2----------
  c(3,0) = -h(2)
  c(3,1) = 2._mp*n12/den
  c(3,2) = 2._mp*nh2/den
  c(3,3) = -40._mp/den
  !----phi3---------
  c(4,0) = 1._mp
  c(4,1) = -2._mp*nh1/den
  c(4,2) = -2._mp*nh2/den
  c(4,3) = 40._mp/den
  !----phi4----------
  c(1,0) = -h(1)
  c(1,1) = 2._mp*nh1/den
  c(1,2) = 2._mp*n21/den
  c(1,3) = -40._mp/den

  !c = c/( (1-h(1))*(1-h(2)))

  if(m==0) then
     b_phi = c(l,0) + c(l,1)*b_x(1) +c(l,2)*b_x(2) +c(l,3)*b_mu(b_x,h,0)!for 
  else if(m==1) then
     b_phi = c(l,1) + c(l,3)*b_mu(b_x,h,m) ! for 
  else if(m==2) then
     b_phi = c(l,2) + c(l,3)*b_mu(b_x,h,m) ! for
  end if
     
  b_phi = b_phi/( (1._mp-h(1))*(1._mp-h(2)))
end function b_phi



function t_phi(l,x,j,k,m) !m denotes the mth derivative, if m=0 fnt value
  use prec
  use common_var
  use maps
  use omp_lib
  implicit none
  integer  :: l,j,k,m
  real(mp) :: t_phi, num, den, x(2)
  real(mp) :: til_x(2),s(2)
  real(mp) :: v1(2), v2(2), v3(2), v4(2)
  real(mp) :: c(4,0:3), ts(2)

  call vv(j,k,v1,v2,v3,v4)

  s     = ss(v1,v2,v3,v4)
  til_x = inv_tA_K(x,v1,v2,v3,v4) ! invA(x-b)

  den=1._mp/(3._mp*s(1)**2 + 3._mp*s(2)**2 + 1._mp)

  !-(1)---t_phi(1)----------1 at (1,0)
  c(1,0) = .25_mp*((s(2)-1._mp)**2-s(1)**2)
  c(1,1) = num(-s(1),-s(2))/4._mp*den
  c(1,2) = num(s(1),s(2))/4._mp*den
  c(1,3) = 0.625_mp*den

  !-(2)---t_phi(2)----------1 at (0,1)
  c(2,0) = .25_mp*((s(1)-1._mp)**2-s(2)**2)
  c(2,1) = -num(-s(1),s(2))/4._mp*den
  c(2,2) = num(-s(1),-s(2))/4._mp*den
  c(2,3) = -0.625_mp*den

  !-(3)---t_phi(3)---------1 at (-1,0)
  c(3,0) = .25_mp*((s(2)+1._mp)**2-s(1)**2)
  c(3,1) = -num(s(1),-s(2))/4._mp*den
  c(3,2) = -num(-s(1),-s(2)) /4._mp*den
  c(3,3) = 0.625_mp*den

  !-(4)---t_phi(4)----------1 at (0,-1)
  c(4,0) = .25_mp*((s(1)+1._mp)**2-s(2)**2)
  c(4,1) = num(s(1),-s(2))/4._mp*den
  c(4,2) = -num(s(1),s(2))/4._mp*den
  c(4,3) = -0.625_mp*den
  !t_phi = sum_{j=1}c_j phi_j on til{K}
  t_phi = c(l,1)*t_ell(2,til_x,s,m) +c(l,2)*t_ell(1,til_x,s,m) &
       +c(l,3)*t_mu(til_x,s,m)
  if (m==0) t_phi = t_phi + c(l,0)
end function t_phi

function num(s1,s2)
  use prec
  implicit none
  real(mp), intent(in):: s1,s2
  real(mp):: num
  num = 2._mp*s1**3 + 3._mp*s1**2 + 2._mp*s1 - 2._mp*s2**3 + 3._mp*s2**2 - 2._mp*s2+ 1._mp
end function num
