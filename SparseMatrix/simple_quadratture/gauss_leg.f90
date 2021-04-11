
subroutine gauss_leg(x1,x2,gp,gw,n,mm)
  use prec
  implicit none
  integer:: m, n, mm, j, k
  real(mp):: x1,x2,gp(mm),gw(mm), eps, dpmeps, pi, xm, xl, z, z1, p1, p2, p3, pp
  real(mp):: zero, one, two, hf, qt
  zero = 0._mp; one = 1._mp;  two=2._mp;  hf=.5_mp;  qt=.25_mp
  
  eps = epsilon(one)
      
  pi=acos(-one)
  m=(n+1)/2
  xm=hf*(x1+x2)
  xl=hf*(x2-x1)
  
  do j=1,m
     z=cos(pi*(real(j,mp)-qt)/(real(n,mp)+hf))
1    continue
     p1=one
     p2=zero
     do k=1,n
        p3=p2
        p2=p1
        p1=((two*real(k,mp)-one)*z*p2 -(real(k,mp)-one)*p3)/real(k,mp)
     enddo
     pp=n*(z*p1-p2)/(z*z-one)
     z1=z
     z=z1-p1/pp
     if (abs(z-z1)>eps) then
        goto 1
     else
!            print*, "z, z1, |z-z1|, eps", z, z1, dabs(z-z1), eps
     endif
     gp(j) = xm - xl*z
     gp(n+1-j) = xm + xl*z
     gw(j) = two*xl/((one-z*z)*pp*pp)
     gw(n+1-j) = gw(j)    
  enddo
end subroutine gauss_leg

