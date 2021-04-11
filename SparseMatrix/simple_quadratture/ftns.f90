function Kron_del(j,k)
  use prec
  integer:: j,k
  real(mp):: Kron_del
  if (j==k) then
     Kron_del = 1._mp
  else
     Kron_del = 0._mp
  end if
end function Kron_del


function u_ex(xx)
  use prec
  use common_var
  implicit none
  real(mp):: xx(2),u_ex,x,y
  x=xx(1); y=xx(2)

  if (epsil==zero) then
     u_ex = sin(2*pi*x)*sin(2*pi*y)*(x**3-y**4+x**2*y**3) !Meng's ex
  else
     u_ex = sin(3*pi*x)* y*(one-y)+ epsil*sin(pi*x/epsil)*sin(pi*y/epsil)
  end if
end function u_ex

function grad_uex(xx)
  use prec
  use common_var
  implicit none
  real(mp):: xx(2),x,y,grad_uex(2)
  x=xx(1); y=xx(2)

  if (epsil==zero) then
     grad_uex(1) = 2*pi*cos(2*pi*x)*sin(2*pi*y)*(x**3-y**4+x**2*y**3) &
          + sin(2*pi*x)*sin(2*pi*y)*(3*x**2+2*x*y**3)
     grad_uex(2) = 2*pi*sin(2*pi*x)*cos(2*pi*y)*(x**3-y**4+x**2*y**3) &
          + sin(2*pi*x)*sin(2*pi*y)*(-4*y**3 + 3*x**2*y**2)
  else 
     grad_uex(1) = 3*pi*cos(3*pi*x)*y*(one-y)+pi*cos(pi*x/epsil)*sin(pi*y/epsil)
     grad_uex(2) = sin(3*pi*x)*(one-2*y)+pi*sin(pi*x/epsil)*cos(pi*y/epsil)
  end if
end function grad_uex

function hess_uex(xx)
  use prec
  use common_var
  implicit none
  real(mp):: xx(2),x,y,hess_uex(2,2)
  x=xx(1); y=xx(2)
  hess_uex = zero
  if (epsil==zero) then
     hess_uex(1,1) = sin(2*pi*x)*sin(2*pi*y) &
          & *(-(2*pi)**2*(x**3-y**4+x**2*y**3) + 6*x + 2*y**3) &
          & + 4*pi*cos(2*pi*x)*sin(2*pi*y)*(3*x**2+2*x*y**3)
     hess_uex(2,2) = sin(2*pi*x)*sin(2*pi*y) &
          & *(-(2*pi)**2*(x**3-y**4+x**2*y**3) - 12*y**2 + 6*x**2*y) &
          & + 4*pi*sin(2*pi*x)*cos(2*pi*y)*(-4*y**3 + 3*x**2*y**2)
     hess_uex(1,2)=4.*pi**2*cos(2*pi*x)*cos(2*pi*y)*(x**3-y**4+x**2*y**3) &
          & + 2*pi*cos(2*pi*x)*sin(2*pi*y)*(-4.*y**3+ 3*x**2*y**2) &
          & + 2*pi*sin(2*pi*x)*cos(2*pi*y)*(3*x**2+ 2*x*y**3) &
          & + sin(2*pi*x)*sin(2*pi*y)*(6*x*y**2)
     hess_uex(2,1)=hess_uex(1,2)
  else 
     hess_uex(1,1) = -9*pi**2*sin(3*pi*x)*(y-y**2) - pi**2/epsil*sin(pi*x/epsil)*sin(pi*y/epsil)
     hess_uex(2,2) = -2*sin(3*pi*x) - pi**2/epsil*sin(pi*x/epsil)*sin(pi*y/epsil)
     hess_uex(1,2) = 3*pi*cos(3*pi*x)*(one-2*y) + pi**2/epsil*cos(pi*x/epsil)*cos(pi*y/epsil)
     hess_uex(2,1)=hess_uex(1,2)
  end if
end function hess_uex

function kap(xx)
  use prec
  use common_var
  implicit none
  real(mp):: xx(2),x,y,kap
  x=xx(1); y=xx(2)
  if (x<x_beg .or. x>x_end .or. y<y_beg .or. y>y_end) then
     kap = zero
  else
     if (epsil==zero) then
        kap = one
     else 
        kap = one + (one+x)*(one+y) + epsil*sin(10*pi*x)*sin(5*pi*y)
     end if
  end if
end function kap

function grad_kap(xx)
  use prec
  use common_var
  implicit none
  real(mp):: xx(2),x,y,grad_kap(2)
  x=xx(1); y=xx(2)
  if (x<x_beg .or. x>x_end .or. y<y_beg .or. y>y_end) then
     grad_kap = zero
  else
     if (epsil==zero) then
        grad_kap(1) =zero;           grad_kap(2) =zero;
     else 
        grad_kap(1) = (one+y) + 10*pi*epsil*cos(10*pi*x)*sin(5*pi*y)
        grad_kap(2) = (one+x) + 5*pi*epsil*sin(10*pi*x)*cos(5*pi*y)
     end if
  end if
end function grad_kap

function f(xx)
  use prec
  use common_var
  implicit none
  real(mp):: xx(2),f,kap, trace
  real(mp), dimension(:,:), allocatable:: hess
  interface
     function grad_kap(xx)
       use prec
       use common_var
       implicit none
       real(mp):: xx(2),grad_kap(2)
     end function grad_kap
     function u_ex(xx)
       use prec
       use common_var
       implicit none
       real(mp):: xx(2),u_ex
     end function u_ex

     function grad_uex(xx)
       use prec
       use common_var
       implicit none
       real(mp):: xx(2),grad_uex(2)
     end function grad_uex
     function hess_uex(xx)
       use prec
       use common_var
       implicit none
       real(mp):: xx(2),hess_uex(2,2)
     end function hess_uex

  end interface
  allocate(hess(2,2))
  hess = hess_uex(xx)
  trace = hess(1,1)+hess(2,2)
  f = -kap(xx)*trace - dot_product(grad_kap(xx), grad_uex(xx))
!!!!       if (epsil==zero) then
!!!!          f = 4*pi*(3*x**2+2*x*y**3)*cos(2*pi*x)*sin(2*pi*y) &
!!!!            +(6*x**2*y-12*y**2)*sin(2*pi*x)*sin(2*pi*y) &
!!!!            +4*pi*(3*x**2*y**2-4*y**3)*sin(2*pi*x)*cos(2*pi*y) &
!!!!            -8*pi**2*(x**3+x**2*y**3-y**4)*sin(2*pi*x)*sin(2*pi*y) &
!!!!            +(6*x+2*y**3)*sin(2*pi*x)*sin(2*pi*y)
!!!!          f = - f
!!!!        else
!!!!           f = 9*pi**2* sin(3*pi*x)*y*(one-y) + 2*sin(3*pi*x)&
!!!!                + 2*(pi*pi/epsil)*sin(pi*x/epsil)*sin(pi*y/epsil)
!!!!           f = kap(x,y)*f - dot_product(grad_kap(x,y),grad_uex(x,y))
!!!!        end if
end function f

