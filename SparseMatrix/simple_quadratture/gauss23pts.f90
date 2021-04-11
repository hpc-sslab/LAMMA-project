! This program computes n!
! Dongwoo Sheen (http://www.nasc.snu.kr) <dongwoosheen@gmail.com>

include "prec.f90"
function factorial(n)
 
  use prec
  implicit none
  real(mp):: n, fact, factorial

  fact = n;    n = n-1._mp
  
  do while (n>0._mp)
     fact = fact * n
     n = n-1._mp
  end do
  factorial = fact
end function factorial

function qint(hp,i,j) ! integral of poly on Meng's quadrilateral
  use prec
  implicit none
  real(mp), intent(in):: hp(2)
  integer:: i, j
  real(mp):: factorial, i_r, j_r, one, qint, two
  i_r = real(i,mp); j_r = real(j,mp);  one=1._mp; two=2._mp
  qint=factorial(i_r)*factorial(j_r)/factorial(two+i_r+j_r) &
       *(1._mp-(hp(1)-1._mp)**(i+1))*(1._mp-(hp(2)-1._mp)**(j+1))
end function qint

function mu_tmp(x,hp,m)
  use prec
  implicit none
  real(mp):: u,v,h,k,mu_tmp, x(2), hp(2)
  integer:: m
  u = x(1); v = x(2); h = hp(1); k=hp(2)
  select case (m)
  case (0)
     mu_tmp= u**3*v + u*v**3 -.3_mp*(h*u**2*v+ k*u*v**2) &
          + .15_mp*(h+k-2._mp)*u*v
  case (1)
     mu_tmp = 3._mp*u**2*v + v**3 -.3_mp*(2._mp*h*u*v + k*v**2) &
          + .15_mp*(h+k-2._mp)*v 
  case (2)
     mu_tmp = u**3 + 3._mp*u*v**2 -.3_mp* (h*u**2 + 2._mp*k*u*v) &
          + .15_mp*(h+k-2._mp)*u 
  case default
     stop 'mu_tmp index wrong'
  end select
end function mu_tmp

function f_vec(x,hp) result(y)
  use prec
  implicit none
  real(mp), dimension(2), intent(in):: x, hp
  real(mp), dimension(2):: y
  y(1) = 10._mp*hp(2)*x(1)**2 + 14._mp*hp(1)*x(1)*x(2)+7._mp*hp(2)*x(2)**2
  y(2) = 7._mp*hp(1)*x(1)**2 + 14._mp*hp(2)*x(1)*x(2) +10._mp*hp(1)*x(2)**2
end function f_vec
function J_f(x,hp) result(J)
  use prec
  implicit none
  real(mp), dimension(2), intent(in):: x, hp
  real(mp), dimension(2,2):: J
  J(1,1) = 20._mp*hp(2)*x(1) + 14._mp*hp(1)*x(2)
  J(1,2) = 14._mp*hp(1)*x(1) + 14._mp*hp(2)*x(2)
  J(2,1) = 14._mp*hp(1)*x(1) + 14._mp*hp(2)*x(2)
  J(2,2) = 14._mp*hp(2)*x(1) + 20._mp*hp(1)*x(2)
end function J_f

subroutine quad_bK(j,k,gp_bK,gw_bK) !return gauss points and weights on barK. quadrature_for_quadrilaterals
  use prec
  use common_var
  use maps
  implicit none
  integer, intent(in):: j,k
  real(mp),intent(out):: gp_bK(mg_quad,2), gw_bK(mg_quad) ! Quadrature by Meng
  real(mp),dimension(2):: x, v1, v2, v3, v4, h, c, d, c_opt(2)
  real(mp):: hp(2), r(2), den, T(6), res, f_val(2), RHS(2), tmp, area_K,theta
  integer:: pauseint, l,m
  integer:: max_newton_iter=30
  real(mp):: eps_newton = 1.e-7
  real(mp):: gab_1, gab_2, mu_1_int, mu_2_int, mu_tmp, qint
  logical :: chck_dom

  interface
     function choose_opt(c,d,h) result(c_opt) !Choose one such that hp/3+x is in the bar K domain:
       use prec                 !if |c| < |d|, choose c; otherwise choose d
       implicit none
       real(mp), dimension(2), intent(in):: c, d, h
       real(mp), dimension(2):: c_opt
     end function choose_opt
     function f_vec(x,hp) result(y)
       use prec
       implicit none
       real(mp), dimension(2), intent(in):: x, hp
       real(mp), dimension(2):: y
     end function f_vec

     function J_f(x,hp) result(J)
       use prec
       implicit none
       real(mp), dimension(2), intent(in):: x, hp
       real(mp), dimension(2,2):: J
     end function J_f

  end interface

  call vv(j,k,v1,v2,v3,v4)
  call hh(j,k,h)

  hp = h + 1._mp
  area_K = (1._mp-h(1))*(1._mp-h(2))/two

  gw_bK = area_K/real(mg_quad,mp)

  if(mg_quad==2 .or. mg_quad==3) then !the right hand side
     r(1)= hp(2)*(2._mp/90._mp*hp(1)**2 - .4_mp*hp(1) + 185._mp/999._mp*hp(2)**2 - .6_mp*hp(2) + 1._mp)
     r(2)= hp(1)*(2._mp/90._mp*hp(2)**2 - .4_mp*hp(2) + 185._mp/999._mp*hp(1)**2 - .6_mp*hp(1) + 1._mp)
  end if

  r = (135._mp/36._mp)*r*(real(mg_quad,mp)/3._mp) !adjust the RHS
!  select case(mg_quad)
!  case(2)
!     r = 90._mp*r;
!  case(3)
!     r = 135._mp*r
!  case default
!  end select

  RHS = 270._mp - 162._mp*hp+50._mp*hp**2 ! used for hp(1)=0 or hp(2)=0, derive formula separately
  RHS = RHS* (real(mg_quad,mp)/3._mp) !adjust the RHS
  
  select case(mg_quad)
  case (1)
     c=0._mp
  case(2,3) !two and three points formula new simple formula 

     if (hp(2) .eq. 0._mp) then !if hp(1)=0 or hp(2)=0, derive formula separately
           c(1) = sqrt(RHS(1)/504._mp);          c(2) = 0._mp;
           d(1) = 0._mp;              d(2) = sqrt(RHS(1)/720._mp)
           c = choose_opt(c,d,h)
     else if (hp(1) .eq. 0._mp) then !symmetric to the above case
           c(1) = 0._mp;            c(2) = sqrt(RHS(2)/504._mp)
           d(1) = sqrt(RHS(2)/720._mp);             d(2) = 0._mp;
           c = choose_opt(c,d,h)
     else
        select case(formula)
        case("newf")  !new formula
           T(1) = 7._mp*r(1)*hp(1) - 10._mp*r(2)*hp(2)
           T(2) = 13720._mp*hp(1)**4 - 26603._mp*hp(1)**2*hp(2)**2 + 13720._mp*hp(2)**4
           T(3) = -70._mp*r(1)**2*hp(1)**2 + 49._mp*r(1)**2*hp(2)**2 + 51._mp*r(1)*r(2)*hp(1)*hp(2) &
                + 49._mp*r(2)**2*hp(1)**2 - 70._mp*r(2)**2*hp(2)**2
           T(3) = sqrt(T(3))
           T(4) = 7._mp*(r(1)*hp(2)-r(2)*hp(1))
           T(5) = -1043._mp*r(1)*hp(1)**2*hp(2) + 980._mp*r(1)*hp(2)**3 &
                + 686._mp*r(2)*hp(1)**3 - 470._mp*r(2)*hp(1)*hp(2)**2
           T(6) = 14._mp*(7._mp*hp(1)**2 - 10._mp*hp(2)**2 ) *T(3)

           c(2) = sqrt((T(5)-T(6))/T(2)); c(1) = -((T(4)-T(3))/T(1))*c(2)
           d(2) = sqrt((T(5)+T(6))/T(2)); d(1) = -((T(4)+T(3))/T(1))*d(2)

           c = choose_opt(c,d,h)
        case("newn") !New formula with Newton's method
           x = hp/30._mp ! initial guess
           do iter = 1, max_newton_iter
              f_val = f_vec(x,hp)
              res = sqrt(dot_product(f_val-r, f_val-r))
              if (res < eps_newton) exit
              !           if (abs(det(J_f(x,hp))) < eps_newton*100.) print*, "det J_f(x,hp)=", det(J_f(x,hp)) 
              x = x - matmul(inv_of(J_f(x,hp)), f_val-r)
           end do
           !        if (iter > 29) print*, "iter=", iter, "   res = ", res
        case("meng")
        case default
           stop "No method of finding Gaussian quadrature"
        end select !end select (formula)
     end if
  case default
     stop "Gauss quadrature points are 1,2,3 only"
  end select !end select (mg_quad)

  select case(mg_quad)
     case(1)
        gp_bK(1,:) = hp/3._mp +c
     case(2)
        gp_bK(1,:) = hp/3._mp + c
        gp_bK(2,:) = hp/3._mp - c
     case(3)
        gp_bK(1,:) = hp/3._mp + c
        gp_bK(2,:) = hp/3._mp - c
        gp_bK(3,:) = hp/3._mp
        
        if(formula=="meng") then !formula (5) p.332 of Meng-Cui-Luo
           T(1)=(1._mp-h(1)+h(1)**2)/18._mp;
           T(2)=3._mp*T(1)*( (1._mp+h(1)**2)*(1._mp+h(2)**2) - 2._mp*(1._mp+h(1)*h(2))*(h(1)+h(2)))
           T(1)=sqrt(2._mp*T(1));  T(2)=sqrt(2._mp*T(2));
           
           theta = pi/2._mp
           do l = 1, mg_quad
              theta = theta + 2._mp/3._mp
              c(1)=T(1)*cos(theta)
              c(2)=(T(2)*sin(theta) - T(1)*hp(1)*hp(2)*cos(theta))/ &
                   (2._mp*(1._mp-h(1)+h(1)**2))
              gp_bK(l,:)= hp/3._mp + c
           end do
        end if
     end select

  if (mg_quad==3 .and. .false.) then
     mu_1_int=3._mp*qint(hp,2,1)+qint(hp,0,3) &
       -.3_mp*(2._mp*hp(1)*qint(hp,1,1)+hp(2)*qint(hp,0,2)) &
       +.15_mp*(hp(1)+hp(2)-2._mp)*qint(hp,0,1)
     mu_2_int=qint(hp,3,0)+3._mp*qint(hp,1,2) &
       -.3_mp*(hp(1)*qint(hp,2,0)  +2._mp*hp(2)*qint(hp,1,1)) &
       +.15_mp*(hp(1)+hp(2)-2._mp)*qint(hp,1,0)

     gab_1=0._mp; gab_2=0._mp
     do l = 1, mg_quad
        gab_1 = gab_1 + gw_bK(l)*mu_tmp(gp_bK(l,:),hp,1)
        gab_2 = gab_2 + gw_bK(l)*mu_tmp(gp_bK(l,:),hp,2)
     end do
     write(6,91) gab_1, mu_1_int, (gab_1 - mu_1_int)/mu_1_int
     write(6,92) gab_2, mu_2_int, (gab_2 - mu_2_int)/mu_2_int
  end if
91 format("gab_1, mu_1_int, rel.err =", 3g11.3)
92 format("gab_2, mu_2_int, rel.err =", 3g11.3)

end subroutine quad_bK

function chck_dom(x,h) !Check if x is in the bar K domain:
  use prec
  implicit none
  real(mp), dimension(2), intent(in):: x, h
  logical :: chck_dom
  chck_dom = .true.
  if ( (x(2) > - x(1) + 1._mp) & !e_1
       .or. ( x(2) > -1._mp/h(1)*x(1) +1._mp ) & !e_2
       .or. ( x(2) < -h(2)/h(1)*x(1) + h(2) ) & !e_3
       .or. ( x(2) < -h(2)*x(1) + h(2) ) ) then !e_4
!     print*, "The point",x, "is NOT in bar K; h=", h
     !           stop 565
     chck_dom = .false.
  end if
end function chck_dom

function choose_opt(c,d,h) result(c_opt) !Choose one such that hp/3+x is in the bar K domain:
  use prec                 !if |c| < |d|, choose c; otherwise choose d
  implicit none
  real(mp), dimension(2), intent(in):: c, d, h
  real(mp), dimension(2):: c_opt, h_c
  logical:: chck_dom
  h_c = (h + 1._mp)/3._mp
  c_opt = c

  if( (dot_product(d,d) < dot_product(c,c))) then
     if (chck_dom(h_c+d,h))         c_opt = d
  else if (chck_dom(h_c+c,h)) then
  else
     print*, "the points", h_c+c, h_c+d, " are not in bar K domain", h
     stop 987
  end if
end function choose_opt
