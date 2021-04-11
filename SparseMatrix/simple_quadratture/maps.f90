module maps
  use prec
  implicit none

contains

  subroutine vv(j,k,v1,v2,v3,v4)
    use common_var
    implicit none
    integer,intent(in):: j,k
    real(mp) :: v1(2), v2(2), v3(2), v4(2)
    v1(1) = xj(j,k);     v1(2) = yk(j,k);
    v2(1) = xj(j-1,k);   v2(2) = yk(j-1,k);
    v3(1) = xj(j-1,k-1); v3(2) = yk(j-1,k-1);
    v4(1) = xj(j,k-1);   v4(2) = yk(j,k-1);
  end subroutine vv

  subroutine hh(j,k,h)
    use common_var
    implicit none
    integer,intent(in):: j,k
    real(mp), dimension(2):: h, v1,v2,v3,v4

    call vv(j,k,v1,v2,v3,v4)
    h(1) = ell(2,v3,j,k,0)   ! = ell(2,v_3) < 0
    h(2) = ell(1,v4,j,k,0)   ! = ell(1,v_4) < 0
  end subroutine hh

  function ell(l,x,j,k,m)
    !On quadrilateral R_jk; vertices v1=(j,k), v2=(j-1,k), v3=(j-1,k-1), v4=(j,k-1)
    !ell(1,v1)=ell(1,v3)=0, ell(1,v2)=1, ell(1,v4)=h(2)
    ! (x-v3)//(v1-v3);  (x-v3)x(v1-v3)=0
    !ell(2,v2)=ell(2,v4)=0, ell(2,v1)=1, ell(2,v3)=h(1)
    ! (x-v4) // (v2-v4)
    use prec
    use common_var
    implicit none
    integer,intent(in):: l,j,k,m
    real(mp), intent(in):: x(2)
    real(mp):: ell
    real(mp) :: v1(2), v2(2), v3(2), v4(2)

    call vv(j,k,v1,v2,v3,v4)

    select case(l)
    case(1)
       if (m==0) then
          ell = cross(x-v3,v1-v3) 
       else if (m==1) then
          ell =(v1(2)-v3(2))
       else if (m==2) then
          ell = -(v1(1)-v3(1))
       end if
       ell = ell/cross(v2-v3,v1-v3)
    case(2)
       if (m==0) then
          ell = cross(x-v4,v2-v4)
       else if (m==1) then
          ell = (v2(2)-v4(2))
       else if (m==2) then
          ell = -(v2(1)-v4(1))
       end if
       ell = ell/cross(v1-v4,v2-v4)
    case default
       print*," ell(1) or ell(2) only "
    end select
  end function ell

  function q(x,j,k,m) ! these are defined on physical domain K_jk
    use prec
    use common_var
    implicit none
    integer:: j, k,m
    real(mp):: q,x(2), h(2), hp(2)
    real(mp) :: v1(2), v2(2), v3(2), v4(2)

    call vv(j,k,v1,v2,v3,v4)
    call hh(j,k,h)
    
    hp=one+h

    if (m==0) then
       q = ell(2,x,j,k,m)**2 - 3._mp/10._mp*(hp(1))*ell(2,x,j,k,m) +3._mp/20._mp*h(1) &
            +c_tilde*(ell(1,x,j,k,m)**2 - 3._mp/10._mp*(hp(2))*ell(1,x,j,k,m) &
            + 3._mp/20._mp*h(2))
    else if (m==1 .or. m==2) then
       q = 2._mp*ell(2,x,j,k,0)*ell(2,x,j,k,m) - 3._mp/10._mp*(hp(1))*ell(2,x,j,k,m) &
            +c_tilde*(2._mp*ell(1,x,j,k,0)*ell(1,x,j,k,m) - 3._mp/10._mp*(hp(2))*ell(1,x,j,k,m))
    end if
  end function q

  function mu(x,j,k,m) ! these are defined on physical domain K_jk
    use prec
    use common_var
    implicit none
    integer:: j, k,m
    real(mp):: mu,x(2),h(2),hp(2)
    real(mp) :: v1(2), v2(2), v3(2), v4(2)

    call vv(j,k,v1,v2,v3,v4)
    call hh(j,k,h)

    hp=one+h
    if (m==0) then
       mu = ell(1,x,j,k,m)*ell(2,x,j,k,m)*q(x,j,k,m);
    else if(m==1 .or. m==2) then
       mu = ell(1,x,j,k,m)*ell(2,x,j,k,0)*q(x,j,k,0) &
            + ell(1,x,j,k,0)*ell(2,x,j,k,m)*q(x,j,k,0) &
            + ell(1,x,j,k,0)*ell(2,x,j,k,0)*q(x,j,k,m) 
    end if
    !    mu = mu*(-5._mp/3._mp)
  end function mu

  function b_q(b_x,h,m) ! these are defined on Meng ref. dom b{K_jk}
    use prec
    use common_var
    implicit none
    integer,intent(in):: m
    real(mp), intent(in):: b_x(2),h(2)
    real(mp):: b_q, hp(2)

    hp=one+h

    if (m==0) then
       b_q = b_x(1)**2 - 3._mp/10._mp*hp(1)*b_x(1) +3._mp/20._mp*h(1) &
            +c_tilde*(b_x(2)**2 - 3._mp/10._mp*hp(2)*b_x(2) + 3._mp/20._mp*h(2))
    else if (m==1) then
       b_q = 2._mp*b_x(1) - 3._mp/10._mp*hp(1)
    else if (m==1 .or. m==2) then
       b_q = c_tilde*(2._mp*b_x(2) - 3._mp/10._mp*hp(2))
    end if
  end function b_q

  function b_mu(b_x,h,m) ! these are defined on physical domain K_jk
    use prec
    use common_var
    implicit none
    integer,intent(in):: m
    real(mp), intent(in):: b_x(2),h(2)
    real(mp):: b_mu

    if (m==0) then
       b_mu = b_x(1)*b_x(2)*b_q(b_x,h,m);
    else if(m==1) then
       b_mu = b_x(2)*b_q(b_x,h,0) + b_x(1)*b_x(2)*b_q(b_x,h,m)
    else if(m==2) then
       b_mu = b_x(1)*b_q(b_x,h,0) + b_x(1)*b_x(2)*b_q(b_x,h,m)
    end if
  end function b_mu


  function t_ell(j,t_x,s,m) !m=0 ftn val; m=1 gradient wrt x_1, m=2 gradient wrt x_2
    use prec
    implicit none
    real(mp), dimension(2), intent(in):: t_x,s
    integer, intent(in):: j, m
    real(mp):: t_ell
    select case(m)
    case(0)
       if (j==1) then
          t_ell = t_x(1)-t_x(2)-s(1)+s(2)
       else if (j==2) then
          t_ell = t_x(1)+t_x(2)+s(1)+s(2)
       end if
    case(1) !derivative wrt t_x(1)
       t_ell = 1._mp
    case(2) !derivative wrt t_x(2)
       t_ell = (-1._mp)**j
    end select
  end function t_ell

  function t_q(t_x,s,m) !formula (2.8) and (2.12) [DSSY-np] modified for basis ftns of
    use prec
    implicit none
    real(mp), dimension(2), intent(in):: t_x, s
    integer:: m
    real(mp) :: t_q
    select case(m)
    case(0)
       t_q=(t_x(1)+2._mp*s(2)/5._mp)**2+(t_x(2)+2._mp*s(1)/5._mp)**2 &
            -3._mp/5._mp*(1._mp-2._mp/5._mp*(s(1)**2+s(2)**2))
    case(1)
       t_q = 2._mp*(t_x(1)+2._mp*s(2)/5._mp)
    case(2)
       t_q = 2._mp*(t_x(2)+2._mp*s(1)/5._mp)
    end select
  end function t_q

  function t_mu(t_x,s,m) !formula (2.8) and (2.12) [DSSY-np] modified for basis ftns of
    use prec
    implicit none
    real(mp), dimension(2), intent(in):: t_x, s
    integer:: m
    real(mp) :: t_mu
    select case(m)
    case(0)
       t_mu = t_ell(1,t_x,s,0)*t_ell(2,t_x,s,0)*t_q(t_x,s,0)
    case(1:2) !first derivatives
       t_mu = t_ell(1,t_x,s,m)*t_ell(2,t_x,s,0)*t_q(t_x,s,0) &
            + t_ell(1,t_x,s,0)*t_ell(2,t_x,s,m)*t_q(t_x,s,0) &
            + t_ell(1,t_x,s,0)*t_ell(2,t_x,s,0)*t_q(t_x,s,m)
    end select
  end function t_mu

  subroutine bAK(j,k,jac_bAK,b_x,x) ! this is equal to {bar A}_K in our notation
    use common_var
    integer:: m
    integer, intent(in) :: j,k
    real(mp):: b_zero(2)
    real(mp):: mat(2,2), c(2)
    real(mp), intent(in) :: b_x(2)
    real(mp), intent(out):: jac_bAK, x(2)
    b_zero=0._mp
    !if x_1 = ell(1,x)=a11 b_x_1 + a12 b_x_2 + c1
    !   x_2 = ell(2,x)=a21 b_x_1 + a22 b_x_2 + c2
    ! c = [c1; c2];
    do m = 1, 2   !should change ell(1,...) and ell(2,...) and replace by inv_bA_K and ginv_bA_K
       mat(1,m)= ell(2,b_zero,j,k,m)  ! mat(1,m) = a1m
       mat(2,m)= ell(1,b_zero,j,k,m)  ! mat(2,m) = a2m
    end do ! b_x = mat*x + c
    c(1) = ell(2,b_zero,j,k,0)
    c(2) = ell(1,b_zero,j,k,0)

    mat = inv_of(mat) ! for x = (mat)^{-1}(b_x - c)
    x = matmul(mat, b_x-c)
    jac_bAK = abs(det(mat))
  end subroutine bAK

  function F_K(hx,v1,v2,v3,v4) result(x)
    real(mp), dimension(2), intent(in):: hx, v1,v2,v3,v4
    real(mp), dimension(2) :: x
    real(mp), dimension(2):: vec_d
    real(mp):: zero=0._mp, one=1._mp, hf=.5_mp, qt=.25_mp

    vec_d = (v1-v2+v3-v4)*qt
    x = v1 + hf*(one-hx(1))*(v2-v1)+hf*(one-hx(2))*(v4-v1)&
         +(one-hx(1))*(one-hx(2))*vec_d
  end function F_K

  function gF_K(hx,v1,v2,v3,v4) result(mat) !Jacobian matrix of F_K
    real(mp), dimension(2), intent(in):: hx,v1,v2,v3,v4
    real(mp), dimension(2,2) :: mat
    real(mp), dimension(2):: vec_d
    real(mp):: one=1._mp, hf=.5_mp, qt=.25_mp

    vec_d = qt*(v1-v2+v3-v4)
    mat(:,1) = -hf*(v2-v1) - (one-hx(2))*vec_d
    mat(:,2) = -hf*(v4-v1) - (one-hx(1))*vec_d

  end function gF_K

  function bA_K(b_x,v1,v2,v3,v4,h) result(x)
    real(mp), intent(in):: b_x(2)
    real(mp), dimension(2), intent(in)::v1,v2,v3,v4,h
    real(mp), dimension(2):: x
    real(mp) :: b_A(2,2), b_xi(2), s(2)
    real(mp):: zero=0._mp, one=1._mp, two=2._mp, four=4._mp

    b_A(:,1) = (v1-v3)/(one-h(1))
    b_A(:,2) = (v2-v4)/(one-h(2))
    b_xi = (v3-h(1)*v1)/(one-h(1))
    x = matmul(b_A,b_x)+b_xi
  end function bA_K

    function inv_bA_K(j,k,x) result(b_x) ! this is equal to inverser of {bar A}_K in our notation
      use common_var
      integer:: m
      integer, intent(in) :: j,k
      real(mp), intent(in) :: x(2)
      real(mp):: b_x(2)
      b_x(1) = ell(2,x,j,k,0)
      b_x(2) = ell(1,x,j,k,0)
    end function inv_bA_K

    function ginv_bA_K(j,k) result(mat) ! this is equal to {bar A}_K in our notation Jacobian matrix of inv_bA_K, gradient(inv_bA_K)
    use common_var
    integer, intent(in) :: j,k
    integer:: m
    real(mp):: mat(2,2), b_zero(2)
    b_zero = zero
    do m = 1, 2
       mat(1,m)= ell(2,b_zero,j,k,m)
       mat(2,m)= ell(1,b_zero,j,k,m)
    end do
  end function ginv_bA_K
  
  function tA_K(t_x,v1,v2,v3,v4) result(x) !maps t K to K p.1787 nonparaDSSY
    real(mp), intent(in)::t_x(2)
    real(mp), dimension(2), intent(in)::v1,v2,v3,v4
    real(mp), dimension(2)::x
    real(mp) :: A(2,2), b(2), invA(2,2)
    real(mp) :: detA
    real(mp):: qt=.25

    A(:,1) = (v1-v2-v3+v4)*qt;     A(:,2) = (v1+v2-v3-v4)*qt
    b = (v1+v2+v3+v4)*qt

    x = matmul(A,t_x) + b
  end function tA_K

  function inv_tA_K(x,v1,v2,v3,v4) result(t_x) !maps K to t K
    real(mp), intent(in)::x(2)
    real(mp), dimension(2), intent(in)::v1,v2,v3,v4
    real(mp), dimension(2)::t_x
    real(mp) :: A(2,2), b(2), invA(2,2)
    real(mp) :: detA
    real(mp):: qt=.25
    real(mp):: zero=0._mp, one=1._mp, two=2._mp, four=4._mp

    A(:,1) = (v1-v2-v3+v4)*qt;     A(:,2) = (v1+v2-v3-v4)*qt
    b = (v1+v2+v3+v4)*qt
    invA = inv_of(A)
    t_x = matmul(invA,x-b)
  end function inv_tA_K

  function B_K(t_x,s) result(b_x)
    real(mp), dimension(2), intent(in)::t_x ,s
    real(mp), dimension(2) :: b_x
    real(mp):: zero=0._mp, one=1._mp, two=2._mp, four=4._mp

    b_x(1) = (t_x(1)-t_x(2)-s(1)+s(2) )/(two* (one-s(1)+s(2)))
    b_x(2) = (t_x(1)+t_x(2)+s(1)+s(2))/(two* (one+s(1)+s(2)))
  end function B_K

  function invB_K(b_x,s) result(t_x)
    real(mp), dimension(2), intent(in):: b_x,s
    real(mp), dimension(2) :: t_x
    real(mp):: zero=0._mp, one=1._mp, two=2._mp, four=4._mp

    t_x(1) =(one-s(1)+s(2))*b_x(1)+(one+s(1)+s(2))*b_x(2)-s(2)
    t_x(2) =-(one-s(1)+s(2))*b_x(1)+(one+s(1)+s(2))*b_x(2)-s(1)
  end function invB_K

  function S_K(hat_x,s) result(t_x)
    real(mp), dimension(2), intent(in):: hat_x, s
    real(mp), dimension(2) :: t_x, tmp
    real(mp):: zero=0._mp, one=1._mp, two=2._mp, four=4._mp

    tmp = hat_x(1)*hat_x(2)
    t_x = hat_x+ tmp*s
  end function S_K

  function invS_K(t_x,s) result(hat_x)
    real(mp), dimension(2), intent(in):: t_x, s
    real(mp), dimension(2) :: hat_x, tmp
    real(mp):: zero=0._mp, one=1._mp, two=2._mp, four=4._mp

    if (s(1) == zero) then
       if (s(2) == zero) then
          hat_x = t_x
       else
          hat_x(1) = t_x(1)
          hat_x(2) = t_x(2)/(one+s(2)*t_x(1))
          if (hat_x(1)**2 > one .or. hat_x(1)**2>1 ) stop 800
       endif
    else
       if (s(2) == zero) then
          hat_x(1) = t_x(1)/(one+s(1)*t_x(2))
          hat_x(2) = t_x(2)
          if (hat_x(1)**2 > one .or. hat_x(1)**2>1 ) stop 801
       else !s(1)s(2) /= 0.
          hat_x(1) = two*t_x(1)/( (one-s(2)*t_x(1)+s(1)*t_x(2))  &
               +sqrt((one-s(2)*t_x(1)+s(1)*t_x(2))**2+four*s(2)*t_x(1)) )
          hat_x(2) = two*t_x(2)/ ((one+s(2)*t_x(1)-s(1)*t_x(2)) &
               +sqrt((one+s(2)*t_x(1)-s(1)*t_x(2))**2+four*s(1)*t_x(2)))
          if (hat_x(1)**2 > one .or. hat_x(1)**2>1 ) stop 802
       endif
    endif
    if (hat_x(1)**2 > one .or. hat_x(2)**2 > one) then
       print*, "hat_x=", hat_x, "   t_x =", t_x
       stop 934
    end if
  end function invS_K

  function area_tri(v1,v2,v3)
    real(mp) :: area_tri
    real(mp) :: v1(2), v2(2), v3(2)
    real(mp):: zero=0._mp, one=1._mp, two=2._mp, four=4._mp

    area_tri = abs(v1(1)*v2(2)+v2(1)*v3(2)+v3(1)*v1(2) &
         -v1(2)*v2(1)-v2(2)*v3(1)-v3(2)*v1(1))/two
  end function area_tri

  function area_quad(v1,v2,v3,v4)
    real(mp) :: area_quad
    real(mp) :: v1(2), v2(2), v3(2), v4(2)

    area_quad = area_tri(v1,v2,v4)+area_tri(v2,v3,v4)
  end function area_quad

  function ss(v1,v2,v3,v4) result(s)
    real(mp), dimension(2), intent(in)::v1,v2,v3,v4
    real(mp), dimension(2)::s
    real(mp) :: a(2,2), inv_a(2,2), det_a
    real(mp) :: vec_d(2)
    real(mp):: zero=0._mp, one=1._mp, two=2._mp, four=4._mp

    inv_a = zero ; vec_d = zero
    s = zero 
    a(:,1) = (v1-v2-v3+v4)/four
    a(:,2) = (v1+v2-v3-v4)/four 
    vec_d = (v1-v2+v3-v4)/four

    inv_a = inv_of(A)

    s = matmul(inv_a,vec_d)

    if (abs(s(1)) + abs(s(2))>one) then
       print*, '=============',s(1),s(2)
       stop 998
    end if
  end function ss


  function cross(a,b)
    real(mp), dimension(:), intent(in):: a, b
    integer:: n
    real(mp):: cross 
    if (size(a) .ne. size(b)) then
       write(*,*) 'Error :Input must be two vectors of same size'
       return
    end if
    n = size(a);  
    if (n==2) then
       cross = a(1)*b(2)-a(2)*b(1)
    end if
  end function cross

  function cross_prod(a,b) result(axb) !cross product
    real(mp), dimension(:), intent(in):: a, b
    integer:: asize(2), bsize(2), j,k, n
    real(mp), dimension(:), allocatable:: axb,u,v
    if (size(a) .ne. size(b)) then
       write(*,*) 'Error :Input must be two vectors of same size'
       return
    end if
    n = size(a);  
    if (n==2) then
       allocate(axb(1))
       axb = a(1)*b(2)-a(2)*b(1)
    else if (n==3) then
       allocate(axb(n))
       axb(1)=a(2)*b(3)-a(3)*b(2)
       axb(2)=a(3)*b(1)-a(1)*b(3)
       axb(3)=a(1)*b(2)-a(2)*b(1)
    else if (n==7) then
       allocate(axb(n),u(0:n),v(0:n))
       u(0)=a(7); u(1:7)=a;       v(0)=b(7); v(1:7)=b;
       do j = 1, 7
          axb(j)=u(mod(j+1,7))*v(mod(j+3,7))-u(mod(j+3,7))*v(mod(j+1,7)) &
               + u(mod(j+2,7))*v(mod(j+6,7))-u(mod(j+6,7))*v(mod(j+2,7)) &
               + u(mod(j+4,7))*v(mod(j+5,7))-u(mod(j+5,7))*v(mod(j+4,7))
       end do
    else 
       write(*,*) 'Error: only for n=2,3,7'
       return
    end if
  end function cross_prod

  function inv_of(A) result(invA) ! matrix inverse
    real(mp), dimension(:,:), intent(in):: A
    integer:: msize(2), j,k, n
    real(mp), dimension(:,:), allocatable:: invA
    real(mp):: detA, sgn
    real(mp):: zero=0._mp, one=1._mp, two=2._mp, four=4._mp

    msize = shape(A)
    if (msize(1) .ne. msize(2)) then
       write(*,*) 'Error in Matrix: Input must be a square Matrix!'
       return
    end if
    n = msize(1)
    detA = det(A)
    if (detA == zero) stop "detA = 0"

    allocate(invA(n,n))

    if (n .eq. 1) then
       ! determinant of matrix 1 x 1, i.e. number
       detA = A(1,1); invA(1,1)=one
    else
       do j=1, n
          do k =1, n
             sgn = (-one)**(j+k)
             invA(k,j) = sgn*det(cofactor(A,j,k))
          end do
       end do
    end if
    invA = invA/detA
  end function inv_of

  function cofactor(matrix, mI, mJ)
    real(mp), dimension(:,:), intent(in) :: matrix
    integer, intent(in) :: mI, mJ
    integer :: msize(2), i, j, k, l, n
    real(mp), dimension(:,:), allocatable :: cofactor
    msize = shape(matrix)
    if (msize(1) .ne. msize(2)) then
       write(*,*) 'Error in Cofactor: Input must be a square Matrix!'
       return
    end if
    n = msize(1)
    !
    allocate(cofactor(n-1, n-1))

    k = 1
    do i=1, n
       if (i .ne. mI) then
          l = 1
          do j=1, n
             if (j .ne. mJ) then
                cofactor(k,l) = matrix(i,j)
                l = l+ 1
             end if
          end do
          k = k+ 1
       end if
    end do
    return
  end function cofactor

  ! Expansion of determinants using Laplace formula
  recursive function det(matrix) result(laplace_det)
    real(mp), dimension(:,:), intent(in) :: matrix
    integer :: msize(2), i, n, sgn
    real(mp) :: laplace_det, detA
    real(mp), dimension(:,:), allocatable :: cf

    msize = shape(matrix)
    if (msize(1) .ne. msize(2)) then
       write(*,*) 'Error in Determinant: Input must be a square Matrix!'
       return
    end if
    n = msize(1)

    if (n .eq. 1) then
       ! determinant of matrix 1 x 1, i.e. number
       detA = matrix(1,1)
    else
       detA = 0
       do i=1, n
          sgn = merge(1, -1, mod(i,2) .eq. 1)
          allocate(cf(n-1, n-1))
          cf = cofactor(matrix, i, 1)
          detA = detA + sgn * matrix(i,1) * det(cf)
          deallocate(cf)
       end do
    end if
    !
    laplace_det = detA
  end function det

end module maps
