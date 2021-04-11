  function l2err(al,be)
  ! Nov. 7, 2018, Written by Dongwoo Sheen (sheen@snu.ac.kr) http://www.nasc.snu.ac.kr
  ! 2D elliptic problem solver  using the DSSY nonconforming FEM
    use prec
    use common_var
  use omp_lib
    implicit none
    real(mp), intent(in):: al(0:nx,1:ny), be(1:nx,0:ny)
    real(mp):: c(4)
    real(mp):: l2err, gauss_err
    integer:: j, k
    l2err = zero
!$omp parallel
!$omp do

    do j =1, nx
       do k = 1, ny
          c(1) = al(j,k); c(2) = be(j,k); c(3) = al(j-1,k); c(4) = be(j,k-1)
          l2err = l2err + gauss_err(j,k,c)
       end do
    end do
!$omp end do
!$omp end parallel

    l2err = sqrt(l2err)
  end function l2err

function gauss_err(j,k,c)
  ! Nov. 7, 2018, Written by Dongwoo Sheen (sheen@snu.ac.kr) http://www.nasc.snu.ac.kr
  ! 2D elliptic problem solver  using the DSSY nonconforming FEM
    use prec
    use common_var
  use omp_lib
    implicit none
    real(mp):: phi, u_ex, gauss_err
    integer:: j, k, l, jg, kg
    real(mp):: c(4), phival(4)
    real(mp):: uh, x(2), hx_unif,hy_unif
    real(mp), dimension(mg_err*mg_err):: jac_err, gw_err
    real(mp), dimension(mg_err*mg_err,2):: gp_err
    interface
       subroutine tensor_prod_err(j,k,jac_err,gp_err,gw_err)
         use prec
         use common_var
         implicit none
         integer, intent(in) :: j,k
         real(mp), dimension(mg_err*mg_err), intent(out):: jac_err, gw_err
         real(mp), dimension(mg_err*mg_err,2), intent(out):: gp_err
       end subroutine tensor_prod_err
    end interface

    gauss_err = zero
   
    call tensor_prod_err(j,k,jac_err,gp_err,gw_err)
!$omp parallel
!$omp do

    do jg = 1, mg_err*mg_err
       x(1) = gp_err(jg,1); x(2) = gp_err(jg,2);
       do l = 1, 4
          phival(l) = phi(l,x,j,k,0)
       end do
       uh = dot_product(c,phival)
       gauss_err = gauss_err + (uh-u_ex(x))**2*gw_err(jg)*jac_err(jg)
    end do
!$omp end do
!$omp end parallel
  
  end function gauss_err

function energy_err(al,be)
  use prec
  use common_var
  use omp_lib
  implicit none
  real(mp), intent(in):: al(0:nx,1:ny), be(1:nx,0:ny)
  real(mp):: c(4)
  real(mp):: energy_err, gauss_err_energy
  integer:: j, k
  energy_err = zero
!$omp parallel
!$omp do

  do j =1, nx
     do k = 1, ny
        c(1) = al(j,k); c(2) = be(j,k); c(3) = al(j-1,k); c(4) = be(j,k-1)
        energy_err = energy_err + gauss_err_energy(j,k,c)
     end do
  end do
!$omp end do
!$omp end parallel

  energy_err = sqrt(energy_err)
end function energy_err

function gauss_err_energy(j,k,c)
  use prec
  use common_var
  use omp_lib
  implicit none
  real(mp):: gauss_err_energy, kap
  integer:: j, k, l, jg, kg
  real(mp):: c(4), grad_x(4), grad_y(4), tmp(2)
  real(mp):: uh(2), x(2), hx_unif,hy_unif
  real(mp), dimension(mg_err*mg_err):: jac_err, gw_err
  real(mp), dimension(mg_err*mg_err,2):: gp_err

  interface
     subroutine tensor_prod_err(j,k,jac_err,gp_err,gw_err)
       use prec
       use common_var
       implicit none
       integer, intent(in) :: j,k
       real(mp), dimension(mg_err*mg_err), intent(out):: jac_err, gw_err
       real(mp), dimension(mg_err*mg_err,2), intent(out):: gp_err
     end subroutine tensor_prod_err
     
     function grad_uex(x)
       use prec
       use common_var
       implicit none
       real(mp):: x(2),grad_uex(2)
     end function grad_uex
     
     function phi(l,x,j,k,m) 
       use prec
       use common_var
       integer, intent(in):: l,j,k,m
       real(mp), intent(in):: x(2)
       real(mp):: phi
     end function phi
  end interface
  
  gauss_err_energy = zero
  
  call tensor_prod_err(j,k,jac_err,gp_err,gw_err)
!$omp parallel
!$omp do

  do jg = 1, mg_err*mg_err
     x(1) = gp_err(jg,1); x(2) = gp_err(jg,2);
     do l = 1, 4
        grad_x(l) = phi(l,x,j,k,1)
        grad_y(l) = phi(l,x,j,k,2)
     end do
     uh(1) = dot_product(c,grad_x); uh(2) = dot_product(c,grad_y);
     gauss_err_energy = gauss_err_energy + dot_product(uh-grad_uex(x),uh-grad_uex(x))*gw_err(jg)*jac_err(jg)
  end do
!$omp end do
!$omp end parallel
  
end function gauss_err_energy
