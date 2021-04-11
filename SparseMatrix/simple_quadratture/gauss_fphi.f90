  function gauss_fphi_mengs(l,j,k)
    use prec
    use common_var
    use maps
    use omp_lib
    implicit none
    real(mp):: f, phi
    integer:: l,j,k, jg
    real(mp):: tmp,x(2),gauss_fphi_mengs, hx_unif, hy_unif, jac_bAK, gp_K(mg_quad,2)
    real(mp):: gp_bK(mg_quad,2), gw_bK(mg_quad)
    
    interface
       subroutine quad_bK(j,k,gp_bK,gw_bK)
         use prec
         use common_var
         implicit none
         integer, intent(in):: j,k
         real(mp),intent(out):: gp_bK(mg_quad,2), gw_bK(mg_quad)
       end subroutine quad_bK
    end interface
    
    gauss_fphi_mengs = zero
    call quad_bK(j,k,gp_bK,gw_bK) !gp_bK in bar{K}

!$omp parallel
!$omp do
    do jg = 1, mg_quad
       call bAK(j,k,jac_bAK,gp_bK(jg,:),x) !gp_bK in {bar K} to x in K
       gauss_fphi_mengs = gauss_fphi_mengs + f(x)*phi(l,x,j,k,0)*gw_bK(jg)*jac_bAK
    end do
!$omp end do
!$omp end parallel
    
  end function gauss_fphi_mengs

    function gauss_fphi_tensr(l,j,k)
  ! Nov. 7, 2018, Written by Dongwoo Sheen (sheen@snu.ac.kr)
    ! http://www.nasc.snu.ac.kr
  ! 2D elliptic problem solver  using the DSSY nonconforming FEM
    use prec
    use common_var
    use omp_lib
    implicit none
    real(mp):: f, phi
    integer:: j, k, l, jg, kg
    real(mp):: x(2), gauss_fphi_tensr
    real(mp):: jac(mg_tensr*mg_tensr), gw_K(mg_tensr*mg_tensr), gp_K(mg_tensr*mg_tensr,2)
    interface
       subroutine tensor_prod(j,k,jac,gp_K,gw_K)
         use prec
         use common_var
         implicit none
         integer, intent(in) :: j,k
         real(mp), intent(out) :: jac(1:mg_tensr*mg_tensr), gp_K(mg_tensr*mg_tensr,2), gw_K(1:mg_tensr*mg_tensr)
       end subroutine tensor_prod
    end interface

    gauss_fphi_tensr = zero
    
    call tensor_prod(j,k,jac,gp_K,gw_K)  !gp_K in K
!$omp parallel
!$omp do

    do jg = 1, mg_tensr*mg_tensr
       x(1) = gp_K(jg,1); x(2) = gp_K(jg,2);
       gauss_fphi_tensr = gauss_fphi_tensr + f(x)* phi(l,x,j,k,0)*gw_K(jg)*jac(jg)
    end do
!$omp end do
!$omp end parallel
   
  end function gauss_fphi_tensr
