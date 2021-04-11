  subroutine gauss_stiffness_mengs
    ! Nov. 7, 2018, Written by Dongwoo Sheen (sheen@snu.ac.kr)
    ! http://www.nasc.snu.ac.kr
    ! 2D elliptic problem solver  using the DSSY nonconforming FEM
    use prec
    use common_var
    use maps
      use omp_lib

    implicit none
    
    integer:: j, k, l, m
    integer:: jg
    real(mp):: x(2), kap, jac_bAK, gp_K(mg_quad,2)
    real(mp):: gp_quad(mg_quad,2), gw_quad(mg_quad)
    interface
       function phi(l,x,j,k,m)
         use prec
         use common_var
         integer, intent(in):: l,j,k,m
         real(mp), intent(in):: x(2)
         real(mp):: phi
       end function phi

       subroutine quad_bK(j,k,gp_quad,gw_quad)
         use prec
         use common_var

         implicit none
         integer, intent(in):: j,k
         real(mp),intent(out):: gp_quad(mg_quad,2), gw_quad(mg_quad)
       end subroutine quad_bK
       
    end interface

    a(:,:,:,:)  = zero
!$omp parallel
!$omp do

    do l = 1, 4
       do m = l, 4
          do j = 1, nx
             do k = 1, ny
                call quad_bK(j,k,gp_quad,gw_quad)
                do jg = 1, mg_quad
                   call bAK(j,k,jac_bAK,gp_quad(jg,:),x)
                   a(l,m,j,k)=a(l,m,j,k) + kap(x)*( phi(l,x,j,k,1)*phi(m,x,j,k,1) &
                        + phi(l,x,j,k,2)*phi(m,x,j,k,2))*gw_quad(jg)*jac_bAK
                end do
             end do
          end do
       end do
    end do
!$omp end do
!$omp end parallel
    
    a(2,1,:,:) = a(1,2,:,:); a(3,1,:,:) = a(1,3,:,:); a(3,2,:,:) = a(2,3,:,:);
    a(4,1,:,:) = a(1,4,:,:); a(4,2,:,:) = a(2,4,:,:); a(4,3,:,:) = a(3,4,:,:);
    
   !print *, "gauss_stif=", a(:,:,2,2)
  end subroutine gauss_stiffness_mengs
  subroutine gauss_stiffness_tensr
  ! Nov. 7, 2018, Written by Dongwoo Sheen (sheen@snu.ac.kr)
    ! http://www.nasc.snu.ac.kr
  ! 2D elliptic problem solver  using the DSSY nonconforming FEM
    use prec
    use common_var
    use omp_lib
    implicit none
    integer:: j, k, l, m, pauseint
    integer:: jg
    real(mp):: x(2), kap
    real(mp):: J_F_K(mg_tensr*mg_tensr), gw_K(mg_tensr*mg_tensr), gp_K(mg_tensr*mg_tensr,2)

    interface
       subroutine tensor_prod(j,k,J_F_K,gp_K,gw_K)
         use prec
         use common_var
         implicit none
         integer, intent(in) :: j,k
         real(mp), intent(out) :: J_F_K(1:mg_tensr*mg_tensr), gp_K(mg_tensr*mg_tensr,2), gw_K(1:mg_tensr*mg_tensr)
       end subroutine tensor_prod
       
       function phi(l,x,j,k,m)
         use prec
         use common_var
         integer, intent(in):: l,j,k,m
         real(mp), intent(in):: x(2)
         real(mp):: phi
       end function phi
    end interface
    
    a(:,:,:,:)  = 0._mp
!$omp parallel
!$omp do

    do l = 1, 4
       do m = l, 4
          do j = 1, nx
             do k = 1, ny
                call tensor_prod(j,k,J_F_K,gp_K,gw_K)
                ! print *, "J_F_K=", J_F_K
                ! print *, "gp_K(:,1)=", gp_K(:,1); stop 774
                ! print *, "gw_K=", gw_K
                ! read(*,*), pauseint
                do jg = 1, mg_tensr*mg_tensr
                   x(1) = gp_K(jg,1); x(2) = gp_K(jg,2);
                   a(l,m,j,k)=a(l,m,j,k) + kap(x)*(phi(l,x,j,k,1)*phi(m,x,j,k,1)&
                        +phi(l,x,j,k,2)*phi(m,x,j,k,2))*gw_K(jg)*J_F_K(jg)
                end do
             end do
          end do
       end do
    end do
!$omp end do
!$omp end parallel
    
    a(2,1,:,:) = a(1,2,:,:); a(3,1,:,:) = a(1,3,:,:); a(3,2,:,:) = a(2,3,:,:);
    a(4,1,:,:) = a(1,4,:,:); a(4,2,:,:) = a(2,4,:,:); a(4,3,:,:) = a(3,4,:,:);

  !print *, "gauss_stif=", a(:,:,2,2)
  end subroutine gauss_stiffness_tensr

