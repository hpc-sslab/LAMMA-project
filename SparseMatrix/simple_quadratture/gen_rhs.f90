  subroutine gen_rhs_tensr(f_al,f_be)
  ! Nov. 7, 2018, Written by Dongwoo Sheen (sheen@snu.ac.kr) http://www.nasc.snu.ac.kr
    ! 2D elliptic problem solver  using the DSSY nonconforming FEM
    !
    !                             be(j,k) (x_j,y_k)
    !                      -------------------------
    !                         |     2      |
    !              al(j-1,k)  |3          1| al(j,k)
    !                         |     4      |
    !                      -------------------------
    !                             be(j,k-1)
    !* This (al,be) is the rotated version from the original dssy.f90:
    !
    !                             al(j,k) (x_j,y_k)
    !                      -------------------------
    !                         |     2      |
    !              be(j-1,k)  |3          1| be(j,k)
    !                         |     4      |
    !                      -------------------------
    !                             al(j,k-1)

    use prec
    use common_var
      use omp_lib

    implicit none
    real(mp):: f_al(0:nx,1:ny), f_be(1:nx,0:ny)
    real(mp):: gauss_fphi_tensr
    integer:: j, k, l
    f_al = zero; f_be = zero
!$omp parallel
!$omp do

    do j =1, nx-1
       do k = 1, ny
          f_al(j,k) = f_al(j,k) + gauss_fphi_tensr(1,j,k)+ gauss_fphi_tensr(3,j+1,k)
       end do
    end do
    
    do j =1, nx 
       do k = 1, ny-1
          f_be(j,k) = f_be(j,k) + gauss_fphi_tensr(2,j,k)+ gauss_fphi_tensr(4,j,k+1)
       end do
    end do
!$omp end do
!$omp end parallel
    
  end subroutine gen_rhs_tensr
  
  subroutine gen_rhs_mengs(f_al,f_be)
  ! Nov. 7, 2018, Written by Dongwoo Sheen (sheen@snu.ac.kr) http://www.nasc.snu.ac.kr
    ! 2D elliptic problem solver  using the DSSY nonconforming FEM
    !
    !                             be(j,k) (x_j,y_k)
    !                      -------------------------
    !                         |     2      |
    !              al(j-1,k)  |3          1| al(j,k)
    !                         |     4      |
    !                      -------------------------
    !                             be(j,k-1)
    use prec
    use common_var
  use omp_lib
    implicit none
    real(mp):: f_be(1:nx,0:ny), f_al(0:nx,1:ny)
    real(mp):: gauss_fphi_mengs
    integer:: j, k, l
    f_al = zero; f_be = zero

!$omp parallel
!$omp do
    do j =1, nx-1
       do k = 1, ny
          f_al(j,k) = f_al(j,k) + gauss_fphi_mengs(1,j,k)+ gauss_fphi_mengs(3,j+1,k)
       end do
    end do
    
    do j =1, nx 
       do k = 1, ny-1
          f_be(j,k) = f_be(j,k) + gauss_fphi_mengs(2,j,k)+ gauss_fphi_mengs(4,j,k+1)
       end do
    end do
!$omp end do
!$omp end parallel
 
  end subroutine gen_rhs_mengs
