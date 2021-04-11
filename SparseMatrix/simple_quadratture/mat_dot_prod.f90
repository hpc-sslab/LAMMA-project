  function mat_dot_prod(a,b)
  ! Nov. 7, 2018, Written by Dongwoo Sheen (sheen@snu.ac.kr) http://www.nasc.snu.ac.kr
  ! 2D elliptic problem solver  using the DSSY nonconforming FEM
    use prec
  use omp_lib
    implicit none
    real(mp):: a(:,:), b(:,:)
    real(mp):: mat_dot_prod
    if (size(a,1) == size(b,1) .and. size(a,2) == size(b,2)) then
!$omp parallel
!$omp do

       mat_dot_prod = sum ( a(:,:)*b(:,:))
!$omp end do
!$omp end parallel

    else
       stop "mat_dot_prod with wrong size of matrix"
    end if
  end function mat_dot_prod
