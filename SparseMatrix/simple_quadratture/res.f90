function res(al,be,f_al,f_be,tmpnx,tmpny,stif)
  ! Nov. 7, 2018, Written by Dongwoo Sheen (sheen@snu.ac.kr) http://www.nasc.snu.ac.kr
  ! 2D elliptic problem solver using the DSSY nonconforming FEM
  use prec
  use common_var
  use omp_lib
  implicit none
  integer, intent(in):: tmpnx, tmpny
  real(mp),intent(in):: stif(1:4,1:4,1:tmpnx,tmpny)
  real(mp):: al(0:tmpnx,1:tmpny), be(1:tmpnx,0:tmpny), f_al(0:tmpnx,1:tmpny)&
       , f_be(1:tmpnx,0:tmpny), res_al(0:tmpnx,1:tmpny), res_be(1:tmpnx,0:tmpny)
  real(mp):: res
  real(mp):: al_dx,al_lpux, be_dx,be_lpux
  integer:: j, k
  res = zero
!$omp parallel
!$omp do

  do j =1, tmpnx-1
     do k =1, tmpny
        res_al(j,k) = f_al(j,k) - (al_dx(al,be,j,k,tmpnx,tmpny,stif) + al_lpux(al,be,j,k,tmpnx,tmpny,stif))
        res = res + res_al(j,k)**2
     end do
  end do

  do j =1, tmpnx
     do k =1, tmpny-1
        res_be(j,k) = f_be(j,k) - ( be_dx(al,be,j,k,tmpnx,tmpny,stif) + be_lpux(al,be,j,k,tmpnx,tmpny,stif))
        res = res + res_be(j,k)**2
     end do
  end do
!$omp end do
!$omp end parallel

  res = sqrt(res)
end function res
