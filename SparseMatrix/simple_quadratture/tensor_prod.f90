subroutine tensor_prod(j,k,J_F_K,gp_K,gw_K)
  use prec
  use common_var
  use maps
    use omp_lib

  implicit none
  integer:: l,m, ind
  integer, intent(in) :: j,k
  real(mp):: x(2)
  real(mp):: v1(1:2), v2(1:2), v3(1:2), v4(1:2)
  real(mp), intent(out):: J_F_K(1:mg_tensr*mg_tensr), gp_K(mg_tensr*mg_tensr,2), gw_K(1:mg_tensr*mg_tensr)

  call vv(j,k,v1,v2,v3,v4)
  
!$omp parallel
!$omp do

  do l = 1, mg_tensr  !x-coord
     x(1) = gp(l)
     do m = 1, mg_tensr  !y-coord
        x(2) = gp(m)
        ind = (l-1)*mg_tensr + m
        gp_K(ind,:) = F_K(x,v1,v2,v3,v4) !mapping from {hat K} to K
        J_F_K(ind) = abs(det(gF_K(x,v1,v2,v3,v4)))
        gw_K(ind) = gw(l)*gw(m)
     end do
  end do
!$omp end do
!$omp end parallel

end subroutine tensor_prod

subroutine tensor_prod_err(j,k,J_F_K_err,gp_err,gw_err) !use 3x3 points for error cal.
  use prec
  use common_var
  use maps
  use omp_lib
  implicit none
  integer:: l,m, ind
  integer, intent(in) :: j,k
  real(mp):: x(2)
  real(mp):: v1(1:2), v2(1:2), v3(1:2), v4(1:2)
  real(mp), dimension(mg_err*mg_err), intent(out):: J_F_K_err, gw_err
  real(mp), dimension(mg_err*mg_err,2), intent(out):: gp_err

  call vv(j,k,v1,v2,v3,v4)

!$omp parallel
!$omp do

  do l = 1, mg_err !x-coord
     x(1) = gp_ten(l)
     do m = 1, mg_err  !y-coord
        x(2) = gp_ten(m)
        ind = (l-1)*mg_err + m
        gp_err(ind,:) = F_K(x,v1,v2,v3,v4)
        J_F_K_err(ind) = abs(det(gF_K(x,v1,v2,v3,v4)))
        gw_err(ind) = gw_ten(l)*gw_ten(m)
     end do
  end do
!$omp end do
!$omp end parallel

end subroutine tensor_prod_err
