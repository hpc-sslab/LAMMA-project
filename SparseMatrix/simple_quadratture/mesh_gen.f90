subroutine mesh_gen
  use prec
  use common_var

  implicit none
  real(mp):: hx_unif, hy_unif, rndm
  integer::  j, k

  xj(0,:) = x_beg; xj(nx,:) = x_end; yk(:,0) = y_beg; yk(:,ny) = y_end
  hx_unif = xsize/real(nx,mp);  hy_unif = ysize/real(ny,mp);
  if (abs(ratio) > zero) then
     do k = 0, ny
        do j = 1, nx-1
           call RANDOM_NUMBER(rndm) !0 < rndm < 1
 !          rndm = real((k+1)*(j+2), mp)/real(nx*ny*10,mp)
           xj(j,k) = x_beg + ( real(j,mp) + (two*rndm - one)*ratio)*hx_unif
        end do
        do j = 1, nx
           hx(j,k) = xj(j,k) - xj(j-1,k)
        end do
     end do

     do j = 0, nx
        do k = 1, ny-1
           call RANDOM_NUMBER(rndm) !0 < rndm < 1
!           rndm = real((k+1)*(j+2), mp)/real(nx*ny*10,mp)
           yk(j,k) = y_beg + ( real(k,mp) + (two*rndm - one)*ratio)*hy_unif
        end do
        do k = 1, ny
           hy(j,k) = yk(j,k) - yk(j,k-1)
        end do
     end do
  else
     do k = 0, ny
        do j = 1, nx-1
           xj(j,k) = x_beg + real(j,mp)*hx_unif
        end do
        do j = 1, nx
           hx(j,k) = xj(j,k) - xj(j-1,k)
        end do
     end do

     do j = 0, nx
        do k = 1, ny-1
           yk(j,k) = y_beg + real(k,mp)*hy_unif
        end do
        do k = 1, ny
           hy(j,k) = yk(j,k) - yk(j,k-1)
        end do
     end do

  end if

  if (.false.) then  ! trapezoidal mesh
     !$omp parallel
     !$omp do

     do k = 0, ny, 2
        do j = 1, nx-1, 2
           xj(j,k) = x_beg + ( real(j,mp) - ratio ) * hx_unif
        end do
        do j = 2, nx-1, 2
           xj(j,k) = x_beg + ( real(j,mp) ) * hx_unif
        end do
     end do

     do k = 1, ny, 2
        do j = 1, nx-1, 2
           xj(j,k) = x_beg + ( real(j,mp) + ratio ) * hx_unif
        end do
        do j = 2, nx-1, 2
           xj(j,k) = x_beg + ( real(j,mp) ) * hx_unif
        end do
     end do

     do k = 0, ny
        do j = 1, nx
           hx(j,k) = xj(j,k) - xj(j-1,k)
        end do
     end do

     do k = 0, ny
        yk(:,k) = y_beg + real(k,mp) * hy_unif;
     end do

     do k = 1, ny
        hy(:,k) = hy_unif
     end do
     !$omp end do
     !$omp end parallel

  end if

end subroutine mesh_gen
