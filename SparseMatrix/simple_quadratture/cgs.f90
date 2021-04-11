  subroutine update_jacobi(al,be,f_al,f_be,tmpnx,tmpny,stif)
    ! Nov. 7, 2018, Written by Dongwoo Sheen (sheen@snu.ac.kr) http://www.nasc.snu.ac.kr
    ! 2D elliptic problem solver  using the DSSY nonconforming FEM
    use prec
    use common_var
    implicit none
    integer, intent(in):: tmpnx, tmpny
    real(mp), intent(inout):: al(0:tmpnx,1:tmpny), be(1:tmpnx,0:tmpny)
    real(mp), intent(in):: f_al(0:tmpnx,1:tmpny), f_be(1:tmpnx,0:tmpny)
    real(mp), intent(in):: stif(1:4,1:4,1:tmpnx,1:tmpny)
    real(mp):: n_al(0:tmpnx,1:tmpny), n_be(1:tmpnx,0:tmpny)  ! for the next unknowns
    real(mp):: al_dx,al_lpux,be_dx,be_lpux
    integer:: j, k
    n_al = zero; n_be = zero; 
    do j =1, tmpnx-1
       do k =1, tmpny
          n_al(j,k)=omega*(f_al(j,k) - al_lpux(al,be,j,k,tmpnx,tmpny,stif))/(stif(1,1,j,k)+stif(3,3,j+1,k)) + (one-omega)*al(j,k)
       end do
    end do
    
    do j =1, tmpnx
       do k =1, tmpny-1
          n_be(j,k)=omega*(f_be(j,k) - be_lpux(al,be,j,k,tmpnx,tmpny,stif))/( stif(2,2,j,k)+stif(4,4,j,k+1) ) + (one-omega)*be(j,k)
       end do
    end do

    al = n_al; be = n_be; 
  end subroutine update_jacobi

  subroutine update_seidel(al,be,f_al,f_be,tmpnx,tmpny,stif)
  ! Nov. 7, 2018, Written by Dongwoo Sheen (sheen@snu.ac.kr) http://www.nasc.snu.ac.kr
  ! 2D elliptic problem solver  using the DSSY nonconforming FEM
    use prec
    use common_var
    implicit none
    integer, intent(in):: tmpnx,tmpny
    real(mp), intent(inout):: al(0:tmpnx,1:tmpny), be(1:tmpnx,0:tmpny)
    real(mp), intent(in):: f_al(0:tmpnx,1:tmpny), f_be(1:tmpnx,0:tmpny)
    real(mp), intent(in):: stif(1:4,1:4,1:tmpnx,1:tmpny)
    real(mp):: al_dx,al_lpux,be_dx,be_lpux
    integer:: j, k
    do j =1, tmpnx-1
       do k =1, tmpny
          al(j,k)=omega*(f_al(j,k) - al_lpux(al,be,j,k,tmpnx,tmpny,stif))/(stif(1,1,j,k)+stif(3,3,j+1,k)) + (one-omega)*al(j,k)
          !            print *, a(1,1,j,k), a(3,3,j+1,k)
          
       end do
    end do
    
    do j =1, tmpnx
       do k =1, tmpny-1
          be(j,k)=omega*(f_be(j,k) - be_lpux(al,be,j,k,tmpnx,tmpny,stif))/(stif(2,2,j,k)+stif(4,4,j,k+1)) + (one-omega)*be(j,k)
       end do
    end do
  end subroutine update_seidel
  
  subroutine update_symm_seidel(al,be,f_al,f_be,tmpnx,tmpny,stif)
    ! Nov. 7, 2018, Written by Dongwoo Sheen (sheen@snu.ac.kr) http://www.nasc.snu.ac.kr
    ! 2D elliptic problem solver  using the DSSY nonconforming FEM
    use prec
    use common_var
    implicit none
    integer, intent(in):: tmpnx, tmpny
    real(mp), intent(inout):: al(0:tmpnx,1:tmpny), be(1:tmpnx,0:tmpny) 
    real(mp), intent(in):: f_al(0:tmpnx,1:tmpny), f_be(1:tmpnx,0:tmpny)
    real(mp), intent(in):: stif(1:4,1:4,1:tmpnx,1:tmpny)
    real(mp):: al_dx,al_lpux,be_dx,be_lpux
    integer:: j, k
    do j = 1, tmpnx-1
       do k = 1, tmpny
          al(j,k)=omega*(f_al(j,k) - al_lpux(al,be,j,k,tmpnx,tmpny,stif))/(a(1,1,j,k)+a(3,3,j+1,k))  + (one-omega)*al(j,k)
       end do
    end do
    
    do j = 1, tmpnx
       do k = 1, tmpny-1
          be(j,k)=omega*(f_be(j,k) - be_lpux(al,be,j,k,tmpnx,tmpny,stif))/(stif(2,2,j,k)+stif(4,4,j,k+1)) + (one-omega)*be(j,k)
       end do
    end do
    
    ! symmetrizing step
    do j = tmpnx-1, 1, -1
       do k = tmpny, 1, -1
          al(j,k)=omega*(f_al(j,k) - al_lpux(al,be,j,k,tmpnx,tmpny,stif))/(stif(1,1,j,k)+stif(3,3,j+1,k))  + (one-omega)*al(j,k)
       end do
    end do

    do j = tmpnx, 1, -1
       do k = tmpny-1, 1, -1
          be(j,k)=omega*(f_be(j,k) - be_lpux(al,be,j,k,tmpnx,tmpny,stif))/(stif(2,2,j,k)+stif(4,4,j,k+1)) + (one-omega)*be(j,k)
       end do
    end do
  end subroutine update_symm_seidel

  subroutine cg_sol(al,be,f_al,f_be,tmpnx,tmpny,stif)
    ! Nov. 7, 2018, Written by Dongwoo Sheen (sheen@snu.ac.kr) http://www.nasc.snu.ac.kr
    ! 2D elliptic problem solver  using the DSSY nonconforming FEM

    use prec
    use common_var
    use omp_lib

    implicit none
    integer, intent(in):: tmpnx, tmpny !size of al,be
    real(mp):: al(0:tmpnx,1:tmpny), be(1:tmpnx,0:tmpny)
    real(mp):: f_al(0:tmpnx,1:tmpny), f_be(1:tmpnx,0:tmpny), r_al(0:tmpnx,1:tmpny), r_be(1:tmpnx,0:tmpny)  &
         ,     p_al(0:tmpnx,1:tmpny), p_be(1:tmpnx,0:tmpny), ap_al(0:tmpnx,1:tmpny), ap_be(1:tmpnx,0:tmpny)
    real(mp), intent(in):: stif(1:4,1:4,1:tmpnx,1:tmpny) !Local stiffness matrix
    real(mp):: tmp, res, l2err, alpha, beta, rr, rr_prev, app, energy_err
    integer:: j, k

    interface    
       subroutine matrix_mul(al,be,ax_al,ax_be,tmpnx,tmpny,stif)
         use prec
         use common_var
         implicit none
         integer, intent(in):: tmpnx, tmpny
         real(mp),intent(in):: stif(1:4,1:4,1:tmpnx,1:tmpny)
         real(mp):: al(0:tmpnx,1:tmpny), be(1:tmpnx,0:tmpny), ax_al(0:tmpnx,1:tmpny), ax_be(1:tmpnx,0:tmpny)
       end subroutine matrix_mul

       function mat_dot_prod(a,b)
         use prec
         real(mp):: a(:,:), b(:,:)
         real(mp):: mat_dot_prod
       end function mat_dot_prod

    end interface

    call matrix_mul(al, be, ap_al, ap_be, tmpnx, tmpny, stif)    
    r_al = f_al - ap_al;     r_be = f_be - ap_be;     
    p_al = r_al;             p_be = r_be;            
    rr_prev = mat_dot_prod(r_al,r_al) + mat_dot_prod(r_be,r_be)
!$omp parallel
!$omp do
    do iter = 1, max_iter
       if (.true.) then
          if (mod(iter,nx)==0) then
             res = sqrt(mat_dot_prod(r_al,r_al)+mat_dot_prod(r_be,r_be))
             !          write(6,91) iter, res
             if( res < tol) then 
                !             write(6,*) 'res = ', res
                !             print*, "Converged at ", iter, " iteration."
                return
             end if
          end if
       end if
       call matrix_mul(p_al, p_be, ap_al, ap_be, tmpnx, tmpny, stif)    
       app = mat_dot_prod(ap_al,p_al) + mat_dot_prod(ap_be,p_be)
       alpha = (mat_dot_prod(r_al,p_al) + mat_dot_prod(r_be,p_be)) /app
       al = al + alpha*p_al;         be = be + alpha*p_be;         
       r_al = r_al - alpha*ap_al ;   r_be = r_be - alpha*ap_be ;   
       rr = mat_dot_prod(r_al,r_al) + mat_dot_prod(r_be,r_be)
       beta = rr/rr_prev
       p_al = r_al + beta*p_al;     p_be = r_be + beta*p_be;   
       rr_prev = rr
    enddo
!$omp end do
!$omp end parallel

    print*, "tol = ", tol,";   res = ", res,";   iter = ", iter,";   max_iter = ",    max_iter
    if (iter >= max_iter) stop "CG did not converge! increase max_iter, or decrease tolerance"

91  format(i7,5x,g11.3)
  end subroutine cg_sol
  
  subroutine matrix_mul(al, be, ax_al, ax_be, tmpnx, tmpny, stif)
  ! Nov. 7, 2018, Written by Dongwoo Sheen (sheen@snu.ac.kr) http://www.nasc.snu.ac.kr
  ! 2D elliptic problem solver  using the DSSY nonconforming FEM
    use prec
    use common_var
  use omp_lib
    implicit none
    integer:: j,k
    integer, intent(in):: tmpnx, tmpny
    real(mp), intent(in):: stif(1:4,1:4,1:tmpnx,1:tmpny)
    real(mp):: al(0:tmpnx,1:tmpny), be(1:tmpnx,0:tmpny), ax_al(0:tmpnx,1:tmpny), ax_be(1:tmpnx,0:tmpny) 
    real(mp):: al_dx,al_lpux, be_dx,be_lpux
    ax_al = zero;  ax_be = zero
!$omp parallel
!$omp do
    
    do j =1, tmpnx-1
       do k =1, tmpny
          ax_al(j,k) = al_dx(al,be,j,k,tmpnx,tmpny,stif) + al_lpux(al,be,j,k,tmpnx,tmpny,stif)
       end do
    end do
    do j =1, tmpnx
       do k =1, tmpny-1
          ax_be(j,k) = be_dx(al,be,j,k,tmpnx,tmpny,stif) + be_lpux(al,be,j,k,tmpnx,tmpny,stif)
       end do
    end do
!$omp end do
!$omp end parallel

  end subroutine matrix_mul
