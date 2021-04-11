include "prec.f90"
  
  module common_var
    use prec
    SAVE
    integer, parameter::max_step=7
    integer:: step, n_ens
    integer:: nx, ny, iter, max_iter, it !iter = 2*it for SSOR; = it otherwise
    integer :: mg, lg  !lg = length of gauss quad points
    integer:: mg_tensr, mg_quad, mg_err=3
    real(mp):: c_tilde=1.0
    real(mp), allocatable:: gp(:), gw(:), a(:,:,:,:), gp_ten(:), gw_ten(:)
    !use mg_err points for gp_err, gw_err for error calc.
    real(mp):: x_beg, y_beg, x_end, y_end, xsize, ysize, tol, omega
    real(mp), allocatable:: xj(:,:), yk(:,:), hx(:,:), hy(:,:) !Mesh information
    real(mp), allocatable:: ref_xj(:,:), ref_yk(:,:) !Mesh information
    real(mp):: qt, hf, zero, one, two, three, four, pi
    real(mp):: ratio, epsil
    logical :: print_res
    real(mp):: ctime(6)
    character(4):: homo, formula
    character(4):: method !cgnc, jaco, seid, ssei
    character(4) :: fem! dssy, bili, rann, meng
    character(5) :: quad! "mengs"  "tensr"
    character(3) :: quad_type! mcl, cho, kns
  end module common_var
