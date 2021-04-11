program dssy
  ! Nov. 7, 2018, Written by Dongwoo Sheen (sheen@snu.ac.kr) http://www.nasc.snu.ac.kr
  ! 2D elliptic problem solver  using the DSSY nonconforming FEM
  use prec
  use common_var
  use omp_lib
  implicit none
  integer:: j, k, l, inx, iny, pauseint, j_ens
  real(mp):: tmp, x, y, res, u, f, gauss_gphigphi
  real(mp), dimension(:,:), allocatable:: al, be, f_al, f_be
  character(len=100):: filename !For save
  integer:: pj, pk !For save
  real(mp):: l2err, energy_err
  real(mp):: l2error(0:max_step), l2error_ratio(1:max_step)
  real(mp):: energyerror(0:max_step), energyerror_ratio(1:max_step)
  real(mp):: l2error_ens(0:max_step)
  real(mp):: energyerror_ens(0:max_step)

  real(mp), dimension(:,:,:,:), allocatable:: loc_al, loc_be

  one = 1._mp; two=2._mp; hf = .5_mp; qt = .25_mp; four=4._mp;  three=3._mp
  pi = acos(-one)

  call read_data
  if (ratio == zero) n_ens = 1
  if (quad=="tensr") then
     allocate(gp(mg_tensr), gw(mg_tensr))
     call gauss_leg(-one,one,gp,gw,mg_tensr,mg_tensr) !Gauss-Legendre quadrature over [-1,1]
     !print*, "gp, gw = ",  gp, gw         !gp:point, gw:weight
     ! for error calc. use 3x3=(mg_err=3) tensor product of Gauss points always
  end if
  allocate(gp_ten(mg_err), gw_ten(mg_err)) !will be used only error calc.
  call gauss_leg(-one,one,gp_ten,gw_ten,mg_err,mg_err)
  inx = nx;  iny=ny

  l2error_ens = zero; energyerror_ens = zero
  !$omp parallel
  !$omp do
  do j_ens = 1, n_ens  !for ensemble average
     print*, "------------------------------"
     print*, "Begining ensemble average for ",j_ens
     nx=inx/2; ny=iny/2
     l2error(0)=one; energyerror(0)=one;
     do step = 1, max_step
        nx = nx*2; ny=ny*2
        print*, "-------------------------------"
        !hx = xsize/real(nx,mp);      hy = ysize/real(ny,mp);
        allocate( xj(0:nx,0:ny), yk(0:nx,0:ny), hx(1:nx,0:ny), hy(0:nx,1:ny) )
        call mesh_gen !Mesh generation

        allocate(a(1:4,1:4,nx,ny))
        allocate(al(0:nx,1:ny), be(1:nx,0:ny), f_al(0:nx,1:ny), f_be(1:nx,0:ny))
        ! See the comment in gen_rhs.f90 for alpha and beta unknowns
        al = zero; be = zero;

        if (quad=="mengs") then
           call gauss_stiffness_mengs
           call gen_rhs_mengs(f_al,f_be)
        else if (quad=="tensr") then ! tensor product
           call gauss_stiffness_tensr
           call gen_rhs_tensr(f_al,f_be)
        else
           stop "Quadrature Rule is undefined"
        end if

        !       write(6,90)
        call cpu_time(ctime(1))
        if (method=="cgnc") then
           call cg_sol(al,be,f_al,f_be,nx,ny,a)
        else 
           !The Gauss-Jacobi iteration
           do it = 1, max_iter
              if (method=="jaco") then
                 iter = it
                 call update_jacobi(al,be,f_al,f_be,nx,ny,a) 
              else if (method=="seid") then
                 iter = it
                 call update_seidel(al,be,f_al,f_be,nx,ny,a)
              else if (method=="ssei") then
                 iter = 2*it
                 call update_symm_seidel(al,be,f_al,f_be,nx,ny,a) 
              end if
              if (mod(iter,10)==0) then
                 tmp = res(al,be,f_al,f_be,nx,ny,a)
                 !                write(6,91) iter, tmp 
                 if (tmp < tol) then
                    !                   write(6,90)
                    !                   print*, "Converged!"
                    !                   write(6,93) iter, tmp, l2err(al,be)
                    exit
                 end if
              end if
           end do
        end if
        call cpu_time(ctime(2))
        write(6,97) nx,ny
        write(6,98) ctime(2) - ctime(1), l2err(al,be), energy_err(al,be)
        l2error(step) = l2err(al,be); energyerror(step) = energy_err(al,be)
        if (step /= max_step) then
           deallocate( xj, yk, hx, hy, a, al, be, f_al, f_be )
        end if
     end do !end do step
     l2error_ens = l2error_ens + l2error**2
     energyerror_ens = energyerror_ens + energyerror**2
     deallocate( xj, yk, hx, hy )
     deallocate(a)
     deallocate(al, be, f_al, f_be)

  end do !end ensemble average summation
  !$omp end do
  !$omp end parallel

  l2error=sqrt(l2error_ens/real(n_ens,mp))
  energyerror=sqrt(energyerror_ens/real(n_ens,mp))

  call print_summary
  print*, "-----------------------------"
  print*, "L^2-errors and error ratios are as follows"
  do step = 1, max_step
     l2error_ratio(step) = log(l2error(step-1)/l2error(step))/log(two)
  end do
  write(6,96) ((nx/2**max_step)*2**j, j =1,max_step)
  write(6,94) l2error(1:max_step)
  write(6,95) l2error_ratio(2:max_step)
  print*, "-----------------------------"
  print*, "Energy-errors and error ratios are as follows"
  do step = 1, max_step
     energyerror_ratio(step) = log(energyerror(step-1)/energyerror(step))/log(two)
  end do
  write(6,96) ((nx/2**max_step)*2**j, j =1,max_step)
  write(6,94) energyerror(1:max_step)
  write(6,95) energyerror_ratio(2:max_step)
  print*, "-----------------------------"
  write(6,990)
  j=1
  write(6,992) (nx/2**max_step)*2**j, energyerror(j), l2error(j) 
  do j =2, max_step
     write(6,991) (nx/2**max_step)*2**j, energyerror(j), energyerror_ratio(j), l2error(j), l2error_ratio(j)
  end do
991 format(i7,3x, 4g11.3)
992 format(i7,3x, g11.3,11x,g11.3)
990 format(5x,"nx",2x,"$|u_h-u|_{1,h}$",2x, "ratio",2x "$||u_h-u||_{0,\O}$",2x "ratio")
  ! if (.true.) then
  if (.false.) then
     !Save the multiscale system and solution.
     OPEN(UNIT=10, FILE="micro_al.txt", ACTION="write", STATUS="replace")
     DO j=0,nx
        WRITE(10,1000) (al(j,k), k=1,ny)
     END DO

     OPEN(UNIT=11, FILE="micro_be.txt", ACTION="write", STATUS="replace")
     DO j=1,nx
        WRITE(11,1000) (be(j,k), k=0,ny)
     END DO

     !Save the MESH
     OPEN(UNIT=12, FILE="mesh_x.txt", ACTION="write", STATUS="replace")
     DO j=0,nx
        WRITE(12,1000) (xj(j,k), k=0,ny)
     END DO

     OPEN(UNIT=13, FILE="mesh_y.txt", ACTION="write", STATUS="replace")
     DO j=0,nx
        WRITE(13,1000) (yk(j,k), k=0,ny)
     END DO
  end if
97 format("nx, ny = ",i4,"  x",i4)
98 format("CPU time =",g12.3, 10x, "l2-err = ",g12.3, ",   energy-err = ", g12.3)
94 format(8g11.3)
95 format(11x,7g11.3)
96 format("nx=",i4.3,7i11.3)

90 format(2x,"iter",8x,"residual",8x,"L^2-err")
91 format(i7,5x,g11.3)
93 format(i7,5x,g11.4,5x,g11.3)

  !1000 FORMAT(1000E18.3)
1000 FORMAT(40000E30.18)

end program dssy
  
