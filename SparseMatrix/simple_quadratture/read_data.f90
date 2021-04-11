subroutine read_data
  use prec
  use common_var
  implicit none

  open(11,FILE='data',status='old')
  read(11,*) x_beg, x_end, y_beg, y_end
  read(11,*) homo   !homogeneous BC or inhogeneous BC
  read(11,*) epsil !"cons"tant coeff or "hete"rogeneous kappa
  read(11,*) nx, ny
  read(11,*) ratio
  read(11,*) lg
  read(11,*) fem
  read(11,*) formula
  read(11,*) method
  read(11,*) max_iter
  read(11,*) tol
  read(11,*) print_res
  read(11,*) n_ens

  xsize = x_end-x_beg;      ysize = y_end-y_beg

  if (lg == 2 .or. lg == 3 ) then
     mg_quad = lg;  quad = "mengs"
  else if (lg == 1 .or. lg == 4 .or. lg == 9 .or. lg == 16 ) then
     mg_tensr = int(sqrt(real(lg)));   quad = "tensr"
  endif

  print*," x_beg, x_end, y_beg, y_end = ",  x_beg, x_end, y_beg, y_end
  print*," homo (homogeneous BC or inhogeneous BC)=", homo
  print*, "epsil in exact solution = ", epsil
  print*, "nx, ny = ", nx, ny
  print*, "ratio for random perturbation mesh = ", ratio
  print*, "max_iter = ", max_iter
  print*, "method = ", method
  print*, "tol = ", tol
  print*, "quad = ", quad
  print*, "formula = ", formula
  print*, "lg =", lg
  if (lg == 2 .or. lg == 3 ) then
     print*, "mg_quad = ", mg_quad 
  else if (lg == 1 .or. lg == 4 .or. lg == 9 .or. lg == 16 ) then
     print*, "mg_tensr = ", mg_tensr,"x",mg_tensr
  endif
  print*, "mg_err =", mg_err
  print*, "fem =", fem
  print*, "print_res =", print_res, ";  n_ens(emble average) =", n_ens
end subroutine read_data
