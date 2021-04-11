subroutine print_summary
  use prec
  use common_var
  implicit none
  print*," homo (homogeneous BC or inhogeneous BC)=", homo
  print*, "epsil in exact solution = ", epsil
  print*, "fem =", fem
  print*, "ratio for random perturbation mesh = ", ratio
  print*, "method = ", method
  print*, "tol = ", tol,  ";   max_iter = ", max_iter, ";  mg_err =", mg_err
  print*, "quad = ", quad, ";   formula = ", formula
  if (lg == 2 .or. lg == 3 ) then
     print*, "mg_quad = ", mg_quad 
  else if (lg == 1 .or. lg == 4 .or. lg == 9 .or. lg == 16 ) then
     print*, "mg_tensr = ", mg_tensr,"x",mg_tensr
  endif
  print*, "print_res =", print_res, ";  n_ens(emble average) =", n_ens
end subroutine print_summary
