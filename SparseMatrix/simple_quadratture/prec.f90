! D. Sheen (http://www.nasc.snu.ac.kr) as of Feb 1, 2015 
module prec
  integer, parameter::hpi = 1   !-128 to 127
  integer, parameter::spi = 2   !-32,768 to 32,767
  integer, parameter::dpi = 4   !-2,147,483,648 to 2,147,483,647 (default)
  integer, parameter::qpi = 8   !-9,223,372,036,854,775,808 to 9,223,372,036,854,775,807
!  integer, parameter::mpi = 8   !-9,223,372,036,854,775,808 to 9,223,372,036,854,775,807
  integer, parameter::mpi = 4   !-2,147,483,648 to 2,147,483,647 (default)

  integer, parameter::sp=selected_real_kind(p=6,r=37) !standard   = 4
  integer, parameter::dp=selected_real_kind(p=15,r=307) !standard = 8
  integer, parameter::qp=selected_real_kind(p=33,r=4931) !standard=16
! integer, parameter::mp=selected_real_kind(p=33,r=4931) !standard=16
  integer, parameter::mp=selected_real_kind(p=15,r=307) !standard = 8

  integer, parameter::spc = kind((1.0, 1.0))
  integer, parameter::dpc = kind((1._dp, 1._dp))
  integer, parameter::qpc = kind((1._qp, 1._qp))
  integer, parameter::mpc = kind((1._mp, 1._mp))

  integer, parameter::lgt = kind(.true.)

end module prec

