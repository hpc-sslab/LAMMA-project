Notice for pdgemm


1/ Installation
ScaLAPACK can be installed with "make". The configuration must be set in the SLmake.inc file. A SLmake.inc.example for a Linux machine running GNU compilers is given in the main directory. 
Download: http://www.netlib.org/scalapack/#_scalapack_version_2_1_0

a/ Firstly, you need to copy SLmake.in.example to SLmake.inc. In SLmake.inc file, you need to modify BLAS, LAPACK (and possibly other) libraries are available on your machine.
 
b/ Add userdgemm routine in libscalapack.a
+ Define userdgemm in PBLAS/SRC/PBblas.h file
+ Make object file: userdgemm.o
+ Add object file to library: ar -crv libscalapack.a userdgemm.o
+ Modify dgemm routine in PBLAS/SRC/PTOOL/PB_Cdtypeset.c to userdgemm

c/ Add packdata routine in libscalapack.a
+ Define packdata in PBLAS/SRC/PBblas.h file
+ Make object file: packdata.o
+ Add object file to library: ar -crv libscalapack.a packdata.o
+ Modify dmmadd_ routine in PBLAS/SRC/PTOOL/PB_Cdtypeset.c to packdata

d/ Add newPB_COutV routine in libscalapack.a on KNL
+ Define newPB_COutV in PBLAS/SRC/PBtools.h file
+ Make object file: newPB_COutV.o
+ Add object file to library: ar -crv libscalapack.a newPB_COutV.o
+ Modify PB_COutV routine in PBLAS/SRC/PTOOL/PB_CpgemmAB.c to newPB_COutV

e/ Add PB_Chbmalloc routine in libscalapack.a on KNL
+ Define PB_Chbmalloc in PBLAS/SRC/PBtools.h file
+ Make object file: PB_Chbmalloc.o
+ Add object file to library: ar -crv libscalapack.a PB_Chbmalloc.o
+ Modify PB_Cmalloc routine in PBLAS/SRC/PTOOL/newPB_COutV.c to PB_Chbmalloc

c/ Modify value of pilaenv.f
	On KNL and SKL, the value in pilaenv.f file need to set as 336 and 384, respectively.
