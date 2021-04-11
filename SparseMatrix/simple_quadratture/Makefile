# To use: make 
.SUFFIXES: .o .vo .f .f90

COMP	= gfortran
OPTS    = 
#CFLAGS  = -g -c -fno-align-commons -fimplicit-none -fbounds-check -fbacktrace
CFLAGS  = -g -c -fdefault-real-8 -fno-align-commons -fimplicit-none -fbounds-check -fbacktrace 
MFLAGS  = -vo
LINK	= $(COMP)
MODS	=
LIB	=

PROG    = a.out
LFLAGS  = -fopenmp -O3 -o $(PROG)

PLAT = LINUX

LIBS =
ALIBS =
LALIBS =

ECHO    = /bin/echo
LN      = ln
LNFLAGS = -s

MODS     = prec.mod common_var.mod maps.mod
DEP     = Makefile $(MODS)

OBJ   = common_var.o read_data.o print_summary.o main.o\
	maps.o phis.o ftns.o \
	tensor_prod.o \
	mesh_gen.o gauss_leg.o \
	gauss23pts.o gauss_stiffness.o gauss_fphi.o gen_rhs.o \
	albe.o mat_dot_prod.o cgs.o res.o \
	errs.o

.f90.o:
	@$(ECHO) Making $@ from $<
	@$(COMP) $(OPTS) $(CFLAGS) $(LIBS) $(LALIBS) $(ALIBS) $<

#commons: commons.f90 Makefile
#	$(COMP) -c $(OPTS) $(CFLAG) commons.f90

a.out:	$(OBJ) $(DEP)
	@$(LINK) $(LFLAGS) $(OBJ) 

common_var.o: common_var.f90
	gfortran -c $<

common_var.mod: common_var.f90 common_var.o
	@true

maps.o: maps.f90
	gfortran -c $<

maps.mod: maps.f90 maps.o
	@true

clean: 
	/bin/rm -f *.o *.mod

cleanall: 
	/bin/rm -f *.o *.mod ../a.out \#* *\~
