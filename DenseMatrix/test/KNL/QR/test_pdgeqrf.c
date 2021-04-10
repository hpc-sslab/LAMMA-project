#include <stdio.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>
#include "mpi.h"
#include<mkl_vsl.h>
#include"hbwmalloc.h"
#include"mkl_scalapack.h"

#ifndef MM
#define MM 48000
#endif


#ifndef KB
#define KB 336 
#endif 
void row_major_matrix(double      *mm,
                      const int    nrow,
                      const int    ncol,
                      const int    seed,
                      const double lb,
                      const double rb)
{
  int i;
  VSLStreamStatePtr stm;
  //
  vslNewStream(&stm, VSL_BRNG_MT19937, seed);
  #pragma ivdep
  for(i = 0; i < nrow; ++i)
  {
    vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD_ACCURATE,stm,ncol,&mm[i*ncol],lb,rb);
  }
  vslDeleteStream(&stm);
}

int main(int argc, char **argv) {
   int i, j, k;
/************  MPI ***************************/
   int myrank_mpi, nprocs_mpi;
//   MPI_Init( &argc, &argv);
//   MPI_Comm_rank(MPI_COMM_WORLD, &myrank_mpi);
//   MPI_Comm_size(MPI_COMM_WORLD, &nprocs_mpi);
/************  BLACS ***************************/
   int ictxt, nprow, npcol, myrow, mycol;
   int info,itemp;
   int ZERO=0,ONE=1;
   double norm;
   nprow = 2; npcol = 2; 
   int M=MM;
   int K=MM;
   int nb = KB;
   int kkb =KB;
   const int i_zero=0, i_one=1, i_four=4, i_negone=-1;
   blacs_pinfo_( &myrank_mpi, &nprocs_mpi ) ;
   blacs_get_( &i_negone, &i_zero, &ictxt );
   blacs_gridinit_( &ictxt, "R", &nprow, &npcol );
   blacs_gridinfo_( &ictxt, &nprow, &npcol, &myrow, &mycol );

   int rA = numroc_( &M, &nb, &myrow, &ZERO, &nprow );
   int cA = numroc_( &K, &nb, &mycol, &ZERO, &npcol );

//   double *A = (double*) malloc(rA*cA*sizeof(double));

   double sizeQuery;
   int lwork;
   int minus1 = -1;
   int descA[9];
   descinit_(descA, &M,   &K,   &nb,  &nb,  &ZERO, &ZERO, &ictxt, &rA,  &info);
//   double *A = (double*) malloc(rA*cA*sizeof(double));
   double *A = (double *) hbw_malloc(sizeof(double)*rA*cA);
   hbw_posix_memalign((void *) A, 64, sizeof(double)*rA*cA);
   double *tau = (double*)hbw_malloc(cA*sizeof(double));
   hbw_posix_memalign((void *) tau, 64, sizeof(double)*cA);
   
//   double *tau = (double*)malloc(cA*sizeof(double));
// find correct size of lwork   
   pdgeqrf_(&M, &K, A, &ONE, &ONE, descA, tau, &sizeQuery, &minus1, &info);
   lwork = sizeQuery;
   double *work = (double*)hbw_malloc(lwork*sizeof(double));
   hbw_posix_memalign((void *) work, 64, sizeof(double)*lwork);

//   double *work = (double*)malloc(lwork*sizeof(double));

     
     double alpha = 1.0; double beta = 1.0;	
     double start, end, flops;

	 srand(time(NULL));
	 int randa = 999 + npcol*myrow + mycol;
  	 row_major_matrix(&A[0], rA, cA, randa,  -1.0,  1.0);
	
	
for(i=0;i<3;i++){
     Cblacs_barrier(ictxt, "All");
	 start=MPI_Wtime();

	 pdgeqrf_(&M, &K, A, &ONE, &ONE, descA, tau, work, &lwork, &info);
 
	 Cblacs_barrier(ictxt, "All");
     end=MPI_Wtime();

	double duration = (double)(end - start); 
	 if (myrow==0 && mycol==0)
	 {
	 	printf("---------------M=%d\t nb=%d\t nprow=%d\tnpcol=%d\n",M,nb,nprow,npcol);
	    printf("%f Gigaflops\n", ((double)4 *(double)K *(double)K * (double)K  / (double)3) * 1.0e-9 / duration);
		
	 }
	}
    hbw_free(A);
    hbw_free(tau);
    hbw_free(work);
   Cblacs_gridexit( 0 );
   MPI_Finalize();
   return 0;
}

