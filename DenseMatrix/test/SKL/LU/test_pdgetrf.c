#include <stdio.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>
#include "mpi.h"
#include<mkl_vsl.h>

#ifndef KB
#define KB 320
#endif
#ifndef MM
#define MM 1600
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
   nprow = 1; 
   npcol = 1; 
   int nb =KB;
   int M=MM;
   int K=MM; 

   Cblacs_pinfo( &myrank_mpi, &nprocs_mpi ) ;
   Cblacs_get( -1, 0, &ictxt );
   Cblacs_gridinit( &ictxt, "Row", nprow, npcol );
   Cblacs_gridinfo( ictxt, &nprow, &npcol, &myrow, &mycol );


   int rA = numroc_( &M, &nb, &myrow, &ZERO, &nprow );
   int cA = numroc_( &K, &nb, &mycol, &ZERO, &npcol );

   double *A = (double*) malloc(rA*cA*sizeof(double));

   int descA[9], descD[9];
   int *IPIV;
   IPIV = (int *)calloc(rA + nb, sizeof(int));
   descinit_(descA, &M,   &K,   &nb,  &nb,  &ZERO, &ZERO, &ictxt, &rA,  &info);
   
   double alpha = 1.0; double beta = 1.0;	
   double start, end, flops;

   srand(time(NULL));
   int randa = 999 + npcol*myrow + mycol;
   row_major_matrix(&A[0], rA, cA, randa,  -1.0,  1.0);
 

   for(j=0;j<4;j++)
   {
	MPI_Barrier(MPI_COMM_WORLD);
 	start=MPI_Wtime();
	 
	pdgetrf_(&M, &K, A, &ONE, &ONE, descA, IPIV, &info);

	MPI_Barrier(MPI_COMM_WORLD);
        end=MPI_Wtime();
	 
	double duration = (double)(end - start); 
	if (myrow==0 && mycol==0)
	{
	if(j==0){
	}
        printf("------M=%d\tK=%d\tnb=%d\tnprow=%d\tnpcol=%d--------\n",M,K,nb,nprow,npcol);
          if (M > K)
	  {
	     printf("duration=%f\t PDGETRF=%f Gigaflops\n",duration, ((double)K * (double)K * (double)M - (double)K * (double)K * (double)K / (double)3) * 1.0e-9 / duration);
	  }
	  else if (K < M)
	  {
	     printf("duration=%f\t PDGETRF=%f Gigaflops\n",duration, ((double)M * (double)M * (double)K - (double)M * (double)M * (double)M / (double)3) * 1.0e-9 / duration);
          }
	  else
	  {
	     printf("duration=%f\t PDGETRF=%f Gigaflops\n",duration, ((double)2*(double)K *(double)K * (double)K  / (double)3) * 1.0e-9 / duration);
		
          }
	}
   }
   free(A);

   Cblacs_gridexit( 0 );
   MPI_Finalize();
   return 0;
}

