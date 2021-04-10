#include <stdio.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>
#include "mpi.h"
#include<mkl_vsl.h>
#include<hbwmalloc.h>

#ifndef MM
#define MM 2000 
#endif

#ifndef NPR
#define NPR 2 
#endif

#ifndef NPC
#define NPC 2 
#endif

#ifndef NB
#define NB 336 
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
	int myrank_mpi, nprocs_mpi;
	int ictxt, nprow, npcol, myrow, mycol,nb;
	int info,itemp;
	int ZERO=0,ONE=1, TWO=2;
	nprow = NPR; npcol = NPC; nb =KB;
	int M=MM;
	int N=MM;
	int K=MM;

	const int i_zero=0, i_one=1, i_four=4, i_negone=-1;
	blacs_pinfo_( &myrank_mpi, &nprocs_mpi ) ;
	blacs_get_( &i_negone, &i_zero, &ictxt );
	blacs_gridinit_( &ictxt, "R", &nprow, &npcol );
	blacs_gridinfo_( &ictxt, &nprow, &npcol, &myrow, &mycol );

	int rA = numroc_( &M, &nb, &myrow, &ZERO, &nprow );
	int cA = numroc_( &K, &nb, &mycol, &ZERO, &npcol );
	int rB = numroc_( &K, &nb, &myrow, &ZERO, &nprow );
	int cB = numroc_( &N, &nb, &mycol, &ZERO, &npcol );
	int rC = numroc_( &M, &nb, &myrow, &ZERO, &nprow );
	int cC = numroc_( &N, &nb, &mycol, &ZERO, &npcol );

	double *A = (double *) hbw_malloc(sizeof(double)*rA*cA);
	hbw_posix_memalign((void *) A, 64, sizeof(double)*rA*cA);
	double *B = (double *) hbw_malloc(sizeof(double)*rB*cB);
	hbw_posix_memalign((void *) B, 64, sizeof(double)*rB*cB);
	double *C = (double *) hbw_malloc(sizeof(double)*rC*cC);
	hbw_posix_memalign((void *) C, 64, sizeof(double)*rC*cC);
	
	int descA[9],descB[9],descC[9];

	descinit_(descA, &M,   &K,   &nb,  &nb,  &ZERO, &ZERO, &ictxt, &rA,  &info);
	descinit_(descB, &K,   &N,   &nb,  &nb,  &ZERO, &ZERO, &ictxt, &rB,  &info);
	descinit_(descC, &M,   &N,   &nb,  &nb,  &ZERO, &ZERO, &ictxt, &rC,  &info);

	double alpha = 1.0; double beta = 1.0;	
	double start, end, flops,f, norm, start2, end2;
	srand(time(NULL));

	int randa = 777 + npcol*myrow + mycol;	
	int randb = 888 + npcol*myrow + mycol;	
	int randc = 999 + npcol*myrow + mycol;	
  	
	row_major_matrix(&A[0], rA, cA, randa,  -1.0,  1.0);
  	row_major_matrix(&B[0], rB, cB, randb,  -1.0,  1.0);
  	row_major_matrix(&C[0], rC, cC, randc,  -1.0,  1.0);
	row_major_matrix(&D[0], rC, cC, randc,  -1.0,  1.0);

	for (j=0; j<4; j++)
	{     

		Cblacs_barrier(ictxt, "All");
		start2=MPI_Wtime();

		pdgemm_("N", "N", &M , &N , &K , &alpha, A , &ONE, &ONE , descA , B , &ONE, &ONE , descB , &beta , C , &ONE, &ONE , descD );
		
		Cblacs_barrier(ictxt, "All");
		end2=MPI_Wtime();//*/
		if (myrow==0 && mycol==0)
		{
	      
		 flops = 2 * (double) M * (double) N * (double) K / (end2-start2) / 1e9;
		 printf("----M=%d\t nb=%d\tnnb2=%d\tkkb2=%d\tduration=%f\t flops=%f Gflops\t nprow=%d\tnpcol=%d\n", M,nb,nnb2,kkb2,end2-start2,flops,nprow,npcol);
		
		}
	}
	hbw_free(A);
	hbw_free(B);
	hbw_free(C);

	Cblacs_gridexit( 0 );
	MPI_Finalize();
	return 0;
}
