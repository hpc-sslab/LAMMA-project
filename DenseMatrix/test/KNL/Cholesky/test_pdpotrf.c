#include <stdio.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>
#include "mpi.h"
#include<mkl_vsl.h>
#include<hbwmalloc.h>
#include"mkl_scalapack.h"

#ifndef MM
#define MM 48000
#endif


#ifndef KB
#define KB 336 
#endif 


int main(int argc, char **argv) {
   int i, j;
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
   blacs_get_( &i_zero, &i_zero, &ictxt );
   blacs_gridinit_( &ictxt, "R", &nprow, &npcol );
   blacs_gridinfo_( &ictxt, &nprow, &npcol, &myrow, &mycol );

   int rA = numroc_( &M, &nb, &myrow, &ZERO, &nprow );
   int cA = numroc_( &K, &nb, &mycol, &ZERO, &npcol );

   char uplo='L';
   int descA[9];
   descinit_(descA, &M,   &K,   &nb,  &nb,  &ZERO, &ZERO, &ictxt, &rA,  &info);

   double *A = (double *) hbw_malloc(sizeof(double)*rA*cA);
   hbw_posix_memalign((void *) A, 64, sizeof(double)*rA*cA);
   
	     
   double alpha = 1.0; double beta = 1.0;	
   double start, end, flops;

	
	 int k = 0;
 for ( j = 0; j < cA; j++) { // local col
     int l_j = j / nb; // which block
     int x_j = j % nb; // where within that block
     int J   = (l_j * npcol + mycol) * nb + x_j; // global col
     for ( i = 0; i < rA; i++) { // local row
         int l_i = i / nb; // which block
         int x_i = i % nb; // where within that block
         int I   = (l_i * nprow + myrow) * nb + x_i; // global row

         if(I == J) {
             A[k] = M*M;
         } else {
             A[k] = I+J;
         }

         k++;
     }
 }	
	

   for(i=0;i<3;i++){
       
	   Cblacs_barrier(ictxt, "All");
	   start=MPI_Wtime();

	   pdpotrf_(&uplo, &M, A, &ONE, &ONE, descA, &info);
 
	   Cblacs_barrier(ictxt, "All");
       end=MPI_Wtime();
	   
	   double duration = (double)(end - start); 
	   if (myrow==0 && mycol==0)
	   {
	 	  printf("---------------M=%d\t nb=%d\n",M,nb);
	      printf("%f Gigaflops\n", ((double)1 *(double)K *(double)K * (double)K  / (double)3) * 1.0e-9 / duration);
		
	   }
   }
   hbw_free(A);
   Cblacs_gridexit( 0 );
   MPI_Finalize();
   return 0;
}

