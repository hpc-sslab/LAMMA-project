#include<time.h>
#include<stdio.h>
#include<malloc.h>
#include<hbwmalloc.h>

#include<mkl_vsl.h>
#include<mkl_cblas.h>
#include<omp.h>

// col-major store
void col_major_matrix(double      *mm,
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
  for(i = 0; i < ncol; ++i)
  {
    vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD_ACCURATE,stm,nrow,&mm[i*nrow],lb,rb);
  }
  vslDeleteStream(&stm);
}

// column-major store
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

// General Matrix-Matrix Multiplication (GEMM)
// Test routine

int main(int    argc,
         char **argv)
{
#define NUM_ITER 5
  // test type
  const int iter = 10;
  int       i,j,q;
	int m2, n2, k2;
  // iteration, time, validity
  double startTime;
  double endTime;
  double gap;
  double flops, maxFlops = 0;
	double flopsArray[NUM_ITER];
  double norm;
  // matrices
  double *A;
  double *B;
  double *C;

  // matrix dimensions
  const int m = MSIZE;
  const int n = NSIZE;
  const int k = KSIZE;

/*  A = (double *) malloc(sizeof(double)*m*k);
  B = (double *) malloc(sizeof(double)*k*n);
  C = (double *) malloc(sizeof(double)*m*n);//  */
  // MCDRAM version
  A = (double *) hbw_malloc(sizeof(double)*m*k);
  hbw_posix_memalign((void *) A, 64, sizeof(double)*m*k);
  B = (double *) hbw_malloc(sizeof(double)*k*n);
  hbw_posix_memalign((void *) B, 64, sizeof(double)*k*n);
  C = (double *) hbw_malloc(sizeof(double)*m*n);
  hbw_posix_memalign((void *) C, 64, sizeof(double)*m*n);  // */
  // generate matrices
  row_major_matrix(&A[0], m, k, 777,  -1.0,  1.0);
  row_major_matrix(&B[0], k, n, 888, -10.0, 10.0);
  row_major_matrix(&C[0], m, n, 999, -20.0, 20.0);
  //col_major_matrix(&A[0], m, k, 777,  -1.0,  1.0);
  //col_major_matrix(&B[0], k, n, 888, -10.0, 10.0);
  //col_major_matrix(&C[0], m, n, 999, -20.0, 20.0);
//  printf("m=%d n=%d k=%d\n mb=%d kb=%d\n",m,n,k,MB,KB);
  // test
//  double * bufToFlushLlc;
//  static const double *bufToFlushLlc = NULL;
	printf("m, n, k : %d %d %d\n", m, n, k);
	for(q = 0; q<NUM_ITER ; q++)
	{
  startTime = omp_get_wtime();
  for(i = 0; i < iter; ++i)
  {
    // dgemm
//    micro_kernel0(k, A, B, C, n);
		ijrloop(m,n,0,k,0,A,k,B,C,n);
   //cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
   //  m, n, k, 1.0, A, k, B, n, 1.0, C, n);
  //    
   // p1d_dgemm(m ,n, k, A, k, B, n, C, n);
	//printf("user_dgemm\n");
    // flops
  }
  endTime = omp_get_wtime();
  //printf("done\n");
  gap = (double)( endTime - startTime );
  flops = flopsArray[q]=(2.0 * (double)m * (double)n * (double)k) * 1.0e-9 / gap * iter;

  //get maxFlops
  if(maxFlops < flops){
    maxFlops = flops;
  }
  //sort flops
  for(j = q; j>0; --j)
  {
    if(flops > flopsArray[j-1]){
      flopsArray[j] = flopsArray[j-1];
    }else{
      break;
    }
  }
  flopsArray[j] = flops;
  printf("dgemm time (sec)\t%f, Gflops\t%f \n", gap, flops); 
}

flops = 0;
for(i = 0; i < NUM_ITER-2; ++i)
{
  flops += flopsArray[i];
}

//printf("done\n");
printf("maximum Gflops : %f\n", maxFlops);
printf("average Gflops : %f\n", flops/(NUM_ITER-2));	
	
	

  /*free(A);
  free(B);
  free(C); // */
	//printf("free\n");
  // MCDRAM version
  hbw_free(A);
  hbw_free(B);
  hbw_free(C);  // */
  //free(bufToFlushLlc);
  }
