//Fig10_col
#include "PBLAS/SRC/pblas.h"
#include "PBLAS/SRC/PBpblas.h"
#include "PBLAS/SRC/PBtools.h"
#include "PBLAS/SRC/PBblacs.h"
#include "PBLAS/SRC/PBblas.h"
#include "immintrin.h"
#include "x86intrin.h"
#include "zmmintrin.h"

#define MR 16
#define NR 14
#define MB 4688
#define NB 84
#define KB 501

#define NT 20
#define NT1 20
#define NT2 2

#define L1_DIST_A 320
#define L1_DIST_B 280

void micro_kernel0(int k, const double * A, const double * B, double * C, int ncol)
{
  int i;
  register __m512d _A0, _A1;
  register __m512d _C0_0, _C1_0, _C2_0, _C3_0, _C4_0, _C5_0, _C6_0, _C7_0, _C8_0, _C9_0, _C10_0, _C11_0, _C12_0, _C13_0;
  register __m512d _C0_1, _C1_1, _C2_1, _C3_1, _C4_1, _C5_1, _C6_1, _C7_1, _C8_1, _C9_1, _C10_1, _C11_1, _C12_1, _C13_1;
    _C0_0 = _mm512_loadu_pd(&C[0*ncol+0]);
    _C0_1 = _mm512_loadu_pd(&C[0*ncol+8]);
    _C1_0 = _mm512_loadu_pd(&C[1*ncol+0]);
    _C1_1 = _mm512_loadu_pd(&C[1*ncol+8]);
    _C2_0 = _mm512_loadu_pd(&C[2*ncol+0]);
    _C2_1 = _mm512_loadu_pd(&C[2*ncol+8]);
    _C3_0 = _mm512_loadu_pd(&C[3*ncol+0]);
    _C3_1 = _mm512_loadu_pd(&C[3*ncol+8]);
    _C4_0 = _mm512_loadu_pd(&C[4*ncol+0]);
    _C4_1 = _mm512_loadu_pd(&C[4*ncol+8]);
    _C5_0 = _mm512_loadu_pd(&C[5*ncol+0]);
    _C5_1 = _mm512_loadu_pd(&C[5*ncol+8]);
    _C6_0 = _mm512_loadu_pd(&C[6*ncol+0]);
    _C6_1 = _mm512_loadu_pd(&C[6*ncol+8]);
    _C7_0 = _mm512_loadu_pd(&C[7*ncol+0]);
    _C7_1 = _mm512_loadu_pd(&C[7*ncol+8]);
    _C8_0 = _mm512_loadu_pd(&C[8*ncol+0]);
    _C8_1 = _mm512_loadu_pd(&C[8*ncol+8]);
    _C9_0 = _mm512_loadu_pd(&C[9*ncol+0]);
    _C9_1 = _mm512_loadu_pd(&C[9*ncol+8]);
    _C10_0 = _mm512_loadu_pd(&C[10*ncol+0]);
    _C10_1 = _mm512_loadu_pd(&C[10*ncol+8]);
    _C11_0 = _mm512_loadu_pd(&C[11*ncol+0]);
    _C11_1 = _mm512_loadu_pd(&C[11*ncol+8]);
    _C12_0 = _mm512_loadu_pd(&C[12*ncol+0]);
    _C12_1 = _mm512_loadu_pd(&C[12*ncol+8]);
    _C13_0 = _mm512_loadu_pd(&C[13*ncol+0]);
    _C13_1 = _mm512_loadu_pd(&C[13*ncol+8]);
  // _C += A*B
  #pragma unroll(1)
  for(i=0; i<k ; i++)
  {
    // A L1 prefetch
    _mm_prefetch((const void*) &A[L1_DIST_A+0],_MM_HINT_T0);
    _mm_prefetch((const void*) &A[L1_DIST_A+8],_MM_HINT_T0);
    // B L1 prefetch
    _mm_prefetch((const void*) &B[L1_DIST_B+0],_MM_HINT_T0);
    _mm_prefetch((const void*) &B[L1_DIST_B+8],_MM_HINT_T0);
    _A0 = _mm512_loadu_pd(&A[0]);
    _A1 = _mm512_loadu_pd(&A[8]);
    _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0]), _A0, _C0_0);
    _C0_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0]), _A1, _C0_1);
    _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1]), _A0, _C1_0);
    _C1_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1]), _A1, _C1_1);
    _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2]), _A0, _C2_0);
    _C2_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2]), _A1, _C2_1);
    _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3]), _A0, _C3_0);
    _C3_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3]), _A1, _C3_1);
    _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4]), _A0, _C4_0);
    _C4_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4]), _A1, _C4_1);
    _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5]), _A0, _C5_0);
    _C5_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5]), _A1, _C5_1);
    _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6]), _A0, _C6_0);
    _C6_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6]), _A1, _C6_1);
    _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7]), _A0, _C7_0);
    _C7_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7]), _A1, _C7_1);
    _C8_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8]), _A0, _C8_0);
    _C8_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8]), _A1, _C8_1);
    _C9_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[9]), _A0, _C9_0);
    _C9_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[9]), _A1, _C9_1);
    _C10_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[10]), _A0, _C10_0);
    _C10_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[10]), _A1, _C10_1);
    _C11_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[11]), _A0, _C11_0);
    _C11_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[11]), _A1, _C11_1);
    _C12_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[12]), _A0, _C12_0);
    _C12_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[12]), _A1, _C12_1);
    _C13_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[13]), _A0, _C13_0);
    _C13_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[13]), _A1, _C13_1);
    A += MR;
    B += NR;
  }
  // store _C -> C
  _mm512_storeu_pd(&C[0*ncol+0], _C0_0);
  _mm512_storeu_pd(&C[0*ncol+8], _C0_1);
  _mm512_storeu_pd(&C[1*ncol+0], _C1_0);
  _mm512_storeu_pd(&C[1*ncol+8], _C1_1);
  _mm512_storeu_pd(&C[2*ncol+0], _C2_0);
  _mm512_storeu_pd(&C[2*ncol+8], _C2_1);
  _mm512_storeu_pd(&C[3*ncol+0], _C3_0);
  _mm512_storeu_pd(&C[3*ncol+8], _C3_1);
  _mm512_storeu_pd(&C[4*ncol+0], _C4_0);
  _mm512_storeu_pd(&C[4*ncol+8], _C4_1);
  _mm512_storeu_pd(&C[5*ncol+0], _C5_0);
  _mm512_storeu_pd(&C[5*ncol+8], _C5_1);
  _mm512_storeu_pd(&C[6*ncol+0], _C6_0);
  _mm512_storeu_pd(&C[6*ncol+8], _C6_1);
  _mm512_storeu_pd(&C[7*ncol+0], _C7_0);
  _mm512_storeu_pd(&C[7*ncol+8], _C7_1);
  _mm512_storeu_pd(&C[8*ncol+0], _C8_0);
  _mm512_storeu_pd(&C[8*ncol+8], _C8_1);
  _mm512_storeu_pd(&C[9*ncol+0], _C9_0);
  _mm512_storeu_pd(&C[9*ncol+8], _C9_1);
  _mm512_storeu_pd(&C[10*ncol+0], _C10_0);
  _mm512_storeu_pd(&C[10*ncol+8], _C10_1);
  _mm512_storeu_pd(&C[11*ncol+0], _C11_0);
  _mm512_storeu_pd(&C[11*ncol+8], _C11_1);
  _mm512_storeu_pd(&C[12*ncol+0], _C12_0);
  _mm512_storeu_pd(&C[12*ncol+8], _C12_1);
  _mm512_storeu_pd(&C[13*ncol+0], _C13_0);
  _mm512_storeu_pd(&C[13*ncol+8], _C13_1);
}
void micro_kernel1(int k, const double * A, const double * B, double * C, int ncol)
{
  int i;
  register __m512d _A0, _A1;
  register __m512d _C0_0, _C1_0, _C2_0, _C3_0, _C4_0, _C5_0, _C6_0, _C7_0, _C8_0, _C9_0, _C10_0, _C11_0, _C12_0, _C13_0;
  register __m512d _C0_1, _C1_1, _C2_1, _C3_1, _C4_1, _C5_1, _C6_1, _C7_1, _C8_1, _C9_1, _C10_1, _C11_1, _C12_1, _C13_1;
    _C0_0 = _mm512_setzero_pd();
    _C0_1 = _mm512_setzero_pd();
    _C1_0 = _mm512_setzero_pd();
    _C1_1 = _mm512_setzero_pd();
    _C2_0 = _mm512_setzero_pd();
    _C2_1 = _mm512_setzero_pd();
    _C3_0 = _mm512_setzero_pd();
    _C3_1 = _mm512_setzero_pd();
    _C4_0 = _mm512_setzero_pd();
    _C4_1 = _mm512_setzero_pd();
    _C5_0 = _mm512_setzero_pd();
    _C5_1 = _mm512_setzero_pd();
    _C6_0 = _mm512_setzero_pd();
    _C6_1 = _mm512_setzero_pd();
    _C7_0 = _mm512_setzero_pd();
    _C7_1 = _mm512_setzero_pd();
    _C8_0 = _mm512_setzero_pd();
    _C8_1 = _mm512_setzero_pd();
    _C9_0 = _mm512_setzero_pd();
    _C9_1 = _mm512_setzero_pd();
    _C10_0 = _mm512_setzero_pd();
    _C10_1 = _mm512_setzero_pd();
    _C11_0 = _mm512_setzero_pd();
    _C11_1 = _mm512_setzero_pd();
    _C12_0 = _mm512_setzero_pd();
    _C12_1 = _mm512_setzero_pd();
    _C13_0 = _mm512_setzero_pd();
    _C13_1 = _mm512_setzero_pd();
  // _C += A*B
  #pragma unroll(1)
  for(i=0; i<k ; i++)
  {
    // A L1 prefetch
    _mm_prefetch((const void*) &A[L1_DIST_A+0],_MM_HINT_T0);
    _mm_prefetch((const void*) &A[L1_DIST_A+8],_MM_HINT_T0);
    // B L1 prefetch
    _mm_prefetch((const void*) &B[L1_DIST_B+0],_MM_HINT_T0);
    _mm_prefetch((const void*) &B[L1_DIST_B+8],_MM_HINT_T0);
    _A0 = _mm512_loadu_pd(&A[0]);
    _A1 = _mm512_loadu_pd(&A[8]);
    _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0]), _A0, _C0_0);
    _C0_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0]), _A1, _C0_1);
    _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1]), _A0, _C1_0);
    _C1_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1]), _A1, _C1_1);
    _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2]), _A0, _C2_0);
    _C2_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2]), _A1, _C2_1);
    _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3]), _A0, _C3_0);
    _C3_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3]), _A1, _C3_1);
    _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4]), _A0, _C4_0);
    _C4_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4]), _A1, _C4_1);
    _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5]), _A0, _C5_0);
    _C5_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5]), _A1, _C5_1);
    _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6]), _A0, _C6_0);
    _C6_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6]), _A1, _C6_1);
    _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7]), _A0, _C7_0);
    _C7_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7]), _A1, _C7_1);
    _C8_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8]), _A0, _C8_0);
    _C8_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8]), _A1, _C8_1);
    _C9_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[9]), _A0, _C9_0);
    _C9_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[9]), _A1, _C9_1);
    _C10_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[10]), _A0, _C10_0);
    _C10_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[10]), _A1, _C10_1);
    _C11_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[11]), _A0, _C11_0);
    _C11_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[11]), _A1, _C11_1);
    _C12_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[12]), _A0, _C12_0);
    _C12_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[12]), _A1, _C12_1);
    _C13_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[13]), _A0, _C13_0);
    _C13_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[13]), _A1, _C13_1);
    A += MR;
    B += NR;
  }
  // store _C -> C
  _mm512_storeu_pd(&C[0*ncol+0], _C0_0);
  _mm512_storeu_pd(&C[0*ncol+8], _C0_1);
  _mm512_storeu_pd(&C[1*ncol+0], _C1_0);
  _mm512_storeu_pd(&C[1*ncol+8], _C1_1);
  _mm512_storeu_pd(&C[2*ncol+0], _C2_0);
  _mm512_storeu_pd(&C[2*ncol+8], _C2_1);
  _mm512_storeu_pd(&C[3*ncol+0], _C3_0);
  _mm512_storeu_pd(&C[3*ncol+8], _C3_1);
  _mm512_storeu_pd(&C[4*ncol+0], _C4_0);
  _mm512_storeu_pd(&C[4*ncol+8], _C4_1);
  _mm512_storeu_pd(&C[5*ncol+0], _C5_0);
  _mm512_storeu_pd(&C[5*ncol+8], _C5_1);
  _mm512_storeu_pd(&C[6*ncol+0], _C6_0);
  _mm512_storeu_pd(&C[6*ncol+8], _C6_1);
  _mm512_storeu_pd(&C[7*ncol+0], _C7_0);
  _mm512_storeu_pd(&C[7*ncol+8], _C7_1);
  _mm512_storeu_pd(&C[8*ncol+0], _C8_0);
  _mm512_storeu_pd(&C[8*ncol+8], _C8_1);
  _mm512_storeu_pd(&C[9*ncol+0], _C9_0);
  _mm512_storeu_pd(&C[9*ncol+8], _C9_1);
  _mm512_storeu_pd(&C[10*ncol+0], _C10_0);
  _mm512_storeu_pd(&C[10*ncol+8], _C10_1);
  _mm512_storeu_pd(&C[11*ncol+0], _C11_0);
  _mm512_storeu_pd(&C[11*ncol+8], _C11_1);
  _mm512_storeu_pd(&C[12*ncol+0], _C12_0);
  _mm512_storeu_pd(&C[12*ncol+8], _C12_1);
  _mm512_storeu_pd(&C[13*ncol+0], _C13_0);
  _mm512_storeu_pd(&C[13*ncol+8], _C13_1);
}

// notation
// m = n*q + d
// quotient : Xq
// remainder: Xd

// Packing A, c-major -> c-major
void packacc(int           row, // # of rows
		int           col, // # of cols
		const double *mt, // from *mt
		int           inc, // distance
		double       *bk) // to *bk
{
	int q = row / MR; // quotient
	int r = row % MR; // remainder
	int i, j;
	omp_set_nested(1); // just in case
	// main part
	// for type 10 algorithm, \tilde{A} is quite big -> rich to parallelize
	#pragma omp parallel for num_threads(NT) private(i) schedule(dynamic)
	for(j = 0; j < q; ++j)
	{
		for(i = 0; i < col; ++i)
		{
			// copy NR elements from &mt[i*inc+j*NR] with interval 1
			// loop order is j-loop -> i-loop to access bk contiguously
			bk[i*MR+j*col*MR:MR] = mt[i*inc+j*MR:MR];
		}
	}
	// move pointers
	bk += q*col*MR;
	mt += q*MR;
	// remainder part
	if(r > 0)
	{
		for(i = 0; i < col; ++i)
		{
			bk[0:r] = mt[0:r];
			bk[r:MR-r] = 0.0;
			bk += MR;
			mt += inc;
		}
	}
}
// Packing A, r-major -> c-major
void packarc(int           row, // # of rows
        int           col, // # of cols
        const double *mt, //  from *mt
        int           inc, // distance
        double       *bk ) // to *bk
{
    int q = row / MR; // quotient
    int r = row % MR; // remainder
    int i, j;
    omp_set_nested(1); // just in case
    // main part
    // for type 10 algorithm, \tilde{A} is quite big -> rich to parallelize
    #pragma omp parallel for num_threads(NT) private(j) schedule(dynamic)
    for(i = 0; i < q; ++i)
    {
        for(j = 0; j < col; ++j)
        {
            // copy MR elements from &mt[j+i*MR*inc] with interval inc
            // loop order is i-loop -> j-loop to access bk contiguously
            bk[j*MR+i*col*MR:MR]=mt[j+i*MR*inc:MR:inc];
        }
    }
    // move pointers
    bk += q*col*MR;
    mt += q*inc*MR;
    // remainder part
    if(r > 0)
    {
        for(j = 0; j < col; ++j)
        {
            bk[0:r] = mt[0:r:inc];
            bk[r:MR-r] = 0.0;
            bk += MR;
            mt += 1;
        }
    }
}

// Packing B, c-major -> r-major
void packbcr(int           row, // # of rows
		int           col, // # of cols
		const double *mt, //  from *mt
		int           inc, // distance
		double       *bk ) // to *bk
{
	int q = col / NR; // quotient
	int r = col % NR; // remainder
	int i, j;
	omp_set_nested(1); // just in case
	// main part
	// for type 10 algorithm, \tilde{B} is quite small and inside of jir loop
	// so poor chance to parallelize
	for(i = 0; i < q; ++i)
	{
		for(j = 0; j < row; ++j)
		{
			// copy MR elements from &mt[j+i*MR*inc] with interval inc
			// loop order is i-loop -> j-loop to access bk contiguously
			bk[j*NR+i*row*NR:NR]=mt[j+i*NR*inc:NR:inc]; 
		}
	}
	// move pointers
	bk += q*row*NR;
	mt += q*inc*NR;
	// remainder part
	if(r > 0)
	{
		for(j = 0; j < row; ++j)
		{
			bk[0:r] = mt[0:r:inc];
			bk[r:NR-r] = 0.0;
			bk += NR;
			mt += 1;
		}
	}
}
// Packing B, r-major -> r-major
void packbrr(int           row, // # of rows
        int           col, // # of cols
        const double *mt, // from *mt
        int           inc, // distance
        double       *bk) // to *bk
{
    int q = col / NR; // quotient
    int r = col % NR; // remainder
    int i, j;

    // main part
    // for type 10 algorithm, \tilde{B} is quite small and inside of jir loop
    // so poor chance to parallelize
    for(j = 0; j < q; ++j)
    {
        for(i = 0; i < row; ++i)
        {
            // copy NR elements from &mt[i*inc+j*NR] with interval 1
            // loop order is j-loop -> i-loop to access bk contiguously
            bk[i*NR+j*row*NR:NR] = mt[i*inc+j*NR:NR];
        }
    }
    // move pointers
    bk += q*row*NR;
    mt += q*NR;
    // remainder part
    if(r > 0)
    {
        for(i = 0; i < row; ++i)
        {
            bk[0:r] = mt[0:r];
            bk[r:NR-r] = 0.0;
            bk += NR;
            mt += inc;
        }
    }
}
// Micro C = _C + C
void micro_dxpy(int                    m,
		int                    n,
		double       * C,
		const double * D,
		int                    ncol)
{
	int i;
	for(i = 0; i < n; ++i)
	{
		C[0:m] += D[i*MR:m];
		C += ncol;
	}
}
//
void jirloop(const int              m,
		const int              n,
		const int              mi,
		const int              k,
		const int              ki,
		const double * A,
		const int              la,
		const double * B,
		const int              lb,
		double       * C,
		const int              lc)
{
	int nq = (n+NB-1) / NB;
	int nd = n % NB;
	int nc;
	int mq = (m+MR-1) / MR;
	int md = m % MR;
	int mc;
	int pq;
	int pd;
	int pc;
	int j;
	int ir;
	int p;
	int ielem;

	double _B[KB*NB] __attribute__((aligned(64)));
	double _C[MR*NR] __attribute__((aligned(64)));
	//
	{
		omp_set_nested(1);
		#pragma omp parallel num_threads(NT1) private(_B, nc, j)
		{
			#pragma omp for schedule(dynamic)
			// j-loop
			for(j = 0; j < nq; ++j)
			{
				nc = (j != nq-1 || nd == 0) ? NB : nd;
				packbcr(k,nc,&B[ki*KB+j*NB*lb],lb,_B);
				#pragma omp parallel num_threads(NT2) private(ir,mc,pq,p,pd,pc,_C) shared(_B, j,nc)
				{
					#pragma omp for //schedule(dynamic)  	
					// ir-loop
					for(ir = 0; ir <mq ; ++ir)
					{
						mc = (ir != mq-1 || md == 0) ? MR : md;
						pq = (nc+NR-1) / NR;
						pd = nc % NR;
						// jr-loop
						for(p = 0; p < pq; ++p)
						{
							pc = (p != pq-1 || pd == 0) ? NR : pd;
							if(pc == NR && mc == MR)
							{
								micro_kernel0(k,&A[ir*MR*k],&_B[p*NR*k],&C[j*NB*lc+p*NR*lc+mi*MB+ir*MR],lc);
							}
							else
							{
								micro_kernel1(k,&A[ir*MR*k],&_B[p*NR*k],_C,MR);
								micro_dxpy(mc,pc,&C[j*NB*lc+p*NR*lc+mi*MB+ir*MR],_C,lc);
							}
						}
					}
				}
			}
		}
	}
}
//
void jirloopT(const int              m,
        const int              n,
        const int              mi,
        const int              k,
        const int              ki,
        const double * A,
        const int              la,
        const double * B,
        const int              lb,
        double       * C,
        const int              lc)
{
    int nq = (n+NB-1) / NB;
    int nd = n % NB;
    int nc;
    int mq = (m+MR-1) / MR;
    int md = m % MR;
    int mc;
    int pq;
    int pd;
    int pc;
    int j;
    int ir;
    int p;
    int ielem;

    double _B[KB*NB] __attribute__((aligned(64)));
    double _C[MR*NR] __attribute__((aligned(64)));
    //
    {
        omp_set_nested(1);
        #pragma omp parallel num_threads(NT1) private(_B, nc, j)
        {
            #pragma omp for schedule(dynamic)
            // j-loop
            for(j = 0; j < nq; ++j)
            {
                nc = (j != nq-1 || nd == 0) ? NB : nd;
                packbrr(k,nc,&B[ki*KB*lb+j*NB],lb,_B);
                #pragma omp parallel num_threads(NT2) private(ir,mc,pq,p,pd,pc,_C) shared(_B, j,nc)
                {
                    #pragma omp for //schedule(dynamic)
                    // ir-loop
                    for(ir = 0; ir <mq ; ++ir)
                    {
                        mc = (ir != mq-1 || md == 0) ? MR : md;
                        pq = (nc+NR-1) / NR;
                        pd = nc % NR;
                        // jr-loop
                        for(p = 0; p < pq; ++p)
                        {
                            pc = (p != pq-1 || pd == 0) ? NR : pd;
                            if(pc == NR && mc == MR)
                            {
								micro_kernel0(k,&A[ir*MR*k],&_B[p*NR*k],&C[j*NB*lc+p*NR*lc+mi*MB+ir*MR],lc);
                             }
                             else
                             {
                                 micro_kernel1(k,&A[ir*MR*k],&_B[p*NR*k],_C,MR);
                                 micro_dxpy(mc,pc,&C[j*NB*lc+p*NR*lc+mi*MB+ir*MR],_C,lc);
                             }
                         }
                     }
                 }
             }
         }
     }
 }
 //
/*void userdgemm(
       char  *transa,
	   char *transb,
	    const int              m,
		const int              n,
		const int              k,
		const double * A,
		const int              la, // leading dimension of A
		const double * B,
		const int              lb, // leading dimension of B
		double       * C,
		const int              lc) // leading dimension of C
{//*/

void userdgemm_(
    char  *transa,
    char *transb,
    int *_m,
    int *_n,
    int *_k,
    char *alpha,
    char *AA,
    int *_la, // distance
    char *BB,
    int *_lb, // distance
    char *beta,
    char *CC,
    int *_lc)
{
  int m = *_m, n = *_n, k = *_k;
  int la = *_la, lb = *_lb, lc = *_lc;
  double *A = (double*)AA;
  double *B = (double*)BB;
  double *C = (double*)CC;
 //*/

 int rowA, colA, rowB, colB;
	int nota, notb;
	char TrA, TrB;
//printf("Im in userdgemm\n");
	 nota = ( ( TrA = Mupcase( F2C_CHAR( transa )[0] ) ) == CNOTRAN );
	 notb = ( ( TrB = Mupcase( F2C_CHAR( transb )[0] ) ) == CNOTRAN );

/*
*	if(nota)
*	{	
*		lda = m;
*	}
*	else{
*		lda = k;
*	}
*
*	if(notb){
*		ldb = k;
*	}
*	else{
*		ldb = n;
*	}
*/	
int mq = (m+MB-1) / MB;
int md = m % MB;
int kq = (k+KB-1) / KB;
int kd = k % KB;
	
	int mc, kc,KC;
	int ii;
	int l;
//	double *_A;
	omp_set_nested(1);
//	_A = (double *) hbw_malloc(sizeof(double)*MB*KB);
//	hbw_posix_memalign((void *) _A, 64, sizeof(double)*MB*KB); 
	static double _A[MB*KB] __attribute__((aligned(64)));


if(nota){
  if(notb){  // N x N
	// I-loop
	for(ii = 0; ii < mq; ++ii)
	{
		mc = (ii != mq-1 || md == 0) ? MB : md;
		for(l = 0; l < kq; ++l)
		{
			kc = (l != kq-1 || kd == 0) ? KB : kd;
			packacc(mc,kc,&A[ii*MB+l*KB*la],la,_A);
				//j-ir loop
				jirloop(mc,n,ii,kc,l,_A,la,B,lb,C,lc);
			}
		}
	}
  else{  // N x T
	for(ii = 0; ii < mq; ++ii)
	{
		mc = (ii != mq-1 || md == 0) ? MB : md;
		for(l = 0; l < kq; ++l)
		{
			kc = (l != kq-1 || kd == 0) ? KB : kd;
			packacc(mc,kc,&A[ii*MB+l*KB*la],la,_A);
				//j-ir loop
				jirloopT(mc,n,ii,kc,l,_A,la,B,lb,C,lc);
			}
		}
	}
}
else{
  if(notb){  // T x N
	// I-loop
	for(ii = 0; ii < mq; ++ii)
	{
		mc = (ii != mq-1 || md == 0) ? MB : md;
		for(l = 0; l < kq; ++l)
		{
			kc = (l != kq-1 || kd == 0) ? KB : kd;
	//		packacc(mc,kc,&A[ii*MB+l*KB*la],la,_A);
				packarc(mc,kc,&A[ii*MB*la+l*KB],la,_A);
				//j-ir loop
				jirloop(mc,n,ii,kc,l,_A,la,B,lb,C,lc);
			}
		}
	}
  else{ // T x T
	for(ii = 0; ii < mq; ++ii)
	{
		mc = (ii != mq-1 || md == 0) ? MB : md;
		for(l = 0; l < kq; ++l)
		{
			kc = (l != kq-1 || kd == 0) ? KB : kd;
	//		packacc(mc,kc,&A[ii*MB+l*KB*la],la,_A);
				packarc(mc,kc,&A[ii*MB*la+l*KB],la,_A);
				//j-ir loop
				jirloopT(mc,n,ii,kc,l,_A,la,B,lb,C,lc);
			}
		}
	}
}


}
