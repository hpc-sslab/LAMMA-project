//Fig10_col

#include "immintrin.h"
#include "x86intrin.h"
#include "zmmintrin.h"
#include <hbwmalloc.h>

#define MR 8
#define NR 31
#define MB 1000
#define NB 124
#define KB 336

#define NT 8
#define NT1 8
#define NT2 2

#define L1_DIST_A 160
#define L1_DIST_B 620

void micro_kernel0(int k, const double * A, const double * B, double * C, int ncol)
{
  int i;
  register __m512d _A0;
  register __m512d _C0_0, _C1_0, _C2_0, _C3_0, _C4_0, _C5_0, _C6_0, _C7_0, _C8_0, _C9_0, _C10_0, _C11_0, _C12_0, _C13_0, _C14_0, _C15_0, _C16_0, _C17_0, _C18_0, _C19_0, _C20_0, _C21_0, _C22_0, _C23_0, _C24_0, _C25_0, _C26_0, _C27_0, _C28_0, _C29_0, _C30_0;
    _C0_0 = _mm512_loadu_pd(&C[0*ncol+0]);
    _C1_0 = _mm512_loadu_pd(&C[1*ncol+0]);
    _C2_0 = _mm512_loadu_pd(&C[2*ncol+0]);
    _C3_0 = _mm512_loadu_pd(&C[3*ncol+0]);
    _C4_0 = _mm512_loadu_pd(&C[4*ncol+0]);
    _C5_0 = _mm512_loadu_pd(&C[5*ncol+0]);
    _C6_0 = _mm512_loadu_pd(&C[6*ncol+0]);
    _C7_0 = _mm512_loadu_pd(&C[7*ncol+0]);
    _C8_0 = _mm512_loadu_pd(&C[8*ncol+0]);
    _C9_0 = _mm512_loadu_pd(&C[9*ncol+0]);
    _C10_0 = _mm512_loadu_pd(&C[10*ncol+0]);
    _C11_0 = _mm512_loadu_pd(&C[11*ncol+0]);
    _C12_0 = _mm512_loadu_pd(&C[12*ncol+0]);
    _C13_0 = _mm512_loadu_pd(&C[13*ncol+0]);
    _C14_0 = _mm512_loadu_pd(&C[14*ncol+0]);
    _C15_0 = _mm512_loadu_pd(&C[15*ncol+0]);
    _C16_0 = _mm512_loadu_pd(&C[16*ncol+0]);
    _C17_0 = _mm512_loadu_pd(&C[17*ncol+0]);
    _C18_0 = _mm512_loadu_pd(&C[18*ncol+0]);
    _C19_0 = _mm512_loadu_pd(&C[19*ncol+0]);
    _C20_0 = _mm512_loadu_pd(&C[20*ncol+0]);
    _C21_0 = _mm512_loadu_pd(&C[21*ncol+0]);
    _C22_0 = _mm512_loadu_pd(&C[22*ncol+0]);
    _C23_0 = _mm512_loadu_pd(&C[23*ncol+0]);
    _C24_0 = _mm512_loadu_pd(&C[24*ncol+0]);
    _C25_0 = _mm512_loadu_pd(&C[25*ncol+0]);
    _C26_0 = _mm512_loadu_pd(&C[26*ncol+0]);
    _C27_0 = _mm512_loadu_pd(&C[27*ncol+0]);
    _C28_0 = _mm512_loadu_pd(&C[28*ncol+0]);
    _C29_0 = _mm512_loadu_pd(&C[29*ncol+0]);
    _C30_0 = _mm512_loadu_pd(&C[30*ncol+0]);
  // _C += A*B
  #pragma unroll(3)
  for(i=0; i<k ; i++)
  {
    // A L1 prefetch
    _mm_prefetch((const void*) &A[L1_DIST_A+0],_MM_HINT_T0);
    // B L1 prefetch
    _mm_prefetch((const void*) &B[L1_DIST_B+0],_MM_HINT_T0);
    _mm_prefetch((const void*) &B[L1_DIST_B+8],_MM_HINT_T0);
    _mm_prefetch((const void*) &B[L1_DIST_B+16],_MM_HINT_T0);
    _mm_prefetch((const void*) &B[L1_DIST_B+24],_MM_HINT_T0);
    _A0 = _mm512_loadu_pd(&A[0]);
    _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0]), _A0, _C0_0);
    _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1]), _A0, _C1_0);
    _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2]), _A0, _C2_0);
    _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3]), _A0, _C3_0);
    _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4]), _A0, _C4_0);
    _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5]), _A0, _C5_0);
    _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6]), _A0, _C6_0);
    _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7]), _A0, _C7_0);
    _C8_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8]), _A0, _C8_0);
    _C9_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[9]), _A0, _C9_0);
    _C10_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[10]), _A0, _C10_0);
    _C11_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[11]), _A0, _C11_0);
    _C12_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[12]), _A0, _C12_0);
    _C13_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[13]), _A0, _C13_0);
    _C14_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[14]), _A0, _C14_0);
    _C15_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[15]), _A0, _C15_0);
    _C16_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[16]), _A0, _C16_0);
    _C17_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[17]), _A0, _C17_0);
    _C18_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[18]), _A0, _C18_0);
    _C19_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[19]), _A0, _C19_0);
    _C20_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[20]), _A0, _C20_0);
    _C21_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[21]), _A0, _C21_0);
    _C22_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[22]), _A0, _C22_0);
    _C23_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[23]), _A0, _C23_0);
    _C24_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[24]), _A0, _C24_0);
    _C25_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[25]), _A0, _C25_0);
    _C26_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[26]), _A0, _C26_0);
    _C27_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[27]), _A0, _C27_0);
    _C28_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[28]), _A0, _C28_0);
    _C29_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[29]), _A0, _C29_0);
    _C30_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[30]), _A0, _C30_0);
    A += MR;
    B += NR;
  }
  // store _C -> C
  _mm512_storeu_pd(&C[0*ncol+0], _C0_0);
  _mm512_storeu_pd(&C[1*ncol+0], _C1_0);
  _mm512_storeu_pd(&C[2*ncol+0], _C2_0);
  _mm512_storeu_pd(&C[3*ncol+0], _C3_0);
  _mm512_storeu_pd(&C[4*ncol+0], _C4_0);
  _mm512_storeu_pd(&C[5*ncol+0], _C5_0);
  _mm512_storeu_pd(&C[6*ncol+0], _C6_0);
  _mm512_storeu_pd(&C[7*ncol+0], _C7_0);
  _mm512_storeu_pd(&C[8*ncol+0], _C8_0);
  _mm512_storeu_pd(&C[9*ncol+0], _C9_0);
  _mm512_storeu_pd(&C[10*ncol+0], _C10_0);
  _mm512_storeu_pd(&C[11*ncol+0], _C11_0);
  _mm512_storeu_pd(&C[12*ncol+0], _C12_0);
  _mm512_storeu_pd(&C[13*ncol+0], _C13_0);
  _mm512_storeu_pd(&C[14*ncol+0], _C14_0);
  _mm512_storeu_pd(&C[15*ncol+0], _C15_0);
  _mm512_storeu_pd(&C[16*ncol+0], _C16_0);
  _mm512_storeu_pd(&C[17*ncol+0], _C17_0);
  _mm512_storeu_pd(&C[18*ncol+0], _C18_0);
  _mm512_storeu_pd(&C[19*ncol+0], _C19_0);
  _mm512_storeu_pd(&C[20*ncol+0], _C20_0);
  _mm512_storeu_pd(&C[21*ncol+0], _C21_0);
  _mm512_storeu_pd(&C[22*ncol+0], _C22_0);
  _mm512_storeu_pd(&C[23*ncol+0], _C23_0);
  _mm512_storeu_pd(&C[24*ncol+0], _C24_0);
  _mm512_storeu_pd(&C[25*ncol+0], _C25_0);
  _mm512_storeu_pd(&C[26*ncol+0], _C26_0);
  _mm512_storeu_pd(&C[27*ncol+0], _C27_0);
  _mm512_storeu_pd(&C[28*ncol+0], _C28_0);
  _mm512_storeu_pd(&C[29*ncol+0], _C29_0);
  _mm512_storeu_pd(&C[30*ncol+0], _C30_0);
}
void micro_kernel1(int k, const double * A, const double * B, double * C, int ncol)
{
  int i;
  register __m512d _A0;
  register __m512d _C0_0, _C1_0, _C2_0, _C3_0, _C4_0, _C5_0, _C6_0, _C7_0, _C8_0, _C9_0, _C10_0, _C11_0, _C12_0, _C13_0, _C14_0, _C15_0, _C16_0, _C17_0, _C18_0, _C19_0, _C20_0, _C21_0, _C22_0, _C23_0, _C24_0, _C25_0, _C26_0, _C27_0, _C28_0, _C29_0, _C30_0;
    _C0_0 = _mm512_setzero_pd();
    _C1_0 = _mm512_setzero_pd();
    _C2_0 = _mm512_setzero_pd();
    _C3_0 = _mm512_setzero_pd();
    _C4_0 = _mm512_setzero_pd();
    _C5_0 = _mm512_setzero_pd();
    _C6_0 = _mm512_setzero_pd();
    _C7_0 = _mm512_setzero_pd();
    _C8_0 = _mm512_setzero_pd();
    _C9_0 = _mm512_setzero_pd();
    _C10_0 = _mm512_setzero_pd();
    _C11_0 = _mm512_setzero_pd();
    _C12_0 = _mm512_setzero_pd();
    _C13_0 = _mm512_setzero_pd();
    _C14_0 = _mm512_setzero_pd();
    _C15_0 = _mm512_setzero_pd();
    _C16_0 = _mm512_setzero_pd();
    _C17_0 = _mm512_setzero_pd();
    _C18_0 = _mm512_setzero_pd();
    _C19_0 = _mm512_setzero_pd();
    _C20_0 = _mm512_setzero_pd();
    _C21_0 = _mm512_setzero_pd();
    _C22_0 = _mm512_setzero_pd();
    _C23_0 = _mm512_setzero_pd();
    _C24_0 = _mm512_setzero_pd();
    _C25_0 = _mm512_setzero_pd();
    _C26_0 = _mm512_setzero_pd();
    _C27_0 = _mm512_setzero_pd();
    _C28_0 = _mm512_setzero_pd();
    _C29_0 = _mm512_setzero_pd();
    _C30_0 = _mm512_setzero_pd();
  // _C += A*B
  #pragma unroll(3)
  for(i=0; i<k ; i++)
  {
    // A L1 prefetch
    _mm_prefetch((const void*) &A[L1_DIST_A+0],_MM_HINT_T0);
    // B L1 prefetch
    _mm_prefetch((const void*) &B[L1_DIST_B+0],_MM_HINT_T0);
    _mm_prefetch((const void*) &B[L1_DIST_B+8],_MM_HINT_T0);
    _mm_prefetch((const void*) &B[L1_DIST_B+16],_MM_HINT_T0);
    _mm_prefetch((const void*) &B[L1_DIST_B+24],_MM_HINT_T0);
    _A0 = _mm512_loadu_pd(&A[0]);
    _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0]), _A0, _C0_0);
    _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1]), _A0, _C1_0);
    _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2]), _A0, _C2_0);
    _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3]), _A0, _C3_0);
    _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4]), _A0, _C4_0);
    _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5]), _A0, _C5_0);
    _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6]), _A0, _C6_0);
    _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7]), _A0, _C7_0);
    _C8_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8]), _A0, _C8_0);
    _C9_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[9]), _A0, _C9_0);
    _C10_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[10]), _A0, _C10_0);
    _C11_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[11]), _A0, _C11_0);
    _C12_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[12]), _A0, _C12_0);
    _C13_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[13]), _A0, _C13_0);
    _C14_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[14]), _A0, _C14_0);
    _C15_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[15]), _A0, _C15_0);
    _C16_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[16]), _A0, _C16_0);
    _C17_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[17]), _A0, _C17_0);
    _C18_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[18]), _A0, _C18_0);
    _C19_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[19]), _A0, _C19_0);
    _C20_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[20]), _A0, _C20_0);
    _C21_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[21]), _A0, _C21_0);
    _C22_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[22]), _A0, _C22_0);
    _C23_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[23]), _A0, _C23_0);
    _C24_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[24]), _A0, _C24_0);
    _C25_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[25]), _A0, _C25_0);
    _C26_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[26]), _A0, _C26_0);
    _C27_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[27]), _A0, _C27_0);
    _C28_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[28]), _A0, _C28_0);
    _C29_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[29]), _A0, _C29_0);
    _C30_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[30]), _A0, _C30_0);
    A += MR;
    B += NR;
  }
  // store _C -> C
  _mm512_storeu_pd(&C[0*ncol+0], _C0_0);
  _mm512_storeu_pd(&C[1*ncol+0], _C1_0);
  _mm512_storeu_pd(&C[2*ncol+0], _C2_0);
  _mm512_storeu_pd(&C[3*ncol+0], _C3_0);
  _mm512_storeu_pd(&C[4*ncol+0], _C4_0);
  _mm512_storeu_pd(&C[5*ncol+0], _C5_0);
  _mm512_storeu_pd(&C[6*ncol+0], _C6_0);
  _mm512_storeu_pd(&C[7*ncol+0], _C7_0);
  _mm512_storeu_pd(&C[8*ncol+0], _C8_0);
  _mm512_storeu_pd(&C[9*ncol+0], _C9_0);
  _mm512_storeu_pd(&C[10*ncol+0], _C10_0);
  _mm512_storeu_pd(&C[11*ncol+0], _C11_0);
  _mm512_storeu_pd(&C[12*ncol+0], _C12_0);
  _mm512_storeu_pd(&C[13*ncol+0], _C13_0);
  _mm512_storeu_pd(&C[14*ncol+0], _C14_0);
  _mm512_storeu_pd(&C[15*ncol+0], _C15_0);
  _mm512_storeu_pd(&C[16*ncol+0], _C16_0);
  _mm512_storeu_pd(&C[17*ncol+0], _C17_0);
  _mm512_storeu_pd(&C[18*ncol+0], _C18_0);
  _mm512_storeu_pd(&C[19*ncol+0], _C19_0);
  _mm512_storeu_pd(&C[20*ncol+0], _C20_0);
  _mm512_storeu_pd(&C[21*ncol+0], _C21_0);
  _mm512_storeu_pd(&C[22*ncol+0], _C22_0);
  _mm512_storeu_pd(&C[23*ncol+0], _C23_0);
  _mm512_storeu_pd(&C[24*ncol+0], _C24_0);
  _mm512_storeu_pd(&C[25*ncol+0], _C25_0);
  _mm512_storeu_pd(&C[26*ncol+0], _C26_0);
  _mm512_storeu_pd(&C[27*ncol+0], _C27_0);
  _mm512_storeu_pd(&C[28*ncol+0], _C28_0);
  _mm512_storeu_pd(&C[29*ncol+0], _C29_0);
  _mm512_storeu_pd(&C[30*ncol+0], _C30_0);
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
	register __m512d x0;
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
			//bk[i*MR+j*col*MR:MR] = mt[i*inc+j*MR:MR];
			 x0 = _mm512_loadu_pd(&mt[i*inc+j*MR]);
			 _mm512_storeu_pd(&bk[(i+0)*MR+j*col*MR], x0);
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
/*
void userdgemm(const int              m,
		const int              n,
		const int              k,
		const double * A,
		const int              la, // leading dimension of A
		const double * B,
		const int              lb, // leading dimension of B
		double       * C,
		const int              lc) // leading dimension of C
{//*/

void userdgemm(
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

//printf("Im in userdgemm\n");
	int mq = (m+MB-1) / MB;
	int md = m % MB;
	int mc;
	int kq = (k+KB-1) / KB;
	int kd = k % KB;
	int kc;
	int ii;
	int l;
	double *_A;
	omp_set_nested(1);
	_A = (double *) hbw_malloc(sizeof(double)*MB*KB);
	hbw_posix_memalign((void *) _A, 64, sizeof(double)*MB*KB); 

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
