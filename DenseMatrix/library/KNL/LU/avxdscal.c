#include "../pblas.h"
#include "../PBpblas.h"
#include "../PBtools.h"
#include "../PBblacs.h"
#include "../PBblas.h"
#include "immintrin.h"
#include "x86intrin.h"
#include "zmmintrin.h"
#define HPL_rone 1.0
#define HPL_rzero 0.0
void avxdscal_(int *_N, char *_ALPHA, char *_X, int *_INCX)

{
    /* 
 * Purpose
 * =======
 *
 * HPL_dscal scales the vector x by alpha.
 * 
 *
 * Arguments
 * =========
 *
 * N       (local input)                 const int
 *         On entry, N specifies the length of the vector x. N  must  be
 *         at least zero.
 *
 * ALPHA   (local input)                 const double
 *         On entry, ALPHA specifies the scalar alpha.   When  ALPHA  is
 *         supplied as zero, then the entries of the incremented array X
 *         need not be set on input.
 *
 * X       (local input/output)          double *
 *         On entry,  X  is an incremented array of dimension  at  least
 *         ( 1 + ( n - 1 ) * abs( INCX ) )  that  contains the vector x.
 *         On exit, the entries of the incremented array  X  are  scaled
 *         by the scalar alpha.
 *
 * INCX    (local input)                 const int
 *         On entry, INCX specifies the increment for the elements of X.
 *         INCX must not be zero.
 *
 * ---------------------------------------------------------------------
 */
   int  N = *_N;
   double *alpha = (double*)_ALPHA;
   double ALPHA = *alpha;
   double *X = (double*)_X;
   int  INCX = *_INCX;
//	printf("Im in avxdscal\n");
	
//	const register __m512d scalar = _mm512_set1_pd(alpha[0]);
	register __m512d x0, x1, result, tempt, result1;
	register int M, i, mp1, NINCX;
	double *Y = NULL;

	if(INCX == 1)
	{
		// code for increment equal to 1
		M = N % 8;
		if(M != 0)
		{
			x0 = _mm512_loadu_pd(&X[0]);
			tempt = _mm512_mul_pd(x0, _mm512_set1_pd(alpha[0]));
			mp1 = M;
			for( i = mp1; i < N; i += 8)
			{
      			x0 = _mm512_loadu_pd(&X[i]);
      			result = _mm512_mul_pd(x0, _mm512_set1_pd(alpha[0]));
				if( i == M){
					_mm512_storeu_pd(&X[0], tempt);
				}
      			_mm512_storeu_pd(&X[i], result);
      		}
		}

		if( M == 0 )
		{
			for( i = 0; i < N; i += 8)
			{
      			x0 = _mm512_loadu_pd(&X[i]);
      			result = _mm512_mul_pd(x0, _mm512_set1_pd(alpha[0]));
	      		_mm512_storeu_pd(&X[i], result);

    		}
		}
	}
	else{
		// code for increent not eual to 1
		NINCX = N*INCX;
		for(i = 0; i < NINCX; i+= INCX)
		{
		    X[i] = ALPHA*X[i];
		}
	}
	return;
}

