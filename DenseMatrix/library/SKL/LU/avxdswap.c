#include "../pblas.h"
#include "../PBpblas.h"
#include "../PBtools.h"
#include "../PBblacs.h"
#include "../PBblas.h"
#include "immintrin.h"
#include "x86intrin.h"
#include "zmmintrin.h"

#define MR 8
#define NR 16

void Dswap(double *X, double *Y)
{
    register __m512d x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15;
	register __m512d y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15;

    x0 = _mm512_loadu_pd(&X[0]);
    x1 = _mm512_loadu_pd(&X[8 * 1]);
    x2 = _mm512_loadu_pd(&X[8 * 2]);
    x3 = _mm512_loadu_pd(&X[8 * 3]);
    x4 = _mm512_loadu_pd(&X[8 * 4]);
    x5 = _mm512_loadu_pd(&X[8 * 5]);
    x6 = _mm512_loadu_pd(&X[8 * 6]);
    x7 = _mm512_loadu_pd(&X[8 * 7]);
    x8 = _mm512_loadu_pd(&X[8 * 8]);
    x9 = _mm512_loadu_pd(&X[8 * 9]);
    x10 = _mm512_loadu_pd(&X[8 * 10]);
    x11 = _mm512_loadu_pd(&X[8 * 11]);
    x12 = _mm512_loadu_pd(&X[8 * 12]);
    x13 = _mm512_loadu_pd(&X[8 * 13]);
    x14 = _mm512_loadu_pd(&X[8 * 14]);
    x15 = _mm512_loadu_pd(&X[8 * 15]);

    y0 = _mm512_loadu_pd(&Y[0]);
    y1 = _mm512_loadu_pd(&Y[8 * 1]);
    y2 = _mm512_loadu_pd(&Y[8 * 2]);
    y3 = _mm512_loadu_pd(&Y[8 * 3]);
    y4 = _mm512_loadu_pd(&Y[8 * 4]);
    y5 = _mm512_loadu_pd(&Y[8 * 5]);
    y6 = _mm512_loadu_pd(&Y[8 * 6]);
    y7 = _mm512_loadu_pd(&Y[8 * 7]);
    y8 = _mm512_loadu_pd(&Y[8 * 8]);
    y9 = _mm512_loadu_pd(&Y[8 * 9]);
    y10 = _mm512_loadu_pd(&Y[8 * 10]);
    y11 = _mm512_loadu_pd(&Y[8 * 11]);
    y12 = _mm512_loadu_pd(&Y[8 * 12]);
    y13 = _mm512_loadu_pd(&Y[8 * 13]);
    y14 = _mm512_loadu_pd(&Y[8 * 14]);
    y15 = _mm512_loadu_pd(&Y[8 * 15]);



	_mm512_storeu_pd(&X[0], y0);
    _mm512_storeu_pd(&X[8 * 1], y1);
    _mm512_storeu_pd(&X[8 * 2], y2);
    _mm512_storeu_pd(&X[8 * 3], y3);
    _mm512_storeu_pd(&X[8 * 4], y4);
    _mm512_storeu_pd(&X[8 * 5], y5);
    _mm512_storeu_pd(&X[8 * 6], y6);
    _mm512_storeu_pd(&X[8 * 7], y7);
    _mm512_storeu_pd(&X[8 * 8], y8);
    _mm512_storeu_pd(&X[8 * 9], y9);
    _mm512_storeu_pd(&X[8 * 10], y10);
    _mm512_storeu_pd(&X[8 * 11], y11);
    _mm512_storeu_pd(&X[8 * 12], y12);
    _mm512_storeu_pd(&X[8 * 13], y13);
    _mm512_storeu_pd(&X[8 * 14], y14);
    _mm512_storeu_pd(&X[8 * 15], y15);

    _mm512_storeu_pd(&Y[0], x0);
    _mm512_storeu_pd(&Y[8 * 1], x1);
    _mm512_storeu_pd(&Y[8 * 2], x2);
    _mm512_storeu_pd(&Y[8 * 3], x3);
    _mm512_storeu_pd(&Y[8 * 4], x4);
    _mm512_storeu_pd(&Y[8 * 5], x5);
    _mm512_storeu_pd(&Y[8 * 6], x6);
    _mm512_storeu_pd(&Y[8 * 7], x7);
    _mm512_storeu_pd(&Y[8 * 8], x8);
    _mm512_storeu_pd(&Y[8 * 9], x9);
    _mm512_storeu_pd(&Y[8 * 10], x10);
    _mm512_storeu_pd(&Y[8 * 11], x11);
    _mm512_storeu_pd(&Y[8 * 12], x12);
    _mm512_storeu_pd(&Y[8 * 13], x13);
    _mm512_storeu_pd(&Y[8 * 14], x14);
    _mm512_storeu_pd(&Y[8 * 15], x15);
    
}

void avxdswap_(int * _N,  char *_X,  int *_INCX,  char *_Y,    int *_INCY )
{
	int  N = *_N;
	double *X = (double*)_X;
	int  INCX = *_INCX;
	double *Y = (double*)_Y;
	int  INCY = *_INCY;

//printf("Im in avxdswap.c\n");	
	double temp;	
	int i, ix, iy, M, mp1, Mp1, MQ, MP;
	register __m512d x0, x1, y0, y1, x2, y2;
	if( N < 0) return;
	if(INCX == 1 && INCY == 1)
	{
		if (N < 8)
        {
            x1 = _mm512_loadu_pd(&X[0]);
			y1 = _mm512_loadu_pd(&Y[0]);
			
			_mm512_storeu_pd(&X[0], y1);
			_mm512_storeu_pd(&Y[0], x1);
        }
		else{
			MP = N % ( MR * NR);
			if( MP != 0 )
			{
				Mp1 = MP % 8;
				if (Mp1 != 0)
				{
					x0 = _mm512_loadu_pd(&X[0]);
					y0 = _mm512_loadu_pd(&Y[0]);
					mp1 = Mp1;
					for( i = mp1; i < N; i+=16)
					{
						x1 = _mm512_loadu_pd(&X[i]);
						y1 = _mm512_loadu_pd(&Y[i]);
						x2 = _mm512_loadu_pd(&X[i + 8]);
						y2 = _mm512_loadu_pd(&Y[i + 8]);

						if( i == Mp1){
							_mm512_storeu_pd(&X[0], y0);
							_mm512_storeu_pd(&Y[0], x0);
						}
						_mm512_storeu_pd(&X[i], y1);
						_mm512_storeu_pd(&Y[i], x1);
						_mm512_storeu_pd(&X[i + 8], y2);
						_mm512_storeu_pd(&Y[i + 8], x2);
					}
				}
			}

			MQ = (N - MP)/( MR * NR);
			for( i = 0; i < MQ; i++)
			{
				Dswap(&X[MP + i * MR*NR],&Y[MP + i * MR*NR]);
			}
		}
	}
	
	else{
		ix = 0;
		iy = 0;
		if(INCX < 0) ix = (-N + 1)*INCX;
		if(INCY < 0) iy = (-N + 1)*INCY;
		for( i = 0; i < N; i++)
		{
			temp = X[ix];
			X[ix] = Y[iy];
			Y[iy] = temp;
			ix = ix + INCX;
			iy = iy + INCY;
		}
	}
	return;
}

