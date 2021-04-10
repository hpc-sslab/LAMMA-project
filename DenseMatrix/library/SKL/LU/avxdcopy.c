
/*
 * Include files
 */
#include "../pblas.h"
#include "../PBpblas.h"
#include "../PBtools.h"
#include "../PBblacs.h"
#include "../PBblas.h"
#include "immintrin.h"
#include "x86intrin.h"
#include "zmmintrin.h"
void copy(double *X, double *Y)
{
    register __m512d x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31;

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
    x16 = _mm512_loadu_pd(&X[8 * 16]);
    x17 = _mm512_loadu_pd(&X[8 * 17]);
    x18 = _mm512_loadu_pd(&X[8 * 18]);
    x19 = _mm512_loadu_pd(&X[8 * 19]);
    x20 = _mm512_loadu_pd(&X[8 * 20]);
    x21 = _mm512_loadu_pd(&X[8 * 21]);
    x22 = _mm512_loadu_pd(&X[8 * 22]);
    x23 = _mm512_loadu_pd(&X[8 * 23]);
    x24 = _mm512_loadu_pd(&X[8 * 24]);
    x25 = _mm512_loadu_pd(&X[8 * 25]);
    x26 = _mm512_loadu_pd(&X[8 * 26]);
    x27 = _mm512_loadu_pd(&X[8 * 27]);
    x28 = _mm512_loadu_pd(&X[8 * 28]);
    x29 = _mm512_loadu_pd(&X[8 * 29]);
    x30 = _mm512_loadu_pd(&X[8 * 30]);
    x31 = _mm512_loadu_pd(&X[8 * 31]);

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
    _mm512_storeu_pd(&Y[8 * 16], x16);
    _mm512_storeu_pd(&Y[8 * 17], x17);
    _mm512_storeu_pd(&Y[8 * 18], x18);
    _mm512_storeu_pd(&Y[8 * 19], x19);
    _mm512_storeu_pd(&Y[8 * 20], x20);
    _mm512_storeu_pd(&Y[8 * 21], x21);
    _mm512_storeu_pd(&Y[8 * 22], x22);
    _mm512_storeu_pd(&Y[8 * 23], x23);
    _mm512_storeu_pd(&Y[8 * 24], x24);
    _mm512_storeu_pd(&Y[8 * 25], x25);
    _mm512_storeu_pd(&Y[8 * 26], x26);
    _mm512_storeu_pd(&Y[8 * 27], x27);
    _mm512_storeu_pd(&Y[8 * 28], x28);
    _mm512_storeu_pd(&Y[8 * 29], x29);
    _mm512_storeu_pd(&Y[8 * 30], x30);
    _mm512_storeu_pd(&Y[8 * 31], x31);
}

void avxdcopy_(int *_N, char *_X, int *_INCX, char *_Y, int *_INCY)
{
    int N = *_N;
    double *X = (double *)_X;
    int INCX = *_INCX;
    double *Y = (double *)_Y;
    int INCY = *_INCY;
    int i, j, M, ix, iy, Mp1, MQ;

    register __m512d x0, x1;

    if (INCX == 1 && INCY == 1)
    {

        M = N % 256;
        if (M != 0)
        {

            Mp1 = M % 8;
            if (Mp1 != 0)
            {
                x0 = _mm512_loadu_pd(&X[0]);
                _mm512_storeu_pd(&Y[0], x0);
                if (N < 8)
                    return;
            }
            for (i = Mp1; i < M; i += 16)
            {
                x0 = _mm512_loadu_pd(&X[i]);
                x1 = _mm512_loadu_pd(&X[i + 8]);
                _mm512_storeu_pd(&Y[i], x0);
                _mm512_storeu_pd(&Y[i + 8], x1);
            }
        }
        MQ = (N - M) / 256;

        for (i = 0; i < MQ; i++)
        {
            copy(&X[M + i * 32 * 8], &Y[M + i * 32 * 8]);
        }
    }
    else
    {

        ix = 0;
        iy = 0;
        if (INCX < 0)
            ix = (-N + 1) * INCX + 1;
        if (INCY < 0)
            iy = (-N + 1) * INCY + 1;
        for (i = 0; i < N; i++)
        {
            Y[iy] = X[ix];
            ix = ix + INCX;
            iy = iy + INCY;
        }
    }
    return;
}

