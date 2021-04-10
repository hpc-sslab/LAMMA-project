#include "immintrin.h"
#include "x86intrin.h"
#include "zmmintrin.h"

#define MR 16
#define NR 14
#define MB 4688
#define NB 84
#define KB 501 //500
#define NnB 238
#define KkB 232

#define Mb 1000

#define Nnt 5
#define Nnt1 5
#define Nt 8
#define Nt1 8
#define Nt2 4

#define NT 20
#define NT1 20
#define NT2 2

#define L1_DIST_A 320
#define L1_DIST_B 280

#define min(a, b) (((a) < (b)) ? (a) : (b))
// All micro kernels (micro_kernel[1-16]x[1-14]) for small matrices; This part ends at line 17725

void micro_kernel1x1(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register double _A0;
    register double _C0_0 = C[0 * ldc + 0];

    for(i = 0; i < k; ++i) {
        _A0 = A[0];
        _C0_0 += B[0 * ldb] * _A0;
        A += lda;
        B += 1;
    }

    C[0 * ldc + 0] = _C0_0;
}
void micro_kernel2x1(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m128d _A0;
    register __m128d _C0_0 = _mm_loadu_pd(&C[0 * ldc + 0]);

    for(i = 0; i < k; ++i) {
        _A0 = _mm_loadu_pd(&A[0]);
        _C0_0 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A0, _C0_0);
        A += lda;
        B += 1;
    }

    _mm_storeu_pd(&C[0 * ldc + 0], _C0_0);
}
void micro_kernel3x1(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m128d _A0;
    register double _A1;
    register __m128d _C0_0 = _mm_loadu_pd(&C[0 * ldc + 0]);
    register double _C0_1 = C[0 * ldc + 2];

    for(i = 0; i < k; ++i) {
        _A0 = _mm_loadu_pd(&A[0]);
        _A1 = A[2];
        _C0_0 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 += B[0 * ldb] * _A1;
        A += lda;
        B += 1;
    }

    _mm_storeu_pd(&C[0 * ldc + 0], _C0_0);
    C[0 * ldc + 2] = _C0_1;
}
void micro_kernel4x1(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m256d _A0;
    register __m256d _C0_0 = _mm256_loadu_pd(&C[0 * ldc + 0]);

    for(i = 0; i < k; ++i) {
        _A0 = _mm256_loadu_pd(&A[0]);
        _C0_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A0, _C0_0);
        A += lda;
        B += 1;
    }

    _mm256_storeu_pd(&C[0 * ldc + 0], _C0_0);
}
void micro_kernel5x1(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m256d _A0;
    register double _A1;
    register __m256d _C0_0 = _mm256_loadu_pd(&C[0 * ldc + 0]);
    register double _C0_1 = C[0 * ldc + 4];

    for(i = 0; i < k; ++i) {
        _A0 = _mm256_loadu_pd(&A[0]);
        _A1 = A[4];
        _C0_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 += B[0 * ldb] * _A1;
        A += lda;
        B += 1;
    }

    _mm256_storeu_pd(&C[0 * ldc + 0], _C0_0);
    C[0 * ldc + 4] = _C0_1;
}
void micro_kernel6x1(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m256d _A0;
    register __m128d _A1;
    register __m256d _C0_0 = _mm256_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C0_1 = _mm_loadu_pd(&C[0 * ldc + 4]);

    for(i = 0; i < k; ++i) {
        _A0 = _mm256_loadu_pd(&A[0]);
        _A1 = _mm_loadu_pd(&A[4]);
        _C0_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A1, _C0_1);
        A += lda;
        B += 1;
    }

    _mm256_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[0 * ldc + 4], _C0_1);
}
void micro_kernel7x1(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m256d _A0;
    register __m128d _A1;
    register double _A2;
    register __m256d _C0_0 = _mm256_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C0_1 = _mm_loadu_pd(&C[0 * ldc + 4]);
    register double _C0_2 = C[0 * ldc + 6];

    for(i = 0; i < k; ++i) {
        _A0 = _mm256_loadu_pd(&A[0]);
        _A1 = _mm_loadu_pd(&A[4]);
        _A2 = A[6];
        _C0_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 += B[0 * ldb] * _A2;
        A += lda;
        B += 1;
    }

    _mm256_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[0 * ldc + 4], _C0_1);
    C[0 * ldc + 6] = _C0_2;
}
//================================================================================================================
void micro_kernel8x1(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);

    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);

}
void micro_kernel9x1(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register double _A1;
    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register double _C0_1 = C[0 * ldc + 8];
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = A[8];
        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
         _C0_1 += B[0 * ldb] * _A1;
        A += lda;
        B += 1;
    }
    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    C[0 * ldc + 8] = _C0_1;
}
void micro_kernel10x1(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m128d _A1;
    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C0_1 = _mm_loadu_pd(&C[0 * ldc + 8]);
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm_loadu_pd(&A[8]);
        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A1, _C0_1);
        A += lda;
        B += 1;
    }
    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[0 * ldc + 8], _C0_1);

}
void micro_kernel11x1(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m128d _A1;
    register double _A2;
    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C0_1 = _mm_loadu_pd(&C[0 * ldc + 8]);
    register double _C0_2 = C[0 * ldc + 10];
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm_loadu_pd(&A[8]);
        _A2 = A[10];
        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 += B[0 * ldb] * _A2;
        A += lda;
        B += 1;
    }
    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[0 * ldc + 8], _C0_1);
    C[0 * ldc + 10] = _C0_2;
}
void micro_kernel12x1(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m256d _A1;
    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C0_1 = _mm256_loadu_pd(&C[0 * ldc + 8]);
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm256_loadu_pd(&A[8]);
        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A1, _C0_1);
        A += lda;
        B += 1;
    }
    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[0 * ldc + 8], _C0_1);
}
void micro_kernel13x1(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m256d _A1;
    register double _A2;
    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C0_1 = _mm256_loadu_pd(&C[0 * ldc + 8]);
    register double _C0_2 = C[0 * ldc + 12];
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm256_loadu_pd(&A[8]);
        _A2 = A[12];
        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 += B[0 * ldb] * _A2;
        A += lda;
        B += 1;
    }
    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[0 * ldc + 8], _C0_1);
    C[0 * ldc + 12] = _C0_2;
}
void micro_kernel14x1(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m256d _A1;
    register __m128d _A2;
    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C0_1 = _mm256_loadu_pd(&C[0 * ldc + 8]);
    register __m128d _C0_2 = _mm_loadu_pd(&C[0 * ldc + 12]);
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm256_loadu_pd(&A[8]);
        _A2 = _mm_loadu_pd(&A[12]);
        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A2, _C0_2);
        A += lda;
        B += 1;
    }
    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[0 * ldc + 8], _C0_1);
    _mm_storeu_pd(&C[0 * ldc + 12], _C0_2);
}
void micro_kernel15x1(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m256d _A1;
    register __m128d _A2;
    register double _A3;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C0_1 = _mm256_loadu_pd(&C[0 * ldc + 8]);
    register __m128d _C0_2 = _mm_loadu_pd(&C[0 * ldc + 12]);
    register double _C0_3 = C[0 * ldc + 14];

    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm256_loadu_pd(&A[8]);
        _A2 = _mm_loadu_pd(&A[12]);
        _A3 = A[14];

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A2, _C0_2);
        _C0_3 += B[0 * ldb] * _A3;

        A += lda;
        B += 1;
    }
    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[0 * ldc + 8], _C0_1);
    _mm_storeu_pd(&C[0 * ldc + 12], _C0_2);
     C[0 * ldc + 14] = _C0_3;
}
void micro_kernel16x1(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0, _A1;
    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m512d _C0_1 = _mm512_loadu_pd(&C[0 * ldc + 8]);

    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm512_loadu_pd(&A[8]);
        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A1, _C0_1);
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm512_storeu_pd(&C[0 * ldc + 8], _C0_1);
}

void micro_kernel1x2(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register double _A0;
    register double _C0_0 = C[0 * ldc + 0];
    register double _C1_0 = C[1 * ldc + 0];

    for(i = 0; i < k; ++i) {
        _A0 = A[0];
        _C0_0 += B[0 * ldb] * _A0;
        _C1_0 += B[1 * ldb] * _A0;
        A += lda;
        B += 1;
    }

    C[0 * ldc + 0] = _C0_0;
    C[1 * ldc + 0] = _C1_0;
}
void micro_kernel2x2(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m128d _A0;
    register __m128d _C0_0 = _mm_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C1_0 = _mm_loadu_pd(&C[1 * ldc + 0]);

    for(i = 0; i < k; ++i) {
        _A0 = _mm_loadu_pd(&A[0]);
        _C0_0 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C1_0 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A0, _C1_0);
        A += lda;
        B += 1;
    }

    _mm_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[1 * ldc + 0], _C1_0);
}
void micro_kernel3x2(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m128d _A0;
    register double _A1;
    register __m128d _C0_0 = _mm_loadu_pd(&C[0 * ldc + 0]);
    register double _C0_1 = C[0 * ldc + 2];
    register __m128d _C1_0 = _mm_loadu_pd(&C[1 * ldc + 0]);
    register double _C1_1 = C[1 * ldc + 2];

    for(i = 0; i < k; ++i) {
        _A0 = _mm_loadu_pd(&A[0]);
        _A1 = A[2];
        _C0_0 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 += B[0 * ldb] * _A1;
        _C1_0 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 += B[1 * ldb] * _A1;
        A += lda;
        B += 1;
    }

    _mm_storeu_pd(&C[0 * ldc + 0], _C0_0);
    C[0 * ldc + 2] = _C0_1;
    _mm_storeu_pd(&C[1 * ldc + 0], _C1_0);
    C[1 * ldc + 2] = _C1_1;
}
void micro_kernel4x2(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m256d _A0;
    register __m256d _C0_0 = _mm256_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C1_0 = _mm256_loadu_pd(&C[1 * ldc + 0]);

    for(i = 0; i < k; ++i) {
        _A0 = _mm256_loadu_pd(&A[0]);
        _C0_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C1_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A0, _C1_0);
        A += lda;
        B += 1;
    }

    _mm256_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[1 * ldc + 0], _C1_0);
}
void micro_kernel5x2(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m256d _A0;
    register double _A1;
    register __m256d _C0_0 = _mm256_loadu_pd(&C[0 * ldc + 0]);
    register double _C0_1 = C[0 * ldc + 4];
    register __m256d _C1_0 = _mm256_loadu_pd(&C[1 * ldc + 0]);
    register double _C1_1 = C[1 * ldc + 4];

    for(i = 0; i < k; ++i) {
        _A0 = _mm256_loadu_pd(&A[0]);
        _A1 = A[4];
        _C0_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 += B[0 * ldb] * _A1;
        _C1_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 += B[1 * ldb] * _A1;
        A += lda;
        B += 1;
    }

    _mm256_storeu_pd(&C[0 * ldc + 0], _C0_0);
    C[0 * ldc + 4] = _C0_1;
    _mm256_storeu_pd(&C[1 * ldc + 0], _C1_0);
    C[1 * ldc + 4] = _C1_1;
}
void micro_kernel6x2(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m256d _A0;
    register __m128d _A1;
    register __m256d _C0_0 = _mm256_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C0_1 = _mm_loadu_pd(&C[0 * ldc + 4]);
    register __m256d _C1_0 = _mm256_loadu_pd(&C[1 * ldc + 0]);
    register __m128d _C1_1 = _mm_loadu_pd(&C[1 * ldc + 4]);

    for(i = 0; i < k; ++i) {
        _A0 = _mm256_loadu_pd(&A[0]);
        _A1 = _mm_loadu_pd(&A[4]);
        _C0_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C1_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A1, _C1_1);
        A += lda;
        B += 1;
    }

    _mm256_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[0 * ldc + 4], _C0_1);
    _mm256_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm_storeu_pd(&C[1 * ldc + 4], _C1_1);
}
void micro_kernel7x2(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m256d _A0;
    register __m128d _A1;
    register double _A2;
    register __m256d _C0_0 = _mm256_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C0_1 = _mm_loadu_pd(&C[0 * ldc + 4]);
    register double _C0_2 = C[0 * ldc + 6];
    register __m256d _C1_0 = _mm256_loadu_pd(&C[1 * ldc + 0]);
    register __m128d _C1_1 = _mm_loadu_pd(&C[1 * ldc + 4]);
    register double _C1_2 = C[1 * ldc + 6];

    for(i = 0; i < k; ++i) {
        _A0 = _mm256_loadu_pd(&A[0]);
        _A1 = _mm_loadu_pd(&A[4]);
        _A2 = A[6];
        _C0_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 += B[0 * ldb] * _A2;
        _C1_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C1_2 += B[1 * ldb] * _A2;
        A += lda;
        B += 1;
    }

    _mm256_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[0 * ldc + 4], _C0_1);
    C[0 * ldc + 6] = _C0_2;
    _mm256_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm_storeu_pd(&C[1 * ldc + 4], _C1_1);
    C[1 * ldc + 6] = _C1_2;
}
//======================================================================================================
void micro_kernel8x2(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);

    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
}
void micro_kernel9x2(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register double _A1;
    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register double _C0_1 = C[0 * ldc + 8];
    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);
    register double _C1_1 = C[1 * ldc + 8];

    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = A[8];
        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 += B[0 * ldb] * _A1;
        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 += B[1 * ldb] * _A1;
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    C[0 * ldc + 8] = _C0_1;
    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    C[1 * ldc + 8] = _C1_1;
}
void micro_kernel10x2(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m128d _A1;
    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C0_1 = _mm_loadu_pd(&C[0 * ldc + 8]);
    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);
    register __m128d _C1_1 = _mm_loadu_pd(&C[1 * ldc + 8]);

    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm_loadu_pd(&A[8]);
        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A1, _C1_1);
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[0 * ldc + 8], _C0_1);
    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm_storeu_pd(&C[1 * ldc + 8], _C1_1);
}
void micro_kernel11x2(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m128d _A1;
    register double _A2;
    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C0_1 = _mm_loadu_pd(&C[0 * ldc + 8]);
    register double _C0_2 = C[0 * ldc + 10];
    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);
    register __m128d _C1_1 = _mm_loadu_pd(&C[1 * ldc + 8]);
    register double _C1_2 = C[1 * ldc + 10];

    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm_loadu_pd(&A[8]);
        _A2 = A[10];
        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 += B[0 * ldb] * _A2;
        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C1_2 += B[1 * ldb] * _A2;
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[0 * ldc + 8], _C0_1);
    C[0 * ldc + 10] = _C0_2;
    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm_storeu_pd(&C[1 * ldc + 8], _C1_1);
    C[1 * ldc + 10] = _C1_2;
}
void micro_kernel12x2(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m256d _A1;
    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C0_1 = _mm256_loadu_pd(&C[0 * ldc + 8]);
    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);
    register __m256d _C1_1 = _mm256_loadu_pd(&C[1 * ldc + 8]);

    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm256_loadu_pd(&A[8]);
        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A1, _C1_1);
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[0 * ldc + 8], _C0_1);
    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm256_storeu_pd(&C[1 * ldc + 8], _C1_1);

}
void micro_kernel13x2(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m256d _A1;
    register double _A2;
    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C0_1 = _mm256_loadu_pd(&C[0 * ldc + 8]);
    register double _C0_2 = C[0 * ldc + 12];
    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);
    register __m256d _C1_1 = _mm256_loadu_pd(&C[1 * ldc + 8]);
    register double _C1_2 = C[1 * ldc + 12];

    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm256_loadu_pd(&A[8]);
        _A2 = A[12];
        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 += B[0 * ldb] * _A2;
        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C1_2 += B[1 * ldb] * _A2;
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[0 * ldc + 8], _C0_1);
    C[0 * ldc + 12] = _C0_2;
    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm256_storeu_pd(&C[1 * ldc + 8], _C1_1);
    C[1 * ldc + 12] = _C1_2;
}
void micro_kernel14x2(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
     int i;
    register __m512d _A0;
    register __m256d _A1;
    register __m128d _A2;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C0_1 = _mm256_loadu_pd(&C[0 * ldc + 8]);
    register __m128d _C0_2 = _mm_loadu_pd(&C[0 * ldc + 12]);

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);
    register __m256d _C1_1 = _mm256_loadu_pd(&C[1 * ldc + 8]);
    register __m128d _C1_2 = _mm_loadu_pd(&C[1 * ldc + 12]);


    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm256_loadu_pd(&A[8]);
        _A2 = _mm_loadu_pd(&A[12]);

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A2, _C0_2);

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C1_2 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A2, _C1_2);

        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[0 * ldc + 8], _C0_1);
    _mm_storeu_pd(&C[0 * ldc + 12], _C0_2);

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm256_storeu_pd(&C[1 * ldc + 8], _C1_1);
    _mm_storeu_pd(&C[1 * ldc + 12], _C1_2);

}
void micro_kernel15x2(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
int i;
    register __m512d _A0;
    register __m256d _A1;
    register __m128d _A2;
    register double _A3;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C0_1 = _mm256_loadu_pd(&C[0 * ldc + 8]);
    register __m128d _C0_2 = _mm_loadu_pd(&C[0 * ldc + 12]);
    register double _C0_3 = C[0 * ldc + 14];

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);
    register __m256d _C1_1 = _mm256_loadu_pd(&C[1 * ldc + 8]);
    register __m128d _C1_2 = _mm_loadu_pd(&C[1 * ldc + 12]);
    register double _C1_3 = C[1 * ldc + 14];


    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm256_loadu_pd(&A[8]);
        _A2 = _mm_loadu_pd(&A[12]);
        _A3 = A[14];

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A2, _C0_2);
        _C0_3 += B[0 * ldb] * _A3;

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C1_2 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A2, _C1_2);
        _C1_3 += B[1 * ldb] * _A3;

        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[0 * ldc + 8], _C0_1);
    _mm_storeu_pd(&C[0 * ldc + 12], _C0_2);
    C[0 * ldc + 14] = _C0_3;

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm256_storeu_pd(&C[1 * ldc + 8], _C1_1);
    _mm_storeu_pd(&C[1 * ldc + 12], _C1_2);
    C[1 * ldc + 14] = _C1_3;
}
void micro_kernel16x2(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m512d _A1;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m512d _C0_1 = _mm512_loadu_pd(&C[0 * ldc + 8]);
    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);
    register __m512d _C1_1 = _mm512_loadu_pd(&C[1 * ldc + 8]);

    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm512_loadu_pd(&A[8]);

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A1, _C1_1);

        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm512_storeu_pd(&C[0 * ldc + 8], _C0_1);
    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm512_storeu_pd(&C[1 * ldc + 8], _C1_1);

}
void micro_kernel1x3(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register double _A0;
    register double _C0_0 = C[0 * ldc + 0];
    register double _C1_0 = C[1 * ldc + 0];
    register double _C2_0 = C[2 * ldc + 0];

    for(i = 0; i < k; ++i) {
        _A0 = A[0];
        _C0_0 += B[0 * ldb] * _A0;
        _C1_0 += B[1 * ldb] * _A0;
        _C2_0 += B[2 * ldb] * _A0;
        A += lda;
        B += 1;
    }

    C[0 * ldc + 0] = _C0_0;
    C[1 * ldc + 0] = _C1_0;
    C[2 * ldc + 0] = _C2_0;
}
void micro_kernel2x3(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m128d _A0;
    register __m128d _C0_0 = _mm_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C1_0 = _mm_loadu_pd(&C[1 * ldc + 0]);
    register __m128d _C2_0 = _mm_loadu_pd(&C[2 * ldc + 0]);

    for(i = 0; i < k; ++i) {
        _A0 = _mm_loadu_pd(&A[0]);
        _C0_0 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C1_0 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C2_0 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A0, _C2_0);
        A += lda;
        B += 1;
    }

    _mm_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm_storeu_pd(&C[2 * ldc + 0], _C2_0);
}
void micro_kernel3x3(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m128d _A0;
    register double _A1;
    register __m128d _C0_0 = _mm_loadu_pd(&C[0 * ldc + 0]);
    register double _C0_1 = C[0 * ldc + 2];
    register __m128d _C1_0 = _mm_loadu_pd(&C[1 * ldc + 0]);
    register double _C1_1 = C[1 * ldc + 2];
    register __m128d _C2_0 = _mm_loadu_pd(&C[2 * ldc + 0]);
    register double _C2_1 = C[2 * ldc + 2];

    for(i = 0; i < k; ++i) {
        _A0 = _mm_loadu_pd(&A[0]);
        _A1 = A[2];
        _C0_0 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 += B[0 * ldb] * _A1;
        _C1_0 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 += B[1 * ldb] * _A1;
        _C2_0 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 += B[2 * ldb] * _A1;
        A += lda;
        B += 1;
    }

    _mm_storeu_pd(&C[0 * ldc + 0], _C0_0);
    C[0 * ldc + 2] = _C0_1;
    _mm_storeu_pd(&C[1 * ldc + 0], _C1_0);
    C[1 * ldc + 2] = _C1_1;
    _mm_storeu_pd(&C[2 * ldc + 0], _C2_0);
    C[2 * ldc + 2] = _C2_1;
}
void micro_kernel4x3(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m256d _A0;
    register __m256d _C0_0 = _mm256_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C1_0 = _mm256_loadu_pd(&C[1 * ldc + 0]);
    register __m256d _C2_0 = _mm256_loadu_pd(&C[2 * ldc + 0]);

    for(i = 0; i < k; ++i) {
        _A0 = _mm256_loadu_pd(&A[0]);
        _C0_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C1_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C2_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A0, _C2_0);
        A += lda;
        B += 1;
    }

    _mm256_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm256_storeu_pd(&C[2 * ldc + 0], _C2_0);
}
void micro_kernel5x3(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m256d _A0;
    register double _A1;
    register __m256d _C0_0 = _mm256_loadu_pd(&C[0 * ldc + 0]);
    register double _C0_1 = C[0 * ldc + 4];
    register __m256d _C1_0 = _mm256_loadu_pd(&C[1 * ldc + 0]);
    register double _C1_1 = C[1 * ldc + 4];
    register __m256d _C2_0 = _mm256_loadu_pd(&C[2 * ldc + 0]);
    register double _C2_1 = C[2 * ldc + 4];

    for(i = 0; i < k; ++i) {
        _A0 = _mm256_loadu_pd(&A[0]);
        _A1 = A[4];
        _C0_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 += B[0 * ldb] * _A1;
        _C1_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 += B[1 * ldb] * _A1;
        _C2_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 += B[2 * ldb] * _A1;
        A += lda;
        B += 1;
    }

    _mm256_storeu_pd(&C[0 * ldc + 0], _C0_0);
    C[0 * ldc + 4] = _C0_1;
    _mm256_storeu_pd(&C[1 * ldc + 0], _C1_0);
    C[1 * ldc + 4] = _C1_1;
    _mm256_storeu_pd(&C[2 * ldc + 0], _C2_0);
    C[2 * ldc + 4] = _C2_1;
}
void micro_kernel6x3(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m256d _A0;
    register __m128d _A1;
    register __m256d _C0_0 = _mm256_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C0_1 = _mm_loadu_pd(&C[0 * ldc + 4]);
    register __m256d _C1_0 = _mm256_loadu_pd(&C[1 * ldc + 0]);
    register __m128d _C1_1 = _mm_loadu_pd(&C[1 * ldc + 4]);
    register __m256d _C2_0 = _mm256_loadu_pd(&C[2 * ldc + 0]);
    register __m128d _C2_1 = _mm_loadu_pd(&C[2 * ldc + 4]);

    for(i = 0; i < k; ++i) {
        _A0 = _mm256_loadu_pd(&A[0]);
        _A1 = _mm_loadu_pd(&A[4]);
        _C0_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C1_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C2_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A1, _C2_1);
        A += lda;
        B += 1;
    }

    _mm256_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[0 * ldc + 4], _C0_1);
    _mm256_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm_storeu_pd(&C[1 * ldc + 4], _C1_1);
    _mm256_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm_storeu_pd(&C[2 * ldc + 4], _C2_1);
}
void micro_kernel7x3(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m256d _A0;
    register __m128d _A1;
    register double _A2;
    register __m256d _C0_0 = _mm256_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C0_1 = _mm_loadu_pd(&C[0 * ldc + 4]);
    register double _C0_2 = C[0 * ldc + 6];
    register __m256d _C1_0 = _mm256_loadu_pd(&C[1 * ldc + 0]);
    register __m128d _C1_1 = _mm_loadu_pd(&C[1 * ldc + 4]);
    register double _C1_2 = C[1 * ldc + 6];
    register __m256d _C2_0 = _mm256_loadu_pd(&C[2 * ldc + 0]);
    register __m128d _C2_1 = _mm_loadu_pd(&C[2 * ldc + 4]);
    register double _C2_2 = C[2 * ldc + 6];

    for(i = 0; i < k; ++i) {
        _A0 = _mm256_loadu_pd(&A[0]);
        _A1 = _mm_loadu_pd(&A[4]);
        _A2 = A[6];
        _C0_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 += B[0 * ldb] * _A2;
        _C1_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C1_2 += B[1 * ldb] * _A2;
        _C2_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C2_2 += B[2 * ldb] * _A2;
        A += lda;
        B += 1;
    }

    _mm256_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[0 * ldc + 4], _C0_1);
    C[0 * ldc + 6] = _C0_2;
    _mm256_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm_storeu_pd(&C[1 * ldc + 4], _C1_1);
    C[1 * ldc + 6] = _C1_2;
    _mm256_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm_storeu_pd(&C[2 * ldc + 4], _C2_1);
    C[2 * ldc + 6] = _C2_2;
}
//==================================================================================================================================
void micro_kernel8x3(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);
    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);

    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
}
void micro_kernel9x3(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register double _A1;
    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register double _C0_1 = C[0 * ldc + 8];
    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);
    register double _C1_1 = C[1 * ldc + 8];
    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register double _C2_1 = C[2 * ldc + 8];

    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = A[8];
        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 += B[0 * ldb] * _A1;
        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 += B[1 * ldb] * _A1;
        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 += B[2 * ldb] * _A1;
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    C[0 * ldc + 8] = _C0_1;
    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    C[1 * ldc + 8] = _C1_1;
    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    C[2 * ldc + 8] = _C2_1;
}
void micro_kernel10x3(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m128d _A1;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C0_1 = _mm_loadu_pd(&C[0 * ldc + 8]);

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);
    register __m128d _C1_1 = _mm_loadu_pd(&C[1 * ldc + 8]);

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m128d _C2_1 = _mm_loadu_pd(&C[2 * ldc + 8]);

    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm_loadu_pd(&A[8]);

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A1, _C0_1);

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A1, _C1_1);

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A1, _C2_1);

        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[0 * ldc + 8], _C0_1);

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm_storeu_pd(&C[1 * ldc + 8], _C1_1);

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm_storeu_pd(&C[2 * ldc + 8], _C2_1);
}
void micro_kernel11x3(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m128d _A1;
    register double _A2;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C0_1 = _mm_loadu_pd(&C[0 * ldc + 8]);
    register double _C0_2 = C[0 * ldc + 10];

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);
    register __m128d _C1_1 = _mm_loadu_pd(&C[1 * ldc + 8]);
    register double _C1_2 = C[1 * ldc + 10];

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m128d _C2_1 = _mm_loadu_pd(&C[2 * ldc + 8]);
    register double _C2_2 = C[2 * ldc + 10];

    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm_loadu_pd(&A[8]);
        _A2 = A[10];
        
        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 += B[0 * ldb] * _A2;

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C1_2 += B[1 * ldb] * _A2;

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C2_2 += B[2 * ldb] * _A2;

        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[0 * ldc + 8], _C0_1);
    C[0 * ldc + 10] = _C0_2;

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm_storeu_pd(&C[1 * ldc + 8], _C1_1);
    C[1 * ldc + 10] = _C1_2;

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm_storeu_pd(&C[2 * ldc + 8], _C2_1);
    C[2 * ldc + 10] = _C2_2;
}
void micro_kernel12x3(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m256d _A1;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C0_1 = _mm256_loadu_pd(&C[0 * ldc + 8]);

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);
    register __m256d _C1_1 = _mm256_loadu_pd(&C[1 * ldc + 8]);

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m256d _C2_1 = _mm256_loadu_pd(&C[2 * ldc + 8]);

    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm256_loadu_pd(&A[8]);

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A1, _C0_1);

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A1, _C1_1);

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A1, _C2_1);

        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[0 * ldc + 8], _C0_1);

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm256_storeu_pd(&C[1 * ldc + 8], _C1_1);

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm256_storeu_pd(&C[2 * ldc + 8], _C2_1);
}
void micro_kernel13x3(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m256d _A1;
    register double _A2;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C0_1 = _mm256_loadu_pd(&C[0 * ldc + 8]);
    register double _C0_2 = C[0 * ldc + 12];

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);
    register __m256d _C1_1 = _mm256_loadu_pd(&C[1 * ldc + 8]);
    register double _C1_2 = C[1 * ldc + 12];

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m256d _C2_1 = _mm256_loadu_pd(&C[2 * ldc + 8]);
    register double _C2_2 = C[2 * ldc + 12];

    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm256_loadu_pd(&A[8]);
        _A2 = A[12];

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 += B[0 * ldb] * _A2;

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C1_2 += B[1 * ldb] * _A2;

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C2_2 += B[2 * ldb] * _A2;

        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[0 * ldc + 8], _C0_1);
    C[0 * ldc + 12] = _C0_2;

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm256_storeu_pd(&C[1 * ldc + 8], _C1_1);
    C[1 * ldc + 12] = _C1_2;

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm256_storeu_pd(&C[2 * ldc + 8], _C2_1);
    C[2 * ldc + 12] = _C2_2;
}
void micro_kernel14x3(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m256d _A1;
    register __m128d _A2;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C0_1 = _mm256_loadu_pd(&C[0 * ldc + 8]);
    register __m128d _C0_2 = _mm_loadu_pd(&C[0 * ldc + 12]);

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);
    register __m256d _C1_1 = _mm256_loadu_pd(&C[1 * ldc + 8]);
    register __m128d _C1_2 = _mm_loadu_pd(&C[1 * ldc + 12]);

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m256d _C2_1 = _mm256_loadu_pd(&C[2 * ldc + 8]);
    register __m128d _C2_2 = _mm_loadu_pd(&C[2 * ldc + 12]);

    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm256_loadu_pd(&A[8]);
        _A2 = _mm_loadu_pd(&A[12]);

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A2, _C0_2);

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C1_2 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A2, _C1_2);

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C2_2 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A2, _C2_2);

        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[0 * ldc + 8], _C0_1);
    _mm_storeu_pd(&C[0 * ldc + 12], _C0_2);

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm256_storeu_pd(&C[1 * ldc + 8], _C1_1);
    _mm_storeu_pd(&C[1 * ldc + 12], _C1_2);

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm256_storeu_pd(&C[2 * ldc + 8], _C2_1);
    _mm_storeu_pd(&C[2 * ldc + 12], _C2_2);
}
void micro_kernel15x3(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m256d _A1;
    register __m128d _A2;
    register double _A3;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C0_1 = _mm256_loadu_pd(&C[0 * ldc + 8]);
    register __m128d _C0_2 = _mm_loadu_pd(&C[0 * ldc + 12]);
    register double _C0_3 = C[0 * ldc + 14];

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);
    register __m256d _C1_1 = _mm256_loadu_pd(&C[1 * ldc + 8]);
    register __m128d _C1_2 = _mm_loadu_pd(&C[1 * ldc + 12]);
    register double _C1_3 = C[1 * ldc + 14];

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m256d _C2_1 = _mm256_loadu_pd(&C[2 * ldc + 8]);
    register __m128d _C2_2 = _mm_loadu_pd(&C[2 * ldc + 12]);
    register double _C2_3 = C[2 * ldc + 14];

    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm256_loadu_pd(&A[8]);
        _A2 = _mm_loadu_pd(&A[12]);
        _A3 = A[14];

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A2, _C0_2);
        _C0_3 += B[0 * ldb] * _A3;

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C1_2 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A2, _C1_2);
        _C1_3 += B[1 * ldb] * _A3;

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C2_2 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A2, _C2_2);
        _C2_3 += B[2 * ldb] * _A3;

        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[0 * ldc + 8], _C0_1);
    _mm_storeu_pd(&C[0 * ldc + 12], _C0_2);
    C[0 * ldc + 14] = _C0_3;

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm256_storeu_pd(&C[1 * ldc + 8], _C1_1);
    _mm_storeu_pd(&C[1 * ldc + 12], _C1_2);
    C[1 * ldc + 14] = _C1_3;

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm256_storeu_pd(&C[2 * ldc + 8], _C2_1);
    _mm_storeu_pd(&C[2 * ldc + 12], _C2_2);
    C[2 * ldc + 14] = _C2_3;
}
void micro_kernel16x3(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m512d _A1;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m512d _C0_1 = _mm512_loadu_pd(&C[0 * ldc + 8]);

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);
    register __m512d _C1_1 = _mm512_loadu_pd(&C[1 * ldc + 8]);

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m512d _C2_1 = _mm512_loadu_pd(&C[2 * ldc + 8]);

    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm512_loadu_pd(&A[8]);

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A1, _C0_1);

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A1, _C1_1);

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A1, _C2_1);

        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm512_storeu_pd(&C[0 * ldc + 8], _C0_1);

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm512_storeu_pd(&C[1 * ldc + 8], _C1_1);

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm512_storeu_pd(&C[2 * ldc + 8], _C2_1);
}
void micro_kernel1x4(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register double _A0;
    register double _C0_0 = C[0 * ldc + 0];
    register double _C1_0 = C[1 * ldc + 0];
    register double _C2_0 = C[2 * ldc + 0];
    register double _C3_0 = C[3 * ldc + 0];

    for(i = 0; i < k; ++i) {
        _A0 = A[0];
        _C0_0 += B[0 * ldb] * _A0;
        _C1_0 += B[1 * ldb] * _A0;
        _C2_0 += B[2 * ldb] * _A0;
        _C3_0 += B[3 * ldb] * _A0;
        A += lda;
        B += 1;
    }

    C[0 * ldc + 0] = _C0_0;
    C[1 * ldc + 0] = _C1_0;
    C[2 * ldc + 0] = _C2_0;
    C[3 * ldc + 0] = _C3_0;
}
void micro_kernel2x4(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m128d _A0;
    register __m128d _C0_0 = _mm_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C1_0 = _mm_loadu_pd(&C[1 * ldc + 0]);
    register __m128d _C2_0 = _mm_loadu_pd(&C[2 * ldc + 0]);
    register __m128d _C3_0 = _mm_loadu_pd(&C[3 * ldc + 0]);

    for(i = 0; i < k; ++i) {
        _A0 = _mm_loadu_pd(&A[0]);
        _C0_0 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C1_0 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C2_0 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C3_0 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A0, _C3_0);
        A += lda;
        B += 1;
    }

    _mm_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm_storeu_pd(&C[3 * ldc + 0], _C3_0);
}
void micro_kernel3x4(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m128d _A0;
    register double _A1;
    register __m128d _C0_0 = _mm_loadu_pd(&C[0 * ldc + 0]);
    register double _C0_1 = C[0 * ldc + 2];
    register __m128d _C1_0 = _mm_loadu_pd(&C[1 * ldc + 0]);
    register double _C1_1 = C[1 * ldc + 2];
    register __m128d _C2_0 = _mm_loadu_pd(&C[2 * ldc + 0]);
    register double _C2_1 = C[2 * ldc + 2];
    register __m128d _C3_0 = _mm_loadu_pd(&C[3 * ldc + 0]);
    register double _C3_1 = C[3 * ldc + 2];

    for(i = 0; i < k; ++i) {
        _A0 = _mm_loadu_pd(&A[0]);
        _A1 = A[2];
        _C0_0 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 += B[0 * ldb] * _A1;
        _C1_0 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 += B[1 * ldb] * _A1;
        _C2_0 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 += B[2 * ldb] * _A1;
        _C3_0 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 += B[3 * ldb] * _A1;
        A += lda;
        B += 1;
    }

    _mm_storeu_pd(&C[0 * ldc + 0], _C0_0);
    C[0 * ldc + 2] = _C0_1;
    _mm_storeu_pd(&C[1 * ldc + 0], _C1_0);
    C[1 * ldc + 2] = _C1_1;
    _mm_storeu_pd(&C[2 * ldc + 0], _C2_0);
    C[2 * ldc + 2] = _C2_1;
    _mm_storeu_pd(&C[3 * ldc + 0], _C3_0);
    C[3 * ldc + 2] = _C3_1;
}
void micro_kernel4x4(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m256d _A0;
    register __m256d _C0_0 = _mm256_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C1_0 = _mm256_loadu_pd(&C[1 * ldc + 0]);
    register __m256d _C2_0 = _mm256_loadu_pd(&C[2 * ldc + 0]);
    register __m256d _C3_0 = _mm256_loadu_pd(&C[3 * ldc + 0]);

    for(i = 0; i < k; ++i) {
        _A0 = _mm256_loadu_pd(&A[0]);
        _C0_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C1_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C2_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C3_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A0, _C3_0);
        A += lda;
        B += 1;
    }

    _mm256_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm256_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm256_storeu_pd(&C[3 * ldc + 0], _C3_0);
}
void micro_kernel5x4(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m256d _A0;
    register double _A1;
    register __m256d _C0_0 = _mm256_loadu_pd(&C[0 * ldc + 0]);
    register double _C0_1 = C[0 * ldc + 4];
    register __m256d _C1_0 = _mm256_loadu_pd(&C[1 * ldc + 0]);
    register double _C1_1 = C[1 * ldc + 4];
    register __m256d _C2_0 = _mm256_loadu_pd(&C[2 * ldc + 0]);
    register double _C2_1 = C[2 * ldc + 4];
    register __m256d _C3_0 = _mm256_loadu_pd(&C[3 * ldc + 0]);
    register double _C3_1 = C[3 * ldc + 4];

    for(i = 0; i < k; ++i) {
        _A0 = _mm256_loadu_pd(&A[0]);
        _A1 = A[4];
        _C0_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 += B[0 * ldb] * _A1;
        _C1_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 += B[1 * ldb] * _A1;
        _C2_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 += B[2 * ldb] * _A1;
        _C3_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 += B[3 * ldb] * _A1;
        A += lda;
        B += 1;
    }

    _mm256_storeu_pd(&C[0 * ldc + 0], _C0_0);
    C[0 * ldc + 4] = _C0_1;
    _mm256_storeu_pd(&C[1 * ldc + 0], _C1_0);
    C[1 * ldc + 4] = _C1_1;
    _mm256_storeu_pd(&C[2 * ldc + 0], _C2_0);
    C[2 * ldc + 4] = _C2_1;
    _mm256_storeu_pd(&C[3 * ldc + 0], _C3_0);
    C[3 * ldc + 4] = _C3_1;
}
void micro_kernel6x4(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m256d _A0;
    register __m128d _A1;
    register __m256d _C0_0 = _mm256_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C0_1 = _mm_loadu_pd(&C[0 * ldc + 4]);
    register __m256d _C1_0 = _mm256_loadu_pd(&C[1 * ldc + 0]);
    register __m128d _C1_1 = _mm_loadu_pd(&C[1 * ldc + 4]);
    register __m256d _C2_0 = _mm256_loadu_pd(&C[2 * ldc + 0]);
    register __m128d _C2_1 = _mm_loadu_pd(&C[2 * ldc + 4]);
    register __m256d _C3_0 = _mm256_loadu_pd(&C[3 * ldc + 0]);
    register __m128d _C3_1 = _mm_loadu_pd(&C[3 * ldc + 4]);

    for(i = 0; i < k; ++i) {
        _A0 = _mm256_loadu_pd(&A[0]);
        _A1 = _mm_loadu_pd(&A[4]);
        _C0_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C1_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C2_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C3_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A1, _C3_1);
        A += lda;
        B += 1;
    }

    _mm256_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[0 * ldc + 4], _C0_1);
    _mm256_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm_storeu_pd(&C[1 * ldc + 4], _C1_1);
    _mm256_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm_storeu_pd(&C[2 * ldc + 4], _C2_1);
    _mm256_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm_storeu_pd(&C[3 * ldc + 4], _C3_1);
}
void micro_kernel7x4(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m256d _A0;
    register __m128d _A1;
    register double _A2;
    register __m256d _C0_0 = _mm256_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C0_1 = _mm_loadu_pd(&C[0 * ldc + 4]);
    register double _C0_2 = C[0 * ldc + 6];
    register __m256d _C1_0 = _mm256_loadu_pd(&C[1 * ldc + 0]);
    register __m128d _C1_1 = _mm_loadu_pd(&C[1 * ldc + 4]);
    register double _C1_2 = C[1 * ldc + 6];
    register __m256d _C2_0 = _mm256_loadu_pd(&C[2 * ldc + 0]);
    register __m128d _C2_1 = _mm_loadu_pd(&C[2 * ldc + 4]);
    register double _C2_2 = C[2 * ldc + 6];
    register __m256d _C3_0 = _mm256_loadu_pd(&C[3 * ldc + 0]);
    register __m128d _C3_1 = _mm_loadu_pd(&C[3 * ldc + 4]);
    register double _C3_2 = C[3 * ldc + 6];

    for(i = 0; i < k; ++i) {
        _A0 = _mm256_loadu_pd(&A[0]);
        _A1 = _mm_loadu_pd(&A[4]);
        _A2 = A[6];
        _C0_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 += B[0 * ldb] * _A2;
        _C1_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C1_2 += B[1 * ldb] * _A2;
        _C2_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C2_2 += B[2 * ldb] * _A2;
        _C3_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A1, _C3_1);
        _C3_2 += B[3 * ldb] * _A2;
        A += lda;
        B += 1;
    }

    _mm256_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[0 * ldc + 4], _C0_1);
    C[0 * ldc + 6] = _C0_2;
    _mm256_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm_storeu_pd(&C[1 * ldc + 4], _C1_1);
    C[1 * ldc + 6] = _C1_2;
    _mm256_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm_storeu_pd(&C[2 * ldc + 4], _C2_1);
    C[2 * ldc + 6] = _C2_2;
    _mm256_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm_storeu_pd(&C[3 * ldc + 4], _C3_1);
    C[3 * ldc + 6] = _C3_2;
}
//=============================================================================================================================
void micro_kernel8x4(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);

}
void micro_kernel9x4(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register double _A1;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register double _C0_1 = C[0 * ldc + 8];

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register double _C1_1 = C[1 * ldc + 8];

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register double _C2_1 = C[2 * ldc + 8];

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register double _C3_1 = C[3 * ldc + 8];
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = A[8];

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 += B[0 * ldb] * _A1;

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 += B[1 * ldb] * _A1;

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 += B[2 * ldb] * _A1;

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 += B[3 * ldb] * _A1;
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    C[0 * ldc + 8] = _C0_1;

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    C[1 * ldc + 8] = _C1_1;

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    C[2 * ldc + 8] = _C2_1;

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    C[3 * ldc + 8] = _C3_1;
}
void micro_kernel10x4(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m128d _A1;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C0_1 = _mm_loadu_pd(&C[0 * ldc + 8]);

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m128d _C1_1 = _mm_loadu_pd(&C[1 * ldc + 8]);

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m128d _C2_1 = _mm_loadu_pd(&C[2 * ldc + 8]);

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m128d _C3_1 = _mm_loadu_pd(&C[3 * ldc + 8]);
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm_loadu_pd(&A[8]);

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A1, _C0_1);

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A1, _C1_1);

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A1, _C2_1);

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A1, _C3_1);
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[0 * ldc + 8], _C0_1);

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm_storeu_pd(&C[1 * ldc + 8], _C1_1);

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm_storeu_pd(&C[2 * ldc + 8], _C2_1);

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm_storeu_pd(&C[3 * ldc + 8], _C3_1);
}
void micro_kernel11x4(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m128d _A1;
    register double _A2;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C0_1 = _mm_loadu_pd(&C[0 * ldc + 8]);
    register double _C0_2 = C[0 * ldc + 10];

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m128d _C1_1 = _mm_loadu_pd(&C[1 * ldc + 8]);
    register double _C1_2 = C[1 * ldc + 10];

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m128d _C2_1 = _mm_loadu_pd(&C[2 * ldc + 8]);
    register double _C2_2 = C[2 * ldc + 10];

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m128d _C3_1 = _mm_loadu_pd(&C[3 * ldc + 8]);
    register double _C3_2 = C[3 * ldc + 10];
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm_loadu_pd(&A[8]);
        _A2 = A[10];

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 += B[0 * ldb] * _A2;

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C1_2 += B[1 * ldb] * _A2;

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C2_2 += B[2 * ldb] * _A2;

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A1, _C3_1);
        _C3_2 += B[3 * ldb] * _A2;
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[0 * ldc + 8], _C0_1);
    C[0 * ldc + 10] = _C0_2;

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm_storeu_pd(&C[1 * ldc + 8], _C1_1);
    C[1 * ldc + 10] = _C1_2;

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm_storeu_pd(&C[2 * ldc + 8], _C2_1);
    C[2 * ldc + 10] = _C2_2;

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm_storeu_pd(&C[3 * ldc + 8], _C3_1);
    C[3 * ldc + 10] = _C3_2;
}
void micro_kernel12x4(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m256d _A1;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C0_1 = _mm256_loadu_pd(&C[0 * ldc + 8]);

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m256d _C1_1 = _mm256_loadu_pd(&C[1 * ldc + 8]);

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m256d _C2_1 = _mm256_loadu_pd(&C[2 * ldc + 8]);

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m256d _C3_1 = _mm256_loadu_pd(&C[3 * ldc + 8]);
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm256_loadu_pd(&A[8]);

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A1, _C0_1);

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A1, _C1_1);

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A1, _C2_1);

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A1, _C3_1);
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[0 * ldc + 8], _C0_1);

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm256_storeu_pd(&C[1 * ldc + 8], _C1_1);

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm256_storeu_pd(&C[2 * ldc + 8], _C2_1);

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm256_storeu_pd(&C[3 * ldc + 8], _C3_1);
}
void micro_kernel13x4(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m256d _A1;
    register double _A2;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C0_1 = _mm256_loadu_pd(&C[0 * ldc + 8]);
    register double _C0_2 = C[0 * ldc + 12];

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m256d _C1_1 = _mm256_loadu_pd(&C[1 * ldc + 8]);
    register double _C1_2 = C[1 * ldc + 12];

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m256d _C2_1 = _mm256_loadu_pd(&C[2 * ldc + 8]);
    register double _C2_2 = C[2 * ldc + 12];

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m256d _C3_1 = _mm256_loadu_pd(&C[3 * ldc + 8]);
    register double _C3_2 = C[3 * ldc + 12];
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm256_loadu_pd(&A[8]);
        _A2 = A[12];

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 += B[0 * ldb] * _A2;

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C1_2 += B[1 * ldb] * _A2;

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C2_2 += B[2 * ldb] * _A2;

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A1, _C3_1);
        _C3_2 += B[3 * ldb] * _A2;
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[0 * ldc + 8], _C0_1);
    C[0 * ldc + 12] = _C0_2;

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm256_storeu_pd(&C[1 * ldc + 8], _C1_1);
    C[1 * ldc + 12] = _C1_2;

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm256_storeu_pd(&C[2 * ldc + 8], _C2_1);
    C[2 * ldc + 12] = _C2_2;

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm256_storeu_pd(&C[3 * ldc + 8], _C3_1);
    C[3 * ldc + 12] = _C3_2;
}
void micro_kernel14x4(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m256d _A1;
    register __m128d _A2;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C0_1 = _mm256_loadu_pd(&C[0 * ldc + 8]);
    register __m128d _C0_2 = _mm_loadu_pd(&C[0 * ldc + 12]);

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m256d _C1_1 = _mm256_loadu_pd(&C[1 * ldc + 8]);
    register __m128d _C1_2 = _mm_loadu_pd(&C[1 * ldc + 12]);

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m256d _C2_1 = _mm256_loadu_pd(&C[2 * ldc + 8]);
    register __m128d _C2_2 = _mm_loadu_pd(&C[2 * ldc + 12]);

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m256d _C3_1 = _mm256_loadu_pd(&C[3 * ldc + 8]);
    register __m128d _C3_2 = _mm_loadu_pd(&C[3 * ldc + 12]);
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm256_loadu_pd(&A[8]);
        _A2 = _mm_loadu_pd(&A[12]);

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A2, _C0_2);

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C1_2 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A2, _C1_2);

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C2_2 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A2, _C2_2);

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A1, _C3_1);
        _C3_2 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A2, _C3_2);
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[0 * ldc + 8], _C0_1);
    _mm_storeu_pd(&C[0 * ldc + 12], _C0_2);

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm256_storeu_pd(&C[1 * ldc + 8], _C1_1);
    _mm_storeu_pd(&C[1 * ldc + 12], _C1_2);

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm256_storeu_pd(&C[2 * ldc + 8], _C2_1);
    _mm_storeu_pd(&C[2 * ldc + 12], _C2_2);

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm256_storeu_pd(&C[3 * ldc + 8], _C3_1);
    _mm_storeu_pd(&C[3 * ldc + 12], _C3_2);
}
void micro_kernel15x4(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m256d _A1;
    register __m128d _A2;
    register double _A3;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C0_1 = _mm256_loadu_pd(&C[0 * ldc + 8]);
    register __m128d _C0_2 = _mm_loadu_pd(&C[0 * ldc + 12]);
    register double _C0_3 = C[0 * ldc + 14];

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m256d _C1_1 = _mm256_loadu_pd(&C[1 * ldc + 8]);
    register __m128d _C1_2 = _mm_loadu_pd(&C[1 * ldc + 12]);
    register double _C1_3 = C[1 * ldc + 14];

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m256d _C2_1 = _mm256_loadu_pd(&C[2 * ldc + 8]);
    register __m128d _C2_2 = _mm_loadu_pd(&C[2 * ldc + 12]);
    register double _C2_3 = C[2 * ldc + 14];

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m256d _C3_1 = _mm256_loadu_pd(&C[3 * ldc + 8]);
    register __m128d _C3_2 = _mm_loadu_pd(&C[3 * ldc + 12]);
    register double _C3_3 = C[3 * ldc + 14];
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm256_loadu_pd(&A[8]);
        _A2 = _mm_loadu_pd(&A[12]);
        _A3 = A[14];

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A2, _C0_2);
        _C0_3 += B[0 * ldb] * _A3;

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C1_2 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A2, _C1_2);
        _C1_3 += B[1 * ldb] * _A3;

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C2_2 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A2, _C2_2);
        _C2_3 += B[2 * ldb] * _A3;

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A1, _C3_1);
        _C3_2 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A2, _C3_2);
        _C3_3 += B[3 * ldb] * _A3;
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[0 * ldc + 8], _C0_1);
    _mm_storeu_pd(&C[0 * ldc + 12], _C0_2);
    C[0 * ldc + 14] = _C0_3;

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm256_storeu_pd(&C[1 * ldc + 8], _C1_1);
    _mm_storeu_pd(&C[1 * ldc + 12], _C1_2);
    C[1 * ldc + 14] = _C1_3;

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm256_storeu_pd(&C[2 * ldc + 8], _C2_1);
    _mm_storeu_pd(&C[2 * ldc + 12], _C2_2);
    C[2 * ldc + 14] = _C2_3;

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm256_storeu_pd(&C[3 * ldc + 8], _C3_1);
    _mm_storeu_pd(&C[3 * ldc + 12], _C3_2);
    C[3 * ldc + 14] = _C3_3;
}
void micro_kernel16x4(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m512d _A1;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m512d _C0_1 = _mm512_loadu_pd(&C[0 * ldc + 8]);

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m512d _C1_1 = _mm512_loadu_pd(&C[1 * ldc + 8]);

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m512d _C2_1 = _mm512_loadu_pd(&C[2 * ldc + 8]);

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m512d _C3_1 = _mm512_loadu_pd(&C[3 * ldc + 8]);
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm512_loadu_pd(&A[8]);

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A1, _C0_1);

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A1, _C1_1);

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A1, _C2_1);

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A1, _C3_1);
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm512_storeu_pd(&C[0 * ldc + 8], _C0_1);

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm512_storeu_pd(&C[1 * ldc + 8], _C1_1);

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm512_storeu_pd(&C[2 * ldc + 8], _C2_1);

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm512_storeu_pd(&C[3 * ldc + 8], _C3_1);
}
void micro_kernel1x5(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register double _A0;
    register double _C0_0 = C[0 * ldc + 0];
    register double _C1_0 = C[1 * ldc + 0];
    register double _C2_0 = C[2 * ldc + 0];
    register double _C3_0 = C[3 * ldc + 0];
    register double _C4_0 = C[4 * ldc + 0];

    for(i = 0; i < k; ++i) {
        _A0 = A[0];
        _C0_0 += B[0 * ldb] * _A0;
        _C1_0 += B[1 * ldb] * _A0;
        _C2_0 += B[2 * ldb] * _A0;
        _C3_0 += B[3 * ldb] * _A0;
        _C4_0 += B[4 * ldb] * _A0;
        A += lda;
        B += 1;
    }

    C[0 * ldc + 0] = _C0_0;
    C[1 * ldc + 0] = _C1_0;
    C[2 * ldc + 0] = _C2_0;
    C[3 * ldc + 0] = _C3_0;
    C[4 * ldc + 0] = _C4_0;
}
void micro_kernel2x5(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m128d _A0;
    register __m128d _C0_0 = _mm_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C1_0 = _mm_loadu_pd(&C[1 * ldc + 0]);
    register __m128d _C2_0 = _mm_loadu_pd(&C[2 * ldc + 0]);
    register __m128d _C3_0 = _mm_loadu_pd(&C[3 * ldc + 0]);
    register __m128d _C4_0 = _mm_loadu_pd(&C[4 * ldc + 0]);

    for(i = 0; i < k; ++i) {
        _A0 = _mm_loadu_pd(&A[0]);
        _C0_0 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C1_0 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C2_0 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C3_0 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C4_0 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A0, _C4_0);
        A += lda;
        B += 1;
    }

    _mm_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm_storeu_pd(&C[4 * ldc + 0], _C4_0);
}
void micro_kernel3x5(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m128d _A0;
    register double _A1;
    register __m128d _C0_0 = _mm_loadu_pd(&C[0 * ldc + 0]);
    register double _C0_1 = C[0 * ldc + 2];
    register __m128d _C1_0 = _mm_loadu_pd(&C[1 * ldc + 0]);
    register double _C1_1 = C[1 * ldc + 2];
    register __m128d _C2_0 = _mm_loadu_pd(&C[2 * ldc + 0]);
    register double _C2_1 = C[2 * ldc + 2];
    register __m128d _C3_0 = _mm_loadu_pd(&C[3 * ldc + 0]);
    register double _C3_1 = C[3 * ldc + 2];
    register __m128d _C4_0 = _mm_loadu_pd(&C[4 * ldc + 0]);
    register double _C4_1 = C[4 * ldc + 2];

    for(i = 0; i < k; ++i) {
        _A0 = _mm_loadu_pd(&A[0]);
        _A1 = A[2];
        _C0_0 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 += B[0 * ldb] * _A1;
        _C1_0 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 += B[1 * ldb] * _A1;
        _C2_0 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 += B[2 * ldb] * _A1;
        _C3_0 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 += B[3 * ldb] * _A1;
        _C4_0 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 += B[4 * ldb] * _A1;
        A += lda;
        B += 1;
    }

    _mm_storeu_pd(&C[0 * ldc + 0], _C0_0);
    C[0 * ldc + 2] = _C0_1;
    _mm_storeu_pd(&C[1 * ldc + 0], _C1_0);
    C[1 * ldc + 2] = _C1_1;
    _mm_storeu_pd(&C[2 * ldc + 0], _C2_0);
    C[2 * ldc + 2] = _C2_1;
    _mm_storeu_pd(&C[3 * ldc + 0], _C3_0);
    C[3 * ldc + 2] = _C3_1;
    _mm_storeu_pd(&C[4 * ldc + 0], _C4_0);
    C[4 * ldc + 2] = _C4_1;
}
void micro_kernel4x5(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m256d _A0;
    register __m256d _C0_0 = _mm256_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C1_0 = _mm256_loadu_pd(&C[1 * ldc + 0]);
    register __m256d _C2_0 = _mm256_loadu_pd(&C[2 * ldc + 0]);
    register __m256d _C3_0 = _mm256_loadu_pd(&C[3 * ldc + 0]);
    register __m256d _C4_0 = _mm256_loadu_pd(&C[4 * ldc + 0]);

    for(i = 0; i < k; ++i) {
        _A0 = _mm256_loadu_pd(&A[0]);
        _C0_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C1_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C2_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C3_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C4_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A0, _C4_0);
        A += lda;
        B += 1;
    }

    _mm256_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm256_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm256_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm256_storeu_pd(&C[4 * ldc + 0], _C4_0);
}
void micro_kernel5x5(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m256d _A0;
    register double _A1;
    register __m256d _C0_0 = _mm256_loadu_pd(&C[0 * ldc + 0]);
    register double _C0_1 = C[0 * ldc + 4];
    register __m256d _C1_0 = _mm256_loadu_pd(&C[1 * ldc + 0]);
    register double _C1_1 = C[1 * ldc + 4];
    register __m256d _C2_0 = _mm256_loadu_pd(&C[2 * ldc + 0]);
    register double _C2_1 = C[2 * ldc + 4];
    register __m256d _C3_0 = _mm256_loadu_pd(&C[3 * ldc + 0]);
    register double _C3_1 = C[3 * ldc + 4];
    register __m256d _C4_0 = _mm256_loadu_pd(&C[4 * ldc + 0]);
    register double _C4_1 = C[4 * ldc + 4];

    for(i = 0; i < k; ++i) {
        _A0 = _mm256_loadu_pd(&A[0]);
        _A1 = A[4];
        _C0_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 += B[0 * ldb] * _A1;
        _C1_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 += B[1 * ldb] * _A1;
        _C2_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 += B[2 * ldb] * _A1;
        _C3_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 += B[3 * ldb] * _A1;
        _C4_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 += B[4 * ldb] * _A1;
        A += lda;
        B += 1;
    }

    _mm256_storeu_pd(&C[0 * ldc + 0], _C0_0);
    C[0 * ldc + 4] = _C0_1;
    _mm256_storeu_pd(&C[1 * ldc + 0], _C1_0);
    C[1 * ldc + 4] = _C1_1;
    _mm256_storeu_pd(&C[2 * ldc + 0], _C2_0);
    C[2 * ldc + 4] = _C2_1;
    _mm256_storeu_pd(&C[3 * ldc + 0], _C3_0);
    C[3 * ldc + 4] = _C3_1;
    _mm256_storeu_pd(&C[4 * ldc + 0], _C4_0);
    C[4 * ldc + 4] = _C4_1;
}
void micro_kernel6x5(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m256d _A0;
    register __m128d _A1;
    register __m256d _C0_0 = _mm256_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C0_1 = _mm_loadu_pd(&C[0 * ldc + 4]);
    register __m256d _C1_0 = _mm256_loadu_pd(&C[1 * ldc + 0]);
    register __m128d _C1_1 = _mm_loadu_pd(&C[1 * ldc + 4]);
    register __m256d _C2_0 = _mm256_loadu_pd(&C[2 * ldc + 0]);
    register __m128d _C2_1 = _mm_loadu_pd(&C[2 * ldc + 4]);
    register __m256d _C3_0 = _mm256_loadu_pd(&C[3 * ldc + 0]);
    register __m128d _C3_1 = _mm_loadu_pd(&C[3 * ldc + 4]);
    register __m256d _C4_0 = _mm256_loadu_pd(&C[4 * ldc + 0]);
    register __m128d _C4_1 = _mm_loadu_pd(&C[4 * ldc + 4]);

    for(i = 0; i < k; ++i) {
        _A0 = _mm256_loadu_pd(&A[0]);
        _A1 = _mm_loadu_pd(&A[4]);
        _C0_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C1_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C2_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C3_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A1, _C3_1);
        _C4_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A1, _C4_1);
        A += lda;
        B += 1;
    }

    _mm256_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[0 * ldc + 4], _C0_1);
    _mm256_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm_storeu_pd(&C[1 * ldc + 4], _C1_1);
    _mm256_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm_storeu_pd(&C[2 * ldc + 4], _C2_1);
    _mm256_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm_storeu_pd(&C[3 * ldc + 4], _C3_1);
    _mm256_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm_storeu_pd(&C[4 * ldc + 4], _C4_1);
}
void micro_kernel7x5(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m256d _A0;
    register __m128d _A1;
    register double _A2;
    register __m256d _C0_0 = _mm256_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C0_1 = _mm_loadu_pd(&C[0 * ldc + 4]);
    register double _C0_2 = C[0 * ldc + 6];
    register __m256d _C1_0 = _mm256_loadu_pd(&C[1 * ldc + 0]);
    register __m128d _C1_1 = _mm_loadu_pd(&C[1 * ldc + 4]);
    register double _C1_2 = C[1 * ldc + 6];
    register __m256d _C2_0 = _mm256_loadu_pd(&C[2 * ldc + 0]);
    register __m128d _C2_1 = _mm_loadu_pd(&C[2 * ldc + 4]);
    register double _C2_2 = C[2 * ldc + 6];
    register __m256d _C3_0 = _mm256_loadu_pd(&C[3 * ldc + 0]);
    register __m128d _C3_1 = _mm_loadu_pd(&C[3 * ldc + 4]);
    register double _C3_2 = C[3 * ldc + 6];
    register __m256d _C4_0 = _mm256_loadu_pd(&C[4 * ldc + 0]);
    register __m128d _C4_1 = _mm_loadu_pd(&C[4 * ldc + 4]);
    register double _C4_2 = C[4 * ldc + 6];

    for(i = 0; i < k; ++i) {
        _A0 = _mm256_loadu_pd(&A[0]);
        _A1 = _mm_loadu_pd(&A[4]);
        _A2 = A[6];
        _C0_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 += B[0 * ldb] * _A2;
        _C1_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C1_2 += B[1 * ldb] * _A2;
        _C2_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C2_2 += B[2 * ldb] * _A2;
        _C3_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A1, _C3_1);
        _C3_2 += B[3 * ldb] * _A2;
        _C4_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A1, _C4_1);
        _C4_2 += B[4 * ldb] * _A2;
        A += lda;
        B += 1;
    }

    _mm256_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[0 * ldc + 4], _C0_1);
    C[0 * ldc + 6] = _C0_2;
    _mm256_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm_storeu_pd(&C[1 * ldc + 4], _C1_1);
    C[1 * ldc + 6] = _C1_2;
    _mm256_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm_storeu_pd(&C[2 * ldc + 4], _C2_1);
    C[2 * ldc + 6] = _C2_2;
    _mm256_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm_storeu_pd(&C[3 * ldc + 4], _C3_1);
    C[3 * ldc + 6] = _C3_2;
    _mm256_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm_storeu_pd(&C[4 * ldc + 4], _C4_1);
    C[4 * ldc + 6] = _C4_2;
}
//=========================================================================================================
void micro_kernel8x5(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);
    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);

    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);

        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);

}
void micro_kernel9x5(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register double _A1;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register double _C0_1 = C[0 * ldc + 8];

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);
    register double _C1_1 = C[1 * ldc + 8];

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register double _C2_1 = C[2 * ldc + 8];

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register double _C3_1 = C[3 * ldc + 8];

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register double _C4_1 = C[4 * ldc + 8];

    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = A[8];

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 += B[0 * ldb] * _A1;

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 += B[1 * ldb] * _A1;

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 += B[2 * ldb] * _A1;

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 += B[3 * ldb] * _A1;

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 += B[4 * ldb] * _A1;
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    C[0 * ldc + 8] = _C0_1;

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    C[1 * ldc + 8] = _C1_1;

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    C[2 * ldc + 8] = _C2_1;

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    C[3 * ldc + 8] = _C3_1;

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    C[4 * ldc + 8] = _C4_1;
}
void micro_kernel10x5(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m128d _A1;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C0_1 = _mm_loadu_pd(&C[0 * ldc + 8]);

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m128d _C1_1 = _mm_loadu_pd(&C[1 * ldc + 8]);

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m128d _C2_1 = _mm_loadu_pd(&C[2 * ldc + 8]);

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m128d _C3_1 = _mm_loadu_pd(&C[3 * ldc + 8]);

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m128d _C4_1 = _mm_loadu_pd(&C[4 * ldc + 8]);
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm_loadu_pd(&A[8]);

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A1, _C0_1);

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A1, _C1_1);

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A1, _C2_1);

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A1, _C3_1);

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A1, _C4_1);
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[0 * ldc + 8], _C0_1);

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm_storeu_pd(&C[1 * ldc + 8], _C1_1);

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm_storeu_pd(&C[2 * ldc + 8], _C2_1);

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm_storeu_pd(&C[3 * ldc + 8], _C3_1);

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm_storeu_pd(&C[4 * ldc + 8], _C4_1);
}
void micro_kernel11x5(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m128d _A1;
    register double _A2;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C0_1 = _mm_loadu_pd(&C[0 * ldc + 8]);
    register double _C0_2 = C[0 * ldc + 10];

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m128d _C1_1 = _mm_loadu_pd(&C[1 * ldc + 8]);
    register double _C1_2 = C[1 * ldc + 10];

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m128d _C2_1 = _mm_loadu_pd(&C[2 * ldc + 8]);
    register double _C2_2 = C[2 * ldc + 10];

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m128d _C3_1 = _mm_loadu_pd(&C[3 * ldc + 8]);
    register double _C3_2 = C[3 * ldc + 10];

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m128d _C4_1 = _mm_loadu_pd(&C[4 * ldc + 8]);
    register double _C4_2 = C[4 * ldc + 10];
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm_loadu_pd(&A[8]);
        _A2 = A[10];

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 += B[0 * ldb] * _A2;

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C1_2 += B[1 * ldb] * _A2;

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C2_2 += B[2 * ldb] * _A2;

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A1, _C3_1);
        _C3_2 += B[3 * ldb] * _A2;

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A1, _C4_1);
        _C4_2 += B[4 * ldb] * _A2;
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[0 * ldc + 8], _C0_1);
    C[0 * ldc + 10] = _C0_2;

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm_storeu_pd(&C[1 * ldc + 8], _C1_1);
    C[1 * ldc + 10] = _C1_2;

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm_storeu_pd(&C[2 * ldc + 8], _C2_1);
    C[2 * ldc + 10] = _C2_2;

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm_storeu_pd(&C[3 * ldc + 8], _C3_1);
    C[3 * ldc + 10] = _C3_2;

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm_storeu_pd(&C[4 * ldc + 8], _C4_1);
    C[4 * ldc + 10] = _C4_2;
}
void micro_kernel12x5(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
     int i;
    register __m512d _A0;
    register __m256d _A1;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C0_1 = _mm256_loadu_pd(&C[0 * ldc + 8]);

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m256d _C1_1 = _mm256_loadu_pd(&C[1 * ldc + 8]);

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m256d _C2_1 = _mm256_loadu_pd(&C[2 * ldc + 8]);

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m256d _C3_1 = _mm256_loadu_pd(&C[3 * ldc + 8]);
    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m256d _C4_1 = _mm256_loadu_pd(&C[4 * ldc + 8]);
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm256_loadu_pd(&A[8]);

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A1, _C0_1);

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A1, _C1_1);

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A1, _C2_1);

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A1, _C3_1);
        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A1, _C4_1);
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[0 * ldc + 8], _C0_1);

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm256_storeu_pd(&C[1 * ldc + 8], _C1_1);

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm256_storeu_pd(&C[2 * ldc + 8], _C2_1);

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm256_storeu_pd(&C[3 * ldc + 8], _C3_1);
    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm256_storeu_pd(&C[4 * ldc + 8], _C4_1);
}
void micro_kernel13x5(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m256d _A1;
    register double _A2;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C0_1 = _mm256_loadu_pd(&C[0 * ldc + 8]);
    register double _C0_2 = C[0 * ldc + 12];

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m256d _C1_1 = _mm256_loadu_pd(&C[1 * ldc + 8]);
    register double _C1_2 = C[1 * ldc + 12];

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m256d _C2_1 = _mm256_loadu_pd(&C[2 * ldc + 8]);
    register double _C2_2 = C[2 * ldc + 12];

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m256d _C3_1 = _mm256_loadu_pd(&C[3 * ldc + 8]);
    register double _C3_2 = C[3 * ldc + 12];

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m256d _C4_1 = _mm256_loadu_pd(&C[4 * ldc + 8]);
    register double _C4_2 = C[4 * ldc + 12];
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm256_loadu_pd(&A[8]);
        _A2 = A[12];

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 += B[0 * ldb] * _A2;

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C1_2 += B[1 * ldb] * _A2;

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C2_2 += B[2 * ldb] * _A2;

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A1, _C3_1);
        _C3_2 += B[3 * ldb] * _A2;

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A1, _C4_1);
        _C4_2 += B[4 * ldb] * _A2;
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[0 * ldc + 8], _C0_1);
    C[0 * ldc + 12] = _C0_2;

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm256_storeu_pd(&C[1 * ldc + 8], _C1_1);
    C[1 * ldc + 12] = _C1_2;

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm256_storeu_pd(&C[2 * ldc + 8], _C2_1);
    C[2 * ldc + 12] = _C2_2;

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm256_storeu_pd(&C[3 * ldc + 8], _C3_1);
    C[3 * ldc + 12] = _C3_2;

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm256_storeu_pd(&C[4 * ldc + 8], _C4_1);
    C[4 * ldc + 12] = _C4_2;
}
void micro_kernel14x5(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m256d _A1;
    register __m128d _A2;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C0_1 = _mm256_loadu_pd(&C[0 * ldc + 8]);
    register __m128d _C0_2 = _mm_loadu_pd(&C[0 * ldc + 12]);

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m256d _C1_1 = _mm256_loadu_pd(&C[1 * ldc + 8]);
    register __m128d _C1_2 = _mm_loadu_pd(&C[1 * ldc + 12]);

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m256d _C2_1 = _mm256_loadu_pd(&C[2 * ldc + 8]);
    register __m128d _C2_2 = _mm_loadu_pd(&C[2 * ldc + 12]);

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m256d _C3_1 = _mm256_loadu_pd(&C[3 * ldc + 8]);
    register __m128d _C3_2 = _mm_loadu_pd(&C[3 * ldc + 12]);
    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m256d _C4_1 = _mm256_loadu_pd(&C[4 * ldc + 8]);
    register __m128d _C4_2 = _mm_loadu_pd(&C[4 * ldc + 12]);
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm256_loadu_pd(&A[8]);
        _A2 = _mm_loadu_pd(&A[12]);

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A2, _C0_2);

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C1_2 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A2, _C1_2);

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C2_2 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A2, _C2_2);

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A1, _C3_1);
        _C3_2 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A2, _C3_2);

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A1, _C4_1);
        _C4_2 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A2, _C4_2);
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[0 * ldc + 8], _C0_1);
    _mm_storeu_pd(&C[0 * ldc + 12], _C0_2);

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm256_storeu_pd(&C[1 * ldc + 8], _C1_1);
    _mm_storeu_pd(&C[1 * ldc + 12], _C1_2);

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm256_storeu_pd(&C[2 * ldc + 8], _C2_1);
    _mm_storeu_pd(&C[2 * ldc + 12], _C2_2);

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm256_storeu_pd(&C[3 * ldc + 8], _C3_1);
    _mm_storeu_pd(&C[3 * ldc + 12], _C3_2);

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm256_storeu_pd(&C[4 * ldc + 8], _C4_1);
    _mm_storeu_pd(&C[4 * ldc + 12], _C4_2);
}
void micro_kernel15x5(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m256d _A1;
    register __m128d _A2;
    register double _A3;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C0_1 = _mm256_loadu_pd(&C[0 * ldc + 8]);
    register __m128d _C0_2 = _mm_loadu_pd(&C[0 * ldc + 12]);
    register double _C0_3 = C[0 * ldc + 14];

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m256d _C1_1 = _mm256_loadu_pd(&C[1 * ldc + 8]);
    register __m128d _C1_2 = _mm_loadu_pd(&C[1 * ldc + 12]);
    register double _C1_3 = C[1 * ldc + 14];

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m256d _C2_1 = _mm256_loadu_pd(&C[2 * ldc + 8]);
    register __m128d _C2_2 = _mm_loadu_pd(&C[2 * ldc + 12]);
    register double _C2_3 = C[2 * ldc + 14];

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m256d _C3_1 = _mm256_loadu_pd(&C[3 * ldc + 8]);
    register __m128d _C3_2 = _mm_loadu_pd(&C[3 * ldc + 12]);
    register double _C3_3 = C[3 * ldc + 14];

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m256d _C4_1 = _mm256_loadu_pd(&C[4 * ldc + 8]);
    register __m128d _C4_2 = _mm_loadu_pd(&C[4 * ldc + 12]);
    register double _C4_3 = C[4 * ldc + 14];
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm256_loadu_pd(&A[8]);
        _A2 = _mm_loadu_pd(&A[12]);
        _A3 = A[14];

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A2, _C0_2);
        _C0_3 += B[0 * ldb] * _A3;

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C1_2 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A2, _C1_2);
        _C1_3 += B[1 * ldb] * _A3;

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C2_2 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A2, _C2_2);
        _C2_3 += B[2 * ldb] * _A3;

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A1, _C3_1);
        _C3_2 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A2, _C3_2);
        _C3_3 += B[3 * ldb] * _A3;

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A1, _C4_1);
        _C4_2 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A2, _C4_2);
        _C4_3 += B[4 * ldb] * _A3;
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[0 * ldc + 8], _C0_1);
    _mm_storeu_pd(&C[0 * ldc + 12], _C0_2);
    C[0 * ldc + 14] = _C0_3;

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm256_storeu_pd(&C[1 * ldc + 8], _C1_1);
    _mm_storeu_pd(&C[1 * ldc + 12], _C1_2);
    C[1 * ldc + 14] = _C1_3;

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm256_storeu_pd(&C[2 * ldc + 8], _C2_1);
    _mm_storeu_pd(&C[2 * ldc + 12], _C2_2);
    C[2 * ldc + 14] = _C2_3;

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm256_storeu_pd(&C[3 * ldc + 8], _C3_1);
    _mm_storeu_pd(&C[3 * ldc + 12], _C3_2);
    C[3 * ldc + 14] = _C3_3;

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm256_storeu_pd(&C[4 * ldc + 8], _C4_1);
    _mm_storeu_pd(&C[4 * ldc + 12], _C4_2);
    C[4 * ldc + 14] = _C4_3;
}
void micro_kernel16x5(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m512d _A1;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m512d _C0_1 = _mm512_loadu_pd(&C[0 * ldc + 8]);

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m512d _C1_1 = _mm512_loadu_pd(&C[1 * ldc + 8]);

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m512d _C2_1 = _mm512_loadu_pd(&C[2 * ldc + 8]);

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m512d _C3_1 = _mm512_loadu_pd(&C[3 * ldc + 8]);

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m512d _C4_1 = _mm512_loadu_pd(&C[4 * ldc + 8]);
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm512_loadu_pd(&A[8]);

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A1, _C0_1);

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A1, _C1_1);

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A1, _C2_1);

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A1, _C3_1);

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A1, _C4_1);
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm512_storeu_pd(&C[0 * ldc + 8], _C0_1);

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm512_storeu_pd(&C[1 * ldc + 8], _C1_1);

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm512_storeu_pd(&C[2 * ldc + 8], _C2_1);

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm512_storeu_pd(&C[3 * ldc + 8], _C3_1);

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm512_storeu_pd(&C[4 * ldc + 8], _C4_1);
}
void micro_kernel1x6(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register double _A0;
    register double _C0_0 = C[0 * ldc + 0];
    register double _C1_0 = C[1 * ldc + 0];
    register double _C2_0 = C[2 * ldc + 0];
    register double _C3_0 = C[3 * ldc + 0];
    register double _C4_0 = C[4 * ldc + 0];
    register double _C5_0 = C[5 * ldc + 0];

    for(i = 0; i < k; ++i) {
        _A0 = A[0];
        _C0_0 += B[0 * ldb] * _A0;
        _C1_0 += B[1 * ldb] * _A0;
        _C2_0 += B[2 * ldb] * _A0;
        _C3_0 += B[3 * ldb] * _A0;
        _C4_0 += B[4 * ldb] * _A0;
        _C5_0 += B[5 * ldb] * _A0;
        A += lda;
        B += 1;
    }

    C[0 * ldc + 0] = _C0_0;
    C[1 * ldc + 0] = _C1_0;
    C[2 * ldc + 0] = _C2_0;
    C[3 * ldc + 0] = _C3_0;
    C[4 * ldc + 0] = _C4_0;
    C[5 * ldc + 0] = _C5_0;
}
void micro_kernel2x6(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m128d _A0;
    register __m128d _C0_0 = _mm_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C1_0 = _mm_loadu_pd(&C[1 * ldc + 0]);
    register __m128d _C2_0 = _mm_loadu_pd(&C[2 * ldc + 0]);
    register __m128d _C3_0 = _mm_loadu_pd(&C[3 * ldc + 0]);
    register __m128d _C4_0 = _mm_loadu_pd(&C[4 * ldc + 0]);
    register __m128d _C5_0 = _mm_loadu_pd(&C[5 * ldc + 0]);

    for(i = 0; i < k; ++i) {
        _A0 = _mm_loadu_pd(&A[0]);
        _C0_0 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C1_0 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C2_0 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C3_0 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C4_0 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C5_0 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A0, _C5_0);
        A += lda;
        B += 1;
    }

    _mm_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm_storeu_pd(&C[5 * ldc + 0], _C5_0);
}
void micro_kernel3x6(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m128d _A0;
    register double _A1;
    register __m128d _C0_0 = _mm_loadu_pd(&C[0 * ldc + 0]);
    register double _C0_1 = C[0 * ldc + 2];
    register __m128d _C1_0 = _mm_loadu_pd(&C[1 * ldc + 0]);
    register double _C1_1 = C[1 * ldc + 2];
    register __m128d _C2_0 = _mm_loadu_pd(&C[2 * ldc + 0]);
    register double _C2_1 = C[2 * ldc + 2];
    register __m128d _C3_0 = _mm_loadu_pd(&C[3 * ldc + 0]);
    register double _C3_1 = C[3 * ldc + 2];
    register __m128d _C4_0 = _mm_loadu_pd(&C[4 * ldc + 0]);
    register double _C4_1 = C[4 * ldc + 2];
    register __m128d _C5_0 = _mm_loadu_pd(&C[5 * ldc + 0]);
    register double _C5_1 = C[5 * ldc + 2];

    for(i = 0; i < k; ++i) {
        _A0 = _mm_loadu_pd(&A[0]);
        _A1 = A[2];
        _C0_0 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 += B[0 * ldb] * _A1;
        _C1_0 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 += B[1 * ldb] * _A1;
        _C2_0 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 += B[2 * ldb] * _A1;
        _C3_0 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 += B[3 * ldb] * _A1;
        _C4_0 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 += B[4 * ldb] * _A1;
        _C5_0 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 += B[5 * ldb] * _A1;
        A += lda;
        B += 1;
    }

    _mm_storeu_pd(&C[0 * ldc + 0], _C0_0);
    C[0 * ldc + 2] = _C0_1;
    _mm_storeu_pd(&C[1 * ldc + 0], _C1_0);
    C[1 * ldc + 2] = _C1_1;
    _mm_storeu_pd(&C[2 * ldc + 0], _C2_0);
    C[2 * ldc + 2] = _C2_1;
    _mm_storeu_pd(&C[3 * ldc + 0], _C3_0);
    C[3 * ldc + 2] = _C3_1;
    _mm_storeu_pd(&C[4 * ldc + 0], _C4_0);
    C[4 * ldc + 2] = _C4_1;
    _mm_storeu_pd(&C[5 * ldc + 0], _C5_0);
    C[5 * ldc + 2] = _C5_1;
}
void micro_kernel4x6(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m256d _A0;
    register __m256d _C0_0 = _mm256_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C1_0 = _mm256_loadu_pd(&C[1 * ldc + 0]);
    register __m256d _C2_0 = _mm256_loadu_pd(&C[2 * ldc + 0]);
    register __m256d _C3_0 = _mm256_loadu_pd(&C[3 * ldc + 0]);
    register __m256d _C4_0 = _mm256_loadu_pd(&C[4 * ldc + 0]);
    register __m256d _C5_0 = _mm256_loadu_pd(&C[5 * ldc + 0]);

    for(i = 0; i < k; ++i) {
        _A0 = _mm256_loadu_pd(&A[0]);
        _C0_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C1_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C2_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C3_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C4_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C5_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A0, _C5_0);
        A += lda;
        B += 1;
    }

    _mm256_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm256_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm256_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm256_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm256_storeu_pd(&C[5 * ldc + 0], _C5_0);
}
void micro_kernel5x6(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m256d _A0;
    register double _A1;
    register __m256d _C0_0 = _mm256_loadu_pd(&C[0 * ldc + 0]);
    register double _C0_1 = C[0 * ldc + 4];
    register __m256d _C1_0 = _mm256_loadu_pd(&C[1 * ldc + 0]);
    register double _C1_1 = C[1 * ldc + 4];
    register __m256d _C2_0 = _mm256_loadu_pd(&C[2 * ldc + 0]);
    register double _C2_1 = C[2 * ldc + 4];
    register __m256d _C3_0 = _mm256_loadu_pd(&C[3 * ldc + 0]);
    register double _C3_1 = C[3 * ldc + 4];
    register __m256d _C4_0 = _mm256_loadu_pd(&C[4 * ldc + 0]);
    register double _C4_1 = C[4 * ldc + 4];
    register __m256d _C5_0 = _mm256_loadu_pd(&C[5 * ldc + 0]);
    register double _C5_1 = C[5 * ldc + 4];

    for(i = 0; i < k; ++i) {
        _A0 = _mm256_loadu_pd(&A[0]);
        _A1 = A[4];
        _C0_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 += B[0 * ldb] * _A1;
        _C1_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 += B[1 * ldb] * _A1;
        _C2_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 += B[2 * ldb] * _A1;
        _C3_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 += B[3 * ldb] * _A1;
        _C4_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 += B[4 * ldb] * _A1;
        _C5_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 += B[5 * ldb] * _A1;
        A += lda;
        B += 1;
    }

    _mm256_storeu_pd(&C[0 * ldc + 0], _C0_0);
    C[0 * ldc + 4] = _C0_1;
    _mm256_storeu_pd(&C[1 * ldc + 0], _C1_0);
    C[1 * ldc + 4] = _C1_1;
    _mm256_storeu_pd(&C[2 * ldc + 0], _C2_0);
    C[2 * ldc + 4] = _C2_1;
    _mm256_storeu_pd(&C[3 * ldc + 0], _C3_0);
    C[3 * ldc + 4] = _C3_1;
    _mm256_storeu_pd(&C[4 * ldc + 0], _C4_0);
    C[4 * ldc + 4] = _C4_1;
    _mm256_storeu_pd(&C[5 * ldc + 0], _C5_0);
    C[5 * ldc + 4] = _C5_1;
}
void micro_kernel6x6(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m256d _A0;
    register __m128d _A1;
    register __m256d _C0_0 = _mm256_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C0_1 = _mm_loadu_pd(&C[0 * ldc + 4]);
    register __m256d _C1_0 = _mm256_loadu_pd(&C[1 * ldc + 0]);
    register __m128d _C1_1 = _mm_loadu_pd(&C[1 * ldc + 4]);
    register __m256d _C2_0 = _mm256_loadu_pd(&C[2 * ldc + 0]);
    register __m128d _C2_1 = _mm_loadu_pd(&C[2 * ldc + 4]);
    register __m256d _C3_0 = _mm256_loadu_pd(&C[3 * ldc + 0]);
    register __m128d _C3_1 = _mm_loadu_pd(&C[3 * ldc + 4]);
    register __m256d _C4_0 = _mm256_loadu_pd(&C[4 * ldc + 0]);
    register __m128d _C4_1 = _mm_loadu_pd(&C[4 * ldc + 4]);
    register __m256d _C5_0 = _mm256_loadu_pd(&C[5 * ldc + 0]);
    register __m128d _C5_1 = _mm_loadu_pd(&C[5 * ldc + 4]);

    for(i = 0; i < k; ++i) {
        _A0 = _mm256_loadu_pd(&A[0]);
        _A1 = _mm_loadu_pd(&A[4]);
        _C0_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C1_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C2_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C3_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A1, _C3_1);
        _C4_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A1, _C4_1);
        _C5_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A1, _C5_1);
        A += lda;
        B += 1;
    }

    _mm256_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[0 * ldc + 4], _C0_1);
    _mm256_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm_storeu_pd(&C[1 * ldc + 4], _C1_1);
    _mm256_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm_storeu_pd(&C[2 * ldc + 4], _C2_1);
    _mm256_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm_storeu_pd(&C[3 * ldc + 4], _C3_1);
    _mm256_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm_storeu_pd(&C[4 * ldc + 4], _C4_1);
    _mm256_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm_storeu_pd(&C[5 * ldc + 4], _C5_1);
}
void micro_kernel7x6(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m256d _A0;
    register __m128d _A1;
    register double _A2;
    register __m256d _C0_0 = _mm256_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C0_1 = _mm_loadu_pd(&C[0 * ldc + 4]);
    register double _C0_2 = C[0 * ldc + 6];
    register __m256d _C1_0 = _mm256_loadu_pd(&C[1 * ldc + 0]);
    register __m128d _C1_1 = _mm_loadu_pd(&C[1 * ldc + 4]);
    register double _C1_2 = C[1 * ldc + 6];
    register __m256d _C2_0 = _mm256_loadu_pd(&C[2 * ldc + 0]);
    register __m128d _C2_1 = _mm_loadu_pd(&C[2 * ldc + 4]);
    register double _C2_2 = C[2 * ldc + 6];
    register __m256d _C3_0 = _mm256_loadu_pd(&C[3 * ldc + 0]);
    register __m128d _C3_1 = _mm_loadu_pd(&C[3 * ldc + 4]);
    register double _C3_2 = C[3 * ldc + 6];
    register __m256d _C4_0 = _mm256_loadu_pd(&C[4 * ldc + 0]);
    register __m128d _C4_1 = _mm_loadu_pd(&C[4 * ldc + 4]);
    register double _C4_2 = C[4 * ldc + 6];
    register __m256d _C5_0 = _mm256_loadu_pd(&C[5 * ldc + 0]);
    register __m128d _C5_1 = _mm_loadu_pd(&C[5 * ldc + 4]);
    register double _C5_2 = C[5 * ldc + 6];

    for(i = 0; i < k; ++i) {
        _A0 = _mm256_loadu_pd(&A[0]);
        _A1 = _mm_loadu_pd(&A[4]);
        _A2 = A[6];
        _C0_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 += B[0 * ldb] * _A2;
        _C1_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C1_2 += B[1 * ldb] * _A2;
        _C2_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C2_2 += B[2 * ldb] * _A2;
        _C3_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A1, _C3_1);
        _C3_2 += B[3 * ldb] * _A2;
        _C4_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A1, _C4_1);
        _C4_2 += B[4 * ldb] * _A2;
        _C5_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A1, _C5_1);
        _C5_2 += B[5 * ldb] * _A2;
        A += lda;
        B += 1;
    }

    _mm256_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[0 * ldc + 4], _C0_1);
    C[0 * ldc + 6] = _C0_2;
    _mm256_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm_storeu_pd(&C[1 * ldc + 4], _C1_1);
    C[1 * ldc + 6] = _C1_2;
    _mm256_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm_storeu_pd(&C[2 * ldc + 4], _C2_1);
    C[2 * ldc + 6] = _C2_2;
    _mm256_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm_storeu_pd(&C[3 * ldc + 4], _C3_1);
    C[3 * ldc + 6] = _C3_2;
    _mm256_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm_storeu_pd(&C[4 * ldc + 4], _C4_1);
    C[4 * ldc + 6] = _C4_2;
    _mm256_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm_storeu_pd(&C[5 * ldc + 4], _C5_1);
    C[5 * ldc + 6] = _C5_2;
}
//=====================================================================================================================
void micro_kernel8x6(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);
    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);

    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);

        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
}
void micro_kernel9x6(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register double _A1;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register double _C0_1 = C[0 * ldc + 8];

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);
    register double _C1_1 = C[1 * ldc + 8];

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register double _C2_1 = C[2 * ldc + 8];

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register double _C3_1 = C[3 * ldc + 8];

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register double _C4_1 = C[4 * ldc + 8];

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register double _C5_1 = C[5 * ldc + 8];

    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = A[8];

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 += B[0 * ldb] * _A1;

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 += B[1 * ldb] * _A1;

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 += B[2 * ldb] * _A1;

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 += B[3 * ldb] * _A1;

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 += B[4 * ldb] * _A1;

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 += B[5 * ldb] * _A1;
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    C[0 * ldc + 8] = _C0_1;

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    C[1 * ldc + 8] = _C1_1;

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    C[2 * ldc + 8] = _C2_1;

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    C[3 * ldc + 8] = _C3_1;

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    C[4 * ldc + 8] = _C4_1;

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    C[5 * ldc + 8] = _C5_1;
}
void micro_kernel10x6(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m128d _A1;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C0_1 = _mm_loadu_pd(&C[0 * ldc + 8]);

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m128d _C1_1 = _mm_loadu_pd(&C[1 * ldc + 8]);

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m128d _C2_1 = _mm_loadu_pd(&C[2 * ldc + 8]);

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m128d _C3_1 = _mm_loadu_pd(&C[3 * ldc + 8]);

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m128d _C4_1 = _mm_loadu_pd(&C[4 * ldc + 8]);

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register __m128d _C5_1 = _mm_loadu_pd(&C[5 * ldc + 8]);
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm_loadu_pd(&A[8]);

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A1, _C0_1);

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A1, _C1_1);

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A1, _C2_1);

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A1, _C3_1);

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A1, _C4_1);

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A1, _C5_1);
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[0 * ldc + 8], _C0_1);

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm_storeu_pd(&C[1 * ldc + 8], _C1_1);

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm_storeu_pd(&C[2 * ldc + 8], _C2_1);

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm_storeu_pd(&C[3 * ldc + 8], _C3_1);

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm_storeu_pd(&C[4 * ldc + 8], _C4_1);

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm_storeu_pd(&C[5 * ldc + 8], _C5_1);
}
void micro_kernel11x6(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m128d _A1;
    register double _A2;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C0_1 = _mm_loadu_pd(&C[0 * ldc + 8]);
    register double _C0_2 = C[0 * ldc + 10];

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m128d _C1_1 = _mm_loadu_pd(&C[1 * ldc + 8]);
    register double _C1_2 = C[1 * ldc + 10];

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m128d _C2_1 = _mm_loadu_pd(&C[2 * ldc + 8]);
    register double _C2_2 = C[2 * ldc + 10];

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m128d _C3_1 = _mm_loadu_pd(&C[3 * ldc + 8]);
    register double _C3_2 = C[3 * ldc + 10];

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m128d _C4_1 = _mm_loadu_pd(&C[4 * ldc + 8]);
    register double _C4_2 = C[4 * ldc + 10];

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register __m128d _C5_1 = _mm_loadu_pd(&C[5 * ldc + 8]);
    register double _C5_2 = C[5 * ldc + 10];
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm_loadu_pd(&A[8]);
        _A2 = A[10];

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 += B[0 * ldb] * _A2;

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C1_2 += B[1 * ldb] * _A2;

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C2_2 += B[2 * ldb] * _A2;

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A1, _C3_1);
        _C3_2 += B[3 * ldb] * _A2;

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A1, _C4_1);
        _C4_2 += B[4 * ldb] * _A2;

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A1, _C5_1);
        _C5_2 += B[5 * ldb] * _A2;
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[0 * ldc + 8], _C0_1);
    C[0 * ldc + 10] = _C0_2;

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm_storeu_pd(&C[1 * ldc + 8], _C1_1);
    C[1 * ldc + 10] = _C1_2;

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm_storeu_pd(&C[2 * ldc + 8], _C2_1);
    C[2 * ldc + 10] = _C2_2;

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm_storeu_pd(&C[3 * ldc + 8], _C3_1);
    C[3 * ldc + 10] = _C3_2;

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm_storeu_pd(&C[4 * ldc + 8], _C4_1);
    C[4 * ldc + 10] = _C4_2;

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm_storeu_pd(&C[5 * ldc + 8], _C5_1);
    C[5 * ldc + 10] = _C5_2;
}
void micro_kernel12x6(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
     int i;
    register __m512d _A0;
    register __m256d _A1;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C0_1 = _mm256_loadu_pd(&C[0 * ldc + 8]);

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m256d _C1_1 = _mm256_loadu_pd(&C[1 * ldc + 8]);

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m256d _C2_1 = _mm256_loadu_pd(&C[2 * ldc + 8]);

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m256d _C3_1 = _mm256_loadu_pd(&C[3 * ldc + 8]);

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m256d _C4_1 = _mm256_loadu_pd(&C[4 * ldc + 8]);

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register __m256d _C5_1 = _mm256_loadu_pd(&C[5 * ldc + 8]);
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm256_loadu_pd(&A[8]);

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A1, _C0_1);

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A1, _C1_1);

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A1, _C2_1);

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A1, _C3_1);

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A1, _C4_1);

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A1, _C5_1);
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[0 * ldc + 8], _C0_1);

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm256_storeu_pd(&C[1 * ldc + 8], _C1_1);

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm256_storeu_pd(&C[2 * ldc + 8], _C2_1);

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm256_storeu_pd(&C[3 * ldc + 8], _C3_1);

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm256_storeu_pd(&C[4 * ldc + 8], _C4_1);

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm256_storeu_pd(&C[5 * ldc + 8], _C5_1);
}
void micro_kernel13x6(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m256d _A1;
    register double _A2;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C0_1 = _mm256_loadu_pd(&C[0 * ldc + 8]);
    register double _C0_2 = C[0 * ldc + 12];

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m256d _C1_1 = _mm256_loadu_pd(&C[1 * ldc + 8]);
    register double _C1_2 = C[1 * ldc + 12];

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m256d _C2_1 = _mm256_loadu_pd(&C[2 * ldc + 8]);
    register double _C2_2 = C[2 * ldc + 12];

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m256d _C3_1 = _mm256_loadu_pd(&C[3 * ldc + 8]);
    register double _C3_2 = C[3 * ldc + 12];

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m256d _C4_1 = _mm256_loadu_pd(&C[4 * ldc + 8]);
    register double _C4_2 = C[4 * ldc + 12];

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register __m256d _C5_1 = _mm256_loadu_pd(&C[5 * ldc + 8]);
    register double _C5_2 = C[5 * ldc + 12];
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm256_loadu_pd(&A[8]);
        _A2 = A[12];

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 += B[0 * ldb] * _A2;

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C1_2 += B[1 * ldb] * _A2;

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C2_2 += B[2 * ldb] * _A2;

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A1, _C3_1);
        _C3_2 += B[3 * ldb] * _A2;

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A1, _C4_1);
        _C4_2 += B[4 * ldb] * _A2;

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A1, _C5_1);
        _C5_2 += B[5 * ldb] * _A2;
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[0 * ldc + 8], _C0_1);
    C[0 * ldc + 12] = _C0_2;

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm256_storeu_pd(&C[1 * ldc + 8], _C1_1);
    C[1 * ldc + 12] = _C1_2;

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm256_storeu_pd(&C[2 * ldc + 8], _C2_1);
    C[2 * ldc + 12] = _C2_2;

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm256_storeu_pd(&C[3 * ldc + 8], _C3_1);
    C[3 * ldc + 12] = _C3_2;

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm256_storeu_pd(&C[4 * ldc + 8], _C4_1);
    C[4 * ldc + 12] = _C4_2;

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm256_storeu_pd(&C[5 * ldc + 8], _C5_1);
    C[5 * ldc + 12] = _C5_2;
}
void micro_kernel14x6(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m256d _A1;
    register __m128d _A2;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C0_1 = _mm256_loadu_pd(&C[0 * ldc + 8]);
    register __m128d _C0_2 = _mm_loadu_pd(&C[0 * ldc + 12]);

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m256d _C1_1 = _mm256_loadu_pd(&C[1 * ldc + 8]);
    register __m128d _C1_2 = _mm_loadu_pd(&C[1 * ldc + 12]);

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m256d _C2_1 = _mm256_loadu_pd(&C[2 * ldc + 8]);
    register __m128d _C2_2 = _mm_loadu_pd(&C[2 * ldc + 12]);

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m256d _C3_1 = _mm256_loadu_pd(&C[3 * ldc + 8]);
    register __m128d _C3_2 = _mm_loadu_pd(&C[3 * ldc + 12]);

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m256d _C4_1 = _mm256_loadu_pd(&C[4 * ldc + 8]);
    register __m128d _C4_2 = _mm_loadu_pd(&C[4 * ldc + 12]);

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register __m256d _C5_1 = _mm256_loadu_pd(&C[5 * ldc + 8]);
    register __m128d _C5_2 = _mm_loadu_pd(&C[5 * ldc + 12]);
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm256_loadu_pd(&A[8]);
        _A2 = _mm_loadu_pd(&A[12]);

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A2, _C0_2);

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C1_2 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A2, _C1_2);

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C2_2 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A2, _C2_2);

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A1, _C3_1);
        _C3_2 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A2, _C3_2);

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A1, _C4_1);
        _C4_2 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A2, _C4_2);

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A1, _C5_1);
        _C5_2 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A2, _C5_2);
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[0 * ldc + 8], _C0_1);
    _mm_storeu_pd(&C[0 * ldc + 12], _C0_2);

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm256_storeu_pd(&C[1 * ldc + 8], _C1_1);
    _mm_storeu_pd(&C[1 * ldc + 12], _C1_2);

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm256_storeu_pd(&C[2 * ldc + 8], _C2_1);
    _mm_storeu_pd(&C[2 * ldc + 12], _C2_2);

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm256_storeu_pd(&C[3 * ldc + 8], _C3_1);
    _mm_storeu_pd(&C[3 * ldc + 12], _C3_2);

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm256_storeu_pd(&C[4 * ldc + 8], _C4_1);
    _mm_storeu_pd(&C[4 * ldc + 12], _C4_2);

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm256_storeu_pd(&C[5 * ldc + 8], _C5_1);
    _mm_storeu_pd(&C[5 * ldc + 12], _C5_2);
}
void micro_kernel15x6(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m256d _A1;
    register __m128d _A2;
    register double _A3;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C0_1 = _mm256_loadu_pd(&C[0 * ldc + 8]);
    register __m128d _C0_2 = _mm_loadu_pd(&C[0 * ldc + 12]);
    register double _C0_3 = C[0 * ldc + 14];

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m256d _C1_1 = _mm256_loadu_pd(&C[1 * ldc + 8]);
    register __m128d _C1_2 = _mm_loadu_pd(&C[1 * ldc + 12]);
    register double _C1_3 = C[1 * ldc + 14];

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m256d _C2_1 = _mm256_loadu_pd(&C[2 * ldc + 8]);
    register __m128d _C2_2 = _mm_loadu_pd(&C[2 * ldc + 12]);
    register double _C2_3 = C[2 * ldc + 14];

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m256d _C3_1 = _mm256_loadu_pd(&C[3 * ldc + 8]);
    register __m128d _C3_2 = _mm_loadu_pd(&C[3 * ldc + 12]);
    register double _C3_3 = C[3 * ldc + 14];

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m256d _C4_1 = _mm256_loadu_pd(&C[4 * ldc + 8]);
    register __m128d _C4_2 = _mm_loadu_pd(&C[4 * ldc + 12]);
    register double _C4_3 = C[4 * ldc + 14];

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register __m256d _C5_1 = _mm256_loadu_pd(&C[5 * ldc + 8]);
    register __m128d _C5_2 = _mm_loadu_pd(&C[5 * ldc + 12]);
    register double _C5_3 = C[5 * ldc + 14];
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm256_loadu_pd(&A[8]);
        _A2 = _mm_loadu_pd(&A[12]);
        _A3 = A[14];

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A2, _C0_2);
        _C0_3 += B[0 * ldb] * _A3;

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C1_2 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A2, _C1_2);
        _C1_3 += B[1 * ldb] * _A3;

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C2_2 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A2, _C2_2);
        _C2_3 += B[2 * ldb] * _A3;

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A1, _C3_1);
        _C3_2 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A2, _C3_2);
        _C3_3 += B[3 * ldb] * _A3;

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A1, _C4_1);
        _C4_2 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A2, _C4_2);
        _C4_3 += B[4 * ldb] * _A3;

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A1, _C5_1);
        _C5_2 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A2, _C5_2);
        _C5_3 += B[5 * ldb] * _A3;
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[0 * ldc + 8], _C0_1);
    _mm_storeu_pd(&C[0 * ldc + 12], _C0_2);
    C[0 * ldc + 14] = _C0_3;

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm256_storeu_pd(&C[1 * ldc + 8], _C1_1);
    _mm_storeu_pd(&C[1 * ldc + 12], _C1_2);
    C[1 * ldc + 14] = _C1_3;

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm256_storeu_pd(&C[2 * ldc + 8], _C2_1);
    _mm_storeu_pd(&C[2 * ldc + 12], _C2_2);
    C[2 * ldc + 14] = _C2_3;

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm256_storeu_pd(&C[3 * ldc + 8], _C3_1);
    _mm_storeu_pd(&C[3 * ldc + 12], _C3_2);
    C[3 * ldc + 14] = _C3_3;

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm256_storeu_pd(&C[4 * ldc + 8], _C4_1);
    _mm_storeu_pd(&C[4 * ldc + 12], _C4_2);
    C[4 * ldc + 14] = _C4_3;

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm256_storeu_pd(&C[5 * ldc + 8], _C5_1);
    _mm_storeu_pd(&C[5 * ldc + 12], _C5_2);
    C[5 * ldc + 14] = _C5_3;
}
void micro_kernel16x6(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m512d _A1;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m512d _C0_1 = _mm512_loadu_pd(&C[0 * ldc + 8]);

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m512d _C1_1 = _mm512_loadu_pd(&C[1 * ldc + 8]);

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m512d _C2_1 = _mm512_loadu_pd(&C[2 * ldc + 8]);

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m512d _C3_1 = _mm512_loadu_pd(&C[3 * ldc + 8]);

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m512d _C4_1 = _mm512_loadu_pd(&C[4 * ldc + 8]);

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register __m512d _C5_1 = _mm512_loadu_pd(&C[5 * ldc + 8]);
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm512_loadu_pd(&A[8]);

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A1, _C0_1);

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A1, _C1_1);

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A1, _C2_1);

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A1, _C3_1);

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A1, _C4_1);

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A1, _C5_1);
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm512_storeu_pd(&C[0 * ldc + 8], _C0_1);

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm512_storeu_pd(&C[1 * ldc + 8], _C1_1);

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm512_storeu_pd(&C[2 * ldc + 8], _C2_1);

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm512_storeu_pd(&C[3 * ldc + 8], _C3_1);

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm512_storeu_pd(&C[4 * ldc + 8], _C4_1);

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm512_storeu_pd(&C[5 * ldc + 8], _C5_1);
}
//#########################################################################################################
void micro_kernel1x7(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register double _A0;
    register double _C0_0 = C[0 * ldc + 0];
    register double _C1_0 = C[1 * ldc + 0];
    register double _C2_0 = C[2 * ldc + 0];
    register double _C3_0 = C[3 * ldc + 0];
    register double _C4_0 = C[4 * ldc + 0];
    register double _C5_0 = C[5 * ldc + 0];
    register double _C6_0 = C[6 * ldc + 0];

    for(i = 0; i < k; ++i) {
        _A0 = A[0];
        _C0_0 += B[0 * ldb] * _A0;
        _C1_0 += B[1 * ldb] * _A0;
        _C2_0 += B[2 * ldb] * _A0;
        _C3_0 += B[3 * ldb] * _A0;
        _C4_0 += B[4 * ldb] * _A0;
        _C5_0 += B[5 * ldb] * _A0;
        _C6_0 += B[6 * ldb] * _A0;

        A += lda;
        B += 1;
    }

    C[0 * ldc + 0] = _C0_0;
    C[1 * ldc + 0] = _C1_0;
    C[2 * ldc + 0] = _C2_0;
    C[3 * ldc + 0] = _C3_0;
    C[4 * ldc + 0] = _C4_0;
    C[5 * ldc + 0] = _C5_0;
    C[6 * ldc + 0] = _C6_0;
}
void micro_kernel2x7(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m128d _A0;
    register __m128d _C0_0 = _mm_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C1_0 = _mm_loadu_pd(&C[1 * ldc + 0]);
    register __m128d _C2_0 = _mm_loadu_pd(&C[2 * ldc + 0]);
    register __m128d _C3_0 = _mm_loadu_pd(&C[3 * ldc + 0]);
    register __m128d _C4_0 = _mm_loadu_pd(&C[4 * ldc + 0]);
    register __m128d _C5_0 = _mm_loadu_pd(&C[5 * ldc + 0]);
    register __m128d _C6_0 = _mm_loadu_pd(&C[6 * ldc + 0]);

    for(i = 0; i < k; ++i) {
        _A0 = _mm_loadu_pd(&A[0]);
        _C0_0 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C1_0 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C2_0 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C3_0 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C4_0 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C5_0 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C6_0 = _mm_fnmadd_pd(_mm_set1_pd(B[6 * ldb]), _A0, _C6_0);

        A += lda;
        B += 1;
    }

    _mm_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm_storeu_pd(&C[6 * ldc + 0], _C6_0);
}
void micro_kernel3x7(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m128d _A0;
    register double _A1;
    register __m128d _C0_0 = _mm_loadu_pd(&C[0 * ldc + 0]);
    register double _C0_1 = C[0 * ldc + 2];
    register __m128d _C1_0 = _mm_loadu_pd(&C[1 * ldc + 0]);
    register double _C1_1 = C[1 * ldc + 2];
    register __m128d _C2_0 = _mm_loadu_pd(&C[2 * ldc + 0]);
    register double _C2_1 = C[2 * ldc + 2];
    register __m128d _C3_0 = _mm_loadu_pd(&C[3 * ldc + 0]);
    register double _C3_1 = C[3 * ldc + 2];
    register __m128d _C4_0 = _mm_loadu_pd(&C[4 * ldc + 0]);
    register double _C4_1 = C[4 * ldc + 2];
    register __m128d _C5_0 = _mm_loadu_pd(&C[5 * ldc + 0]);
    register double _C5_1 = C[5 * ldc + 2];

    register __m128d _C6_0 = _mm_loadu_pd(&C[6 * ldc + 0]);
    register double _C6_1 = C[6 * ldc + 2];

    for(i = 0; i < k; ++i) {
        _A0 = _mm_loadu_pd(&A[0]);
        _A1 = A[2];
        _C0_0 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 += B[0 * ldb] * _A1;
        _C1_0 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 += B[1 * ldb] * _A1;
        _C2_0 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 += B[2 * ldb] * _A1;
        _C3_0 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 += B[3 * ldb] * _A1;
        _C4_0 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 += B[4 * ldb] * _A1;
        _C5_0 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 += B[5 * ldb] * _A1;

        _C6_0 = _mm_fnmadd_pd(_mm_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 += B[6 * ldb] * _A1;
        A += lda;
        B += 1;
    }

    _mm_storeu_pd(&C[0 * ldc + 0], _C0_0);
    C[0 * ldc + 2] = _C0_1;
    _mm_storeu_pd(&C[1 * ldc + 0], _C1_0);
    C[1 * ldc + 2] = _C1_1;
    _mm_storeu_pd(&C[2 * ldc + 0], _C2_0);
    C[2 * ldc + 2] = _C2_1;
    _mm_storeu_pd(&C[3 * ldc + 0], _C3_0);
    C[3 * ldc + 2] = _C3_1;
    _mm_storeu_pd(&C[4 * ldc + 0], _C4_0);
    C[4 * ldc + 2] = _C4_1;
    _mm_storeu_pd(&C[5 * ldc + 0], _C5_0);
    C[5 * ldc + 2] = _C5_1;

    _mm_storeu_pd(&C[6 * ldc + 0], _C6_0);
    C[6 * ldc + 2] = _C6_1;
}
void micro_kernel4x7(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m256d _A0;
    register __m256d _C0_0 = _mm256_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C1_0 = _mm256_loadu_pd(&C[1 * ldc + 0]);
    register __m256d _C2_0 = _mm256_loadu_pd(&C[2 * ldc + 0]);
    register __m256d _C3_0 = _mm256_loadu_pd(&C[3 * ldc + 0]);
    register __m256d _C4_0 = _mm256_loadu_pd(&C[4 * ldc + 0]);
    register __m256d _C5_0 = _mm256_loadu_pd(&C[5 * ldc + 0]);

    register __m256d _C6_0 = _mm256_loadu_pd(&C[6 * ldc + 0]);

    for(i = 0; i < k; ++i) {
        _A0 = _mm256_loadu_pd(&A[0]);
        _C0_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C1_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C2_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C3_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C4_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C5_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A0, _C5_0);

        _C6_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[6 * ldb]), _A0, _C6_0);
        A += lda;
        B += 1;
    }

    _mm256_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm256_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm256_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm256_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm256_storeu_pd(&C[5 * ldc + 0], _C5_0);

    _mm256_storeu_pd(&C[6 * ldc + 0], _C6_0);
}
void micro_kernel5x7(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m256d _A0;
    register double _A1;
    register __m256d _C0_0 = _mm256_loadu_pd(&C[0 * ldc + 0]);
    register double _C0_1 = C[0 * ldc + 4];
    register __m256d _C1_0 = _mm256_loadu_pd(&C[1 * ldc + 0]);
    register double _C1_1 = C[1 * ldc + 4];
    register __m256d _C2_0 = _mm256_loadu_pd(&C[2 * ldc + 0]);
    register double _C2_1 = C[2 * ldc + 4];
    register __m256d _C3_0 = _mm256_loadu_pd(&C[3 * ldc + 0]);
    register double _C3_1 = C[3 * ldc + 4];
    register __m256d _C4_0 = _mm256_loadu_pd(&C[4 * ldc + 0]);
    register double _C4_1 = C[4 * ldc + 4];
    register __m256d _C5_0 = _mm256_loadu_pd(&C[5 * ldc + 0]);
    register double _C5_1 = C[5 * ldc + 4];

    register __m256d _C6_0 = _mm256_loadu_pd(&C[6 * ldc + 0]);
    register double _C6_1 = C[6 * ldc + 4];

    for(i = 0; i < k; ++i) {
        _A0 = _mm256_loadu_pd(&A[0]);
        _A1 = A[4];
        _C0_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 += B[0 * ldb] * _A1;
        _C1_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 += B[1 * ldb] * _A1;
        _C2_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 += B[2 * ldb] * _A1;
        _C3_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 += B[3 * ldb] * _A1;
        _C4_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 += B[4 * ldb] * _A1;
        _C5_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 += B[5 * ldb] * _A1;

        _C6_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 += B[6 * ldb] * _A1;
        A += lda;
        B += 1;
    }

    _mm256_storeu_pd(&C[0 * ldc + 0], _C0_0);
    C[0 * ldc + 4] = _C0_1;
    _mm256_storeu_pd(&C[1 * ldc + 0], _C1_0);
    C[1 * ldc + 4] = _C1_1;
    _mm256_storeu_pd(&C[2 * ldc + 0], _C2_0);
    C[2 * ldc + 4] = _C2_1;
    _mm256_storeu_pd(&C[3 * ldc + 0], _C3_0);
    C[3 * ldc + 4] = _C3_1;
    _mm256_storeu_pd(&C[4 * ldc + 0], _C4_0);
    C[4 * ldc + 4] = _C4_1;
    _mm256_storeu_pd(&C[5 * ldc + 0], _C5_0);
    C[5 * ldc + 4] = _C5_1;

    _mm256_storeu_pd(&C[6 * ldc + 0], _C6_0);
    C[6 * ldc + 4] = _C6_1;
}
void micro_kernel6x7(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m256d _A0;
    register __m128d _A1;
    register __m256d _C0_0 = _mm256_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C0_1 = _mm_loadu_pd(&C[0 * ldc + 4]);
    register __m256d _C1_0 = _mm256_loadu_pd(&C[1 * ldc + 0]);
    register __m128d _C1_1 = _mm_loadu_pd(&C[1 * ldc + 4]);
    register __m256d _C2_0 = _mm256_loadu_pd(&C[2 * ldc + 0]);
    register __m128d _C2_1 = _mm_loadu_pd(&C[2 * ldc + 4]);
    register __m256d _C3_0 = _mm256_loadu_pd(&C[3 * ldc + 0]);
    register __m128d _C3_1 = _mm_loadu_pd(&C[3 * ldc + 4]);
    register __m256d _C4_0 = _mm256_loadu_pd(&C[4 * ldc + 0]);
    register __m128d _C4_1 = _mm_loadu_pd(&C[4 * ldc + 4]);
    register __m256d _C5_0 = _mm256_loadu_pd(&C[5 * ldc + 0]);
    register __m128d _C5_1 = _mm_loadu_pd(&C[5 * ldc + 4]);

    register __m256d _C6_0 = _mm256_loadu_pd(&C[6 * ldc + 0]);
    register __m128d _C6_1 = _mm_loadu_pd(&C[6 * ldc + 4]);

    for(i = 0; i < k; ++i) {
        _A0 = _mm256_loadu_pd(&A[0]);
        _A1 = _mm_loadu_pd(&A[4]);
        _C0_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C1_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C2_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C3_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A1, _C3_1);
        _C4_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A1, _C4_1);
        _C5_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A1, _C5_1);

        _C6_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm_fnmadd_pd(_mm_set1_pd(B[6 * ldb]), _A1, _C6_1);
        A += lda;
        B += 1;
    }

    _mm256_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[0 * ldc + 4], _C0_1);
    _mm256_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm_storeu_pd(&C[1 * ldc + 4], _C1_1);
    _mm256_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm_storeu_pd(&C[2 * ldc + 4], _C2_1);
    _mm256_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm_storeu_pd(&C[3 * ldc + 4], _C3_1);
    _mm256_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm_storeu_pd(&C[4 * ldc + 4], _C4_1);
    _mm256_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm_storeu_pd(&C[5 * ldc + 4], _C5_1);

    _mm256_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm_storeu_pd(&C[6 * ldc + 4], _C6_1);
}
void micro_kernel7x7(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m256d _A0;
    register __m128d _A1;
    register double _A2;
    register __m256d _C0_0 = _mm256_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C0_1 = _mm_loadu_pd(&C[0 * ldc + 4]);
    register double _C0_2 = C[0 * ldc + 6];
    register __m256d _C1_0 = _mm256_loadu_pd(&C[1 * ldc + 0]);
    register __m128d _C1_1 = _mm_loadu_pd(&C[1 * ldc + 4]);
    register double _C1_2 = C[1 * ldc + 6];
    register __m256d _C2_0 = _mm256_loadu_pd(&C[2 * ldc + 0]);
    register __m128d _C2_1 = _mm_loadu_pd(&C[2 * ldc + 4]);
    register double _C2_2 = C[2 * ldc + 6];
    register __m256d _C3_0 = _mm256_loadu_pd(&C[3 * ldc + 0]);
    register __m128d _C3_1 = _mm_loadu_pd(&C[3 * ldc + 4]);
    register double _C3_2 = C[3 * ldc + 6];
    register __m256d _C4_0 = _mm256_loadu_pd(&C[4 * ldc + 0]);
    register __m128d _C4_1 = _mm_loadu_pd(&C[4 * ldc + 4]);
    register double _C4_2 = C[4 * ldc + 6];
    register __m256d _C5_0 = _mm256_loadu_pd(&C[5 * ldc + 0]);
    register __m128d _C5_1 = _mm_loadu_pd(&C[5 * ldc + 4]);
    register double _C5_2 = C[5 * ldc + 6];

    register __m256d _C6_0 = _mm256_loadu_pd(&C[6 * ldc + 0]);
    register __m128d _C6_1 = _mm_loadu_pd(&C[6 * ldc + 4]);
    register double _C6_2 = C[6 * ldc + 6];

    for(i = 0; i < k; ++i) {
        _A0 = _mm256_loadu_pd(&A[0]);
        _A1 = _mm_loadu_pd(&A[4]);
        _A2 = A[6];
        _C0_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 += B[0 * ldb] * _A2;
        _C1_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C1_2 += B[1 * ldb] * _A2;
        _C2_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C2_2 += B[2 * ldb] * _A2;
        _C3_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A1, _C3_1);
        _C3_2 += B[3 * ldb] * _A2;
        _C4_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A1, _C4_1);
        _C4_2 += B[4 * ldb] * _A2;
        _C5_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A1, _C5_1);
        _C5_2 += B[5 * ldb] * _A2;

        _C6_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm_fnmadd_pd(_mm_set1_pd(B[6 * ldb]), _A1, _C6_1);
        _C6_2 += B[6 * ldb] * _A2;

        A += lda;
        B += 1;
    }

    _mm256_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[0 * ldc + 4], _C0_1);
    C[0 * ldc + 6] = _C0_2;
    _mm256_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm_storeu_pd(&C[1 * ldc + 4], _C1_1);
    C[1 * ldc + 6] = _C1_2;
    _mm256_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm_storeu_pd(&C[2 * ldc + 4], _C2_1);
    C[2 * ldc + 6] = _C2_2;
    _mm256_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm_storeu_pd(&C[3 * ldc + 4], _C3_1);
    C[3 * ldc + 6] = _C3_2;
    _mm256_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm_storeu_pd(&C[4 * ldc + 4], _C4_1);
    C[4 * ldc + 6] = _C4_2;
    _mm256_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm_storeu_pd(&C[5 * ldc + 4], _C5_1);
    C[5 * ldc + 6] = _C5_2;

    _mm256_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm_storeu_pd(&C[6 * ldc + 4], _C6_1);
    C[6 * ldc + 6] = _C6_2;
}
//=====================================================================================================================
void micro_kernel8x7(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);
    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);

    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);

        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);
}
void micro_kernel9x7(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register double _A1;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register double _C0_1 = C[0 * ldc + 8];

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);
    register double _C1_1 = C[1 * ldc + 8];

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register double _C2_1 = C[2 * ldc + 8];

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register double _C3_1 = C[3 * ldc + 8];

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register double _C4_1 = C[4 * ldc + 8];

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register double _C5_1 = C[5 * ldc + 8];

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);
    register double _C6_1 = C[6 * ldc + 8];

    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = A[8];

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 += B[0 * ldb] * _A1;

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 += B[1 * ldb] * _A1;

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 += B[2 * ldb] * _A1;

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 += B[3 * ldb] * _A1;

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 += B[4 * ldb] * _A1;

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 += B[5 * ldb] * _A1;

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 += B[6 * ldb] * _A1;
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    C[0 * ldc + 8] = _C0_1;

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    C[1 * ldc + 8] = _C1_1;

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    C[2 * ldc + 8] = _C2_1;

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    C[3 * ldc + 8] = _C3_1;

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    C[4 * ldc + 8] = _C4_1;

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    C[5 * ldc + 8] = _C5_1;

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);
    C[6 * ldc + 8] = _C6_1;
}
void micro_kernel10x7(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m128d _A1;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C0_1 = _mm_loadu_pd(&C[0 * ldc + 8]);

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m128d _C1_1 = _mm_loadu_pd(&C[1 * ldc + 8]);

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m128d _C2_1 = _mm_loadu_pd(&C[2 * ldc + 8]);

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m128d _C3_1 = _mm_loadu_pd(&C[3 * ldc + 8]);

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m128d _C4_1 = _mm_loadu_pd(&C[4 * ldc + 8]);

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register __m128d _C5_1 = _mm_loadu_pd(&C[5 * ldc + 8]);

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);
    register __m128d _C6_1 = _mm_loadu_pd(&C[6 * ldc + 8]);
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm_loadu_pd(&A[8]);

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A1, _C0_1);

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A1, _C1_1);

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A1, _C2_1);

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A1, _C3_1);

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A1, _C4_1);

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A1, _C5_1);

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm_fnmadd_pd(_mm_set1_pd(B[6 * ldb]), _A1, _C6_1);
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[0 * ldc + 8], _C0_1);

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm_storeu_pd(&C[1 * ldc + 8], _C1_1);

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm_storeu_pd(&C[2 * ldc + 8], _C2_1);

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm_storeu_pd(&C[3 * ldc + 8], _C3_1);

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm_storeu_pd(&C[4 * ldc + 8], _C4_1);

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm_storeu_pd(&C[5 * ldc + 8], _C5_1);

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm_storeu_pd(&C[6 * ldc + 8], _C6_1);
}
void micro_kernel11x7(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m128d _A1;
    register double _A2;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C0_1 = _mm_loadu_pd(&C[0 * ldc + 8]);
    register double _C0_2 = C[0 * ldc + 10];

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m128d _C1_1 = _mm_loadu_pd(&C[1 * ldc + 8]);
    register double _C1_2 = C[1 * ldc + 10];

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m128d _C2_1 = _mm_loadu_pd(&C[2 * ldc + 8]);
    register double _C2_2 = C[2 * ldc + 10];

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m128d _C3_1 = _mm_loadu_pd(&C[3 * ldc + 8]);
    register double _C3_2 = C[3 * ldc + 10];

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m128d _C4_1 = _mm_loadu_pd(&C[4 * ldc + 8]);
    register double _C4_2 = C[4 * ldc + 10];

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register __m128d _C5_1 = _mm_loadu_pd(&C[5 * ldc + 8]);
    register double _C5_2 = C[5 * ldc + 10];

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);
    register __m128d _C6_1 = _mm_loadu_pd(&C[6 * ldc + 8]);
    register double _C6_2 = C[6 * ldc + 10];
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm_loadu_pd(&A[8]);
        _A2 = A[10];

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 += B[0 * ldb] * _A2;

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C1_2 += B[1 * ldb] * _A2;

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C2_2 += B[2 * ldb] * _A2;

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A1, _C3_1);
        _C3_2 += B[3 * ldb] * _A2;

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A1, _C4_1);
        _C4_2 += B[4 * ldb] * _A2;

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A1, _C5_1);
        _C5_2 += B[5 * ldb] * _A2;

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm_fnmadd_pd(_mm_set1_pd(B[6 * ldb]), _A1, _C6_1);
        _C6_2 += B[6 * ldb] * _A2;
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[0 * ldc + 8], _C0_1);
    C[0 * ldc + 10] = _C0_2;

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm_storeu_pd(&C[1 * ldc + 8], _C1_1);
    C[1 * ldc + 10] = _C1_2;

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm_storeu_pd(&C[2 * ldc + 8], _C2_1);
    C[2 * ldc + 10] = _C2_2;

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm_storeu_pd(&C[3 * ldc + 8], _C3_1);
    C[3 * ldc + 10] = _C3_2;

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm_storeu_pd(&C[4 * ldc + 8], _C4_1);
    C[4 * ldc + 10] = _C4_2;

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm_storeu_pd(&C[5 * ldc + 8], _C5_1);
    C[5 * ldc + 10] = _C5_2;

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm_storeu_pd(&C[6 * ldc + 8], _C6_1);
    C[6 * ldc + 10] = _C6_2;
}
void micro_kernel12x7(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
     int i;
    register __m512d _A0;
    register __m256d _A1;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C0_1 = _mm256_loadu_pd(&C[0 * ldc + 8]);

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m256d _C1_1 = _mm256_loadu_pd(&C[1 * ldc + 8]);

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m256d _C2_1 = _mm256_loadu_pd(&C[2 * ldc + 8]);

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m256d _C3_1 = _mm256_loadu_pd(&C[3 * ldc + 8]);

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m256d _C4_1 = _mm256_loadu_pd(&C[4 * ldc + 8]);

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register __m256d _C5_1 = _mm256_loadu_pd(&C[5 * ldc + 8]);

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);
    register __m256d _C6_1 = _mm256_loadu_pd(&C[6 * ldc + 8]);
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm256_loadu_pd(&A[8]);

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A1, _C0_1);

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A1, _C1_1);

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A1, _C2_1);

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A1, _C3_1);

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A1, _C4_1);

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A1, _C5_1);

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[6 * ldb]), _A1, _C6_1);
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[0 * ldc + 8], _C0_1);

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm256_storeu_pd(&C[1 * ldc + 8], _C1_1);

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm256_storeu_pd(&C[2 * ldc + 8], _C2_1);

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm256_storeu_pd(&C[3 * ldc + 8], _C3_1);

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm256_storeu_pd(&C[4 * ldc + 8], _C4_1);

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm256_storeu_pd(&C[5 * ldc + 8], _C5_1);

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm256_storeu_pd(&C[6 * ldc + 8], _C6_1);
}
void micro_kernel13x7(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m256d _A1;
    register double _A2;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C0_1 = _mm256_loadu_pd(&C[0 * ldc + 8]);
    register double _C0_2 = C[0 * ldc + 12];

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m256d _C1_1 = _mm256_loadu_pd(&C[1 * ldc + 8]);
    register double _C1_2 = C[1 * ldc + 12];

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m256d _C2_1 = _mm256_loadu_pd(&C[2 * ldc + 8]);
    register double _C2_2 = C[2 * ldc + 12];

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m256d _C3_1 = _mm256_loadu_pd(&C[3 * ldc + 8]);
    register double _C3_2 = C[3 * ldc + 12];

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m256d _C4_1 = _mm256_loadu_pd(&C[4 * ldc + 8]);
    register double _C4_2 = C[4 * ldc + 12];

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register __m256d _C5_1 = _mm256_loadu_pd(&C[5 * ldc + 8]);
    register double _C5_2 = C[5 * ldc + 12];

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);
    register __m256d _C6_1 = _mm256_loadu_pd(&C[6 * ldc + 8]);
    register double _C6_2 = C[6 * ldc + 12];
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm256_loadu_pd(&A[8]);
        _A2 = A[12];

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 += B[0 * ldb] * _A2;

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C1_2 += B[1 * ldb] * _A2;

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C2_2 += B[2 * ldb] * _A2;

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A1, _C3_1);
        _C3_2 += B[3 * ldb] * _A2;

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A1, _C4_1);
        _C4_2 += B[4 * ldb] * _A2;

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A1, _C5_1);
        _C5_2 += B[5 * ldb] * _A2;

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[6 * ldb]), _A1, _C6_1);
        _C6_2 += B[6 * ldb] * _A2;
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[0 * ldc + 8], _C0_1);
    C[0 * ldc + 12] = _C0_2;

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm256_storeu_pd(&C[1 * ldc + 8], _C1_1);
    C[1 * ldc + 12] = _C1_2;

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm256_storeu_pd(&C[2 * ldc + 8], _C2_1);
    C[2 * ldc + 12] = _C2_2;

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm256_storeu_pd(&C[3 * ldc + 8], _C3_1);
    C[3 * ldc + 12] = _C3_2;

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm256_storeu_pd(&C[4 * ldc + 8], _C4_1);
    C[4 * ldc + 12] = _C4_2;

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm256_storeu_pd(&C[5 * ldc + 8], _C5_1);
    C[5 * ldc + 12] = _C5_2;

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm256_storeu_pd(&C[6 * ldc + 8], _C6_1);
    C[6 * ldc + 12] = _C6_2;
}
void micro_kernel14x7(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m256d _A1;
    register __m128d _A2;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C0_1 = _mm256_loadu_pd(&C[0 * ldc + 8]);
    register __m128d _C0_2 = _mm_loadu_pd(&C[0 * ldc + 12]);

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m256d _C1_1 = _mm256_loadu_pd(&C[1 * ldc + 8]);
    register __m128d _C1_2 = _mm_loadu_pd(&C[1 * ldc + 12]);

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m256d _C2_1 = _mm256_loadu_pd(&C[2 * ldc + 8]);
    register __m128d _C2_2 = _mm_loadu_pd(&C[2 * ldc + 12]);

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m256d _C3_1 = _mm256_loadu_pd(&C[3 * ldc + 8]);
    register __m128d _C3_2 = _mm_loadu_pd(&C[3 * ldc + 12]);

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m256d _C4_1 = _mm256_loadu_pd(&C[4 * ldc + 8]);
    register __m128d _C4_2 = _mm_loadu_pd(&C[4 * ldc + 12]);

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register __m256d _C5_1 = _mm256_loadu_pd(&C[5 * ldc + 8]);
    register __m128d _C5_2 = _mm_loadu_pd(&C[5 * ldc + 12]);

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);
    register __m256d _C6_1 = _mm256_loadu_pd(&C[6 * ldc + 8]);
    register __m128d _C6_2 = _mm_loadu_pd(&C[6 * ldc + 12]);
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm256_loadu_pd(&A[8]);
        _A2 = _mm_loadu_pd(&A[12]);

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A2, _C0_2);

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C1_2 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A2, _C1_2);

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C2_2 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A2, _C2_2);

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A1, _C3_1);
        _C3_2 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A2, _C3_2);

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A1, _C4_1);
        _C4_2 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A2, _C4_2);

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A1, _C5_1);
        _C5_2 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A2, _C5_2);

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[6 * ldb]), _A1, _C6_1);
        _C6_2 = _mm_fnmadd_pd(_mm_set1_pd(B[6 * ldb]), _A2, _C6_2);
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[0 * ldc + 8], _C0_1);
    _mm_storeu_pd(&C[0 * ldc + 12], _C0_2);

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm256_storeu_pd(&C[1 * ldc + 8], _C1_1);
    _mm_storeu_pd(&C[1 * ldc + 12], _C1_2);

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm256_storeu_pd(&C[2 * ldc + 8], _C2_1);
    _mm_storeu_pd(&C[2 * ldc + 12], _C2_2);

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm256_storeu_pd(&C[3 * ldc + 8], _C3_1);
    _mm_storeu_pd(&C[3 * ldc + 12], _C3_2);

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm256_storeu_pd(&C[4 * ldc + 8], _C4_1);
    _mm_storeu_pd(&C[4 * ldc + 12], _C4_2);

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm256_storeu_pd(&C[5 * ldc + 8], _C5_1);
    _mm_storeu_pd(&C[5 * ldc + 12], _C5_2);

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm256_storeu_pd(&C[6 * ldc + 8], _C6_1);
    _mm_storeu_pd(&C[6 * ldc + 12], _C6_2);
}
void micro_kernel15x7(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m256d _A1;
    register __m128d _A2;
    register double _A3;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C0_1 = _mm256_loadu_pd(&C[0 * ldc + 8]);
    register __m128d _C0_2 = _mm_loadu_pd(&C[0 * ldc + 12]);
    register double _C0_3 = C[0 * ldc + 14];

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m256d _C1_1 = _mm256_loadu_pd(&C[1 * ldc + 8]);
    register __m128d _C1_2 = _mm_loadu_pd(&C[1 * ldc + 12]);
    register double _C1_3 = C[1 * ldc + 14];

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m256d _C2_1 = _mm256_loadu_pd(&C[2 * ldc + 8]);
    register __m128d _C2_2 = _mm_loadu_pd(&C[2 * ldc + 12]);
    register double _C2_3 = C[2 * ldc + 14];

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m256d _C3_1 = _mm256_loadu_pd(&C[3 * ldc + 8]);
    register __m128d _C3_2 = _mm_loadu_pd(&C[3 * ldc + 12]);
    register double _C3_3 = C[3 * ldc + 14];

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m256d _C4_1 = _mm256_loadu_pd(&C[4 * ldc + 8]);
    register __m128d _C4_2 = _mm_loadu_pd(&C[4 * ldc + 12]);
    register double _C4_3 = C[4 * ldc + 14];

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register __m256d _C5_1 = _mm256_loadu_pd(&C[5 * ldc + 8]);
    register __m128d _C5_2 = _mm_loadu_pd(&C[5 * ldc + 12]);
    register double _C5_3 = C[5 * ldc + 14];

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);
    register __m256d _C6_1 = _mm256_loadu_pd(&C[6 * ldc + 8]);
    register __m128d _C6_2 = _mm_loadu_pd(&C[6 * ldc + 12]);
    register double _C6_3 = C[6 * ldc + 14];
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm256_loadu_pd(&A[8]);
        _A2 = _mm_loadu_pd(&A[12]);
        _A3 = A[14];

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A2, _C0_2);
        _C0_3 += B[0 * ldb] * _A3;

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C1_2 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A2, _C1_2);
        _C1_3 += B[1 * ldb] * _A3;

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C2_2 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A2, _C2_2);
        _C2_3 += B[2 * ldb] * _A3;

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A1, _C3_1);
        _C3_2 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A2, _C3_2);
        _C3_3 += B[3 * ldb] * _A3;

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A1, _C4_1);
        _C4_2 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A2, _C4_2);
        _C4_3 += B[4 * ldb] * _A3;

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A1, _C5_1);
        _C5_2 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A2, _C5_2);
        _C5_3 += B[5 * ldb] * _A3;

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[6 * ldb]), _A1, _C6_1);
        _C6_2 = _mm_fnmadd_pd(_mm_set1_pd(B[6 * ldb]), _A2, _C6_2);
        _C6_3 += B[6 * ldb] * _A3;
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[0 * ldc + 8], _C0_1);
    _mm_storeu_pd(&C[0 * ldc + 12], _C0_2);
    C[0 * ldc + 14] = _C0_3;

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm256_storeu_pd(&C[1 * ldc + 8], _C1_1);
    _mm_storeu_pd(&C[1 * ldc + 12], _C1_2);
    C[1 * ldc + 14] = _C1_3;

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm256_storeu_pd(&C[2 * ldc + 8], _C2_1);
    _mm_storeu_pd(&C[2 * ldc + 12], _C2_2);
    C[2 * ldc + 14] = _C2_3;

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm256_storeu_pd(&C[3 * ldc + 8], _C3_1);
    _mm_storeu_pd(&C[3 * ldc + 12], _C3_2);
    C[3 * ldc + 14] = _C3_3;

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm256_storeu_pd(&C[4 * ldc + 8], _C4_1);
    _mm_storeu_pd(&C[4 * ldc + 12], _C4_2);
    C[4 * ldc + 14] = _C4_3;

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm256_storeu_pd(&C[5 * ldc + 8], _C5_1);
    _mm_storeu_pd(&C[5 * ldc + 12], _C5_2);
    C[5 * ldc + 14] = _C5_3;

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm256_storeu_pd(&C[6 * ldc + 8], _C6_1);
    _mm_storeu_pd(&C[6 * ldc + 12], _C6_2);
    C[6 * ldc + 14] = _C6_3;
}
void micro_kernel16x7(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m512d _A1;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m512d _C0_1 = _mm512_loadu_pd(&C[0 * ldc + 8]);

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m512d _C1_1 = _mm512_loadu_pd(&C[1 * ldc + 8]);

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m512d _C2_1 = _mm512_loadu_pd(&C[2 * ldc + 8]);

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m512d _C3_1 = _mm512_loadu_pd(&C[3 * ldc + 8]);

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m512d _C4_1 = _mm512_loadu_pd(&C[4 * ldc + 8]);

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register __m512d _C5_1 = _mm512_loadu_pd(&C[5 * ldc + 8]);

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);
    register __m512d _C6_1 = _mm512_loadu_pd(&C[6 * ldc + 8]);
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm512_loadu_pd(&A[8]);

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A1, _C0_1);

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A1, _C1_1);

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A1, _C2_1);

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A1, _C3_1);

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A1, _C4_1);

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A1, _C5_1);

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A1, _C6_1);
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm512_storeu_pd(&C[0 * ldc + 8], _C0_1);

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm512_storeu_pd(&C[1 * ldc + 8], _C1_1);

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm512_storeu_pd(&C[2 * ldc + 8], _C2_1);

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm512_storeu_pd(&C[3 * ldc + 8], _C3_1);

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm512_storeu_pd(&C[4 * ldc + 8], _C4_1);

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm512_storeu_pd(&C[5 * ldc + 8], _C5_1);

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm512_storeu_pd(&C[6 * ldc + 8], _C6_1);
}
//#############################################################---------------------------------############################################
void micro_kernel1x8(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register double _A0;
    register double _C0_0 = C[0 * ldc + 0];
    register double _C1_0 = C[1 * ldc + 0];
    register double _C2_0 = C[2 * ldc + 0];
    register double _C3_0 = C[3 * ldc + 0];
    register double _C4_0 = C[4 * ldc + 0];
    register double _C5_0 = C[5 * ldc + 0];
    register double _C6_0 = C[6 * ldc + 0];
    register double _C7_0 = C[7 * ldc + 0];

    for(i = 0; i < k; ++i) {
        _A0 = A[0];
        _C0_0 += B[0 * ldb] * _A0;
        _C1_0 += B[1 * ldb] * _A0;
        _C2_0 += B[2 * ldb] * _A0;
        _C3_0 += B[3 * ldb] * _A0;
        _C4_0 += B[4 * ldb] * _A0;
        _C5_0 += B[5 * ldb] * _A0;
        _C6_0 += B[6 * ldb] * _A0;
        _C7_0 += B[7 * ldb] * _A0;

        A += lda;
        B += 1;
    }

    C[0 * ldc + 0] = _C0_0;
    C[1 * ldc + 0] = _C1_0;
    C[2 * ldc + 0] = _C2_0;
    C[3 * ldc + 0] = _C3_0;
    C[4 * ldc + 0] = _C4_0;
    C[5 * ldc + 0] = _C5_0;
    C[6 * ldc + 0] = _C6_0;
    C[7 * ldc + 0] = _C7_0;
}
void micro_kernel2x8(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m128d _A0;
    register __m128d _C0_0 = _mm_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C1_0 = _mm_loadu_pd(&C[1 * ldc + 0]);
    register __m128d _C2_0 = _mm_loadu_pd(&C[2 * ldc + 0]);
    register __m128d _C3_0 = _mm_loadu_pd(&C[3 * ldc + 0]);
    register __m128d _C4_0 = _mm_loadu_pd(&C[4 * ldc + 0]);
    register __m128d _C5_0 = _mm_loadu_pd(&C[5 * ldc + 0]);
    register __m128d _C6_0 = _mm_loadu_pd(&C[6 * ldc + 0]);
    register __m128d _C7_0 = _mm_loadu_pd(&C[7 * ldc + 0]);

    for(i = 0; i < k; ++i) {
        _A0 = _mm_loadu_pd(&A[0]);
        _C0_0 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C1_0 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C2_0 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C3_0 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C4_0 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C5_0 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C6_0 = _mm_fnmadd_pd(_mm_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C7_0 = _mm_fnmadd_pd(_mm_set1_pd(B[7 * ldb]), _A0, _C7_0);

        A += lda;
        B += 1;
    }

    _mm_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm_storeu_pd(&C[7 * ldc + 0], _C7_0);
}
void micro_kernel3x8(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m128d _A0;
    register double _A1;
    register __m128d _C0_0 = _mm_loadu_pd(&C[0 * ldc + 0]);
    register double _C0_1 = C[0 * ldc + 2];
    register __m128d _C1_0 = _mm_loadu_pd(&C[1 * ldc + 0]);
    register double _C1_1 = C[1 * ldc + 2];
    register __m128d _C2_0 = _mm_loadu_pd(&C[2 * ldc + 0]);
    register double _C2_1 = C[2 * ldc + 2];
    register __m128d _C3_0 = _mm_loadu_pd(&C[3 * ldc + 0]);
    register double _C3_1 = C[3 * ldc + 2];
    register __m128d _C4_0 = _mm_loadu_pd(&C[4 * ldc + 0]);
    register double _C4_1 = C[4 * ldc + 2];
    register __m128d _C5_0 = _mm_loadu_pd(&C[5 * ldc + 0]);
    register double _C5_1 = C[5 * ldc + 2];

    register __m128d _C6_0 = _mm_loadu_pd(&C[6 * ldc + 0]);
    register double _C6_1 = C[6 * ldc + 2];

    register __m128d _C7_0 = _mm_loadu_pd(&C[7 * ldc + 0]);
    register double _C7_1 = C[7 * ldc + 2];

    for(i = 0; i < k; ++i) {
        _A0 = _mm_loadu_pd(&A[0]);
        _A1 = A[2];
        _C0_0 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 += B[0 * ldb] * _A1;
        _C1_0 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 += B[1 * ldb] * _A1;
        _C2_0 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 += B[2 * ldb] * _A1;
        _C3_0 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 += B[3 * ldb] * _A1;
        _C4_0 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 += B[4 * ldb] * _A1;
        _C5_0 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 += B[5 * ldb] * _A1;

        _C6_0 = _mm_fnmadd_pd(_mm_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 += B[6 * ldb] * _A1;

        _C7_0 = _mm_fnmadd_pd(_mm_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 += B[7 * ldb] * _A1;

        A += lda;
        B += 1;
    }

    _mm_storeu_pd(&C[0 * ldc + 0], _C0_0);
    C[0 * ldc + 2] = _C0_1;
    _mm_storeu_pd(&C[1 * ldc + 0], _C1_0);
    C[1 * ldc + 2] = _C1_1;
    _mm_storeu_pd(&C[2 * ldc + 0], _C2_0);
    C[2 * ldc + 2] = _C2_1;
    _mm_storeu_pd(&C[3 * ldc + 0], _C3_0);
    C[3 * ldc + 2] = _C3_1;
    _mm_storeu_pd(&C[4 * ldc + 0], _C4_0);
    C[4 * ldc + 2] = _C4_1;
    _mm_storeu_pd(&C[5 * ldc + 0], _C5_0);
    C[5 * ldc + 2] = _C5_1;

    _mm_storeu_pd(&C[6 * ldc + 0], _C6_0);
    C[6 * ldc + 2] = _C6_1;

    _mm_storeu_pd(&C[7 * ldc + 0], _C7_0);
    C[7 * ldc + 2] = _C7_1;
}
void micro_kernel4x8(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m256d _A0;
    register __m256d _C0_0 = _mm256_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C1_0 = _mm256_loadu_pd(&C[1 * ldc + 0]);
    register __m256d _C2_0 = _mm256_loadu_pd(&C[2 * ldc + 0]);
    register __m256d _C3_0 = _mm256_loadu_pd(&C[3 * ldc + 0]);
    register __m256d _C4_0 = _mm256_loadu_pd(&C[4 * ldc + 0]);
    register __m256d _C5_0 = _mm256_loadu_pd(&C[5 * ldc + 0]);

    register __m256d _C6_0 = _mm256_loadu_pd(&C[6 * ldc + 0]);

    register __m256d _C7_0 = _mm256_loadu_pd(&C[7 * ldc + 0]);

    for(i = 0; i < k; ++i) {
        _A0 = _mm256_loadu_pd(&A[0]);
        _C0_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C1_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C2_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C3_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C4_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C5_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A0, _C5_0);

        _C6_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[6 * ldb]), _A0, _C6_0);

        _C7_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[7 * ldb]), _A0, _C7_0);

        A += lda;
        B += 1;
    }

    _mm256_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm256_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm256_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm256_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm256_storeu_pd(&C[5 * ldc + 0], _C5_0);

    _mm256_storeu_pd(&C[6 * ldc + 0], _C6_0);

    _mm256_storeu_pd(&C[7 * ldc + 0], _C7_0);
}
void micro_kernel5x8(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m256d _A0;
    register double _A1;
    register __m256d _C0_0 = _mm256_loadu_pd(&C[0 * ldc + 0]);
    register double _C0_1 = C[0 * ldc + 4];
    register __m256d _C1_0 = _mm256_loadu_pd(&C[1 * ldc + 0]);
    register double _C1_1 = C[1 * ldc + 4];
    register __m256d _C2_0 = _mm256_loadu_pd(&C[2 * ldc + 0]);
    register double _C2_1 = C[2 * ldc + 4];
    register __m256d _C3_0 = _mm256_loadu_pd(&C[3 * ldc + 0]);
    register double _C3_1 = C[3 * ldc + 4];
    register __m256d _C4_0 = _mm256_loadu_pd(&C[4 * ldc + 0]);
    register double _C4_1 = C[4 * ldc + 4];
    register __m256d _C5_0 = _mm256_loadu_pd(&C[5 * ldc + 0]);
    register double _C5_1 = C[5 * ldc + 4];

    register __m256d _C6_0 = _mm256_loadu_pd(&C[6 * ldc + 0]);
    register double _C6_1 = C[6 * ldc + 4];

    register __m256d _C7_0 = _mm256_loadu_pd(&C[7 * ldc + 0]);
    register double _C7_1 = C[7 * ldc + 4];

    for(i = 0; i < k; ++i) {
        _A0 = _mm256_loadu_pd(&A[0]);
        _A1 = A[4];
        _C0_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 += B[0 * ldb] * _A1;
        _C1_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 += B[1 * ldb] * _A1;
        _C2_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 += B[2 * ldb] * _A1;
        _C3_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 += B[3 * ldb] * _A1;
        _C4_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 += B[4 * ldb] * _A1;
        _C5_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 += B[5 * ldb] * _A1;

        _C6_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 += B[6 * ldb] * _A1;

        _C7_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 += B[7 * ldb] * _A1;

        A += lda;
        B += 1;
    }

    _mm256_storeu_pd(&C[0 * ldc + 0], _C0_0);
    C[0 * ldc + 4] = _C0_1;
    _mm256_storeu_pd(&C[1 * ldc + 0], _C1_0);
    C[1 * ldc + 4] = _C1_1;
    _mm256_storeu_pd(&C[2 * ldc + 0], _C2_0);
    C[2 * ldc + 4] = _C2_1;
    _mm256_storeu_pd(&C[3 * ldc + 0], _C3_0);
    C[3 * ldc + 4] = _C3_1;
    _mm256_storeu_pd(&C[4 * ldc + 0], _C4_0);
    C[4 * ldc + 4] = _C4_1;
    _mm256_storeu_pd(&C[5 * ldc + 0], _C5_0);
    C[5 * ldc + 4] = _C5_1;

    _mm256_storeu_pd(&C[6 * ldc + 0], _C6_0);
    C[6 * ldc + 4] = _C6_1;

    _mm256_storeu_pd(&C[7 * ldc + 0], _C7_0);
    C[7 * ldc + 4] = _C7_1;
}
void micro_kernel6x8(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m256d _A0;
    register __m128d _A1;
    register __m256d _C0_0 = _mm256_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C0_1 = _mm_loadu_pd(&C[0 * ldc + 4]);
    register __m256d _C1_0 = _mm256_loadu_pd(&C[1 * ldc + 0]);
    register __m128d _C1_1 = _mm_loadu_pd(&C[1 * ldc + 4]);
    register __m256d _C2_0 = _mm256_loadu_pd(&C[2 * ldc + 0]);
    register __m128d _C2_1 = _mm_loadu_pd(&C[2 * ldc + 4]);
    register __m256d _C3_0 = _mm256_loadu_pd(&C[3 * ldc + 0]);
    register __m128d _C3_1 = _mm_loadu_pd(&C[3 * ldc + 4]);
    register __m256d _C4_0 = _mm256_loadu_pd(&C[4 * ldc + 0]);
    register __m128d _C4_1 = _mm_loadu_pd(&C[4 * ldc + 4]);
    register __m256d _C5_0 = _mm256_loadu_pd(&C[5 * ldc + 0]);
    register __m128d _C5_1 = _mm_loadu_pd(&C[5 * ldc + 4]);

    register __m256d _C6_0 = _mm256_loadu_pd(&C[6 * ldc + 0]);
    register __m128d _C6_1 = _mm_loadu_pd(&C[6 * ldc + 4]);

    register __m256d _C7_0 = _mm256_loadu_pd(&C[7 * ldc + 0]);
    register __m128d _C7_1 = _mm_loadu_pd(&C[7 * ldc + 4]);

    for(i = 0; i < k; ++i) {
        _A0 = _mm256_loadu_pd(&A[0]);
        _A1 = _mm_loadu_pd(&A[4]);
        _C0_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C1_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C2_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C3_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A1, _C3_1);
        _C4_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A1, _C4_1);
        _C5_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A1, _C5_1);

        _C6_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm_fnmadd_pd(_mm_set1_pd(B[6 * ldb]), _A1, _C6_1);

        _C7_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 = _mm_fnmadd_pd(_mm_set1_pd(B[7 * ldb]), _A1, _C7_1);
        A += lda;
        B += 1;
    }

    _mm256_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[0 * ldc + 4], _C0_1);
    _mm256_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm_storeu_pd(&C[1 * ldc + 4], _C1_1);
    _mm256_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm_storeu_pd(&C[2 * ldc + 4], _C2_1);
    _mm256_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm_storeu_pd(&C[3 * ldc + 4], _C3_1);
    _mm256_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm_storeu_pd(&C[4 * ldc + 4], _C4_1);
    _mm256_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm_storeu_pd(&C[5 * ldc + 4], _C5_1);

    _mm256_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm_storeu_pd(&C[6 * ldc + 4], _C6_1);

    _mm256_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm_storeu_pd(&C[7 * ldc + 4], _C7_1);
}
void micro_kernel7x8(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m256d _A0;
    register __m128d _A1;
    register double _A2;
    register __m256d _C0_0 = _mm256_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C0_1 = _mm_loadu_pd(&C[0 * ldc + 4]);
    register double _C0_2 = C[0 * ldc + 6];
    register __m256d _C1_0 = _mm256_loadu_pd(&C[1 * ldc + 0]);
    register __m128d _C1_1 = _mm_loadu_pd(&C[1 * ldc + 4]);
    register double _C1_2 = C[1 * ldc + 6];
    register __m256d _C2_0 = _mm256_loadu_pd(&C[2 * ldc + 0]);
    register __m128d _C2_1 = _mm_loadu_pd(&C[2 * ldc + 4]);
    register double _C2_2 = C[2 * ldc + 6];
    register __m256d _C3_0 = _mm256_loadu_pd(&C[3 * ldc + 0]);
    register __m128d _C3_1 = _mm_loadu_pd(&C[3 * ldc + 4]);
    register double _C3_2 = C[3 * ldc + 6];
    register __m256d _C4_0 = _mm256_loadu_pd(&C[4 * ldc + 0]);
    register __m128d _C4_1 = _mm_loadu_pd(&C[4 * ldc + 4]);
    register double _C4_2 = C[4 * ldc + 6];
    register __m256d _C5_0 = _mm256_loadu_pd(&C[5 * ldc + 0]);
    register __m128d _C5_1 = _mm_loadu_pd(&C[5 * ldc + 4]);
    register double _C5_2 = C[5 * ldc + 6];

    register __m256d _C6_0 = _mm256_loadu_pd(&C[6 * ldc + 0]);
    register __m128d _C6_1 = _mm_loadu_pd(&C[6 * ldc + 4]);
    register double _C6_2 = C[6 * ldc + 6];

    register __m256d _C7_0 = _mm256_loadu_pd(&C[7 * ldc + 0]);
    register __m128d _C7_1 = _mm_loadu_pd(&C[7 * ldc + 4]);
    register double _C7_2 = C[7 * ldc + 6];

    for(i = 0; i < k; ++i) {
        _A0 = _mm256_loadu_pd(&A[0]);
        _A1 = _mm_loadu_pd(&A[4]);
        _A2 = A[6];
        _C0_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 += B[0 * ldb] * _A2;
        _C1_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C1_2 += B[1 * ldb] * _A2;
        _C2_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C2_2 += B[2 * ldb] * _A2;
        _C3_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A1, _C3_1);
        _C3_2 += B[3 * ldb] * _A2;
        _C4_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A1, _C4_1);
        _C4_2 += B[4 * ldb] * _A2;
        _C5_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A1, _C5_1);
        _C5_2 += B[5 * ldb] * _A2;

        _C6_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm_fnmadd_pd(_mm_set1_pd(B[6 * ldb]), _A1, _C6_1);
        _C6_2 += B[6 * ldb] * _A2;

        _C7_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 = _mm_fnmadd_pd(_mm_set1_pd(B[7 * ldb]), _A1, _C7_1);
        _C7_2 += B[7 * ldb] * _A2;

        A += lda;
        B += 1;
    }

    _mm256_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[0 * ldc + 4], _C0_1);
    C[0 * ldc + 6] = _C0_2;
    _mm256_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm_storeu_pd(&C[1 * ldc + 4], _C1_1);
    C[1 * ldc + 6] = _C1_2;
    _mm256_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm_storeu_pd(&C[2 * ldc + 4], _C2_1);
    C[2 * ldc + 6] = _C2_2;
    _mm256_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm_storeu_pd(&C[3 * ldc + 4], _C3_1);
    C[3 * ldc + 6] = _C3_2;
    _mm256_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm_storeu_pd(&C[4 * ldc + 4], _C4_1);
    C[4 * ldc + 6] = _C4_2;
    _mm256_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm_storeu_pd(&C[5 * ldc + 4], _C5_1);
    C[5 * ldc + 6] = _C5_2;

    _mm256_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm_storeu_pd(&C[6 * ldc + 4], _C6_1);
    C[6 * ldc + 6] = _C6_2;

    _mm256_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm_storeu_pd(&C[7 * ldc + 4], _C7_1);
    C[7 * ldc + 6] = _C7_2;
}
//=====================================================================================================================
void micro_kernel8x8(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);
    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);

    register __m512d _C7_0 = _mm512_loadu_pd(&C[7 * ldc + 0]);

    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);

        _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A0, _C7_0);

        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);

    _mm512_storeu_pd(&C[7 * ldc + 0], _C7_0);
}
void micro_kernel9x8(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register double _A1;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register double _C0_1 = C[0 * ldc + 8];

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);
    register double _C1_1 = C[1 * ldc + 8];

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register double _C2_1 = C[2 * ldc + 8];

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register double _C3_1 = C[3 * ldc + 8];

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register double _C4_1 = C[4 * ldc + 8];

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register double _C5_1 = C[5 * ldc + 8];

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);
    register double _C6_1 = C[6 * ldc + 8];

    register __m512d _C7_0 = _mm512_loadu_pd(&C[7 * ldc + 0]);
    register double _C7_1 = C[7 * ldc + 8];

    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = A[8];

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 += B[0 * ldb] * _A1;

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 += B[1 * ldb] * _A1;

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 += B[2 * ldb] * _A1;

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 += B[3 * ldb] * _A1;

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 += B[4 * ldb] * _A1;

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 += B[5 * ldb] * _A1;

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 += B[6 * ldb] * _A1;
        
        _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 += B[7 * ldb] * _A1;

        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    C[0 * ldc + 8] = _C0_1;

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    C[1 * ldc + 8] = _C1_1;

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    C[2 * ldc + 8] = _C2_1;

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    C[3 * ldc + 8] = _C3_1;

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    C[4 * ldc + 8] = _C4_1;

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    C[5 * ldc + 8] = _C5_1;

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);
    C[6 * ldc + 8] = _C6_1;

    _mm512_storeu_pd(&C[7 * ldc + 0], _C7_0);
    C[7 * ldc + 8] = _C7_1;
}
void micro_kernel10x8(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m128d _A1;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C0_1 = _mm_loadu_pd(&C[0 * ldc + 8]);

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m128d _C1_1 = _mm_loadu_pd(&C[1 * ldc + 8]);

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m128d _C2_1 = _mm_loadu_pd(&C[2 * ldc + 8]);

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m128d _C3_1 = _mm_loadu_pd(&C[3 * ldc + 8]);

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m128d _C4_1 = _mm_loadu_pd(&C[4 * ldc + 8]);

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register __m128d _C5_1 = _mm_loadu_pd(&C[5 * ldc + 8]);

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);
    register __m128d _C6_1 = _mm_loadu_pd(&C[6 * ldc + 8]);

    register __m512d _C7_0 = _mm512_loadu_pd(&C[7 * ldc + 0]);
    register __m128d _C7_1 = _mm_loadu_pd(&C[7 * ldc + 8]);
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm_loadu_pd(&A[8]);

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A1, _C0_1);

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A1, _C1_1);

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A1, _C2_1);

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A1, _C3_1);

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A1, _C4_1);

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A1, _C5_1);

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm_fnmadd_pd(_mm_set1_pd(B[6 * ldb]), _A1, _C6_1);

        _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 = _mm_fnmadd_pd(_mm_set1_pd(B[7 * ldb]), _A1, _C7_1);
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[0 * ldc + 8], _C0_1);

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm_storeu_pd(&C[1 * ldc + 8], _C1_1);

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm_storeu_pd(&C[2 * ldc + 8], _C2_1);

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm_storeu_pd(&C[3 * ldc + 8], _C3_1);

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm_storeu_pd(&C[4 * ldc + 8], _C4_1);

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm_storeu_pd(&C[5 * ldc + 8], _C5_1);

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm_storeu_pd(&C[6 * ldc + 8], _C6_1);

    _mm512_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm_storeu_pd(&C[7 * ldc + 8], _C7_1);
}
void micro_kernel11x8(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m128d _A1;
    register double _A2;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C0_1 = _mm_loadu_pd(&C[0 * ldc + 8]);
    register double _C0_2 = C[0 * ldc + 10];

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m128d _C1_1 = _mm_loadu_pd(&C[1 * ldc + 8]);
    register double _C1_2 = C[1 * ldc + 10];

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m128d _C2_1 = _mm_loadu_pd(&C[2 * ldc + 8]);
    register double _C2_2 = C[2 * ldc + 10];

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m128d _C3_1 = _mm_loadu_pd(&C[3 * ldc + 8]);
    register double _C3_2 = C[3 * ldc + 10];

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m128d _C4_1 = _mm_loadu_pd(&C[4 * ldc + 8]);
    register double _C4_2 = C[4 * ldc + 10];

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register __m128d _C5_1 = _mm_loadu_pd(&C[5 * ldc + 8]);
    register double _C5_2 = C[5 * ldc + 10];

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);
    register __m128d _C6_1 = _mm_loadu_pd(&C[6 * ldc + 8]);
    register double _C6_2 = C[6 * ldc + 10];

    register __m512d _C7_0 = _mm512_loadu_pd(&C[7 * ldc + 0]);
    register __m128d _C7_1 = _mm_loadu_pd(&C[7 * ldc + 8]);
    register double _C7_2 = C[7 * ldc + 10];
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm_loadu_pd(&A[8]);
        _A2 = A[10];

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 += B[0 * ldb] * _A2;

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C1_2 += B[1 * ldb] * _A2;

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C2_2 += B[2 * ldb] * _A2;

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A1, _C3_1);
        _C3_2 += B[3 * ldb] * _A2;

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A1, _C4_1);
        _C4_2 += B[4 * ldb] * _A2;

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A1, _C5_1);
        _C5_2 += B[5 * ldb] * _A2;

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm_fnmadd_pd(_mm_set1_pd(B[6 * ldb]), _A1, _C6_1);
        _C6_2 += B[6 * ldb] * _A2;

        _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 = _mm_fnmadd_pd(_mm_set1_pd(B[7 * ldb]), _A1, _C7_1);
        _C7_2 += B[7 * ldb] * _A2;
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[0 * ldc + 8], _C0_1);
    C[0 * ldc + 10] = _C0_2;

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm_storeu_pd(&C[1 * ldc + 8], _C1_1);
    C[1 * ldc + 10] = _C1_2;

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm_storeu_pd(&C[2 * ldc + 8], _C2_1);
    C[2 * ldc + 10] = _C2_2;

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm_storeu_pd(&C[3 * ldc + 8], _C3_1);
    C[3 * ldc + 10] = _C3_2;

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm_storeu_pd(&C[4 * ldc + 8], _C4_1);
    C[4 * ldc + 10] = _C4_2;

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm_storeu_pd(&C[5 * ldc + 8], _C5_1);
    C[5 * ldc + 10] = _C5_2;

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm_storeu_pd(&C[6 * ldc + 8], _C6_1);
    C[6 * ldc + 10] = _C6_2;

    _mm512_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm_storeu_pd(&C[7 * ldc + 8], _C7_1);
    C[7 * ldc + 10] = _C7_2;
}
void micro_kernel12x8(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
     int i;
    register __m512d _A0;
    register __m256d _A1;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C0_1 = _mm256_loadu_pd(&C[0 * ldc + 8]);

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m256d _C1_1 = _mm256_loadu_pd(&C[1 * ldc + 8]);

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m256d _C2_1 = _mm256_loadu_pd(&C[2 * ldc + 8]);

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m256d _C3_1 = _mm256_loadu_pd(&C[3 * ldc + 8]);

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m256d _C4_1 = _mm256_loadu_pd(&C[4 * ldc + 8]);

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register __m256d _C5_1 = _mm256_loadu_pd(&C[5 * ldc + 8]);

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);
    register __m256d _C6_1 = _mm256_loadu_pd(&C[6 * ldc + 8]);

    register __m512d _C7_0 = _mm512_loadu_pd(&C[7 * ldc + 0]);
    register __m256d _C7_1 = _mm256_loadu_pd(&C[7 * ldc + 8]);
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm256_loadu_pd(&A[8]);

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A1, _C0_1);

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A1, _C1_1);

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A1, _C2_1);

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A1, _C3_1);

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A1, _C4_1);

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A1, _C5_1);

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[6 * ldb]), _A1, _C6_1);

        _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[7 * ldb]), _A1, _C7_1);
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[0 * ldc + 8], _C0_1);

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm256_storeu_pd(&C[1 * ldc + 8], _C1_1);

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm256_storeu_pd(&C[2 * ldc + 8], _C2_1);

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm256_storeu_pd(&C[3 * ldc + 8], _C3_1);

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm256_storeu_pd(&C[4 * ldc + 8], _C4_1);

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm256_storeu_pd(&C[5 * ldc + 8], _C5_1);

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm256_storeu_pd(&C[6 * ldc + 8], _C6_1);

    _mm512_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm256_storeu_pd(&C[7 * ldc + 8], _C7_1);
}
void micro_kernel13x8(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m256d _A1;
    register double _A2;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C0_1 = _mm256_loadu_pd(&C[0 * ldc + 8]);
    register double _C0_2 = C[0 * ldc + 12];

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m256d _C1_1 = _mm256_loadu_pd(&C[1 * ldc + 8]);
    register double _C1_2 = C[1 * ldc + 12];

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m256d _C2_1 = _mm256_loadu_pd(&C[2 * ldc + 8]);
    register double _C2_2 = C[2 * ldc + 12];

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m256d _C3_1 = _mm256_loadu_pd(&C[3 * ldc + 8]);
    register double _C3_2 = C[3 * ldc + 12];

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m256d _C4_1 = _mm256_loadu_pd(&C[4 * ldc + 8]);
    register double _C4_2 = C[4 * ldc + 12];

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register __m256d _C5_1 = _mm256_loadu_pd(&C[5 * ldc + 8]);
    register double _C5_2 = C[5 * ldc + 12];

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);
    register __m256d _C6_1 = _mm256_loadu_pd(&C[6 * ldc + 8]);
    register double _C6_2 = C[6 * ldc + 12];

    register __m512d _C7_0 = _mm512_loadu_pd(&C[7 * ldc + 0]);
    register __m256d _C7_1 = _mm256_loadu_pd(&C[7 * ldc + 8]);
    register double _C7_2 = C[7 * ldc + 12];
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm256_loadu_pd(&A[8]);
        _A2 = A[12];

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 += B[0 * ldb] * _A2;

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C1_2 += B[1 * ldb] * _A2;

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C2_2 += B[2 * ldb] * _A2;

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A1, _C3_1);
        _C3_2 += B[3 * ldb] * _A2;

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A1, _C4_1);
        _C4_2 += B[4 * ldb] * _A2;

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A1, _C5_1);
        _C5_2 += B[5 * ldb] * _A2;

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[6 * ldb]), _A1, _C6_1);
        _C6_2 += B[6 * ldb] * _A2;

        _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[7 * ldb]), _A1, _C7_1);
        _C7_2 += B[7 * ldb] * _A2;
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[0 * ldc + 8], _C0_1);
    C[0 * ldc + 12] = _C0_2;

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm256_storeu_pd(&C[1 * ldc + 8], _C1_1);
    C[1 * ldc + 12] = _C1_2;

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm256_storeu_pd(&C[2 * ldc + 8], _C2_1);
    C[2 * ldc + 12] = _C2_2;

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm256_storeu_pd(&C[3 * ldc + 8], _C3_1);
    C[3 * ldc + 12] = _C3_2;

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm256_storeu_pd(&C[4 * ldc + 8], _C4_1);
    C[4 * ldc + 12] = _C4_2;

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm256_storeu_pd(&C[5 * ldc + 8], _C5_1);
    C[5 * ldc + 12] = _C5_2;

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm256_storeu_pd(&C[6 * ldc + 8], _C6_1);
    C[6 * ldc + 12] = _C6_2;

    _mm512_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm256_storeu_pd(&C[7 * ldc + 8], _C7_1);
    C[7 * ldc + 12] = _C7_2;
}
void micro_kernel14x8(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m256d _A1;
    register __m128d _A2;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C0_1 = _mm256_loadu_pd(&C[0 * ldc + 8]);
    register __m128d _C0_2 = _mm_loadu_pd(&C[0 * ldc + 12]);

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m256d _C1_1 = _mm256_loadu_pd(&C[1 * ldc + 8]);
    register __m128d _C1_2 = _mm_loadu_pd(&C[1 * ldc + 12]);

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m256d _C2_1 = _mm256_loadu_pd(&C[2 * ldc + 8]);
    register __m128d _C2_2 = _mm_loadu_pd(&C[2 * ldc + 12]);

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m256d _C3_1 = _mm256_loadu_pd(&C[3 * ldc + 8]);
    register __m128d _C3_2 = _mm_loadu_pd(&C[3 * ldc + 12]);

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m256d _C4_1 = _mm256_loadu_pd(&C[4 * ldc + 8]);
    register __m128d _C4_2 = _mm_loadu_pd(&C[4 * ldc + 12]);

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register __m256d _C5_1 = _mm256_loadu_pd(&C[5 * ldc + 8]);
    register __m128d _C5_2 = _mm_loadu_pd(&C[5 * ldc + 12]);

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);
    register __m256d _C6_1 = _mm256_loadu_pd(&C[6 * ldc + 8]);
    register __m128d _C6_2 = _mm_loadu_pd(&C[6 * ldc + 12]);

    register __m512d _C7_0 = _mm512_loadu_pd(&C[7 * ldc + 0]);
    register __m256d _C7_1 = _mm256_loadu_pd(&C[7 * ldc + 8]);
    register __m128d _C7_2 = _mm_loadu_pd(&C[7 * ldc + 12]);
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm256_loadu_pd(&A[8]);
        _A2 = _mm_loadu_pd(&A[12]);

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A2, _C0_2);

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C1_2 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A2, _C1_2);

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C2_2 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A2, _C2_2);

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A1, _C3_1);
        _C3_2 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A2, _C3_2);

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A1, _C4_1);
        _C4_2 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A2, _C4_2);

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A1, _C5_1);
        _C5_2 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A2, _C5_2);

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[6 * ldb]), _A1, _C6_1);
        _C6_2 = _mm_fnmadd_pd(_mm_set1_pd(B[6 * ldb]), _A2, _C6_2);

        _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[7 * ldb]), _A1, _C7_1);
        _C7_2 = _mm_fnmadd_pd(_mm_set1_pd(B[7 * ldb]), _A2, _C7_2);
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[0 * ldc + 8], _C0_1);
    _mm_storeu_pd(&C[0 * ldc + 12], _C0_2);

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm256_storeu_pd(&C[1 * ldc + 8], _C1_1);
    _mm_storeu_pd(&C[1 * ldc + 12], _C1_2);

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm256_storeu_pd(&C[2 * ldc + 8], _C2_1);
    _mm_storeu_pd(&C[2 * ldc + 12], _C2_2);

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm256_storeu_pd(&C[3 * ldc + 8], _C3_1);
    _mm_storeu_pd(&C[3 * ldc + 12], _C3_2);

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm256_storeu_pd(&C[4 * ldc + 8], _C4_1);
    _mm_storeu_pd(&C[4 * ldc + 12], _C4_2);

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm256_storeu_pd(&C[5 * ldc + 8], _C5_1);
    _mm_storeu_pd(&C[5 * ldc + 12], _C5_2);

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm256_storeu_pd(&C[6 * ldc + 8], _C6_1);
    _mm_storeu_pd(&C[6 * ldc + 12], _C6_2);

    _mm512_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm256_storeu_pd(&C[7 * ldc + 8], _C7_1);
    _mm_storeu_pd(&C[7 * ldc + 12], _C7_2);
}
void micro_kernel15x8(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m256d _A1;
    register __m128d _A2;
    register double _A3;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C0_1 = _mm256_loadu_pd(&C[0 * ldc + 8]);
    register __m128d _C0_2 = _mm_loadu_pd(&C[0 * ldc + 12]);
    register double _C0_3 = C[0 * ldc + 14];

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m256d _C1_1 = _mm256_loadu_pd(&C[1 * ldc + 8]);
    register __m128d _C1_2 = _mm_loadu_pd(&C[1 * ldc + 12]);
    register double _C1_3 = C[1 * ldc + 14];

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m256d _C2_1 = _mm256_loadu_pd(&C[2 * ldc + 8]);
    register __m128d _C2_2 = _mm_loadu_pd(&C[2 * ldc + 12]);
    register double _C2_3 = C[2 * ldc + 14];

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m256d _C3_1 = _mm256_loadu_pd(&C[3 * ldc + 8]);
    register __m128d _C3_2 = _mm_loadu_pd(&C[3 * ldc + 12]);
    register double _C3_3 = C[3 * ldc + 14];

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m256d _C4_1 = _mm256_loadu_pd(&C[4 * ldc + 8]);
    register __m128d _C4_2 = _mm_loadu_pd(&C[4 * ldc + 12]);
    register double _C4_3 = C[4 * ldc + 14];

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register __m256d _C5_1 = _mm256_loadu_pd(&C[5 * ldc + 8]);
    register __m128d _C5_2 = _mm_loadu_pd(&C[5 * ldc + 12]);
    register double _C5_3 = C[5 * ldc + 14];

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);
    register __m256d _C6_1 = _mm256_loadu_pd(&C[6 * ldc + 8]);
    register __m128d _C6_2 = _mm_loadu_pd(&C[6 * ldc + 12]);
    register double _C6_3 = C[6 * ldc + 14];

    register __m512d _C7_0 = _mm512_loadu_pd(&C[7 * ldc + 0]);
    register __m256d _C7_1 = _mm256_loadu_pd(&C[7 * ldc + 8]);
    register __m128d _C7_2 = _mm_loadu_pd(&C[7 * ldc + 12]);
    register double _C7_3 = C[7 * ldc + 14];
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm256_loadu_pd(&A[8]);
        _A2 = _mm_loadu_pd(&A[12]);
        _A3 = A[14];

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A2, _C0_2);
        _C0_3 += B[0 * ldb] * _A3;

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C1_2 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A2, _C1_2);
        _C1_3 += B[1 * ldb] * _A3;

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C2_2 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A2, _C2_2);
        _C2_3 += B[2 * ldb] * _A3;

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A1, _C3_1);
        _C3_2 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A2, _C3_2);
        _C3_3 += B[3 * ldb] * _A3;

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A1, _C4_1);
        _C4_2 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A2, _C4_2);
        _C4_3 += B[4 * ldb] * _A3;

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A1, _C5_1);
        _C5_2 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A2, _C5_2);
        _C5_3 += B[5 * ldb] * _A3;

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[6 * ldb]), _A1, _C6_1);
        _C6_2 = _mm_fnmadd_pd(_mm_set1_pd(B[6 * ldb]), _A2, _C6_2);
        _C6_3 += B[6 * ldb] * _A3;

        _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[7 * ldb]), _A1, _C7_1);
        _C7_2 = _mm_fnmadd_pd(_mm_set1_pd(B[7 * ldb]), _A2, _C7_2);
        _C7_3 += B[7 * ldb] * _A3;
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[0 * ldc + 8], _C0_1);
    _mm_storeu_pd(&C[0 * ldc + 12], _C0_2);
    C[0 * ldc + 14] = _C0_3;

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm256_storeu_pd(&C[1 * ldc + 8], _C1_1);
    _mm_storeu_pd(&C[1 * ldc + 12], _C1_2);
    C[1 * ldc + 14] = _C1_3;

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm256_storeu_pd(&C[2 * ldc + 8], _C2_1);
    _mm_storeu_pd(&C[2 * ldc + 12], _C2_2);
    C[2 * ldc + 14] = _C2_3;

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm256_storeu_pd(&C[3 * ldc + 8], _C3_1);
    _mm_storeu_pd(&C[3 * ldc + 12], _C3_2);
    C[3 * ldc + 14] = _C3_3;

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm256_storeu_pd(&C[4 * ldc + 8], _C4_1);
    _mm_storeu_pd(&C[4 * ldc + 12], _C4_2);
    C[4 * ldc + 14] = _C4_3;

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm256_storeu_pd(&C[5 * ldc + 8], _C5_1);
    _mm_storeu_pd(&C[5 * ldc + 12], _C5_2);
    C[5 * ldc + 14] = _C5_3;

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm256_storeu_pd(&C[6 * ldc + 8], _C6_1);
    _mm_storeu_pd(&C[6 * ldc + 12], _C6_2);
    C[6 * ldc + 14] = _C6_3;

    _mm512_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm256_storeu_pd(&C[7 * ldc + 8], _C7_1);
    _mm_storeu_pd(&C[7 * ldc + 12], _C7_2);
    C[7 * ldc + 14] = _C7_3;
}
void micro_kernel16x8(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m512d _A1;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m512d _C0_1 = _mm512_loadu_pd(&C[0 * ldc + 8]);

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m512d _C1_1 = _mm512_loadu_pd(&C[1 * ldc + 8]);

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m512d _C2_1 = _mm512_loadu_pd(&C[2 * ldc + 8]);

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m512d _C3_1 = _mm512_loadu_pd(&C[3 * ldc + 8]);

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m512d _C4_1 = _mm512_loadu_pd(&C[4 * ldc + 8]);

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register __m512d _C5_1 = _mm512_loadu_pd(&C[5 * ldc + 8]);

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);
    register __m512d _C6_1 = _mm512_loadu_pd(&C[6 * ldc + 8]);

    register __m512d _C7_0 = _mm512_loadu_pd(&C[7 * ldc + 0]);
    register __m512d _C7_1 = _mm512_loadu_pd(&C[7 * ldc + 8]);
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm512_loadu_pd(&A[8]);

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A1, _C0_1);

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A1, _C1_1);

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A1, _C2_1);

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A1, _C3_1);

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A1, _C4_1);

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A1, _C5_1);

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A1, _C6_1);

        _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A1, _C7_1);
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm512_storeu_pd(&C[0 * ldc + 8], _C0_1);

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm512_storeu_pd(&C[1 * ldc + 8], _C1_1);

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm512_storeu_pd(&C[2 * ldc + 8], _C2_1);

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm512_storeu_pd(&C[3 * ldc + 8], _C3_1);

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm512_storeu_pd(&C[4 * ldc + 8], _C4_1);

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm512_storeu_pd(&C[5 * ldc + 8], _C5_1);

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm512_storeu_pd(&C[6 * ldc + 8], _C6_1);

    _mm512_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm512_storeu_pd(&C[7 * ldc + 8], _C7_1);
}
//#############################################################========================================############################################
void micro_kernel1x9(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register double _A0;
    register double _C0_0 = C[0 * ldc + 0];
    register double _C1_0 = C[1 * ldc + 0];
    register double _C2_0 = C[2 * ldc + 0];
    register double _C3_0 = C[3 * ldc + 0];
    register double _C4_0 = C[4 * ldc + 0];
    register double _C5_0 = C[5 * ldc + 0];
    register double _C6_0 = C[6 * ldc + 0];
    register double _C7_0 = C[7 * ldc + 0];
    register double _C8_0 = C[8 * ldc + 0];

    for(i = 0; i < k; ++i) {
        _A0 = A[0];
        _C0_0 += B[0 * ldb] * _A0;
        _C1_0 += B[1 * ldb] * _A0;
        _C2_0 += B[2 * ldb] * _A0;
        _C3_0 += B[3 * ldb] * _A0;
        _C4_0 += B[4 * ldb] * _A0;
        _C5_0 += B[5 * ldb] * _A0;
        _C6_0 += B[6 * ldb] * _A0;
        _C7_0 += B[7 * ldb] * _A0;
        _C8_0 += B[8 * ldb] * _A0;

        A += lda;
        B += 1;
    }

    C[0 * ldc + 0] = _C0_0;
    C[1 * ldc + 0] = _C1_0;
    C[2 * ldc + 0] = _C2_0;
    C[3 * ldc + 0] = _C3_0;
    C[4 * ldc + 0] = _C4_0;
    C[5 * ldc + 0] = _C5_0;
    C[6 * ldc + 0] = _C6_0;
    C[7 * ldc + 0] = _C7_0;
    C[8 * ldc + 0] = _C8_0;
}
void micro_kernel2x9(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m128d _A0;
    register __m128d _C0_0 = _mm_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C1_0 = _mm_loadu_pd(&C[1 * ldc + 0]);
    register __m128d _C2_0 = _mm_loadu_pd(&C[2 * ldc + 0]);
    register __m128d _C3_0 = _mm_loadu_pd(&C[3 * ldc + 0]);
    register __m128d _C4_0 = _mm_loadu_pd(&C[4 * ldc + 0]);
    register __m128d _C5_0 = _mm_loadu_pd(&C[5 * ldc + 0]);
    register __m128d _C6_0 = _mm_loadu_pd(&C[6 * ldc + 0]);
    register __m128d _C7_0 = _mm_loadu_pd(&C[7 * ldc + 0]);
    register __m128d _C8_0 = _mm_loadu_pd(&C[8 * ldc + 0]);

    for(i = 0; i < k; ++i) {
        _A0 = _mm_loadu_pd(&A[0]);
        _C0_0 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C1_0 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C2_0 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C3_0 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C4_0 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C5_0 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C6_0 = _mm_fnmadd_pd(_mm_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C7_0 = _mm_fnmadd_pd(_mm_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C8_0 = _mm_fnmadd_pd(_mm_set1_pd(B[8 * ldb]), _A0, _C8_0);

        A += lda;
        B += 1;
    }

    _mm_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm_storeu_pd(&C[8 * ldc + 0], _C8_0);
}
void micro_kernel3x9(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m128d _A0;
    register double _A1;
    register __m128d _C0_0 = _mm_loadu_pd(&C[0 * ldc + 0]);
    register double _C0_1 = C[0 * ldc + 2];
    register __m128d _C1_0 = _mm_loadu_pd(&C[1 * ldc + 0]);
    register double _C1_1 = C[1 * ldc + 2];
    register __m128d _C2_0 = _mm_loadu_pd(&C[2 * ldc + 0]);
    register double _C2_1 = C[2 * ldc + 2];
    register __m128d _C3_0 = _mm_loadu_pd(&C[3 * ldc + 0]);
    register double _C3_1 = C[3 * ldc + 2];
    register __m128d _C4_0 = _mm_loadu_pd(&C[4 * ldc + 0]);
    register double _C4_1 = C[4 * ldc + 2];
    register __m128d _C5_0 = _mm_loadu_pd(&C[5 * ldc + 0]);
    register double _C5_1 = C[5 * ldc + 2];

    register __m128d _C6_0 = _mm_loadu_pd(&C[6 * ldc + 0]);
    register double _C6_1 = C[6 * ldc + 2];

    register __m128d _C7_0 = _mm_loadu_pd(&C[7 * ldc + 0]);
    register double _C7_1 = C[7 * ldc + 2];

    register __m128d _C8_0 = _mm_loadu_pd(&C[8 * ldc + 0]);
    register double _C8_1 = C[8 * ldc + 2];

    for(i = 0; i < k; ++i) {
        _A0 = _mm_loadu_pd(&A[0]);
        _A1 = A[2];
        _C0_0 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 += B[0 * ldb] * _A1;
        _C1_0 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 += B[1 * ldb] * _A1;
        _C2_0 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 += B[2 * ldb] * _A1;
        _C3_0 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 += B[3 * ldb] * _A1;
        _C4_0 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 += B[4 * ldb] * _A1;
        _C5_0 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 += B[5 * ldb] * _A1;

        _C6_0 = _mm_fnmadd_pd(_mm_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 += B[6 * ldb] * _A1;

        _C7_0 = _mm_fnmadd_pd(_mm_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 += B[7 * ldb] * _A1;

        _C8_0 = _mm_fnmadd_pd(_mm_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 += B[8 * ldb] * _A1;

        A += lda;
        B += 1;
    }

    _mm_storeu_pd(&C[0 * ldc + 0], _C0_0);
    C[0 * ldc + 2] = _C0_1;
    _mm_storeu_pd(&C[1 * ldc + 0], _C1_0);
    C[1 * ldc + 2] = _C1_1;
    _mm_storeu_pd(&C[2 * ldc + 0], _C2_0);
    C[2 * ldc + 2] = _C2_1;
    _mm_storeu_pd(&C[3 * ldc + 0], _C3_0);
    C[3 * ldc + 2] = _C3_1;
    _mm_storeu_pd(&C[4 * ldc + 0], _C4_0);
    C[4 * ldc + 2] = _C4_1;
    _mm_storeu_pd(&C[5 * ldc + 0], _C5_0);
    C[5 * ldc + 2] = _C5_1;

    _mm_storeu_pd(&C[6 * ldc + 0], _C6_0);
    C[6 * ldc + 2] = _C6_1;

    _mm_storeu_pd(&C[7 * ldc + 0], _C7_0);
    C[7 * ldc + 2] = _C7_1;

    _mm_storeu_pd(&C[8 * ldc + 0], _C8_0);
    C[8 * ldc + 2] = _C8_1;
}
void micro_kernel4x9(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m256d _A0;
    register __m256d _C0_0 = _mm256_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C1_0 = _mm256_loadu_pd(&C[1 * ldc + 0]);
    register __m256d _C2_0 = _mm256_loadu_pd(&C[2 * ldc + 0]);
    register __m256d _C3_0 = _mm256_loadu_pd(&C[3 * ldc + 0]);
    register __m256d _C4_0 = _mm256_loadu_pd(&C[4 * ldc + 0]);
    register __m256d _C5_0 = _mm256_loadu_pd(&C[5 * ldc + 0]);

    register __m256d _C6_0 = _mm256_loadu_pd(&C[6 * ldc + 0]);

    register __m256d _C7_0 = _mm256_loadu_pd(&C[7 * ldc + 0]);

    register __m256d _C8_0 = _mm256_loadu_pd(&C[8 * ldc + 0]);

    for(i = 0; i < k; ++i) {
        _A0 = _mm256_loadu_pd(&A[0]);
        _C0_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C1_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C2_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C3_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C4_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C5_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A0, _C5_0);

        _C6_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[6 * ldb]), _A0, _C6_0);

        _C7_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[7 * ldb]), _A0, _C7_0);

        _C8_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[8 * ldb]), _A0, _C8_0);

        A += lda;
        B += 1;
    }

    _mm256_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm256_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm256_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm256_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm256_storeu_pd(&C[5 * ldc + 0], _C5_0);

    _mm256_storeu_pd(&C[6 * ldc + 0], _C6_0);

    _mm256_storeu_pd(&C[7 * ldc + 0], _C7_0);

    _mm256_storeu_pd(&C[8 * ldc + 0], _C8_0);
}
void micro_kernel5x9(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m256d _A0;
    register double _A1;
    register __m256d _C0_0 = _mm256_loadu_pd(&C[0 * ldc + 0]);
    register double _C0_1 = C[0 * ldc + 4];
    register __m256d _C1_0 = _mm256_loadu_pd(&C[1 * ldc + 0]);
    register double _C1_1 = C[1 * ldc + 4];
    register __m256d _C2_0 = _mm256_loadu_pd(&C[2 * ldc + 0]);
    register double _C2_1 = C[2 * ldc + 4];
    register __m256d _C3_0 = _mm256_loadu_pd(&C[3 * ldc + 0]);
    register double _C3_1 = C[3 * ldc + 4];
    register __m256d _C4_0 = _mm256_loadu_pd(&C[4 * ldc + 0]);
    register double _C4_1 = C[4 * ldc + 4];
    register __m256d _C5_0 = _mm256_loadu_pd(&C[5 * ldc + 0]);
    register double _C5_1 = C[5 * ldc + 4];

    register __m256d _C6_0 = _mm256_loadu_pd(&C[6 * ldc + 0]);
    register double _C6_1 = C[6 * ldc + 4];

    register __m256d _C7_0 = _mm256_loadu_pd(&C[7 * ldc + 0]);
    register double _C7_1 = C[7 * ldc + 4];

    register __m256d _C8_0 = _mm256_loadu_pd(&C[8 * ldc + 0]);
    register double _C8_1 = C[8 * ldc + 4];

    for(i = 0; i < k; ++i) {
        _A0 = _mm256_loadu_pd(&A[0]);
        _A1 = A[4];
        _C0_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 += B[0 * ldb] * _A1;
        _C1_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 += B[1 * ldb] * _A1;
        _C2_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 += B[2 * ldb] * _A1;
        _C3_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 += B[3 * ldb] * _A1;
        _C4_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 += B[4 * ldb] * _A1;
        _C5_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 += B[5 * ldb] * _A1;

        _C6_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 += B[6 * ldb] * _A1;

        _C7_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 += B[7 * ldb] * _A1;

        _C8_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 += B[8 * ldb] * _A1;

        A += lda;
        B += 1;
    }

    _mm256_storeu_pd(&C[0 * ldc + 0], _C0_0);
    C[0 * ldc + 4] = _C0_1;
    _mm256_storeu_pd(&C[1 * ldc + 0], _C1_0);
    C[1 * ldc + 4] = _C1_1;
    _mm256_storeu_pd(&C[2 * ldc + 0], _C2_0);
    C[2 * ldc + 4] = _C2_1;
    _mm256_storeu_pd(&C[3 * ldc + 0], _C3_0);
    C[3 * ldc + 4] = _C3_1;
    _mm256_storeu_pd(&C[4 * ldc + 0], _C4_0);
    C[4 * ldc + 4] = _C4_1;
    _mm256_storeu_pd(&C[5 * ldc + 0], _C5_0);
    C[5 * ldc + 4] = _C5_1;

    _mm256_storeu_pd(&C[6 * ldc + 0], _C6_0);
    C[6 * ldc + 4] = _C6_1;

    _mm256_storeu_pd(&C[7 * ldc + 0], _C7_0);
    C[7 * ldc + 4] = _C7_1;

    _mm256_storeu_pd(&C[8 * ldc + 0], _C8_0);
    C[8 * ldc + 4] = _C8_1;
}
void micro_kernel6x9(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m256d _A0;
    register __m128d _A1;
    register __m256d _C0_0 = _mm256_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C0_1 = _mm_loadu_pd(&C[0 * ldc + 4]);
    register __m256d _C1_0 = _mm256_loadu_pd(&C[1 * ldc + 0]);
    register __m128d _C1_1 = _mm_loadu_pd(&C[1 * ldc + 4]);
    register __m256d _C2_0 = _mm256_loadu_pd(&C[2 * ldc + 0]);
    register __m128d _C2_1 = _mm_loadu_pd(&C[2 * ldc + 4]);
    register __m256d _C3_0 = _mm256_loadu_pd(&C[3 * ldc + 0]);
    register __m128d _C3_1 = _mm_loadu_pd(&C[3 * ldc + 4]);
    register __m256d _C4_0 = _mm256_loadu_pd(&C[4 * ldc + 0]);
    register __m128d _C4_1 = _mm_loadu_pd(&C[4 * ldc + 4]);
    register __m256d _C5_0 = _mm256_loadu_pd(&C[5 * ldc + 0]);
    register __m128d _C5_1 = _mm_loadu_pd(&C[5 * ldc + 4]);

    register __m256d _C6_0 = _mm256_loadu_pd(&C[6 * ldc + 0]);
    register __m128d _C6_1 = _mm_loadu_pd(&C[6 * ldc + 4]);

    register __m256d _C7_0 = _mm256_loadu_pd(&C[7 * ldc + 0]);
    register __m128d _C7_1 = _mm_loadu_pd(&C[7 * ldc + 4]);

    register __m256d _C8_0 = _mm256_loadu_pd(&C[8 * ldc + 0]);
    register __m128d _C8_1 = _mm_loadu_pd(&C[8 * ldc + 4]);

    for(i = 0; i < k; ++i) {
        _A0 = _mm256_loadu_pd(&A[0]);
        _A1 = _mm_loadu_pd(&A[4]);
        _C0_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C1_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C2_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C3_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A1, _C3_1);
        _C4_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A1, _C4_1);
        _C5_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A1, _C5_1);

        _C6_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm_fnmadd_pd(_mm_set1_pd(B[6 * ldb]), _A1, _C6_1);

        _C7_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 = _mm_fnmadd_pd(_mm_set1_pd(B[7 * ldb]), _A1, _C7_1);

        _C8_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 = _mm_fnmadd_pd(_mm_set1_pd(B[8 * ldb]), _A1, _C8_1);

        A += lda;
        B += 1;
    }

    _mm256_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[0 * ldc + 4], _C0_1);
    _mm256_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm_storeu_pd(&C[1 * ldc + 4], _C1_1);
    _mm256_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm_storeu_pd(&C[2 * ldc + 4], _C2_1);
    _mm256_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm_storeu_pd(&C[3 * ldc + 4], _C3_1);
    _mm256_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm_storeu_pd(&C[4 * ldc + 4], _C4_1);
    _mm256_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm_storeu_pd(&C[5 * ldc + 4], _C5_1);

    _mm256_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm_storeu_pd(&C[6 * ldc + 4], _C6_1);

    _mm256_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm_storeu_pd(&C[7 * ldc + 4], _C7_1);

    _mm256_storeu_pd(&C[8 * ldc + 0], _C8_0);
    _mm_storeu_pd(&C[8 * ldc + 4], _C8_1);
}
void micro_kernel7x9(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m256d _A0;
    register __m128d _A1;
    register double _A2;
    register __m256d _C0_0 = _mm256_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C0_1 = _mm_loadu_pd(&C[0 * ldc + 4]);
    register double _C0_2 = C[0 * ldc + 6];
    register __m256d _C1_0 = _mm256_loadu_pd(&C[1 * ldc + 0]);
    register __m128d _C1_1 = _mm_loadu_pd(&C[1 * ldc + 4]);
    register double _C1_2 = C[1 * ldc + 6];
    register __m256d _C2_0 = _mm256_loadu_pd(&C[2 * ldc + 0]);
    register __m128d _C2_1 = _mm_loadu_pd(&C[2 * ldc + 4]);
    register double _C2_2 = C[2 * ldc + 6];
    register __m256d _C3_0 = _mm256_loadu_pd(&C[3 * ldc + 0]);
    register __m128d _C3_1 = _mm_loadu_pd(&C[3 * ldc + 4]);
    register double _C3_2 = C[3 * ldc + 6];
    register __m256d _C4_0 = _mm256_loadu_pd(&C[4 * ldc + 0]);
    register __m128d _C4_1 = _mm_loadu_pd(&C[4 * ldc + 4]);
    register double _C4_2 = C[4 * ldc + 6];
    register __m256d _C5_0 = _mm256_loadu_pd(&C[5 * ldc + 0]);
    register __m128d _C5_1 = _mm_loadu_pd(&C[5 * ldc + 4]);
    register double _C5_2 = C[5 * ldc + 6];

    register __m256d _C6_0 = _mm256_loadu_pd(&C[6 * ldc + 0]);
    register __m128d _C6_1 = _mm_loadu_pd(&C[6 * ldc + 4]);
    register double _C6_2 = C[6 * ldc + 6];

    register __m256d _C7_0 = _mm256_loadu_pd(&C[7 * ldc + 0]);
    register __m128d _C7_1 = _mm_loadu_pd(&C[7 * ldc + 4]);
    register double _C7_2 = C[7 * ldc + 6];

    register __m256d _C8_0 = _mm256_loadu_pd(&C[8 * ldc + 0]);
    register __m128d _C8_1 = _mm_loadu_pd(&C[8 * ldc + 4]);
    register double _C8_2 = C[8 * ldc + 6];

    for(i = 0; i < k; ++i) {
        _A0 = _mm256_loadu_pd(&A[0]);
        _A1 = _mm_loadu_pd(&A[4]);
        _A2 = A[6];
        _C0_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 += B[0 * ldb] * _A2;
        _C1_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C1_2 += B[1 * ldb] * _A2;
        _C2_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C2_2 += B[2 * ldb] * _A2;
        _C3_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A1, _C3_1);
        _C3_2 += B[3 * ldb] * _A2;
        _C4_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A1, _C4_1);
        _C4_2 += B[4 * ldb] * _A2;
        _C5_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A1, _C5_1);
        _C5_2 += B[5 * ldb] * _A2;

        _C6_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm_fnmadd_pd(_mm_set1_pd(B[6 * ldb]), _A1, _C6_1);
        _C6_2 += B[6 * ldb] * _A2;

        _C7_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 = _mm_fnmadd_pd(_mm_set1_pd(B[7 * ldb]), _A1, _C7_1);
        _C7_2 += B[7 * ldb] * _A2;

        _C8_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 = _mm_fnmadd_pd(_mm_set1_pd(B[8 * ldb]), _A1, _C8_1);
        _C8_2 += B[8 * ldb] * _A2;

        A += lda;
        B += 1;
    }

    _mm256_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[0 * ldc + 4], _C0_1);
    C[0 * ldc + 6] = _C0_2;
    _mm256_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm_storeu_pd(&C[1 * ldc + 4], _C1_1);
    C[1 * ldc + 6] = _C1_2;
    _mm256_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm_storeu_pd(&C[2 * ldc + 4], _C2_1);
    C[2 * ldc + 6] = _C2_2;
    _mm256_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm_storeu_pd(&C[3 * ldc + 4], _C3_1);
    C[3 * ldc + 6] = _C3_2;
    _mm256_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm_storeu_pd(&C[4 * ldc + 4], _C4_1);
    C[4 * ldc + 6] = _C4_2;
    _mm256_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm_storeu_pd(&C[5 * ldc + 4], _C5_1);
    C[5 * ldc + 6] = _C5_2;

    _mm256_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm_storeu_pd(&C[6 * ldc + 4], _C6_1);
    C[6 * ldc + 6] = _C6_2;

    _mm256_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm_storeu_pd(&C[7 * ldc + 4], _C7_1);
    C[7 * ldc + 6] = _C7_2;

    _mm256_storeu_pd(&C[8 * ldc + 0], _C8_0);
    _mm_storeu_pd(&C[8 * ldc + 4], _C8_1);
    C[8 * ldc + 6] = _C8_2;
}
//=====================================================================================================================
void micro_kernel8x9(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);
    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);

    register __m512d _C7_0 = _mm512_loadu_pd(&C[7 * ldc + 0]);

    register __m512d _C8_0 = _mm512_loadu_pd(&C[8 * ldc + 0]);

    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);

        _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A0, _C7_0);

        _C8_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8 * ldb]), _A0, _C8_0);

        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);

    _mm512_storeu_pd(&C[7 * ldc + 0], _C7_0);

    _mm512_storeu_pd(&C[8 * ldc + 0], _C8_0);
}
void micro_kernel9x9(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register double _A1;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register double _C0_1 = C[0 * ldc + 8];

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);
    register double _C1_1 = C[1 * ldc + 8];

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register double _C2_1 = C[2 * ldc + 8];

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register double _C3_1 = C[3 * ldc + 8];

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register double _C4_1 = C[4 * ldc + 8];

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register double _C5_1 = C[5 * ldc + 8];

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);
    register double _C6_1 = C[6 * ldc + 8];

    register __m512d _C7_0 = _mm512_loadu_pd(&C[7 * ldc + 0]);
    register double _C7_1 = C[7 * ldc + 8];

    register __m512d _C8_0 = _mm512_loadu_pd(&C[8 * ldc + 0]);
    register double _C8_1 = C[8 * ldc + 8];

    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = A[8];

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 += B[0 * ldb] * _A1;

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 += B[1 * ldb] * _A1;

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 += B[2 * ldb] * _A1;

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 += B[3 * ldb] * _A1;

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 += B[4 * ldb] * _A1;

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 += B[5 * ldb] * _A1;

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 += B[6 * ldb] * _A1;
        
        _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 += B[7 * ldb] * _A1;

        _C8_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 += B[8 * ldb] * _A1;

        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    C[0 * ldc + 8] = _C0_1;

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    C[1 * ldc + 8] = _C1_1;

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    C[2 * ldc + 8] = _C2_1;

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    C[3 * ldc + 8] = _C3_1;

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    C[4 * ldc + 8] = _C4_1;

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    C[5 * ldc + 8] = _C5_1;

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);
    C[6 * ldc + 8] = _C6_1;

    _mm512_storeu_pd(&C[7 * ldc + 0], _C7_0);
    C[7 * ldc + 8] = _C7_1;

    _mm512_storeu_pd(&C[8 * ldc + 0], _C8_0);
    C[8 * ldc + 8] = _C8_1;
}
void micro_kernel10x9(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m128d _A1;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C0_1 = _mm_loadu_pd(&C[0 * ldc + 8]);

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m128d _C1_1 = _mm_loadu_pd(&C[1 * ldc + 8]);

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m128d _C2_1 = _mm_loadu_pd(&C[2 * ldc + 8]);

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m128d _C3_1 = _mm_loadu_pd(&C[3 * ldc + 8]);

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m128d _C4_1 = _mm_loadu_pd(&C[4 * ldc + 8]);

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register __m128d _C5_1 = _mm_loadu_pd(&C[5 * ldc + 8]);

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);
    register __m128d _C6_1 = _mm_loadu_pd(&C[6 * ldc + 8]);

    register __m512d _C7_0 = _mm512_loadu_pd(&C[7 * ldc + 0]);
    register __m128d _C7_1 = _mm_loadu_pd(&C[7 * ldc + 8]);

    register __m512d _C8_0 = _mm512_loadu_pd(&C[8 * ldc + 0]);
    register __m128d _C8_1 = _mm_loadu_pd(&C[8 * ldc + 8]);
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm_loadu_pd(&A[8]);

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A1, _C0_1);

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A1, _C1_1);

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A1, _C2_1);

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A1, _C3_1);

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A1, _C4_1);

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A1, _C5_1);

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm_fnmadd_pd(_mm_set1_pd(B[6 * ldb]), _A1, _C6_1);

        _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 = _mm_fnmadd_pd(_mm_set1_pd(B[7 * ldb]), _A1, _C7_1);

        _C8_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 = _mm_fnmadd_pd(_mm_set1_pd(B[8 * ldb]), _A1, _C8_1);
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[0 * ldc + 8], _C0_1);

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm_storeu_pd(&C[1 * ldc + 8], _C1_1);

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm_storeu_pd(&C[2 * ldc + 8], _C2_1);

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm_storeu_pd(&C[3 * ldc + 8], _C3_1);

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm_storeu_pd(&C[4 * ldc + 8], _C4_1);

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm_storeu_pd(&C[5 * ldc + 8], _C5_1);

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm_storeu_pd(&C[6 * ldc + 8], _C6_1);

    _mm512_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm_storeu_pd(&C[7 * ldc + 8], _C7_1);

    _mm512_storeu_pd(&C[8 * ldc + 0], _C8_0);
    _mm_storeu_pd(&C[8 * ldc + 8], _C8_1);
}
void micro_kernel11x9(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m128d _A1;
    register double _A2;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C0_1 = _mm_loadu_pd(&C[0 * ldc + 8]);
    register double _C0_2 = C[0 * ldc + 10];

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m128d _C1_1 = _mm_loadu_pd(&C[1 * ldc + 8]);
    register double _C1_2 = C[1 * ldc + 10];

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m128d _C2_1 = _mm_loadu_pd(&C[2 * ldc + 8]);
    register double _C2_2 = C[2 * ldc + 10];

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m128d _C3_1 = _mm_loadu_pd(&C[3 * ldc + 8]);
    register double _C3_2 = C[3 * ldc + 10];

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m128d _C4_1 = _mm_loadu_pd(&C[4 * ldc + 8]);
    register double _C4_2 = C[4 * ldc + 10];

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register __m128d _C5_1 = _mm_loadu_pd(&C[5 * ldc + 8]);
    register double _C5_2 = C[5 * ldc + 10];

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);
    register __m128d _C6_1 = _mm_loadu_pd(&C[6 * ldc + 8]);
    register double _C6_2 = C[6 * ldc + 10];

    register __m512d _C7_0 = _mm512_loadu_pd(&C[7 * ldc + 0]);
    register __m128d _C7_1 = _mm_loadu_pd(&C[7 * ldc + 8]);
    register double _C7_2 = C[7 * ldc + 10];

    register __m512d _C8_0 = _mm512_loadu_pd(&C[8 * ldc + 0]);
    register __m128d _C8_1 = _mm_loadu_pd(&C[8 * ldc + 8]);
    register double _C8_2 = C[8 * ldc + 10];
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm_loadu_pd(&A[8]);
        _A2 = A[10];

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 += B[0 * ldb] * _A2;

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C1_2 += B[1 * ldb] * _A2;

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C2_2 += B[2 * ldb] * _A2;

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A1, _C3_1);
        _C3_2 += B[3 * ldb] * _A2;

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A1, _C4_1);
        _C4_2 += B[4 * ldb] * _A2;

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A1, _C5_1);
        _C5_2 += B[5 * ldb] * _A2;

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm_fnmadd_pd(_mm_set1_pd(B[6 * ldb]), _A1, _C6_1);
        _C6_2 += B[6 * ldb] * _A2;

        _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 = _mm_fnmadd_pd(_mm_set1_pd(B[7 * ldb]), _A1, _C7_1);
        _C7_2 += B[7 * ldb] * _A2;

        _C8_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 = _mm_fnmadd_pd(_mm_set1_pd(B[8 * ldb]), _A1, _C8_1);
        _C8_2 += B[8 * ldb] * _A2;
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[0 * ldc + 8], _C0_1);
    C[0 * ldc + 10] = _C0_2;

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm_storeu_pd(&C[1 * ldc + 8], _C1_1);
    C[1 * ldc + 10] = _C1_2;

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm_storeu_pd(&C[2 * ldc + 8], _C2_1);
    C[2 * ldc + 10] = _C2_2;

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm_storeu_pd(&C[3 * ldc + 8], _C3_1);
    C[3 * ldc + 10] = _C3_2;

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm_storeu_pd(&C[4 * ldc + 8], _C4_1);
    C[4 * ldc + 10] = _C4_2;

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm_storeu_pd(&C[5 * ldc + 8], _C5_1);
    C[5 * ldc + 10] = _C5_2;

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm_storeu_pd(&C[6 * ldc + 8], _C6_1);
    C[6 * ldc + 10] = _C6_2;

    _mm512_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm_storeu_pd(&C[7 * ldc + 8], _C7_1);
    C[7 * ldc + 10] = _C7_2;

    _mm512_storeu_pd(&C[8 * ldc + 0], _C8_0);
    _mm_storeu_pd(&C[8 * ldc + 8], _C8_1);
    C[8 * ldc + 10] = _C8_2;
}
void micro_kernel12x9(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
     int i;
    register __m512d _A0;
    register __m256d _A1;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C0_1 = _mm256_loadu_pd(&C[0 * ldc + 8]);

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m256d _C1_1 = _mm256_loadu_pd(&C[1 * ldc + 8]);

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m256d _C2_1 = _mm256_loadu_pd(&C[2 * ldc + 8]);

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m256d _C3_1 = _mm256_loadu_pd(&C[3 * ldc + 8]);

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m256d _C4_1 = _mm256_loadu_pd(&C[4 * ldc + 8]);

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register __m256d _C5_1 = _mm256_loadu_pd(&C[5 * ldc + 8]);

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);
    register __m256d _C6_1 = _mm256_loadu_pd(&C[6 * ldc + 8]);

    register __m512d _C7_0 = _mm512_loadu_pd(&C[7 * ldc + 0]);
    register __m256d _C7_1 = _mm256_loadu_pd(&C[7 * ldc + 8]);

    register __m512d _C8_0 = _mm512_loadu_pd(&C[8 * ldc + 0]);
    register __m256d _C8_1 = _mm256_loadu_pd(&C[8 * ldc + 8]);
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm256_loadu_pd(&A[8]);

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A1, _C0_1);

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A1, _C1_1);

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A1, _C2_1);

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A1, _C3_1);

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A1, _C4_1);

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A1, _C5_1);

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[6 * ldb]), _A1, _C6_1);

        _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[7 * ldb]), _A1, _C7_1);

        _C8_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[8 * ldb]), _A1, _C8_1);
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[0 * ldc + 8], _C0_1);

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm256_storeu_pd(&C[1 * ldc + 8], _C1_1);

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm256_storeu_pd(&C[2 * ldc + 8], _C2_1);

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm256_storeu_pd(&C[3 * ldc + 8], _C3_1);

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm256_storeu_pd(&C[4 * ldc + 8], _C4_1);

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm256_storeu_pd(&C[5 * ldc + 8], _C5_1);

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm256_storeu_pd(&C[6 * ldc + 8], _C6_1);

    _mm512_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm256_storeu_pd(&C[7 * ldc + 8], _C7_1);

    _mm512_storeu_pd(&C[8 * ldc + 0], _C8_0);
    _mm256_storeu_pd(&C[8 * ldc + 8], _C8_1);
}
void micro_kernel13x9(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m256d _A1;
    register double _A2;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C0_1 = _mm256_loadu_pd(&C[0 * ldc + 8]);
    register double _C0_2 = C[0 * ldc + 12];

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m256d _C1_1 = _mm256_loadu_pd(&C[1 * ldc + 8]);
    register double _C1_2 = C[1 * ldc + 12];

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m256d _C2_1 = _mm256_loadu_pd(&C[2 * ldc + 8]);
    register double _C2_2 = C[2 * ldc + 12];

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m256d _C3_1 = _mm256_loadu_pd(&C[3 * ldc + 8]);
    register double _C3_2 = C[3 * ldc + 12];

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m256d _C4_1 = _mm256_loadu_pd(&C[4 * ldc + 8]);
    register double _C4_2 = C[4 * ldc + 12];

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register __m256d _C5_1 = _mm256_loadu_pd(&C[5 * ldc + 8]);
    register double _C5_2 = C[5 * ldc + 12];

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);
    register __m256d _C6_1 = _mm256_loadu_pd(&C[6 * ldc + 8]);
    register double _C6_2 = C[6 * ldc + 12];

    register __m512d _C7_0 = _mm512_loadu_pd(&C[7 * ldc + 0]);
    register __m256d _C7_1 = _mm256_loadu_pd(&C[7 * ldc + 8]);
    register double _C7_2 = C[7 * ldc + 12];

    register __m512d _C8_0 = _mm512_loadu_pd(&C[8 * ldc + 0]);
    register __m256d _C8_1 = _mm256_loadu_pd(&C[8 * ldc + 8]);
    register double _C8_2 = C[8 * ldc + 12];
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm256_loadu_pd(&A[8]);
        _A2 = A[12];

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 += B[0 * ldb] * _A2;

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C1_2 += B[1 * ldb] * _A2;

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C2_2 += B[2 * ldb] * _A2;

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A1, _C3_1);
        _C3_2 += B[3 * ldb] * _A2;

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A1, _C4_1);
        _C4_2 += B[4 * ldb] * _A2;

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A1, _C5_1);
        _C5_2 += B[5 * ldb] * _A2;

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[6 * ldb]), _A1, _C6_1);
        _C6_2 += B[6 * ldb] * _A2;

        _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[7 * ldb]), _A1, _C7_1);
        _C7_2 += B[7 * ldb] * _A2;

        _C8_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[8 * ldb]), _A1, _C8_1);
        _C8_2 += B[8 * ldb] * _A2;
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[0 * ldc + 8], _C0_1);
    C[0 * ldc + 12] = _C0_2;

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm256_storeu_pd(&C[1 * ldc + 8], _C1_1);
    C[1 * ldc + 12] = _C1_2;

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm256_storeu_pd(&C[2 * ldc + 8], _C2_1);
    C[2 * ldc + 12] = _C2_2;

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm256_storeu_pd(&C[3 * ldc + 8], _C3_1);
    C[3 * ldc + 12] = _C3_2;

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm256_storeu_pd(&C[4 * ldc + 8], _C4_1);
    C[4 * ldc + 12] = _C4_2;

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm256_storeu_pd(&C[5 * ldc + 8], _C5_1);
    C[5 * ldc + 12] = _C5_2;

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm256_storeu_pd(&C[6 * ldc + 8], _C6_1);
    C[6 * ldc + 12] = _C6_2;

    _mm512_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm256_storeu_pd(&C[7 * ldc + 8], _C7_1);
    C[7 * ldc + 12] = _C7_2;

    _mm512_storeu_pd(&C[8 * ldc + 0], _C8_0);
    _mm256_storeu_pd(&C[8 * ldc + 8], _C8_1);
    C[8 * ldc + 12] = _C8_2;
}
void micro_kernel14x9(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m256d _A1;
    register __m128d _A2;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C0_1 = _mm256_loadu_pd(&C[0 * ldc + 8]);
    register __m128d _C0_2 = _mm_loadu_pd(&C[0 * ldc + 12]);

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m256d _C1_1 = _mm256_loadu_pd(&C[1 * ldc + 8]);
    register __m128d _C1_2 = _mm_loadu_pd(&C[1 * ldc + 12]);

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m256d _C2_1 = _mm256_loadu_pd(&C[2 * ldc + 8]);
    register __m128d _C2_2 = _mm_loadu_pd(&C[2 * ldc + 12]);

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m256d _C3_1 = _mm256_loadu_pd(&C[3 * ldc + 8]);
    register __m128d _C3_2 = _mm_loadu_pd(&C[3 * ldc + 12]);

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m256d _C4_1 = _mm256_loadu_pd(&C[4 * ldc + 8]);
    register __m128d _C4_2 = _mm_loadu_pd(&C[4 * ldc + 12]);

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register __m256d _C5_1 = _mm256_loadu_pd(&C[5 * ldc + 8]);
    register __m128d _C5_2 = _mm_loadu_pd(&C[5 * ldc + 12]);

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);
    register __m256d _C6_1 = _mm256_loadu_pd(&C[6 * ldc + 8]);
    register __m128d _C6_2 = _mm_loadu_pd(&C[6 * ldc + 12]);

    register __m512d _C7_0 = _mm512_loadu_pd(&C[7 * ldc + 0]);
    register __m256d _C7_1 = _mm256_loadu_pd(&C[7 * ldc + 8]);
    register __m128d _C7_2 = _mm_loadu_pd(&C[7 * ldc + 12]);

    register __m512d _C8_0 = _mm512_loadu_pd(&C[8 * ldc + 0]);
    register __m256d _C8_1 = _mm256_loadu_pd(&C[8 * ldc + 8]);
    register __m128d _C8_2 = _mm_loadu_pd(&C[8 * ldc + 12]);
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm256_loadu_pd(&A[8]);
        _A2 = _mm_loadu_pd(&A[12]);

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A2, _C0_2);

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C1_2 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A2, _C1_2);

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C2_2 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A2, _C2_2);

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A1, _C3_1);
        _C3_2 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A2, _C3_2);

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A1, _C4_1);
        _C4_2 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A2, _C4_2);

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A1, _C5_1);
        _C5_2 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A2, _C5_2);

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[6 * ldb]), _A1, _C6_1);
        _C6_2 = _mm_fnmadd_pd(_mm_set1_pd(B[6 * ldb]), _A2, _C6_2);

        _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[7 * ldb]), _A1, _C7_1);
        _C7_2 = _mm_fnmadd_pd(_mm_set1_pd(B[7 * ldb]), _A2, _C7_2);

        _C8_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[8 * ldb]), _A1, _C8_1);
        _C8_2 = _mm_fnmadd_pd(_mm_set1_pd(B[8 * ldb]), _A2, _C8_2);
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[0 * ldc + 8], _C0_1);
    _mm_storeu_pd(&C[0 * ldc + 12], _C0_2);

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm256_storeu_pd(&C[1 * ldc + 8], _C1_1);
    _mm_storeu_pd(&C[1 * ldc + 12], _C1_2);

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm256_storeu_pd(&C[2 * ldc + 8], _C2_1);
    _mm_storeu_pd(&C[2 * ldc + 12], _C2_2);

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm256_storeu_pd(&C[3 * ldc + 8], _C3_1);
    _mm_storeu_pd(&C[3 * ldc + 12], _C3_2);

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm256_storeu_pd(&C[4 * ldc + 8], _C4_1);
    _mm_storeu_pd(&C[4 * ldc + 12], _C4_2);

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm256_storeu_pd(&C[5 * ldc + 8], _C5_1);
    _mm_storeu_pd(&C[5 * ldc + 12], _C5_2);

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm256_storeu_pd(&C[6 * ldc + 8], _C6_1);
    _mm_storeu_pd(&C[6 * ldc + 12], _C6_2);

    _mm512_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm256_storeu_pd(&C[7 * ldc + 8], _C7_1);
    _mm_storeu_pd(&C[7 * ldc + 12], _C7_2);

    _mm512_storeu_pd(&C[8 * ldc + 0], _C8_0);
    _mm256_storeu_pd(&C[8 * ldc + 8], _C8_1);
    _mm_storeu_pd(&C[8 * ldc + 12], _C8_2);
}
void micro_kernel15x9(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m256d _A1;
    register __m128d _A2;
    register double _A3;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C0_1 = _mm256_loadu_pd(&C[0 * ldc + 8]);
    register __m128d _C0_2 = _mm_loadu_pd(&C[0 * ldc + 12]);
    register double _C0_3 = C[0 * ldc + 14];

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m256d _C1_1 = _mm256_loadu_pd(&C[1 * ldc + 8]);
    register __m128d _C1_2 = _mm_loadu_pd(&C[1 * ldc + 12]);
    register double _C1_3 = C[1 * ldc + 14];

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m256d _C2_1 = _mm256_loadu_pd(&C[2 * ldc + 8]);
    register __m128d _C2_2 = _mm_loadu_pd(&C[2 * ldc + 12]);
    register double _C2_3 = C[2 * ldc + 14];

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m256d _C3_1 = _mm256_loadu_pd(&C[3 * ldc + 8]);
    register __m128d _C3_2 = _mm_loadu_pd(&C[3 * ldc + 12]);
    register double _C3_3 = C[3 * ldc + 14];

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m256d _C4_1 = _mm256_loadu_pd(&C[4 * ldc + 8]);
    register __m128d _C4_2 = _mm_loadu_pd(&C[4 * ldc + 12]);
    register double _C4_3 = C[4 * ldc + 14];

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register __m256d _C5_1 = _mm256_loadu_pd(&C[5 * ldc + 8]);
    register __m128d _C5_2 = _mm_loadu_pd(&C[5 * ldc + 12]);
    register double _C5_3 = C[5 * ldc + 14];

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);
    register __m256d _C6_1 = _mm256_loadu_pd(&C[6 * ldc + 8]);
    register __m128d _C6_2 = _mm_loadu_pd(&C[6 * ldc + 12]);
    register double _C6_3 = C[6 * ldc + 14];

    register __m512d _C7_0 = _mm512_loadu_pd(&C[7 * ldc + 0]);
    register __m256d _C7_1 = _mm256_loadu_pd(&C[7 * ldc + 8]);
    register __m128d _C7_2 = _mm_loadu_pd(&C[7 * ldc + 12]);
    register double _C7_3 = C[7 * ldc + 14];

    register __m512d _C8_0 = _mm512_loadu_pd(&C[8 * ldc + 0]);
    register __m256d _C8_1 = _mm256_loadu_pd(&C[8 * ldc + 8]);
    register __m128d _C8_2 = _mm_loadu_pd(&C[8 * ldc + 12]);
    register double _C8_3 = C[8 * ldc + 14];
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm256_loadu_pd(&A[8]);
        _A2 = _mm_loadu_pd(&A[12]);
        _A3 = A[14];

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A2, _C0_2);
        _C0_3 += B[0 * ldb] * _A3;

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C1_2 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A2, _C1_2);
        _C1_3 += B[1 * ldb] * _A3;

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C2_2 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A2, _C2_2);
        _C2_3 += B[2 * ldb] * _A3;

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A1, _C3_1);
        _C3_2 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A2, _C3_2);
        _C3_3 += B[3 * ldb] * _A3;

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A1, _C4_1);
        _C4_2 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A2, _C4_2);
        _C4_3 += B[4 * ldb] * _A3;

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A1, _C5_1);
        _C5_2 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A2, _C5_2);
        _C5_3 += B[5 * ldb] * _A3;

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[6 * ldb]), _A1, _C6_1);
        _C6_2 = _mm_fnmadd_pd(_mm_set1_pd(B[6 * ldb]), _A2, _C6_2);
        _C6_3 += B[6 * ldb] * _A3;

        _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[7 * ldb]), _A1, _C7_1);
        _C7_2 = _mm_fnmadd_pd(_mm_set1_pd(B[7 * ldb]), _A2, _C7_2);
        _C7_3 += B[7 * ldb] * _A3;

        _C8_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[8 * ldb]), _A1, _C8_1);
        _C8_2 = _mm_fnmadd_pd(_mm_set1_pd(B[8 * ldb]), _A2, _C8_2);
        _C8_3 += B[8 * ldb] * _A3;
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[0 * ldc + 8], _C0_1);
    _mm_storeu_pd(&C[0 * ldc + 12], _C0_2);
    C[0 * ldc + 14] = _C0_3;

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm256_storeu_pd(&C[1 * ldc + 8], _C1_1);
    _mm_storeu_pd(&C[1 * ldc + 12], _C1_2);
    C[1 * ldc + 14] = _C1_3;

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm256_storeu_pd(&C[2 * ldc + 8], _C2_1);
    _mm_storeu_pd(&C[2 * ldc + 12], _C2_2);
    C[2 * ldc + 14] = _C2_3;

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm256_storeu_pd(&C[3 * ldc + 8], _C3_1);
    _mm_storeu_pd(&C[3 * ldc + 12], _C3_2);
    C[3 * ldc + 14] = _C3_3;

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm256_storeu_pd(&C[4 * ldc + 8], _C4_1);
    _mm_storeu_pd(&C[4 * ldc + 12], _C4_2);
    C[4 * ldc + 14] = _C4_3;

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm256_storeu_pd(&C[5 * ldc + 8], _C5_1);
    _mm_storeu_pd(&C[5 * ldc + 12], _C5_2);
    C[5 * ldc + 14] = _C5_3;

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm256_storeu_pd(&C[6 * ldc + 8], _C6_1);
    _mm_storeu_pd(&C[6 * ldc + 12], _C6_2);
    C[6 * ldc + 14] = _C6_3;

    _mm512_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm256_storeu_pd(&C[7 * ldc + 8], _C7_1);
    _mm_storeu_pd(&C[7 * ldc + 12], _C7_2);
    C[7 * ldc + 14] = _C7_3;

    _mm512_storeu_pd(&C[8 * ldc + 0], _C8_0);
    _mm256_storeu_pd(&C[8 * ldc + 8], _C8_1);
    _mm_storeu_pd(&C[8 * ldc + 12], _C8_2);
    C[8 * ldc + 14] = _C8_3;
}
void micro_kernel16x9(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m512d _A1;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m512d _C0_1 = _mm512_loadu_pd(&C[0 * ldc + 8]);

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m512d _C1_1 = _mm512_loadu_pd(&C[1 * ldc + 8]);

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m512d _C2_1 = _mm512_loadu_pd(&C[2 * ldc + 8]);

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m512d _C3_1 = _mm512_loadu_pd(&C[3 * ldc + 8]);

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m512d _C4_1 = _mm512_loadu_pd(&C[4 * ldc + 8]);

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register __m512d _C5_1 = _mm512_loadu_pd(&C[5 * ldc + 8]);

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);
    register __m512d _C6_1 = _mm512_loadu_pd(&C[6 * ldc + 8]);

    register __m512d _C7_0 = _mm512_loadu_pd(&C[7 * ldc + 0]);
    register __m512d _C7_1 = _mm512_loadu_pd(&C[7 * ldc + 8]);

    register __m512d _C8_0 = _mm512_loadu_pd(&C[8 * ldc + 0]);
    register __m512d _C8_1 = _mm512_loadu_pd(&C[8 * ldc + 8]);
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm512_loadu_pd(&A[8]);

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A1, _C0_1);

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A1, _C1_1);

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A1, _C2_1);

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A1, _C3_1);

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A1, _C4_1);

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A1, _C5_1);

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A1, _C6_1);

        _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A1, _C7_1);

        _C8_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8 * ldb]), _A1, _C8_1);
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm512_storeu_pd(&C[0 * ldc + 8], _C0_1);

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm512_storeu_pd(&C[1 * ldc + 8], _C1_1);

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm512_storeu_pd(&C[2 * ldc + 8], _C2_1);

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm512_storeu_pd(&C[3 * ldc + 8], _C3_1);

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm512_storeu_pd(&C[4 * ldc + 8], _C4_1);

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm512_storeu_pd(&C[5 * ldc + 8], _C5_1);

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm512_storeu_pd(&C[6 * ldc + 8], _C6_1);

    _mm512_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm512_storeu_pd(&C[7 * ldc + 8], _C7_1);

    _mm512_storeu_pd(&C[8 * ldc + 0], _C8_0);
    _mm512_storeu_pd(&C[8 * ldc + 8], _C8_1);
}
//#####################################@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@####################################################################
void micro_kernel1x10(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register double _A0;
    register double _C0_0 = C[0 * ldc + 0];
    register double _C1_0 = C[1 * ldc + 0];
    register double _C2_0 = C[2 * ldc + 0];
    register double _C3_0 = C[3 * ldc + 0];
    register double _C4_0 = C[4 * ldc + 0];
    register double _C5_0 = C[5 * ldc + 0];
    register double _C6_0 = C[6 * ldc + 0];
    register double _C7_0 = C[7 * ldc + 0];
    register double _C8_0 = C[8 * ldc + 0];
    register double _C9_0 = C[9 * ldc + 0];

    for(i = 0; i < k; ++i) {
        _A0 = A[0];
        _C0_0 += B[0 * ldb] * _A0;
        _C1_0 += B[1 * ldb] * _A0;
        _C2_0 += B[2 * ldb] * _A0;
        _C3_0 += B[3 * ldb] * _A0;
        _C4_0 += B[4 * ldb] * _A0;
        _C5_0 += B[5 * ldb] * _A0;
        _C6_0 += B[6 * ldb] * _A0;
        _C7_0 += B[7 * ldb] * _A0;
        _C8_0 += B[8 * ldb] * _A0;
        _C9_0 += B[9 * ldb] * _A0;

        A += lda;
        B += 1;
    }

    C[0 * ldc + 0] = _C0_0;
    C[1 * ldc + 0] = _C1_0;
    C[2 * ldc + 0] = _C2_0;
    C[3 * ldc + 0] = _C3_0;
    C[4 * ldc + 0] = _C4_0;
    C[5 * ldc + 0] = _C5_0;
    C[6 * ldc + 0] = _C6_0;
    C[7 * ldc + 0] = _C7_0;
    C[8 * ldc + 0] = _C8_0;
    C[9 * ldc + 0] = _C9_0;
}
void micro_kernel2x10(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m128d _A0;
    register __m128d _C0_0 = _mm_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C1_0 = _mm_loadu_pd(&C[1 * ldc + 0]);
    register __m128d _C2_0 = _mm_loadu_pd(&C[2 * ldc + 0]);
    register __m128d _C3_0 = _mm_loadu_pd(&C[3 * ldc + 0]);
    register __m128d _C4_0 = _mm_loadu_pd(&C[4 * ldc + 0]);
    register __m128d _C5_0 = _mm_loadu_pd(&C[5 * ldc + 0]);
    register __m128d _C6_0 = _mm_loadu_pd(&C[6 * ldc + 0]);
    register __m128d _C7_0 = _mm_loadu_pd(&C[7 * ldc + 0]);
    register __m128d _C8_0 = _mm_loadu_pd(&C[8 * ldc + 0]);
    register __m128d _C9_0 = _mm_loadu_pd(&C[9 * ldc + 0]);

    for(i = 0; i < k; ++i) {
        _A0 = _mm_loadu_pd(&A[0]);
        _C0_0 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C1_0 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C2_0 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C3_0 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C4_0 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C5_0 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C6_0 = _mm_fnmadd_pd(_mm_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C7_0 = _mm_fnmadd_pd(_mm_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C8_0 = _mm_fnmadd_pd(_mm_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C9_0 = _mm_fnmadd_pd(_mm_set1_pd(B[9 * ldb]), _A0, _C9_0);

        A += lda;
        B += 1;
    }

    _mm_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm_storeu_pd(&C[8 * ldc + 0], _C8_0);
    _mm_storeu_pd(&C[9 * ldc + 0], _C9_0);
}
void micro_kernel3x10(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m128d _A0;
    register double _A1;
    register __m128d _C0_0 = _mm_loadu_pd(&C[0 * ldc + 0]);
    register double _C0_1 = C[0 * ldc + 2];
    register __m128d _C1_0 = _mm_loadu_pd(&C[1 * ldc + 0]);
    register double _C1_1 = C[1 * ldc + 2];
    register __m128d _C2_0 = _mm_loadu_pd(&C[2 * ldc + 0]);
    register double _C2_1 = C[2 * ldc + 2];
    register __m128d _C3_0 = _mm_loadu_pd(&C[3 * ldc + 0]);
    register double _C3_1 = C[3 * ldc + 2];
    register __m128d _C4_0 = _mm_loadu_pd(&C[4 * ldc + 0]);
    register double _C4_1 = C[4 * ldc + 2];
    register __m128d _C5_0 = _mm_loadu_pd(&C[5 * ldc + 0]);
    register double _C5_1 = C[5 * ldc + 2];

    register __m128d _C6_0 = _mm_loadu_pd(&C[6 * ldc + 0]);
    register double _C6_1 = C[6 * ldc + 2];

    register __m128d _C7_0 = _mm_loadu_pd(&C[7 * ldc + 0]);
    register double _C7_1 = C[7 * ldc + 2];

    register __m128d _C8_0 = _mm_loadu_pd(&C[8 * ldc + 0]);
    register double _C8_1 = C[8 * ldc + 2];

    register __m128d _C9_0 = _mm_loadu_pd(&C[9 * ldc + 0]);
    register double _C9_1 = C[9 * ldc + 2];

    for(i = 0; i < k; ++i) {
        _A0 = _mm_loadu_pd(&A[0]);
        _A1 = A[2];
        _C0_0 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 += B[0 * ldb] * _A1;
        _C1_0 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 += B[1 * ldb] * _A1;
        _C2_0 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 += B[2 * ldb] * _A1;
        _C3_0 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 += B[3 * ldb] * _A1;
        _C4_0 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 += B[4 * ldb] * _A1;
        _C5_0 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 += B[5 * ldb] * _A1;

        _C6_0 = _mm_fnmadd_pd(_mm_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 += B[6 * ldb] * _A1;

        _C7_0 = _mm_fnmadd_pd(_mm_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 += B[7 * ldb] * _A1;

        _C8_0 = _mm_fnmadd_pd(_mm_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 += B[8 * ldb] * _A1;

        _C9_0 = _mm_fnmadd_pd(_mm_set1_pd(B[9 * ldb]), _A0, _C9_0);
        _C9_1 += B[9 * ldb] * _A1;

        A += lda;
        B += 1;
    }

    _mm_storeu_pd(&C[0 * ldc + 0], _C0_0);
    C[0 * ldc + 2] = _C0_1;
    _mm_storeu_pd(&C[1 * ldc + 0], _C1_0);
    C[1 * ldc + 2] = _C1_1;
    _mm_storeu_pd(&C[2 * ldc + 0], _C2_0);
    C[2 * ldc + 2] = _C2_1;
    _mm_storeu_pd(&C[3 * ldc + 0], _C3_0);
    C[3 * ldc + 2] = _C3_1;
    _mm_storeu_pd(&C[4 * ldc + 0], _C4_0);
    C[4 * ldc + 2] = _C4_1;
    _mm_storeu_pd(&C[5 * ldc + 0], _C5_0);
    C[5 * ldc + 2] = _C5_1;

    _mm_storeu_pd(&C[6 * ldc + 0], _C6_0);
    C[6 * ldc + 2] = _C6_1;

    _mm_storeu_pd(&C[7 * ldc + 0], _C7_0);
    C[7 * ldc + 2] = _C7_1;

    _mm_storeu_pd(&C[8 * ldc + 0], _C8_0);
    C[8 * ldc + 2] = _C8_1;

    _mm_storeu_pd(&C[9 * ldc + 0], _C9_0);
    C[9 * ldc + 2] = _C9_1;
}
void micro_kernel4x10(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m256d _A0;
    register __m256d _C0_0 = _mm256_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C1_0 = _mm256_loadu_pd(&C[1 * ldc + 0]);
    register __m256d _C2_0 = _mm256_loadu_pd(&C[2 * ldc + 0]);
    register __m256d _C3_0 = _mm256_loadu_pd(&C[3 * ldc + 0]);
    register __m256d _C4_0 = _mm256_loadu_pd(&C[4 * ldc + 0]);
    register __m256d _C5_0 = _mm256_loadu_pd(&C[5 * ldc + 0]);

    register __m256d _C6_0 = _mm256_loadu_pd(&C[6 * ldc + 0]);

    register __m256d _C7_0 = _mm256_loadu_pd(&C[7 * ldc + 0]);

    register __m256d _C8_0 = _mm256_loadu_pd(&C[8 * ldc + 0]);

    register __m256d _C9_0 = _mm256_loadu_pd(&C[9 * ldc + 0]);


    for(i = 0; i < k; ++i) {
        _A0 = _mm256_loadu_pd(&A[0]);
        _C0_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C1_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C2_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C3_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C4_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C5_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A0, _C5_0);

        _C6_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[6 * ldb]), _A0, _C6_0);

        _C7_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[7 * ldb]), _A0, _C7_0);

        _C8_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[8 * ldb]), _A0, _C8_0);

        _C9_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[9 * ldb]), _A0, _C9_0);

        A += lda;
        B += 1;
    }

    _mm256_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm256_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm256_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm256_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm256_storeu_pd(&C[5 * ldc + 0], _C5_0);

    _mm256_storeu_pd(&C[6 * ldc + 0], _C6_0);

    _mm256_storeu_pd(&C[7 * ldc + 0], _C7_0);

    _mm256_storeu_pd(&C[8 * ldc + 0], _C8_0);

    _mm256_storeu_pd(&C[9 * ldc + 0], _C9_0);
}
void micro_kernel5x10(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m256d _A0;
    register double _A1;
    register __m256d _C0_0 = _mm256_loadu_pd(&C[0 * ldc + 0]);
    register double _C0_1 = C[0 * ldc + 4];
    register __m256d _C1_0 = _mm256_loadu_pd(&C[1 * ldc + 0]);
    register double _C1_1 = C[1 * ldc + 4];
    register __m256d _C2_0 = _mm256_loadu_pd(&C[2 * ldc + 0]);
    register double _C2_1 = C[2 * ldc + 4];
    register __m256d _C3_0 = _mm256_loadu_pd(&C[3 * ldc + 0]);
    register double _C3_1 = C[3 * ldc + 4];
    register __m256d _C4_0 = _mm256_loadu_pd(&C[4 * ldc + 0]);
    register double _C4_1 = C[4 * ldc + 4];
    register __m256d _C5_0 = _mm256_loadu_pd(&C[5 * ldc + 0]);
    register double _C5_1 = C[5 * ldc + 4];

    register __m256d _C6_0 = _mm256_loadu_pd(&C[6 * ldc + 0]);
    register double _C6_1 = C[6 * ldc + 4];

    register __m256d _C7_0 = _mm256_loadu_pd(&C[7 * ldc + 0]);
    register double _C7_1 = C[7 * ldc + 4];

    register __m256d _C8_0 = _mm256_loadu_pd(&C[8 * ldc + 0]);
    register double _C8_1 = C[8 * ldc + 4];

    register __m256d _C9_0 = _mm256_loadu_pd(&C[9 * ldc + 0]);
    register double _C9_1 = C[9 * ldc + 4];

    for(i = 0; i < k; ++i) {
        _A0 = _mm256_loadu_pd(&A[0]);
        _A1 = A[4];
        _C0_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 += B[0 * ldb] * _A1;
        _C1_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 += B[1 * ldb] * _A1;
        _C2_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 += B[2 * ldb] * _A1;
        _C3_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 += B[3 * ldb] * _A1;
        _C4_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 += B[4 * ldb] * _A1;
        _C5_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 += B[5 * ldb] * _A1;

        _C6_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 += B[6 * ldb] * _A1;

        _C7_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 += B[7 * ldb] * _A1;

        _C8_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 += B[8 * ldb] * _A1;

        _C9_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[9 * ldb]), _A0, _C9_0);
        _C9_1 += B[9 * ldb] * _A1;

        A += lda;
        B += 1;
    }

    _mm256_storeu_pd(&C[0 * ldc + 0], _C0_0);
    C[0 * ldc + 4] = _C0_1;
    _mm256_storeu_pd(&C[1 * ldc + 0], _C1_0);
    C[1 * ldc + 4] = _C1_1;
    _mm256_storeu_pd(&C[2 * ldc + 0], _C2_0);
    C[2 * ldc + 4] = _C2_1;
    _mm256_storeu_pd(&C[3 * ldc + 0], _C3_0);
    C[3 * ldc + 4] = _C3_1;
    _mm256_storeu_pd(&C[4 * ldc + 0], _C4_0);
    C[4 * ldc + 4] = _C4_1;
    _mm256_storeu_pd(&C[5 * ldc + 0], _C5_0);
    C[5 * ldc + 4] = _C5_1;

    _mm256_storeu_pd(&C[6 * ldc + 0], _C6_0);
    C[6 * ldc + 4] = _C6_1;

    _mm256_storeu_pd(&C[7 * ldc + 0], _C7_0);
    C[7 * ldc + 4] = _C7_1;

    _mm256_storeu_pd(&C[8 * ldc + 0], _C8_0);
    C[8 * ldc + 4] = _C8_1;

    _mm256_storeu_pd(&C[9 * ldc + 0], _C9_0);
    C[9 * ldc + 4] = _C9_1;
}
void micro_kernel6x10(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m256d _A0;
    register __m128d _A1;
    register __m256d _C0_0 = _mm256_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C0_1 = _mm_loadu_pd(&C[0 * ldc + 4]);
    register __m256d _C1_0 = _mm256_loadu_pd(&C[1 * ldc + 0]);
    register __m128d _C1_1 = _mm_loadu_pd(&C[1 * ldc + 4]);
    register __m256d _C2_0 = _mm256_loadu_pd(&C[2 * ldc + 0]);
    register __m128d _C2_1 = _mm_loadu_pd(&C[2 * ldc + 4]);
    register __m256d _C3_0 = _mm256_loadu_pd(&C[3 * ldc + 0]);
    register __m128d _C3_1 = _mm_loadu_pd(&C[3 * ldc + 4]);
    register __m256d _C4_0 = _mm256_loadu_pd(&C[4 * ldc + 0]);
    register __m128d _C4_1 = _mm_loadu_pd(&C[4 * ldc + 4]);
    register __m256d _C5_0 = _mm256_loadu_pd(&C[5 * ldc + 0]);
    register __m128d _C5_1 = _mm_loadu_pd(&C[5 * ldc + 4]);

    register __m256d _C6_0 = _mm256_loadu_pd(&C[6 * ldc + 0]);
    register __m128d _C6_1 = _mm_loadu_pd(&C[6 * ldc + 4]);

    register __m256d _C7_0 = _mm256_loadu_pd(&C[7 * ldc + 0]);
    register __m128d _C7_1 = _mm_loadu_pd(&C[7 * ldc + 4]);

    register __m256d _C8_0 = _mm256_loadu_pd(&C[8 * ldc + 0]);
    register __m128d _C8_1 = _mm_loadu_pd(&C[8 * ldc + 4]);

    register __m256d _C9_0 = _mm256_loadu_pd(&C[9 * ldc + 0]);
    register __m128d _C9_1 = _mm_loadu_pd(&C[9 * ldc + 4]);

    for(i = 0; i < k; ++i) {
        _A0 = _mm256_loadu_pd(&A[0]);
        _A1 = _mm_loadu_pd(&A[4]);
        _C0_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C1_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C2_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C3_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A1, _C3_1);
        _C4_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A1, _C4_1);
        _C5_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A1, _C5_1);

        _C6_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm_fnmadd_pd(_mm_set1_pd(B[6 * ldb]), _A1, _C6_1);

        _C7_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 = _mm_fnmadd_pd(_mm_set1_pd(B[7 * ldb]), _A1, _C7_1);

        _C8_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 = _mm_fnmadd_pd(_mm_set1_pd(B[8 * ldb]), _A1, _C8_1);

        _C9_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[9 * ldb]), _A0, _C9_0);
        _C9_1 = _mm_fnmadd_pd(_mm_set1_pd(B[9 * ldb]), _A1, _C9_1);

        A += lda;
        B += 1;
    }

    _mm256_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[0 * ldc + 4], _C0_1);
    _mm256_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm_storeu_pd(&C[1 * ldc + 4], _C1_1);
    _mm256_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm_storeu_pd(&C[2 * ldc + 4], _C2_1);
    _mm256_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm_storeu_pd(&C[3 * ldc + 4], _C3_1);
    _mm256_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm_storeu_pd(&C[4 * ldc + 4], _C4_1);
    _mm256_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm_storeu_pd(&C[5 * ldc + 4], _C5_1);

    _mm256_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm_storeu_pd(&C[6 * ldc + 4], _C6_1);

    _mm256_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm_storeu_pd(&C[7 * ldc + 4], _C7_1);

    _mm256_storeu_pd(&C[8 * ldc + 0], _C8_0);
    _mm_storeu_pd(&C[8 * ldc + 4], _C8_1);

    _mm256_storeu_pd(&C[9 * ldc + 0], _C9_0);
    _mm_storeu_pd(&C[9 * ldc + 4], _C9_1);
}
void micro_kernel7x10(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m256d _A0;
    register __m128d _A1;
    register double _A2;
    register __m256d _C0_0 = _mm256_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C0_1 = _mm_loadu_pd(&C[0 * ldc + 4]);
    register double _C0_2 = C[0 * ldc + 6];
    register __m256d _C1_0 = _mm256_loadu_pd(&C[1 * ldc + 0]);
    register __m128d _C1_1 = _mm_loadu_pd(&C[1 * ldc + 4]);
    register double _C1_2 = C[1 * ldc + 6];
    register __m256d _C2_0 = _mm256_loadu_pd(&C[2 * ldc + 0]);
    register __m128d _C2_1 = _mm_loadu_pd(&C[2 * ldc + 4]);
    register double _C2_2 = C[2 * ldc + 6];
    register __m256d _C3_0 = _mm256_loadu_pd(&C[3 * ldc + 0]);
    register __m128d _C3_1 = _mm_loadu_pd(&C[3 * ldc + 4]);
    register double _C3_2 = C[3 * ldc + 6];
    register __m256d _C4_0 = _mm256_loadu_pd(&C[4 * ldc + 0]);
    register __m128d _C4_1 = _mm_loadu_pd(&C[4 * ldc + 4]);
    register double _C4_2 = C[4 * ldc + 6];
    register __m256d _C5_0 = _mm256_loadu_pd(&C[5 * ldc + 0]);
    register __m128d _C5_1 = _mm_loadu_pd(&C[5 * ldc + 4]);
    register double _C5_2 = C[5 * ldc + 6];

    register __m256d _C6_0 = _mm256_loadu_pd(&C[6 * ldc + 0]);
    register __m128d _C6_1 = _mm_loadu_pd(&C[6 * ldc + 4]);
    register double _C6_2 = C[6 * ldc + 6];

    register __m256d _C7_0 = _mm256_loadu_pd(&C[7 * ldc + 0]);
    register __m128d _C7_1 = _mm_loadu_pd(&C[7 * ldc + 4]);
    register double _C7_2 = C[7 * ldc + 6];

    register __m256d _C8_0 = _mm256_loadu_pd(&C[8 * ldc + 0]);
    register __m128d _C8_1 = _mm_loadu_pd(&C[8 * ldc + 4]);
    register double _C8_2 = C[8 * ldc + 6];

    register __m256d _C9_0 = _mm256_loadu_pd(&C[9 * ldc + 0]);
    register __m128d _C9_1 = _mm_loadu_pd(&C[9 * ldc + 4]);
    register double _C9_2 = C[9 * ldc + 6];

    for(i = 0; i < k; ++i) {
        _A0 = _mm256_loadu_pd(&A[0]);
        _A1 = _mm_loadu_pd(&A[4]);
        _A2 = A[6];
        _C0_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 += B[0 * ldb] * _A2;
        _C1_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C1_2 += B[1 * ldb] * _A2;
        _C2_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C2_2 += B[2 * ldb] * _A2;
        _C3_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A1, _C3_1);
        _C3_2 += B[3 * ldb] * _A2;
        _C4_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A1, _C4_1);
        _C4_2 += B[4 * ldb] * _A2;
        _C5_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A1, _C5_1);
        _C5_2 += B[5 * ldb] * _A2;

        _C6_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm_fnmadd_pd(_mm_set1_pd(B[6 * ldb]), _A1, _C6_1);
        _C6_2 += B[6 * ldb] * _A2;

        _C7_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 = _mm_fnmadd_pd(_mm_set1_pd(B[7 * ldb]), _A1, _C7_1);
        _C7_2 += B[7 * ldb] * _A2;

        _C8_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 = _mm_fnmadd_pd(_mm_set1_pd(B[8 * ldb]), _A1, _C8_1);
        _C8_2 += B[8 * ldb] * _A2;

        _C9_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[9 * ldb]), _A0, _C9_0);
        _C9_1 = _mm_fnmadd_pd(_mm_set1_pd(B[9 * ldb]), _A1, _C9_1);
        _C9_2 += B[9 * ldb] * _A2;

        A += lda;
        B += 1;
    }

    _mm256_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[0 * ldc + 4], _C0_1);
    C[0 * ldc + 6] = _C0_2;
    _mm256_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm_storeu_pd(&C[1 * ldc + 4], _C1_1);
    C[1 * ldc + 6] = _C1_2;
    _mm256_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm_storeu_pd(&C[2 * ldc + 4], _C2_1);
    C[2 * ldc + 6] = _C2_2;
    _mm256_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm_storeu_pd(&C[3 * ldc + 4], _C3_1);
    C[3 * ldc + 6] = _C3_2;
    _mm256_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm_storeu_pd(&C[4 * ldc + 4], _C4_1);
    C[4 * ldc + 6] = _C4_2;
    _mm256_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm_storeu_pd(&C[5 * ldc + 4], _C5_1);
    C[5 * ldc + 6] = _C5_2;

    _mm256_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm_storeu_pd(&C[6 * ldc + 4], _C6_1);
    C[6 * ldc + 6] = _C6_2;

    _mm256_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm_storeu_pd(&C[7 * ldc + 4], _C7_1);
    C[7 * ldc + 6] = _C7_2;

    _mm256_storeu_pd(&C[8 * ldc + 0], _C8_0);
    _mm_storeu_pd(&C[8 * ldc + 4], _C8_1);
    C[8 * ldc + 6] = _C8_2;

    _mm256_storeu_pd(&C[9 * ldc + 0], _C9_0);
    _mm_storeu_pd(&C[9 * ldc + 4], _C9_1);
    C[9 * ldc + 6] = _C9_2;
}
//=====================================================================================================================
void micro_kernel8x10(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);
    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);

    register __m512d _C7_0 = _mm512_loadu_pd(&C[7 * ldc + 0]);

    register __m512d _C8_0 = _mm512_loadu_pd(&C[8 * ldc + 0]);

    register __m512d _C9_0 = _mm512_loadu_pd(&C[9 * ldc + 0]);

    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);

        _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A0, _C7_0);

        _C8_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8 * ldb]), _A0, _C8_0);

        _C9_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[9 * ldb]), _A0, _C9_0);

        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);

    _mm512_storeu_pd(&C[7 * ldc + 0], _C7_0);

    _mm512_storeu_pd(&C[8 * ldc + 0], _C8_0);

    _mm512_storeu_pd(&C[9 * ldc + 0], _C9_0);
}
void micro_kernel9x10(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register double _A1;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register double _C0_1 = C[0 * ldc + 8];

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);
    register double _C1_1 = C[1 * ldc + 8];

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register double _C2_1 = C[2 * ldc + 8];

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register double _C3_1 = C[3 * ldc + 8];

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register double _C4_1 = C[4 * ldc + 8];

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register double _C5_1 = C[5 * ldc + 8];

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);
    register double _C6_1 = C[6 * ldc + 8];

    register __m512d _C7_0 = _mm512_loadu_pd(&C[7 * ldc + 0]);
    register double _C7_1 = C[7 * ldc + 8];

    register __m512d _C8_0 = _mm512_loadu_pd(&C[8 * ldc + 0]);
    register double _C8_1 = C[8 * ldc + 8];

    register __m512d _C9_0 = _mm512_loadu_pd(&C[9 * ldc + 0]);
    register double _C9_1 = C[9 * ldc + 8];

    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = A[8];

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 += B[0 * ldb] * _A1;

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 += B[1 * ldb] * _A1;

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 += B[2 * ldb] * _A1;

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 += B[3 * ldb] * _A1;

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 += B[4 * ldb] * _A1;

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 += B[5 * ldb] * _A1;

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 += B[6 * ldb] * _A1;
        
        _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 += B[7 * ldb] * _A1;

        _C8_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 += B[8 * ldb] * _A1;

        _C9_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[9 * ldb]), _A0, _C9_0);
        _C9_1 += B[9 * ldb] * _A1;

        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    C[0 * ldc + 8] = _C0_1;

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    C[1 * ldc + 8] = _C1_1;

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    C[2 * ldc + 8] = _C2_1;

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    C[3 * ldc + 8] = _C3_1;

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    C[4 * ldc + 8] = _C4_1;

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    C[5 * ldc + 8] = _C5_1;

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);
    C[6 * ldc + 8] = _C6_1;

    _mm512_storeu_pd(&C[7 * ldc + 0], _C7_0);
    C[7 * ldc + 8] = _C7_1;

    _mm512_storeu_pd(&C[8 * ldc + 0], _C8_0);
    C[8 * ldc + 8] = _C8_1;

    _mm512_storeu_pd(&C[9 * ldc + 0], _C9_0);
    C[9 * ldc + 8] = _C9_1;
}
void micro_kernel10x10(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m128d _A1;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C0_1 = _mm_loadu_pd(&C[0 * ldc + 8]);

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m128d _C1_1 = _mm_loadu_pd(&C[1 * ldc + 8]);

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m128d _C2_1 = _mm_loadu_pd(&C[2 * ldc + 8]);

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m128d _C3_1 = _mm_loadu_pd(&C[3 * ldc + 8]);

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m128d _C4_1 = _mm_loadu_pd(&C[4 * ldc + 8]);

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register __m128d _C5_1 = _mm_loadu_pd(&C[5 * ldc + 8]);

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);
    register __m128d _C6_1 = _mm_loadu_pd(&C[6 * ldc + 8]);

    register __m512d _C7_0 = _mm512_loadu_pd(&C[7 * ldc + 0]);
    register __m128d _C7_1 = _mm_loadu_pd(&C[7 * ldc + 8]);

    register __m512d _C8_0 = _mm512_loadu_pd(&C[8 * ldc + 0]);
    register __m128d _C8_1 = _mm_loadu_pd(&C[8 * ldc + 8]);

    register __m512d _C9_0 = _mm512_loadu_pd(&C[9 * ldc + 0]);
    register __m128d _C9_1 = _mm_loadu_pd(&C[9 * ldc + 8]);
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm_loadu_pd(&A[8]);

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A1, _C0_1);

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A1, _C1_1);

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A1, _C2_1);

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A1, _C3_1);

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A1, _C4_1);

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A1, _C5_1);

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm_fnmadd_pd(_mm_set1_pd(B[6 * ldb]), _A1, _C6_1);

        _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 = _mm_fnmadd_pd(_mm_set1_pd(B[7 * ldb]), _A1, _C7_1);

        _C8_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 = _mm_fnmadd_pd(_mm_set1_pd(B[8 * ldb]), _A1, _C8_1);

        _C9_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[9 * ldb]), _A0, _C9_0);
        _C9_1 = _mm_fnmadd_pd(_mm_set1_pd(B[9 * ldb]), _A1, _C9_1);
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[0 * ldc + 8], _C0_1);

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm_storeu_pd(&C[1 * ldc + 8], _C1_1);

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm_storeu_pd(&C[2 * ldc + 8], _C2_1);

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm_storeu_pd(&C[3 * ldc + 8], _C3_1);

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm_storeu_pd(&C[4 * ldc + 8], _C4_1);

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm_storeu_pd(&C[5 * ldc + 8], _C5_1);

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm_storeu_pd(&C[6 * ldc + 8], _C6_1);

    _mm512_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm_storeu_pd(&C[7 * ldc + 8], _C7_1);

    _mm512_storeu_pd(&C[8 * ldc + 0], _C8_0);
    _mm_storeu_pd(&C[8 * ldc + 8], _C8_1);

    _mm512_storeu_pd(&C[9 * ldc + 0], _C9_0);
    _mm_storeu_pd(&C[9 * ldc + 8], _C9_1);
}
void micro_kernel11x10(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m128d _A1;
    register double _A2;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C0_1 = _mm_loadu_pd(&C[0 * ldc + 8]);
    register double _C0_2 = C[0 * ldc + 10];

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m128d _C1_1 = _mm_loadu_pd(&C[1 * ldc + 8]);
    register double _C1_2 = C[1 * ldc + 10];

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m128d _C2_1 = _mm_loadu_pd(&C[2 * ldc + 8]);
    register double _C2_2 = C[2 * ldc + 10];

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m128d _C3_1 = _mm_loadu_pd(&C[3 * ldc + 8]);
    register double _C3_2 = C[3 * ldc + 10];

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m128d _C4_1 = _mm_loadu_pd(&C[4 * ldc + 8]);
    register double _C4_2 = C[4 * ldc + 10];

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register __m128d _C5_1 = _mm_loadu_pd(&C[5 * ldc + 8]);
    register double _C5_2 = C[5 * ldc + 10];

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);
    register __m128d _C6_1 = _mm_loadu_pd(&C[6 * ldc + 8]);
    register double _C6_2 = C[6 * ldc + 10];

    register __m512d _C7_0 = _mm512_loadu_pd(&C[7 * ldc + 0]);
    register __m128d _C7_1 = _mm_loadu_pd(&C[7 * ldc + 8]);
    register double _C7_2 = C[7 * ldc + 10];

    register __m512d _C8_0 = _mm512_loadu_pd(&C[8 * ldc + 0]);
    register __m128d _C8_1 = _mm_loadu_pd(&C[8 * ldc + 8]);
    register double _C8_2 = C[8 * ldc + 10];

    register __m512d _C9_0 = _mm512_loadu_pd(&C[9 * ldc + 0]);
    register __m128d _C9_1 = _mm_loadu_pd(&C[9 * ldc + 8]);
    register double _C9_2 = C[9 * ldc + 10];
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm_loadu_pd(&A[8]);
        _A2 = A[10];

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 += B[0 * ldb] * _A2;

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C1_2 += B[1 * ldb] * _A2;

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C2_2 += B[2 * ldb] * _A2;

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A1, _C3_1);
        _C3_2 += B[3 * ldb] * _A2;

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A1, _C4_1);
        _C4_2 += B[4 * ldb] * _A2;

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A1, _C5_1);
        _C5_2 += B[5 * ldb] * _A2;

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm_fnmadd_pd(_mm_set1_pd(B[6 * ldb]), _A1, _C6_1);
        _C6_2 += B[6 * ldb] * _A2;

        _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 = _mm_fnmadd_pd(_mm_set1_pd(B[7 * ldb]), _A1, _C7_1);
        _C7_2 += B[7 * ldb] * _A2;

        _C8_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 = _mm_fnmadd_pd(_mm_set1_pd(B[8 * ldb]), _A1, _C8_1);
        _C8_2 += B[8 * ldb] * _A2;

        _C9_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[9 * ldb]), _A0, _C9_0);
        _C9_1 = _mm_fnmadd_pd(_mm_set1_pd(B[9 * ldb]), _A1, _C9_1);
        _C9_2 += B[9 * ldb] * _A2;
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[0 * ldc + 8], _C0_1);
    C[0 * ldc + 10] = _C0_2;

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm_storeu_pd(&C[1 * ldc + 8], _C1_1);
    C[1 * ldc + 10] = _C1_2;

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm_storeu_pd(&C[2 * ldc + 8], _C2_1);
    C[2 * ldc + 10] = _C2_2;

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm_storeu_pd(&C[3 * ldc + 8], _C3_1);
    C[3 * ldc + 10] = _C3_2;

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm_storeu_pd(&C[4 * ldc + 8], _C4_1);
    C[4 * ldc + 10] = _C4_2;

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm_storeu_pd(&C[5 * ldc + 8], _C5_1);
    C[5 * ldc + 10] = _C5_2;

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm_storeu_pd(&C[6 * ldc + 8], _C6_1);
    C[6 * ldc + 10] = _C6_2;

    _mm512_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm_storeu_pd(&C[7 * ldc + 8], _C7_1);
    C[7 * ldc + 10] = _C7_2;

    _mm512_storeu_pd(&C[8 * ldc + 0], _C8_0);
    _mm_storeu_pd(&C[8 * ldc + 8], _C8_1);
    C[8 * ldc + 10] = _C8_2;

    _mm512_storeu_pd(&C[9 * ldc + 0], _C9_0);
    _mm_storeu_pd(&C[9 * ldc + 8], _C9_1);
    C[9 * ldc + 10] = _C9_2;
}
void micro_kernel12x10(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {//------------------------------------
     int i;
    register __m512d _A0;
    register __m256d _A1;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C0_1 = _mm256_loadu_pd(&C[0 * ldc + 8]);

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m256d _C1_1 = _mm256_loadu_pd(&C[1 * ldc + 8]);

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m256d _C2_1 = _mm256_loadu_pd(&C[2 * ldc + 8]);

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m256d _C3_1 = _mm256_loadu_pd(&C[3 * ldc + 8]);

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m256d _C4_1 = _mm256_loadu_pd(&C[4 * ldc + 8]);

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register __m256d _C5_1 = _mm256_loadu_pd(&C[5 * ldc + 8]);

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);
    register __m256d _C6_1 = _mm256_loadu_pd(&C[6 * ldc + 8]);

    register __m512d _C7_0 = _mm512_loadu_pd(&C[7 * ldc + 0]);
    register __m256d _C7_1 = _mm256_loadu_pd(&C[7 * ldc + 8]);

    register __m512d _C8_0 = _mm512_loadu_pd(&C[8 * ldc + 0]);
    register __m256d _C8_1 = _mm256_loadu_pd(&C[8 * ldc + 8]);

    register __m512d _C9_0 = _mm512_loadu_pd(&C[9 * ldc + 0]);
    register __m256d _C9_1 = _mm256_loadu_pd(&C[9 * ldc + 8]);
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm256_loadu_pd(&A[8]);

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A1, _C0_1);

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A1, _C1_1);

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A1, _C2_1);

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A1, _C3_1);

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A1, _C4_1);

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A1, _C5_1);

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[6 * ldb]), _A1, _C6_1);

        _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[7 * ldb]), _A1, _C7_1);

        _C8_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[8 * ldb]), _A1, _C8_1);

        _C9_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[9 * ldb]), _A0, _C9_0);
        _C9_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[9 * ldb]), _A1, _C9_1);
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[0 * ldc + 8], _C0_1);

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm256_storeu_pd(&C[1 * ldc + 8], _C1_1);

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm256_storeu_pd(&C[2 * ldc + 8], _C2_1);

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm256_storeu_pd(&C[3 * ldc + 8], _C3_1);

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm256_storeu_pd(&C[4 * ldc + 8], _C4_1);

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm256_storeu_pd(&C[5 * ldc + 8], _C5_1);

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm256_storeu_pd(&C[6 * ldc + 8], _C6_1);

    _mm512_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm256_storeu_pd(&C[7 * ldc + 8], _C7_1);

    _mm512_storeu_pd(&C[8 * ldc + 0], _C8_0);
    _mm256_storeu_pd(&C[8 * ldc + 8], _C8_1);

    _mm512_storeu_pd(&C[9 * ldc + 0], _C9_0);
    _mm256_storeu_pd(&C[9 * ldc + 8], _C9_1);
}
void micro_kernel13x10(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m256d _A1;
    register double _A2;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C0_1 = _mm256_loadu_pd(&C[0 * ldc + 8]);
    register double _C0_2 = C[0 * ldc + 12];

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m256d _C1_1 = _mm256_loadu_pd(&C[1 * ldc + 8]);
    register double _C1_2 = C[1 * ldc + 12];

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m256d _C2_1 = _mm256_loadu_pd(&C[2 * ldc + 8]);
    register double _C2_2 = C[2 * ldc + 12];

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m256d _C3_1 = _mm256_loadu_pd(&C[3 * ldc + 8]);
    register double _C3_2 = C[3 * ldc + 12];

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m256d _C4_1 = _mm256_loadu_pd(&C[4 * ldc + 8]);
    register double _C4_2 = C[4 * ldc + 12];

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register __m256d _C5_1 = _mm256_loadu_pd(&C[5 * ldc + 8]);
    register double _C5_2 = C[5 * ldc + 12];

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);
    register __m256d _C6_1 = _mm256_loadu_pd(&C[6 * ldc + 8]);
    register double _C6_2 = C[6 * ldc + 12];

    register __m512d _C7_0 = _mm512_loadu_pd(&C[7 * ldc + 0]);
    register __m256d _C7_1 = _mm256_loadu_pd(&C[7 * ldc + 8]);
    register double _C7_2 = C[7 * ldc + 12];

    register __m512d _C8_0 = _mm512_loadu_pd(&C[8 * ldc + 0]);
    register __m256d _C8_1 = _mm256_loadu_pd(&C[8 * ldc + 8]);
    register double _C8_2 = C[8 * ldc + 12];

    register __m512d _C9_0 = _mm512_loadu_pd(&C[9 * ldc + 0]);
    register __m256d _C9_1 = _mm256_loadu_pd(&C[9 * ldc + 8]);
    register double _C9_2 = C[9 * ldc + 12];
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm256_loadu_pd(&A[8]);
        _A2 = A[12];

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 += B[0 * ldb] * _A2;

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C1_2 += B[1 * ldb] * _A2;

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C2_2 += B[2 * ldb] * _A2;

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A1, _C3_1);
        _C3_2 += B[3 * ldb] * _A2;

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A1, _C4_1);
        _C4_2 += B[4 * ldb] * _A2;

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A1, _C5_1);
        _C5_2 += B[5 * ldb] * _A2;

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[6 * ldb]), _A1, _C6_1);
        _C6_2 += B[6 * ldb] * _A2;

        _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[7 * ldb]), _A1, _C7_1);
        _C7_2 += B[7 * ldb] * _A2;

        _C8_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[8 * ldb]), _A1, _C8_1);
        _C8_2 += B[8 * ldb] * _A2;

        _C9_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[9 * ldb]), _A0, _C9_0);
        _C9_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[9 * ldb]), _A1, _C9_1);
        _C9_2 += B[9 * ldb] * _A2;
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[0 * ldc + 8], _C0_1);
    C[0 * ldc + 12] = _C0_2;

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm256_storeu_pd(&C[1 * ldc + 8], _C1_1);
    C[1 * ldc + 12] = _C1_2;

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm256_storeu_pd(&C[2 * ldc + 8], _C2_1);
    C[2 * ldc + 12] = _C2_2;

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm256_storeu_pd(&C[3 * ldc + 8], _C3_1);
    C[3 * ldc + 12] = _C3_2;

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm256_storeu_pd(&C[4 * ldc + 8], _C4_1);
    C[4 * ldc + 12] = _C4_2;

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm256_storeu_pd(&C[5 * ldc + 8], _C5_1);
    C[5 * ldc + 12] = _C5_2;

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm256_storeu_pd(&C[6 * ldc + 8], _C6_1);
    C[6 * ldc + 12] = _C6_2;

    _mm512_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm256_storeu_pd(&C[7 * ldc + 8], _C7_1);
    C[7 * ldc + 12] = _C7_2;

    _mm512_storeu_pd(&C[8 * ldc + 0], _C8_0);
    _mm256_storeu_pd(&C[8 * ldc + 8], _C8_1);
    C[8 * ldc + 12] = _C8_2;

    _mm512_storeu_pd(&C[9 * ldc + 0], _C9_0);
    _mm256_storeu_pd(&C[9 * ldc + 8], _C9_1);
    C[9 * ldc + 12] = _C9_2;
}
void micro_kernel14x10(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m256d _A1;
    register __m128d _A2;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C0_1 = _mm256_loadu_pd(&C[0 * ldc + 8]);
    register __m128d _C0_2 = _mm_loadu_pd(&C[0 * ldc + 12]);

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m256d _C1_1 = _mm256_loadu_pd(&C[1 * ldc + 8]);
    register __m128d _C1_2 = _mm_loadu_pd(&C[1 * ldc + 12]);

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m256d _C2_1 = _mm256_loadu_pd(&C[2 * ldc + 8]);
    register __m128d _C2_2 = _mm_loadu_pd(&C[2 * ldc + 12]);

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m256d _C3_1 = _mm256_loadu_pd(&C[3 * ldc + 8]);
    register __m128d _C3_2 = _mm_loadu_pd(&C[3 * ldc + 12]);

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m256d _C4_1 = _mm256_loadu_pd(&C[4 * ldc + 8]);
    register __m128d _C4_2 = _mm_loadu_pd(&C[4 * ldc + 12]);

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register __m256d _C5_1 = _mm256_loadu_pd(&C[5 * ldc + 8]);
    register __m128d _C5_2 = _mm_loadu_pd(&C[5 * ldc + 12]);

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);
    register __m256d _C6_1 = _mm256_loadu_pd(&C[6 * ldc + 8]);
    register __m128d _C6_2 = _mm_loadu_pd(&C[6 * ldc + 12]);

    register __m512d _C7_0 = _mm512_loadu_pd(&C[7 * ldc + 0]);
    register __m256d _C7_1 = _mm256_loadu_pd(&C[7 * ldc + 8]);
    register __m128d _C7_2 = _mm_loadu_pd(&C[7 * ldc + 12]);

    register __m512d _C8_0 = _mm512_loadu_pd(&C[8 * ldc + 0]);
    register __m256d _C8_1 = _mm256_loadu_pd(&C[8 * ldc + 8]);
    register __m128d _C8_2 = _mm_loadu_pd(&C[8 * ldc + 12]);

    register __m512d _C9_0 = _mm512_loadu_pd(&C[9 * ldc + 0]);
    register __m256d _C9_1 = _mm256_loadu_pd(&C[9 * ldc + 8]);
    register __m128d _C9_2 = _mm_loadu_pd(&C[9 * ldc + 12]);
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm256_loadu_pd(&A[8]);
        _A2 = _mm_loadu_pd(&A[12]);

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A2, _C0_2);

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C1_2 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A2, _C1_2);

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C2_2 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A2, _C2_2);

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A1, _C3_1);
        _C3_2 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A2, _C3_2);

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A1, _C4_1);
        _C4_2 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A2, _C4_2);

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A1, _C5_1);
        _C5_2 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A2, _C5_2);

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[6 * ldb]), _A1, _C6_1);
        _C6_2 = _mm_fnmadd_pd(_mm_set1_pd(B[6 * ldb]), _A2, _C6_2);

        _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[7 * ldb]), _A1, _C7_1);
        _C7_2 = _mm_fnmadd_pd(_mm_set1_pd(B[7 * ldb]), _A2, _C7_2);

        _C8_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[8 * ldb]), _A1, _C8_1);
        _C8_2 = _mm_fnmadd_pd(_mm_set1_pd(B[8 * ldb]), _A2, _C8_2);

        _C9_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[9 * ldb]), _A0, _C9_0);
        _C9_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[9 * ldb]), _A1, _C9_1);
        _C9_2 = _mm_fnmadd_pd(_mm_set1_pd(B[9 * ldb]), _A2, _C9_2);
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[0 * ldc + 8], _C0_1);
    _mm_storeu_pd(&C[0 * ldc + 12], _C0_2);

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm256_storeu_pd(&C[1 * ldc + 8], _C1_1);
    _mm_storeu_pd(&C[1 * ldc + 12], _C1_2);

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm256_storeu_pd(&C[2 * ldc + 8], _C2_1);
    _mm_storeu_pd(&C[2 * ldc + 12], _C2_2);

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm256_storeu_pd(&C[3 * ldc + 8], _C3_1);
    _mm_storeu_pd(&C[3 * ldc + 12], _C3_2);

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm256_storeu_pd(&C[4 * ldc + 8], _C4_1);
    _mm_storeu_pd(&C[4 * ldc + 12], _C4_2);

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm256_storeu_pd(&C[5 * ldc + 8], _C5_1);
    _mm_storeu_pd(&C[5 * ldc + 12], _C5_2);

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm256_storeu_pd(&C[6 * ldc + 8], _C6_1);
    _mm_storeu_pd(&C[6 * ldc + 12], _C6_2);

    _mm512_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm256_storeu_pd(&C[7 * ldc + 8], _C7_1);
    _mm_storeu_pd(&C[7 * ldc + 12], _C7_2);

    _mm512_storeu_pd(&C[8 * ldc + 0], _C8_0);
    _mm256_storeu_pd(&C[8 * ldc + 8], _C8_1);
    _mm_storeu_pd(&C[8 * ldc + 12], _C8_2);

    _mm512_storeu_pd(&C[9 * ldc + 0], _C9_0);
    _mm256_storeu_pd(&C[9 * ldc + 8], _C9_1);
    _mm_storeu_pd(&C[9 * ldc + 12], _C9_2);
}
void micro_kernel15x10(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m256d _A1;
    register __m128d _A2;
    register double _A3;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C0_1 = _mm256_loadu_pd(&C[0 * ldc + 8]);
    register __m128d _C0_2 = _mm_loadu_pd(&C[0 * ldc + 12]);
    register double _C0_3 = C[0 * ldc + 14];

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m256d _C1_1 = _mm256_loadu_pd(&C[1 * ldc + 8]);
    register __m128d _C1_2 = _mm_loadu_pd(&C[1 * ldc + 12]);
    register double _C1_3 = C[1 * ldc + 14];

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m256d _C2_1 = _mm256_loadu_pd(&C[2 * ldc + 8]);
    register __m128d _C2_2 = _mm_loadu_pd(&C[2 * ldc + 12]);
    register double _C2_3 = C[2 * ldc + 14];

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m256d _C3_1 = _mm256_loadu_pd(&C[3 * ldc + 8]);
    register __m128d _C3_2 = _mm_loadu_pd(&C[3 * ldc + 12]);
    register double _C3_3 = C[3 * ldc + 14];

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m256d _C4_1 = _mm256_loadu_pd(&C[4 * ldc + 8]);
    register __m128d _C4_2 = _mm_loadu_pd(&C[4 * ldc + 12]);
    register double _C4_3 = C[4 * ldc + 14];

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register __m256d _C5_1 = _mm256_loadu_pd(&C[5 * ldc + 8]);
    register __m128d _C5_2 = _mm_loadu_pd(&C[5 * ldc + 12]);
    register double _C5_3 = C[5 * ldc + 14];

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);
    register __m256d _C6_1 = _mm256_loadu_pd(&C[6 * ldc + 8]);
    register __m128d _C6_2 = _mm_loadu_pd(&C[6 * ldc + 12]);
    register double _C6_3 = C[6 * ldc + 14];

    register __m512d _C7_0 = _mm512_loadu_pd(&C[7 * ldc + 0]);
    register __m256d _C7_1 = _mm256_loadu_pd(&C[7 * ldc + 8]);
    register __m128d _C7_2 = _mm_loadu_pd(&C[7 * ldc + 12]);
    register double _C7_3 = C[7 * ldc + 14];

    register __m512d _C8_0 = _mm512_loadu_pd(&C[8 * ldc + 0]);
    register __m256d _C8_1 = _mm256_loadu_pd(&C[8 * ldc + 8]);
    register __m128d _C8_2 = _mm_loadu_pd(&C[8 * ldc + 12]);
    register double _C8_3 = C[8 * ldc + 14];

    register __m512d _C9_0 = _mm512_loadu_pd(&C[9 * ldc + 0]);
    register __m256d _C9_1 = _mm256_loadu_pd(&C[9 * ldc + 8]);
    register __m128d _C9_2 = _mm_loadu_pd(&C[9 * ldc + 12]);
    register double _C9_3 = C[9 * ldc + 14];
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm256_loadu_pd(&A[8]);
        _A2 = _mm_loadu_pd(&A[12]);
        _A3 = A[14];

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A2, _C0_2);
        _C0_3 += B[0 * ldb] * _A3;

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C1_2 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A2, _C1_2);
        _C1_3 += B[1 * ldb] * _A3;

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C2_2 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A2, _C2_2);
        _C2_3 += B[2 * ldb] * _A3;

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A1, _C3_1);
        _C3_2 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A2, _C3_2);
        _C3_3 += B[3 * ldb] * _A3;

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A1, _C4_1);
        _C4_2 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A2, _C4_2);
        _C4_3 += B[4 * ldb] * _A3;

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A1, _C5_1);
        _C5_2 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A2, _C5_2);
        _C5_3 += B[5 * ldb] * _A3;

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[6 * ldb]), _A1, _C6_1);
        _C6_2 = _mm_fnmadd_pd(_mm_set1_pd(B[6 * ldb]), _A2, _C6_2);
        _C6_3 += B[6 * ldb] * _A3;

        _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[7 * ldb]), _A1, _C7_1);
        _C7_2 = _mm_fnmadd_pd(_mm_set1_pd(B[7 * ldb]), _A2, _C7_2);
        _C7_3 += B[7 * ldb] * _A3;

        _C8_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[8 * ldb]), _A1, _C8_1);
        _C8_2 = _mm_fnmadd_pd(_mm_set1_pd(B[8 * ldb]), _A2, _C8_2);
        _C8_3 += B[8 * ldb] * _A3;

        _C9_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[9 * ldb]), _A0, _C9_0);
        _C9_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[9 * ldb]), _A1, _C9_1);
        _C9_2 = _mm_fnmadd_pd(_mm_set1_pd(B[9 * ldb]), _A2, _C9_2);
        _C9_3 += B[9 * ldb] * _A3;
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[0 * ldc + 8], _C0_1);
    _mm_storeu_pd(&C[0 * ldc + 12], _C0_2);
    C[0 * ldc + 14] = _C0_3;

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm256_storeu_pd(&C[1 * ldc + 8], _C1_1);
    _mm_storeu_pd(&C[1 * ldc + 12], _C1_2);
    C[1 * ldc + 14] = _C1_3;

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm256_storeu_pd(&C[2 * ldc + 8], _C2_1);
    _mm_storeu_pd(&C[2 * ldc + 12], _C2_2);
    C[2 * ldc + 14] = _C2_3;

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm256_storeu_pd(&C[3 * ldc + 8], _C3_1);
    _mm_storeu_pd(&C[3 * ldc + 12], _C3_2);
    C[3 * ldc + 14] = _C3_3;

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm256_storeu_pd(&C[4 * ldc + 8], _C4_1);
    _mm_storeu_pd(&C[4 * ldc + 12], _C4_2);
    C[4 * ldc + 14] = _C4_3;

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm256_storeu_pd(&C[5 * ldc + 8], _C5_1);
    _mm_storeu_pd(&C[5 * ldc + 12], _C5_2);
    C[5 * ldc + 14] = _C5_3;

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm256_storeu_pd(&C[6 * ldc + 8], _C6_1);
    _mm_storeu_pd(&C[6 * ldc + 12], _C6_2);
    C[6 * ldc + 14] = _C6_3;

    _mm512_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm256_storeu_pd(&C[7 * ldc + 8], _C7_1);
    _mm_storeu_pd(&C[7 * ldc + 12], _C7_2);
    C[7 * ldc + 14] = _C7_3;

    _mm512_storeu_pd(&C[8 * ldc + 0], _C8_0);
    _mm256_storeu_pd(&C[8 * ldc + 8], _C8_1);
    _mm_storeu_pd(&C[8 * ldc + 12], _C8_2);
    C[8 * ldc + 14] = _C8_3;

    _mm512_storeu_pd(&C[9 * ldc + 0], _C9_0);
    _mm256_storeu_pd(&C[9 * ldc + 8], _C9_1);
    _mm_storeu_pd(&C[9 * ldc + 12], _C9_2);
    C[9 * ldc + 14] = _C9_3;
}
void micro_kernel16x10(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m512d _A1;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m512d _C0_1 = _mm512_loadu_pd(&C[0 * ldc + 8]);

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m512d _C1_1 = _mm512_loadu_pd(&C[1 * ldc + 8]);

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m512d _C2_1 = _mm512_loadu_pd(&C[2 * ldc + 8]);

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m512d _C3_1 = _mm512_loadu_pd(&C[3 * ldc + 8]);

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m512d _C4_1 = _mm512_loadu_pd(&C[4 * ldc + 8]);

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register __m512d _C5_1 = _mm512_loadu_pd(&C[5 * ldc + 8]);

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);
    register __m512d _C6_1 = _mm512_loadu_pd(&C[6 * ldc + 8]);

    register __m512d _C7_0 = _mm512_loadu_pd(&C[7 * ldc + 0]);
    register __m512d _C7_1 = _mm512_loadu_pd(&C[7 * ldc + 8]);

    register __m512d _C8_0 = _mm512_loadu_pd(&C[8 * ldc + 0]);
    register __m512d _C8_1 = _mm512_loadu_pd(&C[8 * ldc + 8]);

    register __m512d _C9_0 = _mm512_loadu_pd(&C[9 * ldc + 0]);
    register __m512d _C9_1 = _mm512_loadu_pd(&C[9 * ldc + 8]);
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm512_loadu_pd(&A[8]);

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A1, _C0_1);

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A1, _C1_1);

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A1, _C2_1);

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A1, _C3_1);

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A1, _C4_1);

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A1, _C5_1);

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A1, _C6_1);

        _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A1, _C7_1);

        _C8_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8 * ldb]), _A1, _C8_1);

        _C9_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[9 * ldb]), _A0, _C9_0);
        _C9_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[9 * ldb]), _A1, _C9_1);
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm512_storeu_pd(&C[0 * ldc + 8], _C0_1);

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm512_storeu_pd(&C[1 * ldc + 8], _C1_1);

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm512_storeu_pd(&C[2 * ldc + 8], _C2_1);

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm512_storeu_pd(&C[3 * ldc + 8], _C3_1);

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm512_storeu_pd(&C[4 * ldc + 8], _C4_1);

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm512_storeu_pd(&C[5 * ldc + 8], _C5_1);

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm512_storeu_pd(&C[6 * ldc + 8], _C6_1);

    _mm512_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm512_storeu_pd(&C[7 * ldc + 8], _C7_1);

    _mm512_storeu_pd(&C[8 * ldc + 0], _C8_0);
    _mm512_storeu_pd(&C[8 * ldc + 8], _C8_1);

    _mm512_storeu_pd(&C[9 * ldc + 0], _C9_0);
    _mm512_storeu_pd(&C[9 * ldc + 8], _C9_1);
}

//################################@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@####################################
void micro_kernel1x11(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register double _A0;
    register double _C0_0 = C[0 * ldc + 0];
    register double _C1_0 = C[1 * ldc + 0];
    register double _C2_0 = C[2 * ldc + 0];
    register double _C3_0 = C[3 * ldc + 0];
    register double _C4_0 = C[4 * ldc + 0];
    register double _C5_0 = C[5 * ldc + 0];
    register double _C6_0 = C[6 * ldc + 0];
    register double _C7_0 = C[7 * ldc + 0];
    register double _C8_0 = C[8 * ldc + 0];
    register double _C9_0 = C[9 * ldc + 0];
    register double _C10_0 = C[10 * ldc + 0];

    for(i = 0; i < k; ++i) {
        _A0 = A[0];
        _C0_0 += B[0 * ldb] * _A0;
        _C1_0 += B[1 * ldb] * _A0;
        _C2_0 += B[2 * ldb] * _A0;
        _C3_0 += B[3 * ldb] * _A0;
        _C4_0 += B[4 * ldb] * _A0;
        _C5_0 += B[5 * ldb] * _A0;
        _C6_0 += B[6 * ldb] * _A0;
        _C7_0 += B[7 * ldb] * _A0;
        _C8_0 += B[8 * ldb] * _A0;
        _C9_0 += B[9 * ldb] * _A0;
        _C10_0 += B[10 * ldb] * _A0;

        A += lda;
        B += 1;
    }

    C[0 * ldc + 0] = _C0_0;
    C[1 * ldc + 0] = _C1_0;
    C[2 * ldc + 0] = _C2_0;
    C[3 * ldc + 0] = _C3_0;
    C[4 * ldc + 0] = _C4_0;
    C[5 * ldc + 0] = _C5_0;
    C[6 * ldc + 0] = _C6_0;
    C[7 * ldc + 0] = _C7_0;
    C[8 * ldc + 0] = _C8_0;
    C[9 * ldc + 0] = _C9_0;
    C[10 * ldc + 0] = _C10_0;
}
void micro_kernel2x11(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m128d _A0;
    register __m128d _C0_0 = _mm_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C1_0 = _mm_loadu_pd(&C[1 * ldc + 0]);
    register __m128d _C2_0 = _mm_loadu_pd(&C[2 * ldc + 0]);
    register __m128d _C3_0 = _mm_loadu_pd(&C[3 * ldc + 0]);
    register __m128d _C4_0 = _mm_loadu_pd(&C[4 * ldc + 0]);
    register __m128d _C5_0 = _mm_loadu_pd(&C[5 * ldc + 0]);
    register __m128d _C6_0 = _mm_loadu_pd(&C[6 * ldc + 0]);
    register __m128d _C7_0 = _mm_loadu_pd(&C[7 * ldc + 0]);
    register __m128d _C8_0 = _mm_loadu_pd(&C[8 * ldc + 0]);
    register __m128d _C9_0 = _mm_loadu_pd(&C[9 * ldc + 0]);
    register __m128d _C10_0 = _mm_loadu_pd(&C[10 * ldc + 0]);

    for(i = 0; i < k; ++i) {
        _A0 = _mm_loadu_pd(&A[0]);
        _C0_0 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C1_0 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C2_0 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C3_0 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C4_0 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C5_0 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C6_0 = _mm_fnmadd_pd(_mm_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C7_0 = _mm_fnmadd_pd(_mm_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C8_0 = _mm_fnmadd_pd(_mm_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C9_0 = _mm_fnmadd_pd(_mm_set1_pd(B[9 * ldb]), _A0, _C9_0);
        _C10_0 = _mm_fnmadd_pd(_mm_set1_pd(B[10 * ldb]), _A0, _C10_0);

        A += lda;
        B += 1;
    }

    _mm_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm_storeu_pd(&C[8 * ldc + 0], _C8_0);
    _mm_storeu_pd(&C[9 * ldc + 0], _C9_0);
    _mm_storeu_pd(&C[10 * ldc + 0], _C10_0);
}
void micro_kernel3x11(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m128d _A0;
    register double _A1;
    register __m128d _C0_0 = _mm_loadu_pd(&C[0 * ldc + 0]);
    register double _C0_1 = C[0 * ldc + 2];
    register __m128d _C1_0 = _mm_loadu_pd(&C[1 * ldc + 0]);
    register double _C1_1 = C[1 * ldc + 2];
    register __m128d _C2_0 = _mm_loadu_pd(&C[2 * ldc + 0]);
    register double _C2_1 = C[2 * ldc + 2];
    register __m128d _C3_0 = _mm_loadu_pd(&C[3 * ldc + 0]);
    register double _C3_1 = C[3 * ldc + 2];
    register __m128d _C4_0 = _mm_loadu_pd(&C[4 * ldc + 0]);
    register double _C4_1 = C[4 * ldc + 2];
    register __m128d _C5_0 = _mm_loadu_pd(&C[5 * ldc + 0]);
    register double _C5_1 = C[5 * ldc + 2];

    register __m128d _C6_0 = _mm_loadu_pd(&C[6 * ldc + 0]);
    register double _C6_1 = C[6 * ldc + 2];

    register __m128d _C7_0 = _mm_loadu_pd(&C[7 * ldc + 0]);
    register double _C7_1 = C[7 * ldc + 2];

    register __m128d _C8_0 = _mm_loadu_pd(&C[8 * ldc + 0]);
    register double _C8_1 = C[8 * ldc + 2];

    register __m128d _C9_0 = _mm_loadu_pd(&C[9 * ldc + 0]);
    register double _C9_1 = C[9 * ldc + 2];

    register __m128d _C10_0 = _mm_loadu_pd(&C[10 * ldc + 0]);
    register double _C10_1 = C[10 * ldc + 2];

    for(i = 0; i < k; ++i) {
        _A0 = _mm_loadu_pd(&A[0]);
        _A1 = A[2];
        _C0_0 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 += B[0 * ldb] * _A1;
        _C1_0 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 += B[1 * ldb] * _A1;
        _C2_0 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 += B[2 * ldb] * _A1;
        _C3_0 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 += B[3 * ldb] * _A1;
        _C4_0 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 += B[4 * ldb] * _A1;
        _C5_0 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 += B[5 * ldb] * _A1;

        _C6_0 = _mm_fnmadd_pd(_mm_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 += B[6 * ldb] * _A1;

        _C7_0 = _mm_fnmadd_pd(_mm_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 += B[7 * ldb] * _A1;

        _C8_0 = _mm_fnmadd_pd(_mm_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 += B[8 * ldb] * _A1;

        _C9_0 = _mm_fnmadd_pd(_mm_set1_pd(B[9 * ldb]), _A0, _C9_0);
        _C9_1 += B[9 * ldb] * _A1;

        _C10_0 = _mm_fnmadd_pd(_mm_set1_pd(B[10 * ldb]), _A0, _C10_0);
        _C10_1 += B[10 * ldb] * _A1;

        A += lda;
        B += 1;
    }

    _mm_storeu_pd(&C[0 * ldc + 0], _C0_0);
    C[0 * ldc + 2] = _C0_1;
    _mm_storeu_pd(&C[1 * ldc + 0], _C1_0);
    C[1 * ldc + 2] = _C1_1;
    _mm_storeu_pd(&C[2 * ldc + 0], _C2_0);
    C[2 * ldc + 2] = _C2_1;
    _mm_storeu_pd(&C[3 * ldc + 0], _C3_0);
    C[3 * ldc + 2] = _C3_1;
    _mm_storeu_pd(&C[4 * ldc + 0], _C4_0);
    C[4 * ldc + 2] = _C4_1;
    _mm_storeu_pd(&C[5 * ldc + 0], _C5_0);
    C[5 * ldc + 2] = _C5_1;

    _mm_storeu_pd(&C[6 * ldc + 0], _C6_0);
    C[6 * ldc + 2] = _C6_1;

    _mm_storeu_pd(&C[7 * ldc + 0], _C7_0);
    C[7 * ldc + 2] = _C7_1;

    _mm_storeu_pd(&C[8 * ldc + 0], _C8_0);
    C[8 * ldc + 2] = _C8_1;

    _mm_storeu_pd(&C[9 * ldc + 0], _C9_0);
    C[9 * ldc + 2] = _C9_1;

    _mm_storeu_pd(&C[10 * ldc + 0], _C10_0);
    C[10 * ldc + 2] = _C10_1;
}
void micro_kernel4x11(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m256d _A0;
    register __m256d _C0_0 = _mm256_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C1_0 = _mm256_loadu_pd(&C[1 * ldc + 0]);
    register __m256d _C2_0 = _mm256_loadu_pd(&C[2 * ldc + 0]);
    register __m256d _C3_0 = _mm256_loadu_pd(&C[3 * ldc + 0]);
    register __m256d _C4_0 = _mm256_loadu_pd(&C[4 * ldc + 0]);
    register __m256d _C5_0 = _mm256_loadu_pd(&C[5 * ldc + 0]);

    register __m256d _C6_0 = _mm256_loadu_pd(&C[6 * ldc + 0]);

    register __m256d _C7_0 = _mm256_loadu_pd(&C[7 * ldc + 0]);

    register __m256d _C8_0 = _mm256_loadu_pd(&C[8 * ldc + 0]);

    register __m256d _C9_0 = _mm256_loadu_pd(&C[9 * ldc + 0]);
    register __m256d _C10_0 = _mm256_loadu_pd(&C[10 * ldc + 0]);

    for(i = 0; i < k; ++i) {
        _A0 = _mm256_loadu_pd(&A[0]);
        _C0_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C1_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C2_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C3_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C4_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C5_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A0, _C5_0);

        _C6_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[6 * ldb]), _A0, _C6_0);

        _C7_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[7 * ldb]), _A0, _C7_0);

        _C8_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[8 * ldb]), _A0, _C8_0);

        _C9_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[9 * ldb]), _A0, _C9_0);
        _C10_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[10 * ldb]), _A0, _C10_0);
        A += lda;
        B += 1;
    }

    _mm256_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm256_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm256_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm256_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm256_storeu_pd(&C[5 * ldc + 0], _C5_0);

    _mm256_storeu_pd(&C[6 * ldc + 0], _C6_0);

    _mm256_storeu_pd(&C[7 * ldc + 0], _C7_0);

    _mm256_storeu_pd(&C[8 * ldc + 0], _C8_0);

    _mm256_storeu_pd(&C[9 * ldc + 0], _C9_0);

    _mm256_storeu_pd(&C[10 * ldc + 0], _C10_0);
}
void micro_kernel5x11(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m256d _A0;
    register double _A1;
    register __m256d _C0_0 = _mm256_loadu_pd(&C[0 * ldc + 0]);
    register double _C0_1 = C[0 * ldc + 4];
    register __m256d _C1_0 = _mm256_loadu_pd(&C[1 * ldc + 0]);
    register double _C1_1 = C[1 * ldc + 4];
    register __m256d _C2_0 = _mm256_loadu_pd(&C[2 * ldc + 0]);
    register double _C2_1 = C[2 * ldc + 4];
    register __m256d _C3_0 = _mm256_loadu_pd(&C[3 * ldc + 0]);
    register double _C3_1 = C[3 * ldc + 4];
    register __m256d _C4_0 = _mm256_loadu_pd(&C[4 * ldc + 0]);
    register double _C4_1 = C[4 * ldc + 4];
    register __m256d _C5_0 = _mm256_loadu_pd(&C[5 * ldc + 0]);
    register double _C5_1 = C[5 * ldc + 4];

    register __m256d _C6_0 = _mm256_loadu_pd(&C[6 * ldc + 0]);
    register double _C6_1 = C[6 * ldc + 4];

    register __m256d _C7_0 = _mm256_loadu_pd(&C[7 * ldc + 0]);
    register double _C7_1 = C[7 * ldc + 4];

    register __m256d _C8_0 = _mm256_loadu_pd(&C[8 * ldc + 0]);
    register double _C8_1 = C[8 * ldc + 4];

    register __m256d _C9_0 = _mm256_loadu_pd(&C[9 * ldc + 0]);
    register double _C9_1 = C[9 * ldc + 4];

    register __m256d _C10_0 = _mm256_loadu_pd(&C[10 * ldc + 0]);
    register double _C10_1 = C[10 * ldc + 4];

    for(i = 0; i < k; ++i) {
        _A0 = _mm256_loadu_pd(&A[0]);
        _A1 = A[4];
        _C0_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 += B[0 * ldb] * _A1;
        _C1_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 += B[1 * ldb] * _A1;
        _C2_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 += B[2 * ldb] * _A1;
        _C3_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 += B[3 * ldb] * _A1;
        _C4_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 += B[4 * ldb] * _A1;
        _C5_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 += B[5 * ldb] * _A1;

        _C6_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 += B[6 * ldb] * _A1;

        _C7_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 += B[7 * ldb] * _A1;

        _C8_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 += B[8 * ldb] * _A1;

        _C9_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[9 * ldb]), _A0, _C9_0);
        _C9_1 += B[9 * ldb] * _A1;

        _C10_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[10 * ldb]), _A0, _C10_0);
        _C10_1 += B[10 * ldb] * _A1;

        A += lda;
        B += 1;
    }

    _mm256_storeu_pd(&C[0 * ldc + 0], _C0_0);
    C[0 * ldc + 4] = _C0_1;
    _mm256_storeu_pd(&C[1 * ldc + 0], _C1_0);
    C[1 * ldc + 4] = _C1_1;
    _mm256_storeu_pd(&C[2 * ldc + 0], _C2_0);
    C[2 * ldc + 4] = _C2_1;
    _mm256_storeu_pd(&C[3 * ldc + 0], _C3_0);
    C[3 * ldc + 4] = _C3_1;
    _mm256_storeu_pd(&C[4 * ldc + 0], _C4_0);
    C[4 * ldc + 4] = _C4_1;
    _mm256_storeu_pd(&C[5 * ldc + 0], _C5_0);
    C[5 * ldc + 4] = _C5_1;

    _mm256_storeu_pd(&C[6 * ldc + 0], _C6_0);
    C[6 * ldc + 4] = _C6_1;

    _mm256_storeu_pd(&C[7 * ldc + 0], _C7_0);
    C[7 * ldc + 4] = _C7_1;

    _mm256_storeu_pd(&C[8 * ldc + 0], _C8_0);
    C[8 * ldc + 4] = _C8_1;

    _mm256_storeu_pd(&C[9 * ldc + 0], _C9_0);
    C[9 * ldc + 4] = _C9_1;

    _mm256_storeu_pd(&C[10 * ldc + 0], _C10_0);
    C[10 * ldc + 4] = _C10_1;
}
void micro_kernel6x11(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m256d _A0;
    register __m128d _A1;
    register __m256d _C0_0 = _mm256_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C0_1 = _mm_loadu_pd(&C[0 * ldc + 4]);
    register __m256d _C1_0 = _mm256_loadu_pd(&C[1 * ldc + 0]);
    register __m128d _C1_1 = _mm_loadu_pd(&C[1 * ldc + 4]);
    register __m256d _C2_0 = _mm256_loadu_pd(&C[2 * ldc + 0]);
    register __m128d _C2_1 = _mm_loadu_pd(&C[2 * ldc + 4]);
    register __m256d _C3_0 = _mm256_loadu_pd(&C[3 * ldc + 0]);
    register __m128d _C3_1 = _mm_loadu_pd(&C[3 * ldc + 4]);
    register __m256d _C4_0 = _mm256_loadu_pd(&C[4 * ldc + 0]);
    register __m128d _C4_1 = _mm_loadu_pd(&C[4 * ldc + 4]);
    register __m256d _C5_0 = _mm256_loadu_pd(&C[5 * ldc + 0]);
    register __m128d _C5_1 = _mm_loadu_pd(&C[5 * ldc + 4]);

    register __m256d _C6_0 = _mm256_loadu_pd(&C[6 * ldc + 0]);
    register __m128d _C6_1 = _mm_loadu_pd(&C[6 * ldc + 4]);

    register __m256d _C7_0 = _mm256_loadu_pd(&C[7 * ldc + 0]);
    register __m128d _C7_1 = _mm_loadu_pd(&C[7 * ldc + 4]);

    register __m256d _C8_0 = _mm256_loadu_pd(&C[8 * ldc + 0]);
    register __m128d _C8_1 = _mm_loadu_pd(&C[8 * ldc + 4]);

    register __m256d _C9_0 = _mm256_loadu_pd(&C[9 * ldc + 0]);
    register __m128d _C9_1 = _mm_loadu_pd(&C[9 * ldc + 4]);

    register __m256d _C10_0 = _mm256_loadu_pd(&C[10 * ldc + 0]);
    register __m128d _C10_1 = _mm_loadu_pd(&C[10 * ldc + 4]);

    for(i = 0; i < k; ++i) {
        _A0 = _mm256_loadu_pd(&A[0]);
        _A1 = _mm_loadu_pd(&A[4]);
        _C0_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C1_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C2_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C3_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A1, _C3_1);
        _C4_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A1, _C4_1);
        _C5_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A1, _C5_1);

        _C6_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm_fnmadd_pd(_mm_set1_pd(B[6 * ldb]), _A1, _C6_1);

        _C7_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 = _mm_fnmadd_pd(_mm_set1_pd(B[7 * ldb]), _A1, _C7_1);

        _C8_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 = _mm_fnmadd_pd(_mm_set1_pd(B[8 * ldb]), _A1, _C8_1);

        _C9_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[9 * ldb]), _A0, _C9_0);
        _C9_1 = _mm_fnmadd_pd(_mm_set1_pd(B[9 * ldb]), _A1, _C9_1);

        _C10_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[10 * ldb]), _A0, _C10_0);
        _C10_1 = _mm_fnmadd_pd(_mm_set1_pd(B[10 * ldb]), _A1, _C10_1);

        A += lda;
        B += 1;
    }

    _mm256_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[0 * ldc + 4], _C0_1);
    _mm256_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm_storeu_pd(&C[1 * ldc + 4], _C1_1);
    _mm256_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm_storeu_pd(&C[2 * ldc + 4], _C2_1);
    _mm256_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm_storeu_pd(&C[3 * ldc + 4], _C3_1);
    _mm256_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm_storeu_pd(&C[4 * ldc + 4], _C4_1);
    _mm256_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm_storeu_pd(&C[5 * ldc + 4], _C5_1);

    _mm256_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm_storeu_pd(&C[6 * ldc + 4], _C6_1);

    _mm256_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm_storeu_pd(&C[7 * ldc + 4], _C7_1);

    _mm256_storeu_pd(&C[8 * ldc + 0], _C8_0);
    _mm_storeu_pd(&C[8 * ldc + 4], _C8_1);

    _mm256_storeu_pd(&C[9 * ldc + 0], _C9_0);
    _mm_storeu_pd(&C[9 * ldc + 4], _C9_1);

    _mm256_storeu_pd(&C[10 * ldc + 0], _C10_0);
    _mm_storeu_pd(&C[10 * ldc + 4], _C10_1);
}
void micro_kernel7x11(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m256d _A0;
    register __m128d _A1;
    register double _A2;
    register __m256d _C0_0 = _mm256_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C0_1 = _mm_loadu_pd(&C[0 * ldc + 4]);
    register double _C0_2 = C[0 * ldc + 6];
    register __m256d _C1_0 = _mm256_loadu_pd(&C[1 * ldc + 0]);
    register __m128d _C1_1 = _mm_loadu_pd(&C[1 * ldc + 4]);
    register double _C1_2 = C[1 * ldc + 6];
    register __m256d _C2_0 = _mm256_loadu_pd(&C[2 * ldc + 0]);
    register __m128d _C2_1 = _mm_loadu_pd(&C[2 * ldc + 4]);
    register double _C2_2 = C[2 * ldc + 6];
    register __m256d _C3_0 = _mm256_loadu_pd(&C[3 * ldc + 0]);
    register __m128d _C3_1 = _mm_loadu_pd(&C[3 * ldc + 4]);
    register double _C3_2 = C[3 * ldc + 6];
    register __m256d _C4_0 = _mm256_loadu_pd(&C[4 * ldc + 0]);
    register __m128d _C4_1 = _mm_loadu_pd(&C[4 * ldc + 4]);
    register double _C4_2 = C[4 * ldc + 6];
    register __m256d _C5_0 = _mm256_loadu_pd(&C[5 * ldc + 0]);
    register __m128d _C5_1 = _mm_loadu_pd(&C[5 * ldc + 4]);
    register double _C5_2 = C[5 * ldc + 6];

    register __m256d _C6_0 = _mm256_loadu_pd(&C[6 * ldc + 0]);
    register __m128d _C6_1 = _mm_loadu_pd(&C[6 * ldc + 4]);
    register double _C6_2 = C[6 * ldc + 6];

    register __m256d _C7_0 = _mm256_loadu_pd(&C[7 * ldc + 0]);
    register __m128d _C7_1 = _mm_loadu_pd(&C[7 * ldc + 4]);
    register double _C7_2 = C[7 * ldc + 6];

    register __m256d _C8_0 = _mm256_loadu_pd(&C[8 * ldc + 0]);
    register __m128d _C8_1 = _mm_loadu_pd(&C[8 * ldc + 4]);
    register double _C8_2 = C[8 * ldc + 6];

    register __m256d _C9_0 = _mm256_loadu_pd(&C[9 * ldc + 0]);
    register __m128d _C9_1 = _mm_loadu_pd(&C[9 * ldc + 4]);
    register double _C9_2 = C[9 * ldc + 6];

    register __m256d _C10_0 = _mm256_loadu_pd(&C[10 * ldc + 0]);
    register __m128d _C10_1 = _mm_loadu_pd(&C[10 * ldc + 4]);
    register double _C10_2 = C[10 * ldc + 6];

    for(i = 0; i < k; ++i) {
        _A0 = _mm256_loadu_pd(&A[0]);
        _A1 = _mm_loadu_pd(&A[4]);
        _A2 = A[6];
        _C0_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 += B[0 * ldb] * _A2;
        _C1_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C1_2 += B[1 * ldb] * _A2;
        _C2_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C2_2 += B[2 * ldb] * _A2;
        _C3_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A1, _C3_1);
        _C3_2 += B[3 * ldb] * _A2;
        _C4_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A1, _C4_1);
        _C4_2 += B[4 * ldb] * _A2;
        _C5_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A1, _C5_1);
        _C5_2 += B[5 * ldb] * _A2;

        _C6_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm_fnmadd_pd(_mm_set1_pd(B[6 * ldb]), _A1, _C6_1);
        _C6_2 += B[6 * ldb] * _A2;

        _C7_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 = _mm_fnmadd_pd(_mm_set1_pd(B[7 * ldb]), _A1, _C7_1);
        _C7_2 += B[7 * ldb] * _A2;

        _C8_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 = _mm_fnmadd_pd(_mm_set1_pd(B[8 * ldb]), _A1, _C8_1);
        _C8_2 += B[8 * ldb] * _A2;

        _C9_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[9 * ldb]), _A0, _C9_0);
        _C9_1 = _mm_fnmadd_pd(_mm_set1_pd(B[9 * ldb]), _A1, _C9_1);
        _C9_2 += B[9 * ldb] * _A2;

        _C10_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[10 * ldb]), _A0, _C10_0);
        _C10_1 = _mm_fnmadd_pd(_mm_set1_pd(B[10 * ldb]), _A1, _C10_1);
        _C10_2 += B[10 * ldb] * _A2;

        A += lda;
        B += 1;
    }

    _mm256_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[0 * ldc + 4], _C0_1);
    C[0 * ldc + 6] = _C0_2;
    _mm256_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm_storeu_pd(&C[1 * ldc + 4], _C1_1);
    C[1 * ldc + 6] = _C1_2;
    _mm256_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm_storeu_pd(&C[2 * ldc + 4], _C2_1);
    C[2 * ldc + 6] = _C2_2;
    _mm256_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm_storeu_pd(&C[3 * ldc + 4], _C3_1);
    C[3 * ldc + 6] = _C3_2;
    _mm256_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm_storeu_pd(&C[4 * ldc + 4], _C4_1);
    C[4 * ldc + 6] = _C4_2;
    _mm256_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm_storeu_pd(&C[5 * ldc + 4], _C5_1);
    C[5 * ldc + 6] = _C5_2;

    _mm256_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm_storeu_pd(&C[6 * ldc + 4], _C6_1);
    C[6 * ldc + 6] = _C6_2;

    _mm256_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm_storeu_pd(&C[7 * ldc + 4], _C7_1);
    C[7 * ldc + 6] = _C7_2;

    _mm256_storeu_pd(&C[8 * ldc + 0], _C8_0);
    _mm_storeu_pd(&C[8 * ldc + 4], _C8_1);
    C[8 * ldc + 6] = _C8_2;

    _mm256_storeu_pd(&C[9 * ldc + 0], _C9_0);
    _mm_storeu_pd(&C[9 * ldc + 4], _C9_1);
    C[9 * ldc + 6] = _C9_2;

    _mm256_storeu_pd(&C[10 * ldc + 0], _C10_0);
    _mm_storeu_pd(&C[10 * ldc + 4], _C10_1);
    C[10 * ldc + 6] = _C10_2;
}
//=====================================================================================================================
void micro_kernel8x11(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);
    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);

    register __m512d _C7_0 = _mm512_loadu_pd(&C[7 * ldc + 0]);

    register __m512d _C8_0 = _mm512_loadu_pd(&C[8 * ldc + 0]);

    register __m512d _C9_0 = _mm512_loadu_pd(&C[9 * ldc + 0]);

    register __m512d _C10_0 = _mm512_loadu_pd(&C[10 * ldc + 0]);

    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);

        _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A0, _C7_0);

        _C8_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8 * ldb]), _A0, _C8_0);

        _C9_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[9 * ldb]), _A0, _C9_0);

        _C10_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[10 * ldb]), _A0, _C10_0);

        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);

    _mm512_storeu_pd(&C[7 * ldc + 0], _C7_0);

    _mm512_storeu_pd(&C[8 * ldc + 0], _C8_0);

    _mm512_storeu_pd(&C[9 * ldc + 0], _C9_0);

    _mm512_storeu_pd(&C[10 * ldc + 0], _C10_0);
}
void micro_kernel9x11(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register double _A1;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register double _C0_1 = C[0 * ldc + 8];

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);
    register double _C1_1 = C[1 * ldc + 8];

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register double _C2_1 = C[2 * ldc + 8];

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register double _C3_1 = C[3 * ldc + 8];

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register double _C4_1 = C[4 * ldc + 8];

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register double _C5_1 = C[5 * ldc + 8];

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);
    register double _C6_1 = C[6 * ldc + 8];

    register __m512d _C7_0 = _mm512_loadu_pd(&C[7 * ldc + 0]);
    register double _C7_1 = C[7 * ldc + 8];

    register __m512d _C8_0 = _mm512_loadu_pd(&C[8 * ldc + 0]);
    register double _C8_1 = C[8 * ldc + 8];

    register __m512d _C9_0 = _mm512_loadu_pd(&C[9 * ldc + 0]);
    register double _C9_1 = C[9 * ldc + 8];

    register __m512d _C10_0 = _mm512_loadu_pd(&C[10 * ldc + 0]);
    register double _C10_1 = C[10 * ldc + 8];

    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = A[8];

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 += B[0 * ldb] * _A1;

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 += B[1 * ldb] * _A1;

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 += B[2 * ldb] * _A1;

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 += B[3 * ldb] * _A1;

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 += B[4 * ldb] * _A1;

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 += B[5 * ldb] * _A1;

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 += B[6 * ldb] * _A1;
        
        _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 += B[7 * ldb] * _A1;

        _C8_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 += B[8 * ldb] * _A1;

        _C9_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[9 * ldb]), _A0, _C9_0);
        _C9_1 += B[9 * ldb] * _A1;

        _C10_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[10 * ldb]), _A0, _C10_0);
        _C10_1 += B[10 * ldb] * _A1;

        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    C[0 * ldc + 8] = _C0_1;

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    C[1 * ldc + 8] = _C1_1;

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    C[2 * ldc + 8] = _C2_1;

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    C[3 * ldc + 8] = _C3_1;

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    C[4 * ldc + 8] = _C4_1;

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    C[5 * ldc + 8] = _C5_1;

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);
    C[6 * ldc + 8] = _C6_1;

    _mm512_storeu_pd(&C[7 * ldc + 0], _C7_0);
    C[7 * ldc + 8] = _C7_1;

    _mm512_storeu_pd(&C[8 * ldc + 0], _C8_0);
    C[8 * ldc + 8] = _C8_1;

    _mm512_storeu_pd(&C[9 * ldc + 0], _C9_0);
    C[9 * ldc + 8] = _C9_1;

    _mm512_storeu_pd(&C[10 * ldc + 0], _C10_0);
    C[10 * ldc + 8] = _C10_1;
}
void micro_kernel10x11(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m128d _A1;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C0_1 = _mm_loadu_pd(&C[0 * ldc + 8]);

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m128d _C1_1 = _mm_loadu_pd(&C[1 * ldc + 8]);

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m128d _C2_1 = _mm_loadu_pd(&C[2 * ldc + 8]);

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m128d _C3_1 = _mm_loadu_pd(&C[3 * ldc + 8]);

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m128d _C4_1 = _mm_loadu_pd(&C[4 * ldc + 8]);

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register __m128d _C5_1 = _mm_loadu_pd(&C[5 * ldc + 8]);

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);
    register __m128d _C6_1 = _mm_loadu_pd(&C[6 * ldc + 8]);

    register __m512d _C7_0 = _mm512_loadu_pd(&C[7 * ldc + 0]);
    register __m128d _C7_1 = _mm_loadu_pd(&C[7 * ldc + 8]);

    register __m512d _C8_0 = _mm512_loadu_pd(&C[8 * ldc + 0]);
    register __m128d _C8_1 = _mm_loadu_pd(&C[8 * ldc + 8]);

    register __m512d _C9_0 = _mm512_loadu_pd(&C[9 * ldc + 0]);
    register __m128d _C9_1 = _mm_loadu_pd(&C[9 * ldc + 8]);

    register __m512d _C10_0 = _mm512_loadu_pd(&C[10 * ldc + 0]);
    register __m128d _C10_1 = _mm_loadu_pd(&C[10 * ldc + 8]);
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm_loadu_pd(&A[8]);

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A1, _C0_1);

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A1, _C1_1);

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A1, _C2_1);

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A1, _C3_1);

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A1, _C4_1);

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A1, _C5_1);

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm_fnmadd_pd(_mm_set1_pd(B[6 * ldb]), _A1, _C6_1);

        _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 = _mm_fnmadd_pd(_mm_set1_pd(B[7 * ldb]), _A1, _C7_1);

        _C8_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 = _mm_fnmadd_pd(_mm_set1_pd(B[8 * ldb]), _A1, _C8_1);

        _C9_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[9 * ldb]), _A0, _C9_0);
        _C9_1 = _mm_fnmadd_pd(_mm_set1_pd(B[9 * ldb]), _A1, _C9_1);

        _C10_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[10 * ldb]), _A0, _C10_0);
        _C10_1 = _mm_fnmadd_pd(_mm_set1_pd(B[10 * ldb]), _A1, _C10_1);
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[0 * ldc + 8], _C0_1);

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm_storeu_pd(&C[1 * ldc + 8], _C1_1);

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm_storeu_pd(&C[2 * ldc + 8], _C2_1);

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm_storeu_pd(&C[3 * ldc + 8], _C3_1);

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm_storeu_pd(&C[4 * ldc + 8], _C4_1);

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm_storeu_pd(&C[5 * ldc + 8], _C5_1);

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm_storeu_pd(&C[6 * ldc + 8], _C6_1);

    _mm512_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm_storeu_pd(&C[7 * ldc + 8], _C7_1);

    _mm512_storeu_pd(&C[8 * ldc + 0], _C8_0);
    _mm_storeu_pd(&C[8 * ldc + 8], _C8_1);

    _mm512_storeu_pd(&C[9 * ldc + 0], _C9_0);
    _mm_storeu_pd(&C[9 * ldc + 8], _C9_1);

    _mm512_storeu_pd(&C[10 * ldc + 0], _C10_0);
    _mm_storeu_pd(&C[10 * ldc + 8], _C10_1);
}
void micro_kernel11x11(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m128d _A1;
    register double _A2;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C0_1 = _mm_loadu_pd(&C[0 * ldc + 8]);
    register double _C0_2 = C[0 * ldc + 10];

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m128d _C1_1 = _mm_loadu_pd(&C[1 * ldc + 8]);
    register double _C1_2 = C[1 * ldc + 10];

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m128d _C2_1 = _mm_loadu_pd(&C[2 * ldc + 8]);
    register double _C2_2 = C[2 * ldc + 10];

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m128d _C3_1 = _mm_loadu_pd(&C[3 * ldc + 8]);
    register double _C3_2 = C[3 * ldc + 10];

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m128d _C4_1 = _mm_loadu_pd(&C[4 * ldc + 8]);
    register double _C4_2 = C[4 * ldc + 10];

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register __m128d _C5_1 = _mm_loadu_pd(&C[5 * ldc + 8]);
    register double _C5_2 = C[5 * ldc + 10];

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);
    register __m128d _C6_1 = _mm_loadu_pd(&C[6 * ldc + 8]);
    register double _C6_2 = C[6 * ldc + 10];

    register __m512d _C7_0 = _mm512_loadu_pd(&C[7 * ldc + 0]);
    register __m128d _C7_1 = _mm_loadu_pd(&C[7 * ldc + 8]);
    register double _C7_2 = C[7 * ldc + 10];

    register __m512d _C8_0 = _mm512_loadu_pd(&C[8 * ldc + 0]);
    register __m128d _C8_1 = _mm_loadu_pd(&C[8 * ldc + 8]);
    register double _C8_2 = C[8 * ldc + 10];

    register __m512d _C9_0 = _mm512_loadu_pd(&C[9 * ldc + 0]);
    register __m128d _C9_1 = _mm_loadu_pd(&C[9 * ldc + 8]);
    register double _C9_2 = C[9 * ldc + 10];

    register __m512d _C10_0 = _mm512_loadu_pd(&C[10 * ldc + 0]);
    register __m128d _C10_1 = _mm_loadu_pd(&C[10 * ldc + 8]);
    register double _C10_2 = C[10 * ldc + 10];
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm_loadu_pd(&A[8]);
        _A2 = A[10];

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 += B[0 * ldb] * _A2;

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C1_2 += B[1 * ldb] * _A2;

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C2_2 += B[2 * ldb] * _A2;

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A1, _C3_1);
        _C3_2 += B[3 * ldb] * _A2;

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A1, _C4_1);
        _C4_2 += B[4 * ldb] * _A2;

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A1, _C5_1);
        _C5_2 += B[5 * ldb] * _A2;

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm_fnmadd_pd(_mm_set1_pd(B[6 * ldb]), _A1, _C6_1);
        _C6_2 += B[6 * ldb] * _A2;

        _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 = _mm_fnmadd_pd(_mm_set1_pd(B[7 * ldb]), _A1, _C7_1);
        _C7_2 += B[7 * ldb] * _A2;

        _C8_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 = _mm_fnmadd_pd(_mm_set1_pd(B[8 * ldb]), _A1, _C8_1);
        _C8_2 += B[8 * ldb] * _A2;

        _C9_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[9 * ldb]), _A0, _C9_0);
        _C9_1 = _mm_fnmadd_pd(_mm_set1_pd(B[9 * ldb]), _A1, _C9_1);
        _C9_2 += B[9 * ldb] * _A2;

        _C10_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[10 * ldb]), _A0, _C10_0);
        _C10_1 = _mm_fnmadd_pd(_mm_set1_pd(B[10 * ldb]), _A1, _C10_1);
        _C10_2 += B[10 * ldb] * _A2;
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[0 * ldc + 8], _C0_1);
    C[0 * ldc + 10] = _C0_2;

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm_storeu_pd(&C[1 * ldc + 8], _C1_1);
    C[1 * ldc + 10] = _C1_2;

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm_storeu_pd(&C[2 * ldc + 8], _C2_1);
    C[2 * ldc + 10] = _C2_2;

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm_storeu_pd(&C[3 * ldc + 8], _C3_1);
    C[3 * ldc + 10] = _C3_2;

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm_storeu_pd(&C[4 * ldc + 8], _C4_1);
    C[4 * ldc + 10] = _C4_2;

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm_storeu_pd(&C[5 * ldc + 8], _C5_1);
    C[5 * ldc + 10] = _C5_2;

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm_storeu_pd(&C[6 * ldc + 8], _C6_1);
    C[6 * ldc + 10] = _C6_2;

    _mm512_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm_storeu_pd(&C[7 * ldc + 8], _C7_1);
    C[7 * ldc + 10] = _C7_2;

    _mm512_storeu_pd(&C[8 * ldc + 0], _C8_0);
    _mm_storeu_pd(&C[8 * ldc + 8], _C8_1);
    C[8 * ldc + 10] = _C8_2;

    _mm512_storeu_pd(&C[9 * ldc + 0], _C9_0);
    _mm_storeu_pd(&C[9 * ldc + 8], _C9_1);
    C[9 * ldc + 10] = _C9_2;

    _mm512_storeu_pd(&C[10 * ldc + 0], _C10_0);
    _mm_storeu_pd(&C[10 * ldc + 8], _C10_1);
    C[10 * ldc + 10] = _C10_2;
}
void micro_kernel12x11(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {//------------------------------------
     int i;
    register __m512d _A0;
    register __m256d _A1;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C0_1 = _mm256_loadu_pd(&C[0 * ldc + 8]);

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m256d _C1_1 = _mm256_loadu_pd(&C[1 * ldc + 8]);

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m256d _C2_1 = _mm256_loadu_pd(&C[2 * ldc + 8]);

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m256d _C3_1 = _mm256_loadu_pd(&C[3 * ldc + 8]);

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m256d _C4_1 = _mm256_loadu_pd(&C[4 * ldc + 8]);

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register __m256d _C5_1 = _mm256_loadu_pd(&C[5 * ldc + 8]);

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);
    register __m256d _C6_1 = _mm256_loadu_pd(&C[6 * ldc + 8]);

    register __m512d _C7_0 = _mm512_loadu_pd(&C[7 * ldc + 0]);
    register __m256d _C7_1 = _mm256_loadu_pd(&C[7 * ldc + 8]);

    register __m512d _C8_0 = _mm512_loadu_pd(&C[8 * ldc + 0]);
    register __m256d _C8_1 = _mm256_loadu_pd(&C[8 * ldc + 8]);

    register __m512d _C9_0 = _mm512_loadu_pd(&C[9 * ldc + 0]);
    register __m256d _C9_1 = _mm256_loadu_pd(&C[9 * ldc + 8]);

    register __m512d _C10_0 = _mm512_loadu_pd(&C[10 * ldc + 0]);
    register __m256d _C10_1 = _mm256_loadu_pd(&C[10 * ldc + 8]);
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm256_loadu_pd(&A[8]);

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A1, _C0_1);

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A1, _C1_1);

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A1, _C2_1);

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A1, _C3_1);

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A1, _C4_1);

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A1, _C5_1);

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[6 * ldb]), _A1, _C6_1);

        _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[7 * ldb]), _A1, _C7_1);

        _C8_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[8 * ldb]), _A1, _C8_1);

        _C9_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[9 * ldb]), _A0, _C9_0);
        _C9_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[9 * ldb]), _A1, _C9_1);

        _C10_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[10 * ldb]), _A0, _C10_0);
        _C10_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[10 * ldb]), _A1, _C10_1);
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[0 * ldc + 8], _C0_1);

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm256_storeu_pd(&C[1 * ldc + 8], _C1_1);

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm256_storeu_pd(&C[2 * ldc + 8], _C2_1);

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm256_storeu_pd(&C[3 * ldc + 8], _C3_1);

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm256_storeu_pd(&C[4 * ldc + 8], _C4_1);

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm256_storeu_pd(&C[5 * ldc + 8], _C5_1);

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm256_storeu_pd(&C[6 * ldc + 8], _C6_1);

    _mm512_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm256_storeu_pd(&C[7 * ldc + 8], _C7_1);

    _mm512_storeu_pd(&C[8 * ldc + 0], _C8_0);
    _mm256_storeu_pd(&C[8 * ldc + 8], _C8_1);

    _mm512_storeu_pd(&C[9 * ldc + 0], _C9_0);
    _mm256_storeu_pd(&C[9 * ldc + 8], _C9_1);

    _mm512_storeu_pd(&C[10 * ldc + 0], _C10_0);
    _mm256_storeu_pd(&C[10 * ldc + 8], _C10_1);
}
void micro_kernel13x11(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m256d _A1;
    register double _A2;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C0_1 = _mm256_loadu_pd(&C[0 * ldc + 8]);
    register double _C0_2 = C[0 * ldc + 12];

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m256d _C1_1 = _mm256_loadu_pd(&C[1 * ldc + 8]);
    register double _C1_2 = C[1 * ldc + 12];

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m256d _C2_1 = _mm256_loadu_pd(&C[2 * ldc + 8]);
    register double _C2_2 = C[2 * ldc + 12];

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m256d _C3_1 = _mm256_loadu_pd(&C[3 * ldc + 8]);
    register double _C3_2 = C[3 * ldc + 12];

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m256d _C4_1 = _mm256_loadu_pd(&C[4 * ldc + 8]);
    register double _C4_2 = C[4 * ldc + 12];

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register __m256d _C5_1 = _mm256_loadu_pd(&C[5 * ldc + 8]);
    register double _C5_2 = C[5 * ldc + 12];

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);
    register __m256d _C6_1 = _mm256_loadu_pd(&C[6 * ldc + 8]);
    register double _C6_2 = C[6 * ldc + 12];

    register __m512d _C7_0 = _mm512_loadu_pd(&C[7 * ldc + 0]);
    register __m256d _C7_1 = _mm256_loadu_pd(&C[7 * ldc + 8]);
    register double _C7_2 = C[7 * ldc + 12];

    register __m512d _C8_0 = _mm512_loadu_pd(&C[8 * ldc + 0]);
    register __m256d _C8_1 = _mm256_loadu_pd(&C[8 * ldc + 8]);
    register double _C8_2 = C[8 * ldc + 12];

    register __m512d _C9_0 = _mm512_loadu_pd(&C[9 * ldc + 0]);
    register __m256d _C9_1 = _mm256_loadu_pd(&C[9 * ldc + 8]);
    register double _C9_2 = C[9 * ldc + 12];

    register __m512d _C10_0 = _mm512_loadu_pd(&C[10 * ldc + 0]);
    register __m256d _C10_1 = _mm256_loadu_pd(&C[10 * ldc + 8]);
    register double _C10_2 = C[10 * ldc + 12];
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm256_loadu_pd(&A[8]);
        _A2 = A[12];

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 += B[0 * ldb] * _A2;

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C1_2 += B[1 * ldb] * _A2;

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C2_2 += B[2 * ldb] * _A2;

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A1, _C3_1);
        _C3_2 += B[3 * ldb] * _A2;

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A1, _C4_1);
        _C4_2 += B[4 * ldb] * _A2;

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A1, _C5_1);
        _C5_2 += B[5 * ldb] * _A2;

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[6 * ldb]), _A1, _C6_1);
        _C6_2 += B[6 * ldb] * _A2;

        _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[7 * ldb]), _A1, _C7_1);
        _C7_2 += B[7 * ldb] * _A2;

        _C8_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[8 * ldb]), _A1, _C8_1);
        _C8_2 += B[8 * ldb] * _A2;

        _C9_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[9 * ldb]), _A0, _C9_0);
        _C9_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[9 * ldb]), _A1, _C9_1);
        _C9_2 += B[9 * ldb] * _A2;

        _C10_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[10 * ldb]), _A0, _C10_0);
        _C10_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[10 * ldb]), _A1, _C10_1);
        _C10_2 += B[10 * ldb] * _A2;
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[0 * ldc + 8], _C0_1);
    C[0 * ldc + 12] = _C0_2;

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm256_storeu_pd(&C[1 * ldc + 8], _C1_1);
    C[1 * ldc + 12] = _C1_2;

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm256_storeu_pd(&C[2 * ldc + 8], _C2_1);
    C[2 * ldc + 12] = _C2_2;

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm256_storeu_pd(&C[3 * ldc + 8], _C3_1);
    C[3 * ldc + 12] = _C3_2;

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm256_storeu_pd(&C[4 * ldc + 8], _C4_1);
    C[4 * ldc + 12] = _C4_2;

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm256_storeu_pd(&C[5 * ldc + 8], _C5_1);
    C[5 * ldc + 12] = _C5_2;

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm256_storeu_pd(&C[6 * ldc + 8], _C6_1);
    C[6 * ldc + 12] = _C6_2;

    _mm512_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm256_storeu_pd(&C[7 * ldc + 8], _C7_1);
    C[7 * ldc + 12] = _C7_2;

    _mm512_storeu_pd(&C[8 * ldc + 0], _C8_0);
    _mm256_storeu_pd(&C[8 * ldc + 8], _C8_1);
    C[8 * ldc + 12] = _C8_2;

    _mm512_storeu_pd(&C[9 * ldc + 0], _C9_0);
    _mm256_storeu_pd(&C[9 * ldc + 8], _C9_1);
    C[9 * ldc + 12] = _C9_2;

    _mm512_storeu_pd(&C[10 * ldc + 0], _C10_0);
    _mm256_storeu_pd(&C[10 * ldc + 8], _C10_1);
    C[10 * ldc + 12] = _C10_2;
}
void micro_kernel14x11(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m256d _A1;
    register __m128d _A2;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C0_1 = _mm256_loadu_pd(&C[0 * ldc + 8]);
    register __m128d _C0_2 = _mm_loadu_pd(&C[0 * ldc + 12]);

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m256d _C1_1 = _mm256_loadu_pd(&C[1 * ldc + 8]);
    register __m128d _C1_2 = _mm_loadu_pd(&C[1 * ldc + 12]);

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m256d _C2_1 = _mm256_loadu_pd(&C[2 * ldc + 8]);
    register __m128d _C2_2 = _mm_loadu_pd(&C[2 * ldc + 12]);

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m256d _C3_1 = _mm256_loadu_pd(&C[3 * ldc + 8]);
    register __m128d _C3_2 = _mm_loadu_pd(&C[3 * ldc + 12]);

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m256d _C4_1 = _mm256_loadu_pd(&C[4 * ldc + 8]);
    register __m128d _C4_2 = _mm_loadu_pd(&C[4 * ldc + 12]);

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register __m256d _C5_1 = _mm256_loadu_pd(&C[5 * ldc + 8]);
    register __m128d _C5_2 = _mm_loadu_pd(&C[5 * ldc + 12]);

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);
    register __m256d _C6_1 = _mm256_loadu_pd(&C[6 * ldc + 8]);
    register __m128d _C6_2 = _mm_loadu_pd(&C[6 * ldc + 12]);

    register __m512d _C7_0 = _mm512_loadu_pd(&C[7 * ldc + 0]);
    register __m256d _C7_1 = _mm256_loadu_pd(&C[7 * ldc + 8]);
    register __m128d _C7_2 = _mm_loadu_pd(&C[7 * ldc + 12]);

    register __m512d _C8_0 = _mm512_loadu_pd(&C[8 * ldc + 0]);
    register __m256d _C8_1 = _mm256_loadu_pd(&C[8 * ldc + 8]);
    register __m128d _C8_2 = _mm_loadu_pd(&C[8 * ldc + 12]);

    register __m512d _C9_0 = _mm512_loadu_pd(&C[9 * ldc + 0]);
    register __m256d _C9_1 = _mm256_loadu_pd(&C[9 * ldc + 8]);
    register __m128d _C9_2 = _mm_loadu_pd(&C[9 * ldc + 12]);

    register __m512d _C10_0 = _mm512_loadu_pd(&C[10 * ldc + 0]);
    register __m256d _C10_1 = _mm256_loadu_pd(&C[10 * ldc + 8]);
    register __m128d _C10_2 = _mm_loadu_pd(&C[10 * ldc + 12]);
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm256_loadu_pd(&A[8]);
        _A2 = _mm_loadu_pd(&A[12]);

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A2, _C0_2);

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C1_2 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A2, _C1_2);

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C2_2 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A2, _C2_2);

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A1, _C3_1);
        _C3_2 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A2, _C3_2);

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A1, _C4_1);
        _C4_2 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A2, _C4_2);

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A1, _C5_1);
        _C5_2 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A2, _C5_2);

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[6 * ldb]), _A1, _C6_1);
        _C6_2 = _mm_fnmadd_pd(_mm_set1_pd(B[6 * ldb]), _A2, _C6_2);

        _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[7 * ldb]), _A1, _C7_1);
        _C7_2 = _mm_fnmadd_pd(_mm_set1_pd(B[7 * ldb]), _A2, _C7_2);

        _C8_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[8 * ldb]), _A1, _C8_1);
        _C8_2 = _mm_fnmadd_pd(_mm_set1_pd(B[8 * ldb]), _A2, _C8_2);

        _C9_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[9 * ldb]), _A0, _C9_0);
        _C9_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[9 * ldb]), _A1, _C9_1);
        _C9_2 = _mm_fnmadd_pd(_mm_set1_pd(B[9 * ldb]), _A2, _C9_2);

        _C10_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[10 * ldb]), _A0, _C10_0);
        _C10_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[10 * ldb]), _A1, _C10_1);
        _C10_2 = _mm_fnmadd_pd(_mm_set1_pd(B[10 * ldb]), _A2, _C10_2);
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[0 * ldc + 8], _C0_1);
    _mm_storeu_pd(&C[0 * ldc + 12], _C0_2);

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm256_storeu_pd(&C[1 * ldc + 8], _C1_1);
    _mm_storeu_pd(&C[1 * ldc + 12], _C1_2);

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm256_storeu_pd(&C[2 * ldc + 8], _C2_1);
    _mm_storeu_pd(&C[2 * ldc + 12], _C2_2);

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm256_storeu_pd(&C[3 * ldc + 8], _C3_1);
    _mm_storeu_pd(&C[3 * ldc + 12], _C3_2);

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm256_storeu_pd(&C[4 * ldc + 8], _C4_1);
    _mm_storeu_pd(&C[4 * ldc + 12], _C4_2);

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm256_storeu_pd(&C[5 * ldc + 8], _C5_1);
    _mm_storeu_pd(&C[5 * ldc + 12], _C5_2);

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm256_storeu_pd(&C[6 * ldc + 8], _C6_1);
    _mm_storeu_pd(&C[6 * ldc + 12], _C6_2);

    _mm512_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm256_storeu_pd(&C[7 * ldc + 8], _C7_1);
    _mm_storeu_pd(&C[7 * ldc + 12], _C7_2);

    _mm512_storeu_pd(&C[8 * ldc + 0], _C8_0);
    _mm256_storeu_pd(&C[8 * ldc + 8], _C8_1);
    _mm_storeu_pd(&C[8 * ldc + 12], _C8_2);

    _mm512_storeu_pd(&C[9 * ldc + 0], _C9_0);
    _mm256_storeu_pd(&C[9 * ldc + 8], _C9_1);
    _mm_storeu_pd(&C[9 * ldc + 12], _C9_2);

    _mm512_storeu_pd(&C[10 * ldc + 0], _C10_0);
    _mm256_storeu_pd(&C[10 * ldc + 8], _C10_1);
    _mm_storeu_pd(&C[10 * ldc + 12], _C10_2);
}
void micro_kernel15x11(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m256d _A1;
    register __m128d _A2;
    register double _A3;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C0_1 = _mm256_loadu_pd(&C[0 * ldc + 8]);
    register __m128d _C0_2 = _mm_loadu_pd(&C[0 * ldc + 12]);
    register double _C0_3 = C[0 * ldc + 14];

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m256d _C1_1 = _mm256_loadu_pd(&C[1 * ldc + 8]);
    register __m128d _C1_2 = _mm_loadu_pd(&C[1 * ldc + 12]);
    register double _C1_3 = C[1 * ldc + 14];

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m256d _C2_1 = _mm256_loadu_pd(&C[2 * ldc + 8]);
    register __m128d _C2_2 = _mm_loadu_pd(&C[2 * ldc + 12]);
    register double _C2_3 = C[2 * ldc + 14];

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m256d _C3_1 = _mm256_loadu_pd(&C[3 * ldc + 8]);
    register __m128d _C3_2 = _mm_loadu_pd(&C[3 * ldc + 12]);
    register double _C3_3 = C[3 * ldc + 14];

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m256d _C4_1 = _mm256_loadu_pd(&C[4 * ldc + 8]);
    register __m128d _C4_2 = _mm_loadu_pd(&C[4 * ldc + 12]);
    register double _C4_3 = C[4 * ldc + 14];

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register __m256d _C5_1 = _mm256_loadu_pd(&C[5 * ldc + 8]);
    register __m128d _C5_2 = _mm_loadu_pd(&C[5 * ldc + 12]);
    register double _C5_3 = C[5 * ldc + 14];

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);
    register __m256d _C6_1 = _mm256_loadu_pd(&C[6 * ldc + 8]);
    register __m128d _C6_2 = _mm_loadu_pd(&C[6 * ldc + 12]);
    register double _C6_3 = C[6 * ldc + 14];

    register __m512d _C7_0 = _mm512_loadu_pd(&C[7 * ldc + 0]);
    register __m256d _C7_1 = _mm256_loadu_pd(&C[7 * ldc + 8]);
    register __m128d _C7_2 = _mm_loadu_pd(&C[7 * ldc + 12]);
    register double _C7_3 = C[7 * ldc + 14];

    register __m512d _C8_0 = _mm512_loadu_pd(&C[8 * ldc + 0]);
    register __m256d _C8_1 = _mm256_loadu_pd(&C[8 * ldc + 8]);
    register __m128d _C8_2 = _mm_loadu_pd(&C[8 * ldc + 12]);
    register double _C8_3 = C[8 * ldc + 14];

    register __m512d _C9_0 = _mm512_loadu_pd(&C[9 * ldc + 0]);
    register __m256d _C9_1 = _mm256_loadu_pd(&C[9 * ldc + 8]);
    register __m128d _C9_2 = _mm_loadu_pd(&C[9 * ldc + 12]);
    register double _C9_3 = C[9 * ldc + 14];

    register __m512d _C10_0 = _mm512_loadu_pd(&C[10 * ldc + 0]);
    register __m256d _C10_1 = _mm256_loadu_pd(&C[10 * ldc + 8]);
    register __m128d _C10_2 = _mm_loadu_pd(&C[10 * ldc + 12]);
    register double _C10_3 = C[10 * ldc + 14];
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm256_loadu_pd(&A[8]);
        _A2 = _mm_loadu_pd(&A[12]);
        _A3 = A[14];

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A2, _C0_2);
        _C0_3 += B[0 * ldb] * _A3;

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C1_2 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A2, _C1_2);
        _C1_3 += B[1 * ldb] * _A3;

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C2_2 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A2, _C2_2);
        _C2_3 += B[2 * ldb] * _A3;

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A1, _C3_1);
        _C3_2 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A2, _C3_2);
        _C3_3 += B[3 * ldb] * _A3;

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A1, _C4_1);
        _C4_2 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A2, _C4_2);
        _C4_3 += B[4 * ldb] * _A3;

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A1, _C5_1);
        _C5_2 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A2, _C5_2);
        _C5_3 += B[5 * ldb] * _A3;

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[6 * ldb]), _A1, _C6_1);
        _C6_2 = _mm_fnmadd_pd(_mm_set1_pd(B[6 * ldb]), _A2, _C6_2);
        _C6_3 += B[6 * ldb] * _A3;

        _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[7 * ldb]), _A1, _C7_1);
        _C7_2 = _mm_fnmadd_pd(_mm_set1_pd(B[7 * ldb]), _A2, _C7_2);
        _C7_3 += B[7 * ldb] * _A3;

        _C8_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[8 * ldb]), _A1, _C8_1);
        _C8_2 = _mm_fnmadd_pd(_mm_set1_pd(B[8 * ldb]), _A2, _C8_2);
        _C8_3 += B[8 * ldb] * _A3;

        _C9_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[9 * ldb]), _A0, _C9_0);
        _C9_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[9 * ldb]), _A1, _C9_1);
        _C9_2 = _mm_fnmadd_pd(_mm_set1_pd(B[9 * ldb]), _A2, _C9_2);
        _C9_3 += B[9 * ldb] * _A3;

        _C10_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[10 * ldb]), _A0, _C10_0);
        _C10_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[10 * ldb]), _A1, _C10_1);
        _C10_2 = _mm_fnmadd_pd(_mm_set1_pd(B[10 * ldb]), _A2, _C10_2);
        _C10_3 += B[10 * ldb] * _A3;
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[0 * ldc + 8], _C0_1);
    _mm_storeu_pd(&C[0 * ldc + 12], _C0_2);
    C[0 * ldc + 14] = _C0_3;

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm256_storeu_pd(&C[1 * ldc + 8], _C1_1);
    _mm_storeu_pd(&C[1 * ldc + 12], _C1_2);
    C[1 * ldc + 14] = _C1_3;

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm256_storeu_pd(&C[2 * ldc + 8], _C2_1);
    _mm_storeu_pd(&C[2 * ldc + 12], _C2_2);
    C[2 * ldc + 14] = _C2_3;

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm256_storeu_pd(&C[3 * ldc + 8], _C3_1);
    _mm_storeu_pd(&C[3 * ldc + 12], _C3_2);
    C[3 * ldc + 14] = _C3_3;

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm256_storeu_pd(&C[4 * ldc + 8], _C4_1);
    _mm_storeu_pd(&C[4 * ldc + 12], _C4_2);
    C[4 * ldc + 14] = _C4_3;

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm256_storeu_pd(&C[5 * ldc + 8], _C5_1);
    _mm_storeu_pd(&C[5 * ldc + 12], _C5_2);
    C[5 * ldc + 14] = _C5_3;

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm256_storeu_pd(&C[6 * ldc + 8], _C6_1);
    _mm_storeu_pd(&C[6 * ldc + 12], _C6_2);
    C[6 * ldc + 14] = _C6_3;

    _mm512_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm256_storeu_pd(&C[7 * ldc + 8], _C7_1);
    _mm_storeu_pd(&C[7 * ldc + 12], _C7_2);
    C[7 * ldc + 14] = _C7_3;

    _mm512_storeu_pd(&C[8 * ldc + 0], _C8_0);
    _mm256_storeu_pd(&C[8 * ldc + 8], _C8_1);
    _mm_storeu_pd(&C[8 * ldc + 12], _C8_2);
    C[8 * ldc + 14] = _C8_3;

    _mm512_storeu_pd(&C[9 * ldc + 0], _C9_0);
    _mm256_storeu_pd(&C[9 * ldc + 8], _C9_1);
    _mm_storeu_pd(&C[9 * ldc + 12], _C9_2);
    C[9 * ldc + 14] = _C9_3;

    _mm512_storeu_pd(&C[10 * ldc + 0], _C10_0);
    _mm256_storeu_pd(&C[10 * ldc + 8], _C10_1);
    _mm_storeu_pd(&C[10 * ldc + 12], _C10_2);
    C[10 * ldc + 14] = _C10_3;
}
void micro_kernel16x11(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m512d _A1;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m512d _C0_1 = _mm512_loadu_pd(&C[0 * ldc + 8]);

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m512d _C1_1 = _mm512_loadu_pd(&C[1 * ldc + 8]);

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m512d _C2_1 = _mm512_loadu_pd(&C[2 * ldc + 8]);

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m512d _C3_1 = _mm512_loadu_pd(&C[3 * ldc + 8]);

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m512d _C4_1 = _mm512_loadu_pd(&C[4 * ldc + 8]);

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register __m512d _C5_1 = _mm512_loadu_pd(&C[5 * ldc + 8]);

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);
    register __m512d _C6_1 = _mm512_loadu_pd(&C[6 * ldc + 8]);

    register __m512d _C7_0 = _mm512_loadu_pd(&C[7 * ldc + 0]);
    register __m512d _C7_1 = _mm512_loadu_pd(&C[7 * ldc + 8]);

    register __m512d _C8_0 = _mm512_loadu_pd(&C[8 * ldc + 0]);
    register __m512d _C8_1 = _mm512_loadu_pd(&C[8 * ldc + 8]);

    register __m512d _C9_0 = _mm512_loadu_pd(&C[9 * ldc + 0]);
    register __m512d _C9_1 = _mm512_loadu_pd(&C[9 * ldc + 8]);

    register __m512d _C10_0 = _mm512_loadu_pd(&C[10 * ldc + 0]);
    register __m512d _C10_1 = _mm512_loadu_pd(&C[10 * ldc + 8]);
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm512_loadu_pd(&A[8]);

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A1, _C0_1);

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A1, _C1_1);

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A1, _C2_1);

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A1, _C3_1);

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A1, _C4_1);

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A1, _C5_1);

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A1, _C6_1);

        _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A1, _C7_1);

        _C8_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8 * ldb]), _A1, _C8_1);

        _C9_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[9 * ldb]), _A0, _C9_0);
        _C9_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[9 * ldb]), _A1, _C9_1);

        _C10_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[10 * ldb]), _A0, _C10_0);
        _C10_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[10 * ldb]), _A1, _C10_1);
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm512_storeu_pd(&C[0 * ldc + 8], _C0_1);

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm512_storeu_pd(&C[1 * ldc + 8], _C1_1);

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm512_storeu_pd(&C[2 * ldc + 8], _C2_1);

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm512_storeu_pd(&C[3 * ldc + 8], _C3_1);

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm512_storeu_pd(&C[4 * ldc + 8], _C4_1);

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm512_storeu_pd(&C[5 * ldc + 8], _C5_1);

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm512_storeu_pd(&C[6 * ldc + 8], _C6_1);

    _mm512_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm512_storeu_pd(&C[7 * ldc + 8], _C7_1);

    _mm512_storeu_pd(&C[8 * ldc + 0], _C8_0);
    _mm512_storeu_pd(&C[8 * ldc + 8], _C8_1);

    _mm512_storeu_pd(&C[9 * ldc + 0], _C9_0);
    _mm512_storeu_pd(&C[9 * ldc + 8], _C9_1);

    _mm512_storeu_pd(&C[10 * ldc + 0], _C10_0);
    _mm512_storeu_pd(&C[10 * ldc + 8], _C10_1);
}
//----------------------------------------------------------------------------------------------
//################################@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@####################################
void micro_kernel1x12(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register double _A0;
    register double _C0_0 = C[0 * ldc + 0];
    register double _C1_0 = C[1 * ldc + 0];
    register double _C2_0 = C[2 * ldc + 0];
    register double _C3_0 = C[3 * ldc + 0];
    register double _C4_0 = C[4 * ldc + 0];
    register double _C5_0 = C[5 * ldc + 0];
    register double _C6_0 = C[6 * ldc + 0];
    register double _C7_0 = C[7 * ldc + 0];
    register double _C8_0 = C[8 * ldc + 0];
    register double _C9_0 = C[9 * ldc + 0];
    register double _C10_0 = C[10 * ldc + 0];
    register double _C11_0 = C[11 * ldc + 0];

    for(i = 0; i < k; ++i) {
        _A0 = A[0];
        _C0_0 += B[0 * ldb] * _A0;
        _C1_0 += B[1 * ldb] * _A0;
        _C2_0 += B[2 * ldb] * _A0;
        _C3_0 += B[3 * ldb] * _A0;
        _C4_0 += B[4 * ldb] * _A0;
        _C5_0 += B[5 * ldb] * _A0;
        _C6_0 += B[6 * ldb] * _A0;
        _C7_0 += B[7 * ldb] * _A0;
        _C8_0 += B[8 * ldb] * _A0;
        _C9_0 += B[9 * ldb] * _A0;
        _C10_0 += B[10 * ldb] * _A0;
        _C11_0 += B[11 * ldb] * _A0;

        A += lda;
        B += 1;
    }

    C[0 * ldc + 0] = _C0_0;
    C[1 * ldc + 0] = _C1_0;
    C[2 * ldc + 0] = _C2_0;
    C[3 * ldc + 0] = _C3_0;
    C[4 * ldc + 0] = _C4_0;
    C[5 * ldc + 0] = _C5_0;
    C[6 * ldc + 0] = _C6_0;
    C[7 * ldc + 0] = _C7_0;
    C[8 * ldc + 0] = _C8_0;
    C[9 * ldc + 0] = _C9_0;
    C[10 * ldc + 0] = _C10_0;
    C[11 * ldc + 0] = _C11_0;
}
void micro_kernel2x12(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m128d _A0;
    register __m128d _C0_0 = _mm_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C1_0 = _mm_loadu_pd(&C[1 * ldc + 0]);
    register __m128d _C2_0 = _mm_loadu_pd(&C[2 * ldc + 0]);
    register __m128d _C3_0 = _mm_loadu_pd(&C[3 * ldc + 0]);
    register __m128d _C4_0 = _mm_loadu_pd(&C[4 * ldc + 0]);
    register __m128d _C5_0 = _mm_loadu_pd(&C[5 * ldc + 0]);
    register __m128d _C6_0 = _mm_loadu_pd(&C[6 * ldc + 0]);
    register __m128d _C7_0 = _mm_loadu_pd(&C[7 * ldc + 0]);
    register __m128d _C8_0 = _mm_loadu_pd(&C[8 * ldc + 0]);
    register __m128d _C9_0 = _mm_loadu_pd(&C[9 * ldc + 0]);
    register __m128d _C10_0 = _mm_loadu_pd(&C[10 * ldc + 0]);
    register __m128d _C11_0 = _mm_loadu_pd(&C[11 * ldc + 0]);

    for(i = 0; i < k; ++i) {
        _A0 = _mm_loadu_pd(&A[0]);
        _C0_0 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C1_0 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C2_0 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C3_0 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C4_0 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C5_0 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C6_0 = _mm_fnmadd_pd(_mm_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C7_0 = _mm_fnmadd_pd(_mm_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C8_0 = _mm_fnmadd_pd(_mm_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C9_0 = _mm_fnmadd_pd(_mm_set1_pd(B[9 * ldb]), _A0, _C9_0);
        _C10_0 = _mm_fnmadd_pd(_mm_set1_pd(B[10 * ldb]), _A0, _C10_0);
        _C11_0 = _mm_fnmadd_pd(_mm_set1_pd(B[11 * ldb]), _A0, _C11_0);

        A += lda;
        B += 1;
    }

    _mm_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm_storeu_pd(&C[8 * ldc + 0], _C8_0);
    _mm_storeu_pd(&C[9 * ldc + 0], _C9_0);
    _mm_storeu_pd(&C[10 * ldc + 0], _C10_0);
    _mm_storeu_pd(&C[11 * ldc + 0], _C11_0);
}
void micro_kernel3x12(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m128d _A0;
    register double _A1;
    register __m128d _C0_0 = _mm_loadu_pd(&C[0 * ldc + 0]);
    register double _C0_1 = C[0 * ldc + 2];
    register __m128d _C1_0 = _mm_loadu_pd(&C[1 * ldc + 0]);
    register double _C1_1 = C[1 * ldc + 2];
    register __m128d _C2_0 = _mm_loadu_pd(&C[2 * ldc + 0]);
    register double _C2_1 = C[2 * ldc + 2];
    register __m128d _C3_0 = _mm_loadu_pd(&C[3 * ldc + 0]);
    register double _C3_1 = C[3 * ldc + 2];
    register __m128d _C4_0 = _mm_loadu_pd(&C[4 * ldc + 0]);
    register double _C4_1 = C[4 * ldc + 2];
    register __m128d _C5_0 = _mm_loadu_pd(&C[5 * ldc + 0]);
    register double _C5_1 = C[5 * ldc + 2];

    register __m128d _C6_0 = _mm_loadu_pd(&C[6 * ldc + 0]);
    register double _C6_1 = C[6 * ldc + 2];

    register __m128d _C7_0 = _mm_loadu_pd(&C[7 * ldc + 0]);
    register double _C7_1 = C[7 * ldc + 2];

    register __m128d _C8_0 = _mm_loadu_pd(&C[8 * ldc + 0]);
    register double _C8_1 = C[8 * ldc + 2];

    register __m128d _C9_0 = _mm_loadu_pd(&C[9 * ldc + 0]);
    register double _C9_1 = C[9 * ldc + 2];

    register __m128d _C10_0 = _mm_loadu_pd(&C[10 * ldc + 0]);
    register double _C10_1 = C[10 * ldc + 2];

    register __m128d _C11_0 = _mm_loadu_pd(&C[11 * ldc + 0]);
    register double _C11_1 = C[11 * ldc + 2];

    for(i = 0; i < k; ++i) {
        _A0 = _mm_loadu_pd(&A[0]);
        _A1 = A[2];
        _C0_0 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 += B[0 * ldb] * _A1;
        _C1_0 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 += B[1 * ldb] * _A1;
        _C2_0 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 += B[2 * ldb] * _A1;
        _C3_0 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 += B[3 * ldb] * _A1;
        _C4_0 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 += B[4 * ldb] * _A1;
        _C5_0 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 += B[5 * ldb] * _A1;

        _C6_0 = _mm_fnmadd_pd(_mm_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 += B[6 * ldb] * _A1;

        _C7_0 = _mm_fnmadd_pd(_mm_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 += B[7 * ldb] * _A1;

        _C8_0 = _mm_fnmadd_pd(_mm_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 += B[8 * ldb] * _A1;

        _C9_0 = _mm_fnmadd_pd(_mm_set1_pd(B[9 * ldb]), _A0, _C9_0);
        _C9_1 += B[9 * ldb] * _A1;

        _C10_0 = _mm_fnmadd_pd(_mm_set1_pd(B[10 * ldb]), _A0, _C10_0);
        _C10_1 += B[10 * ldb] * _A1;

        _C11_0 = _mm_fnmadd_pd(_mm_set1_pd(B[11 * ldb]), _A0, _C11_0);
        _C11_1 += B[11 * ldb] * _A1;

        A += lda;
        B += 1;
    }

    _mm_storeu_pd(&C[0 * ldc + 0], _C0_0);
    C[0 * ldc + 2] = _C0_1;
    _mm_storeu_pd(&C[1 * ldc + 0], _C1_0);
    C[1 * ldc + 2] = _C1_1;
    _mm_storeu_pd(&C[2 * ldc + 0], _C2_0);
    C[2 * ldc + 2] = _C2_1;
    _mm_storeu_pd(&C[3 * ldc + 0], _C3_0);
    C[3 * ldc + 2] = _C3_1;
    _mm_storeu_pd(&C[4 * ldc + 0], _C4_0);
    C[4 * ldc + 2] = _C4_1;
    _mm_storeu_pd(&C[5 * ldc + 0], _C5_0);
    C[5 * ldc + 2] = _C5_1;

    _mm_storeu_pd(&C[6 * ldc + 0], _C6_0);
    C[6 * ldc + 2] = _C6_1;

    _mm_storeu_pd(&C[7 * ldc + 0], _C7_0);
    C[7 * ldc + 2] = _C7_1;

    _mm_storeu_pd(&C[8 * ldc + 0], _C8_0);
    C[8 * ldc + 2] = _C8_1;

    _mm_storeu_pd(&C[9 * ldc + 0], _C9_0);
    C[9 * ldc + 2] = _C9_1;

    _mm_storeu_pd(&C[10 * ldc + 0], _C10_0);
    C[10 * ldc + 2] = _C10_1;

    _mm_storeu_pd(&C[11 * ldc + 0], _C11_0);
    C[11 * ldc + 2] = _C11_1;
}
void micro_kernel4x12(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m256d _A0;
    register __m256d _C0_0 = _mm256_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C1_0 = _mm256_loadu_pd(&C[1 * ldc + 0]);
    register __m256d _C2_0 = _mm256_loadu_pd(&C[2 * ldc + 0]);
    register __m256d _C3_0 = _mm256_loadu_pd(&C[3 * ldc + 0]);
    register __m256d _C4_0 = _mm256_loadu_pd(&C[4 * ldc + 0]);
    register __m256d _C5_0 = _mm256_loadu_pd(&C[5 * ldc + 0]);

    register __m256d _C6_0 = _mm256_loadu_pd(&C[6 * ldc + 0]);

    register __m256d _C7_0 = _mm256_loadu_pd(&C[7 * ldc + 0]);

    register __m256d _C8_0 = _mm256_loadu_pd(&C[8 * ldc + 0]);

    register __m256d _C9_0 = _mm256_loadu_pd(&C[9 * ldc + 0]);

    register __m256d _C10_0 = _mm256_loadu_pd(&C[10 * ldc + 0]);

    register __m256d _C11_0 = _mm256_loadu_pd(&C[11 * ldc + 0]);

    for(i = 0; i < k; ++i) {
        _A0 = _mm256_loadu_pd(&A[0]);
        _C0_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C1_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C2_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C3_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C4_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C5_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A0, _C5_0);

        _C6_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[6 * ldb]), _A0, _C6_0);

        _C7_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[7 * ldb]), _A0, _C7_0);

        _C8_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[8 * ldb]), _A0, _C8_0);

        _C9_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[9 * ldb]), _A0, _C9_0);

        _C10_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[10 * ldb]), _A0, _C10_0);

        _C11_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[11 * ldb]), _A0, _C11_0);

        A += lda;
        B += 1;
    }

    _mm256_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm256_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm256_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm256_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm256_storeu_pd(&C[5 * ldc + 0], _C5_0);

    _mm256_storeu_pd(&C[6 * ldc + 0], _C6_0);

    _mm256_storeu_pd(&C[7 * ldc + 0], _C7_0);

    _mm256_storeu_pd(&C[8 * ldc + 0], _C8_0);

    _mm256_storeu_pd(&C[9 * ldc + 0], _C9_0);

    _mm256_storeu_pd(&C[10 * ldc + 0], _C10_0);

    _mm256_storeu_pd(&C[11 * ldc + 0], _C11_0);
}
void micro_kernel5x12(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m256d _A0;
    register double _A1;
    register __m256d _C0_0 = _mm256_loadu_pd(&C[0 * ldc + 0]);
    register double _C0_1 = C[0 * ldc + 4];
    register __m256d _C1_0 = _mm256_loadu_pd(&C[1 * ldc + 0]);
    register double _C1_1 = C[1 * ldc + 4];
    register __m256d _C2_0 = _mm256_loadu_pd(&C[2 * ldc + 0]);
    register double _C2_1 = C[2 * ldc + 4];
    register __m256d _C3_0 = _mm256_loadu_pd(&C[3 * ldc + 0]);
    register double _C3_1 = C[3 * ldc + 4];
    register __m256d _C4_0 = _mm256_loadu_pd(&C[4 * ldc + 0]);
    register double _C4_1 = C[4 * ldc + 4];
    register __m256d _C5_0 = _mm256_loadu_pd(&C[5 * ldc + 0]);
    register double _C5_1 = C[5 * ldc + 4];

    register __m256d _C6_0 = _mm256_loadu_pd(&C[6 * ldc + 0]);
    register double _C6_1 = C[6 * ldc + 4];

    register __m256d _C7_0 = _mm256_loadu_pd(&C[7 * ldc + 0]);
    register double _C7_1 = C[7 * ldc + 4];

    register __m256d _C8_0 = _mm256_loadu_pd(&C[8 * ldc + 0]);
    register double _C8_1 = C[8 * ldc + 4];

    register __m256d _C9_0 = _mm256_loadu_pd(&C[9 * ldc + 0]);
    register double _C9_1 = C[9 * ldc + 4];

    register __m256d _C10_0 = _mm256_loadu_pd(&C[10 * ldc + 0]);
    register double _C10_1 = C[10 * ldc + 4];

    register __m256d _C11_0 = _mm256_loadu_pd(&C[11 * ldc + 0]);
    register double _C11_1 = C[11 * ldc + 4];

    for(i = 0; i < k; ++i) {
        _A0 = _mm256_loadu_pd(&A[0]);
        _A1 = A[4];
        _C0_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 += B[0 * ldb] * _A1;
        _C1_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 += B[1 * ldb] * _A1;
        _C2_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 += B[2 * ldb] * _A1;
        _C3_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 += B[3 * ldb] * _A1;
        _C4_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 += B[4 * ldb] * _A1;
        _C5_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 += B[5 * ldb] * _A1;

        _C6_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 += B[6 * ldb] * _A1;

        _C7_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 += B[7 * ldb] * _A1;

        _C8_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 += B[8 * ldb] * _A1;

        _C9_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[9 * ldb]), _A0, _C9_0);
        _C9_1 += B[9 * ldb] * _A1;

        _C10_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[10 * ldb]), _A0, _C10_0);
        _C10_1 += B[10 * ldb] * _A1;

        _C11_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[11 * ldb]), _A0, _C11_0);
        _C11_1 += B[11 * ldb] * _A1;

        A += lda;
        B += 1;
    }

    _mm256_storeu_pd(&C[0 * ldc + 0], _C0_0);
    C[0 * ldc + 4] = _C0_1;
    _mm256_storeu_pd(&C[1 * ldc + 0], _C1_0);
    C[1 * ldc + 4] = _C1_1;
    _mm256_storeu_pd(&C[2 * ldc + 0], _C2_0);
    C[2 * ldc + 4] = _C2_1;
    _mm256_storeu_pd(&C[3 * ldc + 0], _C3_0);
    C[3 * ldc + 4] = _C3_1;
    _mm256_storeu_pd(&C[4 * ldc + 0], _C4_0);
    C[4 * ldc + 4] = _C4_1;
    _mm256_storeu_pd(&C[5 * ldc + 0], _C5_0);
    C[5 * ldc + 4] = _C5_1;

    _mm256_storeu_pd(&C[6 * ldc + 0], _C6_0);
    C[6 * ldc + 4] = _C6_1;

    _mm256_storeu_pd(&C[7 * ldc + 0], _C7_0);
    C[7 * ldc + 4] = _C7_1;

    _mm256_storeu_pd(&C[8 * ldc + 0], _C8_0);
    C[8 * ldc + 4] = _C8_1;

    _mm256_storeu_pd(&C[9 * ldc + 0], _C9_0);
    C[9 * ldc + 4] = _C9_1;

    _mm256_storeu_pd(&C[10 * ldc + 0], _C10_0);
    C[10 * ldc + 4] = _C10_1;

    _mm256_storeu_pd(&C[11 * ldc + 0], _C11_0);
    C[11 * ldc + 4] = _C11_1;
}
void micro_kernel6x12(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m256d _A0;
    register __m128d _A1;
    register __m256d _C0_0 = _mm256_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C0_1 = _mm_loadu_pd(&C[0 * ldc + 4]);
    register __m256d _C1_0 = _mm256_loadu_pd(&C[1 * ldc + 0]);
    register __m128d _C1_1 = _mm_loadu_pd(&C[1 * ldc + 4]);
    register __m256d _C2_0 = _mm256_loadu_pd(&C[2 * ldc + 0]);
    register __m128d _C2_1 = _mm_loadu_pd(&C[2 * ldc + 4]);
    register __m256d _C3_0 = _mm256_loadu_pd(&C[3 * ldc + 0]);
    register __m128d _C3_1 = _mm_loadu_pd(&C[3 * ldc + 4]);
    register __m256d _C4_0 = _mm256_loadu_pd(&C[4 * ldc + 0]);
    register __m128d _C4_1 = _mm_loadu_pd(&C[4 * ldc + 4]);
    register __m256d _C5_0 = _mm256_loadu_pd(&C[5 * ldc + 0]);
    register __m128d _C5_1 = _mm_loadu_pd(&C[5 * ldc + 4]);

    register __m256d _C6_0 = _mm256_loadu_pd(&C[6 * ldc + 0]);
    register __m128d _C6_1 = _mm_loadu_pd(&C[6 * ldc + 4]);

    register __m256d _C7_0 = _mm256_loadu_pd(&C[7 * ldc + 0]);
    register __m128d _C7_1 = _mm_loadu_pd(&C[7 * ldc + 4]);

    register __m256d _C8_0 = _mm256_loadu_pd(&C[8 * ldc + 0]);
    register __m128d _C8_1 = _mm_loadu_pd(&C[8 * ldc + 4]);

    register __m256d _C9_0 = _mm256_loadu_pd(&C[9 * ldc + 0]);
    register __m128d _C9_1 = _mm_loadu_pd(&C[9 * ldc + 4]);

    register __m256d _C10_0 = _mm256_loadu_pd(&C[10 * ldc + 0]);
    register __m128d _C10_1 = _mm_loadu_pd(&C[10 * ldc + 4]);

    register __m256d _C11_0 = _mm256_loadu_pd(&C[11 * ldc + 0]);
    register __m128d _C11_1 = _mm_loadu_pd(&C[11 * ldc + 4]);

    for(i = 0; i < k; ++i) {
        _A0 = _mm256_loadu_pd(&A[0]);
        _A1 = _mm_loadu_pd(&A[4]);
        _C0_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C1_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C2_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C3_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A1, _C3_1);
        _C4_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A1, _C4_1);
        _C5_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A1, _C5_1);

        _C6_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm_fnmadd_pd(_mm_set1_pd(B[6 * ldb]), _A1, _C6_1);

        _C7_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 = _mm_fnmadd_pd(_mm_set1_pd(B[7 * ldb]), _A1, _C7_1);

        _C8_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 = _mm_fnmadd_pd(_mm_set1_pd(B[8 * ldb]), _A1, _C8_1);

        _C9_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[9 * ldb]), _A0, _C9_0);
        _C9_1 = _mm_fnmadd_pd(_mm_set1_pd(B[9 * ldb]), _A1, _C9_1);

        _C10_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[10 * ldb]), _A0, _C10_0);
        _C10_1 = _mm_fnmadd_pd(_mm_set1_pd(B[10 * ldb]), _A1, _C10_1);

        _C11_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[11 * ldb]), _A0, _C11_0);
        _C11_1 = _mm_fnmadd_pd(_mm_set1_pd(B[11 * ldb]), _A1, _C11_1);

        A += lda;
        B += 1;
    }

    _mm256_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[0 * ldc + 4], _C0_1);
    _mm256_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm_storeu_pd(&C[1 * ldc + 4], _C1_1);
    _mm256_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm_storeu_pd(&C[2 * ldc + 4], _C2_1);
    _mm256_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm_storeu_pd(&C[3 * ldc + 4], _C3_1);
    _mm256_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm_storeu_pd(&C[4 * ldc + 4], _C4_1);
    _mm256_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm_storeu_pd(&C[5 * ldc + 4], _C5_1);

    _mm256_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm_storeu_pd(&C[6 * ldc + 4], _C6_1);

    _mm256_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm_storeu_pd(&C[7 * ldc + 4], _C7_1);

    _mm256_storeu_pd(&C[8 * ldc + 0], _C8_0);
    _mm_storeu_pd(&C[8 * ldc + 4], _C8_1);

    _mm256_storeu_pd(&C[9 * ldc + 0], _C9_0);
    _mm_storeu_pd(&C[9 * ldc + 4], _C9_1);

    _mm256_storeu_pd(&C[10 * ldc + 0], _C10_0);
    _mm_storeu_pd(&C[10 * ldc + 4], _C10_1);

    _mm256_storeu_pd(&C[11 * ldc + 0], _C11_0);
    _mm_storeu_pd(&C[11 * ldc + 4], _C11_1);
}
void micro_kernel7x12(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m256d _A0;
    register __m128d _A1;
    register double _A2;
    register __m256d _C0_0 = _mm256_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C0_1 = _mm_loadu_pd(&C[0 * ldc + 4]);
    register double _C0_2 = C[0 * ldc + 6];
    register __m256d _C1_0 = _mm256_loadu_pd(&C[1 * ldc + 0]);
    register __m128d _C1_1 = _mm_loadu_pd(&C[1 * ldc + 4]);
    register double _C1_2 = C[1 * ldc + 6];
    register __m256d _C2_0 = _mm256_loadu_pd(&C[2 * ldc + 0]);
    register __m128d _C2_1 = _mm_loadu_pd(&C[2 * ldc + 4]);
    register double _C2_2 = C[2 * ldc + 6];
    register __m256d _C3_0 = _mm256_loadu_pd(&C[3 * ldc + 0]);
    register __m128d _C3_1 = _mm_loadu_pd(&C[3 * ldc + 4]);
    register double _C3_2 = C[3 * ldc + 6];
    register __m256d _C4_0 = _mm256_loadu_pd(&C[4 * ldc + 0]);
    register __m128d _C4_1 = _mm_loadu_pd(&C[4 * ldc + 4]);
    register double _C4_2 = C[4 * ldc + 6];
    register __m256d _C5_0 = _mm256_loadu_pd(&C[5 * ldc + 0]);
    register __m128d _C5_1 = _mm_loadu_pd(&C[5 * ldc + 4]);
    register double _C5_2 = C[5 * ldc + 6];

    register __m256d _C6_0 = _mm256_loadu_pd(&C[6 * ldc + 0]);
    register __m128d _C6_1 = _mm_loadu_pd(&C[6 * ldc + 4]);
    register double _C6_2 = C[6 * ldc + 6];

    register __m256d _C7_0 = _mm256_loadu_pd(&C[7 * ldc + 0]);
    register __m128d _C7_1 = _mm_loadu_pd(&C[7 * ldc + 4]);
    register double _C7_2 = C[7 * ldc + 6];

    register __m256d _C8_0 = _mm256_loadu_pd(&C[8 * ldc + 0]);
    register __m128d _C8_1 = _mm_loadu_pd(&C[8 * ldc + 4]);
    register double _C8_2 = C[8 * ldc + 6];

    register __m256d _C9_0 = _mm256_loadu_pd(&C[9 * ldc + 0]);
    register __m128d _C9_1 = _mm_loadu_pd(&C[9 * ldc + 4]);
    register double _C9_2 = C[9 * ldc + 6];

    register __m256d _C10_0 = _mm256_loadu_pd(&C[10 * ldc + 0]);
    register __m128d _C10_1 = _mm_loadu_pd(&C[10 * ldc + 4]);
    register double _C10_2 = C[10 * ldc + 6];

    register __m256d _C11_0 = _mm256_loadu_pd(&C[11 * ldc + 0]);
    register __m128d _C11_1 = _mm_loadu_pd(&C[11 * ldc + 4]);
    register double _C11_2 = C[11 * ldc + 6];

    for(i = 0; i < k; ++i) {
        _A0 = _mm256_loadu_pd(&A[0]);
        _A1 = _mm_loadu_pd(&A[4]);
        _A2 = A[6];
        _C0_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 += B[0 * ldb] * _A2;
        _C1_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C1_2 += B[1 * ldb] * _A2;
        _C2_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C2_2 += B[2 * ldb] * _A2;
        _C3_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A1, _C3_1);
        _C3_2 += B[3 * ldb] * _A2;
        _C4_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A1, _C4_1);
        _C4_2 += B[4 * ldb] * _A2;
        _C5_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A1, _C5_1);
        _C5_2 += B[5 * ldb] * _A2;

        _C6_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm_fnmadd_pd(_mm_set1_pd(B[6 * ldb]), _A1, _C6_1);
        _C6_2 += B[6 * ldb] * _A2;

        _C7_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 = _mm_fnmadd_pd(_mm_set1_pd(B[7 * ldb]), _A1, _C7_1);
        _C7_2 += B[7 * ldb] * _A2;

        _C8_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 = _mm_fnmadd_pd(_mm_set1_pd(B[8 * ldb]), _A1, _C8_1);
        _C8_2 += B[8 * ldb] * _A2;

        _C9_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[9 * ldb]), _A0, _C9_0);
        _C9_1 = _mm_fnmadd_pd(_mm_set1_pd(B[9 * ldb]), _A1, _C9_1);
        _C9_2 += B[9 * ldb] * _A2;

        _C10_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[10 * ldb]), _A0, _C10_0);
        _C10_1 = _mm_fnmadd_pd(_mm_set1_pd(B[10 * ldb]), _A1, _C10_1);
        _C10_2 += B[10 * ldb] * _A2;

        _C11_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[11 * ldb]), _A0, _C11_0);
        _C11_1 = _mm_fnmadd_pd(_mm_set1_pd(B[11 * ldb]), _A1, _C11_1);
        _C11_2 += B[11 * ldb] * _A2;

        A += lda;
        B += 1;
    }

    _mm256_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[0 * ldc + 4], _C0_1);
    C[0 * ldc + 6] = _C0_2;
    _mm256_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm_storeu_pd(&C[1 * ldc + 4], _C1_1);
    C[1 * ldc + 6] = _C1_2;
    _mm256_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm_storeu_pd(&C[2 * ldc + 4], _C2_1);
    C[2 * ldc + 6] = _C2_2;
    _mm256_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm_storeu_pd(&C[3 * ldc + 4], _C3_1);
    C[3 * ldc + 6] = _C3_2;
    _mm256_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm_storeu_pd(&C[4 * ldc + 4], _C4_1);
    C[4 * ldc + 6] = _C4_2;
    _mm256_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm_storeu_pd(&C[5 * ldc + 4], _C5_1);
    C[5 * ldc + 6] = _C5_2;

    _mm256_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm_storeu_pd(&C[6 * ldc + 4], _C6_1);
    C[6 * ldc + 6] = _C6_2;

    _mm256_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm_storeu_pd(&C[7 * ldc + 4], _C7_1);
    C[7 * ldc + 6] = _C7_2;

    _mm256_storeu_pd(&C[8 * ldc + 0], _C8_0);
    _mm_storeu_pd(&C[8 * ldc + 4], _C8_1);
    C[8 * ldc + 6] = _C8_2;

    _mm256_storeu_pd(&C[9 * ldc + 0], _C9_0);
    _mm_storeu_pd(&C[9 * ldc + 4], _C9_1);
    C[9 * ldc + 6] = _C9_2;

    _mm256_storeu_pd(&C[10 * ldc + 0], _C10_0);
    _mm_storeu_pd(&C[10 * ldc + 4], _C10_1);
    C[10 * ldc + 6] = _C10_2;

    _mm256_storeu_pd(&C[11 * ldc + 0], _C11_0);
    _mm_storeu_pd(&C[11 * ldc + 4], _C11_1);
    C[11 * ldc + 6] = _C11_2;
}
//=====================================================================================================================
void micro_kernel8x12(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);
    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);

    register __m512d _C7_0 = _mm512_loadu_pd(&C[7 * ldc + 0]);

    register __m512d _C8_0 = _mm512_loadu_pd(&C[8 * ldc + 0]);

    register __m512d _C9_0 = _mm512_loadu_pd(&C[9 * ldc + 0]);

    register __m512d _C10_0 = _mm512_loadu_pd(&C[10 * ldc + 0]);

    register __m512d _C11_0 = _mm512_loadu_pd(&C[11 * ldc + 0]);

    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);

        _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A0, _C7_0);

        _C8_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8 * ldb]), _A0, _C8_0);

        _C9_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[9 * ldb]), _A0, _C9_0);

        _C10_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[10 * ldb]), _A0, _C10_0);

        _C11_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[11 * ldb]), _A0, _C11_0);

        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);

    _mm512_storeu_pd(&C[7 * ldc + 0], _C7_0);

    _mm512_storeu_pd(&C[8 * ldc + 0], _C8_0);

    _mm512_storeu_pd(&C[9 * ldc + 0], _C9_0);

    _mm512_storeu_pd(&C[10 * ldc + 0], _C10_0);

    _mm512_storeu_pd(&C[11 * ldc + 0], _C11_0);
}
void micro_kernel9x12(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register double _A1;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register double _C0_1 = C[0 * ldc + 8];

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);
    register double _C1_1 = C[1 * ldc + 8];

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register double _C2_1 = C[2 * ldc + 8];

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register double _C3_1 = C[3 * ldc + 8];

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register double _C4_1 = C[4 * ldc + 8];

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register double _C5_1 = C[5 * ldc + 8];

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);
    register double _C6_1 = C[6 * ldc + 8];

    register __m512d _C7_0 = _mm512_loadu_pd(&C[7 * ldc + 0]);
    register double _C7_1 = C[7 * ldc + 8];

    register __m512d _C8_0 = _mm512_loadu_pd(&C[8 * ldc + 0]);
    register double _C8_1 = C[8 * ldc + 8];

    register __m512d _C9_0 = _mm512_loadu_pd(&C[9 * ldc + 0]);
    register double _C9_1 = C[9 * ldc + 8];

    register __m512d _C10_0 = _mm512_loadu_pd(&C[10 * ldc + 0]);
    register double _C10_1 = C[10 * ldc + 8];

    register __m512d _C11_0 = _mm512_loadu_pd(&C[11 * ldc + 0]);
    register double _C11_1 = C[11 * ldc + 8];

    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = A[8];

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 += B[0 * ldb] * _A1;

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 += B[1 * ldb] * _A1;

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 += B[2 * ldb] * _A1;

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 += B[3 * ldb] * _A1;

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 += B[4 * ldb] * _A1;

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 += B[5 * ldb] * _A1;

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 += B[6 * ldb] * _A1;
        
        _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 += B[7 * ldb] * _A1;

        _C8_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 += B[8 * ldb] * _A1;

        _C9_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[9 * ldb]), _A0, _C9_0);
        _C9_1 += B[9 * ldb] * _A1;

        _C10_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[10 * ldb]), _A0, _C10_0);
        _C10_1 += B[10 * ldb] * _A1;

        _C11_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[11 * ldb]), _A0, _C11_0);
        _C11_1 += B[11 * ldb] * _A1;

        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    C[0 * ldc + 8] = _C0_1;

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    C[1 * ldc + 8] = _C1_1;

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    C[2 * ldc + 8] = _C2_1;

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    C[3 * ldc + 8] = _C3_1;

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    C[4 * ldc + 8] = _C4_1;

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    C[5 * ldc + 8] = _C5_1;

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);
    C[6 * ldc + 8] = _C6_1;

    _mm512_storeu_pd(&C[7 * ldc + 0], _C7_0);
    C[7 * ldc + 8] = _C7_1;

    _mm512_storeu_pd(&C[8 * ldc + 0], _C8_0);
    C[8 * ldc + 8] = _C8_1;

    _mm512_storeu_pd(&C[9 * ldc + 0], _C9_0);
    C[9 * ldc + 8] = _C9_1;

    _mm512_storeu_pd(&C[10 * ldc + 0], _C10_0);
    C[10 * ldc + 8] = _C10_1;

    _mm512_storeu_pd(&C[11 * ldc + 0], _C11_0);
    C[11 * ldc + 8] = _C11_1;
}
void micro_kernel10x12(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m128d _A1;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C0_1 = _mm_loadu_pd(&C[0 * ldc + 8]);

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m128d _C1_1 = _mm_loadu_pd(&C[1 * ldc + 8]);

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m128d _C2_1 = _mm_loadu_pd(&C[2 * ldc + 8]);

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m128d _C3_1 = _mm_loadu_pd(&C[3 * ldc + 8]);

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m128d _C4_1 = _mm_loadu_pd(&C[4 * ldc + 8]);

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register __m128d _C5_1 = _mm_loadu_pd(&C[5 * ldc + 8]);

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);
    register __m128d _C6_1 = _mm_loadu_pd(&C[6 * ldc + 8]);

    register __m512d _C7_0 = _mm512_loadu_pd(&C[7 * ldc + 0]);
    register __m128d _C7_1 = _mm_loadu_pd(&C[7 * ldc + 8]);

    register __m512d _C8_0 = _mm512_loadu_pd(&C[8 * ldc + 0]);
    register __m128d _C8_1 = _mm_loadu_pd(&C[8 * ldc + 8]);

    register __m512d _C9_0 = _mm512_loadu_pd(&C[9 * ldc + 0]);
    register __m128d _C9_1 = _mm_loadu_pd(&C[9 * ldc + 8]);

    register __m512d _C10_0 = _mm512_loadu_pd(&C[10 * ldc + 0]);
    register __m128d _C10_1 = _mm_loadu_pd(&C[10 * ldc + 8]);

    register __m512d _C11_0 = _mm512_loadu_pd(&C[11 * ldc + 0]);
    register __m128d _C11_1 = _mm_loadu_pd(&C[11 * ldc + 8]);
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm_loadu_pd(&A[8]);

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A1, _C0_1);

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A1, _C1_1);

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A1, _C2_1);

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A1, _C3_1);

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A1, _C4_1);

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A1, _C5_1);

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm_fnmadd_pd(_mm_set1_pd(B[6 * ldb]), _A1, _C6_1);

        _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 = _mm_fnmadd_pd(_mm_set1_pd(B[7 * ldb]), _A1, _C7_1);

        _C8_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 = _mm_fnmadd_pd(_mm_set1_pd(B[8 * ldb]), _A1, _C8_1);

        _C9_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[9 * ldb]), _A0, _C9_0);
        _C9_1 = _mm_fnmadd_pd(_mm_set1_pd(B[9 * ldb]), _A1, _C9_1);

        _C10_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[10 * ldb]), _A0, _C10_0);
        _C10_1 = _mm_fnmadd_pd(_mm_set1_pd(B[10 * ldb]), _A1, _C10_1);

        _C11_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[11 * ldb]), _A0, _C11_0);
        _C11_1 = _mm_fnmadd_pd(_mm_set1_pd(B[11 * ldb]), _A1, _C11_1);
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[0 * ldc + 8], _C0_1);

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm_storeu_pd(&C[1 * ldc + 8], _C1_1);

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm_storeu_pd(&C[2 * ldc + 8], _C2_1);

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm_storeu_pd(&C[3 * ldc + 8], _C3_1);

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm_storeu_pd(&C[4 * ldc + 8], _C4_1);

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm_storeu_pd(&C[5 * ldc + 8], _C5_1);

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm_storeu_pd(&C[6 * ldc + 8], _C6_1);

    _mm512_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm_storeu_pd(&C[7 * ldc + 8], _C7_1);

    _mm512_storeu_pd(&C[8 * ldc + 0], _C8_0);
    _mm_storeu_pd(&C[8 * ldc + 8], _C8_1);

    _mm512_storeu_pd(&C[9 * ldc + 0], _C9_0);
    _mm_storeu_pd(&C[9 * ldc + 8], _C9_1);

    _mm512_storeu_pd(&C[10 * ldc + 0], _C10_0);
    _mm_storeu_pd(&C[10 * ldc + 8], _C10_1);

    _mm512_storeu_pd(&C[11 * ldc + 0], _C11_0);
    _mm_storeu_pd(&C[11 * ldc + 8], _C11_1);
}
void micro_kernel11x12(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m128d _A1;
    register double _A2;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C0_1 = _mm_loadu_pd(&C[0 * ldc + 8]);
    register double _C0_2 = C[0 * ldc + 10];

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m128d _C1_1 = _mm_loadu_pd(&C[1 * ldc + 8]);
    register double _C1_2 = C[1 * ldc + 10];

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m128d _C2_1 = _mm_loadu_pd(&C[2 * ldc + 8]);
    register double _C2_2 = C[2 * ldc + 10];

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m128d _C3_1 = _mm_loadu_pd(&C[3 * ldc + 8]);
    register double _C3_2 = C[3 * ldc + 10];

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m128d _C4_1 = _mm_loadu_pd(&C[4 * ldc + 8]);
    register double _C4_2 = C[4 * ldc + 10];

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register __m128d _C5_1 = _mm_loadu_pd(&C[5 * ldc + 8]);
    register double _C5_2 = C[5 * ldc + 10];

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);
    register __m128d _C6_1 = _mm_loadu_pd(&C[6 * ldc + 8]);
    register double _C6_2 = C[6 * ldc + 10];

    register __m512d _C7_0 = _mm512_loadu_pd(&C[7 * ldc + 0]);
    register __m128d _C7_1 = _mm_loadu_pd(&C[7 * ldc + 8]);
    register double _C7_2 = C[7 * ldc + 10];

    register __m512d _C8_0 = _mm512_loadu_pd(&C[8 * ldc + 0]);
    register __m128d _C8_1 = _mm_loadu_pd(&C[8 * ldc + 8]);
    register double _C8_2 = C[8 * ldc + 10];

    register __m512d _C9_0 = _mm512_loadu_pd(&C[9 * ldc + 0]);
    register __m128d _C9_1 = _mm_loadu_pd(&C[9 * ldc + 8]);
    register double _C9_2 = C[9 * ldc + 10];

    register __m512d _C10_0 = _mm512_loadu_pd(&C[10 * ldc + 0]);
    register __m128d _C10_1 = _mm_loadu_pd(&C[10 * ldc + 8]);
    register double _C10_2 = C[10 * ldc + 10];
    
    register __m512d _C11_0 = _mm512_loadu_pd(&C[11 * ldc + 0]);
    register __m128d _C11_1 = _mm_loadu_pd(&C[11 * ldc + 8]);
    register double _C11_2 = C[11 * ldc + 10];

    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm_loadu_pd(&A[8]);
        _A2 = A[10];

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 += B[0 * ldb] * _A2;

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C1_2 += B[1 * ldb] * _A2;

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C2_2 += B[2 * ldb] * _A2;

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A1, _C3_1);
        _C3_2 += B[3 * ldb] * _A2;

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A1, _C4_1);
        _C4_2 += B[4 * ldb] * _A2;

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A1, _C5_1);
        _C5_2 += B[5 * ldb] * _A2;

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm_fnmadd_pd(_mm_set1_pd(B[6 * ldb]), _A1, _C6_1);
        _C6_2 += B[6 * ldb] * _A2;

        _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 = _mm_fnmadd_pd(_mm_set1_pd(B[7 * ldb]), _A1, _C7_1);
        _C7_2 += B[7 * ldb] * _A2;

        _C8_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 = _mm_fnmadd_pd(_mm_set1_pd(B[8 * ldb]), _A1, _C8_1);
        _C8_2 += B[8 * ldb] * _A2;

        _C9_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[9 * ldb]), _A0, _C9_0);
        _C9_1 = _mm_fnmadd_pd(_mm_set1_pd(B[9 * ldb]), _A1, _C9_1);
        _C9_2 += B[9 * ldb] * _A2;

        _C10_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[10 * ldb]), _A0, _C10_0);
        _C10_1 = _mm_fnmadd_pd(_mm_set1_pd(B[10 * ldb]), _A1, _C10_1);
        _C10_2 += B[10 * ldb] * _A2;
        
        _C11_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[11 * ldb]), _A0, _C11_0);
        _C11_1 = _mm_fnmadd_pd(_mm_set1_pd(B[11 * ldb]), _A1, _C11_1);
        _C11_2 += B[11 * ldb] * _A2;

        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[0 * ldc + 8], _C0_1);
    C[0 * ldc + 10] = _C0_2;

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm_storeu_pd(&C[1 * ldc + 8], _C1_1);
    C[1 * ldc + 10] = _C1_2;

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm_storeu_pd(&C[2 * ldc + 8], _C2_1);
    C[2 * ldc + 10] = _C2_2;

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm_storeu_pd(&C[3 * ldc + 8], _C3_1);
    C[3 * ldc + 10] = _C3_2;

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm_storeu_pd(&C[4 * ldc + 8], _C4_1);
    C[4 * ldc + 10] = _C4_2;

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm_storeu_pd(&C[5 * ldc + 8], _C5_1);
    C[5 * ldc + 10] = _C5_2;

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm_storeu_pd(&C[6 * ldc + 8], _C6_1);
    C[6 * ldc + 10] = _C6_2;

    _mm512_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm_storeu_pd(&C[7 * ldc + 8], _C7_1);
    C[7 * ldc + 10] = _C7_2;

    _mm512_storeu_pd(&C[8 * ldc + 0], _C8_0);
    _mm_storeu_pd(&C[8 * ldc + 8], _C8_1);
    C[8 * ldc + 10] = _C8_2;

    _mm512_storeu_pd(&C[9 * ldc + 0], _C9_0);
    _mm_storeu_pd(&C[9 * ldc + 8], _C9_1);
    C[9 * ldc + 10] = _C9_2;

    _mm512_storeu_pd(&C[10 * ldc + 0], _C10_0);
    _mm_storeu_pd(&C[10 * ldc + 8], _C10_1);
    C[10 * ldc + 10] = _C10_2;

    _mm512_storeu_pd(&C[11 * ldc + 0], _C11_0);
    _mm_storeu_pd(&C[11 * ldc + 8], _C11_1);
    C[11 * ldc + 10] = _C11_2;
}
void micro_kernel12x12(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {//------------------------------------
     int i;
    register __m512d _A0;
    register __m256d _A1;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C0_1 = _mm256_loadu_pd(&C[0 * ldc + 8]);

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m256d _C1_1 = _mm256_loadu_pd(&C[1 * ldc + 8]);

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m256d _C2_1 = _mm256_loadu_pd(&C[2 * ldc + 8]);

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m256d _C3_1 = _mm256_loadu_pd(&C[3 * ldc + 8]);

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m256d _C4_1 = _mm256_loadu_pd(&C[4 * ldc + 8]);

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register __m256d _C5_1 = _mm256_loadu_pd(&C[5 * ldc + 8]);

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);
    register __m256d _C6_1 = _mm256_loadu_pd(&C[6 * ldc + 8]);

    register __m512d _C7_0 = _mm512_loadu_pd(&C[7 * ldc + 0]);
    register __m256d _C7_1 = _mm256_loadu_pd(&C[7 * ldc + 8]);

    register __m512d _C8_0 = _mm512_loadu_pd(&C[8 * ldc + 0]);
    register __m256d _C8_1 = _mm256_loadu_pd(&C[8 * ldc + 8]);

    register __m512d _C9_0 = _mm512_loadu_pd(&C[9 * ldc + 0]);
    register __m256d _C9_1 = _mm256_loadu_pd(&C[9 * ldc + 8]);

    register __m512d _C10_0 = _mm512_loadu_pd(&C[10 * ldc + 0]);
    register __m256d _C10_1 = _mm256_loadu_pd(&C[10 * ldc + 8]);

    register __m512d _C11_0 = _mm512_loadu_pd(&C[11 * ldc + 0]);
    register __m256d _C11_1 = _mm256_loadu_pd(&C[11 * ldc + 8]);
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm256_loadu_pd(&A[8]);

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A1, _C0_1);

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A1, _C1_1);

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A1, _C2_1);

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A1, _C3_1);

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A1, _C4_1);

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A1, _C5_1);

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[6 * ldb]), _A1, _C6_1);

        _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[7 * ldb]), _A1, _C7_1);

        _C8_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[8 * ldb]), _A1, _C8_1);

        _C9_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[9 * ldb]), _A0, _C9_0);
        _C9_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[9 * ldb]), _A1, _C9_1);

        _C10_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[10 * ldb]), _A0, _C10_0);
        _C10_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[10 * ldb]), _A1, _C10_1);

        _C11_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[11 * ldb]), _A0, _C11_0);
        _C11_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[11 * ldb]), _A1, _C11_1);
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[0 * ldc + 8], _C0_1);

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm256_storeu_pd(&C[1 * ldc + 8], _C1_1);

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm256_storeu_pd(&C[2 * ldc + 8], _C2_1);

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm256_storeu_pd(&C[3 * ldc + 8], _C3_1);

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm256_storeu_pd(&C[4 * ldc + 8], _C4_1);

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm256_storeu_pd(&C[5 * ldc + 8], _C5_1);

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm256_storeu_pd(&C[6 * ldc + 8], _C6_1);

    _mm512_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm256_storeu_pd(&C[7 * ldc + 8], _C7_1);

    _mm512_storeu_pd(&C[8 * ldc + 0], _C8_0);
    _mm256_storeu_pd(&C[8 * ldc + 8], _C8_1);

    _mm512_storeu_pd(&C[9 * ldc + 0], _C9_0);
    _mm256_storeu_pd(&C[9 * ldc + 8], _C9_1);

    _mm512_storeu_pd(&C[10 * ldc + 0], _C10_0);
    _mm256_storeu_pd(&C[10 * ldc + 8], _C10_1);

    _mm512_storeu_pd(&C[11 * ldc + 0], _C11_0);
    _mm256_storeu_pd(&C[11 * ldc + 8], _C11_1);
}
void micro_kernel13x12(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m256d _A1;
    register double _A2;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C0_1 = _mm256_loadu_pd(&C[0 * ldc + 8]);
    register double _C0_2 = C[0 * ldc + 12];

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m256d _C1_1 = _mm256_loadu_pd(&C[1 * ldc + 8]);
    register double _C1_2 = C[1 * ldc + 12];

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m256d _C2_1 = _mm256_loadu_pd(&C[2 * ldc + 8]);
    register double _C2_2 = C[2 * ldc + 12];

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m256d _C3_1 = _mm256_loadu_pd(&C[3 * ldc + 8]);
    register double _C3_2 = C[3 * ldc + 12];

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m256d _C4_1 = _mm256_loadu_pd(&C[4 * ldc + 8]);
    register double _C4_2 = C[4 * ldc + 12];

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register __m256d _C5_1 = _mm256_loadu_pd(&C[5 * ldc + 8]);
    register double _C5_2 = C[5 * ldc + 12];

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);
    register __m256d _C6_1 = _mm256_loadu_pd(&C[6 * ldc + 8]);
    register double _C6_2 = C[6 * ldc + 12];

    register __m512d _C7_0 = _mm512_loadu_pd(&C[7 * ldc + 0]);
    register __m256d _C7_1 = _mm256_loadu_pd(&C[7 * ldc + 8]);
    register double _C7_2 = C[7 * ldc + 12];

    register __m512d _C8_0 = _mm512_loadu_pd(&C[8 * ldc + 0]);
    register __m256d _C8_1 = _mm256_loadu_pd(&C[8 * ldc + 8]);
    register double _C8_2 = C[8 * ldc + 12];

    register __m512d _C9_0 = _mm512_loadu_pd(&C[9 * ldc + 0]);
    register __m256d _C9_1 = _mm256_loadu_pd(&C[9 * ldc + 8]);
    register double _C9_2 = C[9 * ldc + 12];

    register __m512d _C10_0 = _mm512_loadu_pd(&C[10 * ldc + 0]);
    register __m256d _C10_1 = _mm256_loadu_pd(&C[10 * ldc + 8]);
    register double _C10_2 = C[10 * ldc + 12];

    register __m512d _C11_0 = _mm512_loadu_pd(&C[11 * ldc + 0]);
    register __m256d _C11_1 = _mm256_loadu_pd(&C[11 * ldc + 8]);
    register double _C11_2 = C[11 * ldc + 12];
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm256_loadu_pd(&A[8]);
        _A2 = A[12];

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 += B[0 * ldb] * _A2;

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C1_2 += B[1 * ldb] * _A2;

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C2_2 += B[2 * ldb] * _A2;

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A1, _C3_1);
        _C3_2 += B[3 * ldb] * _A2;

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A1, _C4_1);
        _C4_2 += B[4 * ldb] * _A2;

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A1, _C5_1);
        _C5_2 += B[5 * ldb] * _A2;

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[6 * ldb]), _A1, _C6_1);
        _C6_2 += B[6 * ldb] * _A2;

        _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[7 * ldb]), _A1, _C7_1);
        _C7_2 += B[7 * ldb] * _A2;

        _C8_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[8 * ldb]), _A1, _C8_1);
        _C8_2 += B[8 * ldb] * _A2;

        _C9_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[9 * ldb]), _A0, _C9_0);
        _C9_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[9 * ldb]), _A1, _C9_1);
        _C9_2 += B[9 * ldb] * _A2;

        _C10_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[10 * ldb]), _A0, _C10_0);
        _C10_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[10 * ldb]), _A1, _C10_1);
        _C10_2 += B[10 * ldb] * _A2;

        _C11_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[11 * ldb]), _A0, _C11_0);
        _C11_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[11 * ldb]), _A1, _C11_1);
        _C11_2 += B[11 * ldb] * _A2;
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[0 * ldc + 8], _C0_1);
    C[0 * ldc + 12] = _C0_2;

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm256_storeu_pd(&C[1 * ldc + 8], _C1_1);
    C[1 * ldc + 12] = _C1_2;

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm256_storeu_pd(&C[2 * ldc + 8], _C2_1);
    C[2 * ldc + 12] = _C2_2;

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm256_storeu_pd(&C[3 * ldc + 8], _C3_1);
    C[3 * ldc + 12] = _C3_2;

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm256_storeu_pd(&C[4 * ldc + 8], _C4_1);
    C[4 * ldc + 12] = _C4_2;

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm256_storeu_pd(&C[5 * ldc + 8], _C5_1);
    C[5 * ldc + 12] = _C5_2;

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm256_storeu_pd(&C[6 * ldc + 8], _C6_1);
    C[6 * ldc + 12] = _C6_2;

    _mm512_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm256_storeu_pd(&C[7 * ldc + 8], _C7_1);
    C[7 * ldc + 12] = _C7_2;

    _mm512_storeu_pd(&C[8 * ldc + 0], _C8_0);
    _mm256_storeu_pd(&C[8 * ldc + 8], _C8_1);
    C[8 * ldc + 12] = _C8_2;

    _mm512_storeu_pd(&C[9 * ldc + 0], _C9_0);
    _mm256_storeu_pd(&C[9 * ldc + 8], _C9_1);
    C[9 * ldc + 12] = _C9_2;

    _mm512_storeu_pd(&C[10 * ldc + 0], _C10_0);
    _mm256_storeu_pd(&C[10 * ldc + 8], _C10_1);
    C[10 * ldc + 12] = _C10_2;

    _mm512_storeu_pd(&C[11 * ldc + 0], _C11_0);
    _mm256_storeu_pd(&C[11 * ldc + 8], _C11_1);
    C[11 * ldc + 12] = _C11_2;
}
void micro_kernel14x12(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m256d _A1;
    register __m128d _A2;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C0_1 = _mm256_loadu_pd(&C[0 * ldc + 8]);
    register __m128d _C0_2 = _mm_loadu_pd(&C[0 * ldc + 12]);

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m256d _C1_1 = _mm256_loadu_pd(&C[1 * ldc + 8]);
    register __m128d _C1_2 = _mm_loadu_pd(&C[1 * ldc + 12]);

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m256d _C2_1 = _mm256_loadu_pd(&C[2 * ldc + 8]);
    register __m128d _C2_2 = _mm_loadu_pd(&C[2 * ldc + 12]);

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m256d _C3_1 = _mm256_loadu_pd(&C[3 * ldc + 8]);
    register __m128d _C3_2 = _mm_loadu_pd(&C[3 * ldc + 12]);

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m256d _C4_1 = _mm256_loadu_pd(&C[4 * ldc + 8]);
    register __m128d _C4_2 = _mm_loadu_pd(&C[4 * ldc + 12]);

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register __m256d _C5_1 = _mm256_loadu_pd(&C[5 * ldc + 8]);
    register __m128d _C5_2 = _mm_loadu_pd(&C[5 * ldc + 12]);

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);
    register __m256d _C6_1 = _mm256_loadu_pd(&C[6 * ldc + 8]);
    register __m128d _C6_2 = _mm_loadu_pd(&C[6 * ldc + 12]);

    register __m512d _C7_0 = _mm512_loadu_pd(&C[7 * ldc + 0]);
    register __m256d _C7_1 = _mm256_loadu_pd(&C[7 * ldc + 8]);
    register __m128d _C7_2 = _mm_loadu_pd(&C[7 * ldc + 12]);

    register __m512d _C8_0 = _mm512_loadu_pd(&C[8 * ldc + 0]);
    register __m256d _C8_1 = _mm256_loadu_pd(&C[8 * ldc + 8]);
    register __m128d _C8_2 = _mm_loadu_pd(&C[8 * ldc + 12]);

    register __m512d _C9_0 = _mm512_loadu_pd(&C[9 * ldc + 0]);
    register __m256d _C9_1 = _mm256_loadu_pd(&C[9 * ldc + 8]);
    register __m128d _C9_2 = _mm_loadu_pd(&C[9 * ldc + 12]);

    register __m512d _C10_0 = _mm512_loadu_pd(&C[10 * ldc + 0]);
    register __m256d _C10_1 = _mm256_loadu_pd(&C[10 * ldc + 8]);
    register __m128d _C10_2 = _mm_loadu_pd(&C[10 * ldc + 12]);

    register __m512d _C11_0 = _mm512_loadu_pd(&C[11 * ldc + 0]);
    register __m256d _C11_1 = _mm256_loadu_pd(&C[11 * ldc + 8]);
    register __m128d _C11_2 = _mm_loadu_pd(&C[11 * ldc + 12]);

    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm256_loadu_pd(&A[8]);
        _A2 = _mm_loadu_pd(&A[12]);

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A2, _C0_2);

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C1_2 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A2, _C1_2);

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C2_2 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A2, _C2_2);

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A1, _C3_1);
        _C3_2 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A2, _C3_2);

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A1, _C4_1);
        _C4_2 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A2, _C4_2);

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A1, _C5_1);
        _C5_2 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A2, _C5_2);

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[6 * ldb]), _A1, _C6_1);
        _C6_2 = _mm_fnmadd_pd(_mm_set1_pd(B[6 * ldb]), _A2, _C6_2);

        _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[7 * ldb]), _A1, _C7_1);
        _C7_2 = _mm_fnmadd_pd(_mm_set1_pd(B[7 * ldb]), _A2, _C7_2);

        _C8_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[8 * ldb]), _A1, _C8_1);
        _C8_2 = _mm_fnmadd_pd(_mm_set1_pd(B[8 * ldb]), _A2, _C8_2);

        _C9_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[9 * ldb]), _A0, _C9_0);
        _C9_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[9 * ldb]), _A1, _C9_1);
        _C9_2 = _mm_fnmadd_pd(_mm_set1_pd(B[9 * ldb]), _A2, _C9_2);

        _C10_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[10 * ldb]), _A0, _C10_0);
        _C10_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[10 * ldb]), _A1, _C10_1);
        _C10_2 = _mm_fnmadd_pd(_mm_set1_pd(B[10 * ldb]), _A2, _C10_2);
 

        _C11_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[11 * ldb]), _A0, _C11_0);
        _C11_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[11 * ldb]), _A1, _C11_1);
        _C11_2 = _mm_fnmadd_pd(_mm_set1_pd(B[11 * ldb]), _A2, _C11_2);

        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[0 * ldc + 8], _C0_1);
    _mm_storeu_pd(&C[0 * ldc + 12], _C0_2);

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm256_storeu_pd(&C[1 * ldc + 8], _C1_1);
    _mm_storeu_pd(&C[1 * ldc + 12], _C1_2);

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm256_storeu_pd(&C[2 * ldc + 8], _C2_1);
    _mm_storeu_pd(&C[2 * ldc + 12], _C2_2);

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm256_storeu_pd(&C[3 * ldc + 8], _C3_1);
    _mm_storeu_pd(&C[3 * ldc + 12], _C3_2);

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm256_storeu_pd(&C[4 * ldc + 8], _C4_1);
    _mm_storeu_pd(&C[4 * ldc + 12], _C4_2);

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm256_storeu_pd(&C[5 * ldc + 8], _C5_1);
    _mm_storeu_pd(&C[5 * ldc + 12], _C5_2);

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm256_storeu_pd(&C[6 * ldc + 8], _C6_1);
    _mm_storeu_pd(&C[6 * ldc + 12], _C6_2);

    _mm512_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm256_storeu_pd(&C[7 * ldc + 8], _C7_1);
    _mm_storeu_pd(&C[7 * ldc + 12], _C7_2);

    _mm512_storeu_pd(&C[8 * ldc + 0], _C8_0);
    _mm256_storeu_pd(&C[8 * ldc + 8], _C8_1);
    _mm_storeu_pd(&C[8 * ldc + 12], _C8_2);

    _mm512_storeu_pd(&C[9 * ldc + 0], _C9_0);
    _mm256_storeu_pd(&C[9 * ldc + 8], _C9_1);
    _mm_storeu_pd(&C[9 * ldc + 12], _C9_2);

    _mm512_storeu_pd(&C[10 * ldc + 0], _C10_0);
    _mm256_storeu_pd(&C[10 * ldc + 8], _C10_1);
    _mm_storeu_pd(&C[10 * ldc + 12], _C10_2);

    _mm512_storeu_pd(&C[11 * ldc + 0], _C11_0);
    _mm256_storeu_pd(&C[11 * ldc + 8], _C11_1);
    _mm_storeu_pd(&C[11 * ldc + 12], _C11_2);
}
void micro_kernel15x12(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m256d _A1;
    register __m128d _A2;
    register double _A3;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C0_1 = _mm256_loadu_pd(&C[0 * ldc + 8]);
    register __m128d _C0_2 = _mm_loadu_pd(&C[0 * ldc + 12]);
    register double _C0_3 = C[0 * ldc + 14];

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m256d _C1_1 = _mm256_loadu_pd(&C[1 * ldc + 8]);
    register __m128d _C1_2 = _mm_loadu_pd(&C[1 * ldc + 12]);
    register double _C1_3 = C[1 * ldc + 14];

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m256d _C2_1 = _mm256_loadu_pd(&C[2 * ldc + 8]);
    register __m128d _C2_2 = _mm_loadu_pd(&C[2 * ldc + 12]);
    register double _C2_3 = C[2 * ldc + 14];

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m256d _C3_1 = _mm256_loadu_pd(&C[3 * ldc + 8]);
    register __m128d _C3_2 = _mm_loadu_pd(&C[3 * ldc + 12]);
    register double _C3_3 = C[3 * ldc + 14];

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m256d _C4_1 = _mm256_loadu_pd(&C[4 * ldc + 8]);
    register __m128d _C4_2 = _mm_loadu_pd(&C[4 * ldc + 12]);
    register double _C4_3 = C[4 * ldc + 14];

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register __m256d _C5_1 = _mm256_loadu_pd(&C[5 * ldc + 8]);
    register __m128d _C5_2 = _mm_loadu_pd(&C[5 * ldc + 12]);
    register double _C5_3 = C[5 * ldc + 14];

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);
    register __m256d _C6_1 = _mm256_loadu_pd(&C[6 * ldc + 8]);
    register __m128d _C6_2 = _mm_loadu_pd(&C[6 * ldc + 12]);
    register double _C6_3 = C[6 * ldc + 14];

    register __m512d _C7_0 = _mm512_loadu_pd(&C[7 * ldc + 0]);
    register __m256d _C7_1 = _mm256_loadu_pd(&C[7 * ldc + 8]);
    register __m128d _C7_2 = _mm_loadu_pd(&C[7 * ldc + 12]);
    register double _C7_3 = C[7 * ldc + 14];

    register __m512d _C8_0 = _mm512_loadu_pd(&C[8 * ldc + 0]);
    register __m256d _C8_1 = _mm256_loadu_pd(&C[8 * ldc + 8]);
    register __m128d _C8_2 = _mm_loadu_pd(&C[8 * ldc + 12]);
    register double _C8_3 = C[8 * ldc + 14];

    register __m512d _C9_0 = _mm512_loadu_pd(&C[9 * ldc + 0]);
    register __m256d _C9_1 = _mm256_loadu_pd(&C[9 * ldc + 8]);
    register __m128d _C9_2 = _mm_loadu_pd(&C[9 * ldc + 12]);
    register double _C9_3 = C[9 * ldc + 14];

    register __m512d _C10_0 = _mm512_loadu_pd(&C[10 * ldc + 0]);
    register __m256d _C10_1 = _mm256_loadu_pd(&C[10 * ldc + 8]);
    register __m128d _C10_2 = _mm_loadu_pd(&C[10 * ldc + 12]);
    register double _C10_3 = C[10 * ldc + 14];

    register __m512d _C11_0 = _mm512_loadu_pd(&C[11 * ldc + 0]);
    register __m256d _C11_1 = _mm256_loadu_pd(&C[11 * ldc + 8]);
    register __m128d _C11_2 = _mm_loadu_pd(&C[11 * ldc + 12]);
    register double _C11_3 = C[11 * ldc + 14];
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm256_loadu_pd(&A[8]);
        _A2 = _mm_loadu_pd(&A[12]);
        _A3 = A[14];

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A2, _C0_2);
        _C0_3 += B[0 * ldb] * _A3;

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C1_2 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A2, _C1_2);
        _C1_3 += B[1 * ldb] * _A3;

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C2_2 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A2, _C2_2);
        _C2_3 += B[2 * ldb] * _A3;

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A1, _C3_1);
        _C3_2 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A2, _C3_2);
        _C3_3 += B[3 * ldb] * _A3;

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A1, _C4_1);
        _C4_2 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A2, _C4_2);
        _C4_3 += B[4 * ldb] * _A3;

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A1, _C5_1);
        _C5_2 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A2, _C5_2);
        _C5_3 += B[5 * ldb] * _A3;

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[6 * ldb]), _A1, _C6_1);
        _C6_2 = _mm_fnmadd_pd(_mm_set1_pd(B[6 * ldb]), _A2, _C6_2);
        _C6_3 += B[6 * ldb] * _A3;

        _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[7 * ldb]), _A1, _C7_1);
        _C7_2 = _mm_fnmadd_pd(_mm_set1_pd(B[7 * ldb]), _A2, _C7_2);
        _C7_3 += B[7 * ldb] * _A3;

        _C8_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[8 * ldb]), _A1, _C8_1);
        _C8_2 = _mm_fnmadd_pd(_mm_set1_pd(B[8 * ldb]), _A2, _C8_2);
        _C8_3 += B[8 * ldb] * _A3;

        _C9_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[9 * ldb]), _A0, _C9_0);
        _C9_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[9 * ldb]), _A1, _C9_1);
        _C9_2 = _mm_fnmadd_pd(_mm_set1_pd(B[9 * ldb]), _A2, _C9_2);
        _C9_3 += B[9 * ldb] * _A3;

        _C10_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[10 * ldb]), _A0, _C10_0);
        _C10_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[10 * ldb]), _A1, _C10_1);
        _C10_2 = _mm_fnmadd_pd(_mm_set1_pd(B[10 * ldb]), _A2, _C10_2);
        _C10_3 += B[10 * ldb] * _A3;
        
        _C11_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[11 * ldb]), _A0, _C11_0);
        _C11_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[11 * ldb]), _A1, _C11_1);
        _C11_2 = _mm_fnmadd_pd(_mm_set1_pd(B[11 * ldb]), _A2, _C11_2);
        _C11_3 += B[11 * ldb] * _A3;
        

        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[0 * ldc + 8], _C0_1);
    _mm_storeu_pd(&C[0 * ldc + 12], _C0_2);
    C[0 * ldc + 14] = _C0_3;

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm256_storeu_pd(&C[1 * ldc + 8], _C1_1);
    _mm_storeu_pd(&C[1 * ldc + 12], _C1_2);
    C[1 * ldc + 14] = _C1_3;

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm256_storeu_pd(&C[2 * ldc + 8], _C2_1);
    _mm_storeu_pd(&C[2 * ldc + 12], _C2_2);
    C[2 * ldc + 14] = _C2_3;

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm256_storeu_pd(&C[3 * ldc + 8], _C3_1);
    _mm_storeu_pd(&C[3 * ldc + 12], _C3_2);
    C[3 * ldc + 14] = _C3_3;

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm256_storeu_pd(&C[4 * ldc + 8], _C4_1);
    _mm_storeu_pd(&C[4 * ldc + 12], _C4_2);
    C[4 * ldc + 14] = _C4_3;

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm256_storeu_pd(&C[5 * ldc + 8], _C5_1);
    _mm_storeu_pd(&C[5 * ldc + 12], _C5_2);
    C[5 * ldc + 14] = _C5_3;

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm256_storeu_pd(&C[6 * ldc + 8], _C6_1);
    _mm_storeu_pd(&C[6 * ldc + 12], _C6_2);
    C[6 * ldc + 14] = _C6_3;

    _mm512_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm256_storeu_pd(&C[7 * ldc + 8], _C7_1);
    _mm_storeu_pd(&C[7 * ldc + 12], _C7_2);
    C[7 * ldc + 14] = _C7_3;

    _mm512_storeu_pd(&C[8 * ldc + 0], _C8_0);
    _mm256_storeu_pd(&C[8 * ldc + 8], _C8_1);
    _mm_storeu_pd(&C[8 * ldc + 12], _C8_2);
    C[8 * ldc + 14] = _C8_3;

    _mm512_storeu_pd(&C[9 * ldc + 0], _C9_0);
    _mm256_storeu_pd(&C[9 * ldc + 8], _C9_1);
    _mm_storeu_pd(&C[9 * ldc + 12], _C9_2);
    C[9 * ldc + 14] = _C9_3;

    _mm512_storeu_pd(&C[10 * ldc + 0], _C10_0);
    _mm256_storeu_pd(&C[10 * ldc + 8], _C10_1);
    _mm_storeu_pd(&C[10 * ldc + 12], _C10_2);
    C[10 * ldc + 14] = _C10_3;

    _mm512_storeu_pd(&C[11 * ldc + 0], _C11_0);
    _mm256_storeu_pd(&C[11 * ldc + 8], _C11_1);
    _mm_storeu_pd(&C[11 * ldc + 12], _C11_2);
    C[11 * ldc + 14] = _C11_3;
}
void micro_kernel16x12(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m512d _A1;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m512d _C0_1 = _mm512_loadu_pd(&C[0 * ldc + 8]);

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m512d _C1_1 = _mm512_loadu_pd(&C[1 * ldc + 8]);

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m512d _C2_1 = _mm512_loadu_pd(&C[2 * ldc + 8]);

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m512d _C3_1 = _mm512_loadu_pd(&C[3 * ldc + 8]);

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m512d _C4_1 = _mm512_loadu_pd(&C[4 * ldc + 8]);

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register __m512d _C5_1 = _mm512_loadu_pd(&C[5 * ldc + 8]);

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);
    register __m512d _C6_1 = _mm512_loadu_pd(&C[6 * ldc + 8]);

    register __m512d _C7_0 = _mm512_loadu_pd(&C[7 * ldc + 0]);
    register __m512d _C7_1 = _mm512_loadu_pd(&C[7 * ldc + 8]);

    register __m512d _C8_0 = _mm512_loadu_pd(&C[8 * ldc + 0]);
    register __m512d _C8_1 = _mm512_loadu_pd(&C[8 * ldc + 8]);

    register __m512d _C9_0 = _mm512_loadu_pd(&C[9 * ldc + 0]);
    register __m512d _C9_1 = _mm512_loadu_pd(&C[9 * ldc + 8]);

    register __m512d _C10_0 = _mm512_loadu_pd(&C[10 * ldc + 0]);
    register __m512d _C10_1 = _mm512_loadu_pd(&C[10 * ldc + 8]);

    register __m512d _C11_0 = _mm512_loadu_pd(&C[11 * ldc + 0]);
    register __m512d _C11_1 = _mm512_loadu_pd(&C[11 * ldc + 8]);
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm512_loadu_pd(&A[8]);

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A1, _C0_1);

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A1, _C1_1);

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A1, _C2_1);

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A1, _C3_1);

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A1, _C4_1);

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A1, _C5_1);

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A1, _C6_1);

        _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A1, _C7_1);

        _C8_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8 * ldb]), _A1, _C8_1);

        _C9_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[9 * ldb]), _A0, _C9_0);
        _C9_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[9 * ldb]), _A1, _C9_1);

        _C10_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[10 * ldb]), _A0, _C10_0);
        _C10_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[10 * ldb]), _A1, _C10_1);

        _C11_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[11 * ldb]), _A0, _C11_0);
        _C11_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[11 * ldb]), _A1, _C11_1);
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm512_storeu_pd(&C[0 * ldc + 8], _C0_1);

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm512_storeu_pd(&C[1 * ldc + 8], _C1_1);

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm512_storeu_pd(&C[2 * ldc + 8], _C2_1);

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm512_storeu_pd(&C[3 * ldc + 8], _C3_1);

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm512_storeu_pd(&C[4 * ldc + 8], _C4_1);

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm512_storeu_pd(&C[5 * ldc + 8], _C5_1);

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm512_storeu_pd(&C[6 * ldc + 8], _C6_1);

    _mm512_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm512_storeu_pd(&C[7 * ldc + 8], _C7_1);

    _mm512_storeu_pd(&C[8 * ldc + 0], _C8_0);
    _mm512_storeu_pd(&C[8 * ldc + 8], _C8_1);

    _mm512_storeu_pd(&C[9 * ldc + 0], _C9_0);
    _mm512_storeu_pd(&C[9 * ldc + 8], _C9_1);

    _mm512_storeu_pd(&C[10 * ldc + 0], _C10_0);
    _mm512_storeu_pd(&C[10 * ldc + 8], _C10_1);

    _mm512_storeu_pd(&C[11 * ldc + 0], _C11_0);
    _mm512_storeu_pd(&C[11 * ldc + 8], _C11_1);
}

//=====================------------------------------------------------------------------------------------
void micro_kernel1x13(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register double _A0;
    register double _C0_0 = C[0 * ldc + 0];
    register double _C1_0 = C[1 * ldc + 0];
    register double _C2_0 = C[2 * ldc + 0];
    register double _C3_0 = C[3 * ldc + 0];
    register double _C4_0 = C[4 * ldc + 0];
    register double _C5_0 = C[5 * ldc + 0];
    register double _C6_0 = C[6 * ldc + 0];
    register double _C7_0 = C[7 * ldc + 0];
    register double _C8_0 = C[8 * ldc + 0];
    register double _C9_0 = C[9 * ldc + 0];
    register double _C10_0 = C[10 * ldc + 0];
    register double _C11_0 = C[11 * ldc + 0];
    register double _C12_0 = C[12 * ldc + 0];

    for(i = 0; i < k; ++i) {
        _A0 = A[0];
        _C0_0 += B[0 * ldb] * _A0;
        _C1_0 += B[1 * ldb] * _A0;
        _C2_0 += B[2 * ldb] * _A0;
        _C3_0 += B[3 * ldb] * _A0;
        _C4_0 += B[4 * ldb] * _A0;
        _C5_0 += B[5 * ldb] * _A0;
        _C6_0 += B[6 * ldb] * _A0;
        _C7_0 += B[7 * ldb] * _A0;
        _C8_0 += B[8 * ldb] * _A0;
        _C9_0 += B[9 * ldb] * _A0;
        _C10_0 += B[10 * ldb] * _A0;
        _C11_0 += B[11 * ldb] * _A0;
        _C12_0 += B[12 * ldb] * _A0;

        A += lda;
        B += 1;
    }

    C[0 * ldc + 0] = _C0_0;
    C[1 * ldc + 0] = _C1_0;
    C[2 * ldc + 0] = _C2_0;
    C[3 * ldc + 0] = _C3_0;
    C[4 * ldc + 0] = _C4_0;
    C[5 * ldc + 0] = _C5_0;
    C[6 * ldc + 0] = _C6_0;
    C[7 * ldc + 0] = _C7_0;
    C[8 * ldc + 0] = _C8_0;
    C[9 * ldc + 0] = _C9_0;
    C[10 * ldc + 0] = _C10_0;
    C[11 * ldc + 0] = _C11_0;
    C[12 * ldc + 0] = _C12_0;
}
void micro_kernel2x13(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m128d _A0;
    register __m128d _C0_0 = _mm_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C1_0 = _mm_loadu_pd(&C[1 * ldc + 0]);
    register __m128d _C2_0 = _mm_loadu_pd(&C[2 * ldc + 0]);
    register __m128d _C3_0 = _mm_loadu_pd(&C[3 * ldc + 0]);
    register __m128d _C4_0 = _mm_loadu_pd(&C[4 * ldc + 0]);
    register __m128d _C5_0 = _mm_loadu_pd(&C[5 * ldc + 0]);
    register __m128d _C6_0 = _mm_loadu_pd(&C[6 * ldc + 0]);
    register __m128d _C7_0 = _mm_loadu_pd(&C[7 * ldc + 0]);
    register __m128d _C8_0 = _mm_loadu_pd(&C[8 * ldc + 0]);
    register __m128d _C9_0 = _mm_loadu_pd(&C[9 * ldc + 0]);
    register __m128d _C10_0 = _mm_loadu_pd(&C[10 * ldc + 0]);
    register __m128d _C11_0 = _mm_loadu_pd(&C[11 * ldc + 0]);
    register __m128d _C12_0 = _mm_loadu_pd(&C[12 * ldc + 0]);

    for(i = 0; i < k; ++i) {
        _A0 = _mm_loadu_pd(&A[0]);
        _C0_0 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C1_0 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C2_0 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C3_0 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C4_0 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C5_0 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C6_0 = _mm_fnmadd_pd(_mm_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C7_0 = _mm_fnmadd_pd(_mm_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C8_0 = _mm_fnmadd_pd(_mm_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C9_0 = _mm_fnmadd_pd(_mm_set1_pd(B[9 * ldb]), _A0, _C9_0);
        _C10_0 = _mm_fnmadd_pd(_mm_set1_pd(B[10 * ldb]), _A0, _C10_0);
        _C11_0 = _mm_fnmadd_pd(_mm_set1_pd(B[11 * ldb]), _A0, _C11_0);
        _C12_0 = _mm_fnmadd_pd(_mm_set1_pd(B[12 * ldb]), _A0, _C12_0);

        A += lda;
        B += 1;
    }

    _mm_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm_storeu_pd(&C[8 * ldc + 0], _C8_0);
    _mm_storeu_pd(&C[9 * ldc + 0], _C9_0);
    _mm_storeu_pd(&C[10 * ldc + 0], _C10_0);
    _mm_storeu_pd(&C[11 * ldc + 0], _C11_0);
    _mm_storeu_pd(&C[12 * ldc + 0], _C12_0);
}
void micro_kernel3x13(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m128d _A0;
    register double _A1;
    register __m128d _C0_0 = _mm_loadu_pd(&C[0 * ldc + 0]);
    register double _C0_1 = C[0 * ldc + 2];
    register __m128d _C1_0 = _mm_loadu_pd(&C[1 * ldc + 0]);
    register double _C1_1 = C[1 * ldc + 2];
    register __m128d _C2_0 = _mm_loadu_pd(&C[2 * ldc + 0]);
    register double _C2_1 = C[2 * ldc + 2];
    register __m128d _C3_0 = _mm_loadu_pd(&C[3 * ldc + 0]);
    register double _C3_1 = C[3 * ldc + 2];
    register __m128d _C4_0 = _mm_loadu_pd(&C[4 * ldc + 0]);
    register double _C4_1 = C[4 * ldc + 2];
    register __m128d _C5_0 = _mm_loadu_pd(&C[5 * ldc + 0]);
    register double _C5_1 = C[5 * ldc + 2];

    register __m128d _C6_0 = _mm_loadu_pd(&C[6 * ldc + 0]);
    register double _C6_1 = C[6 * ldc + 2];

    register __m128d _C7_0 = _mm_loadu_pd(&C[7 * ldc + 0]);
    register double _C7_1 = C[7 * ldc + 2];

    register __m128d _C8_0 = _mm_loadu_pd(&C[8 * ldc + 0]);
    register double _C8_1 = C[8 * ldc + 2];

    register __m128d _C9_0 = _mm_loadu_pd(&C[9 * ldc + 0]);
    register double _C9_1 = C[9 * ldc + 2];

    register __m128d _C10_0 = _mm_loadu_pd(&C[10 * ldc + 0]);
    register double _C10_1 = C[10 * ldc + 2];

    register __m128d _C11_0 = _mm_loadu_pd(&C[11 * ldc + 0]);
    register double _C11_1 = C[11 * ldc + 2];

    register __m128d _C12_0 = _mm_loadu_pd(&C[12 * ldc + 0]);
    register double _C12_1 = C[12 * ldc + 2];

    for(i = 0; i < k; ++i) {
        _A0 = _mm_loadu_pd(&A[0]);
        _A1 = A[2];
        _C0_0 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 += B[0 * ldb] * _A1;
        _C1_0 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 += B[1 * ldb] * _A1;
        _C2_0 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 += B[2 * ldb] * _A1;
        _C3_0 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 += B[3 * ldb] * _A1;
        _C4_0 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 += B[4 * ldb] * _A1;
        _C5_0 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 += B[5 * ldb] * _A1;

        _C6_0 = _mm_fnmadd_pd(_mm_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 += B[6 * ldb] * _A1;

        _C7_0 = _mm_fnmadd_pd(_mm_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 += B[7 * ldb] * _A1;

        _C8_0 = _mm_fnmadd_pd(_mm_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 += B[8 * ldb] * _A1;

        _C9_0 = _mm_fnmadd_pd(_mm_set1_pd(B[9 * ldb]), _A0, _C9_0);
        _C9_1 += B[9 * ldb] * _A1;

        _C10_0 = _mm_fnmadd_pd(_mm_set1_pd(B[10 * ldb]), _A0, _C10_0);
        _C10_1 += B[10 * ldb] * _A1;

        _C11_0 = _mm_fnmadd_pd(_mm_set1_pd(B[11 * ldb]), _A0, _C11_0);
        _C11_1 += B[11 * ldb] * _A1;

        _C12_0 = _mm_fnmadd_pd(_mm_set1_pd(B[12 * ldb]), _A0, _C12_0);
        _C12_1 += B[12 * ldb] * _A1;

        A += lda;
        B += 1;
    }

    _mm_storeu_pd(&C[0 * ldc + 0], _C0_0);
    C[0 * ldc + 2] = _C0_1;
    _mm_storeu_pd(&C[1 * ldc + 0], _C1_0);
    C[1 * ldc + 2] = _C1_1;
    _mm_storeu_pd(&C[2 * ldc + 0], _C2_0);
    C[2 * ldc + 2] = _C2_1;
    _mm_storeu_pd(&C[3 * ldc + 0], _C3_0);
    C[3 * ldc + 2] = _C3_1;
    _mm_storeu_pd(&C[4 * ldc + 0], _C4_0);
    C[4 * ldc + 2] = _C4_1;
    _mm_storeu_pd(&C[5 * ldc + 0], _C5_0);
    C[5 * ldc + 2] = _C5_1;

    _mm_storeu_pd(&C[6 * ldc + 0], _C6_0);
    C[6 * ldc + 2] = _C6_1;

    _mm_storeu_pd(&C[7 * ldc + 0], _C7_0);
    C[7 * ldc + 2] = _C7_1;

    _mm_storeu_pd(&C[8 * ldc + 0], _C8_0);
    C[8 * ldc + 2] = _C8_1;

    _mm_storeu_pd(&C[9 * ldc + 0], _C9_0);
    C[9 * ldc + 2] = _C9_1;

    _mm_storeu_pd(&C[10 * ldc + 0], _C10_0);
    C[10 * ldc + 2] = _C10_1;

    _mm_storeu_pd(&C[11 * ldc + 0], _C11_0);
    C[11 * ldc + 2] = _C11_1;
    _mm_storeu_pd(&C[12 * ldc + 0], _C12_0);
    C[12 * ldc + 2] = _C12_1;
}
void micro_kernel4x13(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m256d _A0;
    register __m256d _C0_0 = _mm256_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C1_0 = _mm256_loadu_pd(&C[1 * ldc + 0]);
    register __m256d _C2_0 = _mm256_loadu_pd(&C[2 * ldc + 0]);
    register __m256d _C3_0 = _mm256_loadu_pd(&C[3 * ldc + 0]);
    register __m256d _C4_0 = _mm256_loadu_pd(&C[4 * ldc + 0]);
    register __m256d _C5_0 = _mm256_loadu_pd(&C[5 * ldc + 0]);

    register __m256d _C6_0 = _mm256_loadu_pd(&C[6 * ldc + 0]);

    register __m256d _C7_0 = _mm256_loadu_pd(&C[7 * ldc + 0]);

    register __m256d _C8_0 = _mm256_loadu_pd(&C[8 * ldc + 0]);

    register __m256d _C9_0 = _mm256_loadu_pd(&C[9 * ldc + 0]);

    register __m256d _C10_0 = _mm256_loadu_pd(&C[10 * ldc + 0]);

    register __m256d _C11_0 = _mm256_loadu_pd(&C[11 * ldc + 0]);

    register __m256d _C12_0 = _mm256_loadu_pd(&C[12 * ldc + 0]);

    for(i = 0; i < k; ++i) {
        _A0 = _mm256_loadu_pd(&A[0]);
        _C0_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C1_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C2_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C3_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C4_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C5_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A0, _C5_0);

        _C6_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[6 * ldb]), _A0, _C6_0);

        _C7_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[7 * ldb]), _A0, _C7_0);

        _C8_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[8 * ldb]), _A0, _C8_0);

        _C9_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[9 * ldb]), _A0, _C9_0);

        _C10_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[10 * ldb]), _A0, _C10_0);

        _C11_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[11 * ldb]), _A0, _C11_0);

        _C12_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[12 * ldb]), _A0, _C12_0);

        A += lda;
        B += 1;
    }

    _mm256_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm256_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm256_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm256_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm256_storeu_pd(&C[5 * ldc + 0], _C5_0);

    _mm256_storeu_pd(&C[6 * ldc + 0], _C6_0);

    _mm256_storeu_pd(&C[7 * ldc + 0], _C7_0);

    _mm256_storeu_pd(&C[8 * ldc + 0], _C8_0);

    _mm256_storeu_pd(&C[9 * ldc + 0], _C9_0);

    _mm256_storeu_pd(&C[10 * ldc + 0], _C10_0);

    _mm256_storeu_pd(&C[11 * ldc + 0], _C11_0);

    _mm256_storeu_pd(&C[12 * ldc + 0], _C12_0);
}
void micro_kernel5x13(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m256d _A0;
    register double _A1;
    register __m256d _C0_0 = _mm256_loadu_pd(&C[0 * ldc + 0]);
    register double _C0_1 = C[0 * ldc + 4];
    register __m256d _C1_0 = _mm256_loadu_pd(&C[1 * ldc + 0]);
    register double _C1_1 = C[1 * ldc + 4];
    register __m256d _C2_0 = _mm256_loadu_pd(&C[2 * ldc + 0]);
    register double _C2_1 = C[2 * ldc + 4];
    register __m256d _C3_0 = _mm256_loadu_pd(&C[3 * ldc + 0]);
    register double _C3_1 = C[3 * ldc + 4];
    register __m256d _C4_0 = _mm256_loadu_pd(&C[4 * ldc + 0]);
    register double _C4_1 = C[4 * ldc + 4];
    register __m256d _C5_0 = _mm256_loadu_pd(&C[5 * ldc + 0]);
    register double _C5_1 = C[5 * ldc + 4];

    register __m256d _C6_0 = _mm256_loadu_pd(&C[6 * ldc + 0]);
    register double _C6_1 = C[6 * ldc + 4];

    register __m256d _C7_0 = _mm256_loadu_pd(&C[7 * ldc + 0]);
    register double _C7_1 = C[7 * ldc + 4];

    register __m256d _C8_0 = _mm256_loadu_pd(&C[8 * ldc + 0]);
    register double _C8_1 = C[8 * ldc + 4];

    register __m256d _C9_0 = _mm256_loadu_pd(&C[9 * ldc + 0]);
    register double _C9_1 = C[9 * ldc + 4];

    register __m256d _C10_0 = _mm256_loadu_pd(&C[10 * ldc + 0]);
    register double _C10_1 = C[10 * ldc + 4];

    register __m256d _C11_0 = _mm256_loadu_pd(&C[11 * ldc + 0]);
    register double _C11_1 = C[11 * ldc + 4];

    register __m256d _C12_0 = _mm256_loadu_pd(&C[12 * ldc + 0]);
    register double _C12_1 = C[12 * ldc + 4];

    for(i = 0; i < k; ++i) {
        _A0 = _mm256_loadu_pd(&A[0]);
        _A1 = A[4];
        _C0_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 += B[0 * ldb] * _A1;
        _C1_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 += B[1 * ldb] * _A1;
        _C2_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 += B[2 * ldb] * _A1;
        _C3_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 += B[3 * ldb] * _A1;
        _C4_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 += B[4 * ldb] * _A1;
        _C5_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 += B[5 * ldb] * _A1;

        _C6_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 += B[6 * ldb] * _A1;

        _C7_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 += B[7 * ldb] * _A1;

        _C8_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 += B[8 * ldb] * _A1;

        _C9_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[9 * ldb]), _A0, _C9_0);
        _C9_1 += B[9 * ldb] * _A1;

        _C10_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[10 * ldb]), _A0, _C10_0);
        _C10_1 += B[10 * ldb] * _A1;

        _C11_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[11 * ldb]), _A0, _C11_0);
        _C11_1 += B[11 * ldb] * _A1;

        _C12_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[12 * ldb]), _A0, _C12_0);
        _C12_1 += B[12 * ldb] * _A1;

        A += lda;
        B += 1;
    }

    _mm256_storeu_pd(&C[0 * ldc + 0], _C0_0);
    C[0 * ldc + 4] = _C0_1;
    _mm256_storeu_pd(&C[1 * ldc + 0], _C1_0);
    C[1 * ldc + 4] = _C1_1;
    _mm256_storeu_pd(&C[2 * ldc + 0], _C2_0);
    C[2 * ldc + 4] = _C2_1;
    _mm256_storeu_pd(&C[3 * ldc + 0], _C3_0);
    C[3 * ldc + 4] = _C3_1;
    _mm256_storeu_pd(&C[4 * ldc + 0], _C4_0);
    C[4 * ldc + 4] = _C4_1;
    _mm256_storeu_pd(&C[5 * ldc + 0], _C5_0);
    C[5 * ldc + 4] = _C5_1;

    _mm256_storeu_pd(&C[6 * ldc + 0], _C6_0);
    C[6 * ldc + 4] = _C6_1;

    _mm256_storeu_pd(&C[7 * ldc + 0], _C7_0);
    C[7 * ldc + 4] = _C7_1;

    _mm256_storeu_pd(&C[8 * ldc + 0], _C8_0);
    C[8 * ldc + 4] = _C8_1;

    _mm256_storeu_pd(&C[9 * ldc + 0], _C9_0);
    C[9 * ldc + 4] = _C9_1;

    _mm256_storeu_pd(&C[10 * ldc + 0], _C10_0);
    C[10 * ldc + 4] = _C10_1;

    _mm256_storeu_pd(&C[11 * ldc + 0], _C11_0);
    C[11 * ldc + 4] = _C11_1;

    _mm256_storeu_pd(&C[12 * ldc + 0], _C12_0);
    C[12 * ldc + 4] = _C12_1;
}
void micro_kernel6x13(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m256d _A0;
    register __m128d _A1;
    register __m256d _C0_0 = _mm256_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C0_1 = _mm_loadu_pd(&C[0 * ldc + 4]);
    register __m256d _C1_0 = _mm256_loadu_pd(&C[1 * ldc + 0]);
    register __m128d _C1_1 = _mm_loadu_pd(&C[1 * ldc + 4]);
    register __m256d _C2_0 = _mm256_loadu_pd(&C[2 * ldc + 0]);
    register __m128d _C2_1 = _mm_loadu_pd(&C[2 * ldc + 4]);
    register __m256d _C3_0 = _mm256_loadu_pd(&C[3 * ldc + 0]);
    register __m128d _C3_1 = _mm_loadu_pd(&C[3 * ldc + 4]);
    register __m256d _C4_0 = _mm256_loadu_pd(&C[4 * ldc + 0]);
    register __m128d _C4_1 = _mm_loadu_pd(&C[4 * ldc + 4]);
    register __m256d _C5_0 = _mm256_loadu_pd(&C[5 * ldc + 0]);
    register __m128d _C5_1 = _mm_loadu_pd(&C[5 * ldc + 4]);

    register __m256d _C6_0 = _mm256_loadu_pd(&C[6 * ldc + 0]);
    register __m128d _C6_1 = _mm_loadu_pd(&C[6 * ldc + 4]);

    register __m256d _C7_0 = _mm256_loadu_pd(&C[7 * ldc + 0]);
    register __m128d _C7_1 = _mm_loadu_pd(&C[7 * ldc + 4]);

    register __m256d _C8_0 = _mm256_loadu_pd(&C[8 * ldc + 0]);
    register __m128d _C8_1 = _mm_loadu_pd(&C[8 * ldc + 4]);

    register __m256d _C9_0 = _mm256_loadu_pd(&C[9 * ldc + 0]);
    register __m128d _C9_1 = _mm_loadu_pd(&C[9 * ldc + 4]);

    register __m256d _C10_0 = _mm256_loadu_pd(&C[10 * ldc + 0]);
    register __m128d _C10_1 = _mm_loadu_pd(&C[10 * ldc + 4]);

    register __m256d _C11_0 = _mm256_loadu_pd(&C[11 * ldc + 0]);
    register __m128d _C11_1 = _mm_loadu_pd(&C[11 * ldc + 4]);

    register __m256d _C12_0 = _mm256_loadu_pd(&C[12 * ldc + 0]);
    register __m128d _C12_1 = _mm_loadu_pd(&C[12 * ldc + 4]);

    for(i = 0; i < k; ++i) {
        _A0 = _mm256_loadu_pd(&A[0]);
        _A1 = _mm_loadu_pd(&A[4]);
        _C0_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C1_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C2_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C3_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A1, _C3_1);
        _C4_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A1, _C4_1);
        _C5_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A1, _C5_1);

        _C6_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm_fnmadd_pd(_mm_set1_pd(B[6 * ldb]), _A1, _C6_1);

        _C7_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 = _mm_fnmadd_pd(_mm_set1_pd(B[7 * ldb]), _A1, _C7_1);

        _C8_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 = _mm_fnmadd_pd(_mm_set1_pd(B[8 * ldb]), _A1, _C8_1);

        _C9_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[9 * ldb]), _A0, _C9_0);
        _C9_1 = _mm_fnmadd_pd(_mm_set1_pd(B[9 * ldb]), _A1, _C9_1);

        _C10_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[10 * ldb]), _A0, _C10_0);
        _C10_1 = _mm_fnmadd_pd(_mm_set1_pd(B[10 * ldb]), _A1, _C10_1);

        _C11_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[11 * ldb]), _A0, _C11_0);
        _C11_1 = _mm_fnmadd_pd(_mm_set1_pd(B[11 * ldb]), _A1, _C11_1);

        _C12_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[12 * ldb]), _A0, _C12_0);
        _C12_1 = _mm_fnmadd_pd(_mm_set1_pd(B[12 * ldb]), _A1, _C12_1);

        A += lda;
        B += 1;
    }

    _mm256_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[0 * ldc + 4], _C0_1);
    _mm256_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm_storeu_pd(&C[1 * ldc + 4], _C1_1);
    _mm256_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm_storeu_pd(&C[2 * ldc + 4], _C2_1);
    _mm256_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm_storeu_pd(&C[3 * ldc + 4], _C3_1);
    _mm256_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm_storeu_pd(&C[4 * ldc + 4], _C4_1);
    _mm256_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm_storeu_pd(&C[5 * ldc + 4], _C5_1);

    _mm256_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm_storeu_pd(&C[6 * ldc + 4], _C6_1);

    _mm256_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm_storeu_pd(&C[7 * ldc + 4], _C7_1);

    _mm256_storeu_pd(&C[8 * ldc + 0], _C8_0);
    _mm_storeu_pd(&C[8 * ldc + 4], _C8_1);

    _mm256_storeu_pd(&C[9 * ldc + 0], _C9_0);
    _mm_storeu_pd(&C[9 * ldc + 4], _C9_1);

    _mm256_storeu_pd(&C[10 * ldc + 0], _C10_0);
    _mm_storeu_pd(&C[10 * ldc + 4], _C10_1);

    _mm256_storeu_pd(&C[11 * ldc + 0], _C11_0);
    _mm_storeu_pd(&C[11 * ldc + 4], _C11_1);

    _mm256_storeu_pd(&C[12 * ldc + 0], _C12_0);
    _mm_storeu_pd(&C[12 * ldc + 4], _C12_1);
}
void micro_kernel7x13(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m256d _A0;
    register __m128d _A1;
    register double _A2;
    register __m256d _C0_0 = _mm256_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C0_1 = _mm_loadu_pd(&C[0 * ldc + 4]);
    register double _C0_2 = C[0 * ldc + 6];
    register __m256d _C1_0 = _mm256_loadu_pd(&C[1 * ldc + 0]);
    register __m128d _C1_1 = _mm_loadu_pd(&C[1 * ldc + 4]);
    register double _C1_2 = C[1 * ldc + 6];
    register __m256d _C2_0 = _mm256_loadu_pd(&C[2 * ldc + 0]);
    register __m128d _C2_1 = _mm_loadu_pd(&C[2 * ldc + 4]);
    register double _C2_2 = C[2 * ldc + 6];
    register __m256d _C3_0 = _mm256_loadu_pd(&C[3 * ldc + 0]);
    register __m128d _C3_1 = _mm_loadu_pd(&C[3 * ldc + 4]);
    register double _C3_2 = C[3 * ldc + 6];
    register __m256d _C4_0 = _mm256_loadu_pd(&C[4 * ldc + 0]);
    register __m128d _C4_1 = _mm_loadu_pd(&C[4 * ldc + 4]);
    register double _C4_2 = C[4 * ldc + 6];
    register __m256d _C5_0 = _mm256_loadu_pd(&C[5 * ldc + 0]);
    register __m128d _C5_1 = _mm_loadu_pd(&C[5 * ldc + 4]);
    register double _C5_2 = C[5 * ldc + 6];

    register __m256d _C6_0 = _mm256_loadu_pd(&C[6 * ldc + 0]);
    register __m128d _C6_1 = _mm_loadu_pd(&C[6 * ldc + 4]);
    register double _C6_2 = C[6 * ldc + 6];

    register __m256d _C7_0 = _mm256_loadu_pd(&C[7 * ldc + 0]);
    register __m128d _C7_1 = _mm_loadu_pd(&C[7 * ldc + 4]);
    register double _C7_2 = C[7 * ldc + 6];

    register __m256d _C8_0 = _mm256_loadu_pd(&C[8 * ldc + 0]);
    register __m128d _C8_1 = _mm_loadu_pd(&C[8 * ldc + 4]);
    register double _C8_2 = C[8 * ldc + 6];

    register __m256d _C9_0 = _mm256_loadu_pd(&C[9 * ldc + 0]);
    register __m128d _C9_1 = _mm_loadu_pd(&C[9 * ldc + 4]);
    register double _C9_2 = C[9 * ldc + 6];

    register __m256d _C10_0 = _mm256_loadu_pd(&C[10 * ldc + 0]);
    register __m128d _C10_1 = _mm_loadu_pd(&C[10 * ldc + 4]);
    register double _C10_2 = C[10 * ldc + 6];

    register __m256d _C11_0 = _mm256_loadu_pd(&C[11 * ldc + 0]);
    register __m128d _C11_1 = _mm_loadu_pd(&C[11 * ldc + 4]);
    register double _C11_2 = C[11 * ldc + 6];

    register __m256d _C12_0 = _mm256_loadu_pd(&C[12 * ldc + 0]);
    register __m128d _C12_1 = _mm_loadu_pd(&C[12 * ldc + 4]);
    register double _C12_2 = C[12 * ldc + 6];

    for(i = 0; i < k; ++i) {
        _A0 = _mm256_loadu_pd(&A[0]);
        _A1 = _mm_loadu_pd(&A[4]);
        _A2 = A[6];
        _C0_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 += B[0 * ldb] * _A2;
        _C1_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C1_2 += B[1 * ldb] * _A2;
        _C2_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C2_2 += B[2 * ldb] * _A2;
        _C3_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A1, _C3_1);
        _C3_2 += B[3 * ldb] * _A2;
        _C4_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A1, _C4_1);
        _C4_2 += B[4 * ldb] * _A2;
        _C5_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A1, _C5_1);
        _C5_2 += B[5 * ldb] * _A2;

        _C6_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm_fnmadd_pd(_mm_set1_pd(B[6 * ldb]), _A1, _C6_1);
        _C6_2 += B[6 * ldb] * _A2;

        _C7_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 = _mm_fnmadd_pd(_mm_set1_pd(B[7 * ldb]), _A1, _C7_1);
        _C7_2 += B[7 * ldb] * _A2;

        _C8_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 = _mm_fnmadd_pd(_mm_set1_pd(B[8 * ldb]), _A1, _C8_1);
        _C8_2 += B[8 * ldb] * _A2;

        _C9_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[9 * ldb]), _A0, _C9_0);
        _C9_1 = _mm_fnmadd_pd(_mm_set1_pd(B[9 * ldb]), _A1, _C9_1);
        _C9_2 += B[9 * ldb] * _A2;

        _C10_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[10 * ldb]), _A0, _C10_0);
        _C10_1 = _mm_fnmadd_pd(_mm_set1_pd(B[10 * ldb]), _A1, _C10_1);
        _C10_2 += B[10 * ldb] * _A2;

        _C11_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[11 * ldb]), _A0, _C11_0);
        _C11_1 = _mm_fnmadd_pd(_mm_set1_pd(B[11 * ldb]), _A1, _C11_1);
        _C11_2 += B[11 * ldb] * _A2;

        _C12_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[12 * ldb]), _A0, _C12_0);
        _C12_1 = _mm_fnmadd_pd(_mm_set1_pd(B[12 * ldb]), _A1, _C12_1);
        _C12_2 += B[12 * ldb] * _A2;

        A += lda;
        B += 1;
    }

    _mm256_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[0 * ldc + 4], _C0_1);
    C[0 * ldc + 6] = _C0_2;
    _mm256_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm_storeu_pd(&C[1 * ldc + 4], _C1_1);
    C[1 * ldc + 6] = _C1_2;
    _mm256_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm_storeu_pd(&C[2 * ldc + 4], _C2_1);
    C[2 * ldc + 6] = _C2_2;
    _mm256_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm_storeu_pd(&C[3 * ldc + 4], _C3_1);
    C[3 * ldc + 6] = _C3_2;
    _mm256_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm_storeu_pd(&C[4 * ldc + 4], _C4_1);
    C[4 * ldc + 6] = _C4_2;
    _mm256_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm_storeu_pd(&C[5 * ldc + 4], _C5_1);
    C[5 * ldc + 6] = _C5_2;

    _mm256_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm_storeu_pd(&C[6 * ldc + 4], _C6_1);
    C[6 * ldc + 6] = _C6_2;

    _mm256_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm_storeu_pd(&C[7 * ldc + 4], _C7_1);
    C[7 * ldc + 6] = _C7_2;

    _mm256_storeu_pd(&C[8 * ldc + 0], _C8_0);
    _mm_storeu_pd(&C[8 * ldc + 4], _C8_1);
    C[8 * ldc + 6] = _C8_2;

    _mm256_storeu_pd(&C[9 * ldc + 0], _C9_0);
    _mm_storeu_pd(&C[9 * ldc + 4], _C9_1);
    C[9 * ldc + 6] = _C9_2;

    _mm256_storeu_pd(&C[10 * ldc + 0], _C10_0);
    _mm_storeu_pd(&C[10 * ldc + 4], _C10_1);
    C[10 * ldc + 6] = _C10_2;

    _mm256_storeu_pd(&C[11 * ldc + 0], _C11_0);
    _mm_storeu_pd(&C[11 * ldc + 4], _C11_1);
    C[11 * ldc + 6] = _C11_2;

    _mm256_storeu_pd(&C[12 * ldc + 0], _C12_0);
    _mm_storeu_pd(&C[12 * ldc + 4], _C12_1);
    C[12 * ldc + 6] = _C12_2;
}
//=====================================================================================================================
void micro_kernel8x13(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);
    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);

    register __m512d _C7_0 = _mm512_loadu_pd(&C[7 * ldc + 0]);

    register __m512d _C8_0 = _mm512_loadu_pd(&C[8 * ldc + 0]);

    register __m512d _C9_0 = _mm512_loadu_pd(&C[9 * ldc + 0]);

    register __m512d _C10_0 = _mm512_loadu_pd(&C[10 * ldc + 0]);

    register __m512d _C11_0 = _mm512_loadu_pd(&C[11 * ldc + 0]);

    register __m512d _C12_0 = _mm512_loadu_pd(&C[12 * ldc + 0]);

    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);

        _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A0, _C7_0);

        _C8_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8 * ldb]), _A0, _C8_0);

        _C9_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[9 * ldb]), _A0, _C9_0);

        _C10_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[10 * ldb]), _A0, _C10_0);

        _C11_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[11 * ldb]), _A0, _C11_0);

        _C12_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[12 * ldb]), _A0, _C12_0);

        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);

    _mm512_storeu_pd(&C[7 * ldc + 0], _C7_0);

    _mm512_storeu_pd(&C[8 * ldc + 0], _C8_0);

    _mm512_storeu_pd(&C[9 * ldc + 0], _C9_0);

    _mm512_storeu_pd(&C[10 * ldc + 0], _C10_0);

    _mm512_storeu_pd(&C[11 * ldc + 0], _C11_0);

    _mm512_storeu_pd(&C[12 * ldc + 0], _C12_0);
}
void micro_kernel9x13(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register double _A1;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register double _C0_1 = C[0 * ldc + 8];

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);
    register double _C1_1 = C[1 * ldc + 8];

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register double _C2_1 = C[2 * ldc + 8];

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register double _C3_1 = C[3 * ldc + 8];

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register double _C4_1 = C[4 * ldc + 8];

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register double _C5_1 = C[5 * ldc + 8];

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);
    register double _C6_1 = C[6 * ldc + 8];

    register __m512d _C7_0 = _mm512_loadu_pd(&C[7 * ldc + 0]);
    register double _C7_1 = C[7 * ldc + 8];

    register __m512d _C8_0 = _mm512_loadu_pd(&C[8 * ldc + 0]);
    register double _C8_1 = C[8 * ldc + 8];

    register __m512d _C9_0 = _mm512_loadu_pd(&C[9 * ldc + 0]);
    register double _C9_1 = C[9 * ldc + 8];

    register __m512d _C10_0 = _mm512_loadu_pd(&C[10 * ldc + 0]);
    register double _C10_1 = C[10 * ldc + 8];

    register __m512d _C11_0 = _mm512_loadu_pd(&C[11 * ldc + 0]);
    register double _C11_1 = C[11 * ldc + 8];

    register __m512d _C12_0 = _mm512_loadu_pd(&C[12 * ldc + 0]);
    register double _C12_1 = C[12 * ldc + 8];

    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = A[8];

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 += B[0 * ldb] * _A1;

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 += B[1 * ldb] * _A1;

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 += B[2 * ldb] * _A1;

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 += B[3 * ldb] * _A1;

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 += B[4 * ldb] * _A1;

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 += B[5 * ldb] * _A1;

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 += B[6 * ldb] * _A1;
        
        _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 += B[7 * ldb] * _A1;

        _C8_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 += B[8 * ldb] * _A1;

        _C9_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[9 * ldb]), _A0, _C9_0);
        _C9_1 += B[9 * ldb] * _A1;

        _C10_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[10 * ldb]), _A0, _C10_0);
        _C10_1 += B[10 * ldb] * _A1;

        _C11_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[11 * ldb]), _A0, _C11_0);
        _C11_1 += B[11 * ldb] * _A1;

        _C12_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[12 * ldb]), _A0, _C12_0);
        _C12_1 += B[12 * ldb] * _A1;

        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    C[0 * ldc + 8] = _C0_1;

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    C[1 * ldc + 8] = _C1_1;

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    C[2 * ldc + 8] = _C2_1;

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    C[3 * ldc + 8] = _C3_1;

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    C[4 * ldc + 8] = _C4_1;

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    C[5 * ldc + 8] = _C5_1;

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);
    C[6 * ldc + 8] = _C6_1;

    _mm512_storeu_pd(&C[7 * ldc + 0], _C7_0);
    C[7 * ldc + 8] = _C7_1;

    _mm512_storeu_pd(&C[8 * ldc + 0], _C8_0);
    C[8 * ldc + 8] = _C8_1;

    _mm512_storeu_pd(&C[9 * ldc + 0], _C9_0);
    C[9 * ldc + 8] = _C9_1;

    _mm512_storeu_pd(&C[10 * ldc + 0], _C10_0);
    C[10 * ldc + 8] = _C10_1;

    _mm512_storeu_pd(&C[11 * ldc + 0], _C11_0);
    C[11 * ldc + 8] = _C11_1;

    _mm512_storeu_pd(&C[12 * ldc + 0], _C12_0);
    C[12 * ldc + 8] = _C12_1;
}
void micro_kernel10x13(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m128d _A1;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C0_1 = _mm_loadu_pd(&C[0 * ldc + 8]);

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m128d _C1_1 = _mm_loadu_pd(&C[1 * ldc + 8]);

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m128d _C2_1 = _mm_loadu_pd(&C[2 * ldc + 8]);

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m128d _C3_1 = _mm_loadu_pd(&C[3 * ldc + 8]);

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m128d _C4_1 = _mm_loadu_pd(&C[4 * ldc + 8]);

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register __m128d _C5_1 = _mm_loadu_pd(&C[5 * ldc + 8]);

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);
    register __m128d _C6_1 = _mm_loadu_pd(&C[6 * ldc + 8]);

    register __m512d _C7_0 = _mm512_loadu_pd(&C[7 * ldc + 0]);
    register __m128d _C7_1 = _mm_loadu_pd(&C[7 * ldc + 8]);

    register __m512d _C8_0 = _mm512_loadu_pd(&C[8 * ldc + 0]);
    register __m128d _C8_1 = _mm_loadu_pd(&C[8 * ldc + 8]);

    register __m512d _C9_0 = _mm512_loadu_pd(&C[9 * ldc + 0]);
    register __m128d _C9_1 = _mm_loadu_pd(&C[9 * ldc + 8]);

    register __m512d _C10_0 = _mm512_loadu_pd(&C[10 * ldc + 0]);
    register __m128d _C10_1 = _mm_loadu_pd(&C[10 * ldc + 8]);

    register __m512d _C11_0 = _mm512_loadu_pd(&C[11 * ldc + 0]);
    register __m128d _C11_1 = _mm_loadu_pd(&C[11 * ldc + 8]);

    register __m512d _C12_0 = _mm512_loadu_pd(&C[12 * ldc + 0]);
    register __m128d _C12_1 = _mm_loadu_pd(&C[12 * ldc + 8]);
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm_loadu_pd(&A[8]);

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A1, _C0_1);

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A1, _C1_1);

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A1, _C2_1);

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A1, _C3_1);

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A1, _C4_1);

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A1, _C5_1);

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm_fnmadd_pd(_mm_set1_pd(B[6 * ldb]), _A1, _C6_1);

        _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 = _mm_fnmadd_pd(_mm_set1_pd(B[7 * ldb]), _A1, _C7_1);

        _C8_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 = _mm_fnmadd_pd(_mm_set1_pd(B[8 * ldb]), _A1, _C8_1);

        _C9_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[9 * ldb]), _A0, _C9_0);
        _C9_1 = _mm_fnmadd_pd(_mm_set1_pd(B[9 * ldb]), _A1, _C9_1);

        _C10_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[10 * ldb]), _A0, _C10_0);
        _C10_1 = _mm_fnmadd_pd(_mm_set1_pd(B[10 * ldb]), _A1, _C10_1);

        _C11_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[11 * ldb]), _A0, _C11_0);
        _C11_1 = _mm_fnmadd_pd(_mm_set1_pd(B[11 * ldb]), _A1, _C11_1);

        _C12_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[12 * ldb]), _A0, _C12_0);
        _C12_1 = _mm_fnmadd_pd(_mm_set1_pd(B[12 * ldb]), _A1, _C12_1);
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[0 * ldc + 8], _C0_1);

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm_storeu_pd(&C[1 * ldc + 8], _C1_1);

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm_storeu_pd(&C[2 * ldc + 8], _C2_1);

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm_storeu_pd(&C[3 * ldc + 8], _C3_1);

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm_storeu_pd(&C[4 * ldc + 8], _C4_1);

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm_storeu_pd(&C[5 * ldc + 8], _C5_1);

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm_storeu_pd(&C[6 * ldc + 8], _C6_1);

    _mm512_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm_storeu_pd(&C[7 * ldc + 8], _C7_1);

    _mm512_storeu_pd(&C[8 * ldc + 0], _C8_0);
    _mm_storeu_pd(&C[8 * ldc + 8], _C8_1);

    _mm512_storeu_pd(&C[9 * ldc + 0], _C9_0);
    _mm_storeu_pd(&C[9 * ldc + 8], _C9_1);

    _mm512_storeu_pd(&C[10 * ldc + 0], _C10_0);
    _mm_storeu_pd(&C[10 * ldc + 8], _C10_1);

    _mm512_storeu_pd(&C[11 * ldc + 0], _C11_0);
    _mm_storeu_pd(&C[11 * ldc + 8], _C11_1);

    _mm512_storeu_pd(&C[12 * ldc + 0], _C12_0);
    _mm_storeu_pd(&C[12 * ldc + 8], _C12_1);
}
void micro_kernel11x13(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m128d _A1;
    register double _A2;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C0_1 = _mm_loadu_pd(&C[0 * ldc + 8]);
    register double _C0_2 = C[0 * ldc + 10];

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m128d _C1_1 = _mm_loadu_pd(&C[1 * ldc + 8]);
    register double _C1_2 = C[1 * ldc + 10];

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m128d _C2_1 = _mm_loadu_pd(&C[2 * ldc + 8]);
    register double _C2_2 = C[2 * ldc + 10];

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m128d _C3_1 = _mm_loadu_pd(&C[3 * ldc + 8]);
    register double _C3_2 = C[3 * ldc + 10];

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m128d _C4_1 = _mm_loadu_pd(&C[4 * ldc + 8]);
    register double _C4_2 = C[4 * ldc + 10];

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register __m128d _C5_1 = _mm_loadu_pd(&C[5 * ldc + 8]);
    register double _C5_2 = C[5 * ldc + 10];

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);
    register __m128d _C6_1 = _mm_loadu_pd(&C[6 * ldc + 8]);
    register double _C6_2 = C[6 * ldc + 10];

    register __m512d _C7_0 = _mm512_loadu_pd(&C[7 * ldc + 0]);
    register __m128d _C7_1 = _mm_loadu_pd(&C[7 * ldc + 8]);
    register double _C7_2 = C[7 * ldc + 10];

    register __m512d _C8_0 = _mm512_loadu_pd(&C[8 * ldc + 0]);
    register __m128d _C8_1 = _mm_loadu_pd(&C[8 * ldc + 8]);
    register double _C8_2 = C[8 * ldc + 10];

    register __m512d _C9_0 = _mm512_loadu_pd(&C[9 * ldc + 0]);
    register __m128d _C9_1 = _mm_loadu_pd(&C[9 * ldc + 8]);
    register double _C9_2 = C[9 * ldc + 10];

    register __m512d _C10_0 = _mm512_loadu_pd(&C[10 * ldc + 0]);
    register __m128d _C10_1 = _mm_loadu_pd(&C[10 * ldc + 8]);
    register double _C10_2 = C[10 * ldc + 10];
    
    register __m512d _C11_0 = _mm512_loadu_pd(&C[11 * ldc + 0]);
    register __m128d _C11_1 = _mm_loadu_pd(&C[11 * ldc + 8]);
    register double _C11_2 = C[11 * ldc + 10];

    register __m512d _C12_0 = _mm512_loadu_pd(&C[12 * ldc + 0]);
    register __m128d _C12_1 = _mm_loadu_pd(&C[12 * ldc + 8]);
    register double _C12_2 = C[12 * ldc + 10];

    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm_loadu_pd(&A[8]);
        _A2 = A[10];

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 += B[0 * ldb] * _A2;

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C1_2 += B[1 * ldb] * _A2;

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C2_2 += B[2 * ldb] * _A2;

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A1, _C3_1);
        _C3_2 += B[3 * ldb] * _A2;

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A1, _C4_1);
        _C4_2 += B[4 * ldb] * _A2;

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A1, _C5_1);
        _C5_2 += B[5 * ldb] * _A2;

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm_fnmadd_pd(_mm_set1_pd(B[6 * ldb]), _A1, _C6_1);
        _C6_2 += B[6 * ldb] * _A2;

        _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 = _mm_fnmadd_pd(_mm_set1_pd(B[7 * ldb]), _A1, _C7_1);
        _C7_2 += B[7 * ldb] * _A2;

        _C8_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 = _mm_fnmadd_pd(_mm_set1_pd(B[8 * ldb]), _A1, _C8_1);
        _C8_2 += B[8 * ldb] * _A2;

        _C9_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[9 * ldb]), _A0, _C9_0);
        _C9_1 = _mm_fnmadd_pd(_mm_set1_pd(B[9 * ldb]), _A1, _C9_1);
        _C9_2 += B[9 * ldb] * _A2;

        _C10_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[10 * ldb]), _A0, _C10_0);
        _C10_1 = _mm_fnmadd_pd(_mm_set1_pd(B[10 * ldb]), _A1, _C10_1);
        _C10_2 += B[10 * ldb] * _A2;
        
        _C11_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[11 * ldb]), _A0, _C11_0);
        _C11_1 = _mm_fnmadd_pd(_mm_set1_pd(B[11 * ldb]), _A1, _C11_1);
        _C11_2 += B[11 * ldb] * _A2;

        _C12_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[12 * ldb]), _A0, _C12_0);
        _C12_1 = _mm_fnmadd_pd(_mm_set1_pd(B[12 * ldb]), _A1, _C12_1);
        _C12_2 += B[12 * ldb] * _A2; 

        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[0 * ldc + 8], _C0_1);
    C[0 * ldc + 10] = _C0_2;

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm_storeu_pd(&C[1 * ldc + 8], _C1_1);
    C[1 * ldc + 10] = _C1_2;

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm_storeu_pd(&C[2 * ldc + 8], _C2_1);
    C[2 * ldc + 10] = _C2_2;

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm_storeu_pd(&C[3 * ldc + 8], _C3_1);
    C[3 * ldc + 10] = _C3_2;

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm_storeu_pd(&C[4 * ldc + 8], _C4_1);
    C[4 * ldc + 10] = _C4_2;

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm_storeu_pd(&C[5 * ldc + 8], _C5_1);
    C[5 * ldc + 10] = _C5_2;

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm_storeu_pd(&C[6 * ldc + 8], _C6_1);
    C[6 * ldc + 10] = _C6_2;

    _mm512_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm_storeu_pd(&C[7 * ldc + 8], _C7_1);
    C[7 * ldc + 10] = _C7_2;

    _mm512_storeu_pd(&C[8 * ldc + 0], _C8_0);
    _mm_storeu_pd(&C[8 * ldc + 8], _C8_1);
    C[8 * ldc + 10] = _C8_2;

    _mm512_storeu_pd(&C[9 * ldc + 0], _C9_0);
    _mm_storeu_pd(&C[9 * ldc + 8], _C9_1);
    C[9 * ldc + 10] = _C9_2;

    _mm512_storeu_pd(&C[10 * ldc + 0], _C10_0);
    _mm_storeu_pd(&C[10 * ldc + 8], _C10_1);
    C[10 * ldc + 10] = _C10_2;

    _mm512_storeu_pd(&C[11 * ldc + 0], _C11_0);
    _mm_storeu_pd(&C[11 * ldc + 8], _C11_1);
    C[11 * ldc + 10] = _C11_2;

    _mm512_storeu_pd(&C[12 * ldc + 0], _C12_0);
    _mm_storeu_pd(&C[12 * ldc + 8], _C12_1);
    C[12 * ldc + 10] = _C12_2;
}
void micro_kernel12x13(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {//------------------------------------
     int i;
    register __m512d _A0;
    register __m256d _A1;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C0_1 = _mm256_loadu_pd(&C[0 * ldc + 8]);

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m256d _C1_1 = _mm256_loadu_pd(&C[1 * ldc + 8]);

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m256d _C2_1 = _mm256_loadu_pd(&C[2 * ldc + 8]);

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m256d _C3_1 = _mm256_loadu_pd(&C[3 * ldc + 8]);

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m256d _C4_1 = _mm256_loadu_pd(&C[4 * ldc + 8]);

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register __m256d _C5_1 = _mm256_loadu_pd(&C[5 * ldc + 8]);

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);
    register __m256d _C6_1 = _mm256_loadu_pd(&C[6 * ldc + 8]);

    register __m512d _C7_0 = _mm512_loadu_pd(&C[7 * ldc + 0]);
    register __m256d _C7_1 = _mm256_loadu_pd(&C[7 * ldc + 8]);

    register __m512d _C8_0 = _mm512_loadu_pd(&C[8 * ldc + 0]);
    register __m256d _C8_1 = _mm256_loadu_pd(&C[8 * ldc + 8]);

    register __m512d _C9_0 = _mm512_loadu_pd(&C[9 * ldc + 0]);
    register __m256d _C9_1 = _mm256_loadu_pd(&C[9 * ldc + 8]);

    register __m512d _C10_0 = _mm512_loadu_pd(&C[10 * ldc + 0]);
    register __m256d _C10_1 = _mm256_loadu_pd(&C[10 * ldc + 8]);

    register __m512d _C11_0 = _mm512_loadu_pd(&C[11 * ldc + 0]);
    register __m256d _C11_1 = _mm256_loadu_pd(&C[11 * ldc + 8]);

    register __m512d _C12_0 = _mm512_loadu_pd(&C[12 * ldc + 0]);
    register __m256d _C12_1 = _mm256_loadu_pd(&C[12 * ldc + 8]);
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm256_loadu_pd(&A[8]);

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A1, _C0_1);

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A1, _C1_1);

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A1, _C2_1);

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A1, _C3_1);

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A1, _C4_1);

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A1, _C5_1);

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[6 * ldb]), _A1, _C6_1);

        _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[7 * ldb]), _A1, _C7_1);

        _C8_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[8 * ldb]), _A1, _C8_1);

        _C9_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[9 * ldb]), _A0, _C9_0);
        _C9_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[9 * ldb]), _A1, _C9_1);

        _C10_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[10 * ldb]), _A0, _C10_0);
        _C10_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[10 * ldb]), _A1, _C10_1);

        _C11_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[11 * ldb]), _A0, _C11_0);
        _C11_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[11 * ldb]), _A1, _C11_1);

        _C12_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[12 * ldb]), _A0, _C12_0);
        _C12_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[12 * ldb]), _A1, _C12_1);
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[0 * ldc + 8], _C0_1);

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm256_storeu_pd(&C[1 * ldc + 8], _C1_1);

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm256_storeu_pd(&C[2 * ldc + 8], _C2_1);

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm256_storeu_pd(&C[3 * ldc + 8], _C3_1);

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm256_storeu_pd(&C[4 * ldc + 8], _C4_1);

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm256_storeu_pd(&C[5 * ldc + 8], _C5_1);

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm256_storeu_pd(&C[6 * ldc + 8], _C6_1);

    _mm512_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm256_storeu_pd(&C[7 * ldc + 8], _C7_1);

    _mm512_storeu_pd(&C[8 * ldc + 0], _C8_0);
    _mm256_storeu_pd(&C[8 * ldc + 8], _C8_1);

    _mm512_storeu_pd(&C[9 * ldc + 0], _C9_0);
    _mm256_storeu_pd(&C[9 * ldc + 8], _C9_1);

    _mm512_storeu_pd(&C[10 * ldc + 0], _C10_0);
    _mm256_storeu_pd(&C[10 * ldc + 8], _C10_1);

    _mm512_storeu_pd(&C[11 * ldc + 0], _C11_0);
    _mm256_storeu_pd(&C[11 * ldc + 8], _C11_1);

    _mm512_storeu_pd(&C[12 * ldc + 0], _C12_0);
    _mm256_storeu_pd(&C[12 * ldc + 8], _C12_1);
}
void micro_kernel13x13(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m256d _A1;
    register double _A2;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C0_1 = _mm256_loadu_pd(&C[0 * ldc + 8]);
    register double _C0_2 = C[0 * ldc + 12];

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m256d _C1_1 = _mm256_loadu_pd(&C[1 * ldc + 8]);
    register double _C1_2 = C[1 * ldc + 12];

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m256d _C2_1 = _mm256_loadu_pd(&C[2 * ldc + 8]);
    register double _C2_2 = C[2 * ldc + 12];

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m256d _C3_1 = _mm256_loadu_pd(&C[3 * ldc + 8]);
    register double _C3_2 = C[3 * ldc + 12];

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m256d _C4_1 = _mm256_loadu_pd(&C[4 * ldc + 8]);
    register double _C4_2 = C[4 * ldc + 12];

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register __m256d _C5_1 = _mm256_loadu_pd(&C[5 * ldc + 8]);
    register double _C5_2 = C[5 * ldc + 12];

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);
    register __m256d _C6_1 = _mm256_loadu_pd(&C[6 * ldc + 8]);
    register double _C6_2 = C[6 * ldc + 12];

    register __m512d _C7_0 = _mm512_loadu_pd(&C[7 * ldc + 0]);
    register __m256d _C7_1 = _mm256_loadu_pd(&C[7 * ldc + 8]);
    register double _C7_2 = C[7 * ldc + 12];

    register __m512d _C8_0 = _mm512_loadu_pd(&C[8 * ldc + 0]);
    register __m256d _C8_1 = _mm256_loadu_pd(&C[8 * ldc + 8]);
    register double _C8_2 = C[8 * ldc + 12];

    register __m512d _C9_0 = _mm512_loadu_pd(&C[9 * ldc + 0]);
    register __m256d _C9_1 = _mm256_loadu_pd(&C[9 * ldc + 8]);
    register double _C9_2 = C[9 * ldc + 12];

    register __m512d _C10_0 = _mm512_loadu_pd(&C[10 * ldc + 0]);
    register __m256d _C10_1 = _mm256_loadu_pd(&C[10 * ldc + 8]);
    register double _C10_2 = C[10 * ldc + 12];

    register __m512d _C11_0 = _mm512_loadu_pd(&C[11 * ldc + 0]);
    register __m256d _C11_1 = _mm256_loadu_pd(&C[11 * ldc + 8]);
    register double _C11_2 = C[11 * ldc + 12];

    register __m512d _C12_0 = _mm512_loadu_pd(&C[12 * ldc + 0]);
    register __m256d _C12_1 = _mm256_loadu_pd(&C[12 * ldc + 8]);
    register double _C12_2 = C[12 * ldc + 12];
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm256_loadu_pd(&A[8]);
        _A2 = A[12];

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 += B[0 * ldb] * _A2;

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C1_2 += B[1 * ldb] * _A2;

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C2_2 += B[2 * ldb] * _A2;

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A1, _C3_1);
        _C3_2 += B[3 * ldb] * _A2;

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A1, _C4_1);
        _C4_2 += B[4 * ldb] * _A2;

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A1, _C5_1);
        _C5_2 += B[5 * ldb] * _A2;

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[6 * ldb]), _A1, _C6_1);
        _C6_2 += B[6 * ldb] * _A2;

        _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[7 * ldb]), _A1, _C7_1);
        _C7_2 += B[7 * ldb] * _A2;

        _C8_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[8 * ldb]), _A1, _C8_1);
        _C8_2 += B[8 * ldb] * _A2;

        _C9_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[9 * ldb]), _A0, _C9_0);
        _C9_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[9 * ldb]), _A1, _C9_1);
        _C9_2 += B[9 * ldb] * _A2;

        _C10_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[10 * ldb]), _A0, _C10_0);
        _C10_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[10 * ldb]), _A1, _C10_1);
        _C10_2 += B[10 * ldb] * _A2;

        _C11_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[11 * ldb]), _A0, _C11_0);
        _C11_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[11 * ldb]), _A1, _C11_1);
        _C11_2 += B[11 * ldb] * _A2;

        _C12_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[12 * ldb]), _A0, _C12_0);
        _C12_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[12 * ldb]), _A1, _C12_1);
        _C12_2 += B[12 * ldb] * _A2;
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[0 * ldc + 8], _C0_1);
    C[0 * ldc + 12] = _C0_2;

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm256_storeu_pd(&C[1 * ldc + 8], _C1_1);
    C[1 * ldc + 12] = _C1_2;

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm256_storeu_pd(&C[2 * ldc + 8], _C2_1);
    C[2 * ldc + 12] = _C2_2;

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm256_storeu_pd(&C[3 * ldc + 8], _C3_1);
    C[3 * ldc + 12] = _C3_2;

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm256_storeu_pd(&C[4 * ldc + 8], _C4_1);
    C[4 * ldc + 12] = _C4_2;

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm256_storeu_pd(&C[5 * ldc + 8], _C5_1);
    C[5 * ldc + 12] = _C5_2;

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm256_storeu_pd(&C[6 * ldc + 8], _C6_1);
    C[6 * ldc + 12] = _C6_2;

    _mm512_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm256_storeu_pd(&C[7 * ldc + 8], _C7_1);
    C[7 * ldc + 12] = _C7_2;

    _mm512_storeu_pd(&C[8 * ldc + 0], _C8_0);
    _mm256_storeu_pd(&C[8 * ldc + 8], _C8_1);
    C[8 * ldc + 12] = _C8_2;

    _mm512_storeu_pd(&C[9 * ldc + 0], _C9_0);
    _mm256_storeu_pd(&C[9 * ldc + 8], _C9_1);
    C[9 * ldc + 12] = _C9_2;

    _mm512_storeu_pd(&C[10 * ldc + 0], _C10_0);
    _mm256_storeu_pd(&C[10 * ldc + 8], _C10_1);
    C[10 * ldc + 12] = _C10_2;

    _mm512_storeu_pd(&C[11 * ldc + 0], _C11_0);
    _mm256_storeu_pd(&C[11 * ldc + 8], _C11_1);
    C[11 * ldc + 12] = _C11_2;

    _mm512_storeu_pd(&C[12 * ldc + 0], _C12_0);
    _mm256_storeu_pd(&C[12 * ldc + 8], _C12_1);
    C[12 * ldc + 12] = _C12_2;
}
void micro_kernel14x13(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m256d _A1;
    register __m128d _A2;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C0_1 = _mm256_loadu_pd(&C[0 * ldc + 8]);
    register __m128d _C0_2 = _mm_loadu_pd(&C[0 * ldc + 12]);

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m256d _C1_1 = _mm256_loadu_pd(&C[1 * ldc + 8]);
    register __m128d _C1_2 = _mm_loadu_pd(&C[1 * ldc + 12]);

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m256d _C2_1 = _mm256_loadu_pd(&C[2 * ldc + 8]);
    register __m128d _C2_2 = _mm_loadu_pd(&C[2 * ldc + 12]);

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m256d _C3_1 = _mm256_loadu_pd(&C[3 * ldc + 8]);
    register __m128d _C3_2 = _mm_loadu_pd(&C[3 * ldc + 12]);

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m256d _C4_1 = _mm256_loadu_pd(&C[4 * ldc + 8]);
    register __m128d _C4_2 = _mm_loadu_pd(&C[4 * ldc + 12]);

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register __m256d _C5_1 = _mm256_loadu_pd(&C[5 * ldc + 8]);
    register __m128d _C5_2 = _mm_loadu_pd(&C[5 * ldc + 12]);

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);
    register __m256d _C6_1 = _mm256_loadu_pd(&C[6 * ldc + 8]);
    register __m128d _C6_2 = _mm_loadu_pd(&C[6 * ldc + 12]);

    register __m512d _C7_0 = _mm512_loadu_pd(&C[7 * ldc + 0]);
    register __m256d _C7_1 = _mm256_loadu_pd(&C[7 * ldc + 8]);
    register __m128d _C7_2 = _mm_loadu_pd(&C[7 * ldc + 12]);

    register __m512d _C8_0 = _mm512_loadu_pd(&C[8 * ldc + 0]);
    register __m256d _C8_1 = _mm256_loadu_pd(&C[8 * ldc + 8]);
    register __m128d _C8_2 = _mm_loadu_pd(&C[8 * ldc + 12]);

    register __m512d _C9_0 = _mm512_loadu_pd(&C[9 * ldc + 0]);
    register __m256d _C9_1 = _mm256_loadu_pd(&C[9 * ldc + 8]);
    register __m128d _C9_2 = _mm_loadu_pd(&C[9 * ldc + 12]);

    register __m512d _C10_0 = _mm512_loadu_pd(&C[10 * ldc + 0]);
    register __m256d _C10_1 = _mm256_loadu_pd(&C[10 * ldc + 8]);
    register __m128d _C10_2 = _mm_loadu_pd(&C[10 * ldc + 12]);

    register __m512d _C11_0 = _mm512_loadu_pd(&C[11 * ldc + 0]);
    register __m256d _C11_1 = _mm256_loadu_pd(&C[11 * ldc + 8]);
    register __m128d _C11_2 = _mm_loadu_pd(&C[11 * ldc + 12]);

    register __m512d _C12_0 = _mm512_loadu_pd(&C[12 * ldc + 0]);
    register __m256d _C12_1 = _mm256_loadu_pd(&C[12 * ldc + 8]);
    register __m128d _C12_2 = _mm_loadu_pd(&C[12 * ldc + 12]);

    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm256_loadu_pd(&A[8]);
        _A2 = _mm_loadu_pd(&A[12]);

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A2, _C0_2);

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C1_2 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A2, _C1_2);

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C2_2 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A2, _C2_2);

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A1, _C3_1);
        _C3_2 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A2, _C3_2);

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A1, _C4_1);
        _C4_2 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A2, _C4_2);

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A1, _C5_1);
        _C5_2 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A2, _C5_2);

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[6 * ldb]), _A1, _C6_1);
        _C6_2 = _mm_fnmadd_pd(_mm_set1_pd(B[6 * ldb]), _A2, _C6_2);

        _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[7 * ldb]), _A1, _C7_1);
        _C7_2 = _mm_fnmadd_pd(_mm_set1_pd(B[7 * ldb]), _A2, _C7_2);

        _C8_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[8 * ldb]), _A1, _C8_1);
        _C8_2 = _mm_fnmadd_pd(_mm_set1_pd(B[8 * ldb]), _A2, _C8_2);

        _C9_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[9 * ldb]), _A0, _C9_0);
        _C9_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[9 * ldb]), _A1, _C9_1);
        _C9_2 = _mm_fnmadd_pd(_mm_set1_pd(B[9 * ldb]), _A2, _C9_2);

        _C10_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[10 * ldb]), _A0, _C10_0);
        _C10_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[10 * ldb]), _A1, _C10_1);
        _C10_2 = _mm_fnmadd_pd(_mm_set1_pd(B[10 * ldb]), _A2, _C10_2);
 

        _C11_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[11 * ldb]), _A0, _C11_0);
        _C11_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[11 * ldb]), _A1, _C11_1);
        _C11_2 = _mm_fnmadd_pd(_mm_set1_pd(B[11 * ldb]), _A2, _C11_2);

        _C12_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[12 * ldb]), _A0, _C12_0);
        _C12_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[12 * ldb]), _A1, _C12_1);
        _C12_2 = _mm_fnmadd_pd(_mm_set1_pd(B[12 * ldb]), _A2, _C12_2);

        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[0 * ldc + 8], _C0_1);
    _mm_storeu_pd(&C[0 * ldc + 12], _C0_2);

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm256_storeu_pd(&C[1 * ldc + 8], _C1_1);
    _mm_storeu_pd(&C[1 * ldc + 12], _C1_2);

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm256_storeu_pd(&C[2 * ldc + 8], _C2_1);
    _mm_storeu_pd(&C[2 * ldc + 12], _C2_2);

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm256_storeu_pd(&C[3 * ldc + 8], _C3_1);
    _mm_storeu_pd(&C[3 * ldc + 12], _C3_2);

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm256_storeu_pd(&C[4 * ldc + 8], _C4_1);
    _mm_storeu_pd(&C[4 * ldc + 12], _C4_2);

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm256_storeu_pd(&C[5 * ldc + 8], _C5_1);
    _mm_storeu_pd(&C[5 * ldc + 12], _C5_2);

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm256_storeu_pd(&C[6 * ldc + 8], _C6_1);
    _mm_storeu_pd(&C[6 * ldc + 12], _C6_2);

    _mm512_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm256_storeu_pd(&C[7 * ldc + 8], _C7_1);
    _mm_storeu_pd(&C[7 * ldc + 12], _C7_2);

    _mm512_storeu_pd(&C[8 * ldc + 0], _C8_0);
    _mm256_storeu_pd(&C[8 * ldc + 8], _C8_1);
    _mm_storeu_pd(&C[8 * ldc + 12], _C8_2);

    _mm512_storeu_pd(&C[9 * ldc + 0], _C9_0);
    _mm256_storeu_pd(&C[9 * ldc + 8], _C9_1);
    _mm_storeu_pd(&C[9 * ldc + 12], _C9_2);

    _mm512_storeu_pd(&C[10 * ldc + 0], _C10_0);
    _mm256_storeu_pd(&C[10 * ldc + 8], _C10_1);
    _mm_storeu_pd(&C[10 * ldc + 12], _C10_2);

    _mm512_storeu_pd(&C[11 * ldc + 0], _C11_0);
    _mm256_storeu_pd(&C[11 * ldc + 8], _C11_1);
    _mm_storeu_pd(&C[11 * ldc + 12], _C11_2);

    _mm512_storeu_pd(&C[12 * ldc + 0], _C12_0);
    _mm256_storeu_pd(&C[12 * ldc + 8], _C12_1);
    _mm_storeu_pd(&C[12 * ldc + 12], _C12_2);
}
void micro_kernel15x13(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m256d _A1;
    register __m128d _A2;
    register double _A3;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C0_1 = _mm256_loadu_pd(&C[0 * ldc + 8]);
    register __m128d _C0_2 = _mm_loadu_pd(&C[0 * ldc + 12]);
    register double _C0_3 = C[0 * ldc + 14];

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m256d _C1_1 = _mm256_loadu_pd(&C[1 * ldc + 8]);
    register __m128d _C1_2 = _mm_loadu_pd(&C[1 * ldc + 12]);
    register double _C1_3 = C[1 * ldc + 14];

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m256d _C2_1 = _mm256_loadu_pd(&C[2 * ldc + 8]);
    register __m128d _C2_2 = _mm_loadu_pd(&C[2 * ldc + 12]);
    register double _C2_3 = C[2 * ldc + 14];

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m256d _C3_1 = _mm256_loadu_pd(&C[3 * ldc + 8]);
    register __m128d _C3_2 = _mm_loadu_pd(&C[3 * ldc + 12]);
    register double _C3_3 = C[3 * ldc + 14];

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m256d _C4_1 = _mm256_loadu_pd(&C[4 * ldc + 8]);
    register __m128d _C4_2 = _mm_loadu_pd(&C[4 * ldc + 12]);
    register double _C4_3 = C[4 * ldc + 14];

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register __m256d _C5_1 = _mm256_loadu_pd(&C[5 * ldc + 8]);
    register __m128d _C5_2 = _mm_loadu_pd(&C[5 * ldc + 12]);
    register double _C5_3 = C[5 * ldc + 14];

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);
    register __m256d _C6_1 = _mm256_loadu_pd(&C[6 * ldc + 8]);
    register __m128d _C6_2 = _mm_loadu_pd(&C[6 * ldc + 12]);
    register double _C6_3 = C[6 * ldc + 14];

    register __m512d _C7_0 = _mm512_loadu_pd(&C[7 * ldc + 0]);
    register __m256d _C7_1 = _mm256_loadu_pd(&C[7 * ldc + 8]);
    register __m128d _C7_2 = _mm_loadu_pd(&C[7 * ldc + 12]);
    register double _C7_3 = C[7 * ldc + 14];

    register __m512d _C8_0 = _mm512_loadu_pd(&C[8 * ldc + 0]);
    register __m256d _C8_1 = _mm256_loadu_pd(&C[8 * ldc + 8]);
    register __m128d _C8_2 = _mm_loadu_pd(&C[8 * ldc + 12]);
    register double _C8_3 = C[8 * ldc + 14];

    register __m512d _C9_0 = _mm512_loadu_pd(&C[9 * ldc + 0]);
    register __m256d _C9_1 = _mm256_loadu_pd(&C[9 * ldc + 8]);
    register __m128d _C9_2 = _mm_loadu_pd(&C[9 * ldc + 12]);
    register double _C9_3 = C[9 * ldc + 14];

    register __m512d _C10_0 = _mm512_loadu_pd(&C[10 * ldc + 0]);
    register __m256d _C10_1 = _mm256_loadu_pd(&C[10 * ldc + 8]);
    register __m128d _C10_2 = _mm_loadu_pd(&C[10 * ldc + 12]);
    register double _C10_3 = C[10 * ldc + 14];

    register __m512d _C11_0 = _mm512_loadu_pd(&C[11 * ldc + 0]);
    register __m256d _C11_1 = _mm256_loadu_pd(&C[11 * ldc + 8]);
    register __m128d _C11_2 = _mm_loadu_pd(&C[11 * ldc + 12]);
    register double _C11_3 = C[11 * ldc + 14];

    register __m512d _C12_0 = _mm512_loadu_pd(&C[12 * ldc + 0]);
    register __m256d _C12_1 = _mm256_loadu_pd(&C[12 * ldc + 8]);
    register __m128d _C12_2 = _mm_loadu_pd(&C[12 * ldc + 12]);
    register double _C12_3 = C[12 * ldc + 14];
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm256_loadu_pd(&A[8]);
        _A2 = _mm_loadu_pd(&A[12]);
        _A3 = A[14];

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A2, _C0_2);
        _C0_3 += B[0 * ldb] * _A3;

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C1_2 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A2, _C1_2);
        _C1_3 += B[1 * ldb] * _A3;

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C2_2 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A2, _C2_2);
        _C2_3 += B[2 * ldb] * _A3;

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A1, _C3_1);
        _C3_2 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A2, _C3_2);
        _C3_3 += B[3 * ldb] * _A3;

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A1, _C4_1);
        _C4_2 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A2, _C4_2);
        _C4_3 += B[4 * ldb] * _A3;

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A1, _C5_1);
        _C5_2 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A2, _C5_2);
        _C5_3 += B[5 * ldb] * _A3;

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[6 * ldb]), _A1, _C6_1);
        _C6_2 = _mm_fnmadd_pd(_mm_set1_pd(B[6 * ldb]), _A2, _C6_2);
        _C6_3 += B[6 * ldb] * _A3;

        _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[7 * ldb]), _A1, _C7_1);
        _C7_2 = _mm_fnmadd_pd(_mm_set1_pd(B[7 * ldb]), _A2, _C7_2);
        _C7_3 += B[7 * ldb] * _A3;

        _C8_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[8 * ldb]), _A1, _C8_1);
        _C8_2 = _mm_fnmadd_pd(_mm_set1_pd(B[8 * ldb]), _A2, _C8_2);
        _C8_3 += B[8 * ldb] * _A3;

        _C9_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[9 * ldb]), _A0, _C9_0);
        _C9_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[9 * ldb]), _A1, _C9_1);
        _C9_2 = _mm_fnmadd_pd(_mm_set1_pd(B[9 * ldb]), _A2, _C9_2);
        _C9_3 += B[9 * ldb] * _A3;

        _C10_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[10 * ldb]), _A0, _C10_0);
        _C10_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[10 * ldb]), _A1, _C10_1);
        _C10_2 = _mm_fnmadd_pd(_mm_set1_pd(B[10 * ldb]), _A2, _C10_2);
        _C10_3 += B[10 * ldb] * _A3;
        
        _C11_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[11 * ldb]), _A0, _C11_0);
        _C11_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[11 * ldb]), _A1, _C11_1);
        _C11_2 = _mm_fnmadd_pd(_mm_set1_pd(B[11 * ldb]), _A2, _C11_2);
        _C11_3 += B[11 * ldb] * _A3;
        
        _C12_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[12 * ldb]), _A0, _C12_0);
        _C12_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[12 * ldb]), _A1, _C12_1);
        _C12_2 = _mm_fnmadd_pd(_mm_set1_pd(B[12 * ldb]), _A2, _C12_2);
        _C12_3 += B[12 * ldb] * _A3;

        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[0 * ldc + 8], _C0_1);
    _mm_storeu_pd(&C[0 * ldc + 12], _C0_2);
    C[0 * ldc + 14] = _C0_3;

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm256_storeu_pd(&C[1 * ldc + 8], _C1_1);
    _mm_storeu_pd(&C[1 * ldc + 12], _C1_2);
    C[1 * ldc + 14] = _C1_3;

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm256_storeu_pd(&C[2 * ldc + 8], _C2_1);
    _mm_storeu_pd(&C[2 * ldc + 12], _C2_2);
    C[2 * ldc + 14] = _C2_3;

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm256_storeu_pd(&C[3 * ldc + 8], _C3_1);
    _mm_storeu_pd(&C[3 * ldc + 12], _C3_2);
    C[3 * ldc + 14] = _C3_3;

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm256_storeu_pd(&C[4 * ldc + 8], _C4_1);
    _mm_storeu_pd(&C[4 * ldc + 12], _C4_2);
    C[4 * ldc + 14] = _C4_3;

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm256_storeu_pd(&C[5 * ldc + 8], _C5_1);
    _mm_storeu_pd(&C[5 * ldc + 12], _C5_2);
    C[5 * ldc + 14] = _C5_3;

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm256_storeu_pd(&C[6 * ldc + 8], _C6_1);
    _mm_storeu_pd(&C[6 * ldc + 12], _C6_2);
    C[6 * ldc + 14] = _C6_3;

    _mm512_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm256_storeu_pd(&C[7 * ldc + 8], _C7_1);
    _mm_storeu_pd(&C[7 * ldc + 12], _C7_2);
    C[7 * ldc + 14] = _C7_3;

    _mm512_storeu_pd(&C[8 * ldc + 0], _C8_0);
    _mm256_storeu_pd(&C[8 * ldc + 8], _C8_1);
    _mm_storeu_pd(&C[8 * ldc + 12], _C8_2);
    C[8 * ldc + 14] = _C8_3;

    _mm512_storeu_pd(&C[9 * ldc + 0], _C9_0);
    _mm256_storeu_pd(&C[9 * ldc + 8], _C9_1);
    _mm_storeu_pd(&C[9 * ldc + 12], _C9_2);
    C[9 * ldc + 14] = _C9_3;

    _mm512_storeu_pd(&C[10 * ldc + 0], _C10_0);
    _mm256_storeu_pd(&C[10 * ldc + 8], _C10_1);
    _mm_storeu_pd(&C[10 * ldc + 12], _C10_2);
    C[10 * ldc + 14] = _C10_3;

    _mm512_storeu_pd(&C[11 * ldc + 0], _C11_0);
    _mm256_storeu_pd(&C[11 * ldc + 8], _C11_1);
    _mm_storeu_pd(&C[11 * ldc + 12], _C11_2);
    C[11 * ldc + 14] = _C11_3;

    _mm512_storeu_pd(&C[12 * ldc + 0], _C12_0);
    _mm256_storeu_pd(&C[12 * ldc + 8], _C12_1);
    _mm_storeu_pd(&C[12 * ldc + 12], _C12_2);
    C[12 * ldc + 14] = _C12_3;
}
void micro_kernel16x13(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m512d _A1;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m512d _C0_1 = _mm512_loadu_pd(&C[0 * ldc + 8]);

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m512d _C1_1 = _mm512_loadu_pd(&C[1 * ldc + 8]);

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m512d _C2_1 = _mm512_loadu_pd(&C[2 * ldc + 8]);

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m512d _C3_1 = _mm512_loadu_pd(&C[3 * ldc + 8]);

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m512d _C4_1 = _mm512_loadu_pd(&C[4 * ldc + 8]);

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register __m512d _C5_1 = _mm512_loadu_pd(&C[5 * ldc + 8]);

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);
    register __m512d _C6_1 = _mm512_loadu_pd(&C[6 * ldc + 8]);

    register __m512d _C7_0 = _mm512_loadu_pd(&C[7 * ldc + 0]);
    register __m512d _C7_1 = _mm512_loadu_pd(&C[7 * ldc + 8]);

    register __m512d _C8_0 = _mm512_loadu_pd(&C[8 * ldc + 0]);
    register __m512d _C8_1 = _mm512_loadu_pd(&C[8 * ldc + 8]);

    register __m512d _C9_0 = _mm512_loadu_pd(&C[9 * ldc + 0]);
    register __m512d _C9_1 = _mm512_loadu_pd(&C[9 * ldc + 8]);

    register __m512d _C10_0 = _mm512_loadu_pd(&C[10 * ldc + 0]);
    register __m512d _C10_1 = _mm512_loadu_pd(&C[10 * ldc + 8]);

    register __m512d _C11_0 = _mm512_loadu_pd(&C[11 * ldc + 0]);
    register __m512d _C11_1 = _mm512_loadu_pd(&C[11 * ldc + 8]);

    register __m512d _C12_0 = _mm512_loadu_pd(&C[12 * ldc + 0]);
    register __m512d _C12_1 = _mm512_loadu_pd(&C[12 * ldc + 8]);
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm512_loadu_pd(&A[8]);

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A1, _C0_1);

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A1, _C1_1);

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A1, _C2_1);

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A1, _C3_1);

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A1, _C4_1);

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A1, _C5_1);

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A1, _C6_1);

        _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A1, _C7_1);

        _C8_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8 * ldb]), _A1, _C8_1);

        _C9_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[9 * ldb]), _A0, _C9_0);
        _C9_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[9 * ldb]), _A1, _C9_1);

        _C10_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[10 * ldb]), _A0, _C10_0);
        _C10_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[10 * ldb]), _A1, _C10_1);

        _C11_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[11 * ldb]), _A0, _C11_0);
        _C11_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[11 * ldb]), _A1, _C11_1);

        _C12_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[12 * ldb]), _A0, _C12_0);
        _C12_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[12 * ldb]), _A1, _C12_1);
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm512_storeu_pd(&C[0 * ldc + 8], _C0_1);

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm512_storeu_pd(&C[1 * ldc + 8], _C1_1);

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm512_storeu_pd(&C[2 * ldc + 8], _C2_1);

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm512_storeu_pd(&C[3 * ldc + 8], _C3_1);

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm512_storeu_pd(&C[4 * ldc + 8], _C4_1);

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm512_storeu_pd(&C[5 * ldc + 8], _C5_1);

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm512_storeu_pd(&C[6 * ldc + 8], _C6_1);

    _mm512_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm512_storeu_pd(&C[7 * ldc + 8], _C7_1);

    _mm512_storeu_pd(&C[8 * ldc + 0], _C8_0);
    _mm512_storeu_pd(&C[8 * ldc + 8], _C8_1);

    _mm512_storeu_pd(&C[9 * ldc + 0], _C9_0);
    _mm512_storeu_pd(&C[9 * ldc + 8], _C9_1);

    _mm512_storeu_pd(&C[10 * ldc + 0], _C10_0);
    _mm512_storeu_pd(&C[10 * ldc + 8], _C10_1);

    _mm512_storeu_pd(&C[11 * ldc + 0], _C11_0);
    _mm512_storeu_pd(&C[11 * ldc + 8], _C11_1);

    _mm512_storeu_pd(&C[12 * ldc + 0], _C12_0);
    _mm512_storeu_pd(&C[12 * ldc + 8], _C12_1);
}

//======================-----------------======================-------------------======================

void micro_kernel1x14(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register double _A0;
    register double _C0_0 = C[0 * ldc + 0];
    register double _C1_0 = C[1 * ldc + 0];
    register double _C2_0 = C[2 * ldc + 0];
    register double _C3_0 = C[3 * ldc + 0];
    register double _C4_0 = C[4 * ldc + 0];
    register double _C5_0 = C[5 * ldc + 0];
    register double _C6_0 = C[6 * ldc + 0];
    register double _C7_0 = C[7 * ldc + 0];
    register double _C8_0 = C[8 * ldc + 0];
    register double _C9_0 = C[9 * ldc + 0];
    register double _C10_0 = C[10 * ldc + 0];
    register double _C11_0 = C[11 * ldc + 0];
    register double _C12_0 = C[12 * ldc + 0];
    register double _C13_0 = C[13 * ldc + 0];

    for(i = 0; i < k; ++i) {
        _A0 = A[0];
        _C0_0 += B[0 * ldb] * _A0;
        _C1_0 += B[1 * ldb] * _A0;
        _C2_0 += B[2 * ldb] * _A0;
        _C3_0 += B[3 * ldb] * _A0;
        _C4_0 += B[4 * ldb] * _A0;
        _C5_0 += B[5 * ldb] * _A0;
        _C6_0 += B[6 * ldb] * _A0;
        _C7_0 += B[7 * ldb] * _A0;
        _C8_0 += B[8 * ldb] * _A0;
        _C9_0 += B[9 * ldb] * _A0;
        _C10_0 += B[10 * ldb] * _A0;
        _C11_0 += B[11 * ldb] * _A0;
        _C12_0 += B[12 * ldb] * _A0;
        _C13_0 += B[13 * ldb] * _A0;

        A += lda;
        B += 1;
    }

    C[0 * ldc + 0] = _C0_0;
    C[1 * ldc + 0] = _C1_0;
    C[2 * ldc + 0] = _C2_0;
    C[3 * ldc + 0] = _C3_0;
    C[4 * ldc + 0] = _C4_0;
    C[5 * ldc + 0] = _C5_0;
    C[6 * ldc + 0] = _C6_0;
    C[7 * ldc + 0] = _C7_0;
    C[8 * ldc + 0] = _C8_0;
    C[9 * ldc + 0] = _C9_0;
    C[10 * ldc + 0] = _C10_0;
    C[11 * ldc + 0] = _C11_0;
    C[12 * ldc + 0] = _C12_0;
    C[13 * ldc + 0] = _C13_0;
}
void micro_kernel2x14(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m128d _A0;
    register __m128d _C0_0 = _mm_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C1_0 = _mm_loadu_pd(&C[1 * ldc + 0]);
    register __m128d _C2_0 = _mm_loadu_pd(&C[2 * ldc + 0]);
    register __m128d _C3_0 = _mm_loadu_pd(&C[3 * ldc + 0]);
    register __m128d _C4_0 = _mm_loadu_pd(&C[4 * ldc + 0]);
    register __m128d _C5_0 = _mm_loadu_pd(&C[5 * ldc + 0]);
    register __m128d _C6_0 = _mm_loadu_pd(&C[6 * ldc + 0]);
    register __m128d _C7_0 = _mm_loadu_pd(&C[7 * ldc + 0]);
    register __m128d _C8_0 = _mm_loadu_pd(&C[8 * ldc + 0]);
    register __m128d _C9_0 = _mm_loadu_pd(&C[9 * ldc + 0]);
    register __m128d _C10_0 = _mm_loadu_pd(&C[10 * ldc + 0]);
    register __m128d _C11_0 = _mm_loadu_pd(&C[11 * ldc + 0]);
    register __m128d _C12_0 = _mm_loadu_pd(&C[12 * ldc + 0]);
    register __m128d _C13_0 = _mm_loadu_pd(&C[13 * ldc + 0]);

    for(i = 0; i < k; ++i) {
        _A0 = _mm_loadu_pd(&A[0]);
        _C0_0 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C1_0 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C2_0 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C3_0 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C4_0 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C5_0 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C6_0 = _mm_fnmadd_pd(_mm_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C7_0 = _mm_fnmadd_pd(_mm_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C8_0 = _mm_fnmadd_pd(_mm_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C9_0 = _mm_fnmadd_pd(_mm_set1_pd(B[9 * ldb]), _A0, _C9_0);
        _C10_0 = _mm_fnmadd_pd(_mm_set1_pd(B[10 * ldb]), _A0, _C10_0);
        _C11_0 = _mm_fnmadd_pd(_mm_set1_pd(B[11 * ldb]), _A0, _C11_0);
        _C12_0 = _mm_fnmadd_pd(_mm_set1_pd(B[12 * ldb]), _A0, _C12_0);
        _C13_0 = _mm_fnmadd_pd(_mm_set1_pd(B[13 * ldb]), _A0, _C13_0);

        A += lda;
        B += 1;
    }

    _mm_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm_storeu_pd(&C[8 * ldc + 0], _C8_0);
    _mm_storeu_pd(&C[9 * ldc + 0], _C9_0);
    _mm_storeu_pd(&C[10 * ldc + 0], _C10_0);
    _mm_storeu_pd(&C[11 * ldc + 0], _C11_0);
    _mm_storeu_pd(&C[12 * ldc + 0], _C12_0);
    _mm_storeu_pd(&C[13 * ldc + 0], _C13_0);
}
void micro_kernel3x14(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m128d _A0;
    register double _A1;
    register __m128d _C0_0 = _mm_loadu_pd(&C[0 * ldc + 0]);
    register double _C0_1 = C[0 * ldc + 2];
    register __m128d _C1_0 = _mm_loadu_pd(&C[1 * ldc + 0]);
    register double _C1_1 = C[1 * ldc + 2];
    register __m128d _C2_0 = _mm_loadu_pd(&C[2 * ldc + 0]);
    register double _C2_1 = C[2 * ldc + 2];
    register __m128d _C3_0 = _mm_loadu_pd(&C[3 * ldc + 0]);
    register double _C3_1 = C[3 * ldc + 2];
    register __m128d _C4_0 = _mm_loadu_pd(&C[4 * ldc + 0]);
    register double _C4_1 = C[4 * ldc + 2];
    register __m128d _C5_0 = _mm_loadu_pd(&C[5 * ldc + 0]);
    register double _C5_1 = C[5 * ldc + 2];

    register __m128d _C6_0 = _mm_loadu_pd(&C[6 * ldc + 0]);
    register double _C6_1 = C[6 * ldc + 2];

    register __m128d _C7_0 = _mm_loadu_pd(&C[7 * ldc + 0]);
    register double _C7_1 = C[7 * ldc + 2];

    register __m128d _C8_0 = _mm_loadu_pd(&C[8 * ldc + 0]);
    register double _C8_1 = C[8 * ldc + 2];

    register __m128d _C9_0 = _mm_loadu_pd(&C[9 * ldc + 0]);
    register double _C9_1 = C[9 * ldc + 2];

    register __m128d _C10_0 = _mm_loadu_pd(&C[10 * ldc + 0]);
    register double _C10_1 = C[10 * ldc + 2];

    register __m128d _C11_0 = _mm_loadu_pd(&C[11 * ldc + 0]);
    register double _C11_1 = C[11 * ldc + 2];

    register __m128d _C12_0 = _mm_loadu_pd(&C[12 * ldc + 0]);
    register double _C12_1 = C[12 * ldc + 2];

    register __m128d _C13_0 = _mm_loadu_pd(&C[13 * ldc + 0]);
    register double _C13_1 = C[13 * ldc + 2];

    for(i = 0; i < k; ++i) {
        _A0 = _mm_loadu_pd(&A[0]);
        _A1 = A[2];
        _C0_0 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 += B[0 * ldb] * _A1;
        _C1_0 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 += B[1 * ldb] * _A1;
        _C2_0 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 += B[2 * ldb] * _A1;
        _C3_0 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 += B[3 * ldb] * _A1;
        _C4_0 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 += B[4 * ldb] * _A1;
        _C5_0 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 += B[5 * ldb] * _A1;

        _C6_0 = _mm_fnmadd_pd(_mm_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 += B[6 * ldb] * _A1;

        _C7_0 = _mm_fnmadd_pd(_mm_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 += B[7 * ldb] * _A1;

        _C8_0 = _mm_fnmadd_pd(_mm_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 += B[8 * ldb] * _A1;

        _C9_0 = _mm_fnmadd_pd(_mm_set1_pd(B[9 * ldb]), _A0, _C9_0);
        _C9_1 += B[9 * ldb] * _A1;

        _C10_0 = _mm_fnmadd_pd(_mm_set1_pd(B[10 * ldb]), _A0, _C10_0);
        _C10_1 += B[10 * ldb] * _A1;

        _C11_0 = _mm_fnmadd_pd(_mm_set1_pd(B[11 * ldb]), _A0, _C11_0);
        _C11_1 += B[11 * ldb] * _A1;

        _C12_0 = _mm_fnmadd_pd(_mm_set1_pd(B[12 * ldb]), _A0, _C12_0);
        _C12_1 += B[12 * ldb] * _A1;

        _C13_0 = _mm_fnmadd_pd(_mm_set1_pd(B[13 * ldb]), _A0, _C13_0);
        _C13_1 += B[13 * ldb] * _A1;

        A += lda;
        B += 1;
    }

    _mm_storeu_pd(&C[0 * ldc + 0], _C0_0);
    C[0 * ldc + 2] = _C0_1;
    _mm_storeu_pd(&C[1 * ldc + 0], _C1_0);
    C[1 * ldc + 2] = _C1_1;
    _mm_storeu_pd(&C[2 * ldc + 0], _C2_0);
    C[2 * ldc + 2] = _C2_1;
    _mm_storeu_pd(&C[3 * ldc + 0], _C3_0);
    C[3 * ldc + 2] = _C3_1;
    _mm_storeu_pd(&C[4 * ldc + 0], _C4_0);
    C[4 * ldc + 2] = _C4_1;
    _mm_storeu_pd(&C[5 * ldc + 0], _C5_0);
    C[5 * ldc + 2] = _C5_1;

    _mm_storeu_pd(&C[6 * ldc + 0], _C6_0);
    C[6 * ldc + 2] = _C6_1;

    _mm_storeu_pd(&C[7 * ldc + 0], _C7_0);
    C[7 * ldc + 2] = _C7_1;

    _mm_storeu_pd(&C[8 * ldc + 0], _C8_0);
    C[8 * ldc + 2] = _C8_1;

    _mm_storeu_pd(&C[9 * ldc + 0], _C9_0);
    C[9 * ldc + 2] = _C9_1;

    _mm_storeu_pd(&C[10 * ldc + 0], _C10_0);
    C[10 * ldc + 2] = _C10_1;

    _mm_storeu_pd(&C[11 * ldc + 0], _C11_0);
    C[11 * ldc + 2] = _C11_1;
    _mm_storeu_pd(&C[12 * ldc + 0], _C12_0);
    C[12 * ldc + 2] = _C12_1;

    _mm_storeu_pd(&C[13 * ldc + 0], _C13_0);
    C[13 * ldc + 2] = _C13_1;
}
void micro_kernel4x14(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m256d _A0;
    register __m256d _C0_0 = _mm256_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C1_0 = _mm256_loadu_pd(&C[1 * ldc + 0]);
    register __m256d _C2_0 = _mm256_loadu_pd(&C[2 * ldc + 0]);
    register __m256d _C3_0 = _mm256_loadu_pd(&C[3 * ldc + 0]);
    register __m256d _C4_0 = _mm256_loadu_pd(&C[4 * ldc + 0]);
    register __m256d _C5_0 = _mm256_loadu_pd(&C[5 * ldc + 0]);

    register __m256d _C6_0 = _mm256_loadu_pd(&C[6 * ldc + 0]);

    register __m256d _C7_0 = _mm256_loadu_pd(&C[7 * ldc + 0]);

    register __m256d _C8_0 = _mm256_loadu_pd(&C[8 * ldc + 0]);

    register __m256d _C9_0 = _mm256_loadu_pd(&C[9 * ldc + 0]);

    register __m256d _C10_0 = _mm256_loadu_pd(&C[10 * ldc + 0]);

    register __m256d _C11_0 = _mm256_loadu_pd(&C[11 * ldc + 0]);

    register __m256d _C12_0 = _mm256_loadu_pd(&C[12 * ldc + 0]);

    register __m256d _C13_0 = _mm256_loadu_pd(&C[13 * ldc + 0]);

    for(i = 0; i < k; ++i) {
        _A0 = _mm256_loadu_pd(&A[0]);
        _C0_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C1_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C2_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C3_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C4_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C5_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A0, _C5_0);

        _C6_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[6 * ldb]), _A0, _C6_0);

        _C7_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[7 * ldb]), _A0, _C7_0);

        _C8_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[8 * ldb]), _A0, _C8_0);

        _C9_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[9 * ldb]), _A0, _C9_0);

        _C10_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[10 * ldb]), _A0, _C10_0);

        _C11_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[11 * ldb]), _A0, _C11_0);

        _C12_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[12 * ldb]), _A0, _C12_0);

        _C13_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[13 * ldb]), _A0, _C13_0);

        A += lda;
        B += 1;
    }

    _mm256_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm256_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm256_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm256_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm256_storeu_pd(&C[5 * ldc + 0], _C5_0);

    _mm256_storeu_pd(&C[6 * ldc + 0], _C6_0);

    _mm256_storeu_pd(&C[7 * ldc + 0], _C7_0);

    _mm256_storeu_pd(&C[8 * ldc + 0], _C8_0);

    _mm256_storeu_pd(&C[9 * ldc + 0], _C9_0);

    _mm256_storeu_pd(&C[10 * ldc + 0], _C10_0);

    _mm256_storeu_pd(&C[11 * ldc + 0], _C11_0);

    _mm256_storeu_pd(&C[12 * ldc + 0], _C12_0);

    _mm256_storeu_pd(&C[13 * ldc + 0], _C13_0);
}
void micro_kernel5x14(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m256d _A0;
    register double _A1;
    register __m256d _C0_0 = _mm256_loadu_pd(&C[0 * ldc + 0]);
    register double _C0_1 = C[0 * ldc + 4];
    register __m256d _C1_0 = _mm256_loadu_pd(&C[1 * ldc + 0]);
    register double _C1_1 = C[1 * ldc + 4];
    register __m256d _C2_0 = _mm256_loadu_pd(&C[2 * ldc + 0]);
    register double _C2_1 = C[2 * ldc + 4];
    register __m256d _C3_0 = _mm256_loadu_pd(&C[3 * ldc + 0]);
    register double _C3_1 = C[3 * ldc + 4];
    register __m256d _C4_0 = _mm256_loadu_pd(&C[4 * ldc + 0]);
    register double _C4_1 = C[4 * ldc + 4];
    register __m256d _C5_0 = _mm256_loadu_pd(&C[5 * ldc + 0]);
    register double _C5_1 = C[5 * ldc + 4];

    register __m256d _C6_0 = _mm256_loadu_pd(&C[6 * ldc + 0]);
    register double _C6_1 = C[6 * ldc + 4];

    register __m256d _C7_0 = _mm256_loadu_pd(&C[7 * ldc + 0]);
    register double _C7_1 = C[7 * ldc + 4];

    register __m256d _C8_0 = _mm256_loadu_pd(&C[8 * ldc + 0]);
    register double _C8_1 = C[8 * ldc + 4];

    register __m256d _C9_0 = _mm256_loadu_pd(&C[9 * ldc + 0]);
    register double _C9_1 = C[9 * ldc + 4];

    register __m256d _C10_0 = _mm256_loadu_pd(&C[10 * ldc + 0]);
    register double _C10_1 = C[10 * ldc + 4];

    register __m256d _C11_0 = _mm256_loadu_pd(&C[11 * ldc + 0]);
    register double _C11_1 = C[11 * ldc + 4];

    register __m256d _C12_0 = _mm256_loadu_pd(&C[12 * ldc + 0]);
    register double _C12_1 = C[12 * ldc + 4];

    register __m256d _C13_0 = _mm256_loadu_pd(&C[13 * ldc + 0]);
    register double _C13_1 = C[13 * ldc + 4];

    for(i = 0; i < k; ++i) {
        _A0 = _mm256_loadu_pd(&A[0]);
        _A1 = A[4];
        _C0_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 += B[0 * ldb] * _A1;
        _C1_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 += B[1 * ldb] * _A1;
        _C2_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 += B[2 * ldb] * _A1;
        _C3_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 += B[3 * ldb] * _A1;
        _C4_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 += B[4 * ldb] * _A1;
        _C5_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 += B[5 * ldb] * _A1;

        _C6_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 += B[6 * ldb] * _A1;

        _C7_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 += B[7 * ldb] * _A1;

        _C8_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 += B[8 * ldb] * _A1;

        _C9_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[9 * ldb]), _A0, _C9_0);
        _C9_1 += B[9 * ldb] * _A1;

        _C10_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[10 * ldb]), _A0, _C10_0);
        _C10_1 += B[10 * ldb] * _A1;

        _C11_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[11 * ldb]), _A0, _C11_0);
        _C11_1 += B[11 * ldb] * _A1;

        _C12_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[12 * ldb]), _A0, _C12_0);
        _C12_1 += B[12 * ldb] * _A1;

        _C13_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[13 * ldb]), _A0, _C13_0);
        _C13_1 += B[13 * ldb] * _A1;
        A += lda;
        B += 1;
    }

    _mm256_storeu_pd(&C[0 * ldc + 0], _C0_0);
    C[0 * ldc + 4] = _C0_1;
    _mm256_storeu_pd(&C[1 * ldc + 0], _C1_0);
    C[1 * ldc + 4] = _C1_1;
    _mm256_storeu_pd(&C[2 * ldc + 0], _C2_0);
    C[2 * ldc + 4] = _C2_1;
    _mm256_storeu_pd(&C[3 * ldc + 0], _C3_0);
    C[3 * ldc + 4] = _C3_1;
    _mm256_storeu_pd(&C[4 * ldc + 0], _C4_0);
    C[4 * ldc + 4] = _C4_1;
    _mm256_storeu_pd(&C[5 * ldc + 0], _C5_0);
    C[5 * ldc + 4] = _C5_1;

    _mm256_storeu_pd(&C[6 * ldc + 0], _C6_0);
    C[6 * ldc + 4] = _C6_1;

    _mm256_storeu_pd(&C[7 * ldc + 0], _C7_0);
    C[7 * ldc + 4] = _C7_1;

    _mm256_storeu_pd(&C[8 * ldc + 0], _C8_0);
    C[8 * ldc + 4] = _C8_1;

    _mm256_storeu_pd(&C[9 * ldc + 0], _C9_0);
    C[9 * ldc + 4] = _C9_1;

    _mm256_storeu_pd(&C[10 * ldc + 0], _C10_0);
    C[10 * ldc + 4] = _C10_1;

    _mm256_storeu_pd(&C[11 * ldc + 0], _C11_0);
    C[11 * ldc + 4] = _C11_1;

    _mm256_storeu_pd(&C[12 * ldc + 0], _C12_0);
    C[12 * ldc + 4] = _C12_1;

    _mm256_storeu_pd(&C[13 * ldc + 0], _C13_0);
    C[13 * ldc + 4] = _C13_1;
}
void micro_kernel6x14(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m256d _A0;
    register __m128d _A1;
    register __m256d _C0_0 = _mm256_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C0_1 = _mm_loadu_pd(&C[0 * ldc + 4]);
    register __m256d _C1_0 = _mm256_loadu_pd(&C[1 * ldc + 0]);
    register __m128d _C1_1 = _mm_loadu_pd(&C[1 * ldc + 4]);
    register __m256d _C2_0 = _mm256_loadu_pd(&C[2 * ldc + 0]);
    register __m128d _C2_1 = _mm_loadu_pd(&C[2 * ldc + 4]);
    register __m256d _C3_0 = _mm256_loadu_pd(&C[3 * ldc + 0]);
    register __m128d _C3_1 = _mm_loadu_pd(&C[3 * ldc + 4]);
    register __m256d _C4_0 = _mm256_loadu_pd(&C[4 * ldc + 0]);
    register __m128d _C4_1 = _mm_loadu_pd(&C[4 * ldc + 4]);
    register __m256d _C5_0 = _mm256_loadu_pd(&C[5 * ldc + 0]);
    register __m128d _C5_1 = _mm_loadu_pd(&C[5 * ldc + 4]);

    register __m256d _C6_0 = _mm256_loadu_pd(&C[6 * ldc + 0]);
    register __m128d _C6_1 = _mm_loadu_pd(&C[6 * ldc + 4]);

    register __m256d _C7_0 = _mm256_loadu_pd(&C[7 * ldc + 0]);
    register __m128d _C7_1 = _mm_loadu_pd(&C[7 * ldc + 4]);

    register __m256d _C8_0 = _mm256_loadu_pd(&C[8 * ldc + 0]);
    register __m128d _C8_1 = _mm_loadu_pd(&C[8 * ldc + 4]);

    register __m256d _C9_0 = _mm256_loadu_pd(&C[9 * ldc + 0]);
    register __m128d _C9_1 = _mm_loadu_pd(&C[9 * ldc + 4]);

    register __m256d _C10_0 = _mm256_loadu_pd(&C[10 * ldc + 0]);
    register __m128d _C10_1 = _mm_loadu_pd(&C[10 * ldc + 4]);

    register __m256d _C11_0 = _mm256_loadu_pd(&C[11 * ldc + 0]);
    register __m128d _C11_1 = _mm_loadu_pd(&C[11 * ldc + 4]);

    register __m256d _C12_0 = _mm256_loadu_pd(&C[12 * ldc + 0]);
    register __m128d _C12_1 = _mm_loadu_pd(&C[12 * ldc + 4]);

    register __m256d _C13_0 = _mm256_loadu_pd(&C[13 * ldc + 0]);
    register __m128d _C13_1 = _mm_loadu_pd(&C[13 * ldc + 4]);

    for(i = 0; i < k; ++i) {
        _A0 = _mm256_loadu_pd(&A[0]);
        _A1 = _mm_loadu_pd(&A[4]);
        _C0_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C1_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C2_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C3_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A1, _C3_1);
        _C4_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A1, _C4_1);
        _C5_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A1, _C5_1);

        _C6_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm_fnmadd_pd(_mm_set1_pd(B[6 * ldb]), _A1, _C6_1);

        _C7_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 = _mm_fnmadd_pd(_mm_set1_pd(B[7 * ldb]), _A1, _C7_1);

        _C8_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 = _mm_fnmadd_pd(_mm_set1_pd(B[8 * ldb]), _A1, _C8_1);

        _C9_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[9 * ldb]), _A0, _C9_0);
        _C9_1 = _mm_fnmadd_pd(_mm_set1_pd(B[9 * ldb]), _A1, _C9_1);

        _C10_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[10 * ldb]), _A0, _C10_0);
        _C10_1 = _mm_fnmadd_pd(_mm_set1_pd(B[10 * ldb]), _A1, _C10_1);

        _C11_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[11 * ldb]), _A0, _C11_0);
        _C11_1 = _mm_fnmadd_pd(_mm_set1_pd(B[11 * ldb]), _A1, _C11_1);

        _C12_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[12 * ldb]), _A0, _C12_0);
        _C12_1 = _mm_fnmadd_pd(_mm_set1_pd(B[12 * ldb]), _A1, _C12_1);

        _C13_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[13 * ldb]), _A0, _C13_0);
        _C13_1 = _mm_fnmadd_pd(_mm_set1_pd(B[13 * ldb]), _A1, _C13_1);

        A += lda;
        B += 1;
    }

    _mm256_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[0 * ldc + 4], _C0_1);
    _mm256_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm_storeu_pd(&C[1 * ldc + 4], _C1_1);
    _mm256_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm_storeu_pd(&C[2 * ldc + 4], _C2_1);
    _mm256_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm_storeu_pd(&C[3 * ldc + 4], _C3_1);
    _mm256_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm_storeu_pd(&C[4 * ldc + 4], _C4_1);
    _mm256_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm_storeu_pd(&C[5 * ldc + 4], _C5_1);

    _mm256_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm_storeu_pd(&C[6 * ldc + 4], _C6_1);

    _mm256_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm_storeu_pd(&C[7 * ldc + 4], _C7_1);

    _mm256_storeu_pd(&C[8 * ldc + 0], _C8_0);
    _mm_storeu_pd(&C[8 * ldc + 4], _C8_1);

    _mm256_storeu_pd(&C[9 * ldc + 0], _C9_0);
    _mm_storeu_pd(&C[9 * ldc + 4], _C9_1);

    _mm256_storeu_pd(&C[10 * ldc + 0], _C10_0);
    _mm_storeu_pd(&C[10 * ldc + 4], _C10_1);

    _mm256_storeu_pd(&C[11 * ldc + 0], _C11_0);
    _mm_storeu_pd(&C[11 * ldc + 4], _C11_1);

    _mm256_storeu_pd(&C[12 * ldc + 0], _C12_0);
    _mm_storeu_pd(&C[12 * ldc + 4], _C12_1);

    _mm256_storeu_pd(&C[13 * ldc + 0], _C13_0);
    _mm_storeu_pd(&C[13 * ldc + 4], _C13_1);
}
void micro_kernel7x14(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m256d _A0;
    register __m128d _A1;
    register double _A2;
    register __m256d _C0_0 = _mm256_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C0_1 = _mm_loadu_pd(&C[0 * ldc + 4]);
    register double _C0_2 = C[0 * ldc + 6];
    register __m256d _C1_0 = _mm256_loadu_pd(&C[1 * ldc + 0]);
    register __m128d _C1_1 = _mm_loadu_pd(&C[1 * ldc + 4]);
    register double _C1_2 = C[1 * ldc + 6];
    register __m256d _C2_0 = _mm256_loadu_pd(&C[2 * ldc + 0]);
    register __m128d _C2_1 = _mm_loadu_pd(&C[2 * ldc + 4]);
    register double _C2_2 = C[2 * ldc + 6];
    register __m256d _C3_0 = _mm256_loadu_pd(&C[3 * ldc + 0]);
    register __m128d _C3_1 = _mm_loadu_pd(&C[3 * ldc + 4]);
    register double _C3_2 = C[3 * ldc + 6];
    register __m256d _C4_0 = _mm256_loadu_pd(&C[4 * ldc + 0]);
    register __m128d _C4_1 = _mm_loadu_pd(&C[4 * ldc + 4]);
    register double _C4_2 = C[4 * ldc + 6];
    register __m256d _C5_0 = _mm256_loadu_pd(&C[5 * ldc + 0]);
    register __m128d _C5_1 = _mm_loadu_pd(&C[5 * ldc + 4]);
    register double _C5_2 = C[5 * ldc + 6];

    register __m256d _C6_0 = _mm256_loadu_pd(&C[6 * ldc + 0]);
    register __m128d _C6_1 = _mm_loadu_pd(&C[6 * ldc + 4]);
    register double _C6_2 = C[6 * ldc + 6];

    register __m256d _C7_0 = _mm256_loadu_pd(&C[7 * ldc + 0]);
    register __m128d _C7_1 = _mm_loadu_pd(&C[7 * ldc + 4]);
    register double _C7_2 = C[7 * ldc + 6];

    register __m256d _C8_0 = _mm256_loadu_pd(&C[8 * ldc + 0]);
    register __m128d _C8_1 = _mm_loadu_pd(&C[8 * ldc + 4]);
    register double _C8_2 = C[8 * ldc + 6];

    register __m256d _C9_0 = _mm256_loadu_pd(&C[9 * ldc + 0]);
    register __m128d _C9_1 = _mm_loadu_pd(&C[9 * ldc + 4]);
    register double _C9_2 = C[9 * ldc + 6];

    register __m256d _C10_0 = _mm256_loadu_pd(&C[10 * ldc + 0]);
    register __m128d _C10_1 = _mm_loadu_pd(&C[10 * ldc + 4]);
    register double _C10_2 = C[10 * ldc + 6];

    register __m256d _C11_0 = _mm256_loadu_pd(&C[11 * ldc + 0]);
    register __m128d _C11_1 = _mm_loadu_pd(&C[11 * ldc + 4]);
    register double _C11_2 = C[11 * ldc + 6];

    register __m256d _C12_0 = _mm256_loadu_pd(&C[12 * ldc + 0]);
    register __m128d _C12_1 = _mm_loadu_pd(&C[12 * ldc + 4]);
    register double _C12_2 = C[12 * ldc + 6];

    register __m256d _C13_0 = _mm256_loadu_pd(&C[13 * ldc + 0]);
    register __m128d _C13_1 = _mm_loadu_pd(&C[13 * ldc + 4]);
    register double _C13_2 = C[13 * ldc + 6];

    for(i = 0; i < k; ++i) {
        _A0 = _mm256_loadu_pd(&A[0]);
        _A1 = _mm_loadu_pd(&A[4]);
        _A2 = A[6];
        _C0_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 += B[0 * ldb] * _A2;
        _C1_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C1_2 += B[1 * ldb] * _A2;
        _C2_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C2_2 += B[2 * ldb] * _A2;
        _C3_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A1, _C3_1);
        _C3_2 += B[3 * ldb] * _A2;
        _C4_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A1, _C4_1);
        _C4_2 += B[4 * ldb] * _A2;
        _C5_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A1, _C5_1);
        _C5_2 += B[5 * ldb] * _A2;

        _C6_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm_fnmadd_pd(_mm_set1_pd(B[6 * ldb]), _A1, _C6_1);
        _C6_2 += B[6 * ldb] * _A2;

        _C7_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 = _mm_fnmadd_pd(_mm_set1_pd(B[7 * ldb]), _A1, _C7_1);
        _C7_2 += B[7 * ldb] * _A2;

        _C8_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 = _mm_fnmadd_pd(_mm_set1_pd(B[8 * ldb]), _A1, _C8_1);
        _C8_2 += B[8 * ldb] * _A2;

        _C9_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[9 * ldb]), _A0, _C9_0);
        _C9_1 = _mm_fnmadd_pd(_mm_set1_pd(B[9 * ldb]), _A1, _C9_1);
        _C9_2 += B[9 * ldb] * _A2;

        _C10_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[10 * ldb]), _A0, _C10_0);
        _C10_1 = _mm_fnmadd_pd(_mm_set1_pd(B[10 * ldb]), _A1, _C10_1);
        _C10_2 += B[10 * ldb] * _A2;

        _C11_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[11 * ldb]), _A0, _C11_0);
        _C11_1 = _mm_fnmadd_pd(_mm_set1_pd(B[11 * ldb]), _A1, _C11_1);
        _C11_2 += B[11 * ldb] * _A2;

        _C12_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[12 * ldb]), _A0, _C12_0);
        _C12_1 = _mm_fnmadd_pd(_mm_set1_pd(B[12 * ldb]), _A1, _C12_1);
        _C12_2 += B[12 * ldb] * _A2;

        _C13_0 = _mm256_fnmadd_pd(_mm256_set1_pd(B[13 * ldb]), _A0, _C13_0);
        _C13_1 = _mm_fnmadd_pd(_mm_set1_pd(B[13 * ldb]), _A1, _C13_1);
        _C13_2 += B[13 * ldb] * _A2;

        A += lda;
        B += 1;
    }

    _mm256_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[0 * ldc + 4], _C0_1);
    C[0 * ldc + 6] = _C0_2;
    _mm256_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm_storeu_pd(&C[1 * ldc + 4], _C1_1);
    C[1 * ldc + 6] = _C1_2;
    _mm256_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm_storeu_pd(&C[2 * ldc + 4], _C2_1);
    C[2 * ldc + 6] = _C2_2;
    _mm256_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm_storeu_pd(&C[3 * ldc + 4], _C3_1);
    C[3 * ldc + 6] = _C3_2;
    _mm256_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm_storeu_pd(&C[4 * ldc + 4], _C4_1);
    C[4 * ldc + 6] = _C4_2;
    _mm256_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm_storeu_pd(&C[5 * ldc + 4], _C5_1);
    C[5 * ldc + 6] = _C5_2;

    _mm256_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm_storeu_pd(&C[6 * ldc + 4], _C6_1);
    C[6 * ldc + 6] = _C6_2;

    _mm256_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm_storeu_pd(&C[7 * ldc + 4], _C7_1);
    C[7 * ldc + 6] = _C7_2;

    _mm256_storeu_pd(&C[8 * ldc + 0], _C8_0);
    _mm_storeu_pd(&C[8 * ldc + 4], _C8_1);
    C[8 * ldc + 6] = _C8_2;

    _mm256_storeu_pd(&C[9 * ldc + 0], _C9_0);
    _mm_storeu_pd(&C[9 * ldc + 4], _C9_1);
    C[9 * ldc + 6] = _C9_2;

    _mm256_storeu_pd(&C[10 * ldc + 0], _C10_0);
    _mm_storeu_pd(&C[10 * ldc + 4], _C10_1);
    C[10 * ldc + 6] = _C10_2;

    _mm256_storeu_pd(&C[11 * ldc + 0], _C11_0);
    _mm_storeu_pd(&C[11 * ldc + 4], _C11_1);
    C[11 * ldc + 6] = _C11_2;

    _mm256_storeu_pd(&C[12 * ldc + 0], _C12_0);
    _mm_storeu_pd(&C[12 * ldc + 4], _C12_1);
    C[12 * ldc + 6] = _C12_2;

    _mm256_storeu_pd(&C[13 * ldc + 0], _C13_0);
    _mm_storeu_pd(&C[13 * ldc + 4], _C13_1);
    C[13 * ldc + 6] = _C13_2;
}
//=====================================================================================================================
void micro_kernel8x14(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);
    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);

    register __m512d _C7_0 = _mm512_loadu_pd(&C[7 * ldc + 0]);

    register __m512d _C8_0 = _mm512_loadu_pd(&C[8 * ldc + 0]);

    register __m512d _C9_0 = _mm512_loadu_pd(&C[9 * ldc + 0]);

    register __m512d _C10_0 = _mm512_loadu_pd(&C[10 * ldc + 0]);

    register __m512d _C11_0 = _mm512_loadu_pd(&C[11 * ldc + 0]);

    register __m512d _C12_0 = _mm512_loadu_pd(&C[12 * ldc + 0]);

    register __m512d _C13_0 = _mm512_loadu_pd(&C[13 * ldc + 0]);

    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);

        _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A0, _C7_0);

        _C8_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8 * ldb]), _A0, _C8_0);

        _C9_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[9 * ldb]), _A0, _C9_0);

        _C10_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[10 * ldb]), _A0, _C10_0);

        _C11_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[11 * ldb]), _A0, _C11_0);

        _C12_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[12 * ldb]), _A0, _C12_0);

        _C13_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[13 * ldb]), _A0, _C13_0);

        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);

    _mm512_storeu_pd(&C[7 * ldc + 0], _C7_0);

    _mm512_storeu_pd(&C[8 * ldc + 0], _C8_0);

    _mm512_storeu_pd(&C[9 * ldc + 0], _C9_0);

    _mm512_storeu_pd(&C[10 * ldc + 0], _C10_0);

    _mm512_storeu_pd(&C[11 * ldc + 0], _C11_0);

    _mm512_storeu_pd(&C[12 * ldc + 0], _C12_0);

    _mm512_storeu_pd(&C[13 * ldc + 0], _C13_0);
}
void micro_kernel9x14(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register double _A1;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register double _C0_1 = C[0 * ldc + 8];

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);
    register double _C1_1 = C[1 * ldc + 8];

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register double _C2_1 = C[2 * ldc + 8];

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register double _C3_1 = C[3 * ldc + 8];

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register double _C4_1 = C[4 * ldc + 8];

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register double _C5_1 = C[5 * ldc + 8];

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);
    register double _C6_1 = C[6 * ldc + 8];

    register __m512d _C7_0 = _mm512_loadu_pd(&C[7 * ldc + 0]);
    register double _C7_1 = C[7 * ldc + 8];

    register __m512d _C8_0 = _mm512_loadu_pd(&C[8 * ldc + 0]);
    register double _C8_1 = C[8 * ldc + 8];

    register __m512d _C9_0 = _mm512_loadu_pd(&C[9 * ldc + 0]);
    register double _C9_1 = C[9 * ldc + 8];

    register __m512d _C10_0 = _mm512_loadu_pd(&C[10 * ldc + 0]);
    register double _C10_1 = C[10 * ldc + 8];

    register __m512d _C11_0 = _mm512_loadu_pd(&C[11 * ldc + 0]);
    register double _C11_1 = C[11 * ldc + 8];

    register __m512d _C12_0 = _mm512_loadu_pd(&C[12 * ldc + 0]);
    register double _C12_1 = C[12 * ldc + 8];

    register __m512d _C13_0 = _mm512_loadu_pd(&C[13 * ldc + 0]);
    register double _C13_1 = C[13 * ldc + 8];

    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = A[8];

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 += B[0 * ldb] * _A1;

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 += B[1 * ldb] * _A1;

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 += B[2 * ldb] * _A1;

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 += B[3 * ldb] * _A1;

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 += B[4 * ldb] * _A1;

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 += B[5 * ldb] * _A1;

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 += B[6 * ldb] * _A1;
        
        _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 += B[7 * ldb] * _A1;

        _C8_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 += B[8 * ldb] * _A1;

        _C9_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[9 * ldb]), _A0, _C9_0);
        _C9_1 += B[9 * ldb] * _A1;

        _C10_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[10 * ldb]), _A0, _C10_0);
        _C10_1 += B[10 * ldb] * _A1;

        _C11_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[11 * ldb]), _A0, _C11_0);
        _C11_1 += B[11 * ldb] * _A1;

        _C12_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[12 * ldb]), _A0, _C12_0);
        _C12_1 += B[12 * ldb] * _A1;

        _C13_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[13 * ldb]), _A0, _C13_0);
        _C13_1 += B[13 * ldb] * _A1;

        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    C[0 * ldc + 8] = _C0_1;

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    C[1 * ldc + 8] = _C1_1;

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    C[2 * ldc + 8] = _C2_1;

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    C[3 * ldc + 8] = _C3_1;

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    C[4 * ldc + 8] = _C4_1;

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    C[5 * ldc + 8] = _C5_1;

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);
    C[6 * ldc + 8] = _C6_1;

    _mm512_storeu_pd(&C[7 * ldc + 0], _C7_0);
    C[7 * ldc + 8] = _C7_1;

    _mm512_storeu_pd(&C[8 * ldc + 0], _C8_0);
    C[8 * ldc + 8] = _C8_1;

    _mm512_storeu_pd(&C[9 * ldc + 0], _C9_0);
    C[9 * ldc + 8] = _C9_1;

    _mm512_storeu_pd(&C[10 * ldc + 0], _C10_0);
    C[10 * ldc + 8] = _C10_1;

    _mm512_storeu_pd(&C[11 * ldc + 0], _C11_0);
    C[11 * ldc + 8] = _C11_1;

    _mm512_storeu_pd(&C[12 * ldc + 0], _C12_0);
    C[12 * ldc + 8] = _C12_1;

    _mm512_storeu_pd(&C[13 * ldc + 0], _C13_0);
    C[13 * ldc + 8] = _C13_1;
}
void micro_kernel10x14(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m128d _A1;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C0_1 = _mm_loadu_pd(&C[0 * ldc + 8]);

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m128d _C1_1 = _mm_loadu_pd(&C[1 * ldc + 8]);

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m128d _C2_1 = _mm_loadu_pd(&C[2 * ldc + 8]);

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m128d _C3_1 = _mm_loadu_pd(&C[3 * ldc + 8]);

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m128d _C4_1 = _mm_loadu_pd(&C[4 * ldc + 8]);

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register __m128d _C5_1 = _mm_loadu_pd(&C[5 * ldc + 8]);

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);
    register __m128d _C6_1 = _mm_loadu_pd(&C[6 * ldc + 8]);

    register __m512d _C7_0 = _mm512_loadu_pd(&C[7 * ldc + 0]);
    register __m128d _C7_1 = _mm_loadu_pd(&C[7 * ldc + 8]);

    register __m512d _C8_0 = _mm512_loadu_pd(&C[8 * ldc + 0]);
    register __m128d _C8_1 = _mm_loadu_pd(&C[8 * ldc + 8]);

    register __m512d _C9_0 = _mm512_loadu_pd(&C[9 * ldc + 0]);
    register __m128d _C9_1 = _mm_loadu_pd(&C[9 * ldc + 8]);

    register __m512d _C10_0 = _mm512_loadu_pd(&C[10 * ldc + 0]);
    register __m128d _C10_1 = _mm_loadu_pd(&C[10 * ldc + 8]);

    register __m512d _C11_0 = _mm512_loadu_pd(&C[11 * ldc + 0]);
    register __m128d _C11_1 = _mm_loadu_pd(&C[11 * ldc + 8]);

    register __m512d _C12_0 = _mm512_loadu_pd(&C[12 * ldc + 0]);
    register __m128d _C12_1 = _mm_loadu_pd(&C[12 * ldc + 8]);

    register __m512d _C13_0 = _mm512_loadu_pd(&C[13 * ldc + 0]);
    register __m128d _C13_1 = _mm_loadu_pd(&C[13 * ldc + 8]);
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm_loadu_pd(&A[8]);

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A1, _C0_1);

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A1, _C1_1);

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A1, _C2_1);

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A1, _C3_1);

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A1, _C4_1);

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A1, _C5_1);

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm_fnmadd_pd(_mm_set1_pd(B[6 * ldb]), _A1, _C6_1);

        _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 = _mm_fnmadd_pd(_mm_set1_pd(B[7 * ldb]), _A1, _C7_1);

        _C8_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 = _mm_fnmadd_pd(_mm_set1_pd(B[8 * ldb]), _A1, _C8_1);

        _C9_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[9 * ldb]), _A0, _C9_0);
        _C9_1 = _mm_fnmadd_pd(_mm_set1_pd(B[9 * ldb]), _A1, _C9_1);

        _C10_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[10 * ldb]), _A0, _C10_0);
        _C10_1 = _mm_fnmadd_pd(_mm_set1_pd(B[10 * ldb]), _A1, _C10_1);

        _C11_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[11 * ldb]), _A0, _C11_0);
        _C11_1 = _mm_fnmadd_pd(_mm_set1_pd(B[11 * ldb]), _A1, _C11_1);

        _C12_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[12 * ldb]), _A0, _C12_0);
        _C12_1 = _mm_fnmadd_pd(_mm_set1_pd(B[12 * ldb]), _A1, _C12_1);

        _C13_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[13 * ldb]), _A0, _C13_0);
        _C13_1 = _mm_fnmadd_pd(_mm_set1_pd(B[13 * ldb]), _A1, _C13_1);
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[0 * ldc + 8], _C0_1);

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm_storeu_pd(&C[1 * ldc + 8], _C1_1);

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm_storeu_pd(&C[2 * ldc + 8], _C2_1);

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm_storeu_pd(&C[3 * ldc + 8], _C3_1);

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm_storeu_pd(&C[4 * ldc + 8], _C4_1);

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm_storeu_pd(&C[5 * ldc + 8], _C5_1);

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm_storeu_pd(&C[6 * ldc + 8], _C6_1);

    _mm512_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm_storeu_pd(&C[7 * ldc + 8], _C7_1);

    _mm512_storeu_pd(&C[8 * ldc + 0], _C8_0);
    _mm_storeu_pd(&C[8 * ldc + 8], _C8_1);

    _mm512_storeu_pd(&C[9 * ldc + 0], _C9_0);
    _mm_storeu_pd(&C[9 * ldc + 8], _C9_1);

    _mm512_storeu_pd(&C[10 * ldc + 0], _C10_0);
    _mm_storeu_pd(&C[10 * ldc + 8], _C10_1);

    _mm512_storeu_pd(&C[11 * ldc + 0], _C11_0);
    _mm_storeu_pd(&C[11 * ldc + 8], _C11_1);

    _mm512_storeu_pd(&C[12 * ldc + 0], _C12_0);
    _mm_storeu_pd(&C[12 * ldc + 8], _C12_1);

    _mm512_storeu_pd(&C[13 * ldc + 0], _C13_0);
    _mm_storeu_pd(&C[13 * ldc + 8], _C13_1);
}
void micro_kernel11x14(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m128d _A1;
    register double _A2;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m128d _C0_1 = _mm_loadu_pd(&C[0 * ldc + 8]);
    register double _C0_2 = C[0 * ldc + 10];

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m128d _C1_1 = _mm_loadu_pd(&C[1 * ldc + 8]);
    register double _C1_2 = C[1 * ldc + 10];

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m128d _C2_1 = _mm_loadu_pd(&C[2 * ldc + 8]);
    register double _C2_2 = C[2 * ldc + 10];

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m128d _C3_1 = _mm_loadu_pd(&C[3 * ldc + 8]);
    register double _C3_2 = C[3 * ldc + 10];

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m128d _C4_1 = _mm_loadu_pd(&C[4 * ldc + 8]);
    register double _C4_2 = C[4 * ldc + 10];

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register __m128d _C5_1 = _mm_loadu_pd(&C[5 * ldc + 8]);
    register double _C5_2 = C[5 * ldc + 10];

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);
    register __m128d _C6_1 = _mm_loadu_pd(&C[6 * ldc + 8]);
    register double _C6_2 = C[6 * ldc + 10];

    register __m512d _C7_0 = _mm512_loadu_pd(&C[7 * ldc + 0]);
    register __m128d _C7_1 = _mm_loadu_pd(&C[7 * ldc + 8]);
    register double _C7_2 = C[7 * ldc + 10];

    register __m512d _C8_0 = _mm512_loadu_pd(&C[8 * ldc + 0]);
    register __m128d _C8_1 = _mm_loadu_pd(&C[8 * ldc + 8]);
    register double _C8_2 = C[8 * ldc + 10];

    register __m512d _C9_0 = _mm512_loadu_pd(&C[9 * ldc + 0]);
    register __m128d _C9_1 = _mm_loadu_pd(&C[9 * ldc + 8]);
    register double _C9_2 = C[9 * ldc + 10];

    register __m512d _C10_0 = _mm512_loadu_pd(&C[10 * ldc + 0]);
    register __m128d _C10_1 = _mm_loadu_pd(&C[10 * ldc + 8]);
    register double _C10_2 = C[10 * ldc + 10];
    
    register __m512d _C11_0 = _mm512_loadu_pd(&C[11 * ldc + 0]);
    register __m128d _C11_1 = _mm_loadu_pd(&C[11 * ldc + 8]);
    register double _C11_2 = C[11 * ldc + 10];

    register __m512d _C12_0 = _mm512_loadu_pd(&C[12 * ldc + 0]);
    register __m128d _C12_1 = _mm_loadu_pd(&C[12 * ldc + 8]);
    register double _C12_2 = C[12 * ldc + 10];

    register __m512d _C13_0 = _mm512_loadu_pd(&C[13 * ldc + 0]);
    register __m128d _C13_1 = _mm_loadu_pd(&C[13 * ldc + 8]);
    register double _C13_2 = C[13 * ldc + 10];

    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm_loadu_pd(&A[8]);
        _A2 = A[10];

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 += B[0 * ldb] * _A2;

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C1_2 += B[1 * ldb] * _A2;

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C2_2 += B[2 * ldb] * _A2;

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A1, _C3_1);
        _C3_2 += B[3 * ldb] * _A2;

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A1, _C4_1);
        _C4_2 += B[4 * ldb] * _A2;

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A1, _C5_1);
        _C5_2 += B[5 * ldb] * _A2;

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm_fnmadd_pd(_mm_set1_pd(B[6 * ldb]), _A1, _C6_1);
        _C6_2 += B[6 * ldb] * _A2;

        _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 = _mm_fnmadd_pd(_mm_set1_pd(B[7 * ldb]), _A1, _C7_1);
        _C7_2 += B[7 * ldb] * _A2;

        _C8_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 = _mm_fnmadd_pd(_mm_set1_pd(B[8 * ldb]), _A1, _C8_1);
        _C8_2 += B[8 * ldb] * _A2;

        _C9_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[9 * ldb]), _A0, _C9_0);
        _C9_1 = _mm_fnmadd_pd(_mm_set1_pd(B[9 * ldb]), _A1, _C9_1);
        _C9_2 += B[9 * ldb] * _A2;

        _C10_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[10 * ldb]), _A0, _C10_0);
        _C10_1 = _mm_fnmadd_pd(_mm_set1_pd(B[10 * ldb]), _A1, _C10_1);
        _C10_2 += B[10 * ldb] * _A2;
        
        _C11_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[11 * ldb]), _A0, _C11_0);
        _C11_1 = _mm_fnmadd_pd(_mm_set1_pd(B[11 * ldb]), _A1, _C11_1);
        _C11_2 += B[11 * ldb] * _A2;

        _C12_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[12 * ldb]), _A0, _C12_0);
        _C12_1 = _mm_fnmadd_pd(_mm_set1_pd(B[12 * ldb]), _A1, _C12_1);
        _C12_2 += B[12 * ldb] * _A2; 

        _C13_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[13 * ldb]), _A0, _C13_0);
        _C13_1 = _mm_fnmadd_pd(_mm_set1_pd(B[13 * ldb]), _A1, _C13_1);
        _C13_2 += B[13 * ldb] * _A2; 

        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm_storeu_pd(&C[0 * ldc + 8], _C0_1);
    C[0 * ldc + 10] = _C0_2;

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm_storeu_pd(&C[1 * ldc + 8], _C1_1);
    C[1 * ldc + 10] = _C1_2;

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm_storeu_pd(&C[2 * ldc + 8], _C2_1);
    C[2 * ldc + 10] = _C2_2;

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm_storeu_pd(&C[3 * ldc + 8], _C3_1);
    C[3 * ldc + 10] = _C3_2;

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm_storeu_pd(&C[4 * ldc + 8], _C4_1);
    C[4 * ldc + 10] = _C4_2;

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm_storeu_pd(&C[5 * ldc + 8], _C5_1);
    C[5 * ldc + 10] = _C5_2;

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm_storeu_pd(&C[6 * ldc + 8], _C6_1);
    C[6 * ldc + 10] = _C6_2;

    _mm512_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm_storeu_pd(&C[7 * ldc + 8], _C7_1);
    C[7 * ldc + 10] = _C7_2;

    _mm512_storeu_pd(&C[8 * ldc + 0], _C8_0);
    _mm_storeu_pd(&C[8 * ldc + 8], _C8_1);
    C[8 * ldc + 10] = _C8_2;

    _mm512_storeu_pd(&C[9 * ldc + 0], _C9_0);
    _mm_storeu_pd(&C[9 * ldc + 8], _C9_1);
    C[9 * ldc + 10] = _C9_2;

    _mm512_storeu_pd(&C[10 * ldc + 0], _C10_0);
    _mm_storeu_pd(&C[10 * ldc + 8], _C10_1);
    C[10 * ldc + 10] = _C10_2;

    _mm512_storeu_pd(&C[11 * ldc + 0], _C11_0);
    _mm_storeu_pd(&C[11 * ldc + 8], _C11_1);
    C[11 * ldc + 10] = _C11_2;

    _mm512_storeu_pd(&C[12 * ldc + 0], _C12_0);
    _mm_storeu_pd(&C[12 * ldc + 8], _C12_1);
    C[12 * ldc + 10] = _C12_2;

    _mm512_storeu_pd(&C[13 * ldc + 0], _C13_0);
    _mm_storeu_pd(&C[13 * ldc + 8], _C13_1);
    C[13 * ldc + 10] = _C13_2;
}
void micro_kernel12x14(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {//------------------------------------
     int i;
    register __m512d _A0;
    register __m256d _A1;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C0_1 = _mm256_loadu_pd(&C[0 * ldc + 8]);

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m256d _C1_1 = _mm256_loadu_pd(&C[1 * ldc + 8]);

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m256d _C2_1 = _mm256_loadu_pd(&C[2 * ldc + 8]);

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m256d _C3_1 = _mm256_loadu_pd(&C[3 * ldc + 8]);

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m256d _C4_1 = _mm256_loadu_pd(&C[4 * ldc + 8]);

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register __m256d _C5_1 = _mm256_loadu_pd(&C[5 * ldc + 8]);

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);
    register __m256d _C6_1 = _mm256_loadu_pd(&C[6 * ldc + 8]);

    register __m512d _C7_0 = _mm512_loadu_pd(&C[7 * ldc + 0]);
    register __m256d _C7_1 = _mm256_loadu_pd(&C[7 * ldc + 8]);

    register __m512d _C8_0 = _mm512_loadu_pd(&C[8 * ldc + 0]);
    register __m256d _C8_1 = _mm256_loadu_pd(&C[8 * ldc + 8]);

    register __m512d _C9_0 = _mm512_loadu_pd(&C[9 * ldc + 0]);
    register __m256d _C9_1 = _mm256_loadu_pd(&C[9 * ldc + 8]);

    register __m512d _C10_0 = _mm512_loadu_pd(&C[10 * ldc + 0]);
    register __m256d _C10_1 = _mm256_loadu_pd(&C[10 * ldc + 8]);

    register __m512d _C11_0 = _mm512_loadu_pd(&C[11 * ldc + 0]);
    register __m256d _C11_1 = _mm256_loadu_pd(&C[11 * ldc + 8]);

    register __m512d _C12_0 = _mm512_loadu_pd(&C[12 * ldc + 0]);
    register __m256d _C12_1 = _mm256_loadu_pd(&C[12 * ldc + 8]);

    register __m512d _C13_0 = _mm512_loadu_pd(&C[13 * ldc + 0]);
    register __m256d _C13_1 = _mm256_loadu_pd(&C[13 * ldc + 8]);
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm256_loadu_pd(&A[8]);

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A1, _C0_1);

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A1, _C1_1);

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A1, _C2_1);

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A1, _C3_1);

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A1, _C4_1);

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A1, _C5_1);

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[6 * ldb]), _A1, _C6_1);

        _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[7 * ldb]), _A1, _C7_1);

        _C8_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[8 * ldb]), _A1, _C8_1);

        _C9_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[9 * ldb]), _A0, _C9_0);
        _C9_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[9 * ldb]), _A1, _C9_1);

        _C10_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[10 * ldb]), _A0, _C10_0);
        _C10_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[10 * ldb]), _A1, _C10_1);

        _C11_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[11 * ldb]), _A0, _C11_0);
        _C11_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[11 * ldb]), _A1, _C11_1);

        _C12_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[12 * ldb]), _A0, _C12_0);
        _C12_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[12 * ldb]), _A1, _C12_1);

        _C13_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[13 * ldb]), _A0, _C13_0);
        _C13_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[13 * ldb]), _A1, _C13_1);
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[0 * ldc + 8], _C0_1);

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm256_storeu_pd(&C[1 * ldc + 8], _C1_1);

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm256_storeu_pd(&C[2 * ldc + 8], _C2_1);

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm256_storeu_pd(&C[3 * ldc + 8], _C3_1);

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm256_storeu_pd(&C[4 * ldc + 8], _C4_1);

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm256_storeu_pd(&C[5 * ldc + 8], _C5_1);

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm256_storeu_pd(&C[6 * ldc + 8], _C6_1);

    _mm512_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm256_storeu_pd(&C[7 * ldc + 8], _C7_1);

    _mm512_storeu_pd(&C[8 * ldc + 0], _C8_0);
    _mm256_storeu_pd(&C[8 * ldc + 8], _C8_1);

    _mm512_storeu_pd(&C[9 * ldc + 0], _C9_0);
    _mm256_storeu_pd(&C[9 * ldc + 8], _C9_1);

    _mm512_storeu_pd(&C[10 * ldc + 0], _C10_0);
    _mm256_storeu_pd(&C[10 * ldc + 8], _C10_1);

    _mm512_storeu_pd(&C[11 * ldc + 0], _C11_0);
    _mm256_storeu_pd(&C[11 * ldc + 8], _C11_1);

    _mm512_storeu_pd(&C[12 * ldc + 0], _C12_0);
    _mm256_storeu_pd(&C[12 * ldc + 8], _C12_1);

    _mm512_storeu_pd(&C[13 * ldc + 0], _C13_0);
    _mm256_storeu_pd(&C[13 * ldc + 8], _C13_1);
}
void micro_kernel13x14(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m256d _A1;
    register double _A2;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C0_1 = _mm256_loadu_pd(&C[0 * ldc + 8]);
    register double _C0_2 = C[0 * ldc + 12];

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m256d _C1_1 = _mm256_loadu_pd(&C[1 * ldc + 8]);
    register double _C1_2 = C[1 * ldc + 12];

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m256d _C2_1 = _mm256_loadu_pd(&C[2 * ldc + 8]);
    register double _C2_2 = C[2 * ldc + 12];

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m256d _C3_1 = _mm256_loadu_pd(&C[3 * ldc + 8]);
    register double _C3_2 = C[3 * ldc + 12];

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m256d _C4_1 = _mm256_loadu_pd(&C[4 * ldc + 8]);
    register double _C4_2 = C[4 * ldc + 12];

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register __m256d _C5_1 = _mm256_loadu_pd(&C[5 * ldc + 8]);
    register double _C5_2 = C[5 * ldc + 12];

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);
    register __m256d _C6_1 = _mm256_loadu_pd(&C[6 * ldc + 8]);
    register double _C6_2 = C[6 * ldc + 12];

    register __m512d _C7_0 = _mm512_loadu_pd(&C[7 * ldc + 0]);
    register __m256d _C7_1 = _mm256_loadu_pd(&C[7 * ldc + 8]);
    register double _C7_2 = C[7 * ldc + 12];

    register __m512d _C8_0 = _mm512_loadu_pd(&C[8 * ldc + 0]);
    register __m256d _C8_1 = _mm256_loadu_pd(&C[8 * ldc + 8]);
    register double _C8_2 = C[8 * ldc + 12];

    register __m512d _C9_0 = _mm512_loadu_pd(&C[9 * ldc + 0]);
    register __m256d _C9_1 = _mm256_loadu_pd(&C[9 * ldc + 8]);
    register double _C9_2 = C[9 * ldc + 12];

    register __m512d _C10_0 = _mm512_loadu_pd(&C[10 * ldc + 0]);
    register __m256d _C10_1 = _mm256_loadu_pd(&C[10 * ldc + 8]);
    register double _C10_2 = C[10 * ldc + 12];

    register __m512d _C11_0 = _mm512_loadu_pd(&C[11 * ldc + 0]);
    register __m256d _C11_1 = _mm256_loadu_pd(&C[11 * ldc + 8]);
    register double _C11_2 = C[11 * ldc + 12];

    register __m512d _C12_0 = _mm512_loadu_pd(&C[12 * ldc + 0]);
    register __m256d _C12_1 = _mm256_loadu_pd(&C[12 * ldc + 8]);
    register double _C12_2 = C[12 * ldc + 12];

    register __m512d _C13_0 = _mm512_loadu_pd(&C[13 * ldc + 0]);
    register __m256d _C13_1 = _mm256_loadu_pd(&C[13 * ldc + 8]);
    register double _C13_2 = C[13 * ldc + 12];
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm256_loadu_pd(&A[8]);
        _A2 = A[12];

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 += B[0 * ldb] * _A2;

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C1_2 += B[1 * ldb] * _A2;

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C2_2 += B[2 * ldb] * _A2;

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A1, _C3_1);
        _C3_2 += B[3 * ldb] * _A2;

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A1, _C4_1);
        _C4_2 += B[4 * ldb] * _A2;

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A1, _C5_1);
        _C5_2 += B[5 * ldb] * _A2;

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[6 * ldb]), _A1, _C6_1);
        _C6_2 += B[6 * ldb] * _A2;

        _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[7 * ldb]), _A1, _C7_1);
        _C7_2 += B[7 * ldb] * _A2;

        _C8_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[8 * ldb]), _A1, _C8_1);
        _C8_2 += B[8 * ldb] * _A2;

        _C9_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[9 * ldb]), _A0, _C9_0);
        _C9_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[9 * ldb]), _A1, _C9_1);
        _C9_2 += B[9 * ldb] * _A2;

        _C10_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[10 * ldb]), _A0, _C10_0);
        _C10_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[10 * ldb]), _A1, _C10_1);
        _C10_2 += B[10 * ldb] * _A2;

        _C11_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[11 * ldb]), _A0, _C11_0);
        _C11_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[11 * ldb]), _A1, _C11_1);
        _C11_2 += B[11 * ldb] * _A2;

        _C12_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[12 * ldb]), _A0, _C12_0);
        _C12_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[12 * ldb]), _A1, _C12_1);
        _C12_2 += B[12 * ldb] * _A2;

        _C13_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[13 * ldb]), _A0, _C13_0);
        _C13_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[13 * ldb]), _A1, _C13_1);
        _C13_2 += B[13 * ldb] * _A2;
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[0 * ldc + 8], _C0_1);
    C[0 * ldc + 12] = _C0_2;

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm256_storeu_pd(&C[1 * ldc + 8], _C1_1);
    C[1 * ldc + 12] = _C1_2;

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm256_storeu_pd(&C[2 * ldc + 8], _C2_1);
    C[2 * ldc + 12] = _C2_2;

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm256_storeu_pd(&C[3 * ldc + 8], _C3_1);
    C[3 * ldc + 12] = _C3_2;

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm256_storeu_pd(&C[4 * ldc + 8], _C4_1);
    C[4 * ldc + 12] = _C4_2;

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm256_storeu_pd(&C[5 * ldc + 8], _C5_1);
    C[5 * ldc + 12] = _C5_2;

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm256_storeu_pd(&C[6 * ldc + 8], _C6_1);
    C[6 * ldc + 12] = _C6_2;

    _mm512_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm256_storeu_pd(&C[7 * ldc + 8], _C7_1);
    C[7 * ldc + 12] = _C7_2;

    _mm512_storeu_pd(&C[8 * ldc + 0], _C8_0);
    _mm256_storeu_pd(&C[8 * ldc + 8], _C8_1);
    C[8 * ldc + 12] = _C8_2;

    _mm512_storeu_pd(&C[9 * ldc + 0], _C9_0);
    _mm256_storeu_pd(&C[9 * ldc + 8], _C9_1);
    C[9 * ldc + 12] = _C9_2;

    _mm512_storeu_pd(&C[10 * ldc + 0], _C10_0);
    _mm256_storeu_pd(&C[10 * ldc + 8], _C10_1);
    C[10 * ldc + 12] = _C10_2;

    _mm512_storeu_pd(&C[11 * ldc + 0], _C11_0);
    _mm256_storeu_pd(&C[11 * ldc + 8], _C11_1);
    C[11 * ldc + 12] = _C11_2;

    _mm512_storeu_pd(&C[12 * ldc + 0], _C12_0);
    _mm256_storeu_pd(&C[12 * ldc + 8], _C12_1);
    C[12 * ldc + 12] = _C12_2;

    _mm512_storeu_pd(&C[13 * ldc + 0], _C13_0);
    _mm256_storeu_pd(&C[13 * ldc + 8], _C13_1);
    C[13 * ldc + 12] = _C13_2;
}
void micro_kernel14x14(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m256d _A1;
    register __m128d _A2;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C0_1 = _mm256_loadu_pd(&C[0 * ldc + 8]);
    register __m128d _C0_2 = _mm_loadu_pd(&C[0 * ldc + 12]);

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m256d _C1_1 = _mm256_loadu_pd(&C[1 * ldc + 8]);
    register __m128d _C1_2 = _mm_loadu_pd(&C[1 * ldc + 12]);

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m256d _C2_1 = _mm256_loadu_pd(&C[2 * ldc + 8]);
    register __m128d _C2_2 = _mm_loadu_pd(&C[2 * ldc + 12]);

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m256d _C3_1 = _mm256_loadu_pd(&C[3 * ldc + 8]);
    register __m128d _C3_2 = _mm_loadu_pd(&C[3 * ldc + 12]);

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m256d _C4_1 = _mm256_loadu_pd(&C[4 * ldc + 8]);
    register __m128d _C4_2 = _mm_loadu_pd(&C[4 * ldc + 12]);

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register __m256d _C5_1 = _mm256_loadu_pd(&C[5 * ldc + 8]);
    register __m128d _C5_2 = _mm_loadu_pd(&C[5 * ldc + 12]);

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);
    register __m256d _C6_1 = _mm256_loadu_pd(&C[6 * ldc + 8]);
    register __m128d _C6_2 = _mm_loadu_pd(&C[6 * ldc + 12]);

    register __m512d _C7_0 = _mm512_loadu_pd(&C[7 * ldc + 0]);
    register __m256d _C7_1 = _mm256_loadu_pd(&C[7 * ldc + 8]);
    register __m128d _C7_2 = _mm_loadu_pd(&C[7 * ldc + 12]);

    register __m512d _C8_0 = _mm512_loadu_pd(&C[8 * ldc + 0]);
    register __m256d _C8_1 = _mm256_loadu_pd(&C[8 * ldc + 8]);
    register __m128d _C8_2 = _mm_loadu_pd(&C[8 * ldc + 12]);

    register __m512d _C9_0 = _mm512_loadu_pd(&C[9 * ldc + 0]);
    register __m256d _C9_1 = _mm256_loadu_pd(&C[9 * ldc + 8]);
    register __m128d _C9_2 = _mm_loadu_pd(&C[9 * ldc + 12]);

    register __m512d _C10_0 = _mm512_loadu_pd(&C[10 * ldc + 0]);
    register __m256d _C10_1 = _mm256_loadu_pd(&C[10 * ldc + 8]);
    register __m128d _C10_2 = _mm_loadu_pd(&C[10 * ldc + 12]);

    register __m512d _C11_0 = _mm512_loadu_pd(&C[11 * ldc + 0]);
    register __m256d _C11_1 = _mm256_loadu_pd(&C[11 * ldc + 8]);
    register __m128d _C11_2 = _mm_loadu_pd(&C[11 * ldc + 12]);

    register __m512d _C12_0 = _mm512_loadu_pd(&C[12 * ldc + 0]);
    register __m256d _C12_1 = _mm256_loadu_pd(&C[12 * ldc + 8]);
    register __m128d _C12_2 = _mm_loadu_pd(&C[12 * ldc + 12]);

    register __m512d _C13_0 = _mm512_loadu_pd(&C[13 * ldc + 0]);
    register __m256d _C13_1 = _mm256_loadu_pd(&C[13 * ldc + 8]);
    register __m128d _C13_2 = _mm_loadu_pd(&C[13 * ldc + 12]);

    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm256_loadu_pd(&A[8]);
        _A2 = _mm_loadu_pd(&A[12]);

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A2, _C0_2);

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C1_2 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A2, _C1_2);

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C2_2 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A2, _C2_2);

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A1, _C3_1);
        _C3_2 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A2, _C3_2);

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A1, _C4_1);
        _C4_2 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A2, _C4_2);

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A1, _C5_1);
        _C5_2 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A2, _C5_2);

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[6 * ldb]), _A1, _C6_1);
        _C6_2 = _mm_fnmadd_pd(_mm_set1_pd(B[6 * ldb]), _A2, _C6_2);

        _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[7 * ldb]), _A1, _C7_1);
        _C7_2 = _mm_fnmadd_pd(_mm_set1_pd(B[7 * ldb]), _A2, _C7_2);

        _C8_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[8 * ldb]), _A1, _C8_1);
        _C8_2 = _mm_fnmadd_pd(_mm_set1_pd(B[8 * ldb]), _A2, _C8_2);

        _C9_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[9 * ldb]), _A0, _C9_0);
        _C9_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[9 * ldb]), _A1, _C9_1);
        _C9_2 = _mm_fnmadd_pd(_mm_set1_pd(B[9 * ldb]), _A2, _C9_2);

        _C10_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[10 * ldb]), _A0, _C10_0);
        _C10_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[10 * ldb]), _A1, _C10_1);
        _C10_2 = _mm_fnmadd_pd(_mm_set1_pd(B[10 * ldb]), _A2, _C10_2);
 

        _C11_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[11 * ldb]), _A0, _C11_0);
        _C11_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[11 * ldb]), _A1, _C11_1);
        _C11_2 = _mm_fnmadd_pd(_mm_set1_pd(B[11 * ldb]), _A2, _C11_2);

        _C12_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[12 * ldb]), _A0, _C12_0);
        _C12_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[12 * ldb]), _A1, _C12_1);
        _C12_2 = _mm_fnmadd_pd(_mm_set1_pd(B[12 * ldb]), _A2, _C12_2);

        _C13_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[13 * ldb]), _A0, _C13_0);
        _C13_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[13 * ldb]), _A1, _C13_1);
        _C13_2 = _mm_fnmadd_pd(_mm_set1_pd(B[13 * ldb]), _A2, _C13_2);

        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[0 * ldc + 8], _C0_1);
    _mm_storeu_pd(&C[0 * ldc + 12], _C0_2);

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm256_storeu_pd(&C[1 * ldc + 8], _C1_1);
    _mm_storeu_pd(&C[1 * ldc + 12], _C1_2);

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm256_storeu_pd(&C[2 * ldc + 8], _C2_1);
    _mm_storeu_pd(&C[2 * ldc + 12], _C2_2);

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm256_storeu_pd(&C[3 * ldc + 8], _C3_1);
    _mm_storeu_pd(&C[3 * ldc + 12], _C3_2);

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm256_storeu_pd(&C[4 * ldc + 8], _C4_1);
    _mm_storeu_pd(&C[4 * ldc + 12], _C4_2);

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm256_storeu_pd(&C[5 * ldc + 8], _C5_1);
    _mm_storeu_pd(&C[5 * ldc + 12], _C5_2);

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm256_storeu_pd(&C[6 * ldc + 8], _C6_1);
    _mm_storeu_pd(&C[6 * ldc + 12], _C6_2);

    _mm512_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm256_storeu_pd(&C[7 * ldc + 8], _C7_1);
    _mm_storeu_pd(&C[7 * ldc + 12], _C7_2);

    _mm512_storeu_pd(&C[8 * ldc + 0], _C8_0);
    _mm256_storeu_pd(&C[8 * ldc + 8], _C8_1);
    _mm_storeu_pd(&C[8 * ldc + 12], _C8_2);

    _mm512_storeu_pd(&C[9 * ldc + 0], _C9_0);
    _mm256_storeu_pd(&C[9 * ldc + 8], _C9_1);
    _mm_storeu_pd(&C[9 * ldc + 12], _C9_2);

    _mm512_storeu_pd(&C[10 * ldc + 0], _C10_0);
    _mm256_storeu_pd(&C[10 * ldc + 8], _C10_1);
    _mm_storeu_pd(&C[10 * ldc + 12], _C10_2);

    _mm512_storeu_pd(&C[11 * ldc + 0], _C11_0);
    _mm256_storeu_pd(&C[11 * ldc + 8], _C11_1);
    _mm_storeu_pd(&C[11 * ldc + 12], _C11_2);

    _mm512_storeu_pd(&C[12 * ldc + 0], _C12_0);
    _mm256_storeu_pd(&C[12 * ldc + 8], _C12_1);
    _mm_storeu_pd(&C[12 * ldc + 12], _C12_2);

    _mm512_storeu_pd(&C[13 * ldc + 0], _C13_0);
    _mm256_storeu_pd(&C[13 * ldc + 8], _C13_1);
    _mm_storeu_pd(&C[13 * ldc + 12], _C13_2);
}
void micro_kernel15x14(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m256d _A1;
    register __m128d _A2;
    register double _A3;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m256d _C0_1 = _mm256_loadu_pd(&C[0 * ldc + 8]);
    register __m128d _C0_2 = _mm_loadu_pd(&C[0 * ldc + 12]);
    register double _C0_3 = C[0 * ldc + 14];

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m256d _C1_1 = _mm256_loadu_pd(&C[1 * ldc + 8]);
    register __m128d _C1_2 = _mm_loadu_pd(&C[1 * ldc + 12]);
    register double _C1_3 = C[1 * ldc + 14];

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m256d _C2_1 = _mm256_loadu_pd(&C[2 * ldc + 8]);
    register __m128d _C2_2 = _mm_loadu_pd(&C[2 * ldc + 12]);
    register double _C2_3 = C[2 * ldc + 14];

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m256d _C3_1 = _mm256_loadu_pd(&C[3 * ldc + 8]);
    register __m128d _C3_2 = _mm_loadu_pd(&C[3 * ldc + 12]);
    register double _C3_3 = C[3 * ldc + 14];

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m256d _C4_1 = _mm256_loadu_pd(&C[4 * ldc + 8]);
    register __m128d _C4_2 = _mm_loadu_pd(&C[4 * ldc + 12]);
    register double _C4_3 = C[4 * ldc + 14];

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register __m256d _C5_1 = _mm256_loadu_pd(&C[5 * ldc + 8]);
    register __m128d _C5_2 = _mm_loadu_pd(&C[5 * ldc + 12]);
    register double _C5_3 = C[5 * ldc + 14];

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);
    register __m256d _C6_1 = _mm256_loadu_pd(&C[6 * ldc + 8]);
    register __m128d _C6_2 = _mm_loadu_pd(&C[6 * ldc + 12]);
    register double _C6_3 = C[6 * ldc + 14];

    register __m512d _C7_0 = _mm512_loadu_pd(&C[7 * ldc + 0]);
    register __m256d _C7_1 = _mm256_loadu_pd(&C[7 * ldc + 8]);
    register __m128d _C7_2 = _mm_loadu_pd(&C[7 * ldc + 12]);
    register double _C7_3 = C[7 * ldc + 14];

    register __m512d _C8_0 = _mm512_loadu_pd(&C[8 * ldc + 0]);
    register __m256d _C8_1 = _mm256_loadu_pd(&C[8 * ldc + 8]);
    register __m128d _C8_2 = _mm_loadu_pd(&C[8 * ldc + 12]);
    register double _C8_3 = C[8 * ldc + 14];

    register __m512d _C9_0 = _mm512_loadu_pd(&C[9 * ldc + 0]);
    register __m256d _C9_1 = _mm256_loadu_pd(&C[9 * ldc + 8]);
    register __m128d _C9_2 = _mm_loadu_pd(&C[9 * ldc + 12]);
    register double _C9_3 = C[9 * ldc + 14];

    register __m512d _C10_0 = _mm512_loadu_pd(&C[10 * ldc + 0]);
    register __m256d _C10_1 = _mm256_loadu_pd(&C[10 * ldc + 8]);
    register __m128d _C10_2 = _mm_loadu_pd(&C[10 * ldc + 12]);
    register double _C10_3 = C[10 * ldc + 14];

    register __m512d _C11_0 = _mm512_loadu_pd(&C[11 * ldc + 0]);
    register __m256d _C11_1 = _mm256_loadu_pd(&C[11 * ldc + 8]);
    register __m128d _C11_2 = _mm_loadu_pd(&C[11 * ldc + 12]);
    register double _C11_3 = C[11 * ldc + 14];

    register __m512d _C12_0 = _mm512_loadu_pd(&C[12 * ldc + 0]);
    register __m256d _C12_1 = _mm256_loadu_pd(&C[12 * ldc + 8]);
    register __m128d _C12_2 = _mm_loadu_pd(&C[12 * ldc + 12]);
    register double _C12_3 = C[12 * ldc + 14];

    register __m512d _C13_0 = _mm512_loadu_pd(&C[13 * ldc + 0]);
    register __m256d _C13_1 = _mm256_loadu_pd(&C[13 * ldc + 8]);
    register __m128d _C13_2 = _mm_loadu_pd(&C[13 * ldc + 12]);
    register double _C13_3 = C[13 * ldc + 14];
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm256_loadu_pd(&A[8]);
        _A2 = _mm_loadu_pd(&A[12]);
        _A3 = A[14];

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[0 * ldb]), _A1, _C0_1);
        _C0_2 = _mm_fnmadd_pd(_mm_set1_pd(B[0 * ldb]), _A2, _C0_2);
        _C0_3 += B[0 * ldb] * _A3;

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[1 * ldb]), _A1, _C1_1);
        _C1_2 = _mm_fnmadd_pd(_mm_set1_pd(B[1 * ldb]), _A2, _C1_2);
        _C1_3 += B[1 * ldb] * _A3;

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[2 * ldb]), _A1, _C2_1);
        _C2_2 = _mm_fnmadd_pd(_mm_set1_pd(B[2 * ldb]), _A2, _C2_2);
        _C2_3 += B[2 * ldb] * _A3;

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[3 * ldb]), _A1, _C3_1);
        _C3_2 = _mm_fnmadd_pd(_mm_set1_pd(B[3 * ldb]), _A2, _C3_2);
        _C3_3 += B[3 * ldb] * _A3;

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[4 * ldb]), _A1, _C4_1);
        _C4_2 = _mm_fnmadd_pd(_mm_set1_pd(B[4 * ldb]), _A2, _C4_2);
        _C4_3 += B[4 * ldb] * _A3;

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[5 * ldb]), _A1, _C5_1);
        _C5_2 = _mm_fnmadd_pd(_mm_set1_pd(B[5 * ldb]), _A2, _C5_2);
        _C5_3 += B[5 * ldb] * _A3;

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[6 * ldb]), _A1, _C6_1);
        _C6_2 = _mm_fnmadd_pd(_mm_set1_pd(B[6 * ldb]), _A2, _C6_2);
        _C6_3 += B[6 * ldb] * _A3;

        _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[7 * ldb]), _A1, _C7_1);
        _C7_2 = _mm_fnmadd_pd(_mm_set1_pd(B[7 * ldb]), _A2, _C7_2);
        _C7_3 += B[7 * ldb] * _A3;

        _C8_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[8 * ldb]), _A1, _C8_1);
        _C8_2 = _mm_fnmadd_pd(_mm_set1_pd(B[8 * ldb]), _A2, _C8_2);
        _C8_3 += B[8 * ldb] * _A3;

        _C9_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[9 * ldb]), _A0, _C9_0);
        _C9_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[9 * ldb]), _A1, _C9_1);
        _C9_2 = _mm_fnmadd_pd(_mm_set1_pd(B[9 * ldb]), _A2, _C9_2);
        _C9_3 += B[9 * ldb] * _A3;

        _C10_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[10 * ldb]), _A0, _C10_0);
        _C10_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[10 * ldb]), _A1, _C10_1);
        _C10_2 = _mm_fnmadd_pd(_mm_set1_pd(B[10 * ldb]), _A2, _C10_2);
        _C10_3 += B[10 * ldb] * _A3;
        
        _C11_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[11 * ldb]), _A0, _C11_0);
        _C11_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[11 * ldb]), _A1, _C11_1);
        _C11_2 = _mm_fnmadd_pd(_mm_set1_pd(B[11 * ldb]), _A2, _C11_2);
        _C11_3 += B[11 * ldb] * _A3;
        
        _C12_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[12 * ldb]), _A0, _C12_0);
        _C12_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[12 * ldb]), _A1, _C12_1);
        _C12_2 = _mm_fnmadd_pd(_mm_set1_pd(B[12 * ldb]), _A2, _C12_2);
        _C12_3 += B[12 * ldb] * _A3;

        _C13_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[13 * ldb]), _A0, _C13_0);
        _C13_1 = _mm256_fnmadd_pd(_mm256_set1_pd(B[13 * ldb]), _A1, _C13_1);
        _C13_2 = _mm_fnmadd_pd(_mm_set1_pd(B[13 * ldb]), _A2, _C13_2);
        _C13_3 += B[13 * ldb] * _A3;

        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm256_storeu_pd(&C[0 * ldc + 8], _C0_1);
    _mm_storeu_pd(&C[0 * ldc + 12], _C0_2);
    C[0 * ldc + 14] = _C0_3;

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm256_storeu_pd(&C[1 * ldc + 8], _C1_1);
    _mm_storeu_pd(&C[1 * ldc + 12], _C1_2);
    C[1 * ldc + 14] = _C1_3;

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm256_storeu_pd(&C[2 * ldc + 8], _C2_1);
    _mm_storeu_pd(&C[2 * ldc + 12], _C2_2);
    C[2 * ldc + 14] = _C2_3;

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm256_storeu_pd(&C[3 * ldc + 8], _C3_1);
    _mm_storeu_pd(&C[3 * ldc + 12], _C3_2);
    C[3 * ldc + 14] = _C3_3;

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm256_storeu_pd(&C[4 * ldc + 8], _C4_1);
    _mm_storeu_pd(&C[4 * ldc + 12], _C4_2);
    C[4 * ldc + 14] = _C4_3;

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm256_storeu_pd(&C[5 * ldc + 8], _C5_1);
    _mm_storeu_pd(&C[5 * ldc + 12], _C5_2);
    C[5 * ldc + 14] = _C5_3;

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm256_storeu_pd(&C[6 * ldc + 8], _C6_1);
    _mm_storeu_pd(&C[6 * ldc + 12], _C6_2);
    C[6 * ldc + 14] = _C6_3;

    _mm512_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm256_storeu_pd(&C[7 * ldc + 8], _C7_1);
    _mm_storeu_pd(&C[7 * ldc + 12], _C7_2);
    C[7 * ldc + 14] = _C7_3;

    _mm512_storeu_pd(&C[8 * ldc + 0], _C8_0);
    _mm256_storeu_pd(&C[8 * ldc + 8], _C8_1);
    _mm_storeu_pd(&C[8 * ldc + 12], _C8_2);
    C[8 * ldc + 14] = _C8_3;

    _mm512_storeu_pd(&C[9 * ldc + 0], _C9_0);
    _mm256_storeu_pd(&C[9 * ldc + 8], _C9_1);
    _mm_storeu_pd(&C[9 * ldc + 12], _C9_2);
    C[9 * ldc + 14] = _C9_3;

    _mm512_storeu_pd(&C[10 * ldc + 0], _C10_0);
    _mm256_storeu_pd(&C[10 * ldc + 8], _C10_1);
    _mm_storeu_pd(&C[10 * ldc + 12], _C10_2);
    C[10 * ldc + 14] = _C10_3;

    _mm512_storeu_pd(&C[11 * ldc + 0], _C11_0);
    _mm256_storeu_pd(&C[11 * ldc + 8], _C11_1);
    _mm_storeu_pd(&C[11 * ldc + 12], _C11_2);
    C[11 * ldc + 14] = _C11_3;

    _mm512_storeu_pd(&C[12 * ldc + 0], _C12_0);
    _mm256_storeu_pd(&C[12 * ldc + 8], _C12_1);
    _mm_storeu_pd(&C[12 * ldc + 12], _C12_2);
    C[12 * ldc + 14] = _C12_3;

    _mm512_storeu_pd(&C[13 * ldc + 0], _C13_0);
    _mm256_storeu_pd(&C[13 * ldc + 8], _C13_1);
    _mm_storeu_pd(&C[13 * ldc + 12], _C13_2);
    C[13 * ldc + 14] = _C13_3;
}
void micro_kernel16x14(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) {
    int i;
    register __m512d _A0;
    register __m512d _A1;

    register __m512d _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    register __m512d _C0_1 = _mm512_loadu_pd(&C[0 * ldc + 8]);

    register __m512d _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);  
    register __m512d _C1_1 = _mm512_loadu_pd(&C[1 * ldc + 8]);

    register __m512d _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    register __m512d _C2_1 = _mm512_loadu_pd(&C[2 * ldc + 8]);

    register __m512d _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    register __m512d _C3_1 = _mm512_loadu_pd(&C[3 * ldc + 8]);

    register __m512d _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    register __m512d _C4_1 = _mm512_loadu_pd(&C[4 * ldc + 8]);

    register __m512d _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    register __m512d _C5_1 = _mm512_loadu_pd(&C[5 * ldc + 8]);

    register __m512d _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);
    register __m512d _C6_1 = _mm512_loadu_pd(&C[6 * ldc + 8]);

    register __m512d _C7_0 = _mm512_loadu_pd(&C[7 * ldc + 0]);
    register __m512d _C7_1 = _mm512_loadu_pd(&C[7 * ldc + 8]);

    register __m512d _C8_0 = _mm512_loadu_pd(&C[8 * ldc + 0]);
    register __m512d _C8_1 = _mm512_loadu_pd(&C[8 * ldc + 8]);

    register __m512d _C9_0 = _mm512_loadu_pd(&C[9 * ldc + 0]);
    register __m512d _C9_1 = _mm512_loadu_pd(&C[9 * ldc + 8]);

    register __m512d _C10_0 = _mm512_loadu_pd(&C[10 * ldc + 0]);
    register __m512d _C10_1 = _mm512_loadu_pd(&C[10 * ldc + 8]);

    register __m512d _C11_0 = _mm512_loadu_pd(&C[11 * ldc + 0]);
    register __m512d _C11_1 = _mm512_loadu_pd(&C[11 * ldc + 8]);

    register __m512d _C12_0 = _mm512_loadu_pd(&C[12 * ldc + 0]);
    register __m512d _C12_1 = _mm512_loadu_pd(&C[12 * ldc + 8]);

    register __m512d _C13_0 = _mm512_loadu_pd(&C[13 * ldc + 0]);
    register __m512d _C13_1 = _mm512_loadu_pd(&C[13 * ldc + 8]);
    
    for(i = 0; i < k; ++i) {
        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm512_loadu_pd(&A[8]);

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A0, _C0_0);
        _C0_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0 * ldb]), _A1, _C0_1);

        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A0, _C1_0);
        _C1_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1 * ldb]), _A1, _C1_1);

        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A0, _C2_0);
        _C2_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2 * ldb]), _A1, _C2_1);

        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A0, _C3_0);
        _C3_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3 * ldb]), _A1, _C3_1);

        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A0, _C4_0);
        _C4_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4 * ldb]), _A1, _C4_1);

        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A0, _C5_0);
        _C5_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5 * ldb]), _A1, _C5_1);

        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A0, _C6_0);
        _C6_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6 * ldb]), _A1, _C6_1);

        _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A0, _C7_0);
        _C7_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7 * ldb]), _A1, _C7_1);

        _C8_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8 * ldb]), _A0, _C8_0);
        _C8_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8 * ldb]), _A1, _C8_1);

        _C9_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[9 * ldb]), _A0, _C9_0);
        _C9_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[9 * ldb]), _A1, _C9_1);

        _C10_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[10 * ldb]), _A0, _C10_0);
        _C10_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[10 * ldb]), _A1, _C10_1);

        _C11_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[11 * ldb]), _A0, _C11_0);
        _C11_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[11 * ldb]), _A1, _C11_1);

        _C12_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[12 * ldb]), _A0, _C12_0);
        _C12_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[12 * ldb]), _A1, _C12_1);

        _C13_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[13 * ldb]), _A0, _C13_0);
        _C13_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[13 * ldb]), _A1, _C13_1);
        
        A += lda;
        B += 1;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm512_storeu_pd(&C[0 * ldc + 8], _C0_1);

    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm512_storeu_pd(&C[1 * ldc + 8], _C1_1);

    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm512_storeu_pd(&C[2 * ldc + 8], _C2_1);

    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm512_storeu_pd(&C[3 * ldc + 8], _C3_1);

    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm512_storeu_pd(&C[4 * ldc + 8], _C4_1);

    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm512_storeu_pd(&C[5 * ldc + 8], _C5_1);

    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm512_storeu_pd(&C[6 * ldc + 8], _C6_1);

    _mm512_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm512_storeu_pd(&C[7 * ldc + 8], _C7_1);

    _mm512_storeu_pd(&C[8 * ldc + 0], _C8_0);
    _mm512_storeu_pd(&C[8 * ldc + 8], _C8_1);

    _mm512_storeu_pd(&C[9 * ldc + 0], _C9_0);
    _mm512_storeu_pd(&C[9 * ldc + 8], _C9_1);

    _mm512_storeu_pd(&C[10 * ldc + 0], _C10_0);
    _mm512_storeu_pd(&C[10 * ldc + 8], _C10_1);

    _mm512_storeu_pd(&C[11 * ldc + 0], _C11_0);
    _mm512_storeu_pd(&C[11 * ldc + 8], _C11_1);

    _mm512_storeu_pd(&C[12 * ldc + 0], _C12_0);
    _mm512_storeu_pd(&C[12 * ldc + 8], _C12_1);

    _mm512_storeu_pd(&C[13 * ldc + 0], _C13_0);
    _mm512_storeu_pd(&C[13 * ldc + 8], _C13_1);
}


void (* micro_kernels[16][14])(int k, const double *A, const double *B, double *C, int lda, int ldb, int ldc) = {
    {micro_kernel1x1,micro_kernel1x2,micro_kernel1x3,micro_kernel1x4,micro_kernel1x5,micro_kernel1x6, micro_kernel1x7,micro_kernel1x8,micro_kernel1x9,micro_kernel1x10,micro_kernel1x11,micro_kernel1x12, micro_kernel1x13,micro_kernel1x14},
    {micro_kernel2x1,micro_kernel2x2,micro_kernel2x3,micro_kernel2x4,micro_kernel2x5,micro_kernel2x6, micro_kernel2x7,micro_kernel2x8,micro_kernel2x9,micro_kernel2x10,micro_kernel2x11,micro_kernel2x12, micro_kernel2x13,micro_kernel2x14},
    {micro_kernel3x1,micro_kernel3x2,micro_kernel3x3,micro_kernel3x4,micro_kernel3x5,micro_kernel3x6, micro_kernel3x7,micro_kernel3x8,micro_kernel3x9,micro_kernel3x10,micro_kernel3x11,micro_kernel3x12, micro_kernel3x13,micro_kernel3x14},
    {micro_kernel4x1,micro_kernel4x2,micro_kernel4x3,micro_kernel4x4,micro_kernel4x5,micro_kernel4x6, micro_kernel4x7,micro_kernel4x8,micro_kernel4x9,micro_kernel4x10,micro_kernel4x11,micro_kernel4x12, micro_kernel4x13,micro_kernel4x14},
    {micro_kernel5x1,micro_kernel5x2,micro_kernel5x3,micro_kernel5x4,micro_kernel5x5,micro_kernel5x6, micro_kernel5x7,micro_kernel5x8,micro_kernel5x9,micro_kernel5x10,micro_kernel5x11,micro_kernel5x12, micro_kernel5x13,micro_kernel5x14},
    {micro_kernel6x1,micro_kernel6x2,micro_kernel6x3,micro_kernel6x4,micro_kernel6x5,micro_kernel6x6, micro_kernel6x7,micro_kernel6x8,micro_kernel6x9,micro_kernel6x10,micro_kernel6x11,micro_kernel6x12, micro_kernel6x13,micro_kernel6x14},
    {micro_kernel7x1,micro_kernel7x2,micro_kernel7x3,micro_kernel7x4,micro_kernel7x5,micro_kernel7x6, micro_kernel7x7,micro_kernel7x8,micro_kernel7x9,micro_kernel7x10,micro_kernel7x11,micro_kernel7x12, micro_kernel7x13,micro_kernel7x14},
    {micro_kernel8x1,micro_kernel8x2,micro_kernel8x3,micro_kernel8x4,micro_kernel8x5,micro_kernel8x6, micro_kernel8x7,micro_kernel8x8,micro_kernel8x9,micro_kernel8x10,micro_kernel8x11,micro_kernel8x12, micro_kernel8x13,micro_kernel8x14},
    {micro_kernel9x1,micro_kernel9x2,micro_kernel9x3,micro_kernel9x4,micro_kernel9x5,micro_kernel9x6, micro_kernel9x7,micro_kernel9x8,micro_kernel9x9,micro_kernel9x10,micro_kernel9x11,micro_kernel9x12, micro_kernel9x13,micro_kernel9x14},
    {micro_kernel10x1,micro_kernel10x2,micro_kernel10x3,micro_kernel10x4,micro_kernel10x5,micro_kernel10x6, micro_kernel10x7,micro_kernel10x8,micro_kernel10x9,micro_kernel10x10,micro_kernel10x11,micro_kernel10x12, micro_kernel10x13,micro_kernel10x14},
    {micro_kernel11x1,micro_kernel11x2,micro_kernel11x3,micro_kernel11x4,micro_kernel11x5,micro_kernel11x6, micro_kernel11x7,micro_kernel11x8,micro_kernel11x9,micro_kernel11x10,micro_kernel11x11,micro_kernel11x12, micro_kernel11x13,micro_kernel11x14},
    {micro_kernel12x1,micro_kernel12x2,micro_kernel12x3,micro_kernel12x4,micro_kernel12x5,micro_kernel12x6, micro_kernel12x7,micro_kernel12x8,micro_kernel12x9,micro_kernel12x10,micro_kernel12x11,micro_kernel12x12, micro_kernel12x13,micro_kernel12x14},
    {micro_kernel13x1,micro_kernel13x2,micro_kernel13x3,micro_kernel13x4,micro_kernel13x5,micro_kernel13x6, micro_kernel13x7,micro_kernel13x8,micro_kernel13x9,micro_kernel13x10,micro_kernel13x11,micro_kernel13x12, micro_kernel13x13,micro_kernel13x14},
    {micro_kernel14x1,micro_kernel14x2,micro_kernel14x3,micro_kernel14x4,micro_kernel14x5,micro_kernel14x6, micro_kernel14x7,micro_kernel14x8,micro_kernel14x9,micro_kernel14x10,micro_kernel14x11,micro_kernel14x12, micro_kernel14x13,micro_kernel14x14},
    {micro_kernel15x1,micro_kernel15x2,micro_kernel15x3,micro_kernel15x4,micro_kernel15x5,micro_kernel15x6, micro_kernel15x7,micro_kernel15x8,micro_kernel15x9,micro_kernel15x10,micro_kernel15x11,micro_kernel15x12, micro_kernel15x13,micro_kernel15x14},
    {micro_kernel16x1,micro_kernel16x2,micro_kernel16x3,micro_kernel16x4,micro_kernel16x5,micro_kernel16x6, micro_kernel16x7,micro_kernel16x8,micro_kernel16x9,micro_kernel16x10,micro_kernel16x11,micro_kernel16x12, micro_kernel16x13,micro_kernel16x14}
};


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
    #pragma omp parallel for num_threads(NT2) private(j) //schedule(dynamic)
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

void dgemm_110k(const int              m,
		const int              n,
		const int              k,
		const double * A,
		const int              la, // leading dimension of A
		const double * B,
		const int              lb, // leading dimension of B
		double       * C,
		const int              lc) // leading dimension of C
{



	int mq = (m+MB-1) / MB;
	int md = m % MB;
	int mc;
	int kq = (k+KB-1) / KB;
	int kd = k % KB;
	int kc;
	int ii;
	int l;
	
	omp_set_nested(1);
    static double _A[KB*MB] __attribute__((aligned(64)));

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
/*----------------------------------------------------------------------------------------------------------------------
    ==============dgemm_500======================    */

// Packing A, c-major -> c-major
void packacc500(int           row, // # of rows
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
	#pragma omp parallel for num_threads(Nt) private(i) schedule(dynamic)
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

// Packing B, c-major -> r-major
void packbcr500(int           row, // # of rows
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
    #pragma omp parallel for num_threads(Nt2) private(j) //schedule(dynamic)
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


void jirloop500(const int              m,
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
		#pragma omp parallel num_threads(Nt1) private(_B, nc, j)
		{
			#pragma omp for schedule(dynamic)
			// j-loop
			for(j = 0; j < nq; ++j)
			{
				nc = (j != nq-1 || nd == 0) ? NB : nd;
				packbcr500(k,nc,&B[ki*KB+j*NB*lb],lb,_B);
				#pragma omp parallel num_threads(Nt2) private(ir,mc,pq,p,pd,pc,_C) shared(_B, j,nc)
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
								micro_kernel0(k,&A[ir*MR*k],&_B[p*NR*k],&C[j*NB*lc+p*NR*lc+mi*Mb+ir*MR],lc);
							}
							else
							{
								micro_kernel1(k,&A[ir*MR*k],&_B[p*NR*k],_C,MR);
								micro_dxpy(mc,pc,&C[j*NB*lc+p*NR*lc+mi*Mb+ir*MR],_C,lc);
							}
						}
					}
				}
			}
		}
	}
}
//

void dgemm_500(const int              m,
		const int              n,
		const int              k,
		const double * A,
		const int              la, // leading dimension of A
		const double * B,
		const int              lb, // leading dimension of B
		double       * C,
		const int              lc) // leading dimension of C

{

	int mq = (m+Mb-1) / Mb;
	int md = m % Mb;
	int mc;
	int kq = (k+KB-1) / KB;
	int kd = k % KB;
	int kc;
	int ii;
	int l;
	
	omp_set_nested(1);
    static double _A[KB*Mb] __attribute__((aligned(64)));

	// I-loop
	for(ii = 0; ii < mq; ++ii)
	{
		mc = (ii != mq-1 || md == 0) ? Mb : md;
		for(l = 0; l < kq; ++l)
		{
			kc = (l != kq-1 || kd == 0) ? KB : kd;
			packacc500(mc,kc,&A[ii*Mb+l*KB*la],la,_A);
				//j-ir loop
				jirloop500(mc,n,ii,kc,l,_A,la,B,lb,C,lc);
			}
		}
	}
/*========================dgemm_300====================================================*/


// Packing A, c-major -> c-major
void packacc300(int           row, // # of rows
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
	#pragma omp parallel for num_threads(Nnt) private(i) schedule(dynamic)
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

// Packing B, c-major -> r-major
void packbcr300(int           row, // # of rows
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
    #pragma omp parallel for num_threads(Nt2) private(j) //schedule(dynamic)
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


void jirloop300(const int              m,
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
	int nq = (n+NnB-1) / NnB;
	int nd = n % NnB;
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

	double _B[KkB*NnB] __attribute__((aligned(64)));
	double _C[MR*NR] __attribute__((aligned(64)));
	//
	{
		omp_set_nested(1);
		#pragma omp parallel num_threads(Nnt1) private(_B, nc, j)
		{
			#pragma omp for schedule(dynamic)
			// j-loop
			for(j = 0; j < nq; ++j)
			{
				nc = (j != nq-1 || nd == 0) ? NnB : nd;
				packbcr500(k,nc,&B[ki*KkB+j*NnB*lb],lb,_B);
				#pragma omp parallel num_threads(Nt2) private(ir,mc,pq,p,pd,pc,_C) shared(_B, j,nc)
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
								micro_kernel0(k,&A[ir*MR*k],&_B[p*NR*k],&C[j*NnB*lc+p*NR*lc+mi*Mb+ir*MR],lc);
							}
							else
							{
								micro_kernel1(k,&A[ir*MR*k],&_B[p*NR*k],_C,MR);
								micro_dxpy(mc,pc,&C[j*NnB*lc+p*NR*lc+mi*Mb+ir*MR],_C,lc);
							}
						}
					}
				}
			}
		}
	}
}
//

void dgemm_300(const int              m,
		const int              n,
		const int              k,
		const double * A,
		const int              la, // leading dimension of A
		const double * B,
		const int              lb, // leading dimension of B
		double       * C,
		const int              lc) // leading dimension of C

{

	int mq = (m+Mb-1) / Mb;
	int md = m % Mb;
	int mc;
	int kq = (k+KkB-1) / KkB;
	int kd = k % KkB;
	int kc;
	int ii;
	int l;
	
	omp_set_nested(1);
    static double _A[KkB*Mb] __attribute__((aligned(64)));

	// I-loop
	for(ii = 0; ii < mq; ++ii)
	{
		mc = (ii != mq-1 || md == 0) ? Mb : md;
		for(l = 0; l < kq; ++l)
		{
			kc = (l != kq-1 || kd == 0) ? KkB : kd;
			packacc300(mc,kc,&A[ii*Mb+l*KkB*la],la,_A);
				//j-ir loop
				jirloop300(mc,n,ii,kc,l,_A,la,B,lb,C,lc);
			}
		}
	}
//------------------------small kernel dgemm-------------------------------------
void blocked_dgemm_small(const int              m,
                        const int              n,
                        const int              k,
                        const double * A,
                        const int              la, // leading dimension of A
                        const double * B,
                        const int              lb, // leading dimension of B
                        double       * C,
                        const int              lc) // leading dimension of C
{
    int mq = (m+MR-1) / MR; // quotient
    int md = m % MR; // remainder
    int mc;

    int pq; // quotient
    int pd; // remainder
    int pc;

    int ir;
    int p;

    for(ir = 0; ir <mq ; ++ir)
    {
        mc = (ir != mq-1 || md == 0) ? MR : md; // A sliver size : mc X kc
        pq = (n+NR-1) / NR; // quotient
        pd = n % NR; // remainder
        // jr-loop
        for(p = 0; p < pq; ++p)
        {
            pc = (p != pq-1 || pd == 0) ? NR : pd;
            micro_kernels[mc-1][pc-1](k,&A[ir*MR],&B[lb*NR*p],&C[p*NR*lc+ir*MR],la,lb,lc);
        }
    }
}
//-----------------------------------------------------------------------------------------

 //   void userdgemm(int m, int n, int k, double *A, int lda, double *B, int ldb, double *C, int ldc)
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
  int lda = *_la, ldb = *_lb, ldc = *_lc;
  double *A = (double*)AA;
  double *B = (double*)BB;
  double *C = (double*)CC;


        if(lda >=1100)
        {
            dgemm_110k(m, n, k, A, lda, B, ldb, C, ldc);
        }
        else if((lda < 1100) && (lda >= 500))
        {
            dgemm_500(m, n, k, A, lda, B, ldb, C, ldc);
        }
       else if((lda < 500) && (lda > 224))
        {
            dgemm_300(m, n, k, A, lda, B, ldb, C, ldc);
 //           small_dgemm(m, n, k, A, lda, B, ldb, C, ldc);
        }
	   else
	   {
		   blocked_dgemm_small(m, n, k, A, lda, B, ldb, C, ldc);
	   }
    }


