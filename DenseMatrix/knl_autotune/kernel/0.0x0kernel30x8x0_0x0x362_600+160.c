#include "immintrin.h"
#include "x86intrin.h"
#include "zmmintrin.h"
#include <hbwmalloc.h>

#define MR 30
#define NR 8
#define MB 0
#define NB 0
#define KB 362

#define L1_DIST_A 600
#define L1_DIST_B 160

void micro_kernel0(int k, const double * A, const double * B, double * C, int ncol)
{
  int i;
  register __m512d _B0;
  register __m512d _C0_0, _C0_1, _C0_2, _C0_3, _C0_4, _C0_5, _C0_6, _C0_7, _C0_8, _C0_9, _C0_10, _C0_11, _C0_12, _C0_13, _C0_14, _C0_15, _C0_16, _C0_17, _C0_18, _C0_19, _C0_20, _C0_21, _C0_22, _C0_23, _C0_24, _C0_25, _C0_26, _C0_27, _C0_28, _C0_29;
    _C0_0 = _mm512_loadu_pd(&C[0*ncol+0]);
    _C0_1 = _mm512_loadu_pd(&C[1*ncol+0]);
    _C0_2 = _mm512_loadu_pd(&C[2*ncol+0]);
    _C0_3 = _mm512_loadu_pd(&C[3*ncol+0]);
    _C0_4 = _mm512_loadu_pd(&C[4*ncol+0]);
    _C0_5 = _mm512_loadu_pd(&C[5*ncol+0]);
    _C0_6 = _mm512_loadu_pd(&C[6*ncol+0]);
    _C0_7 = _mm512_loadu_pd(&C[7*ncol+0]);
    _C0_8 = _mm512_loadu_pd(&C[8*ncol+0]);
    _C0_9 = _mm512_loadu_pd(&C[9*ncol+0]);
    _C0_10 = _mm512_loadu_pd(&C[10*ncol+0]);
    _C0_11 = _mm512_loadu_pd(&C[11*ncol+0]);
    _C0_12 = _mm512_loadu_pd(&C[12*ncol+0]);
    _C0_13 = _mm512_loadu_pd(&C[13*ncol+0]);
    _C0_14 = _mm512_loadu_pd(&C[14*ncol+0]);
    _C0_15 = _mm512_loadu_pd(&C[15*ncol+0]);
    _C0_16 = _mm512_loadu_pd(&C[16*ncol+0]);
    _C0_17 = _mm512_loadu_pd(&C[17*ncol+0]);
    _C0_18 = _mm512_loadu_pd(&C[18*ncol+0]);
    _C0_19 = _mm512_loadu_pd(&C[19*ncol+0]);
    _C0_20 = _mm512_loadu_pd(&C[20*ncol+0]);
    _C0_21 = _mm512_loadu_pd(&C[21*ncol+0]);
    _C0_22 = _mm512_loadu_pd(&C[22*ncol+0]);
    _C0_23 = _mm512_loadu_pd(&C[23*ncol+0]);
    _C0_24 = _mm512_loadu_pd(&C[24*ncol+0]);
    _C0_25 = _mm512_loadu_pd(&C[25*ncol+0]);
    _C0_26 = _mm512_loadu_pd(&C[26*ncol+0]);
    _C0_27 = _mm512_loadu_pd(&C[27*ncol+0]);
    _C0_28 = _mm512_loadu_pd(&C[28*ncol+0]);
    _C0_29 = _mm512_loadu_pd(&C[29*ncol+0]);
  // _C += A*B
  for(i=0; i<k ; i++)
  {
    // A L1 prefetch
    _mm_prefetch((const void*) &A[L1_DIST_A+0],_MM_HINT_T0);
    _mm_prefetch((const void*) &A[L1_DIST_A+8],_MM_HINT_T0);
    _mm_prefetch((const void*) &A[L1_DIST_A+16],_MM_HINT_T0);
    _mm_prefetch((const void*) &A[L1_DIST_A+24],_MM_HINT_T0);
    // B L1 prefetch
    _mm_prefetch((const void*) &B[L1_DIST_B+0],_MM_HINT_T0);
    _B0 = _mm512_loadu_pd(&B[0]);
    _C0_0 = _mm512_fmadd_pd(_mm512_set1_pd(A[0]), _B0, _C0_0);
    _C0_1 = _mm512_fmadd_pd(_mm512_set1_pd(A[1]), _B0, _C0_1);
    _C0_2 = _mm512_fmadd_pd(_mm512_set1_pd(A[2]), _B0, _C0_2);
    _C0_3 = _mm512_fmadd_pd(_mm512_set1_pd(A[3]), _B0, _C0_3);
    _C0_4 = _mm512_fmadd_pd(_mm512_set1_pd(A[4]), _B0, _C0_4);
    _C0_5 = _mm512_fmadd_pd(_mm512_set1_pd(A[5]), _B0, _C0_5);
    _C0_6 = _mm512_fmadd_pd(_mm512_set1_pd(A[6]), _B0, _C0_6);
    _C0_7 = _mm512_fmadd_pd(_mm512_set1_pd(A[7]), _B0, _C0_7);
    _C0_8 = _mm512_fmadd_pd(_mm512_set1_pd(A[8]), _B0, _C0_8);
    _C0_9 = _mm512_fmadd_pd(_mm512_set1_pd(A[9]), _B0, _C0_9);
    _C0_10 = _mm512_fmadd_pd(_mm512_set1_pd(A[10]), _B0, _C0_10);
    _C0_11 = _mm512_fmadd_pd(_mm512_set1_pd(A[11]), _B0, _C0_11);
    _C0_12 = _mm512_fmadd_pd(_mm512_set1_pd(A[12]), _B0, _C0_12);
    _C0_13 = _mm512_fmadd_pd(_mm512_set1_pd(A[13]), _B0, _C0_13);
    _C0_14 = _mm512_fmadd_pd(_mm512_set1_pd(A[14]), _B0, _C0_14);
    _C0_15 = _mm512_fmadd_pd(_mm512_set1_pd(A[15]), _B0, _C0_15);
    _C0_16 = _mm512_fmadd_pd(_mm512_set1_pd(A[16]), _B0, _C0_16);
    _C0_17 = _mm512_fmadd_pd(_mm512_set1_pd(A[17]), _B0, _C0_17);
    _C0_18 = _mm512_fmadd_pd(_mm512_set1_pd(A[18]), _B0, _C0_18);
    _C0_19 = _mm512_fmadd_pd(_mm512_set1_pd(A[19]), _B0, _C0_19);
    _C0_20 = _mm512_fmadd_pd(_mm512_set1_pd(A[20]), _B0, _C0_20);
    _C0_21 = _mm512_fmadd_pd(_mm512_set1_pd(A[21]), _B0, _C0_21);
    _C0_22 = _mm512_fmadd_pd(_mm512_set1_pd(A[22]), _B0, _C0_22);
    _C0_23 = _mm512_fmadd_pd(_mm512_set1_pd(A[23]), _B0, _C0_23);
    _C0_24 = _mm512_fmadd_pd(_mm512_set1_pd(A[24]), _B0, _C0_24);
    _C0_25 = _mm512_fmadd_pd(_mm512_set1_pd(A[25]), _B0, _C0_25);
    _C0_26 = _mm512_fmadd_pd(_mm512_set1_pd(A[26]), _B0, _C0_26);
    _C0_27 = _mm512_fmadd_pd(_mm512_set1_pd(A[27]), _B0, _C0_27);
    _C0_28 = _mm512_fmadd_pd(_mm512_set1_pd(A[28]), _B0, _C0_28);
    _C0_29 = _mm512_fmadd_pd(_mm512_set1_pd(A[29]), _B0, _C0_29);
    A += MR;
    B += NR;
  }
  // store _C -> C
  _mm512_storeu_pd(&C[0*ncol+0], _C0_0);
  _mm512_storeu_pd(&C[1*ncol+0], _C0_1);
  _mm512_storeu_pd(&C[2*ncol+0], _C0_2);
  _mm512_storeu_pd(&C[3*ncol+0], _C0_3);
  _mm512_storeu_pd(&C[4*ncol+0], _C0_4);
  _mm512_storeu_pd(&C[5*ncol+0], _C0_5);
  _mm512_storeu_pd(&C[6*ncol+0], _C0_6);
  _mm512_storeu_pd(&C[7*ncol+0], _C0_7);
  _mm512_storeu_pd(&C[8*ncol+0], _C0_8);
  _mm512_storeu_pd(&C[9*ncol+0], _C0_9);
  _mm512_storeu_pd(&C[10*ncol+0], _C0_10);
  _mm512_storeu_pd(&C[11*ncol+0], _C0_11);
  _mm512_storeu_pd(&C[12*ncol+0], _C0_12);
  _mm512_storeu_pd(&C[13*ncol+0], _C0_13);
  _mm512_storeu_pd(&C[14*ncol+0], _C0_14);
  _mm512_storeu_pd(&C[15*ncol+0], _C0_15);
  _mm512_storeu_pd(&C[16*ncol+0], _C0_16);
  _mm512_storeu_pd(&C[17*ncol+0], _C0_17);
  _mm512_storeu_pd(&C[18*ncol+0], _C0_18);
  _mm512_storeu_pd(&C[19*ncol+0], _C0_19);
  _mm512_storeu_pd(&C[20*ncol+0], _C0_20);
  _mm512_storeu_pd(&C[21*ncol+0], _C0_21);
  _mm512_storeu_pd(&C[22*ncol+0], _C0_22);
  _mm512_storeu_pd(&C[23*ncol+0], _C0_23);
  _mm512_storeu_pd(&C[24*ncol+0], _C0_24);
  _mm512_storeu_pd(&C[25*ncol+0], _C0_25);
  _mm512_storeu_pd(&C[26*ncol+0], _C0_26);
  _mm512_storeu_pd(&C[27*ncol+0], _C0_27);
  _mm512_storeu_pd(&C[28*ncol+0], _C0_28);
  _mm512_storeu_pd(&C[29*ncol+0], _C0_29);
}
void micro_kernel1(int k, const double * A, const double * B, double * C, int ncol)
{
  int i;
  register __m512d _B0;
  register __m512d _C0_0, _C0_1, _C0_2, _C0_3, _C0_4, _C0_5, _C0_6, _C0_7, _C0_8, _C0_9, _C0_10, _C0_11, _C0_12, _C0_13, _C0_14, _C0_15, _C0_16, _C0_17, _C0_18, _C0_19, _C0_20, _C0_21, _C0_22, _C0_23, _C0_24, _C0_25, _C0_26, _C0_27, _C0_28, _C0_29;
    _C0_0 = _mm512_setzero_pd();
    _C0_1 = _mm512_setzero_pd();
    _C0_2 = _mm512_setzero_pd();
    _C0_3 = _mm512_setzero_pd();
    _C0_4 = _mm512_setzero_pd();
    _C0_5 = _mm512_setzero_pd();
    _C0_6 = _mm512_setzero_pd();
    _C0_7 = _mm512_setzero_pd();
    _C0_8 = _mm512_setzero_pd();
    _C0_9 = _mm512_setzero_pd();
    _C0_10 = _mm512_setzero_pd();
    _C0_11 = _mm512_setzero_pd();
    _C0_12 = _mm512_setzero_pd();
    _C0_13 = _mm512_setzero_pd();
    _C0_14 = _mm512_setzero_pd();
    _C0_15 = _mm512_setzero_pd();
    _C0_16 = _mm512_setzero_pd();
    _C0_17 = _mm512_setzero_pd();
    _C0_18 = _mm512_setzero_pd();
    _C0_19 = _mm512_setzero_pd();
    _C0_20 = _mm512_setzero_pd();
    _C0_21 = _mm512_setzero_pd();
    _C0_22 = _mm512_setzero_pd();
    _C0_23 = _mm512_setzero_pd();
    _C0_24 = _mm512_setzero_pd();
    _C0_25 = _mm512_setzero_pd();
    _C0_26 = _mm512_setzero_pd();
    _C0_27 = _mm512_setzero_pd();
    _C0_28 = _mm512_setzero_pd();
    _C0_29 = _mm512_setzero_pd();
  // _C += A*B
  for(i=0; i<k ; i++)
  {
    // A L1 prefetch
    _mm_prefetch((const void*) &A[L1_DIST_A+0],_MM_HINT_T0);
    _mm_prefetch((const void*) &A[L1_DIST_A+8],_MM_HINT_T0);
    _mm_prefetch((const void*) &A[L1_DIST_A+16],_MM_HINT_T0);
    _mm_prefetch((const void*) &A[L1_DIST_A+24],_MM_HINT_T0);
    // B L1 prefetch
    _mm_prefetch((const void*) &B[L1_DIST_B+0],_MM_HINT_T0);
    _B0 = _mm512_loadu_pd(&B[0]);
    _C0_0 = _mm512_fmadd_pd(_mm512_set1_pd(A[0]), _B0, _C0_0);
    _C0_1 = _mm512_fmadd_pd(_mm512_set1_pd(A[1]), _B0, _C0_1);
    _C0_2 = _mm512_fmadd_pd(_mm512_set1_pd(A[2]), _B0, _C0_2);
    _C0_3 = _mm512_fmadd_pd(_mm512_set1_pd(A[3]), _B0, _C0_3);
    _C0_4 = _mm512_fmadd_pd(_mm512_set1_pd(A[4]), _B0, _C0_4);
    _C0_5 = _mm512_fmadd_pd(_mm512_set1_pd(A[5]), _B0, _C0_5);
    _C0_6 = _mm512_fmadd_pd(_mm512_set1_pd(A[6]), _B0, _C0_6);
    _C0_7 = _mm512_fmadd_pd(_mm512_set1_pd(A[7]), _B0, _C0_7);
    _C0_8 = _mm512_fmadd_pd(_mm512_set1_pd(A[8]), _B0, _C0_8);
    _C0_9 = _mm512_fmadd_pd(_mm512_set1_pd(A[9]), _B0, _C0_9);
    _C0_10 = _mm512_fmadd_pd(_mm512_set1_pd(A[10]), _B0, _C0_10);
    _C0_11 = _mm512_fmadd_pd(_mm512_set1_pd(A[11]), _B0, _C0_11);
    _C0_12 = _mm512_fmadd_pd(_mm512_set1_pd(A[12]), _B0, _C0_12);
    _C0_13 = _mm512_fmadd_pd(_mm512_set1_pd(A[13]), _B0, _C0_13);
    _C0_14 = _mm512_fmadd_pd(_mm512_set1_pd(A[14]), _B0, _C0_14);
    _C0_15 = _mm512_fmadd_pd(_mm512_set1_pd(A[15]), _B0, _C0_15);
    _C0_16 = _mm512_fmadd_pd(_mm512_set1_pd(A[16]), _B0, _C0_16);
    _C0_17 = _mm512_fmadd_pd(_mm512_set1_pd(A[17]), _B0, _C0_17);
    _C0_18 = _mm512_fmadd_pd(_mm512_set1_pd(A[18]), _B0, _C0_18);
    _C0_19 = _mm512_fmadd_pd(_mm512_set1_pd(A[19]), _B0, _C0_19);
    _C0_20 = _mm512_fmadd_pd(_mm512_set1_pd(A[20]), _B0, _C0_20);
    _C0_21 = _mm512_fmadd_pd(_mm512_set1_pd(A[21]), _B0, _C0_21);
    _C0_22 = _mm512_fmadd_pd(_mm512_set1_pd(A[22]), _B0, _C0_22);
    _C0_23 = _mm512_fmadd_pd(_mm512_set1_pd(A[23]), _B0, _C0_23);
    _C0_24 = _mm512_fmadd_pd(_mm512_set1_pd(A[24]), _B0, _C0_24);
    _C0_25 = _mm512_fmadd_pd(_mm512_set1_pd(A[25]), _B0, _C0_25);
    _C0_26 = _mm512_fmadd_pd(_mm512_set1_pd(A[26]), _B0, _C0_26);
    _C0_27 = _mm512_fmadd_pd(_mm512_set1_pd(A[27]), _B0, _C0_27);
    _C0_28 = _mm512_fmadd_pd(_mm512_set1_pd(A[28]), _B0, _C0_28);
    _C0_29 = _mm512_fmadd_pd(_mm512_set1_pd(A[29]), _B0, _C0_29);
    A += MR;
    B += NR;
  }
  // store _C -> C
  _mm512_storeu_pd(&C[0*ncol+0], _C0_0);
  _mm512_storeu_pd(&C[1*ncol+0], _C0_1);
  _mm512_storeu_pd(&C[2*ncol+0], _C0_2);
  _mm512_storeu_pd(&C[3*ncol+0], _C0_3);
  _mm512_storeu_pd(&C[4*ncol+0], _C0_4);
  _mm512_storeu_pd(&C[5*ncol+0], _C0_5);
  _mm512_storeu_pd(&C[6*ncol+0], _C0_6);
  _mm512_storeu_pd(&C[7*ncol+0], _C0_7);
  _mm512_storeu_pd(&C[8*ncol+0], _C0_8);
  _mm512_storeu_pd(&C[9*ncol+0], _C0_9);
  _mm512_storeu_pd(&C[10*ncol+0], _C0_10);
  _mm512_storeu_pd(&C[11*ncol+0], _C0_11);
  _mm512_storeu_pd(&C[12*ncol+0], _C0_12);
  _mm512_storeu_pd(&C[13*ncol+0], _C0_13);
  _mm512_storeu_pd(&C[14*ncol+0], _C0_14);
  _mm512_storeu_pd(&C[15*ncol+0], _C0_15);
  _mm512_storeu_pd(&C[16*ncol+0], _C0_16);
  _mm512_storeu_pd(&C[17*ncol+0], _C0_17);
  _mm512_storeu_pd(&C[18*ncol+0], _C0_18);
  _mm512_storeu_pd(&C[19*ncol+0], _C0_19);
  _mm512_storeu_pd(&C[20*ncol+0], _C0_20);
  _mm512_storeu_pd(&C[21*ncol+0], _C0_21);
  _mm512_storeu_pd(&C[22*ncol+0], _C0_22);
  _mm512_storeu_pd(&C[23*ncol+0], _C0_23);
  _mm512_storeu_pd(&C[24*ncol+0], _C0_24);
  _mm512_storeu_pd(&C[25*ncol+0], _C0_25);
  _mm512_storeu_pd(&C[26*ncol+0], _C0_26);
  _mm512_storeu_pd(&C[27*ncol+0], _C0_27);
  _mm512_storeu_pd(&C[28*ncol+0], _C0_28);
  _mm512_storeu_pd(&C[29*ncol+0], _C0_29);
}
