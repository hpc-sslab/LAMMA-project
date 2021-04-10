#include "lbm.h"
#include "config.h"
#ifndef LBM_CUDA_H_
#define LBM_CUDA_H_

#define THREADS_X SIZE_X
#define BLOCKS_X SIZE_Y
#define BLOCKS_Y SIZE_Z
#define TOTAL_THREADS_AoS (THREADS_X * BLOCKS_X * BLOCKS_Y)
#define TOTAL_THREADS_SoA (THREADS_X * (BLOCKS_X+1) * (BLOCKS_Y+1))
#define uxuy(ux,uy,u2) ((1.0 + (+ux + uy) * (4.5 * (+ux + uy) + 3.0) - u2))

__device__ int calc_idx(int x, int y, int z, int e);
__device__ int lc(int e, int i);
__device__ int nb_c(int i);
__device__ int nb_n(int i) ;
__device__ int nb_s(int i) ;
__device__ int nb_e(int i);
__device__ int nb_w(int i);
__device__ int nb_t(int i) ;
__device__ int nb_b(int i);
__device__ int nb_ne(int i);
__device__ int nb_nw(int i) ;
__device__ int nb_se(int i) ;
__device__ int nb_sw(int i) ;
__device__ int nb_nt(int i) ;
__device__ int nb_nb(int i) ;
__device__ int nb_st(int i) ;
__device__ int nb_sb(int i) ;
__device__ int nb_et(int i) ;
__device__ int nb_eb(int i) ;
__device__ int nb_wt(int i) ;
__device__ int nb_wb(int i) ;
__device__ int test_flag(int flag_value, int test_flag);


__global__ void lbm_kernel1(float *sGrid, float *dGrid);
__global__ void lbm_kernel2(float *sGrid, float *dGrid);
__global__ void lbm_kernel_AoS(float *sGrid, float *dGrid);
__global__ void lbm_kernel_AoS_with_branch(float *sGrid, float *dGrid);//, int* statusGrid);
__global__ void lbm_kernel_SoA(float* sGrid, float*  dGrid, unsigned char* flags);
__global__ void lbm_kernel_SoA_bk(float* __restrict__ const sGrid, float *dGrid, unsigned char* __restrict__ const flags);
__global__ void lbm_kernel_SoA_Branch(float *sGrid, float *dGrid, unsigned char *flags);
__global__ void /*__launch_bounds__(128,8)*/ lbm_kernel_SoA_shuffle(float* __restrict__ const sGrid, float *dGrid, unsigned char* __restrict__ const flags);
__global__ void lbm_kernel_SoA_sm(float *sGrid, float *dGrid, unsigned char *flags);
__global__ void lbm_kernel_SoA_Pull(float* sGrid, float* dGrid, const unsigned char* __restrict__ flags);
__global__ void lbm_kernel_SoA_CG(float* sGrid, float*  dGrid, unsigned char* __restrict__ const flags);
__global__ void lbm_kernel_SoA_Pull_CG(float* sGrid, float* dGrid, const unsigned char* __restrict__ flags);
__global__ void test_sm(float *test_arr);
void copyToConstantMem(int *h_lcArray, int *h_nbArray, float one_minus_o, float dfl1_o, float dfl2_o, float dfl3_o);
__global__ void lbm_kernel_SoA_Struct(Distributions src, Distributions dst, unsigned char *flags);
__global__ void lbm_kernel_SoA_Struct_sm(Distributions src, Distributions dst, int *flags);
__global__ void lbm_kernel_SoA_Struct_sm_arr(float *sfc, float *sfn, float *sfs, float *sfe,float *sfw,float *sft,float *sfb,float *sfne,float *sfnw,float *sfse,float *sfsw,float *sfnt,float *sfnb,float *sfst,float *sfsb,float *sfet,float *sfeb,float *sfwt,float *sfwb,
											float *dfc, float *dfn, float *dfs, float *dfe,float *dfw,float *dft,float *dfb,float *dfne,float *dfnw,float *dfse,float *dfsw,float *dfnt,float *dfnb,float *dfst,float *dfsb,float *dfet,float *dfeb,float *dfwt,float *dfwb,
											int *flags);
__global__ void lbm_kernel_SoA_19_Arrays(float* __restrict__ sfc, float* __restrict__ sfn, float* __restrict__ sfs, float* __restrict__ sfe,float* __restrict__ sfw,float* __restrict__ sft,float* __restrict__ sfb,float* __restrict__ sfne,float* __restrict__ sfnw,float* __restrict__ sfse,float* __restrict__ sfsw,float* __restrict__ sfnt,float* __restrict__ sfnb,float* __restrict__ sfst,float* __restrict__ sfsb,float* __restrict__ sfet,float* __restrict__ sfeb,float* __restrict__ sfwt,float* __restrict__ sfwb,
											float *dfc, float *dfn, float *dfs, float *dfe,float *dfw,float *dft,float *dfb,float *dfne,float *dfnw,float *dfse,float *dfsw,float *dfnt,float *dfnb,float *dfst,float *dfsb,float *dfet,float *dfeb,float *dfwt,float *dfwb,
											unsigned char* __restrict__ flags);
/*__global__ void lbm_kernel_SoA_19_Arrays(float*  sfc, float*  sfn, float*  sfs, float*  sfe,float*  sfw,float*  sft,float*  sfb,float*  sfne,float*  sfnw,float*  sfse,float*  sfsw,float*  sfnt,float*  sfnb,float*  sfst,float*  sfsb,float*  sfet,float*  sfeb,float*  sfwt,float*  sfwb,
											float *dfc, float *dfn, float *dfs, float *dfe,float *dfw,float *dft,float *dfb,float *dfne,float *dfnw,float *dfse,float *dfsw,float *dfnt,float *dfnb,float *dfst,float *dfsb,float *dfet,float *dfeb,float *dfwt,float *dfwb,
											unsigned char* __restrict__ flags);*/

__device__ int calc_idx_soa(int x, int y, int z, int e);
__global__ void collide_cuda(float* sGrid, unsigned char* __restrict__ const flags);
__global__ void stream_cuda(float *sGrid, float *dGrid);
__global__ void lbm_kernel_partitioned(float* sGrid, float *dGrid, unsigned char* __restrict__ const flags, int sPos);
inline __device__ int getGlobalIdx_3D_3D(int z);
__inline__ __device__ int index3D(int nx, int ny, int x, int y, int z);

__global__ void SoA_Push_Only(float *sGrid, float *dGrid, unsigned char *flags);
__global__ void SoA_Pull_Only(float *sGrid, float *dGrid, unsigned char *flags);
__global__ void SoA_Pull_Branch_Removal(float *sGrid, float *dGrid, unsigned char *flags);
__global__ void SoA_Pull_Register_Reduction(float *sGrid, float *dGrid, unsigned char *flags);
__global__ void SoA_Pull_DPFP_Reduction(float *sGrid, float *dGrid, unsigned char *flags);
__global__ void SoA_Pull_3_Techniques(float *sGrid, float *dGrid, unsigned char *flags);


#endif /* LBM_CUDA_H_ */
