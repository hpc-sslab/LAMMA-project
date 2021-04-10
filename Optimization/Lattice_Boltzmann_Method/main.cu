/*############################################################################*/

#include "main.h"
#include "lbm.h"
#include "lbm_cuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <errno.h>

#if defined(SPEC)
#   include <time.h>
#else
#   include <sys/times.h>
#   include <unistd.h>
#endif

#include <sys/stat.h>
#define COMPUTE_CAP (3)
//#define AoS 0
/*############################################################################*/

static LBM_GridPtr srcGrid, dstGrid;
size_t gridSize;
size_t marginSize;
/*float  __align__(8) * src;
float  __align__(8) * dst;

float  __align__(8) *d_srcGrid,  __align__(8) *d_dstGrid;
float  __align__(8) *u_srcGrid,  __align__(8) *u_dstGrid,  __align__(8) *u_temp;
float  __align__(8) *srcGrid_SoA, __align__(8) *dstGrid_SoA,  __align__(8) *tmpGrid_SoA;
*/

float * src;
float * dst;

float *d_srcGrid, *d_dstGrid;
float *u_srcGrid, *u_dstGrid, *u_temp;
float *srcGrid_SoA, *dstGrid_SoA, *tmpGrid_SoA1, *tmpGrid_SoA2 ;

int *aasrc, *aadst,*aatmp;
float *srcfc, *srcfn, *srcfs, *srcfe, *srcfw, *srcft, *srcfb, *srcfne, *srcfnw, *srcfse, *srcfsw, *srcfnt, *srcfnb, *srcfst, *srcfsb, *srcfet, *srcfeb, *srcfwt, *srcfwb;
float *dstfc, *dstfn, *dstfs, *dstfe, *dstfw, *dstft, *dstfb, *dstfne, *dstfnw, *dstfse, *dstfsw, *dstfnt, *dstfnb, *dstfst, *dstfsb, *dstfet, *dstfeb, *dstfwt, *dstfwb;
float *tmpfc, *tmpfn, *tmpfs, *tmpfe, *tmpfw, *tmpft, *tmpfb, *tmpfne, *tmpfnw, *tmpfse, *tmpfsw, *tmpfnt, *tmpfnb, *tmpfst, *tmpfsb, *tmpfet, *tmpfeb, *tmpfwt, *tmpfwb;


#if(AoS==1)
dim3 GRID(BLOCKS_X, BLOCKS_Y);
dim3 BLOCKS(THREADS_X, 1, 1);
#else
	#if(PARTITIONED==0)
		dim3 GRID(SIZE_Y, SIZE_Z);
		dim3 BLOCKS(SIZE_X, 1, 1);
		//dim3 GRID(SIZE_X/TILED_WIDTH_X, SIZE_Y/TILED_WIDTH_Y, SIZE_Z/TILED_WIDTH_Z);
		//dim3 BLOCKS(TILED_WIDTH_X, TILED_WIDTH_Y);
	Distributions srcDist, dstDist, tmpDist;
	#else
		dim3 GRID(SIZE_Y, PARTITION_SIZE_Z);
		dim3 BLOCKS(SIZE_X, 1, 1);
	#endif
	//int *flags;
		unsigned char *flags;
#endif

float *t_src, *t_dst;
float *h_dst;
float *temp_arr1, *temp_arr2;
           // C, N, S, E, W, T, B,NE,NW,SE,SW,NT,NB,ST,SB,ET,EB,WT,WB
int ex[QQ] = {0, 0, 0, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0, 0, 0, 1, 1,-1,-1};
int ey[QQ] = {0, 1,-1, 0, 0, 0, 0, 1, 1,-1,-1, 1, 1,-1,-1, 0, 0, 0, 0};
int ez[QQ] = {0, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1, 1,-1};

/*############################################################################*/
#if(AoS==1)
void initialize_nbArray_AoS(int *h_nbArray) {
	h_nbArray[0] = CALC_INDEX( 0, 0, 0, 0); //C    	+0
	h_nbArray[1] = CALC_INDEX( 0,+1, 0, 1); //N    	+2001
	h_nbArray[2] = CALC_INDEX( 0,-1, 0, 2); //S    	-1998
	h_nbArray[3] = CALC_INDEX(+1, 0, 0, 3); //E    	+23
	h_nbArray[4] = CALC_INDEX(-1, 0, 0, 4); //W		-16
	h_nbArray[5] = CALC_INDEX( 0, 0,+1, 5); //T  	+200005
	h_nbArray[6] = CALC_INDEX( 0, 0,-1, 6); //B  	-199994
	h_nbArray[7] = CALC_INDEX(+1,+1, 0, 7); //NE 	+2027
	h_nbArray[8] = CALC_INDEX(-1,+1, 0, 8); //NW 	+1988
	h_nbArray[9] = CALC_INDEX(+1,-1, 0, 9); //SE	-1971
	h_nbArray[10] = CALC_INDEX(-1,-1, 0, 10); //SW	-2010
	h_nbArray[11] = CALC_INDEX( 0,+1,+1, 11); //NT	+202011
	h_nbArray[12] = CALC_INDEX( 0,+1,-1, 12); //NB	-197988
	h_nbArray[13] = CALC_INDEX( 0,-1,+1, 13); //ST	+198013
	h_nbArray[14] = CALC_INDEX( 0,-1,-1, 14); //SB	-201986
	h_nbArray[15] = CALC_INDEX(+1, 0,+1, 15); //ET	+200035
	h_nbArray[16] = CALC_INDEX(+1, 0,-1, 16); //EB	-199964
	h_nbArray[17] = CALC_INDEX(-1, 0,+1, 17); //WT	+199997
	h_nbArray[18] = CALC_INDEX(-1, 0,-1, 18); //WB	-200002

}
void initialize_lcArray_AoS(int *h_lcArray) {
	h_lcArray[0] = CALC_INDEX( 0, 0, 0, 0); //C
	h_lcArray[1] = CALC_INDEX( 0, 0, 0, 1); //N
	h_lcArray[2] = CALC_INDEX( 0, 0, 0, 2); //S
	h_lcArray[3] = CALC_INDEX( 0, 0, 0, 3); //E
	h_lcArray[4] = CALC_INDEX( 0, 0, 0, 4); //W
	h_lcArray[5] = CALC_INDEX( 0, 0, 0, 5); //T
	h_lcArray[6] = CALC_INDEX( 0, 0, 0, 6); //B
	h_lcArray[7] = CALC_INDEX( 0, 0, 0, 7); //NE
	h_lcArray[8] = CALC_INDEX( 0, 0, 0, 8); //NW
	h_lcArray[9] = CALC_INDEX( 0, 0, 0, 9); //SE
	h_lcArray[10] = CALC_INDEX( 0, 0, 0, 10); //SW
	h_lcArray[11] = CALC_INDEX( 0, 0, 0, 11); //NT
	h_lcArray[12] = CALC_INDEX( 0, 0, 0, 12); //NB
	h_lcArray[13] = CALC_INDEX( 0, 0, 0, 13); //ST
	h_lcArray[14] = CALC_INDEX( 0, 0, 0, 14); //SB
	h_lcArray[15] = CALC_INDEX( 0, 0, 0, 15); //ET
	h_lcArray[16] = CALC_INDEX( 0, 0, 0, 16); //EB
	h_lcArray[17] = CALC_INDEX( 0, 0, 0, 17); //WT
	h_lcArray[18] = CALC_INDEX( 0, 0, 0, 18); //WB
}
#else
void initialize_nbArray_SoA_w_Struct(int *h_nbArray) {
	printf("\ninitialize_nbArray_SoA_w_Struct...");
	h_nbArray[0] = CALC_INDEX_SOA_W_STRUCT( 0, 0, 0, 0); //C    	+0
	//printf("\nC %4d", h_nbArray[0]);
	h_nbArray[1] = CALC_INDEX_SOA_W_STRUCT( 0,+1, 0, 1); //N    	+20
	//printf("\nN %4d", h_nbArray[1]);
	h_nbArray[2] = CALC_INDEX_SOA_W_STRUCT( 0,-1, 0, 2); //S    	+34
	//printf("\nS %4d", h_nbArray[2]);
	h_nbArray[3] = CALC_INDEX_SOA_W_STRUCT(+1, 0, 0, 3); //E    	+55
	//printf("\nE %4d", h_nbArray[3]);
	h_nbArray[4] = CALC_INDEX_SOA_W_STRUCT(-1, 0, 0, 4); //W		+71
	//printf("\nW %4d", h_nbArray[4]);
	h_nbArray[5] = CALC_INDEX_SOA_W_STRUCT( 0, 0,+1, 5); //T  	+96
	//printf("\nT %4d", h_nbArray[5]);
	h_nbArray[6] = CALC_INDEX_SOA_W_STRUCT( 0, 0,-1, 6); //B  	+102
	//printf("\nB %4d", h_nbArray[6]);
	h_nbArray[7] = CALC_INDEX_SOA_W_STRUCT(+1,+1, 0, 7); //NE 	+129
	//printf("\nNE %4d", h_nbArray[7]);
	h_nbArray[8] = CALC_INDEX_SOA_W_STRUCT(-1,+1, 0, 8); //NW 	+145
	//printf("\nNW %4d", h_nbArray[8]);
	h_nbArray[9] = CALC_INDEX_SOA_W_STRUCT(+1,-1, 0, 9); //SE	+161
	//printf("\nSE %4d", h_nbArray[9]);
	h_nbArray[10] = CALC_INDEX_SOA_W_STRUCT(-1,-1, 0, 10); //SW	+177
	//printf("\nSW %4d", h_nbArray[10]);
	h_nbArray[11] = CALC_INDEX_SOA_W_STRUCT( 0,+1,+1, 11); //NT	+206
	//printf("\nNT %4d", h_nbArray[11]);
	h_nbArray[12] = CALC_INDEX_SOA_W_STRUCT( 0,+1,-1, 12); //NB	+212
	//printf("\nNB %4d", h_nbArray[12]);
	h_nbArray[13] = CALC_INDEX_SOA_W_STRUCT( 0,-1,+1, 13); //ST	+238
	//printf("\nST %4d", h_nbArray[13]);
	h_nbArray[14] = CALC_INDEX_SOA_W_STRUCT( 0,-1,-1, 14); //SB	+244
	//printf("\nSB %4d", h_nbArray[14]);
	h_nbArray[15] = CALC_INDEX_SOA_W_STRUCT(+1, 0,+1, 15); //ET	+277
	//printf("\nET %4d", h_nbArray[15]);
	h_nbArray[16] = CALC_INDEX_SOA_W_STRUCT(+1, 0,-1, 16); //EB	+283
	//printf("\nEB %4d", h_nbArray[16]);
	h_nbArray[17] = CALC_INDEX_SOA_W_STRUCT(-1, 0,+1, 17); //WT	+311
	//printf("\nWT %4d", h_nbArray[17]);
	h_nbArray[18] = CALC_INDEX_SOA_W_STRUCT(-1, 0,-1, 18); //WB	+317
	//printf("\nWB %4d", h_nbArray[18]);

}
void initialize_lcArray_SoA_w_Struct(int *h_lcArray) {
	printf("\ninitialize_lcArray_SoA_w_Struct...");
	h_lcArray[0] = CALC_INDEX_SOA_W_STRUCT( 0, 0, 0, 0); //C
	h_lcArray[1] = CALC_INDEX_SOA_W_STRUCT( 0, 0, 0, 1); //N
	h_lcArray[2] = CALC_INDEX_SOA_W_STRUCT( 0, 0, 0, 2); //S
	h_lcArray[3] = CALC_INDEX_SOA_W_STRUCT( 0, 0, 0, 3); //E
	h_lcArray[4] = CALC_INDEX_SOA_W_STRUCT( 0, 0, 0, 4); //W
	h_lcArray[5] = CALC_INDEX_SOA_W_STRUCT( 0, 0, 0, 5); //T
	h_lcArray[6] = CALC_INDEX_SOA_W_STRUCT( 0, 0, 0, 6); //B
	h_lcArray[7] = CALC_INDEX_SOA_W_STRUCT( 0, 0, 0, 7); //NE
	h_lcArray[8] = CALC_INDEX_SOA_W_STRUCT( 0, 0, 0, 8); //NW
	h_lcArray[9] = CALC_INDEX_SOA_W_STRUCT( 0, 0, 0, 9); //SE
	h_lcArray[10] = CALC_INDEX_SOA_W_STRUCT( 0, 0, 0, 10); //SW
	h_lcArray[11] = CALC_INDEX_SOA_W_STRUCT( 0, 0, 0, 11); //NT
	h_lcArray[12] = CALC_INDEX_SOA_W_STRUCT( 0, 0, 0, 12); //NB
	h_lcArray[13] = CALC_INDEX_SOA_W_STRUCT( 0, 0, 0, 13); //ST
	h_lcArray[14] = CALC_INDEX_SOA_W_STRUCT( 0, 0, 0, 14); //SB
	h_lcArray[15] = CALC_INDEX_SOA_W_STRUCT( 0, 0, 0, 15); //ET
	h_lcArray[16] = CALC_INDEX_SOA_W_STRUCT( 0, 0, 0, 16); //EB
	h_lcArray[17] = CALC_INDEX_SOA_W_STRUCT( 0, 0, 0, 17); //WT
	h_lcArray[18] = CALC_INDEX_SOA_W_STRUCT( 0, 0, 0, 18); //WB
}
void initialize_nbArray_SoA_wo_Struct(int *h_nbArray) {
	printf("\ninitialize_nbArray_SoA_wo_Struct...");
	h_nbArray[0] = CALC_INDEX_SOA_WO_STRUCT( 0, 0, 0, 0); //C    	+0
	//printf("\n%d", h_nbArray[0]);
	h_nbArray[1] = CALC_INDEX_SOA_WO_STRUCT( 0,+1, 0, 1); //N    	+20
	//printf("\n%d", h_nbArray[1]);
	h_nbArray[2] = CALC_INDEX_SOA_WO_STRUCT( 0,-1, 0, 2); //S    	+34
	//printf("\n%d", h_nbArray[2]);
	h_nbArray[3] = CALC_INDEX_SOA_WO_STRUCT(+1, 0, 0, 3); //E    	+55
	//printf("\n%d", h_nbArray[3]);
	h_nbArray[4] = CALC_INDEX_SOA_WO_STRUCT(-1, 0, 0, 4); //W		+71
	//printf("\n%d", h_nbArray[4]);
	h_nbArray[5] = CALC_INDEX_SOA_WO_STRUCT( 0, 0,+1, 5); //T  	+96
	//printf("\n%d", h_nbArray[5]);
	h_nbArray[6] = CALC_INDEX_SOA_WO_STRUCT( 0, 0,-1, 6); //B  	+102
	//printf("\n%d", h_nbArray[6]);
	h_nbArray[7] = CALC_INDEX_SOA_WO_STRUCT(+1,+1, 0, 7); //NE 	+129
	//printf("\n%d", h_nbArray[7]);
	h_nbArray[8] = CALC_INDEX_SOA_WO_STRUCT(-1,+1, 0, 8); //NW 	+145
	//printf("\n%d", h_nbArray[8]);
	h_nbArray[9] = CALC_INDEX_SOA_WO_STRUCT(+1,-1, 0, 9); //SE	+161
	//printf("\n%d", h_nbArray[9]);
	h_nbArray[10] = CALC_INDEX_SOA_WO_STRUCT(-1,-1, 0, 10); //SW	+177
	//printf("\n%d", h_nbArray[10]);
	h_nbArray[11] = CALC_INDEX_SOA_WO_STRUCT( 0,+1,+1, 11); //NT	+206
	//printf("\n%d", h_nbArray[11]);
	h_nbArray[12] = CALC_INDEX_SOA_WO_STRUCT( 0,+1,-1, 12); //NB	+212
	//printf("\n%d", h_nbArray[12]);
	h_nbArray[13] = CALC_INDEX_SOA_WO_STRUCT( 0,-1,+1, 13); //ST	+238
	//printf("\n%d", h_nbArray[13]);
	h_nbArray[14] = CALC_INDEX_SOA_WO_STRUCT( 0,-1,-1, 14); //SB	+244
	//printf("\n%d", h_nbArray[14]);
	h_nbArray[15] = CALC_INDEX_SOA_WO_STRUCT(+1, 0,+1, 15); //ET	+277
	//printf("\n%d", h_nbArray[15]);
	h_nbArray[16] = CALC_INDEX_SOA_WO_STRUCT(+1, 0,-1, 16); //EB	+283
	//printf("\n%d", h_nbArray[16]);
	h_nbArray[17] = CALC_INDEX_SOA_WO_STRUCT(-1, 0,+1, 17); //WT	+311
	//printf("\n%d", h_nbArray[17]);
	h_nbArray[18] = CALC_INDEX_SOA_WO_STRUCT(-1, 0,-1, 18); //WB	+317
	//printf("\n%d", h_nbArray[18]);

}
void initialize_lcArray_SoA_wo_Struct(int *h_lcArray) {
	printf("\ninitialize_lcArray_SoA_wo_Struct...");
	h_lcArray[0] = CALC_INDEX_SOA_WO_STRUCT( 0, 0, 0, 0); //C
	h_lcArray[1] = CALC_INDEX_SOA_WO_STRUCT( 0, 0, 0, 1); //N
	h_lcArray[2] = CALC_INDEX_SOA_WO_STRUCT( 0, 0, 0, 2); //S
	h_lcArray[3] = CALC_INDEX_SOA_WO_STRUCT( 0, 0, 0, 3); //E
	h_lcArray[4] = CALC_INDEX_SOA_WO_STRUCT( 0, 0, 0, 4); //W
	h_lcArray[5] = CALC_INDEX_SOA_WO_STRUCT( 0, 0, 0, 5); //T
	h_lcArray[6] = CALC_INDEX_SOA_WO_STRUCT( 0, 0, 0, 6); //B
	h_lcArray[7] = CALC_INDEX_SOA_WO_STRUCT( 0, 0, 0, 7); //NE
	h_lcArray[8] = CALC_INDEX_SOA_WO_STRUCT( 0, 0, 0, 8); //NW
	h_lcArray[9] = CALC_INDEX_SOA_WO_STRUCT( 0, 0, 0, 9); //SE
	h_lcArray[10] = CALC_INDEX_SOA_WO_STRUCT( 0, 0, 0, 10); //SW
	h_lcArray[11] = CALC_INDEX_SOA_WO_STRUCT( 0, 0, 0, 11); //NT
	h_lcArray[12] = CALC_INDEX_SOA_WO_STRUCT( 0, 0, 0, 12); //NB
	h_lcArray[13] = CALC_INDEX_SOA_WO_STRUCT( 0, 0, 0, 13); //ST
	h_lcArray[14] = CALC_INDEX_SOA_WO_STRUCT( 0, 0, 0, 14); //SB
	h_lcArray[15] = CALC_INDEX_SOA_WO_STRUCT( 0, 0, 0, 15); //ET
	h_lcArray[16] = CALC_INDEX_SOA_WO_STRUCT( 0, 0, 0, 16); //EB
	h_lcArray[17] = CALC_INDEX_SOA_WO_STRUCT( 0, 0, 0, 17); //WT
	h_lcArray[18] = CALC_INDEX_SOA_WO_STRUCT( 0, 0, 0, 18); //WB
}

#endif

void transpose_Array(float *dst,float *src, int length)
{
	int new_pos;
	int Q = 20;
	printf("\nTransposing...");
	int jump = (length) /Q;
	for(int i=0; i<length;i++)
	{
		new_pos = (i%Q)* jump + (i/Q);
		dst[new_pos] = src[i];

	}
}
void reverse_Array(float *dst, float *src, int length)
{
	int new_pos;
	int Q = 20;
	int jump = (length) /Q;
	printf("\nReversing...");
	for(int i=0; i<length;i++)
	{
		new_pos = (i%Q)* jump + (i/Q);
		dst[i] = src[new_pos];
	}
}
void refresh_Array(float *dst,float *src, int length)
{
	for(int i=0;i< length; i++)
		dst[i] = -1.0;
}
__device__ int getGlobalIdx_2D_1D()
{
	int blockId = blockIdx.y * gridDim.x + blockIdx.x;
	int threadId = blockId * blockDim.x + threadIdx.x;
	return threadId;
}

__global__ void kernel_ungathered_uncoalesced_accesses(float *src, float * dst)
{
	int offset = SIZE_X * SIZE_Y * SIZE_Z;
	int idx = getGlobalIdx_2D_1D();
	dst[idx               ] = src[idx           ];
	dst[idx +   offset + 1] = src[idx +   offset];
	dst[idx + 2*offset    ] = src[idx + 2*offset];
	dst[idx + 3*offset + 1] = src[idx + 3*offset];
	dst[idx + 4*offset    ] = src[idx + 4*offset];
	dst[idx + 5*offset + 1] = src[idx + 5*offset];

}
__global__ void kernel_gathered_uncoalesced_accesses(float *src, float * dst)
{
	int offset = SIZE_X * SIZE_Y * SIZE_Z;
	int idx = getGlobalIdx_2D_1D();
	dst[idx               ] = src[idx           ];
	dst[idx +   offset	  ] = src[idx +   offset];
	dst[idx + 2*offset    ] = src[idx + 2*offset];
	dst[idx + 3*offset + 1] = src[idx + 3*offset];
	dst[idx + 4*offset + 1] = src[idx + 4*offset];
	dst[idx + 5*offset + 1] = src[idx + 5*offset];
}

int main(int nArgs, char* arg[]) {

	MAIN_Param param;

	int t;
	int timeSteps;
	float h_elapsedTime, d_elapsedTime;
	int num_diff;
	int h_nbArray[QQ];
	int h_lcArray[QQ];
	int executedSize;
	float one_minus_o = ONEMINUSOMEGA;
	float dfl1_o = DFL1_OMEGA;
	float dfl2_o = DFL2_OMEGA;
	float dfl3_o = DFL3_OMEGA;
	size_t total_memory =0;

	//FILE* file;
//////////////////////////////////////////////////////
	MAIN_parseCommandLine(nArgs, arg, &param);
	MAIN_printInfo(&param);
	MAIN_initialize(&param);


	size_t size = (gridSize) * sizeof(float);
	executedSize = gridSize - marginSize; //size/sizeof(float) - MARGIN; //800000; //26000000

//#if(PARTITIONED==0)
//	const size_t sz_grid = (MARGIN_L_SIZE + GRID_SIZE_SOA + MARGIN_R_SIZE)*sizeof(float);
//	const size_t sz_flags = sizeof(unsigned char)*(MARGIN_L + DIST_SIZE + MARGIN_R);
//#else
//	const size_t sz_grid = (MARGIN_L_SIZE + (SIZE_X+LAYERS_NUM)*(SIZE_Y+LAYERS_NUM)*(SIZE_Z+LAYERS_NUM)*QQ + MARGIN_R_SIZE)*sizeof(float);
//	const size_t sz_flags = sizeof(unsigned char)*(MARGIN_L + (SIZE_X+LAYERS_NUM)*(SIZE_Y+LAYERS_NUM)*(SIZE_Z+LAYERS_NUM) + MARGIN_R);
//#endif

////allocate and initialize data for using SoA style/////

	printf("\ngridSize  = %d, executedSize = %d, marginSize = %d", gridSize, executedSize, marginSize);//800000);
//printf("\nSize of LBM_Grid with margin = %d (%d elements)",(gridSize) * sizeof(float), (gridSize));
//
///////////////////////////////////////////////////////////////////////////////////
//
//////Initialize for constant memory///////////////////////////////////////////////
#if(AoS==1)
	initialize_nbArray_AoS(h_nbArray);
	initialize_lcArray_AoS(h_lcArray);
#else
#if(SoA_w_Struct==1)
	initialize_nbArray_SoA_w_Struct(h_nbArray);
	initialize_lcArray_SoA_w_Struct(h_lcArray);
#else
	initialize_nbArray_SoA_wo_Struct(h_nbArray);
	initialize_lcArray_SoA_wo_Struct(h_lcArray);
#endif
#endif
	//copy to constant memory
	copyToConstantMem(h_lcArray, h_nbArray, one_minus_o, dfl1_o, dfl2_o, dfl3_o);
///////////////////////////////////////////////////////////////////////////////////
	cudaError err;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
//
//////Allocate memory for device//////////////////////////////////////////////////
#if (COMPUTE_CAP < 3)
	cudaMalloc(&u_srcGrid, size);
	cudaMalloc(&u_dstGrid, size);
	cudaMalloc(&u_temp, size);
	total_memory+=3*size;
#else
	cudaMallocManaged(&u_srcGrid, size);
	cudaMallocManaged(&u_dstGrid, size);
	cudaMallocManaged(&u_temp, size);
	total_memory+=3*size;
#endif
//
///////////////////////////////////////////////////////////////////////////////
#if (AoS==1)
#if COMPUTE_CAP < 3
#else
	for (int i = 0; i < gridSize; i++) {
		u_srcGrid[i] = src[i];
		u_dstGrid[i] = dst[i];

	}

	u_srcGrid += marginSize;
	u_dstGrid += marginSize;
#endif
#else

	//t_src = (float*)malloc(size);
	//t_dst = (float*)malloc(size);

#if COMPUTE_CAP < 3
	cudaMemcpy(u_srcGrid, src, size, cudaMemcpyHostToDevice);
	cudaMemcpy(u_dstGrid, dst, size, cudaMemcpyHostToDevice);


#else
	for (int i = 0; i < gridSize; i++) {
		u_srcGrid[i] = src[i];
		u_dstGrid[i] = dst[i];
		//t_src[i] = src[i];
		//t_dst[i] = dst[i];
	}

	u_srcGrid += marginSize;
	u_dstGrid += marginSize;
#endif
	//t_src +=marginSize;
	//t_dst +=marginSize;

//	file = fopen("compare1.txt", "w");
//	for(int i=19;i<26000000; i+=20)
//	{
//		fprintf(file, "\n<(%d, %d) , (%d, %d)>", (*MAGIC_CAST( u_srcGrid[i]) & (OBSTACLE)),(*MAGIC_CAST( u_srcGrid[i]) & (ACCEL)),  (*MAGIC_CAST( u_dstGrid[i]) & (OBSTACLE)), (*MAGIC_CAST( u_srcGrid[i]) & (ACCEL)));
//	}
//	fclose(file);
	//transpose_Array(u_srcGrid, t_src, executedSize );
	//transpose_Array(u_dstGrid, t_dst, executedSize);

#if(SoA_w_Struct==1)
	LBM_allocateGrid_SoA_w_Struct(&tmpDist, MARGIN_L + DIST_SIZE + MARGIN_R);
	LBM_allocateGrid_SoA_w_Struct(&srcDist, MARGIN_L + DIST_SIZE + MARGIN_R);
	LBM_allocateGrid_SoA_w_Struct(&dstDist, MARGIN_L + DIST_SIZE + MARGIN_R);

	LBM_allocateGrid_SoA_w_Struct_arr(&srcfc, &srcfn, &srcfs, &srcfe, &srcfw, &srcft, &srcfb, &srcfne, &srcfnw, &srcfse, &srcfsw, &srcfnt, &srcfnb, &srcfst, &srcfsb, &srcfet, &srcfeb, &srcfwt, &srcfwb, MARGIN_L + DIST_SIZE + MARGIN_R);
	if(srcfc==NULL) printf("\nCannot allocate memory.");
	LBM_allocateGrid_SoA_w_Struct_arr(&dstfc, &dstfn, &dstfs, &dstfe, &dstfw, &dstft, &dstfb, &dstfne, &dstfnw, &dstfse, &dstfsw, &dstfnt, &dstfnb, &dstfst, &dstfsb, &dstfet, &dstfeb, &dstfwt, &dstfwb, MARGIN_L + DIST_SIZE + MARGIN_R);
	LBM_allocateGrid_SoA_w_Struct_arr(&tmpfc, &tmpfn, &tmpfs, &tmpfe, &tmpfw, &tmpft, &tmpfb, &tmpfne, &tmpfnw, &tmpfse, &tmpfsw, &tmpfnt, &tmpfnb, &tmpfst, &tmpfsb, &tmpfet, &tmpfeb, &tmpfwt, &tmpfwb, MARGIN_L + DIST_SIZE + MARGIN_R);

	cudaMallocManaged(&flags, sizeof(unsigned int)*(MARGIN_L + DIST_SIZE + MARGIN_R));

	LBM_convertToSoA_w_Struct(u_srcGrid, executedSize, &srcDist, flags);
	LBM_convertToSoA_w_Struct(u_dstGrid, executedSize, &dstDist, flags);

	LBM_convertToSoA_w_Struct_arr(u_srcGrid, executedSize, srcfc, srcfn, srcfs, srcfe, srcfw, srcft, srcfb, srcfne, srcfnw, srcfse, srcfsw, srcfnt, srcfnb, srcfst, srcfsb, srcfet, srcfeb, srcfwt, srcfwb, flags);
	LBM_convertToSoA_w_Struct_arr(u_dstGrid, executedSize, dstfc, dstfn, dstfs, dstfe, dstfw, dstft, dstfb, dstfne, dstfnw, dstfse, dstfsw, dstfnt, dstfnb, dstfst, dstfsb, dstfet, dstfeb, dstfwt, dstfwb, flags);
	//LBM_displayFlags_SoA_w_Struct(flags);
#else
#if(PARTITIONED==0)
#if COMPUTE_CAP <3 //GTX285


	cudaMalloc(&srcGrid_SoA, sz_grid);
	cudaMalloc(&dstGrid_SoA, sz_grid);
	//cudaMalloc(&tmpGrid_SoA, sz_grid);
	tmpGrid_SoA1 =  (float*)malloc(sz_grid);
	tmpGrid_SoA2 =  (float*)malloc(sz_grid);
	cudaMalloc(&flags, sz_flags);

	float* convered_arr_SoA;
	convered_arr_SoA = (float*)malloc(sz_grid);
	unsigned char* convered_arr_flags;
	convered_arr_flags = (unsigned char*)malloc(sz_flags);
	src+=marginSize;
	dst+=marginSize;


	LBM_convertToSoA_wo_Struct(src, convered_arr_SoA, 	convered_arr_flags);
	cudaMemcpy(srcGrid_SoA, convered_arr_SoA, sz_grid, cudaMemcpyHostToDevice );

	LBM_convertToSoA_wo_Struct(dst, convered_arr_SoA, 	convered_arr_flags);
	cudaMemcpy(dstGrid_SoA, convered_arr_SoA, sz_grid, cudaMemcpyHostToDevice );

	cudaMemcpy(flags, convered_arr_flags, sz_flags, cudaMemcpyHostToDevice );

	free(convered_arr_SoA);
	free(convered_arr_flags);

#else //K20
	cudaMallocManaged(&srcGrid_SoA, (MARGIN_L_SIZE + GRID_SIZE_SOA + MARGIN_R_SIZE)*sizeof(float));
	cudaMallocManaged(&dstGrid_SoA, (MARGIN_L_SIZE + GRID_SIZE_SOA + MARGIN_R_SIZE)*sizeof(float));
	cudaMallocManaged(&tmpGrid_SoA1, (MARGIN_L_SIZE + GRID_SIZE_SOA + MARGIN_R_SIZE)*sizeof(float));
	total_memory+= (MARGIN_L_SIZE + GRID_SIZE_SOA + MARGIN_R_SIZE)*sizeof(float);
	//cudaMallocManaged(&flags, sizeof(unsigned int)*(MARGIN_L + DIST_SIZE + MARGIN_R));
	cudaMallocManaged(&flags, sizeof(unsigned char)*(MARGIN_L + DIST_SIZE + MARGIN_R));
	total_memory += sizeof(unsigned char)*(MARGIN_L + DIST_SIZE + MARGIN_R);
	LBM_convertToSoA_wo_Struct(u_srcGrid, srcGrid_SoA, flags);
	LBM_convertToSoA_wo_Struct(u_dstGrid, dstGrid_SoA, flags);
	//LBM_convertToSoA_wo_Struct_New_Layout(u_srcGrid, srcGrid_SoA, flags);
	//LBM_convertToSoA_wo_Struct_New_Layout(u_dstGrid, dstGrid_SoA, flags);
#endif
#else
#if COMPUTE_CAP < 3 //GTX285

	cudaMalloc(&srcGrid_SoA, sz_grid);
	cudaMalloc(&dstGrid_SoA, sz_grid);
	//cudaMalloc(&tmpGrid_SoA, sz);
	tmpGrid_SoA1 =  (float*)malloc(sz_grid);
	tmpGrid_SoA2 =  (float*)malloc(sz_grid);
	//cudaMallocManaged(&flags, sizeof(unsigned int)*(MARGIN_L + DIST_SIZE + MARGIN_R));
	cudaMalloc(&flags, sz_flags);

	err = cudaGetLastError();
	if (cudaSuccess != err) {
		printf("\nError : %s, %d", cudaGetErrorString(err), __LINE__);
		exit(1);
	}

	float* convered_arr_SoA;
	convered_arr_SoA = (float*)malloc(sz_grid);
	unsigned char* convered_arr_flags;
	convered_arr_flags = (unsigned char*)malloc(sz_flags);
	scr+=marginSize;
	dst+=marginSize;

	LBM_convertToSoA_partitioned_3dim(src, convered_arr_SoA, convered_arr_flags);
	cudaMemcpy(srcGrid_SoA, converconvered_arr_SoA, sz_grid, cudaMemcpyHostToDevice);

	LBM_convertToSoA_partitioned_3dim(dst, convered_arr_SoA, convered_arr_flags);
	cudaMemcpy(dstGrid_SoA, converconvered_arr_SoA, sz_grid, cudaMemcpyHostToDevice);

	cudaMemcpy(flags, converconvered_arr_flags, sz_flags, cudaMemcpyHostToDevice);

	free(convered_arr_SoA);
	free(convered_arr_flags);
#else //K20

	cudaMallocManaged(&srcGrid_SoA, (MARGIN_L_SIZE + (SIZE_X+LAYERS_NUM)*(SIZE_Y+LAYERS_NUM)*(SIZE_Z+LAYERS_NUM)*QQ + MARGIN_R_SIZE)*sizeof(float));
	cudaMallocManaged(&dstGrid_SoA, (MARGIN_L_SIZE + (SIZE_X+LAYERS_NUM)*(SIZE_Y+LAYERS_NUM)*(SIZE_Z+LAYERS_NUM)*QQ + MARGIN_R_SIZE)*sizeof(float));
	cudaMallocManaged(&tmpGrid_SoA, (MARGIN_L_SIZE + (SIZE_X+LAYERS_NUM)*(SIZE_Y+LAYERS_NUM)*(SIZE_Z+LAYERS_NUM)*QQ + MARGIN_R_SIZE)*sizeof(float));
	//cudaMallocManaged(&flags, sizeof(unsigned int)*(MARGIN_L + DIST_SIZE + MARGIN_R));
	cudaMallocManaged(&flags, sizeof(unsigned char)*(MARGIN_L + (SIZE_X+LAYERS_NUM)*(SIZE_Y+LAYERS_NUM)*(SIZE_Z+LAYERS_NUM) + MARGIN_R));

	err = cudaGetLastError();
	if (cudaSuccess != err) {
		printf("\nError : %s, %d", cudaGetErrorString(err), __LINE__);
		exit(1);
	}

	LBM_convertToSoA_partitioned_3dim(u_srcGrid, srcGrid_SoA, flags);

	LBM_convertToSoA_partitioned_3dim(u_dstGrid, dstGrid_SoA, flags);


	//printf("\nTotal size = %d", (MARGIN_L_SIZE + GRID_SIZE_SOA + MARGIN_R_SIZE + ALL_LAYERS_SIZE)*sizeof(float));
#endif
#endif
#endif

#endif

	timeSteps = param.nTimeSteps;

////GPU execution/////////////////////////////////////////////////////////////////////////
#if COMPUTE_CAP >= 3 //K20
	/*printf("\nBefore GPU run...");
	num_diff = checkSameGrid(u_srcGrid, u_dstGrid);
	if (num_diff == 0)
		printf("\n>>>Grids are same.\n");
	else
		printf("\n>>>Grid are different by %d elements.\n", num_diff);

	num_diff = checkSameGrid_SoA_wo_Struct(*srcGrid, srcGrid_SoA);
	if (num_diff == 0)
		printf("\n>>>Grids are same.\n");
	else
		printf("\n>>>Grid are different by %d elements.\n", num_diff);
		*/
#endif
#if(AoS==1)
	PRINT_LINE;
	printf("\nAoS VERSION");
	PRINT_LINE;
#else
#if(SoA_w_Struct==1)
	PRINT_LINE;
	printf("\nSoA VERSION USING STRUCT FOR SCHEME");
	PRINT_LINE;
#else
	PRINT_LINE;
	printf("\nSoA VERSION USING ARRAY FOR SCHEME");
	PRINT_LINE;
#endif
#endif

#if(PARTITIONED==1)
	cudaStream_t stream[PARTITIONS_NUM];
	for (int i = 0; i < PARTITIONS_NUM; ++i)
		cudaStreamCreate(&stream[i]);

	/*cudaStreamAttachMemAsync(stream[0], srcGrid_SoA, 0, cudaMemAttachSingle);
	cudaStreamAttachMemAsync(stream[0], flags, 0, cudaMemAttachSingle);
	cudaStreamAttachMemAsync(stream[1], dstGrid_SoA, 0, cudaMemAttachSingle);*/

	int a = MARGIN_L_SIZE;
	int b = PARTITION_SIZE;
	int c = ONE_LAYER_SIZE;
	int d = MARGIN_R_SIZE;
	for(int i=0;i < PARTITIONS_NUM; i++)
	{
		if(i==0){
			cudaStreamAttachMemAsync(stream[i], srcGrid_SoA, 0 , cudaMemAttachGlobal);
			cudaStreamAttachMemAsync(stream[i], dstGrid_SoA, 0 , cudaMemAttachGlobal);
			cudaStreamAttachMemAsync(stream[i], flags, 0 , cudaMemAttachGlobal);
			//printf("\ni = %d, start at = %d", i, 0);
		}

		else if(i==PARTITIONS_NUM-1){
			cudaStreamAttachMemAsync(stream[i], srcGrid_SoA, 0, cudaMemAttachGlobal);
			cudaStreamAttachMemAsync(stream[i], dstGrid_SoA, 0, cudaMemAttachGlobal);
			cudaStreamAttachMemAsync(stream[i], flags, 0 , cudaMemAttachGlobal);
			//printf("\ni = %d, start at = %d", i, a + i * (b+c) + b + d);
		}
		else{
			cudaStreamAttachMemAsync(stream[i], srcGrid_SoA, 0, cudaMemAttachGlobal);
			cudaStreamAttachMemAsync(stream[i], dstGrid_SoA, 0, cudaMemAttachGlobal);
			cudaStreamAttachMemAsync(stream[i], flags, 0 , cudaMemAttachGlobal);
			//printf("\ni = %d, start at = %d", i, a + i * (b+c) );
		}
	}

	err = cudaGetLastError();
	if (cudaSuccess != err) {
		printf("\nError : %s, %d", cudaGetErrorString(err), __LINE__);
		exit(1);
	}
#endif

	cudaDeviceSynchronize();
	cudaEventRecord(start, 0);
	for (t = 1; t <= timeSteps; t++) {
		//test_data1<<<GRID, BLOCKS>>>(u_srcGrid, u_dstGrid);
		//test_data2<<<GRID, BLOCKS>>>(u_srcGrid, u_dstGrid);
#if (AoS==1)
		cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
		lbm_kernel_AoS<<<GRID, BLOCKS>>>(u_srcGrid, u_dstGrid);
		//lbm_kernel_AoS_with_branch<<<GRID, BLOCKS>>>(u_srcGrid, u_dstGrid);//, statusGrid);
		cudaDeviceSynchronize();
		u_temp = u_srcGrid;
		u_srcGrid = u_dstGrid;
		u_dstGrid = u_temp;

#else

#if(SoA_w_Struct==1)

		//lbm_kernel_SoA_Struct<<<GRID, BLOCKS>>>(srcDist, dstDist, flags);
		//cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
		//cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
		//cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
		//lbm_kernel_SoA_Struct_sm<<<GRID, BLOCKS>>>(srcDist, dstDist, flags);
//		cudaDeviceSynchronize();
//		tmpDist = srcDist;
//		srcDist = dstDist;
//		dstDist = tmpDist;

		//cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
		//cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
		cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
		/*lbm_kernel_SoA_Struct_sm_arr<<<GRID, BLOCKS>>>(srcfc, srcfn, srcfs, srcfe, srcfw, srcft, srcfb, srcfne, srcfnw, srcfse, srcfsw, srcfnt, srcfnb, srcfst, srcfsb, srcfet, srcfeb, srcfwt, srcfwb,
														dstfc, dstfn, dstfs, dstfe, dstfw, dstft, dstfb, dstfne, dstfnw, dstfse, dstfsw, dstfnt, dstfnb, dstfst, dstfsb, dstfet, dstfeb, dstfwt, dstfwb,
														flags);*/
		lbm_kernel_SoA_19_Arrays<<<GRID, BLOCKS>>>((float* const)srcfc, (float* const)srcfn, (float* const)srcfs, (float* const)srcfe, (float* const)srcfw, (float* const)srcft, (float* const)srcfb, (float* const)srcfne, (float* const)srcfnw, (float* const)srcfse, (float* const)srcfsw, (float* const)srcfnt, (float* const)srcfnb, (float* const)srcfst, (float* const)srcfsb, (float* const)srcfet, (float* const)srcfeb, (float* const)srcfwt, (float* const)srcfwb,
																dstfc, dstfn, dstfs, dstfe, dstfw, dstft, dstfb, dstfne, dstfnw, dstfse, dstfsw, dstfnt, dstfnb, dstfst, dstfsb, dstfet, dstfeb, dstfwt, dstfwb,
																(unsigned char* const)flags);
/*		lbm_kernel_SoA_19_Arrays<<<GRID, BLOCKS>>>(srcfc, srcfn, srcfs, srcfe, srcfw, srcft, srcfb, srcfne, srcfnw, srcfse, srcfsw, srcfnt, srcfnb, srcfst, srcfsb, srcfet, srcfeb, srcfwt, srcfwb,
																		dstfc, dstfn, dstfs, dstfe, dstfw, dstft, dstfb, dstfne, dstfnw, dstfse, dstfsw, dstfnt, dstfnb, dstfst, dstfsb, dstfet, dstfeb, dstfwt, dstfwb,
																		(unsigned char* const)flags);*/

		cudaDeviceSynchronize();

		tmpfc = srcfc; srcfc = dstfc; dstfc = tmpfc;
		tmpfn = srcfn; srcfn = dstfn; dstfn = tmpfn;
		tmpfs = srcfs; srcfs = dstfs; dstfs = tmpfs;
		tmpfe = srcfe; srcfe = dstfe; dstfe = tmpfe;
		tmpfw = srcfw; srcfw = dstfw; dstfw = tmpfw;
		tmpft = srcft; srcft = dstft; dstft = tmpft;
		tmpfb = srcfb; srcfb = dstfb; dstfb = tmpfb;
		tmpfne = srcfne; srcfne = dstfne; dstfne = tmpfne;
		tmpfnw = srcfnw; srcfnw = dstfnw; dstfnw = tmpfnw;
		tmpfse = srcfse; srcfse = dstfse; dstfse = tmpfse;
		tmpfsw = srcfsw; srcfsw = dstfsw; dstfsw = tmpfsw;
		tmpfnt = srcfnt; srcfnt = dstfnt; dstfnt = tmpfnt;
		tmpfnb = srcfnb; srcfnb = dstfnb; dstfnb = tmpfnb;
		tmpfst = srcfst; srcfst = dstfst; dstfst = tmpfst;
		tmpfsb = srcfsb; srcfsb = dstfsb; dstfsb = tmpfsb;
		tmpfet = srcfet; srcfet = dstfet; dstfet = tmpfet;
		tmpfeb = srcfeb; srcfeb = dstfeb; dstfeb = tmpfeb;
		tmpfwt = srcfwt; srcfwt = dstfwt; dstfwt = tmpfwt;
		tmpfwb = srcfwb; srcfwb = dstfwb; dstfwb = tmpfwb;



#else
#if(PARTITIONED==0)

		cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
		//cudaDeviceSetCacheConfig(cudaFuncCachePreferEqual);
		//cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
		//cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte );
		//lbm_kernel_SoA_Pull_CG<<<GRID, BLOCKS>>>(srcGrid_SoA, dstGrid_SoA, (unsigned char* const) flags);
		//SoA_Push_Only<<<GRID, BLOCKS>>>(srcGrid_SoA, dstGrid_SoA, flags);
		//SoA_Pull_Only<<<GRID, BLOCKS>>>(srcGrid_SoA, dstGrid_SoA, flags);
		//SoA_Pull_Branch_Removal<<<GRID, BLOCKS>>>(srcGrid_SoA, dstGrid_SoA, flags);
		//SoA_Pull_Register_Reduction<<<GRID, BLOCKS>>>(srcGrid_SoA, dstGrid_SoA, flags);
		//SoA_Pull_DPFP_Reduction<<<GRID, BLOCKS>>>(srcGrid_SoA, dstGrid_SoA, flags);


		lbm_kernel_SoA<<<GRID, BLOCKS>>>(srcGrid_SoA, dstGrid_SoA, flags);
		//lbm_kernel_SoA_Pull<<<GRID, BLOCKS>>>(srcGrid_SoA, (const float*)dstGrid_SoA, (const unsigned char*)flags);
		//lbm_kernel_SoA_Branch<<<GRID, BLOCKS>>>(srcGrid_SoA, dstGrid_SoA, flags);
		//lbm_kernel_SoA_shuffle<<<GRID, BLOCKS>>>((float* const) srcGrid_SoA, dstGrid_SoA, (unsigned char* const) flags);
		//cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
		//cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
		//lbm_kernel_SoA_sm<<<GRID, BLOCKS>>>(srcGrid_SoA, dstGrid_SoA, flags);
#else
		//collide_cuda<<<GRID, BLOCKS, 0 , stream[0] >>>(srcGrid_SoA,(unsigned char* const) flags);
		//cudaStreamSynchronize(stream[0]);
		//stream_cuda<<<GRID, BLOCKS, 0, stream[1]>>>(srcGrid_SoA, dstGrid_SoA);
//		for(int i=0;i<PARTITIONS_NUM;i++)
//		{	lbm_kernel_partitioned<<<GRID, BLOCKS, 0, stream[i]>>>(srcGrid_SoA, dstGrid_SoA, (unsigned char* const) flags, i*((b+c)/QQ));
//			cudaStreamSynchronize(stream[i]);
//		//lbm_kernel_partitioned<<<GRID, BLOCKS, 0, stream[1]>>>(srcGrid_SoA, dstGrid_SoA, (unsigned char* const) flags, b/QQ + c/QQ);
//		//cudaStreamSynchronize(stream[1]);
//		}
#endif
		cudaDeviceSynchronize();

#if COMPUTE_CAP < 3
		cudaMemcpy(tmpGrid_SoA1, srcGrid_SoA, sz_grid, cudaMemcpyDeviceToHost);
		cudaMemcpy(tmpGrid_SoA2, dstGrid_SoA, sz_grid, cudaMemcpyDeviceToHost);
		cudaMemcpy(srcGrid_SoA, tmpGrid_SoA2, sz_grid, cudaMemcpyHostToDevice);
		cudaMemcpy(dstGrid_SoA, tmpGrid_SoA1, sz_grid, cudaMemcpyHostToDevice);
#else
		tmpGrid_SoA1 = srcGrid_SoA;
		srcGrid_SoA = dstGrid_SoA;
		dstGrid_SoA = tmpGrid_SoA1;
#endif

#endif
#endif


	}

	//FILE* file2;
	/*file = fopen("empty_pos.txt", "w+");
	//file2 = fopen("filled_pos.txt", "w+");
	for(int i=0;i<size/sizeof(float);i++)
	{
		if(statusGrid[i]==0)
			fprintf(file,"%d\n", i);
		//else
			//fprintf(file2, "%d\n", i);
	}
	fclose(file);
	//fclose(file2);
*/
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&d_elapsedTime, start, stop);
	err = cudaGetLastError();
	if(cudaSuccess!=err)
	{	printf("\nError : %s, %d", cudaGetErrorString(err), __LINE__);
		exit(1);
	}
	printf("\n>>>Device - Elapsed Time : %.8f\n", d_elapsedTime / 1000);


#if(PARTITIONED==1)
	for (int i = 0; i < 2; ++i) cudaStreamDestroy(stream[i]);
#endif

/////////////////////////////////////////////////////////////////////////////////////

////Linear Execution/////////////////////////////////////////////////////////////////
	printf("\nTotal allocated memory = %ld MB", total_memory / (1024*1024));
	//timeSteps = 20;
	PRINT_LINE;
	printf("\nLINEAR VERSION...");
	cudaEventRecord(start, 0);
	for (t = 1; t <= timeSteps; t++) {
		if (param.simType == CHANNEL) {
			LBM_handleInOutFlow(*srcGrid);
		}
		LBM_performStreamCollide(*srcGrid, *dstGrid);
		LBM_swapGrids(&srcGrid, &dstGrid);
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&h_elapsedTime, start, stop);
	PRINT_LINE;
	printf("\n>>>Host - Elapsed Time : %.8f\n", h_elapsedTime / 1000);
	PRINT_LINE;
	printf("\n>>>Speedup GPU/CPU = %.8f (times)\n", h_elapsedTime / d_elapsedTime);
	PRINT_LINE;
/////////////////////////////////////////////////////////////////////////////////////
#if (AoS==1)
	LBM_showGridStatistics2(*srcGrid, u_srcGrid);
	printf("\nResults Comparison between Linear version and GPU version...");
	num_diff = checkSameGrid(*srcGrid, u_srcGrid);

	if (num_diff == 0)
		printf("\n>>>Grids are same.\n");
	else
		printf("\n>>>Grid are different by %d elements.\n", num_diff);

#else
	printf("\nResults Comparison between Linear version and GPU version...");
#if(SoA_w_Struct==1)
	LBM_showGridStatistics2_SoA_w_Struct(*srcGrid, &srcDist, flags);

	num_diff = checkSameGrid_SoA_w_Struct(*srcGrid, &srcDist);
	//num_diff = checkSameGrid_SoA_w_Struct_arr(*srcGrid, srcfc, srcfn, srcfs, srcfe, srcfw, srcft, srcfb, srcfne, srcfnw, srcfse, srcfsw, srcfnt, srcfnb, srcfst, srcfsb, srcfet, srcfeb, srcfwt, srcfwb);

	if (num_diff == 0)
		printf("\n>>>Grids are same.\n");
	else
		printf("\n>>>Grid are different by %d elements.\n", num_diff);
#else
#if COMPUTE_CAP < 3
#else
	num_diff = checkSameGrid_SoA_wo_Struct(*srcGrid, srcGrid_SoA);
	if (num_diff == 0)
		printf("\n>>>Grids are same.\n");
	else
		printf("\n>>>Grid are different by %d elements.\n", num_diff);
#endif
#endif

	//fclose(file);

#endif

	/*if (num_diff == 0)
		printf("\nGrids are same.\n");
	else
		printf("\nGrid are different by %d elements.\n", num_diff);*/

//	cudaMallocManaged(&temp_arr1, executedSize*sizeof(float));
//
//	for(int i=0;i< executedSize; i++)
//		temp_arr1[i]= 0.0;
//	//dim3 gr(4,4);
//	//dim3 bk(16);
//	test_sm<<<GRID, BLOCKS, 100 * sizeof(float)>>>(temp_arr1);
//	cudaDeviceSynchronize(); //must call this function after the kernel function if we use cudaMallocManaged function
//	PRINT_LINE;
//	for(int i=0;i<200; i++)
//		printf("%.3f ", temp_arr1[i]);



	//MAIN_finalize(&param);

//	cudaFree(temp_arr1);
	//test();

	//cudaFree(u_srcGrid-marginSize);
	//cudaFree(u_dstGrid-marginSize);
	//cudaFree(u_temp);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
#if (AoS==0)
//	free(t_src-marginSize);	t_src = NULL;
//	free(t_dst-marginSize);	t_dst = NULL;
#if(SoA_w_Struct==1)
	LBM_freeGrid_SoA_w_Struct(&srcDist);
	LBM_freeGrid_SoA_w_Struct(&dstDist);
	LBM_freeGrid_SoA_w_Struct(&tmpDist);

	LBM_freeGrid_SoA_w_Struct_arr(srcfc, srcfn, srcfs, srcfe, srcfw, srcft, srcfb, srcfne, srcfnw, srcfse, srcfsw, srcfnt, srcfnb, srcfst, srcfsb, srcfet, srcfeb, srcfwt, srcfwb);
	LBM_freeGrid_SoA_w_Struct_arr(dstfc, dstfn, dstfs, dstfe, dstfw, dstft, dstfb, dstfne, dstfnw, dstfse, dstfsw, dstfnt, dstfnb, dstfst, dstfsb, dstfet, dstfeb, dstfwt, dstfwb);
	LBM_freeGrid_SoA_w_Struct_arr(tmpfc, tmpfn, tmpfs, tmpfe, tmpfw, tmpft, tmpfb, tmpfne, tmpfnw, tmpfse, tmpfsw, tmpfnt, tmpfnb, tmpfst, tmpfsb, tmpfet, tmpfeb, tmpfwt, tmpfwb);

#else
	//cudaFree(srcGrid_SoA);
	//cudaFree(dstGrid_SoA);
	//cudaFree(tmpGrid_SoA);
#endif
	//cudaFree(flags);
#endif

	cudaFree(u_srcGrid-marginSize);
	cudaFree(u_dstGrid-marginSize);
	cudaFree(u_temp);
	//cudaDeviceReset();
	MAIN_finalize(&param);
	return 0;
}


/*############################################################################*/

void MAIN_parseCommandLine(int nArgs, char* arg[], MAIN_Param* param) {
	//struct stat fileStat;

	int adjustArgs = 0;

	/* SPEC - handle one of --device/--platform */
	if (nArgs == 8)
		adjustArgs += 2;
	/* SPEC - handle both --device/--platform */
	if (nArgs == 10)
		adjustArgs += 4;

	if (nArgs < adjustArgs + 5 || nArgs > adjustArgs + 6) {
		printf(
				"syntax: lbm <time steps> <result file> <0: nil, 1: cmp, 2: str> <0: ldc, 1: channel flow> [<obstacle file>]\n");
		exit(1);
	}

	param->nTimeSteps = atoi(arg[adjustArgs + 1]);
	param->resultFilename = arg[adjustArgs + 2];
	printf("\nResult file : %s\n", param->resultFilename);
	param->action = (MAIN_Action) atoi(arg[adjustArgs + 3]);
	param->simType = (MAIN_SimType) atoi(arg[adjustArgs + 4]);

	if (nArgs == adjustArgs + 6) {
		param->obstacleFilename = arg[adjustArgs + 5];
		printf("\nObstacle File : %s\n", param->obstacleFilename);

		/*if (stat(param->obstacleFilename, &fileStat) != 0) {
			printf("MAIN_parseCommandLine: cannot stat obstacle file '%s'\n",
					param->obstacleFilename);
			exit(1);
		}*/
		/*if (fileStat.st_size
				!= SIZE_X * SIZE_Y * SIZE_Z + (SIZE_Y + 1) * SIZE_Z) {
			printf("MAIN_parseCommandLine:\n"
					"\tsize of file '%s' is %i bytes\n"
					"\texpected size is %i bytes\n", param->obstacleFilename,
					(int) fileStat.st_size,
					SIZE_X * SIZE_Y * SIZE_Z + (SIZE_Y + 1) * SIZE_Z);
			exit(1);
		}*/ //tam thoi remove
	} else
		param->obstacleFilename = NULL;

	/*if (param->action == COMPARE
			&& stat(param->resultFilename, &fileStat) != 0) {
		printf("MAIN_parseCommandLine: cannot stat result file '%s'\n",
				param->resultFilename);
		exit(1);
	}*/
}

/*############################################################################*/

void MAIN_printInfo(const MAIN_Param* param) {
	const char actionString[3][32] = { "nothing", "compare", "store" };
	const char simTypeString[3][32] = { "lid-driven cavity", "channel flow" };
	printf("MAIN_printInfo:\n"
			"\tgrid size      : %i x %i x %i = %.2f * 10^6 Cells\n"
			"\tnTimeSteps     : %i\n"
			"\tresult file    : %s\n"
			"\taction         : %s\n"
			"\tsimulation type: %s\n"
			"\tobstacle file  : %s\n\n", SIZE_X, SIZE_Y, SIZE_Z,
			1e-6 * SIZE_X * SIZE_Y * SIZE_Z, param->nTimeSteps,
			param->resultFilename, actionString[param->action],
			simTypeString[param->simType],
			(param->obstacleFilename == NULL) ?
					"<none>" : param->obstacleFilename);
}

/*############################################################################*/

void MAIN_initialize(const MAIN_Param* param) {

	LBM_allocateGrid((float**) &srcGrid, (float**) &src);
	LBM_allocateGrid((float**) &dstGrid, (float**) &dst);

	LBM_initializeGrid(*srcGrid);
	LBM_initializeGrid(*dstGrid);


	if (param->obstacleFilename != NULL) {
		LBM_loadObstacleFile(*srcGrid, param->obstacleFilename);
		LBM_loadObstacleFile(*dstGrid, param->obstacleFilename);
	}

	if (param->simType == CHANNEL) {
		LBM_initializeSpecialCellsForChannel(*srcGrid);
		LBM_initializeSpecialCellsForChannel(*dstGrid);
	} else {
		LBM_initializeSpecialCellsForLDC(*srcGrid);
		LBM_initializeSpecialCellsForLDC(*dstGrid);
	}

	LBM_showGridStatistics(*srcGrid);

}

/*############################################################################*/

void MAIN_finalize(const MAIN_Param* param) {
//LBM_showGridStatistics(*srcGrid);

	if (param->action == COMPARE)
		LBM_compareVelocityField(*srcGrid, param->resultFilename, TRUE);
	if (param->action == STORE)
		LBM_storeVelocityField(*srcGrid, param->resultFilename, TRUE);

	LBM_freeGrid((float**) &srcGrid);
	LBM_freeGrid((float**) &dstGrid);
}

#if !defined(SPEC)
/*############################################################################*/

void MAIN_startClock( MAIN_Time* time ) {
	time->timeScale = 1.0 / sysconf( _SC_CLK_TCK );
	time->tickStart = times( &(time->timeStart) );
}

/*############################################################################*/

void MAIN_stopClock( MAIN_Time* time, const MAIN_Param* param ) {
	time->tickStop = times( &(time->timeStop) );

	printf( "MAIN_stopClock:\n"
			"\tusr: %7.2f sys: %7.2f tot: %7.2f wct: %7.2f MLUPS: %5.2f\n\n",
			(time->timeStop.tms_utime - time->timeStart.tms_utime) * time->timeScale,
			(time->timeStop.tms_stime - time->timeStart.tms_stime) * time->timeScale,
			(time->timeStop.tms_utime - time->timeStart.tms_utime +
					time->timeStop.tms_stime - time->timeStart.tms_stime) * time->timeScale,
			(time->tickStop - time->tickStart ) * time->timeScale,
			1.0e-6 * SIZE_X * SIZE_Y * SIZE_Z * param->nTimeSteps /
			(time->tickStop - time->tickStart ) / time->timeScale );
}
#endif
