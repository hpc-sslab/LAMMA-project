#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include "mm.h"

#define IJK 0
#define IKJ 1
#define JIK 2
#define KIJ 3
#define JKI 4
#define KJI 5

#define LOOP_VER JIK

//ijk
__global__ void NaiveMatrixMulKernel_ijk(float *C, float* const __restrict__ A,float* const __restrict__ B, const unsigned int m, const unsigned int k, const unsigned int n)
{

	//get the row index

	const unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
	//get the column index
	const unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;

	//initialize the sum variable to store temporarily result
	float sum = 0;
	//calculate product of (k)th row of matrix M and (k)th column of matrix N
	for(unsigned int kk = 0; kk < n; kk++)
	{	float a = A[i*n + kk];
		float b = B[kk*n + j];
		sum +=  a*b;
	}

	//assign the sum to matrix P
	C[i*n + j] = sum;
}
//ikj
__global__ void NaiveMatrixMulKernel_ikj(float *C, float* const __restrict__ A,float* const __restrict__ B, const unsigned int m, const unsigned int k, const unsigned int n)
{

	//get the column index
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	//get the row index
	const unsigned int kk = blockIdx.y * blockDim.y + threadIdx.y;


	//initialize the sum variable to store temporarily result
	//float sum = 0;
	float a = A[i*n + kk];
	for(unsigned int j = 0; j < n; j++)
	{

		float b = B[kk*n + j];
		//printf("C[%d][%d]+=A[%d][%d]*B[%d][%d] >> %5.3f = %5.3f + %5.3f*%5.3f Threadx = %d, Thready = %d, Blockx = %d, Blocky = %d\n", i, j, i, kk, kk, j, C[i*n+j]+a*b, C[i*n+j], a, b, threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);
		atomicAdd(&C[i*n+j] , a*b);
	}
	//assign the sum to matrix P
	//C[i*n + j] = sum;
}
//jik
__global__ void NaiveMatrixMulKernel_jik(float *C, float* const __restrict__ A,float* const __restrict__ B, const unsigned int m, const unsigned int k, const unsigned int n)
{

	//get the row index
	const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
	//get the column index
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	//initialize the sum variable to store temporarily result
	float sum = 0;
	//calculate product of (k)th row of matrix M and (k)th column of matrix N
	for(unsigned int kk = 0; kk < n; kk++)
	{	float a = A[i*n + kk];
		float b = B[kk*n + j];
		sum +=  a*b;
	}

	//assign the sum to matrix P
	C[i*n + j] = sum;
}
//kij
__global__ void NaiveMatrixMulKernel_kij(float *C, float* const __restrict__ A,float* const __restrict__ B, const unsigned int m, const unsigned int k, const unsigned int n)
{

	//get the row index
	const unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
	//get the column index
	const unsigned int kk = blockIdx.x * blockDim.x + threadIdx.x;

	//initialize the sum variable to store temporarily result
	//float sum = 0;
	float a = A[i*n + kk];
	for(unsigned int j = 0; j < n; j++)
	{

		float b = B[kk*n + j];
		//printf("C[%d][%d]+=A[%d][%d]*B[%d][%d] >> %5.3f = %5.3f + %5.3f*%5.3f Threadx = %d, Thready = %d, Blockx = %d, Blocky = %d\n", i, j, i, kk, kk, j, C[i*n+j]+a*b, C[i*n+j], a, b, threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);
		atomicAdd(&C[i*n+j] , a*b);
	}
	//assign the sum to matrix P
	//C[i*n + j] = sum;
}

//jki
__global__ void NaiveMatrixMulKernel_jki(float *C, float* const __restrict__ A,float* const __restrict__ B, const unsigned int m, const unsigned int k, const unsigned int n)
{

	//get the column index
	const unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
	//get the row index
	const unsigned int kk = blockIdx.y * blockDim.y + threadIdx.y;


	//initialize the sum variable to store temporarily result
	//float sum = 0;
	float b = B[kk*n + j];
	for(unsigned int i = 0; i < n; i++)
	{
		float a = A[i*n + kk];
		//printf("C[%d][%d]+=A[%d][%d]*B[%d][%d] >> %5.3f = %5.3f + %5.3f*%5.3f Threadx = %d, Thready = %d, Blockx = %d, Blocky = %d\n", i, j, i, kk, kk, j, C[i*n+j]+a*b, C[i*n+j], a, b, threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);
		atomicAdd(&C[i*n+j] , a*b);
	}
	//assign the sum to matrix P
	//C[i*n + j] = sum;
}

//kji
__global__ void NaiveMatrixMulKernel_kji(float *C, float* const __restrict__ A,float* const __restrict__ B, const unsigned int m, const unsigned int k, const unsigned int n)
{

	//get the row index
	const unsigned int kk = blockIdx.y * blockDim.y + threadIdx.y;
	//get the column index
	const unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;

	//initialize the sum variable to store temporarily result
	//float sum = 0;
	float b = B[kk*n + j];
	for(unsigned int i = 0; i < n; i++)
	{
		float a = A[i*n + kk];
		//printf("C[%d][%d]+=A[%d][%d]*B[%d][%d] >> %5.3f = %5.3f + %5.3f*%5.3f Threadx = %d, Thready = %d, Blockx = %d, Blocky = %d\n", i, j, i, kk, kk, j, C[i*n+j]+a*b, C[i*n+j], a, b, threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);
		atomicAdd(&C[i*n+j] , a*b);
	}
	//assign the sum to matrix P
	//C[i*n + j] = sum;
}
__global__ void NaiveMatrixMulKernel(float *C, float* const __restrict__ A,float* const __restrict__ B, const unsigned int m, const unsigned int k, const unsigned int n)
{

	//get the row index

	const unsigned int gRow = blockIdx.y * blockDim.y + threadIdx.y;
	//get the column index
	const unsigned int gCol = blockIdx.x * blockDim.x + threadIdx.x;

	//initialize the sum variable to store temporarily result
	float sum = 0;
#pragma unroll
	//calculate product of (k)th row of matrix M and (k)th column of matrix N
	for(unsigned int kk = 0; kk < k; kk++)
	{	//float a = __ldg(&A[gRow*k + kk]);
		//float b = __ldg(&B[kk * n + gCol]);
		float a = A[gRow*k + kk];
		float b = B[kk * n + gCol];
		sum +=  a*b;
	}

	//assign the sum to matrix P
	C[gRow*m + gCol] = sum;
}
//kernel function 
__global__ void MatrixMulKernel(float *C, float* const A,float* const B, const unsigned int m, const unsigned int k, const unsigned int n)
{

	//get the row index
	const unsigned int gRow = blockIdx.y * TILE_WIDTH + threadIdx.y;
	//const int row = blockIdx.y * blockDim.y + threadIdx.y;
	//get the column index
	const unsigned int gCol = blockIdx.x * TILE_WIDTH + threadIdx.x;
	//const int col = blockIdx.x * blockDim.x + threadIdx.x;

	//initialize the sum variable to store temporarily result
	float sum = 0;

#pragma unroll
	//calculate product of (k)th row of matrix M and (k)th column of matrix N
	for(int kk = 0; kk < k; kk++)
		sum += A[gRow * k + kk] * B[kk * n + gCol];
	//sum += __ldg(&A[gRow * k + kk]) * __ldg(&B[kk * n + gCol]);

	//assign the sum to matrix P 
	C[gRow * m + gCol] = sum;
}

int main()
{
	float *M, *N, *KP, *CP;
	cudaEvent_t start, stop;
	float elapsedTime;

	//initialize host memory

	M = GenMatrix(M_DIM, K_DIM);
	//PrintMatrix(M, WIDTH);
	N = GenMatrix(K_DIM, N_DIM);
	//Nnew = GenMatrix(K_DIM, N_DIM);
	//PrintMatrix(N, WIDTH);
	KP = GenMatrix(M_DIM, N_DIM);

	CP = GenMatrix(M_DIM, N_DIM);

	//initialize device memory
	Init_Cuda(M, N, M_DIM, K_DIM, N_DIM);



	dim3 blocks(M_DIM / TILE_WIDTH , N_DIM / TILE_WIDTH);
	dim3 threads(TILE_WIDTH  , TILE_WIDTH);

	//create cudaEvent start and stop to record elapsed time
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	for(int ver=0; ver<6; ver++)
	{
		//record start time to start event
		cudaEventRecord(start, 0);

		switch (ver)
		{	case IJK:
				NaiveMatrixMulKernel_ijk<<<blocks, threads>>>(Pd, Md, Nd, M_DIM, K_DIM, N_DIM);
				break;
			case IKJ:
				NaiveMatrixMulKernel_ikj<<<blocks, threads>>>(Pd, Md, Nd, M_DIM, K_DIM, N_DIM);
				break;
			case JIK:
				NaiveMatrixMulKernel_jik<<<blocks, threads>>>(Pd, Md, Nd, M_DIM, K_DIM, N_DIM);
				break;
			case KIJ:
				NaiveMatrixMulKernel_kij<<<blocks, threads>>>(Pd, Md, Nd, M_DIM, K_DIM, N_DIM);
				break;
			case JKI:
				NaiveMatrixMulKernel_jki<<<blocks, threads>>>(Pd, Md, Nd, M_DIM, K_DIM, N_DIM);
				break;
			case KJI:
				NaiveMatrixMulKernel_kji<<<blocks, threads>>>(Pd, Md, Nd, M_DIM, K_DIM, N_DIM);
				break;
			default:
				NaiveMatrixMulKernel_ijk<<<blocks, threads>>>(Pd, Md, Nd, M_DIM, K_DIM, N_DIM);
				break;
		}

		//record start time to stop event
		cudaEventRecord(stop, 0);
		//synchronize the stop event
		cudaEventSynchronize(stop);
		//calculate the elapsed time
		cudaEventElapsedTime(&elapsedTime, start, stop);

		switch (ver)
		{
			case IJK:
				printf("ijk %.3f\n", elapsedTime);
				break;
			case IKJ:
				printf("ikj %.3f\n", elapsedTime);
				break;
			case JIK:
				printf("jik %.3f\n", elapsedTime);
				break;
			case KIJ:
				printf("kij %.3f\n", elapsedTime);
				break;
			case JKI:
				printf("jki %.3f\n", elapsedTime);
				break;
			case KJI:
				printf("kji %.3f\n", elapsedTime);
				break;
			default:
				printf("ijk %.3f\n", elapsedTime);
				break;
		}
		//cudaMemcpy(KP, Pd, M_DIM*N_DIM*sizeof(float), cudaMemcpyDeviceToHost);
		//printf("Checking the result...\n");
		//MatrixMul(CP, M, N, M_DIM, K_DIM, N_DIM);
		//printf("CP >>>>>>>>>>>>>>>>>>>>>>>>>>\n");
		//PrintMatrix(CP, M_DIM, N_DIM);
		//if(Check(KP, CP, M_DIM, N_DIM))
		//	printf("We do it.\n");
		//else
		//	printf("Something is wrong.\n");
		cudaMemset(Pd, 0, M_DIM*N_DIM*sizeof(float));
	}

	//copy data from device memory to host memory
	//destroy the start and stop event
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	/*printf("M >>>>>>>>>>>>>>>>>>>>>>>>>>\n");
	  PrintMatrix(M, WIDTH);
	  printf("N >>>>>>>>>>>>>>>>>>>>>>>>>>\n");
	  PrintMatrix(N, WIDTH);
	  printf("KP >>>>>>>>>>>>>>>>>>>>>>>>>>\n");
	  PrintMatrix(KP, WIDTH);*/

	//free host memory
	free(M);
	free(N);
	free(KP);
	free(CP);
	//free device memory
	Free_Cuda();
	return 0;
}
