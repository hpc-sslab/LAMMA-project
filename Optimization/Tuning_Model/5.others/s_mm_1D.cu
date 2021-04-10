#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>

#define TILE_WIDTH 8
//const unsigned int WIDTH  = 4096;
float *Md, *Nd, *Pd;
//generate matrix

#define IJK 0
#define IKJ 1
#define JIK 2
#define KIJ 3
#define JKI 4
#define KJI 5

#define LOOP_VER JIK
#define CHECK 0

float *GenMatrix(const unsigned int n)
{
	float *matrix;
	const unsigned int M_SIZE = n*n;
	unsigned int i = 0, j = 0;
	matrix  = (float*) malloc(M_SIZE * sizeof(float));
	for(i = 0 ;i < n; i++){
		for(j = 0 ;j < n; j++){
			matrix[i * n + j] = (rand()*1.0)/ RAND_MAX;
		}
	}
	return matrix;
}
//display matrix
int PrintMatrix(float *P, const unsigned int n)
{
	unsigned int i = 0, j = 0;
	printf("\n");
	for(i = 0 ;i < n; i++){
		for(j = 0 ;j < n; j++){
			printf("%.3f\t", P[i * n + j]);
		}
		printf("\n");
	}
	return 1;
}
//Init data
void Init_Cuda(float *M, float *N, unsigned int width)
{
	const unsigned int size = width*width*sizeof(float);
	//allocate matrix
	cudaMalloc((void**)&Md, size);
	cudaMemcpy(Md, M, size, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&Nd, size);
	cudaMemcpy(Nd, N, size,cudaMemcpyHostToDevice);
	cudaMalloc((void**)&Pd, size);
	cudaMemset(Pd, 0, size);
}
//Free memory
void Free_Cuda()
{
	cudaFree(Md);
	cudaFree(Nd);
	cudaFree(Pd);
}
//kernel function
__global__ void MatrixMulKernel(float *P, float* M, float* N, int width)
//__global__ void MatrixMulKernel(float *P, float* const __restrict__ M, float* const __restrict__ N, int width)
{
	int i, r, c;
	i = blockIdx.x*blockDim.x + threadIdx.x;
	while(i<width*width)
	{
		r = i / width;
		c = i % width;
		float sum = 0.0;

		for(int k=0; k<width; k++)
		{	//sum+= M[r*width + k] * N[k * width + c];
			float a = M[r*width +k];
			float b = N[k*width +c];
			//sum+= __ldg(&M[r*width + k]) * __ldg(&N[k * width + c]);
			sum+= a * b;
		}
		P[r * width + c] = sum;

		i+= gridDim.x * blockDim.x;
	}

}
//ijk
__global__ void MatrixMulKernel_ijk(float *P, float* M, float* N, int width)
//__global__ void MatrixMulKernel(float *P, float* const __restrict__ M, float* const __restrict__ N, int width)
{
	int idx, i, j;
	idx = blockIdx.x*blockDim.x + threadIdx.x;
	while(idx<width*width)
	{
		i = idx / width;
		j = idx % width;
		float sum = 0.0;

		for(int k=0; k<width; k++)
		{
			float a = M[i*width + k];
			float b = N[k*width + j];
			sum+= a * b;
		}
		P[i * width + j] = sum;

		idx+= gridDim.x * blockDim.x;
	}

}
//ikj
__global__ void MatrixMulKernel_ikj(float *P, float* M, float* N, int width)
//__global__ void MatrixMulKernel(float *P, float* const __restrict__ M, float* const __restrict__ N, int width)
{
	int idx, i, j, k;
	idx = blockIdx.x*blockDim.x + threadIdx.x;
	while(idx<width*width)
	{
		i = idx / width;
		k = idx % width;

		float a = M[i*width + k];
		for(j=0; j<width; j++)
		{
			float b = N[k*width + j];
			atomicAdd(&P[i*width + j], a * b);
		}
		idx+= gridDim.x * blockDim.x;
	}
}
//jik
__global__ void MatrixMulKernel_jik(float *P, float* M, float* N, int width)
//__global__ void MatrixMulKernel(float *P, float* const __restrict__ M, float* const __restrict__ N, int width)
{
	int idx, i, j, k;
	idx = blockIdx.x*blockDim.x + threadIdx.x;
	while(idx<width*width)
	{
		j = idx / width;
		i = idx % width;
		float sum = 0.0;

		for(k=0; k<width; k++)
		{
			float a = M[i*width + k];
			float b = N[k*width + j];
			sum+= a * b;
		}
		P[i * width + j] = sum;

		idx+= gridDim.x * blockDim.x;
	}

}
//kij
__global__ void MatrixMulKernel_kij(float *P, float* M, float* N, int width)
//__global__ void MatrixMulKernel(float *P, float* const __restrict__ M, float* const __restrict__ N, int width)
{
	int idx, i, j, k;
	idx = blockIdx.x*blockDim.x + threadIdx.x;
	while(idx<width*width)
	{
		k = idx / width;
		i = idx % width;

		float a = M[i*width + k];
		for(j=0; j<width; j++)
		{
			float b = N[k*width + j];
			atomicAdd(&P[i*width + j], a * b);
		}
		idx+= gridDim.x * blockDim.x;
	}
}
//jki
__global__ void MatrixMulKernel_jki(float *P, float* M, float* N, int width)
//__global__ void MatrixMulKernel(float *P, float* const __restrict__ M, float* const __restrict__ N, int width)
{
	int idx, i, j, k;
	idx = blockIdx.x*blockDim.x + threadIdx.x;
	while(idx<width*width)
	{
		j = idx / width;
		k = idx % width;

		float b = N[k*width + j];
		for(i=0; i<width; i++)
		{
			float a = M[i*width + k];
			atomicAdd(&P[i*width + j], a * b);
		}
		idx+= gridDim.x * blockDim.x;
	}
}
//kji
__global__ void MatrixMulKernel_kji(float *P, float* M, float* N, int width)
//__global__ void MatrixMulKernel(float *P, float* const __restrict__ M, float* const __restrict__ N, int width)
{
	int idx, i, j, k;
	idx = blockIdx.x*blockDim.x + threadIdx.x;
	while(idx<width*width)
	{
		k = idx / width;
		j = idx % width;
		float b = N[k*width + j];
		for(i=0; i<width; i++)
		{
			float a = M[i*width + k];
			atomicAdd(&P[i*width + j], a * b);
		}
		idx+= gridDim.x * blockDim.x;
	}
}

int MatrixMul(float *P, const float *M, const float *N, unsigned int n)
{
	int i, j, k ;
	float sum, a, b ;
	for(i = 0 ;i < n; i++)
		for(j = 0 ;j < n; j++)
		{	sum  = 0;
			for(k = 0 ;k < n; k++)
			{
				a = M[i * n + k];
				b = N[k * n + j];
				sum += a*b;
			}
			P [ i* n + j] = (float)sum;
		}
	return 1;
}
int Check(float *KP, float *CP, unsigned int n)
{
	int i, j;
	float e = 0.001;
	int correct = 1;
	for(i = 0; i < n ; i++)
		for(j = 0; j < n; j++)
		{	if(abs(KP[i * n + j] - CP[i * n + j]) > e)
			{	printf("%.10f %.10f\n", KP[i * n + j], CP[i * n + j]);
				return 0;
			}

		}
	return correct;
}
int main(int argc, char * argv[])
{
	float *M, *N, *KP, *CP;
	int width = 512;
	cudaEvent_t start, stop;
	float elapsedTime;

	//initialize host memory



	//create number of blocks and number of threads
	int T = 128;
	dim3 block(T, 1, 1);
	dim3 grid(((width*width)+ T - 1) / T, 1, 1);
	if (argc != 4) /* argc should be 2 for correct execution */
	{
		/* We print argv[0] assuming it is the program name */
		printf("Usage: %s %s %s %s.\n", argv[0], "[matrix_size]", "[block_div]", "[num_threads]");
		exit(0);
	} else {
		//printf("Arguments: %d %d", atoi(argv[1]),	atoi(argv[2]));
		width = atoi(argv[1]);
		block.x = atoi(argv[3]);
		block.y = 1;
		grid.x = ((width*width)/atoi(argv[2]) + (block.x * block.y) - 1) / (block.x*block.y);
		//grid.x = ((width*width)/atoi(argv[2]) + block.x - 1) / block.x;

	}

    
	M = GenMatrix(width);
	//PrintMatrix(M, width);
	N = GenMatrix(width);
	//PrintMatrix(N, width);
	KP = GenMatrix(width);

	CP = GenMatrix(width);

	//initialize device memory
	Init_Cuda(M, N, width);

	//create cudaEvent start and stop to record elapsed time
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//record start time to start event

	for (int ver = 0; ver < 6; ver++) {
		//record start time to start event
		cudaEventRecord(start, 0);

		switch (ver) {
		case IJK:
			MatrixMulKernel_ijk<<<grid, block>>>(Pd, Md, Nd, width);
			break;
		case IKJ:
			MatrixMulKernel_ikj<<<grid, block>>>(Pd, Md, Nd, width);
			break;
		case JIK:
			MatrixMulKernel_jik<<<grid, block>>>(Pd, Md, Nd, width);
			break;
		case KIJ:
			MatrixMulKernel_kij<<<grid, block>>>(Pd, Md, Nd, width);
			break;
		case JKI:
			MatrixMulKernel_jki<<<grid, block>>>(Pd, Md, Nd, width);
			break;
		case KJI:
			MatrixMulKernel_kji<<<grid, block>>>(Pd, Md, Nd, width);
			break;
		default:
			MatrixMulKernel_ijk<<<grid, block>>>(Pd, Md, Nd, width);
			break;
		}

		//record start time to stop event
		cudaEventRecord(stop, 0);
		//synchronize the stop event
		cudaEventSynchronize(stop);
		//calculate the elapsed time
		cudaEventElapsedTime(&elapsedTime, start, stop);

		switch (ver) {
		case IJK:
			printf("ijk %.3f (ms)\n", elapsedTime);
			break;
		case IKJ:
			printf("ikj %.3f (ms)\n", elapsedTime);
			break;
		case JIK:
			printf("jik %.3f (ms)\n", elapsedTime);
			break;
		case KIJ:
			printf("kij %.3f (ms)\n", elapsedTime);
			break;
		case JKI:
			printf("jki %.3f (ms)\n", elapsedTime);
			break;
		case KJI:
			printf("kji %.3f (ms)\n", elapsedTime);
			break;
		default:
			printf("ijk %.3f (ms)\n", elapsedTime);
			break;
		}
		//copy data from device memory to host memory
		cudaMemcpy(KP, Pd, width*width*sizeof(float), cudaMemcpyDeviceToHost);

#if (CHECK)
		MatrixMul(CP, M, N, width);
			//printf("CP >>>>>>>>>>>>>>>>>>>>>>>>>>\n");
			//PrintMatrix(CP, width);

		if(Check(KP, CP, width))
			printf("  We do it.\n");
		else
			printf("  Something is wrong.\n");

#endif

		cudaMemset(Pd, 0, width*width*sizeof(float));
	}

	//destroy the start and stop event
	cudaEventDestroy(start);
	cudaEventDestroy(stop);


	//PrintMatrix(P, width);



	/*printf("M >>>>>>>>>>>>>>>>>>>>>>>>>>\n");
	PrintMatrix(M, width);
	printf("N >>>>>>>>>>>>>>>>>>>>>>>>>>\n");
	PrintMatrix(N, width);
	printf("KP >>>>>>>>>>>>>>>>>>>>>>>>>>\n");
	PrintMatrix(KP, width);
	*/
		//free host memory
	free(M);
	free(N);
	free(KP);
	free(CP);
	//free device memory
	Free_Cuda();
	return 0;
}
