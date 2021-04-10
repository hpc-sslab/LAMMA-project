#include<cuda.h>
#include<stdio.h>
#include<stdlib.h>
#include<iostream>

float *hs_device, *gs_device;
double *hd_device, *gd_device;
const unsigned int SINGLE_PRECISION = 1;
const unsigned int DOUBLE_PRECISION = 0;

//generate matrix
template<typename T>
T *GenMatrix(const unsigned int width, const unsigned int height)
{
	T *matrix;
	const unsigned int M_SIZE = width*height;
	unsigned int i = 0, j = 0;
	matrix  = (T*) malloc(M_SIZE * sizeof(double));
	for(i = 0 ;i < height; i++){
		for(j = 0 ;j < width; j++){
			matrix[i * width + j] = (rand()*1.0)/ RAND_MAX;
		}
	}
	return matrix;
}

//display matrix
template<typename T>
int PrintMatrix(T *P, const unsigned int width, const unsigned int height)
{
	unsigned int i = 0, j = 0;
	printf("\n");
	for(i = 0 ;i < height; i++){
		for(j = 0 ;j < width; j++){
			printf("%.3f\t", P[i * width + j]);
		}
		printf("\n");
	}
	return 1;
}
//Init data
template<typename T>
void Init_Cuda(T *M, T *N, const unsigned int width, const unsigned int height, const int sp)
{
	const unsigned int size = width*height*sizeof(T);
	//allocate matrix
	if(sp==SINGLE_PRECISION){
		cudaMalloc((void**)&hs_device, size);
		cudaMemcpy(hs_device, M, size, cudaMemcpyHostToDevice);
		cudaMalloc((void**)&gs_device, size);
		cudaMemcpy(gs_device, N, size,cudaMemcpyHostToDevice);
	}
	else
	{
		cudaMalloc((void**)&hd_device, size);
		cudaMemcpy(hd_device, M, size, cudaMemcpyHostToDevice);
		cudaMalloc((void**)&gd_device, size);
		cudaMemcpy(gd_device, N, size,cudaMemcpyHostToDevice);
	}
}
//Free memory
void Free_Cuda(const int sp)
{
	if(sp==SINGLE_PRECISION){
		cudaFree(hs_device);
		cudaFree(gs_device);
	}
	else
	{
		cudaFree(hd_device);
		cudaFree(gd_device);
	}
}
template<typename T>
__global__ void sor_kernel (T *h, T *g, const unsigned int width, const unsigned int height) {

	int i, r, c;
	i = blockIdx.x*blockDim.x + threadIdx.x;

	while(i<width*height)
	{
		r = i / width;
		c = i % width;

		if (r > 0 && r < height - 1 && c > 0 && c < width - 1) {	//printf("(%d, %d) %.2f %.2f %.2f %.2f\n", i, j, h[(i-1) * N + j] ,  h[(i+1) * N + j] ,  h[i * N + (j-1)] , h[i * N + (j+1)] );
			//printf("\n[%d, %d]", r, c);
			g[r * width + c] = 0.25
					* (h[(r - 1) * width + c] + h[(r + 1) * width + c]
							+ h[r * width + (c - 1)] + h[r * width + (c + 1)]);

		}
		i+=blockDim.x*gridDim.x;
	}

}
template<typename T>
void sor_cpu (T *h, T *g, const unsigned int width, const unsigned int height) {
// note this does Gauss-Seidel relaxation, not Jacobi that is done on gpu
	int i, j;
	for(i=1; i < height-1; i++)
		for(j=1; j < width-1; j++){
			//printf("\n[%d, %d]", i, j);
			//printf("(%d, %d) %.2f %.2f %.2f %.2f\n", i, j, h[(i-1) * N + j] ,  h[(i+1) * N + j] ,  h[i * N + (j-1)] , h[i * N + (j+1)] );
			g[i * width + j] = 0.25 * (h[(i-1) * width + j] + h[(i+1) * width + j]
							+ h[i * width + (j-1)] + h[i * width + (j+1)]);
		}

}
template<typename T>
int Check(const T *KP, const T *CP, const unsigned int width, const unsigned int height)
{
	int i, j;
	float e = 0.001;
	int correct = 1;
	for(i = 1; i < height-1 ; i++)
		for(j = 1; j < width-1; j++)
		{	if(abs(KP[i * width + j] - CP[i * width + j]) > e)
			{	printf("%.5f %.5f\n", KP[i * width + j], CP[i * width + j]);
				return 0;
			}

		}
	return correct;
}
int main(int argc, char *argv[])  {
	// loop counters
	//const int N = 16384;
	unsigned int width = 512;//16384;

	float *hs_host, *gs_host, *ks_host;		// ptr to array holding numbers on host and device to include fixed borders
	double *hd_host, *gd_host, *kd_host;

	int single_double = SINGLE_PRECISION; //1 single, 0 double

	cudaEvent_t start, stop;
	float elapsed_time_ms;

	int T = 128;
	dim3 block(T, 1, 1);
	dim3 grid(((width*width)+ T - 1) / T, 1, 1);
	if (argc != 5) /* argc should be 4 for correct execution */
	{
		/* We print argv[0] assuming it is the program name */
		printf("Wrong parameters. Please use the following format for running.\n");
		printf(" Usage: %s %s %s %s %s", argv[0], "[matrix_size]", "[single|double]", "[divide_val]", "[num_threads]\n");
		exit(EXIT_FAILURE);
	} else {
		//printf("Arguments: %d %d", atoi(argv[1]),	atoi(argv[2]));
		if(atoi(argv[2])!=0)
			single_double = SINGLE_PRECISION;
		else
			single_double = DOUBLE_PRECISION;
		width = atoi(argv[1]);
		block.x = atoi(argv[4]);
		grid.x = ((width*width)/atoi(argv[3]) + block.x - 1) / block.x;

	}
if(single_double==SINGLE_PRECISION)
{
	hs_host = GenMatrix<float>(width, width);
	gs_host = GenMatrix<float>(width, width);
	ks_host = GenMatrix<float>(width, width);

	//PrintMatrix<float>(hs_host, N);
	Init_Cuda<float>(hs_host, gs_host, width, width, SINGLE_PRECISION);
}
else
{
	hd_host = GenMatrix<double>(width, width);
	gd_host = GenMatrix<double>(width, width);
	kd_host = GenMatrix<double>(width, width);
	Init_Cuda<double>(hd_host, gd_host, width, width, DOUBLE_PRECISION);
}

	cudaEventCreate( &start );     // instrument code to measure start time
	cudaEventCreate( &stop );
	cudaEventRecord( start, 0 );
	cudaEventSynchronize( start );

if(single_double==SINGLE_PRECISION)
{
	sor_kernel<float><<<grid,block>>>(hs_device, gs_device, width, width);
	cudaMemcpy(gs_host, gs_device, width*width*sizeof(float) ,cudaMemcpyDeviceToHost);	// copy results back to host, array g
}
else
{
	sor_kernel<double><<<grid,block>>>(hd_device, gd_device, width, width);
	cudaMemcpy(gd_host, gd_device, width*width*sizeof(float) ,cudaMemcpyDeviceToHost);	// copy results back to host, array g
}

	//PrintMatrix(g, width, height);

	cudaEventRecord( stop, 0 );     // instrument code to measue end time
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &elapsed_time_ms, start, stop );

	printf("[ %s ][ %4dx%4d ][ %10d blocks ][ %5d threads ]\t>\t[ %7.3f (ms) ]\n", ((single_double==SINGLE_PRECISION)?"Single Precision":"Double Precision"), width, width, grid.x, block.x, elapsed_time_ms);

if(single_double==SINGLE_PRECISION)
{
	//sor_cpu<float>(hs_host, ks_host, width, height);
//	/PrintMatrix(ks_host, width, height);
	//if(Check(gs_host, ks_host, width, height)==1)
	//	printf("We do it.\n");
	//else
	//	printf("Something is wrong.\n");
	free(hs_host);
	free(gs_host);
	free(ks_host);
	Free_Cuda(SINGLE_PRECISION);
}
else
{
	//sor_cpu<double>(hd_host, kd_host, width, height);
//	/PrintMatrix(kd_host, width, height);
	//if(Check(gd_host, kd_host, width, height)==1)
	//	printf("We do it.\n");
	//else
	//	printf("Something is wrong.\n");
	free(hd_host);
	free(gd_host);
	free(kd_host);
	Free_Cuda(DOUBLE_PRECISION);
}

	return 0;
}
