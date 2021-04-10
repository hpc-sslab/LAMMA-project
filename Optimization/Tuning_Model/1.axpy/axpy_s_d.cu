#include <stdio.h>
#include <iostream>


#define CHECK 0

const unsigned int SINGLE_PRECISION = 1;
const unsigned int DOUBLE_PRECISION = 0;


template<typename T>
__global__ void axpy(const unsigned int n, const T a, const T *x, T *y) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	while (i < n) {
		y[i] = a * x[i] + y[i];
		i += blockDim.x * gridDim.x;
	}
	//if (i < n) y[i] = a*x[i] + y[i];
}

int main(int argc, char * argv[]) {
	unsigned int N = 1024 * 1024 * 32;
	float *sx_host, *sy_host;
	double *dx_host, *dy_host;
	float *sx_device, *sy_device;
	double *dx_device, *dy_device;
	int sp = SINGLE_PRECISION;
	cudaEvent_t start, stop;
	float elapsedTime;

	int T = 128;
	dim3 block(T, 1, 1);
	dim3 grid((N + T - 1) / T, 1, 1);
	if (argc != 5) /* argc should be 2 for correct execution */
	{
		printf("Wrong parameters. Please use the following format for running.\n");
		printf(" Usage: %s %s %s %s %s", argv[0], "[problem_size]", "[single|double]", "[divide_val]", "[num_threads]\n");
		exit(EXIT_FAILURE);
	} else {
		N  = atoi(argv[1])*1024*1024;
		block.x = atoi(argv[4]);
		grid.x = (N / atoi(argv[3]) + block.x - 1) / block.x;
		sp = atoi(argv[2]);
		if(sp!=0)
			sp = SINGLE_PRECISION;
		else
			sp= DOUBLE_PRECISION;
	}
if(sp==SINGLE_PRECISION)
{
	sx_host = (float*) malloc(N * sizeof(float));
	sy_host = (float*) malloc(N * sizeof(float));

	cudaMalloc(&sx_device, N * sizeof(float));
	cudaMalloc(&sy_device, N * sizeof(float));
	for (int i = 0; i < N; i++) {
			sx_host[i] = 1.0f;
			sy_host[i] = 2.0f;
		}
	cudaMemcpy(sx_device, sx_host, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(sy_device, sy_host, N * sizeof(float), cudaMemcpyHostToDevice);
}
else
{
	dx_host = (double*) malloc(N * sizeof(double));
	dy_host = (double*) malloc(N * sizeof(double));

	cudaMalloc(&dx_device, N * sizeof(double));
	cudaMalloc(&dy_device, N * sizeof(double));

	for (int i = 0; i < N; i++) {
			dx_host[i] = 1.0f;
			dy_host[i] = 2.0f;
		}

	cudaMemcpy(dx_device, dx_host, N * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dy_device, dy_host, N * sizeof(double), cudaMemcpyHostToDevice);
}

		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		cudaFree(0);
		cudaEventRecord(start, 0);
if(sp==SINGLE_PRECISION)
{
	axpy<float><<<grid, block>>>(N, 2.0f, sx_device, sy_device);
}
else
{
	axpy<double><<<grid, block>>>(N, 2.0f, dx_device, dy_device);
}

		cudaEventRecord(stop, 0);

		cudaEventSynchronize(stop);

		cudaEventElapsedTime(&elapsedTime, start, stop);

		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		printf("[ %s ][ %4d MB ][ %10d blocks ][ %5d threads ]\t>\t[ %7.3f (ms) ]\n", ((sp==SINGLE_PRECISION)?"Single Precision":"Double Precision"), N / 1024/1024, grid.x, block.x, elapsedTime);

		if(sp==SINGLE_PRECISION)
		{
			cudaMemcpy(sy_host, sy_device, N * sizeof(float), cudaMemcpyDeviceToHost);
		}
		else
		{
			cudaMemcpy(dy_host, dy_device, N * sizeof(double), cudaMemcpyDeviceToHost);
		}

#if(CHECK==1)

	printf("\nN = %ld\n", N);
	if(sp==SINGLE_PRECISION)
	{
		float maxError = 0.0f;
		for (int i = 0; i < N; i++)
			maxError = max((float) maxError, (float) abs(sy_host[i] - 4.0f));
		printf("Max error: %fn\n", maxError);
	}
	else
	{
		double maxError = 0.0f;
		for (int i = 0; i < N; i++)
			maxError = max((double) maxError, (double) abs(dy_host[i] - 4.0f));
		printf("Max error: %fn\n", maxError);
	}


#endif

if(sp==SINGLE_PRECISION)
{
	cudaFree(sx_device);
	cudaFree(sy_device);
	free(sx_host);
	free(sy_host);
}
else
{
	cudaFree(dx_device);
	cudaFree(dy_device);
	free(dx_host);
	free(dy_host);
}
	cudaDeviceReset();



}
