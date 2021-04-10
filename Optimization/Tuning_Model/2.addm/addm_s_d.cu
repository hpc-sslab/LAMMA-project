#include<cuda.h>
#include<stdio.h>
#include<stdlib.h>
#include<iostream>

#define CHECK 0

const unsigned int SINGLE_PRECISION = 1;
const unsigned int DOUBLE_PRECISION = 0;
float *SMd, *SNd, *SPd;
double *DMd, *DNd, *DPd;
const unsigned int WIDTH = 1024;


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
void Init_Cuda(T *M, T *N, const unsigned int width, const unsigned int height, bool sp)
{
	const unsigned int size = width*height*sizeof(T);
	//allocate matrix
	if(sp==SINGLE_PRECISION){
		cudaMalloc((void**)&SMd, size);
		cudaMemcpy(SMd, M, size, cudaMemcpyHostToDevice);
		cudaMalloc((void**)&SNd, size);
		cudaMemcpy(SNd, N, size,cudaMemcpyHostToDevice);
		cudaMalloc((void**)&SPd, size);
		cudaMemset(SPd, 0, size);
	}
	else
	{
		cudaMalloc((void**)&DMd, size);
		cudaMemcpy(DMd, M, size, cudaMemcpyHostToDevice);
		cudaMalloc((void**)&DNd, size);
		cudaMemcpy(DNd, N, size,cudaMemcpyHostToDevice);
		cudaMalloc((void**)&DPd, size);
		cudaMemset(DPd, 0, size);
	}
}
//Free memory
void Free_Cuda(bool sp)
{
	if(sp==SINGLE_PRECISION){
		cudaFree(SMd);
		cudaFree(SNd);
		cudaFree(SPd);
	}
	else
	{
		cudaFree(DMd);
		cudaFree(DNd);
		cudaFree(DPd);
	}
}
//kernel function
template<typename T>
__global__ void MatrixAddKernel(T *P, const T *M, const T *N, const unsigned int width)
{

	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int length = width * width;
	while (i < length) {
		P[i] = M[i] + N[i];
		i += gridDim.x * blockDim.x;
	}
}

template<typename T>
int MatrixAdd(T *P, const T *M, const T *N, const unsigned int n)
{
	int i, j ;
	for(i = 0 ;i < n; i++)
		for(j = 0 ;j < n; j++)
		{	P [ i* n + j] = M[ i* n + j] + N[ i* n + j];
		}
	return 0;
}
template<typename T>
int Check(const T *KP, const T *CP, const unsigned int n)
{
	int i, j;
	T e = 0.001;
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
	bool sp = 1;
	float *SM, *SN, *SKP, *SCP;
	double *DM, *DN, *DKP, *DCP;
	cudaEvent_t start, stop;
	float elapsedTime;
	unsigned int width;

	width = WIDTH;

	//create number of blocks and number of threads
	int Thr = 128;
	dim3 block(Thr, 1, 1);
	dim3 grid(((width*width)+ Thr - 1) / Thr, 1, 1);
	if (argc != 5)
	{
		/* We print argv[0] assuming it is the program name */
		printf("Wrong parameters. Please use the following format for running.\n");
		printf(" Usage: %s %s %s %s %s", argv[0], "[matrix_size]", "[single|double]", "[divide_val]", "[num_threads]\n");
		exit(EXIT_FAILURE);
	} else {
		width =  atoi(argv[1]);
		sp = atoi(argv[2]);
		block.x = atoi(argv[4]);
		grid.x = ((width*width)/atoi(argv[3]) + block.x - 1) / block.x;
		if(atoi(argv[2])!=0)
			sp = SINGLE_PRECISION;
		else
			sp = DOUBLE_PRECISION;

	}
	//for using MatrixMul_Kernel_Tiled_SM kernel
	//block.x = TILE_WIDTH; block.y=TILE_WIDTH;
	//grid.x = WIDTH/TILE_WIDTH; grid.y = WIDTH/TILE_WIDTH;

	//initialize host memory
	if(sp==SINGLE_PRECISION)
	{

		SM = GenMatrix<float>(width, width);
		//PrintMatrix(M, width, width);
		SN = GenMatrix<float>(width, width);
		//PrintMatrix(N, width, width);
		SKP = GenMatrix<float>(width, width);

		SCP = GenMatrix<float>(width, width);

		//initialize device memory
		Init_Cuda<float>(SM, SN, width, width, SINGLE_PRECISION);
	}
	else
	{

		DM = GenMatrix<double>(width, width);
		//PrintMatrix(M, width, width);
		DN = GenMatrix<double>(width, width);
		//PrintMatrix(N, width, width);
		DKP = GenMatrix<double>(width, width);

		DCP = GenMatrix<double>(width, width);

		//initialize device memory
		Init_Cuda<double>(DM, DN, width, width, DOUBLE_PRECISION);
	}


    //create cudaEvent start and stop to record elapsed time
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//record start time to start event
	cudaEventRecord(start, 0);

	//launch kernel
	if(sp==SINGLE_PRECISION)
	{
		MatrixAddKernel<float><<<grid, block>>>(SPd, SMd, SNd, width);
	}
	else
	{
		MatrixAddKernel<double><<<grid, block>>>(DPd, DMd, DNd, width);
	}

	//record start time to stop event
	cudaEventRecord(stop, 0);
	//synchronize the stop event
	cudaEventSynchronize(stop);
	//calculate the elapsed time
	cudaEventElapsedTime(&elapsedTime, start, stop);

	//destroy the start and stop event
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	//copy data from device memory to host memory
	if(sp==SINGLE_PRECISION)
		cudaMemcpy(SKP, SPd, width*width*sizeof(float), cudaMemcpyDeviceToHost);
	else
		cudaMemcpy(DKP, DPd, width*width*sizeof(double), cudaMemcpyDeviceToHost);
	//PrintMatrix(P, width, width);

	//print runtime
	printf("[ %s ][ %4dx%4d ][ %10d blocks ][ %5d threads ]\t>\t[ %7.3f (ms) ]\n", ((sp==SINGLE_PRECISION)?"Single Precision":"Double Precision"), width, width, grid.x, block.x, elapsedTime);

#if (CHECK==1)
	if(sp==SINGLE_PRECISION)
	{
		/*printf("M >>>>>>>>>>>>>>>>>>>>>>>>>>\n");
		PrintMatrix<float>(SM, width, width);
		printf("N >>>>>>>>>>>>>>>>>>>>>>>>>>\n");
		PrintMatrix<float>(SN, width, width);
		printf("KP >>>>>>>>>>>>>>>>>>>>>>>>>>\n");
		PrintMatrix<float>(SKP, width, width);
		*/
		MatrixAdd<float>(SCP, SM, SN, width);
		//printf("CP >>>>>>>>>>>>>>>>>>>>>>>>>>\n");
		//PrintMatrix<float>(CP, width, width);

		if(Check<float>(SKP, SCP, width))
			printf("We do it.\n");
		else
			printf("Something is wrong.\n");

	}
	else
	{
		/*printf("M >>>>>>>>>>>>>>>>>>>>>>>>>>\n");
		PrintMatrix<double>(DM, width, width);
		printf("N >>>>>>>>>>>>>>>>>>>>>>>>>>\n");
		PrintMatrix<double>(DN, width, width);
		printf("KP >>>>>>>>>>>>>>>>>>>>>>>>>>\n");
		PrintMatrix<double>(DKP, width, width);
		*/
		MatrixAdd<double>(DCP, DM, DN, width);
		//printf("CP >>>>>>>>>>>>>>>>>>>>>>>>>>\n");
		//PrintMatrix<double>(DCP, width, width);

		if(Check<double>(DKP, DCP, width))
			printf("We do it.\n");
		else
			printf("Something is wrong.\n");

	}
#endif
	//free host memory
	if(sp==SINGLE_PRECISION)
	{
		free(SM);
		free(SN);
		free(SKP);
		free(SCP);
	//free device memory
		Free_Cuda(sp);
	}
	else
	{
		free(DM);
		free(DN);
		free(DKP);
		free(DCP);
		//free device memory
		Free_Cuda(sp);
	}
	return 0;
}
