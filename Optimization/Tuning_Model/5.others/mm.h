
#ifndef MM_H_
#define MM_H_

#define TILE_WIDTH 16


#define M_DIM 512
#define K_DIM 512
#define N_DIM 512
float *Md, *Nd, *Pd;

float *GenMatrix(const unsigned int height, const unsigned int width)
{
	float *matrix;
	const unsigned int M_SIZE = width*height;
	unsigned int i = 0, j = 0;
	matrix  = (float*) malloc(M_SIZE * sizeof(float));
	for(i = 0 ;i < height; i++){
		for(j = 0 ;j < width; j++){
			matrix[i * width + j] = 1.0;//(rand()*1.0)/ RAND_MAX;
		}
	}
	return matrix;
}
int PrintMatrix(float *P, const unsigned int height, const unsigned int width)
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
int MatrixMul(float *P, const float *M, const float *N, const unsigned int m, const unsigned int k, const unsigned int n)
{
	int i, j, kk ;
	float sum, a, b ;
	for(i = 0 ;i < m; i++)
		for(j = 0 ;j < n; j++)
		{	sum  = 0;
			for(kk = 0 ;kk < k; kk++)
			{
				a = M[i * k + kk];
				b = N[kk * n + j];
				sum += a*b;
			}
			P [ i * n + j] = (float)sum;
		}
	return 1;
}
int Check(float *KP, float *CP, const unsigned int height, const unsigned int width)
{
	int i, j;
	float e = 0.001;
	int correct = 1;
	for(i = 0; i < height ; i++)
		for(j = 0; j < width; j++)
		{	if(abs(KP[i * width + j] - CP[i * width + j]) > e)
			{	printf("%.10f %.10f\n", KP[i * width + j], CP[i * width + j]);
				return 0;
			}

		}
	return correct;
}
//Init data
void Init_Cuda(float *M, float *N, const unsigned int m, const unsigned int k, const unsigned int n)
{

	//allocate matrix
	cudaMalloc((void**)&Md, m*k*sizeof(float));
	cudaMemcpy(Md, M, m*k*sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&Nd, k*n*sizeof(float));
	cudaMemcpy(Nd, N, k*n*sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&Pd, m*n*sizeof(float));
	cudaMemset(Pd, 0, m*n*sizeof(float));
}
//Free memory
void Free_Cuda()
{
	cudaFree(Md);
	cudaFree(Nd);
	cudaFree(Pd);
}


#endif /* MM_H_ */
