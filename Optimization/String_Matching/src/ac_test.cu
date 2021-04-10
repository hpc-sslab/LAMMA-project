
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <omp.h>
#include <cuda.h>
#include "../inc/readfile.h"
#include "../inc/logg.h"
#include "../inc/ac.h"
#include "../inc/helper_stt.h"
//#include "../inc/ac_kernels.h"
#include "../inc/functions.h"
#include <helper_cuda.h>
#include <helper_functions.h>

//#define NUM_BANKS 32
#define CPU_NUM_THREADS 4
//#define NUM_STREAMS 2
//#define RUN_VER  1//0 GM, 1 SM
//#define RUN_CASE GPU_SM
//#define DFA_PARTS 1
#define ELE_PER_THREAD 4
#define SM_SIZE 16384//1024//2048//4096//8192//16384//32768
#define LENGTH_PER_THREAD 16//64//256

///void (*Print_Matrix_DFA)(ACSM_STRUCT2 *) = &Print_DFA;
__constant__ unsigned char c_xlatcase[256];

static unsigned char xlatcase[256];

static void initXlatcase() {
	int i;
	for (i = 0; i < 256; i++) {
		xlatcase[i] = (unsigned char) toupper(i);
	}
}
cudaArray* generateCudaArray(unsigned int *dfa_arr, int width, int height);
cudaTextureObject_t generateTexture(cudaArray *cuArray, int width, int height);
__global__ void ac_gm(cudaTextureObject_t texObj, char* d_input, long d_input_len, long chunk_len_per_block, unsigned int pattern_max_len, float* d_results) ;
__global__ void ac_sm(cudaTextureObject_t texObj, char* d_input, long d_input_len, unsigned int pattern_max_len, float* d_results) ;

int main(int argc, char **argv) {
	const char *filename1[] = {
			//"/home/phuong/test/test64KB_d.txt", //0
			//"/home/phuong/test/test512KB_d.txt", //1
			"/home/phuong/test/test1MB.txt", //0
			"/home/phuong/test/test20MB.txt", //1
			"/home/phuong/test/test50MB.txt", //2
			"/home/phuong/test/test100MB.txt", //3
			"/home/phuong/test/test200MB.txt", //4
			"/home/phuong/test/test300MB.txt", //5
			"/home/phuong/test/test500MB.txt", //6
			"/home/phuong/test/test400MB.txt", //7
			"/home/phuong/test/test1KB_d.txt" //8
			//"/home/phuong/test/SampleTest.txt" //10
			};
	const char *filename2[] = {
			"/home/phuong/dict/dict100.txt", //0
			"/home/phuong/dict/dict200.txt", //1
			"/home/phuong/dict/dict500.txt", //2
			"/home/phuong/dict/dict1000.txt", //3
			"/home/phuong/dict/dict2000.txt", //4
			"/home/phuong/dict/dict5000.txt", //5
			"/home/phuong/dict/dict10000.txt", //6
			"/home/phuong/dict/dict20000.txt", //7
			"/home/phuong/dict/dict30000.txt", //8
			"/home/phuong/dict/dict40000.txt", //9
			"/home/phuong/dict/dict50000.txt", //10
			"/home/phuong/dict/dict60000.txt", //11
			"/home/phuong/dict/dict70000.txt", //12
			"/home/phuong/dict/dict80000.txt", //13
			"/home/phuong/dict/dict90000.txt", //14
			"/home/phuong/dict/dict100000.txt", //15
			"/home/phuong/dict/dict150000.txt", //16
			"/home/phuong/dict/dict200000.txt", //17
			"/home/phuong/dict/dict250000.txt", //18
			"/home/phuong/dict/dict300000.txt", //19
			"/home/phuong/dict/dict400000.txt", //20
			"/home/phuong/dict/SampleDict.txt" //20
			};

	int dt, ts, type=1;
	char* h_input;

	char* d_input = NULL;
	//unsigned char* d_xlatecase = NULL;
	float * d_results = NULL;
	float* h_results = NULL;
	long h_input_len;
	long xlatecase_len = 256;
	//long chunk_len = 0;
	unsigned int pattern_max_len = 8;
	//int start_state = 0;
	//unsigned int blocks_arr[] = {2048, 16384, 32768, 32768, 32768, 32768, 32768 };
	//unsigned int threads_arr[] = { 1, 1, 1, 16, 64, 128, 256 };
	unsigned int blocks;
	unsigned int threads;

	//char* h_input_test = NULL;
	//dt = 10; //0,1,2,3,4,5,6,7
	//ts = 2; //0,1,2,3,4,5,6,7,8,

	getParametersFromUser(&type, &dt, &ts);
	dt = dt-1;
	ts = ts-1;

	unsigned int width, height;
	initXlatcase();

	//fill data for input array on host memory
	//h_input = fillInput(filename1[ts]);
	//fillInput(&h_input,filename1[ts]);

	long bufsize = getFileSize(filename1[ts]);
	checkCudaErrors(
			cudaHostAlloc((void**)&h_input, sizeof(char)*(bufsize+1),cudaHostAllocDefault));
	if (h_input) {
		readFilesToArray(filename1[ts], h_input, bufsize, bufsize + 1);
	}

	//for (int i = 0; i < 500; i++)
		//printf("%c", h_input[i]);
	//get the input length
	h_input_len = strlen(h_input);
	//printf("\nh_input_len = %ld\n", h_input_len);

	//fill data for input array on device memory
	d_input = fillDeviceData(h_input, h_input_len);

	//fill data for xlatecase array on device memory
	//d_xlatecase = fillDeviceData(xlatcase, xlatecase_len);
	checkCudaErrors(cudaMemcpyToSymbol(c_xlatcase, xlatcase, sizeof(unsigned char)*xlatecase_len));

	//define used GPU device
	int devID = findCudaDevice(argc, (const char**) argv);

	ACSM_STRUCT2 *acsm[DFA_PARTS];

	unsigned int *arrDFA[DFA_PARTS];



	for (int i = 0; i < DFA_PARTS; i++) {
		acsm[i] = NULL;
		acsm[i] = createACSM();
		arrDFA[i] = NULL;
	}
	//patternsToACSM(filename2[dt], acsm);
	patternsToACSM2(filename2[dt], acsm, DFA_PARTS);

	//pPrintDFA(acsm);
	//Print_Matrix_DFA(acsm[i]);
	for (int i = 0; i < DFA_PARTS; i++) {

		//Print_Matrix_DFA(acsm[i]);
		width = acsm[i]->acsmAlphabetSize + 2;
		height = acsm[i]->acsmNumStates;
		printf("Width of STT : %d\n", width);
		printf("Height of STT : %d\n", height);

		arrDFA[i] = createDFA(acsm[i]);

		//printDFA(acsm[i], arrDFA[i]);
	}

	//printf("Without matching array, found %d patterns\n",searchDFA(arrDFA[1], (unsigned char*)h_input, h_input_len, xlatcase, &start_state));

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("\nCuda Error: %s\n", cudaGetErrorString(error));
	}

	/*for (int i = 0; i < height; i++) {
	 for (int j = 0; j < width; j++)
	 printf("%4d", arrDFA[3][i*258 + j]);
	 printf("\n");
	 }*/

	cudaArray** cuArray = NULL;
	cuArray = (cudaArray**) malloc(DFA_PARTS * sizeof(cudaArray*));

	cudaTextureObject_t *texObj = 0;
	texObj = (cudaTextureObject_t*) malloc(
			DFA_PARTS * sizeof(cudaTextureObject_t));

	for (int i = 0; i < DFA_PARTS; i++) {
		cuArray[i] = generateCudaArray(arrDFA[i], width, height);
		texObj[i] = generateTexture(cuArray[i], width, height);
	}

	StopWatchInterface *timer = NULL;
	sdkCreateTimer(&timer);


	cudaStream_t *streams = (cudaStream_t*) malloc(
			NUM_STREAMS * sizeof(cudaStream_t));

	//omp_set_num_threads(CPU_NUM_THREADS);
	//printf("Number of host CPU = %d\n", omp_get_num_procs());

#if(RUN_VER==0)
	//blocks = 1;//8192;//blocks_arr[ts];
	threads= 32; //threads_arr[ts];
	blocks = create_grid(&threads, h_input_len, 0, LENGTH_PER_THREAD, NUM_STREAMS );
#else
	 /*threads = SM_SIZE / LENGTH_PER_THREAD;
	 if(ts==0)
		 blocks = 1;
	 else
		 blocks = (h_input_len )/ SM_SIZE / NUM_STREAMS;
		 */
	threads= 64; //threads_arr[ts];
	blocks = create_grid(&threads, h_input_len, SM_SIZE, LENGTH_PER_THREAD, NUM_STREAMS );
#endif


	printf("Blocks/stream = %d, threads/stream =  %d\n", blocks,
			threads);

	checkCudaErrors(
			cudaMalloc((void**)&d_results, blocks* sizeof(float )));
	checkCudaErrors(
			cudaMemset(d_results, -1.0f, blocks * sizeof(float )));

	//allocate results array on host memory for getting the results back from device memory
	h_results = (float *) malloc(blocks * sizeof(float));
	memset(h_results, 0.0f, blocks * sizeof(float));

	checkCudaErrors(
				cudaMemcpy(d_results, h_results, sizeof(float)*blocks, cudaMemcpyHostToDevice));

	//testTexture<<<1,1>>>(texObj, width, height);
	sdkStartTimer(&timer);

//#pragma omp parallel for shared(h_input_len, h_input, NUM_STREAMS, d_xlatecase,pattern_max_len , texObj, d_input)
#if(DFA_PARTS>1)
	int k = 0;
#endif
	//checkCudaErrors(cudaMemcpy(d_input , h_input, h_input_len*sizeof(char), cudaMemcpyHostToDevice));
	for (int j = 0; j < NUM_STREAMS; j++) {
		printf("Stream [%d/%d] \n", j, NUM_STREAMS);

		checkCudaErrors(cudaStreamCreate(&(streams[j])));
	}
	for (int j = 0; j < NUM_STREAMS; j++) {
#if (DFA_PARTS>1)
		long offset = k * h_input_len / NUM_STREAMS;

#else
		long offset = j *h_input_len/NUM_STREAMS;
#endif
		cudaMemcpyAsync(d_input + offset, h_input + offset,
				h_input_len * sizeof(char) / NUM_STREAMS,
				cudaMemcpyHostToDevice);


#if(DFA_PARTS>1)

#if(RUN_VER == 0)
		//cudaFuncSetCacheConfig(ac_gm, cudaFuncCachePreferShared);
		//cudaFuncSetCacheConfig(ac_gm, cudaFuncCachePreferEqual);
		cudaFuncSetCacheConfig(ac_gm, cudaFuncCachePreferL1);
		ac_gm<<<blocks, threads, 0, streams[j]>>>(texObj[j],
					d_input + offset, h_input_len / NUM_STREAMS,
					(h_input_len / NUM_STREAMS) / blocks,
					pattern_max_len, d_results);
#else
			cudaFuncSetCacheConfig(ac_sm, cudaFuncCachePreferShared);
//			cudaFuncSetCacheConfig(ac_sm, cudaFuncCachePreferEqual);
//			cudaFuncSetCacheConfig(ac_sm, cudaFuncCachePreferL1);
			ac_sm<<<blocks, threads, (SM_SIZE + threads*pattern_max_len)  * sizeof(char), streams[j]>>>(texObj[j],
				d_input+offset, h_input_len/NUM_STREAMS, pattern_max_len,
				d_results);
#endif


		k++;


#else
#if(RUN_VER==0)
		//cudaFuncSetCacheConfig(ac_gm, cudaFuncCachePreferShared);
		//cudaFuncSetCacheConfig(ac_gm, cudaFuncCachePreferEqual);
		cudaFuncSetCacheConfig(ac_gm, cudaFuncCachePreferL1);

		ac_gm<<<blocks,threads, 0, streams[j] >>>(texObj[0] ,d_input + offset, h_input_len/NUM_STREAMS, (h_input_len/NUM_STREAMS)/blocks, pattern_max_len, d_results);

#else
		cudaFuncSetCacheConfig(ac_sm, cudaFuncCachePreferShared);
//		cudaFuncSetCacheConfig(ac_sm, cudaFuncCachePreferEqual);
//		cudaFuncSetCacheConfig(ac_sm, cudaFuncCachePreferL1);

		ac_sm<<<blocks, threads, (SM_SIZE + threads*pattern_max_len) * sizeof(char), streams[j]>>>(texObj[0],
						d_input+offset, h_input_len/NUM_STREAMS, pattern_max_len,
						d_results);

#endif
#endif

		checkCudaErrors(cudaMemcpy(h_results, d_results, blocks * sizeof(float ), cudaMemcpyDeviceToHost));
		printf("Stream %d ok!\n", j);

	}

	sdkStopTimer(&timer);
	displayTime(sdkGetTimerValue(&timer));
	sdkDeleteTimer(&timer);

	for (int i = 0; i < NUM_STREAMS; i++) {
		cudaStreamDestroy(streams[i]);
	}
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("\nCuda Error: %s\n", cudaGetErrorString(error));
	}




	//printf("%ld", generateNumOfBlocks(64*1024,threads,20));
	//for (int i = 0; i < NUM_STREAMS; i++) {
		//acsmFree2(acsm[i]);
		//free(arrDFA[i]);
	//}
	cudaFree(d_results);
	free(h_results);

	//free(h_input);
	cudaFreeHost(h_input);
	cudaFree(d_input);
	//cudaFree(d_xlatecase);

}

cudaArray* generateCudaArray(unsigned int *dfa_arr, int width, int height) {
	cudaArray* cuArray = NULL;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned int>();
	checkCudaErrors(cudaMallocArray(&cuArray, &channelDesc, width, height));
	checkCudaErrors(
			cudaMemcpyToArray(cuArray, 0, 0, (unsigned int*)dfa_arr, width*height*sizeof(unsigned int), cudaMemcpyHostToDevice));
	printf("generateCudaArray...Done.\n");
	return cuArray;
}
cudaTextureObject_t generateTexture(cudaArray *cuArray, int width, int height) {

	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cuArray;

	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeClamp;
	texDesc.addressMode[1] = cudaAddressModeClamp;
	texDesc.filterMode = cudaFilterModePoint;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = false;

	cudaTextureObject_t texObj = 0;
	checkCudaErrors(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));
	printf("generateTexture...Done.\n");
	return texObj;
}
__global__ void ac_gm(cudaTextureObject_t texObj, char* d_input,
		long d_input_len, long chunk_len_per_block,
		unsigned int pattern_max_len, float* d_results) {

	long chunk_len = chunk_len_per_block / blockDim.x;
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	long is = idx * chunk_len;
	long ie =
			(is + chunk_len + pattern_max_len) > d_input_len ?
					d_input_len : (is + chunk_len + pattern_max_len);
	int state, sindex;
	long thread_rs;
	state = 0; //*current_state;
	if(threadIdx.x == 0)
		d_results[blockIdx.x] = 0;

	thread_rs = 0;
	for (int i = is; i < ie; i++) {
		sindex = c_xlatcase[d_input[i]];
		if (tex2D<int>(texObj, 1, state) == 1) {
			thread_rs += tex2D<int>(texObj, 0, state);

		}
		state = tex2D<int>(texObj, 2u + sindex, state);
	}

	__syncthreads();
	atomicAdd(&d_results[blockIdx.x], (float) thread_rs);



}
__global__ void ac_sm(cudaTextureObject_t texObj, char* d_input,
		long d_input_len, unsigned int pattern_max_len, float* d_results) {

	unsigned int is;
	//unsigned long pos_input =0;
	extern __shared__ unsigned char s_data[];
	//unsigned int off1 = 0;

	unsigned char *s_bytes = &s_data[0];


	unsigned int thread_len = (threadIdx.x == (gridDim.x * blockDim.x - 1)) ? LENGTH_PER_THREAD : (LENGTH_PER_THREAD + pattern_max_len);

	unsigned int times = 0;
	unsigned int times_total = thread_len / ELE_PER_THREAD;
	unsigned long pos = 0;
	while(times < times_total)
	{
		pos = threadIdx.x*ELE_PER_THREAD + times*blockDim.x*ELE_PER_THREAD;
		*(unsigned int*) (&s_bytes[pos]) = *(unsigned int*) (&d_input[pos]);
		times++;
	}
	__syncthreads();

	int state, sindex;
	unsigned int thread_sum = 0;

	thread_sum = 0;
	times = 0 ;
	//pos = (threadIdx.x % ELE_PER_THREAD) * LENGTH_PER_THREAD *times_total +   ((threadIdx.x / ELE_PER_THREAD)*ELE_PER_THREAD);
	pos = (threadIdx.x % ELE_PER_THREAD) * thread_len *times_total +   ((threadIdx.x / ELE_PER_THREAD)*ELE_PER_THREAD);
	while(times < times_total){
		is = pos + times* thread_len;//LENGTH_PER_THREAD;
		//printf("(%d)", is);
		for (int i = is; i < is + ELE_PER_THREAD; i++) {
			//printf("%c", s_bytes[i]);
			sindex = c_xlatcase[s_bytes[i]];
			if (tex2D<int>(texObj, 1, state) == 1) {
				thread_sum += tex2D<int>(texObj, 0, state);
			}
			state = tex2D<int>(texObj, 2u + sindex, state);

		}
		times++;
	}

	__syncthreads();
	atomicAdd(&d_results[blockIdx.x], (float) thread_sum);


}

/*__global__ void ac_sm(cudaTextureObject_t texObj, char* d_input,
		long d_input_len, unsigned int pattern_max_len, float* d_results) {

	unsigned int is;
	//unsigned long pos_input =0;
	extern __shared__ unsigned char s_data[];
	//unsigned int off1 = 0;

	unsigned char *s_bytes = &s_data[0];


	unsigned int thread_len = (threadIdx.x == (gridDim.x * blockDim.x - 1)) ? LENGTH_PER_THREAD : (LENGTH_PER_THREAD + pattern_max_len);

	unsigned int times = 0;
	unsigned int times_total = thread_len / ELE_PER_THREAD;
	unsigned long pos = 0;
	while(times < times_total)
	{
		pos = threadIdx.x*ELE_PER_THREAD + times*blockDim.x*ELE_PER_THREAD;
		*(unsigned int*) (&s_bytes[pos]) = *(unsigned int*) (&d_input[pos]);
		times++;
	}
	__syncthreads();

	int state, sindex;
	unsigned int thread_sum = 0;

	thread_sum = 0;
	times = 0 ;
	//pos = (threadIdx.x % ELE_PER_THREAD) * LENGTH_PER_THREAD *times_total +   ((threadIdx.x / ELE_PER_THREAD)*ELE_PER_THREAD);
	pos = (threadIdx.x % ELE_PER_THREAD) * thread_len *times_total +   ((threadIdx.x / ELE_PER_THREAD)*ELE_PER_THREAD);
	while(times < times_total){
		is = pos + times* thread_len;//LENGTH_PER_THREAD;
		//printf("(%d)", is);
		for (int i = is; i < is + ELE_PER_THREAD; i++) {
			//printf("%c", s_bytes[i]);
			sindex = c_xlatcase[s_bytes[i]];
			if (tex2D<int>(texObj, 1, state) == 1) {
				thread_sum += tex2D<int>(texObj, 0, state);
			}
			state = tex2D<int>(texObj, 2u + sindex, state);

		}
		times++;
	}

	__syncthreads();
	atomicAdd(&d_results[blockIdx.x], (float) thread_sum);


}*/

