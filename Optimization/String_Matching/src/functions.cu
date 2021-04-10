
#include "../inc/functions.h"

long generateNumOfBlocks(long input_len, long num_threads,
		unsigned int chunk_len) {
	long num_blocks = input_len / num_threads / chunk_len;
	while (!((num_blocks > 0) && ((num_blocks & (num_blocks - 1)) == 0))) {
		chunk_len++;
		num_blocks = input_len / num_threads / chunk_len;
	}
	printf("Chunk len = %d\n", chunk_len);
	return num_blocks;
}

int* generateTaskArray(int num_threads) {
	int* task = NULL;
	task = (int*) malloc(num_threads * sizeof(int));
	if (task) {
		for (int t = 0; t < num_threads; t++) {
			//	task[t * num_threads + i] = num_threads - 1 - i - t;
			task[t] = t;
		}
	}

	return task;
}
int* generateTaskArray2(int num_threads) {
	int* task = NULL;
	task = (int*) malloc(num_threads * num_threads * sizeof(int));
	if (task) {
		for (int t = 0; t < num_threads; t++)
			for (int i = 0; i < num_threads; i++) {
				task[t * num_threads + i] = num_threads - 1 - i - t;
				//task[t] = t;
			}
	}

	return task;
}
/*
 void fillInput(char **h_input, char *filename)
 {
 long bufsize = getFileSize(filename);
 checkCudaErrors(cudaHostAlloc((void**)&h_input, sizeof(char)*(bufsize+1),cudaHostAllocDefault));
 if (h_input) {
 readFilesToArray(filename, h_input, bufsize, bufsize + 1);
 }

 }
 */
char* fillInput(char* filename) {
	char* buffer = NULL;
	long bufsize = getFileSize(filename);
	buffer = (char*) malloc(sizeof(char) * (bufsize + 1));

	if (buffer) {
		readFilesToArray(filename, buffer, bufsize, bufsize + 1);
	}
	return buffer;
}
char* fillInput(long input_len, char c) {
	char* buffer = NULL;
	buffer = (char*) malloc(sizeof(char) * input_len);
	memset(buffer, c, sizeof(char) * input_len);
	return buffer;
}
char* fillDeviceData(char* h_data, long data_len) {
	char* d_input = NULL;
	checkCudaErrors(cudaMalloc((void**)&d_input,sizeof(char)*data_len));
	checkCudaErrors(
			cudaMemcpy(d_input, h_data, sizeof(char)*data_len, cudaMemcpyHostToDevice));
	return d_input;
}
unsigned char* fillDeviceData(unsigned char* h_data, long data_len) {
	unsigned char* d_input = NULL;
	checkCudaErrors(
			cudaMalloc((void**)&d_input,sizeof(unsigned char)*data_len));
	checkCudaErrors(
			cudaMemcpy(d_input, h_data, sizeof(unsigned char)*data_len, cudaMemcpyHostToDevice));
	printf("fillDeviceData...Done.\n");
	return d_input;
}

char* fillDeviceData(long input_len, char c) {
	char* d_input = NULL;
	checkCudaErrors(cudaMalloc((void**)&d_input, sizeof(char)*input_len));
	checkCudaErrors(cudaMemset(d_input, c, sizeof(char)*input_len));
	return d_input;
}

long calcChunkLength(unsigned int num_blocks, unsigned int num_threads,
		long input_len) {
	return input_len / (num_blocks * num_threads);

}
void getParametersFromUser(int *ex_type, int *dt, int *ts) {

	//int type = 1;
	int dict = 1;
	int test = 1;

	/*printf("Please choose the parallelization approach(1=>2):\n");
	 printf("   [1] => Global Memory only\n");
	 printf("   [2] => Shared Memory\n");
	 printf("Entered approach:");
	 scanf("%d", &type);
	 if (type != 1 && type != 2)
	 *ex_type = 1;
	 else
	 *ex_type = type;
	 */

	int P[11] = {100, 200, 500, 1000, 2000, 5000, 10000, 20000, 30000, 40000, 50000};
	int L[7] = {1, 20, 50, 100, 200, 300, 500};
	printf("Please choose the number of patterns(1=>11):\n");
	printf("   [1] =>   100 patterns\n");
	printf("   [2] =>   200 patterns\n");
	printf("   [3] =>   500 patterns\n");
	printf("   [4] =>  1000 patterns\n");
	printf("   [5] =>  2000 patterns\n");
	printf("   [6] =>  5000 patterns\n");
	printf("   [7] => 10000 patterns\n");
	printf("   [8] => 20000 patterns\n");
	printf("   [9] => 30000 patterns\n");
	printf("  [10] => 40000 patterns\n");
	printf("  [11] => 50000 patterns\n");

	printf("Entered number of patterns:");
	scanf("%d", &dict);
	if (dict >= 1 && dict <= 11)
		*dt = dict;
	else
		*dt = 1;

	printf("\nNumber of patterns = %d\n", P[*dt-1]);
	printf("Please choose the the length of input string(1=>7):\n");
	//printf("   [1] =>  64 KB\n");
	//printf("   [2] => 512 KB\n");
	printf("   [1] =>   1 MB\n");
	printf("   [2] =>  20 MB\n");
	printf("   [3] =>  50 MB\n");
	printf("   [4] => 100 MB\n");
	printf("   [5] => 200 MB\n");
	printf("   [6] => 300 MB\n");
	printf("   [7] => 500 MB\n");

	printf("Entered input length:");
	scanf("%d", &test);
	if (test >= 1 && test <= 7)
		*ts = test;
	else
		*ts = 1;
	printf("\nText Length = %d MB\n", L[*ts-1] );
}
float sum_found_patterns(float *h_results, int len) {
	float sum = 0.0;
	for (int i = 0; i < len; i++)
		sum = sum + h_results[i];
	return sum;
}
void displayTime(float t)
{
#if(RUN_VER==0)
	printf("++++++++++++++++++++++++++++++++++\n"
				"Using global memory.\nProcessing time: %f (ms)\n", t);
#else
	printf("++++++++++++++++++++++++++++++++++\n"
				"Using shared memory.\nProcessing time: %f (ms)\n", t);
#endif
}
//create the number of blocks when we know the number of threads, the input_length and shared memory size
//shared memory size = 0 means we don't use the shared memory
unsigned int create_grid(unsigned int *num_threads, unsigned long input_length, unsigned int sm_size, unsigned int length_per_thread, unsigned int num_streams)
{
	//unsigned int multiple 	= 2;
	//unsigned int roundDown;
	unsigned int num_blocks;
	unsigned int t = 0;
	unsigned int length_per_block = 0;
	if(sm_size == 0)
	{
		t = *num_threads;
		length_per_block = t * length_per_thread;
		if(num_streams == 1)
		{
			//roundDown = ((int)(input_length / length_per_block) / multiple) * multiple;
			//num_blocks = roundDown + multiple;
			num_blocks = (int)(input_length / length_per_block);
		}
		else
		{
			//roundDown = ((int)((input_length / num_streams) / length_per_block) / multiple) * multiple;
			//num_blocks = roundDown + multiple;
			num_blocks = (int)(input_length / length_per_block / num_streams);
		}
	}
	else
	{	if(num_streams == 1)
		{
			t = sm_size / length_per_thread;
			//roundDown = ((int)(input_length / sm_size) / multiple) * multiple;
			//num_blocks = roundDown + multiple;
			num_blocks = (int)(input_length / sm_size);
		}
		else
		{
			t = sm_size / length_per_thread;
			//roundDown = ((int)((input_length / sm_size) / sm_size) / multiple) * multiple;
			//num_blocks = roundDown + multiple;
			num_blocks = (int)(input_length / sm_size / num_streams);
		}
		*num_threads = t;
	}
	printf("create_grid...Done.\n");
	return num_blocks;

}
