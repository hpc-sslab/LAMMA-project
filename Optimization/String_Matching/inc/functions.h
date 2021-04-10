#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include "readfile.h"



#define NUM_BANKS 32
#define CPU_NUM_THREADS 4
#define NUM_STREAMS 2
#define RUN_VER 0 //0 GM, 1 SM
//#define RUN_CASE GPU_SM
#define DFA_PARTS (NUM_STREAMS) 

long generateNumOfBlocks(long input_len, long num_threads,	unsigned int chunk_len);
int* generateTaskArray(int num_threads) ;
int* generateTaskArray2(int num_threads) ;
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
char* fillInput(char* filename) ;
char* fillInput(long input_len, char c) ;
char* fillDeviceData(char* h_data, long data_len) ;
unsigned char* fillDeviceData(unsigned char* h_data, long data_len) ;

char* fillDeviceData(long input_len, char c) ;
long calcChunkLength(unsigned int num_blocks, unsigned int num_threads,	long input_len) ;
void getParametersFromUser(int *ex_type, int *dt, int *ts) ;
float sum_found_patterns(float *h_results, int len) ;
unsigned int create_grid(unsigned int *num_threads, unsigned long input_length, unsigned int sm_size, unsigned int length_per_thread, unsigned int num_streams);
void displayTime(float t);
