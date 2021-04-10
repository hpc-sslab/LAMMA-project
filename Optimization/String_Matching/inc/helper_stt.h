

#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <omp.h>

#define MATCHING_MAX 3 //assumed maximum of matching

ACSM_STRUCT2* createACSM();

//num_threads =0 , transfer all patterns to 1 acsm
int patternsToACSM(char* filename, ACSM_STRUCT2 *acsm, int num_threads,	int thread_num);
int patternsToACSM(char* filename, ACSM_STRUCT2 *acsm);
int patternsToACSM(char* filename, ACSM_STRUCT2 *acsm[], int num_threads,
		int *task) ;
int patternsToACSM2(const char* filename, ACSM_STRUCT2 *acsm[], int num_threads);//,int *task) ;
void fillMatrixDFA(ACSM_STRUCT2 * acsm, acstate_t *arr) ;
void updateMatchingListOnDFA(ACSM_STRUCT2 *acsm, acstate_t *dfa_arr) ;
unsigned int* createDFA(ACSM_STRUCT2 *acsm) ;
void printDFA(ACSM_STRUCT2 *acsm, unsigned int *dfa_arr) ;
//search on DFA without matching_arr
int searchDFA(acstate_t *dfa_arr, unsigned char *Tx, long tx_len, unsigned char *xlatcase,  int *current_state);
//search on DFA use matching_arr
//int searchDFA(acstate_t *dfa_arr, unsigned char *Tx, long tx_len, unsigned char *xlatcase,  int *matching_arr, int *current_state);
