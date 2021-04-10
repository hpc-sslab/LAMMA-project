/*
 * P_Helper_STT.cu
 *
 *  Created on: Jun 25, 2013
 *      Author: Nhat-Phuong Tran
 */

#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <omp.h>
#include "../inc/readfile.h"
#include "../inc/logg.h"
#include "../inc/ac.h"
#include "../inc/helper_stt.h"


ACSM_STRUCT2* createACSM() {
	ACSM_STRUCT2 *acsm;
	acsm = acsmNew2(NULL, NULL, NULL);
	if (!acsm) {
		printf("acsm-no memory\n");
		exit(0);
	}
	acsm->acsmFormat = ACF_FULL;
	return acsm;
}

//num_threads =0 , transfer all patterns to 1 acsm
int patternsToACSM(char* filename, ACSM_STRUCT2 *acsm, int num_threads,
		int thread_num) {

	if (thread_num < num_threads) {
		long bufsize = getFileSize(filename);
		char *patterns = (char*) malloc(sizeof(char) * (bufsize + 1));
		if (patterns != NULL)
			readText(filename, patterns, bufsize + 1);
		char *pch = strtok(patterns, " ");
		int nc = 0;
		int i = 0;
		int count = 0;
		while (pch != NULL) {

			if (count == thread_num) {
				acsmAddPattern2(acsm, (unsigned char*) pch, strlen(pch), nc, 0,
						0, 0, (void*) pch, i);
				i++;
				count = (count == (num_threads - 1) ? count = 0 : count);
				count = count + 1;
			} else {
				i++;
				count = (count == (num_threads - 1) ? count = 0 : count);
				count = count + 1;
			}

			pch = strtok(NULL, " ");
		}

		//if (strlen(pch) > pattern_max_len)
		//pattern_max_len = strlen(pch);

		//printf("\n%d",i);

		free(patterns);
		acsmCompile2(acsm, NULL, NULL);
		return 1;
	} else
		return 0;

}
int patternsToACSM(char* filename, ACSM_STRUCT2 *acsm) {

	long bufsize = getFileSize(filename);
	char *patterns = (char*) malloc(sizeof(char) * (bufsize + 1));
	if (patterns != NULL)
		readText(filename, patterns, bufsize + 1);
	char *pch = strtok(patterns, " ");
	int nc = 0;
	int i = 0;

	while (pch != NULL) {

		acsmAddPattern2(acsm, (unsigned char*) pch, strlen(pch), nc, 0, 0, 0,
				(void*) pch, i);
		i++;

		pch = strtok(NULL, " ");
	}

	//if (strlen(pch) > pattern_max_len)
	//pattern_max_len = strlen(pch);

	//printf("\n%d",i);

	free(patterns);
	acsmCompile2(acsm, NULL, NULL);
	return 1;

}
int patternsToACSM(char* filename, ACSM_STRUCT2 *acsm[], int num_threads,
		int *task) {

	long bufsize = getFileSize(filename);
	char *patterns = (char*) malloc(sizeof(char) * (bufsize + 1));
	if (patterns != NULL)
		readText(filename, patterns, bufsize + 1);
	char *pch = strtok(patterns, " ");

	int nc = 0;
	int i = 0;
	int order = 0;
	printf("Contributing patterns to ACSMs...\n");
	while (pch != NULL) {
		for (int thread_num = 0; (thread_num < num_threads) && (pch != NULL);
				thread_num++) {
			//printf("thread_num/num_threads = %d/%d\n",thread_num, num_threads );
			for (int p = 0; (p < num_threads) && (pch != NULL); p++) {

				order = thread_num + task[thread_num * num_threads + p];
				acsmAddPattern2(acsm[order], (unsigned char*) pch, strlen(pch),
						nc, 0, 0, 0, (void*) pch, i);
				i++;
				//printf("\tp = %d, order = %d, added %d patterns\n", p, order, i-1);
				if (p + 1 < num_threads)
					pch = strtok(NULL, " ");
			}
			if (thread_num + 1 < num_threads)
				pch = strtok(NULL, " ");
		}
		pch = strtok(NULL, " ");
	}
	printf("Compiling acsm...\n");
	order = 0;
	for (int thread_num = 0; (thread_num < num_threads); thread_num++) {
		for (int p = 0; p < num_threads; p++) {
			order = thread_num + task[thread_num * num_threads + p];
			acsmCompile2(acsm[order], NULL, NULL);
		}
	}

	free(pch);
	free(patterns);
	return 1;
}
int patternsToACSM2(const char* filename, ACSM_STRUCT2 *acsm[], int num_threads) /*,
		int *task)*/ {

	long bufsize = getFileSize(filename);
	char *patterns = (char*) malloc(sizeof(char) * (bufsize + 1));
	if (patterns != NULL)
		readText(filename, patterns, bufsize + 1);
	char *pch = strtok(patterns, " ");

	int nc = 0;
	int i = 0;
	long count = 0;
	//int order = 0;
	printf("Contributing patterns to ACSMs...\n");
	while (pch != NULL) {
		for (int thread_num = 0; (thread_num < num_threads) && (pch != NULL);
				thread_num++) {
			count = count + 1;
			//printf("Added pattern \"%.*s\" [%ld] => ACSM[%d]\n", strlen(pch), pch, count, thread_num);
			acsmAddPattern2(acsm[thread_num], (unsigned char*) pch, strlen(pch),
					nc, 0, 0, 0, (void*) pch, i);
			i++;
			//printf("\tp = %d, order = %d, added %d patterns\n", p, order, i-1);
			if (thread_num + 1 < num_threads)
				pch = strtok(NULL, " ");
		}
		pch = strtok(NULL, " ");
	}
	printf("Compiling acsm...\n");
	for (int thread_num = 0; (thread_num < num_threads); thread_num++) {
		acsmCompile2(acsm[thread_num], NULL, NULL);
	}

	free(pch);
	free(patterns);
	printf("patternsToACSM2...Done.\n");
	return 1;
}

void fillMatrixDFA(ACSM_STRUCT2 * acsm, acstate_t *arr) {
	int k, i;
	acstate_t * p, state;
	acstate_t ** NextState = acsm->acsmNextState;

	//printf("Print DFA - %d active states\n",acsm->acsmNumStates);
	for (k = 0; k < acsm->acsmNumStates; k++) {
		p = NextState[k];

		if (!p)
			continue;

		for (i = 0; i < (acsm->acsmAlphabetSize + 2); i++) {
			state = p[i];
			arr[k * ((acsm->acsmAlphabetSize) + 2) + i] = state;
		}


	}
}
/*fill DFA Matrix with values from ACSM struct
 *input: acsm
 *output: dfa_arr - dfa matrix in one dimension
 *output: matching_arr - array of matching in format ([state], [order_of_matching_1], [order_of_matching_2], ... [order_of_matching_MATCHING_MAX-1])
 *output: matching_total - total of states which have matching
 */
void fillMatrixDFA(ACSM_STRUCT2 * acsm, acstate_t *dfa_arr, unsigned int *matching_arr, unsigned int *matching_total) {
	int k, i, t;
	acstate_t * p, state;
	acstate_t ** NextState = acsm->acsmNextState;
	ACSM_PATTERN2 *mlist;

	t= 0;
	//printf("Print DFA - %d active states\n",acsm->acsmNumStates);
	for (k = 0; k < acsm->acsmNumStates; k++) {
		p = NextState[k];

		if (!p)
			continue;

		for (i = 0; i < (acsm->acsmAlphabetSize + 2); i++) {
			state = p[i];
			dfa_arr[k * ((acsm->acsmAlphabetSize) + 2) + i] = state;
		}

		mlist = acsm->acsmMatchList[k];
		if(mlist)
		{
			*matching_total = * matching_total + 1;
			matching_arr[t] = k;//state
			for(mlist=acsm->acsmMatchList[k] ; mlist; mlist = mlist->next)
			{
				//printf("%.*s %d", mlist->n, mlist->patrn, GetValueOfPattern(acsm, mlist->patrn, mlist->n));
				matching_arr[t++] = GetValueOfPattern(acsm, mlist->patrn, mlist->n);
			}
		}

		t = t + (MATCHING_MAX - (t % MATCHING_MAX));//make sure t jump to positions which are multiple of MATCHING_MAX

	}
}

void updateMatchingListOnDFA(ACSM_STRUCT2 *acsm, acstate_t *dfa_arr) {
	ACSM_PATTERN2 * mlist;
	int state, j, count;
	for (state = 0; state < acsm->acsmNumStates; state++) //duyet qua cac trang thai
			{
		count = 0; // moi trang thai khoi tao lai bien count
		//lay gia tri tai cot 2 cua trang thai do
		j = dfa_arr[state * (acsm->acsmAlphabetSize + 2) + 1];

		if (j == 1) {
			for (mlist = acsm->acsmMatchList[state]; mlist; mlist =	mlist->next)
			{
				count = count + 1;

			}
		}
		dfa_arr[state * (acsm->acsmAlphabetSize + 2)] = count;
		/*#if TEST_CASE
		 if(count>0)
		 printf("Found %d matching at state %d\n",arr_dfa[state * (acsm->acsmAlphabetSize+2)], state);
		 #endif
		 */
	}
}
unsigned int* createDFA(ACSM_STRUCT2 *acsm) {
	unsigned int *arrDFA = NULL;
	arrDFA = (unsigned int*) malloc(
			sizeof(unsigned int) * acsm->acsmNumStates
					* (acsm->acsmAlphabetSize + 2));
	if (arrDFA) {
		fillMatrixDFA(acsm, arrDFA);
		updateMatchingListOnDFA(acsm, arrDFA);

	}
	return arrDFA;
}
void printDFA(ACSM_STRUCT2 *acsm, unsigned int *dfa_arr) {
	for (int t = 0; t < acsm->acsmNumStates; t++) {
		if (dfa_arr[t * (acsm->acsmAlphabetSize + 2) + 1] == 1) {
			printf("state %d,  \n", t);
			for (int j = 0; j < acsm->acsmAlphabetSize + 2; j++) {
				int k = dfa_arr[t * (acsm->acsmAlphabetSize + 2) + j];
				//if( k !=0 || ((j % (acsm->acsmAlphabetSize+2) == 0) || (j % (acsm->acsmAlphabetSize+2) == 1 )))
				if ((j % (acsm->acsmAlphabetSize + 2) == 0) || (j % (acsm->acsmAlphabetSize + 2) == 1))
					printf("%d  ", k);
				else if (isascii(j) && isprint(j) && k != 0)
					printf("(%c)%d  ", (char) (j - 2), k);

			}
			printf("\n");
		}
	}
}

//void (*pPrintDFA)(ACSM_STRUCT2*) = Print_DFA_Link;

//search on DFA use matching_arr
int searchDFA(acstate_t *dfa_arr, unsigned char *Tx, long tx_len, unsigned char *xlatcase,  int *matching_arr, int *current_state)
{

	unsigned int *ps;
	int i = 0, j ;
	int state, sindex;
	int nfound = 0;


	//if (current_state == NULL);
	//exit(0);

	state = 0;//*current_state;

	for(i= 0; i < tx_len; i++) {
		ps = dfa_arr + state*258;
		sindex = xlatcase[Tx[i]];
		printf("%c\n",sindex);
		printf("ps = %d\n", ps[1]);
		if (ps[1])
		{
			//nfound++;
			//printf("\nnfound = %d", nfound);
			for(j = state * MATCHING_MAX + 1; j < (state +1 ) * MATCHING_MAX; j++)
				if(matching_arr[j] != -1)
					nfound++;

		}
		state = ps[2u+ sindex];
	}

	//check for the last state which was not processed earlier
	ps = dfa_arr + state*258;
	printf("ps = %d\n", ps[1]);
	if (ps[1])
	{
		//nfound++;
		//printf("\nnfound = %d", nfound);
		for(j = state * MATCHING_MAX + 1; j < (state +1 ) * MATCHING_MAX; j++)
			if(matching_arr[j] != -1)
				nfound++;

	}
	return nfound;

}

//search on DFA without matching_arr
int searchDFA(acstate_t *dfa_arr, unsigned char *Tx, long tx_len, unsigned char *xlatcase,  int *current_state)
{

	unsigned int *ps;
	int i = 0 ;
	int state, sindex;
	int nfound = 0;


	//if (current_state == NULL);
	//exit(0);

	state = 0;//*current_state;

	for(i= 0; i < tx_len; i++) {
		ps = dfa_arr + state*258;
		sindex = xlatcase[Tx[i]];
		printf("%c\n",sindex);
		printf("ps = %d\n", ps[1]);
		if (ps[1])
		{
			nfound = nfound + ps[0];

		}
		state = ps[2u+ sindex];
	}

	//check for the last state which was not processed earlier
	ps = dfa_arr + state*258;
	printf("ps = %d\n", ps[1]);
	if (ps[1])
	{
		nfound = nfound + ps[0];

	}
	return nfound;

}

