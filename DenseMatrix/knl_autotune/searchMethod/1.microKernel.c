#include<stdio.h>
#include<assert.h>
#include<malloc.h>
#include<omp.h>
#include"search_10.h"
#include"function.h"

extern data_t* data;

kernel_t** search(int* count){ // count : output parameter. it returns how many kernels are tested
	FILE* fin;
	FILE* resf=NULL;
	int i, j;
	int mu, nu, ku;
	int mb, nb, kb;
	int L1_A, L1_B;
	int type_byte = 8;
	int ni = 17;
	int njr= 4;
	kernel_t **bestKernel1, **bestKernel1_2, **bestKernel2, **bestKernel3;

	/*
	 * get hardware information. fixed part
	 */
	fin = fopen("data/data", "r");
	assert(fin);
	data = readData(fin);

	/*
	 * clear the previous results
	 */
	remove("./munuRes.txt");

	/*
	 * bestKernel for best 5 (mu,nu,mb,kb) sets.
	 * because of 6 space : if there is 5 space, then have to make exception case for final case.
	 *                      so i added 1 more space at last space.
	 */
	bestKernel1 = newKernelList(20);
	bestKernel1_2 = newKernelList(20);
	bestKernel2 = newKernelList(20);
	bestKernel3 = newKernelList(20);


	/*
	 * search part. you have to implement only here
	 *
	 * search part contain these
	 *  1. set variable for the making kernel.
	 *  2. call prepareKernel(kernel_t* kernel). if it return -1, it means kernel compile is failed.
	 *  3. call validTest(kernel_t* kernel). if it return -1, it means kernel result is wrong.
	 *     it can be skipped.
	 *  4. call testKernel->mflop[0] = flopsTest(kernel_t* kernel, int thread). 
	 *  5. *count ++;
	 *  6. if testKernel is faster then bestKernel, change bestKernel
	 */
	kernel_t *testKernel;
	ku = 1;
	int regSize=32*64/type_byte;
	for(nu = 8; nu <= regSize/2; nu+=8){
		for(mu = (regSize/nu)-1; mu>0; mu--){
//			for(kb = 40; kb < 4000; kb*=2)
			{		int kb=(int)sqrt((data->L2size)/type_byte/2/2)*2;
					testKernel=setKernel(mu,nu,0,0,0,kb,20*mu,0,20*nu,0,0,0,0);
					printf("-----------------------1-----------------------------\n");
					if(prepareMicroKernel(testKernel) == -1){ 
						//it means that it is failed to make kernel.
						//it is skipped.
						printf("%c[32m\n", 27);
						printf("skip!!!!!!!!\n");
						printf("mu = %d, nu = %d, ku = %d, mb = %d, nb = %d, kb = %d\n",
								testKernel->mu, testKernel->nu, testKernel->ku,
								testKernel->mb, testKernel->nb, testKernel->kb);
						printf("L1_distanceA = %d, L1_distanceB = %d\n",
								testKernel->prefetchA1, testKernel->prefetchB1);
					printf("npack = %d, ni = %d, njr = %d\n", 
							testKernel->nPack, testKernel->ni, testKernel->njr);
						printf("%c[0m\n", 27);
					}else{
						printf("%c[32m", 27);
						printf("count = %d\n", *count);
						printf("mu = %d, nu = %d, ku = %d, mb = %d, nb = %d, kb = %d\n", 
								testKernel->mu, testKernel->nu, testKernel->ku,
								testKernel->mb, testKernel->nb, testKernel->kb);
						printf("L1_distanceA = %d, L1_distanceB = %d\n", 
								testKernel->prefetchA1, testKernel->prefetchB1);
					printf("npack = %d, ni = %d, njr = %d\n", 
							testKernel->nPack, testKernel->ni, testKernel->njr);
						printf("%c[0m\n", 27);
						testKernel->mflop[0] = microKernelFlopsTest(testKernel, count, 0);
						printf("mflop(check) = %f\n", testKernel->mflop[0]);

						resf=fopen("munuRes.txt","a");
						fprintf(resf,"%d\t%d\t%f\n",testKernel->mu,testKernel->nu,testKernel->mflop[0]);
						fclose(resf);
						/*
						 * bestKernel choose best 5 (nu,mu,mb,kb)set.
						 */
						printf(" bestKernel1 - ");
						putBestKernel(bestKernel1, testKernel, 20);
					}
					printf("-----------------------------------------------------\n");
			}
		}
	}
	printf("-----------------------1_refine-----------------------------\n");
	//kernelListRepeatTest(bestKernel1, 3, 20, count);
	printf("---------------------phase 1 best ---------------------------\n");
	for(i = 0; i < 10; i++){
		printf("%c[32m\n", 27);
		printf("%d : mu = %d, nu = %d, ku = %d, mb = %d, nb = %d, kb = %d\n",
				i, bestKernel1[i]->mu, bestKernel1[i]->nu, bestKernel1[i]->ku,
				bestKernel1[i]->mb, bestKernel1[i]->nb, bestKernel1[i]->kb);
		printf("L1_distanceA = %d, L1_distanceB = %d\n",
				bestKernel1[i]->prefetchA1, bestKernel1[i]->prefetchB1);
			printf("npack = %d, ni = %d, njr = %d\n", 
					bestKernel1[i]->nPack, bestKernel1[i]->ni, bestKernel1[i]->njr);
		printf("%c[0m\n", 27);
	}
	
			return bestKernel1;
}
