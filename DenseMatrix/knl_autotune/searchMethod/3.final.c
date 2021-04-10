#include<stdio.h>

#include<assert.h>
#include<malloc.h>
#include<omp.h>
#include"search_10.h"
#include"function.h"

extern data_t* data;

kernel_t** search(int* count){ // count : output parameter. it returns how many kernels are tested
	FILE* fin;
	int i, j,ii,jj;
	int mu_max, mu_min;
	int mu, nu, ku;
	int mb, nb, kb;
	int L1_A, L1_B;
	int type_byte = 8;
	int ni = 17;
	int njr= 4;
	kernel_t **bestKernel1, **bestKernel1_2, **bestKernel2, **bestKernel3;
	double L2inf, L2sup;
	/*
	 * get hardware information. fixed part
	 */
	fin = fopen("data/data", "r");
	assert(fin);
	data = readData(fin);

	/*
	 * set kernel list for each step's ranking.
	 */
	bestKernel1 = newKernelList(20);
	bestKernel1_2 = newKernelList(20);
	bestKernel2 = newKernelList(20);
	bestKernel3 = newKernelList(20);

	int muList[5]={31,13,9,8,6};
	int	nuList[5]={8,16,24,24,32};

	L2inf=0.3-0.05;
	L2sup=0.6+0.05;


	/*
	 * used funtions descriptions
	 *
	 * setKernel() : set kernel parameter
	 * prepareKernel() : generate kernel code & compile using parameters
	 * validTest() : check valid. kernel must be compiled before checking.
	 * flopsTest() : check flops. kernel must be compiled before checking.
	 * putBestKernel() : put the kernel to kernel list. it sort by flops.
	 * kernelListRepeatTest() : repeat the test on that kernel lists.
	 *
	 */
	kernel_t *testKernel;
	ku = 1;
	int regSize=32*64/type_byte;
	//	for(nu = 16; nu <= data->numberOfRegister; nu+=16)
//	for (ii=0; ii<5; ii++)
	{ ii=0;
		mu=muList[ii];
		nu=nuList[ii];
		{
			int mbBase = (int)sqrt((data->L2size)/type_byte/2/2);
			int mbInt = mbBase/10;
		//	for(i = mbInt; i<=mbInt*10; i+=mbInt){
			for(i = 217; i>=93; i-=31){
				for(mb = mu; mb<i ; mb+=mu);
				//(mb*kb + 2core*(kb*nr + mr*nr))*4byte < 1MB L2cache / 2
				//			int range_byte = 65536; //64KB
				int kb_min = (((data->L2size*L2inf))/type_byte - 2*(mu*nu)) / (mb + 2*nu); 
				int kb_max = (((data->L2size*L2sup))/type_byte - 2*(mu*nu)) / (mb + 2*nu); 
				int kb_int = (kb_max-kb_min)/20;
				kb_max = kb_min+20*kb_int;
				for(kb = kb_min; kb<=kb_max; kb+=kb_int){	
					//				int range_byte = 1048576; //1MB
//					for(j = nb_min; j<=nb_max; j+=nb_int)
					{
						nb = 20000;
						while(nb % nu != 0){
							nb++;
						}
						testKernel=setKernel(mu,nu,ku,mb,nb,kb,20*mu,0,20*nu,0,ni,ni,njr);
						printf("-----------------------1-----------------------------\n");
						if(prepareKernel(testKernel) == -1){ 
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
							//testKernel->mflop[0] = validTest(testKernel, count);
							testKernel->mflop[0] = flopsTest(testKernel, count, 0);
							printf("mflop(check) = %f\n", testKernel->mflop[0]);
							/*
							 * bestKernel choose best 5 (nu,mu,mb,kb)set.
							 */
							printf(" bestKernel1 - ");
							putBestKernel(bestKernel1, testKernel, 20);
						}
					}
					printf("-----------------------------------------------------\n");
				}
			}
		}
	}
	printf("-----------------------1_refine-----------------------------\n");
	kernelListRepeatTest(bestKernel1, 3, 20, count);
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

/*
	printf("****************** phase 1_2 ************************\n");
	for(i = 0; i < 10; i++){
		int niList[4] = {34,17,2,1};
		for(j = 0; j < 4; j++){
			ni = niList[j];
			njr = 68 / ni;
			testKernel = copyKernel(bestKernel1[i]);
			testKernel->nPack=ni;
			testKernel->ni=ni;
			testKernel->njr=njr;
			printf("-----------------------1_2-----------------------------\n");
			if(prepareKernel(testKernel) == -1){

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
				printf("%c[32m\n", 27);
				printf("count = %d\n", ++(*count));
				printf("mu = %d, nu = %d, ku = %d, mb = %d, nb = %d, kb = %d\n", 
						testKernel->mu, testKernel->nu, testKernel->ku,
						testKernel->mb, testKernel->nb, testKernel->kb);
				printf("L1_distanceA = %d, L1_distanceB = %d\n", 
						testKernel->prefetchA1, testKernel->prefetchB1);
				printf("npack = %d, ni = %d, njr = %d\n", 
						testKernel->nPack, testKernel->ni, testKernel->njr);
				printf("%c[0m\n", 27);
				//testKernel->mflop[0] = (float)*count;
				testKernel->mflop[0] = flopsTest(testKernel, count, 0);
				printf(" bestKernel2 - ");
				putBestKernel(bestKernel1_2, testKernel, 20);
			}
			printf("-----------------------------------------------------\n");
		}
	}

	printf("-----------------------1_2_refine-----------------------------\n");
	kernelListRepeatTest(bestKernel1_2, 3, 20, count);
	printf("--------------------phase 1_2 best ------------------------\n");
	for(i = 0; i < 10; i++){
		printf("%c[32m\n", 27);
		printf("%d : mu = %d, nu = %d, ku = %d, mb = %d, nb = %d, kb = %d\n", 
				i, bestKernel1_2[i]->mu, bestKernel1_2[i]->nu, bestKernel1_2[i]->ku,
				bestKernel1_2[i]->mb, bestKernel1_2[i]->nb, bestKernel1_2[i]->kb);
		printf("L1_distanceA = %d, L1_distanceB = %d\n", 
				bestKernel1_2[i]->prefetchA1, bestKernel1_2[i]->prefetchB1);
		printf("npack = %d, ni = %d, njr = %d\n", 
				bestKernel1_2[i]->nPack, bestKernel1_2[i]->ni, bestKernel1_2[i]->njr);
		printf("%c[0m\n", 27);
	}


	printf("****************** phase 2 ************************\n");
	for(i = 0; i < 10; i++){
		for(L1_A = 12; L1_A < 41; L1_A+=4){
			for(L1_B = 12; L1_B < 61; L1_B+=8)
			{ 
				testKernel = copyKernel(bestKernel1_2[i]);
				testKernel->prefetchA1 = L1_A*(testKernel->mu);
				testKernel->prefetchB1 = L1_B*(testKernel->nu);
				printf("-----------------------2-----------------------------\n");
				if(prepareKernel(testKernel) == -1){
					//if(0){
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
					printf("%c[32m\n", 27);
					printf("count = %d\n", ++(*count));
					printf("mu = %d, nu = %d, ku = %d, mb = %d, nb = %d, kb = %d\n", 
							testKernel->mu, testKernel->nu, testKernel->ku,
							testKernel->mb, testKernel->nb, testKernel->kb);
					printf("L1_distanceA = %d, L1_distanceB = %d\n", 
							testKernel->prefetchA1, testKernel->prefetchB1);
					printf("npack = %d, ni = %d, njr = %d\n", 
							testKernel->nPack, testKernel->ni, testKernel->njr);
					printf("%c[0m\n", 27);
					//testKernel->mflop[0] = (float)*count;
					testKernel->mflop[0] = flopsTest(testKernel, count, 0);
					printf(" bestKernel2 - ");
					putBestKernel(bestKernel2, testKernel, 20);

				}
				printf("-----------------------------------------------------\n");
				}
			}
		}
		printf("-----------------------2_refine-----------------------------\n");
		kernelListRepeatTest(bestKernel2, 3, 20, count);
		printf("--------------------phase 2 best ------------------------\n");
		for(i = 0; i < 10; i++){
			printf("%c[32m\n", 27);
			printf("%d : mu = %d, nu = %d, ku = %d, mb = %d, nb = %d, kb = %d\n", 
					i, bestKernel2[i]->mu, bestKernel2[i]->nu, bestKernel2[i]->ku,
					bestKernel2[i]->mb, bestKernel2[i]->nb, bestKernel2[i]->kb);
			printf("L1_distanceA = %d, L1_distanceB = %d\n", 
					bestKernel2[i]->prefetchA1, bestKernel2[i]->prefetchB1);
			printf("npack = %d, ni = %d, njr = %d\n", 
					bestKernel2[i]->nPack, bestKernel2[i]->ni, bestKernel2[i]->njr);
			printf("%c[0m\n", 27);
		}

		printf("****************** phase 3 ************************\n");
		for(i = 0; i < 10; i++){
			// in the for(i), check best ku
			for(ku = 1; ku<6 ; ku++){
				testKernel = copyKernel(bestKernel2[i]);
				testKernel->ku = ku;
				printf("------------------------3----------------------------\n");
				if(prepareKernel(testKernel) == -1){
					//if(0){
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
					printf("%c[32m\n", 27);
					printf("count = %d\n", ++(*count));
					printf("mu = %d, nu = %d, ku = %d, mb = %d, nb = %d, kb = %d\n", 
							testKernel->mu, testKernel->nu, testKernel->ku,
							testKernel->mb, testKernel->nb, testKernel->kb);
					printf("L1_distanceA = %d, L1_distanceB = %d\n", 
							testKernel->prefetchA1, testKernel->prefetchB1);
					printf("npack = %d, ni = %d, njr = %d\n", 
							testKernel->nPack, testKernel->ni, testKernel->njr);
					printf("%c[0m\n", 27);
					//testKernel->mflop[0] = (float)*count;
					testKernel->mflop[0] = flopsTest(testKernel, count, 0);
					printf(" bestKernel3 - ");
					putBestKernel(bestKernel3, testKernel, 20);
					printf("-----------------------------------------------------\n");
				}
				}
			}
			printf("-----------------------3_refine-----------------------------\n");
			kernelListRepeatTest(bestKernel3, 3, 20, count);
			printf("-------------------phase 3 best ---------------------------\n");
			for(i = 0; i < 10; i++){
				printf("%c[32m\n", 27);
				printf("%d : mu = %d, nu = %d, ku = %d, mb = %d, nb = %d, kb = %d\n", 
						i, bestKernel3[i]->mu, bestKernel3[i]->nu, bestKernel3[i]->ku,
						bestKernel3[i]->mb, bestKernel3[i]->nb, bestKernel3[i]->kb);
				printf("L1_distanceA = %d, L1_distanceB = %d\n", 
						bestKernel3[i]->prefetchA1, bestKernel3[i]->prefetchB1);
				printf("npack = %d, ni = %d, njr = %d\n", 
						bestKernel3[i]->nPack, bestKernel3[i]->ni, bestKernel3[i]->njr);
				printf("%c[0m\n", 27);
			}
*/


			printf("%c[33m", 27);
			printf("count = %d", *count);
			printf("%c[0m", 27);

			//killKernelList(bestKernel1,20);
			//killKernelList(bestKernel2,20);

			//return the best kernel list.
			//return bestKernel3;
			return bestKernel1;

		}
