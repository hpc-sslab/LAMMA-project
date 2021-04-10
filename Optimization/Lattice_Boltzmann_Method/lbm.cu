/*############################################################################*/

#include "lbm.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

/*############################################################################*/

/*#define DFL1 (1.0/ 3.0)
 #define DFL2 (1.0/18.0)
 #define DFL3 (1.0/36.0)
 */
extern size_t gridSize;
extern size_t marginSize;


/*############################################################################*/

void LBM_allocateGrid(float** ptr, float ** org_ptr) {
	const size_t margin = 2 * SIZE_X * SIZE_Y * N_CELL_ENTRIES, size =
			sizeof(LBM_Grid) + 2 * margin * sizeof(float);
	*ptr = (float*) malloc(size);
	if (!*ptr) {
		printf("LBM_allocateGrid: could not allocate %.1f MByte\n",
				size / (1024.0 * 1024.0));
		exit(1);
	}
#if !defined(SPEC)
	printf("LBM_allocateGrid: allocated %.1f MByte %d\n",
			size / (1024.0 * 1024.0), size);
#endif
	*org_ptr = *ptr;
	*ptr += margin;
	marginSize = margin;
	gridSize = size / sizeof(float);

	printf("\nLBM_allocateGrid done.\n");
}

/*############################################################################*/

void LBM_freeGrid(float** ptr) {
	const size_t margin = 2 * SIZE_X * SIZE_Y * N_CELL_ENTRIES;

	free(*ptr - margin);
	*ptr = NULL;
}

/*############################################################################*/

void LBM_initializeGrid(LBM_Grid grid) {
	SWEEP_VAR

	/*voption indep*/
	//printf("\nFrom %d to %d\n", CALC_INDEX(0, 0, -2,0) ,  CALC_INDEX(0, 0, SIZE_Z+2,0 ));
	SWEEP_START( 0, 0, -2, 0, 0, SIZE_Z+2 )
		//printf("\ni = %d", i);
		LOCAL( grid, C )= DFL1; //printf("\ni = %d, grid[%d] = %.3f\n", i, CALC_INDEX(0, 0, 0, C)+(i), grid[CALC_INDEX(0, 0, 0, C)+(i)]);
		LOCAL( grid, N ) = DFL2;//printf("\ni = %d, grid[%d] = %.3f\n", i, CALC_INDEX(0, 0, 0, N)+(i), grid[CALC_INDEX(0, 0, 0, N)+(i)]);
		LOCAL( grid, S ) = DFL2;//printf("\ni = %d, grid[%d] = %.3f\n", i, CALC_INDEX(0, 0, 0, S)+(i), grid[CALC_INDEX(0, 0, 0, S)+(i)]);
		LOCAL( grid, E ) = DFL2;//printf("\ni = %d, grid[%d] = %.3f\n", i, CALC_INDEX(0, 0, 0, E)+(i), grid[CALC_INDEX(0, 0, 0, E)+(i)]);
		LOCAL( grid, W ) = DFL2;//printf("\ni = %d, grid[%d] = %.3f\n", i, CALC_INDEX(0, 0, 0, W)+(i), grid[CALC_INDEX(0, 0, 0, W)+(i)]);
		LOCAL( grid, T ) = DFL2;//printf("\ni = %d, grid[%d] = %.3f\n", i, CALC_INDEX(0, 0, 0, T)+(i), grid[CALC_INDEX(0, 0, 0, T)+(i)]);
		LOCAL( grid, B ) = DFL2;//printf("\ni = %d, grid[%d] = %.3f\n", i, CALC_INDEX(0, 0, 0, B)+(i), grid[CALC_INDEX(0, 0, 0, B)+(i)]);
		LOCAL( grid, NE ) = DFL3;//printf("\ni = %d, grid[%d] = %.3f\n", i, CALC_INDEX(0, 0, 0, NE)+(i), grid[CALC_INDEX(0, 0, 0, NE)+(i)]);
		LOCAL( grid, NW ) = DFL3;//printf("\ni = %d, grid[%d] = %.3f\n", i, CALC_INDEX(0, 0, 0, NW)+(i), grid[CALC_INDEX(0, 0, 0, NW)+(i)]);
		LOCAL( grid, SE ) = DFL3;//printf("\ni = %d, grid[%d] = %.3f\n", i, CALC_INDEX(0, 0, 0, SE)+(i), grid[CALC_INDEX(0, 0, 0, SE)+(i)]);
		LOCAL( grid, SW ) = DFL3;//printf("\ni = %d, grid[%d] = %.3f\n", i, CALC_INDEX(0, 0, 0, SW)+(i), grid[CALC_INDEX(0, 0, 0, SW)+(i)]);
		LOCAL( grid, NT ) = DFL3;//printf("\ni = %d, grid[%d] = %.3f\n", i, CALC_INDEX(0, 0, 0, NT)+(i), grid[CALC_INDEX(0, 0, 0, NT)+(i)]);
		LOCAL( grid, NB ) = DFL3;//printf("\ni = %d, grid[%d] = %.3f\n", i, CALC_INDEX(0, 0, 0, NB)+(i), grid[CALC_INDEX(0, 0, 0, NB)+(i)]);
		LOCAL( grid, ST ) = DFL3;//printf("\ni = %d, grid[%d] = %.3f\n", i, CALC_INDEX(0, 0, 0, ST)+(i), grid[CALC_INDEX(0, 0, 0, ST)+(i)]);
		LOCAL( grid, SB ) = DFL3;//printf("\ni = %d, grid[%d] = %.3f\n", i, CALC_INDEX(0, 0, 0, SB)+(i), grid[CALC_INDEX(0, 0, 0, SB)+(i)]);
		LOCAL( grid, ET ) = DFL3;//printf("\ni = %d, grid[%d] = %.3f\n", i, CALC_INDEX(0, 0, 0, ET)+(i), grid[CALC_INDEX(0, 0, 0, ET)+(i)]);
		LOCAL( grid, EB ) = DFL3;//printf("\ni = %d, grid[%d] = %.3f\n", i, CALC_INDEX(0, 0, 0, EB)+(i), grid[CALC_INDEX(0, 0, 0, EB)+(i)]);
		LOCAL( grid, WT ) = DFL3;//printf("\ni = %d, grid[%d] = %.3f\n", i, CALC_INDEX(0, 0, 0, WT)+(i), grid[CALC_INDEX(0, 0, 0, WT)+(i)]);
		LOCAL( grid, WB ) = DFL3;//printf("\ni = %d, grid[%d] = %.3f\n", i, CALC_INDEX(0, 0, 0, WB)+(i), grid[CALC_INDEX(0, 0, 0, WB)+(i)]);

		CLEAR_ALL_FLAGS_SWEEP( grid );

		SWEEP_END
		printf("\nLBM_initializeGrid done.");
	}

	/*############################################################################*/

void LBM_swapGrids(LBM_GridPtr* grid1, LBM_GridPtr* grid2) {
	LBM_GridPtr aux = *grid1;
	*grid1 = *grid2;
	*grid2 = aux;
}

/*############################################################################*/

void LBM_loadObstacleFile(LBM_Grid grid, const char* filename) {
	int x, y, z;

	FILE* file = fopen(filename, "r"); //rb
	if(file==NULL)
	{
		printf("\nCannot open obstacle file.");
		exit(1);
	}
	for (z = 0; z < SIZE_Z; z++) {
		for (y = 0; y < SIZE_Y; y++) {
			for (x = 0; x < SIZE_X; x++) {
				if (fgetc(file) != '.')
					SET_FLAG(grid, x, y, z, OBSTACLE);
			}
			fgetc(file);
		}
		fgetc(file);
	}

	fclose(file);
	printf("\nLBM_loadObstacleFile done.");
}

/*############################################################################*/

void LBM_initializeSpecialCellsForLDC(LBM_Grid grid) {
	int x, y, z;

	/*voption indep*/
	//int count = 0;

	//#pragma acc parallel loop
	for (z = -2; z < SIZE_Z + 2; z++) {
		for (y = 0; y < SIZE_Y; y++) {
			for (x = 0; x < SIZE_X; x++) {

				if (x == 0 || x == SIZE_X - 1 || y == 0 || y == SIZE_Y - 1
						|| z == 0 || z == SIZE_Z - 1) {
					SET_FLAG(grid, x, y, z, OBSTACLE);
					//printf("1 ");
					//count++;
				} else {
					if ((z == 1 || z == SIZE_Z - 2) && x > 1 && x < SIZE_X - 2
							&& y > 1 && y < SIZE_Y - 2) {
						SET_FLAG(grid, x, y, z, ACCEL);
						//printf("2 ");
						//count++;
					}
					//else printf("0 ");
					//count++;
				}
			}
		}
	}
	printf("\nLBM_initializeSpecialCellsForLDC done.");
}
/*############################################################################*/

void LBM_initializeSpecialCellsForChannel(LBM_Grid grid) {
	int x, y, z;

	/*voption indep*/
#if !defined(SPEC)
#ifdef _OPENMP
#pragma omp parallel for private( x, y )
#endif
#endif
	for (z = -2; z < SIZE_Z + 2; z++) {
		for (y = 0; y < SIZE_Y; y++) {
			for (x = 0; x < SIZE_X; x++) {
				if (x == 0 || x == SIZE_X - 1 || y == 0 || y == SIZE_Y - 1) {
					SET_FLAG(grid, x, y, z, OBSTACLE);

					if ((z == 0 || z == SIZE_Z - 1)
							&& !TEST_FLAG( grid, x, y, z, OBSTACLE ))
						SET_FLAG(grid, x, y, z, IN_OUT_FLOW);
				}
			}
		}
	}
	printf("\nLBM_initializeSpecialCellsForChannel done.");
}

/*############################################################################*/

void LBM_performStreamCollide(LBM_Grid srcGrid, LBM_Grid dstGrid) {

	SWEEP_VAR

	float ux, uy, uz, u2, rho;

	int count = 0;
	//FILE *f = fopen("aos_data.txt", "a");
	/*voption indep*/
	//printf("\nFrom %d to %d\n", CALC_INDEX(0, 0, 0,0) ,  CALC_INDEX(0, 0, SIZE_Z,0 ));
	SWEEP_START( 0, 0, 0, 0, 0, SIZE_Z )
		if (TEST_FLAG_SWEEP( srcGrid, OBSTACLE )) {

			//fprintf(f,"\n%d <- %d, %.8f <- %.8f", CALC_INDEX( 0,  0,   0, C)+i, CALC_INDEX(0, 0,  0, C)+i, DST_C ( dstGrid ), SRC_C ( srcGrid ));
			DST_C ( dstGrid )= SRC_C ( srcGrid );
			//fprintf(f,"\n%d <- %d, %.8f <- %.8f", CALC_INDEX( 0, -1,   0, S)+i, CALC_INDEX(0, 0,  0, N)+i, DST_S ( dstGrid ) , SRC_N ( srcGrid ));
			DST_S ( dstGrid ) = SRC_N ( srcGrid );
			//fprintf(f,"\n%d <- %d, %.8f <- %.8f", CALC_INDEX( 0, +1,   0, N)+i, CALC_INDEX(0, 0,  0, S)+i, DST_N ( dstGrid ) , SRC_S ( srcGrid ));
			DST_N ( dstGrid ) = SRC_S ( srcGrid );
			//fprintf(f,"\n%d <- %d, %.8f <- %.8f", CALC_INDEX(-1,  0,   0, W)+i, CALC_INDEX(0, 0,  0, E)+i, DST_W ( dstGrid ) , SRC_E ( srcGrid ));
			DST_W ( dstGrid ) = SRC_E ( srcGrid );
			//fprintf(f,"\n%d <- %d, %.8f <- %.8f", CALC_INDEX(+1,  0,   0, E)+i, CALC_INDEX(0, 0,  0, W)+i, DST_E ( dstGrid ) , SRC_W ( srcGrid ));
			DST_E ( dstGrid ) = SRC_W ( srcGrid );
			//fprintf(f,"\n%d <- %d, %.8f <- %.8f", CALC_INDEX( 0,  0,  -1, B)+i, CALC_INDEX(0, 0,  0, T)+i, DST_B ( dstGrid ) , SRC_T ( srcGrid ));
			DST_B ( dstGrid ) = SRC_T ( srcGrid );
			//fprintf(f,"\n%d <- %d, %.8f <- %.8f", CALC_INDEX( 0,  0,  +1, T)+i, CALC_INDEX(0, 0,  0, B)+i, DST_T ( dstGrid ) , SRC_B ( srcGrid ));
			DST_T ( dstGrid ) = SRC_B ( srcGrid );
			//fprintf(f,"\n%d <- %d, %.8f <- %.8f", CALC_INDEX(-1, -1,   0, SW)+i, CALC_INDEX(0, 0,  0, NE)+i, DST_SW( dstGrid ) , SRC_NE( srcGrid ));
			DST_SW( dstGrid ) = SRC_NE( srcGrid );
			//fprintf(f,"\n%d <- %d, %.8f <- %.8f", CALC_INDEX(+1, -1,   0, SE)+i, CALC_INDEX(0, 0,  0, NW)+i, DST_SE( dstGrid ) , SRC_NW( srcGrid ));
			DST_SE( dstGrid ) = SRC_NW( srcGrid );
			//fprintf(f,"\n%d <- %d, %.8f <- %.8f", CALC_INDEX(-1, +1,   0, NW)+i, CALC_INDEX(0, 0,  0, SE)+i, DST_NW( dstGrid ) , SRC_SE( srcGrid ));
			DST_NW( dstGrid ) = SRC_SE( srcGrid );
			//fprintf(f,"\n%d <- %d, %.8f <- %.8f", CALC_INDEX(+1, +1,   0, NE)+i, CALC_INDEX(0, 0,  0, SW)+i, DST_NE( dstGrid ) , SRC_SW( srcGrid ));
			DST_NE( dstGrid ) = SRC_SW( srcGrid );
			//fprintf(f,"\n%d <- %d, %.8f <- %.8f", CALC_INDEX( 0, -1,  -1, SB)+i, CALC_INDEX(0, 0,  0, NT)+i, DST_SB( dstGrid ) , SRC_NT( srcGrid ));
			DST_SB( dstGrid ) = SRC_NT( srcGrid );
			//fprintf(f,"\n%d <- %d, %.8f <- %.8f", CALC_INDEX( 0, -1,  +1, ST)+i, CALC_INDEX(0, 0,  0, NB)+i, DST_ST( dstGrid ) , SRC_NB( srcGrid ));
			DST_ST( dstGrid ) = SRC_NB( srcGrid );
			//fprintf(f,"\n%d <- %d, %.8f <- %.8f", CALC_INDEX( 0, +1,  -1, NB)+i, CALC_INDEX(0, 0,  0, ST)+i, DST_NB( dstGrid ) , SRC_ST( srcGrid ));
			DST_NB( dstGrid ) = SRC_ST( srcGrid );
			//fprintf(f,"\n%d <- %d, %.8f <- %.8f", CALC_INDEX( 0, +1,  +1, NT)+i, CALC_INDEX(0, 0,  0, SB)+i, DST_NT( dstGrid ) , SRC_SB( srcGrid ));
			DST_NT( dstGrid ) = SRC_SB( srcGrid );
			//fprintf(f,"\n%d <- %d, %.8f <- %.8f", CALC_INDEX(-1,  0,  -1, WB)+i, CALC_INDEX(0, 0,  0, ET)+i, DST_WB( dstGrid ) , SRC_ET( srcGrid ));
			DST_WB( dstGrid ) = SRC_ET( srcGrid );
			//fprintf(f,"\n%d <- %d, %.8f <- %.8f", CALC_INDEX(-1,  0,  +1, WT)+i, CALC_INDEX(0, 0,  0, EB)+i, DST_WT( dstGrid ) , SRC_EB( srcGrid ));
			DST_WT( dstGrid ) = SRC_EB( srcGrid );
			//fprintf(f,"\n%d <- %d, %.8f <- %.8f", CALC_INDEX(+1,  0,  -1, EB)+i, CALC_INDEX(0, 0,  0, WT)+i, DST_EB( dstGrid ) , SRC_WT( srcGrid ));
			DST_EB( dstGrid ) = SRC_WT( srcGrid );
			//fprintf(f,"\n%d <- %d, %.8f <- %.8f", CALC_INDEX(+1,  0,  +1, ET)+i, CALC_INDEX(0, 0,  0, WB)+i, DST_ET( dstGrid ) , SRC_WB( srcGrid ));
			DST_ET( dstGrid ) = SRC_WB( srcGrid );
			//system("pause");
//			if(i<100)
//			fprintf(f, "\n%d OBS\t%.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f",i, SRC_C ( srcGrid ), SRC_N ( srcGrid ), SRC_S ( srcGrid ), SRC_E ( srcGrid ),
//					SRC_W ( srcGrid ), SRC_W ( srcGrid ), SRC_T ( srcGrid ), SRC_B ( srcGrid ), SRC_NE ( srcGrid ), SRC_NW ( srcGrid ), SRC_SE ( srcGrid ), SRC_SW ( srcGrid ), SRC_NT ( srcGrid ), SRC_NB ( srcGrid ),
//					SRC_ST ( srcGrid ), SRC_SB ( srcGrid ), SRC_ET ( srcGrid ), SRC_EB ( srcGrid ), SRC_WT ( srcGrid ), SRC_WB ( srcGrid ));
			continue;

		}
		count++;
//		if(i==83220 || i == (83220 + 64*64*20) || i == (83220 + 2*64*64*20) || i == (83220 + 3*64*64*20)){
//			fprintf(f, "\n%d FLU\t%.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f",i, SRC_C ( srcGrid ), SRC_N ( srcGrid ), SRC_S ( srcGrid ), SRC_E ( srcGrid ),
//							SRC_W ( srcGrid ), SRC_W ( srcGrid ), SRC_T ( srcGrid ), SRC_B ( srcGrid ), SRC_NE ( srcGrid ), SRC_NW ( srcGrid ), SRC_SE ( srcGrid ), SRC_SW ( srcGrid ), SRC_NT ( srcGrid ), SRC_NB ( srcGrid ),
//							SRC_ST ( srcGrid ), SRC_SB ( srcGrid ), SRC_ET ( srcGrid ), SRC_EB ( srcGrid ), SRC_WT ( srcGrid ), SRC_WB ( srcGrid ));
//
//		}

		//printf("\n%d", count);
		//calculate the mass density rho
		rho = + SRC_C ( srcGrid ) + SRC_N ( srcGrid )
		+ SRC_S ( srcGrid ) + SRC_E ( srcGrid )
		+ SRC_W ( srcGrid ) + SRC_T ( srcGrid )
		+ SRC_B ( srcGrid ) + SRC_NE( srcGrid )
		+ SRC_NW( srcGrid ) + SRC_SE( srcGrid )
		+ SRC_SW( srcGrid ) + SRC_NT( srcGrid )
		+ SRC_NB( srcGrid ) + SRC_ST( srcGrid )
		+ SRC_SB( srcGrid ) + SRC_ET( srcGrid )
		+ SRC_EB( srcGrid ) + SRC_WT( srcGrid )
		+ SRC_WB( srcGrid );
		//calculate fluid velocity u
		ux = + SRC_E ( srcGrid ) - SRC_W ( srcGrid )
		+ SRC_NE( srcGrid ) - SRC_NW( srcGrid )
		+ SRC_SE( srcGrid ) - SRC_SW( srcGrid )
		+ SRC_ET( srcGrid ) + SRC_EB( srcGrid )
		- SRC_WT( srcGrid ) - SRC_WB( srcGrid );
		//printf("\nux = %.8f", ux);
		uy = + SRC_N ( srcGrid ) - SRC_S ( srcGrid )
		+ SRC_NE( srcGrid ) + SRC_NW( srcGrid )
		- SRC_SE( srcGrid ) - SRC_SW( srcGrid )
		+ SRC_NT( srcGrid ) + SRC_NB( srcGrid )
		- SRC_ST( srcGrid ) - SRC_SB( srcGrid );
		//printf("\nuy = %.8f", uy);
		uz = + SRC_T ( srcGrid ) - SRC_B ( srcGrid )
		+ SRC_NT( srcGrid ) - SRC_NB( srcGrid )
		+ SRC_ST( srcGrid ) - SRC_SB( srcGrid )
		+ SRC_ET( srcGrid ) - SRC_EB( srcGrid )
		+ SRC_WT( srcGrid ) - SRC_WB( srcGrid );
		//printf("\nuz = %.8f", uz);
		ux /= rho;
		uy /= rho;
		uz /= rho;

		if( TEST_FLAG_SWEEP( srcGrid, ACCEL )) {
			ux = 0.005;
			uy = 0.002;
			uz = 0.000;
		}

		u2 = 1.5 * (ux*ux + uy*uy + uz*uz);

		//calculate the local equilibrium distribution function at C
		DST_C ( dstGrid ) = (1.0-OMEGA)*SRC_C ( srcGrid ) + DFL1*OMEGA*rho*(1.0 - u2);
		//calculate the local equilibrium distribution function at N, S, E, W, T, B
		DST_N ( dstGrid ) = (1.0-OMEGA)*SRC_N ( srcGrid ) + DFL2*OMEGA*rho*(1.0 + uy*(4.5*uy + 3.0) - u2);
		DST_S ( dstGrid ) = (1.0-OMEGA)*SRC_S ( srcGrid ) + DFL2*OMEGA*rho*(1.0 + uy*(4.5*uy - 3.0) - u2);
		DST_E ( dstGrid ) = (1.0-OMEGA)*SRC_E ( srcGrid ) + DFL2*OMEGA*rho*(1.0 + ux*(4.5*ux + 3.0) - u2);
		DST_W ( dstGrid ) = (1.0-OMEGA)*SRC_W ( srcGrid ) + DFL2*OMEGA*rho*(1.0 + ux*(4.5*ux - 3.0) - u2);
		DST_T ( dstGrid ) = (1.0-OMEGA)*SRC_T ( srcGrid ) + DFL2*OMEGA*rho*(1.0 + uz*(4.5*uz + 3.0) - u2);
		DST_B ( dstGrid ) = (1.0-OMEGA)*SRC_B ( srcGrid ) + DFL2*OMEGA*rho*(1.0 + uz*(4.5*uz - 3.0) - u2);
		//calculate the local equilibrium distribution function at NE, NW, SE, SW, NT, NB, ST, SB, ET, EB, WT, WB
		DST_NE( dstGrid ) = (1.0-OMEGA)*SRC_NE( srcGrid ) + DFL3*OMEGA*rho*(1.0 + (+ux+uy)*(4.5*(+ux+uy) + 3.0) - u2);
		DST_NW( dstGrid ) = (1.0-OMEGA)*SRC_NW( srcGrid ) + DFL3*OMEGA*rho*(1.0 + (-ux+uy)*(4.5*(-ux+uy) + 3.0) - u2);
		DST_SE( dstGrid ) = (1.0-OMEGA)*SRC_SE( srcGrid ) + DFL3*OMEGA*rho*(1.0 + (+ux-uy)*(4.5*(+ux-uy) + 3.0) - u2);
		DST_SW( dstGrid ) = (1.0-OMEGA)*SRC_SW( srcGrid ) + DFL3*OMEGA*rho*(1.0 + (-ux-uy)*(4.5*(-ux-uy) + 3.0) - u2);
		DST_NT( dstGrid ) = (1.0-OMEGA)*SRC_NT( srcGrid ) + DFL3*OMEGA*rho*(1.0 + (+uy+uz)*(4.5*(+uy+uz) + 3.0) - u2);
		DST_NB( dstGrid ) = (1.0-OMEGA)*SRC_NB( srcGrid ) + DFL3*OMEGA*rho*(1.0 + (+uy-uz)*(4.5*(+uy-uz) + 3.0) - u2);
		DST_ST( dstGrid ) = (1.0-OMEGA)*SRC_ST( srcGrid ) + DFL3*OMEGA*rho*(1.0 + (-uy+uz)*(4.5*(-uy+uz) + 3.0) - u2);
		DST_SB( dstGrid ) = (1.0-OMEGA)*SRC_SB( srcGrid ) + DFL3*OMEGA*rho*(1.0 + (-uy-uz)*(4.5*(-uy-uz) + 3.0) - u2);
		DST_ET( dstGrid ) = (1.0-OMEGA)*SRC_ET( srcGrid ) + DFL3*OMEGA*rho*(1.0 + (+ux+uz)*(4.5*(+ux+uz) + 3.0) - u2);
		DST_EB( dstGrid ) = (1.0-OMEGA)*SRC_EB( srcGrid ) + DFL3*OMEGA*rho*(1.0 + (+ux-uz)*(4.5*(+ux-uz) + 3.0) - u2);
		DST_WT( dstGrid ) = (1.0-OMEGA)*SRC_WT( srcGrid ) + DFL3*OMEGA*rho*(1.0 + (-ux+uz)*(4.5*(-ux+uz) + 3.0) - u2);
		DST_WB( dstGrid ) = (1.0-OMEGA)*SRC_WB( srcGrid ) + DFL3*OMEGA*rho*(1.0 + (-ux-uz)*(4.5*(-ux-uz) + 3.0) - u2);
		SWEEP_END
		//fclose(f);
	}
void LBM_performStreamCollide_Pull(LBM_Grid srcGrid, LBM_Grid dstGrid) {

	SWEEP_VAR

	float ux, uy, uz, u2, rho;

	int count = 0;
	//FILE *f = fopen("aos_data.txt", "w");
	/*voption indep*/
	//printf("\nFrom %d to %d\n", CALC_INDEX(0, 0, 0,0) ,  CALC_INDEX(0, 0, SIZE_Z,0 ));
	SWEEP_START( 0, 0, 0, 0, 0, SIZE_Z )
		if (TEST_FLAG_SWEEP( srcGrid, OBSTACLE )) {
			count++;
			//fprintf(f,"\n%d <- %d, %.8f <- %.8f", CALC_INDEX( 0,  0,   0, C)+i, CALC_INDEX(0, 0,  0, C)+i, DST_C ( dstGrid ), SRC_C ( srcGrid ));
			DST_C ( dstGrid )= SRC_C ( srcGrid );
			//fprintf(f,"\n%d <- %d, %.8f <- %.8f", CALC_INDEX( 0, -1,   0, S)+i, CALC_INDEX(0, 0,  0, N)+i, DST_S ( dstGrid ) , SRC_N ( srcGrid ));
			DST_S ( dstGrid ) = SRC_N ( srcGrid );
			//fprintf(f,"\n%d <- %d, %.8f <- %.8f", CALC_INDEX( 0, +1,   0, N)+i, CALC_INDEX(0, 0,  0, S)+i, DST_N ( dstGrid ) , SRC_S ( srcGrid ));
			DST_N ( dstGrid ) = SRC_S ( srcGrid );
			//fprintf(f,"\n%d <- %d, %.8f <- %.8f", CALC_INDEX(-1,  0,   0, W)+i, CALC_INDEX(0, 0,  0, E)+i, DST_W ( dstGrid ) , SRC_E ( srcGrid ));
			DST_W ( dstGrid ) = SRC_E ( srcGrid );
			//fprintf(f,"\n%d <- %d, %.8f <- %.8f", CALC_INDEX(+1,  0,   0, E)+i, CALC_INDEX(0, 0,  0, W)+i, DST_E ( dstGrid ) , SRC_W ( srcGrid ));
			DST_E ( dstGrid ) = SRC_W ( srcGrid );
			//fprintf(f,"\n%d <- %d, %.8f <- %.8f", CALC_INDEX( 0,  0,  -1, B)+i, CALC_INDEX(0, 0,  0, T)+i, DST_B ( dstGrid ) , SRC_T ( srcGrid ));
			DST_B ( dstGrid ) = SRC_T ( srcGrid );
			//fprintf(f,"\n%d <- %d, %.8f <- %.8f", CALC_INDEX( 0,  0,  +1, T)+i, CALC_INDEX(0, 0,  0, B)+i, DST_T ( dstGrid ) , SRC_B ( srcGrid ));
			DST_T ( dstGrid ) = SRC_B ( srcGrid );
			//fprintf(f,"\n%d <- %d, %.8f <- %.8f", CALC_INDEX(-1, -1,   0, SW)+i, CALC_INDEX(0, 0,  0, NE)+i, DST_SW( dstGrid ) , SRC_NE( srcGrid ));
			DST_SW( dstGrid ) = SRC_NE( srcGrid );
			//fprintf(f,"\n%d <- %d, %.8f <- %.8f", CALC_INDEX(+1, -1,   0, SE)+i, CALC_INDEX(0, 0,  0, NW)+i, DST_SE( dstGrid ) , SRC_NW( srcGrid ));
			DST_SE( dstGrid ) = SRC_NW( srcGrid );
			//fprintf(f,"\n%d <- %d, %.8f <- %.8f", CALC_INDEX(-1, +1,   0, NW)+i, CALC_INDEX(0, 0,  0, SE)+i, DST_NW( dstGrid ) , SRC_SE( srcGrid ));
			DST_NW( dstGrid ) = SRC_SE( srcGrid );
			//fprintf(f,"\n%d <- %d, %.8f <- %.8f", CALC_INDEX(+1, +1,   0, NE)+i, CALC_INDEX(0, 0,  0, SW)+i, DST_NE( dstGrid ) , SRC_SW( srcGrid ));
			DST_NE( dstGrid ) = SRC_SW( srcGrid );
			//fprintf(f,"\n%d <- %d, %.8f <- %.8f", CALC_INDEX( 0, -1,  -1, SB)+i, CALC_INDEX(0, 0,  0, NT)+i, DST_SB( dstGrid ) , SRC_NT( srcGrid ));
			DST_SB( dstGrid ) = SRC_NT( srcGrid );
			//fprintf(f,"\n%d <- %d, %.8f <- %.8f", CALC_INDEX( 0, -1,  +1, ST)+i, CALC_INDEX(0, 0,  0, NB)+i, DST_ST( dstGrid ) , SRC_NB( srcGrid ));
			DST_ST( dstGrid ) = SRC_NB( srcGrid );
			//fprintf(f,"\n%d <- %d, %.8f <- %.8f", CALC_INDEX( 0, +1,  -1, NB)+i, CALC_INDEX(0, 0,  0, ST)+i, DST_NB( dstGrid ) , SRC_ST( srcGrid ));
			DST_NB( dstGrid ) = SRC_ST( srcGrid );
			//fprintf(f,"\n%d <- %d, %.8f <- %.8f", CALC_INDEX( 0, +1,  +1, NT)+i, CALC_INDEX(0, 0,  0, SB)+i, DST_NT( dstGrid ) , SRC_SB( srcGrid ));
			DST_NT( dstGrid ) = SRC_SB( srcGrid );
			//fprintf(f,"\n%d <- %d, %.8f <- %.8f", CALC_INDEX(-1,  0,  -1, WB)+i, CALC_INDEX(0, 0,  0, ET)+i, DST_WB( dstGrid ) , SRC_ET( srcGrid ));
			DST_WB( dstGrid ) = SRC_ET( srcGrid );
			//fprintf(f,"\n%d <- %d, %.8f <- %.8f", CALC_INDEX(-1,  0,  +1, WT)+i, CALC_INDEX(0, 0,  0, EB)+i, DST_WT( dstGrid ) , SRC_EB( srcGrid ));
			DST_WT( dstGrid ) = SRC_EB( srcGrid );
			//fprintf(f,"\n%d <- %d, %.8f <- %.8f", CALC_INDEX(+1,  0,  -1, EB)+i, CALC_INDEX(0, 0,  0, WT)+i, DST_EB( dstGrid ) , SRC_WT( srcGrid ));
			DST_EB( dstGrid ) = SRC_WT( srcGrid );
			//fprintf(f,"\n%d <- %d, %.8f <- %.8f", CALC_INDEX(+1,  0,  +1, ET)+i, CALC_INDEX(0, 0,  0, WB)+i, DST_ET( dstGrid ) , SRC_WB( srcGrid ));
			DST_ET( dstGrid ) = SRC_WB( srcGrid );
			//system("pause");
			continue;

		}
		//calculate the mass density rho
		rho = + SRC_C ( srcGrid ) + SRC_N ( srcGrid )
		+ SRC_S ( srcGrid ) + SRC_E ( srcGrid )
		+ SRC_W ( srcGrid ) + SRC_T ( srcGrid )
		+ SRC_B ( srcGrid ) + SRC_NE( srcGrid )
		+ SRC_NW( srcGrid ) + SRC_SE( srcGrid )
		+ SRC_SW( srcGrid ) + SRC_NT( srcGrid )
		+ SRC_NB( srcGrid ) + SRC_ST( srcGrid )
		+ SRC_SB( srcGrid ) + SRC_ET( srcGrid )
		+ SRC_EB( srcGrid ) + SRC_WT( srcGrid )
		+ SRC_WB( srcGrid );
		//calculate fluid velocity u
		ux = + SRC_E ( srcGrid ) - SRC_W ( srcGrid )
		+ SRC_NE( srcGrid ) - SRC_NW( srcGrid )
		+ SRC_SE( srcGrid ) - SRC_SW( srcGrid )
		+ SRC_ET( srcGrid ) + SRC_EB( srcGrid )
		- SRC_WT( srcGrid ) - SRC_WB( srcGrid );
		//printf("\nux = %.8f", ux);
		uy = + SRC_N ( srcGrid ) - SRC_S ( srcGrid )
		+ SRC_NE( srcGrid ) + SRC_NW( srcGrid )
		- SRC_SE( srcGrid ) - SRC_SW( srcGrid )
		+ SRC_NT( srcGrid ) + SRC_NB( srcGrid )
		- SRC_ST( srcGrid ) - SRC_SB( srcGrid );
		//printf("\nuy = %.8f", uy);
		uz = + SRC_T ( srcGrid ) - SRC_B ( srcGrid )
		+ SRC_NT( srcGrid ) - SRC_NB( srcGrid )
		+ SRC_ST( srcGrid ) - SRC_SB( srcGrid )
		+ SRC_ET( srcGrid ) - SRC_EB( srcGrid )
		+ SRC_WT( srcGrid ) - SRC_WB( srcGrid );
		//printf("\nuz = %.8f", uz);
		ux /= rho;
		uy /= rho;
		uz /= rho;

		if( TEST_FLAG_SWEEP( srcGrid, ACCEL )) {
			ux = 0.005;
			uy = 0.002;
			uz = 0.000;
		}

		u2 = 1.5 * (ux*ux + uy*uy + uz*uz);
		//calculate the local equilibrium distribution function at C
		DST_C ( dstGrid ) = (1.0-OMEGA)*SRC_C ( srcGrid ) + DFL1*OMEGA*rho*(1.0 - u2);
		//calculate the local equilibrium distribution function at N, S, E, W, T, B
		DST_N ( dstGrid ) = (1.0-OMEGA)*SRC_N ( srcGrid ) + DFL2*OMEGA*rho*(1.0 + uy*(4.5*uy + 3.0) - u2);
		DST_S ( dstGrid ) = (1.0-OMEGA)*SRC_S ( srcGrid ) + DFL2*OMEGA*rho*(1.0 + uy*(4.5*uy - 3.0) - u2);
		DST_E ( dstGrid ) = (1.0-OMEGA)*SRC_E ( srcGrid ) + DFL2*OMEGA*rho*(1.0 + ux*(4.5*ux + 3.0) - u2);
		DST_W ( dstGrid ) = (1.0-OMEGA)*SRC_W ( srcGrid ) + DFL2*OMEGA*rho*(1.0 + ux*(4.5*ux - 3.0) - u2);
		DST_T ( dstGrid ) = (1.0-OMEGA)*SRC_T ( srcGrid ) + DFL2*OMEGA*rho*(1.0 + uz*(4.5*uz + 3.0) - u2);
		DST_B ( dstGrid ) = (1.0-OMEGA)*SRC_B ( srcGrid ) + DFL2*OMEGA*rho*(1.0 + uz*(4.5*uz - 3.0) - u2);
		//calculate the local equilibrium distribution function at NE, NW, SE, SW, NT, NB, ST, SB, ET, EB, WT, WB
		DST_NE( dstGrid ) = (1.0-OMEGA)*SRC_NE( srcGrid ) + DFL3*OMEGA*rho*(1.0 + (+ux+uy)*(4.5*(+ux+uy) + 3.0) - u2);
		DST_NW( dstGrid ) = (1.0-OMEGA)*SRC_NW( srcGrid ) + DFL3*OMEGA*rho*(1.0 + (-ux+uy)*(4.5*(-ux+uy) + 3.0) - u2);
		DST_SE( dstGrid ) = (1.0-OMEGA)*SRC_SE( srcGrid ) + DFL3*OMEGA*rho*(1.0 + (+ux-uy)*(4.5*(+ux-uy) + 3.0) - u2);
		DST_SW( dstGrid ) = (1.0-OMEGA)*SRC_SW( srcGrid ) + DFL3*OMEGA*rho*(1.0 + (-ux-uy)*(4.5*(-ux-uy) + 3.0) - u2);
		DST_NT( dstGrid ) = (1.0-OMEGA)*SRC_NT( srcGrid ) + DFL3*OMEGA*rho*(1.0 + (+uy+uz)*(4.5*(+uy+uz) + 3.0) - u2);
		DST_NB( dstGrid ) = (1.0-OMEGA)*SRC_NB( srcGrid ) + DFL3*OMEGA*rho*(1.0 + (+uy-uz)*(4.5*(+uy-uz) + 3.0) - u2);
		DST_ST( dstGrid ) = (1.0-OMEGA)*SRC_ST( srcGrid ) + DFL3*OMEGA*rho*(1.0 + (-uy+uz)*(4.5*(-uy+uz) + 3.0) - u2);
		DST_SB( dstGrid ) = (1.0-OMEGA)*SRC_SB( srcGrid ) + DFL3*OMEGA*rho*(1.0 + (-uy-uz)*(4.5*(-uy-uz) + 3.0) - u2);
		DST_ET( dstGrid ) = (1.0-OMEGA)*SRC_ET( srcGrid ) + DFL3*OMEGA*rho*(1.0 + (+ux+uz)*(4.5*(+ux+uz) + 3.0) - u2);
		DST_EB( dstGrid ) = (1.0-OMEGA)*SRC_EB( srcGrid ) + DFL3*OMEGA*rho*(1.0 + (+ux-uz)*(4.5*(+ux-uz) + 3.0) - u2);
		DST_WT( dstGrid ) = (1.0-OMEGA)*SRC_WT( srcGrid ) + DFL3*OMEGA*rho*(1.0 + (-ux+uz)*(4.5*(-ux+uz) + 3.0) - u2);
		DST_WB( dstGrid ) = (1.0-OMEGA)*SRC_WB( srcGrid ) + DFL3*OMEGA*rho*(1.0 + (-ux-uz)*(4.5*(-ux-uz) + 3.0) - u2);
		SWEEP_END
		//fclose(f);
	}
	/*############################################################################*/

void LBM_handleInOutFlow(LBM_Grid srcGrid) {
	float ux, uy, uz, rho, ux1, uy1, uz1, rho1, ux2, uy2, uz2, rho2, u2, px,
			py;
	SWEEP_VAR

	/* inflow */
	/*voption indep*/

#ifdef DBG
	printf("srcGrid = %x, %d\n",srcGrid, GRID_SIZE);
#endif

	SWEEP_START( 0, 0, 0, 0, 0, 1 )
		rho1 = +GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, C )+ GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, N  )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, S  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, E  )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, W  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, T  )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, B  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, NE )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, NW ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, SE )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, SW ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, NT )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, NB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, ST )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, SB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, ET )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, EB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, WT )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, WB ) ;
		rho2 = + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, C ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, N )
		+ GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, S ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, E )
		+ GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, W ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, T )
		+ GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, B ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, NE )
		+ GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, NW ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, SE )
		+ GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, SW ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, NT )
		+ GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, NB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, ST )
		+ GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, SB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, ET )
		+ GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, EB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, WT )
		+ GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, WB );

		rho = 2.0*rho1 - rho2;

		px = (SWEEP_X / (0.5*(SIZE_X-1))) - 1.0;
		py = (SWEEP_Y / (0.5*(SIZE_Y-1))) - 1.0;
		ux = 0.00;
		uy = 0.00;
		uz = 0.01 * (1.0-px*px) * (1.0-py*py);

		u2 = 1.5 * (ux*ux + uy*uy + uz*uz);

		LOCAL( srcGrid, C ) = DFL1*rho*(1.0 - u2);

		LOCAL( srcGrid, N ) = DFL2*rho*(1.0 + uy*(4.5*uy + 3.0) - u2);
		LOCAL( srcGrid, S ) = DFL2*rho*(1.0 + uy*(4.5*uy - 3.0) - u2);
		LOCAL( srcGrid, E ) = DFL2*rho*(1.0 + ux*(4.5*ux + 3.0) - u2);
		LOCAL( srcGrid, W ) = DFL2*rho*(1.0 + ux*(4.5*ux - 3.0) - u2);
		LOCAL( srcGrid, T ) = DFL2*rho*(1.0 + uz*(4.5*uz + 3.0) - u2);
		LOCAL( srcGrid, B ) = DFL2*rho*(1.0 + uz*(4.5*uz - 3.0) - u2);

		LOCAL( srcGrid, NE) = DFL3*rho*(1.0 + (+ux+uy)*(4.5*(+ux+uy) + 3.0) - u2);
		LOCAL( srcGrid, NW) = DFL3*rho*(1.0 + (-ux+uy)*(4.5*(-ux+uy) + 3.0) - u2);
		LOCAL( srcGrid, SE) = DFL3*rho*(1.0 + (+ux-uy)*(4.5*(+ux-uy) + 3.0) - u2);
		LOCAL( srcGrid, SW) = DFL3*rho*(1.0 + (-ux-uy)*(4.5*(-ux-uy) + 3.0) - u2);
		LOCAL( srcGrid, NT) = DFL3*rho*(1.0 + (+uy+uz)*(4.5*(+uy+uz) + 3.0) - u2);
		LOCAL( srcGrid, NB) = DFL3*rho*(1.0 + (+uy-uz)*(4.5*(+uy-uz) + 3.0) - u2);
		LOCAL( srcGrid, ST) = DFL3*rho*(1.0 + (-uy+uz)*(4.5*(-uy+uz) + 3.0) - u2);
		LOCAL( srcGrid, SB) = DFL3*rho*(1.0 + (-uy-uz)*(4.5*(-uy-uz) + 3.0) - u2);
		LOCAL( srcGrid, ET) = DFL3*rho*(1.0 + (+ux+uz)*(4.5*(+ux+uz) + 3.0) - u2);
		LOCAL( srcGrid, EB) = DFL3*rho*(1.0 + (+ux-uz)*(4.5*(+ux-uz) + 3.0) - u2);
		LOCAL( srcGrid, WT) = DFL3*rho*(1.0 + (-ux+uz)*(4.5*(-ux+uz) + 3.0) - u2);
		LOCAL( srcGrid, WB) = DFL3*rho*(1.0 + (-ux-uz)*(4.5*(-ux-uz) + 3.0) - u2);
		SWEEP_END

	SWEEP_START( 0, 0, SIZE_Z-1, 0, 0, SIZE_Z )
		rho1 = +GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, C )+ GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, N )
		+ GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, S ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, E )
		+ GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, W ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, T )
		+ GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, B ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NE )
		+ GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NW ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SE )
		+ GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SW ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NT )
		+ GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, ST )
		+ GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, ET )
		+ GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, EB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, WT )
		+ GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, WB );
		ux1 = + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, E ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, W )
		+ GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NE ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NW )
		+ GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SE ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SW )
		+ GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, ET ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, EB )
		- GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, WT ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, WB );
		uy1 = + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, N ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, S )
		+ GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NE ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NW )
		- GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SE ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SW )
		+ GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NT ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NB )
		- GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, ST ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SB );
		uz1 = + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, T ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, B )
		+ GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NT ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NB )
		+ GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, ST ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SB )
		+ GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, ET ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, EB )
		+ GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, WT ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, WB );

		ux1 /= rho1;
		uy1 /= rho1;
		uz1 /= rho1;

		rho2 = + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, C ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, N )
		+ GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, S ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, E )
		+ GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, W ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, T )
		+ GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, B ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NE )
		+ GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NW ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SE )
		+ GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SW ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NT )
		+ GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, ST )
		+ GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, ET )
		+ GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, EB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, WT )
		+ GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, WB );
		ux2 = + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, E ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, W )
		+ GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NE ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NW )
		+ GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SE ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SW )
		+ GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, ET ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, EB )
		- GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, WT ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, WB );
		uy2 = + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, N ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, S )
		+ GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NE ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NW )
		- GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SE ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SW )
		+ GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NT ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NB )
		- GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, ST ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SB );
		uz2 = + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, T ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, B )
		+ GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NT ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NB )
		+ GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, ST ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SB )
		+ GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, ET ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, EB )
		+ GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, WT ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, WB );

		ux2 /= rho2;
		uy2 /= rho2;
		uz2 /= rho2;

		rho = 1.0;

		ux = 2*ux1 - ux2;
		uy = 2*uy1 - uy2;
		uz = 2*uz1 - uz2;

		u2 = 1.5 * (ux*ux + uy*uy + uz*uz);

		LOCAL( srcGrid, C ) = DFL1*rho*(1.0 - u2);

		LOCAL( srcGrid, N ) = DFL2*rho*(1.0 + uy*(4.5*uy + 3.0) - u2);
		LOCAL( srcGrid, S ) = DFL2*rho*(1.0 + uy*(4.5*uy - 3.0) - u2);
		LOCAL( srcGrid, E ) = DFL2*rho*(1.0 + ux*(4.5*ux + 3.0) - u2);
		LOCAL( srcGrid, W ) = DFL2*rho*(1.0 + ux*(4.5*ux - 3.0) - u2);
		LOCAL( srcGrid, T ) = DFL2*rho*(1.0 + uz*(4.5*uz + 3.0) - u2);
		LOCAL( srcGrid, B ) = DFL2*rho*(1.0 + uz*(4.5*uz - 3.0) - u2);

		LOCAL( srcGrid, NE) = DFL3*rho*(1.0 + (+ux+uy)*(4.5*(+ux+uy) + 3.0) - u2);
		LOCAL( srcGrid, NW) = DFL3*rho*(1.0 + (-ux+uy)*(4.5*(-ux+uy) + 3.0) - u2);
		LOCAL( srcGrid, SE) = DFL3*rho*(1.0 + (+ux-uy)*(4.5*(+ux-uy) + 3.0) - u2);
		LOCAL( srcGrid, SW) = DFL3*rho*(1.0 + (-ux-uy)*(4.5*(-ux-uy) + 3.0) - u2);
		LOCAL( srcGrid, NT) = DFL3*rho*(1.0 + (+uy+uz)*(4.5*(+uy+uz) + 3.0) - u2);
		LOCAL( srcGrid, NB) = DFL3*rho*(1.0 + (+uy-uz)*(4.5*(+uy-uz) + 3.0) - u2);
		LOCAL( srcGrid, ST) = DFL3*rho*(1.0 + (-uy+uz)*(4.5*(-uy+uz) + 3.0) - u2);
		LOCAL( srcGrid, SB) = DFL3*rho*(1.0 + (-uy-uz)*(4.5*(-uy-uz) + 3.0) - u2);
		LOCAL( srcGrid, ET) = DFL3*rho*(1.0 + (+ux+uz)*(4.5*(+ux+uz) + 3.0) - u2);
		LOCAL( srcGrid, EB) = DFL3*rho*(1.0 + (+ux-uz)*(4.5*(+ux-uz) + 3.0) - u2);
		LOCAL( srcGrid, WT) = DFL3*rho*(1.0 + (-ux+uz)*(4.5*(-ux+uz) + 3.0) - u2);
		LOCAL( srcGrid, WB) = DFL3*rho*(1.0 + (-ux-uz)*(4.5*(-ux-uz) + 3.0) - u2);
		SWEEP_END

	}

	/*############################################################################*/

void LBM_showGridStatistics(LBM_Grid grid) {
	int nObstacleCells = 0, nAccelCells = 0, nFluidCells = 0;
	float ux, uy, uz;
	float minU2 = 1e+30, maxU2 = -1e+30, u2;
	float minRho = 1e+30, maxRho = -1e+30, rho;
	float mass = 0;

	SWEEP_VAR

	SWEEP_START( 0, 0, 0, 0, 0, SIZE_Z )
		rho = +LOCAL( grid, C )+ LOCAL( grid, N )
		+ LOCAL( grid, S ) + LOCAL( grid, E )
		+ LOCAL( grid, W ) + LOCAL( grid, T )
		+ LOCAL( grid, B ) + LOCAL( grid, NE )
		+ LOCAL( grid, NW ) + LOCAL( grid, SE )
		+ LOCAL( grid, SW ) + LOCAL( grid, NT )
		+ LOCAL( grid, NB ) + LOCAL( grid, ST )
		+ LOCAL( grid, SB ) + LOCAL( grid, ET )
		+ LOCAL( grid, EB ) + LOCAL( grid, WT )
		+ LOCAL( grid, WB );
		if( rho < minRho ) minRho = rho;
		if( rho > maxRho ) maxRho = rho;
		mass += rho;

		if( TEST_FLAG_SWEEP( grid, OBSTACLE )) {
			nObstacleCells++;
		}
		else {
			if( TEST_FLAG_SWEEP( grid, ACCEL ))
			nAccelCells++;
			else
			nFluidCells++;

			ux = + LOCAL( grid, E ) - LOCAL( grid, W )
			+ LOCAL( grid, NE ) - LOCAL( grid, NW )
			+ LOCAL( grid, SE ) - LOCAL( grid, SW )
			+ LOCAL( grid, ET ) + LOCAL( grid, EB )
			- LOCAL( grid, WT ) - LOCAL( grid, WB );
			uy = + LOCAL( grid, N ) - LOCAL( grid, S )
			+ LOCAL( grid, NE ) + LOCAL( grid, NW )
			- LOCAL( grid, SE ) - LOCAL( grid, SW )
			+ LOCAL( grid, NT ) + LOCAL( grid, NB )
			- LOCAL( grid, ST ) - LOCAL( grid, SB );
			uz = + LOCAL( grid, T ) - LOCAL( grid, B )
			+ LOCAL( grid, NT ) - LOCAL( grid, NB )
			+ LOCAL( grid, ST ) - LOCAL( grid, SB )
			+ LOCAL( grid, ET ) - LOCAL( grid, EB )
			+ LOCAL( grid, WT ) - LOCAL( grid, WB );
			u2 = (ux*ux + uy*uy + uz*uz) / (rho*rho);
			if( u2 < minU2 ) minU2 = u2;
			if( u2 > maxU2 ) maxU2 = u2;
		}
		SWEEP_END

	printf("LBM_showGridStatistics:\n"
			"\tnObstacleCells: %7i nAccelCells: %7i nFluidCells: %7i\n"
			"\tminRho: %8.4f maxRho: %8.4f mass: %e\n"
			"\tminU: %e maxU: %e\n\n", nObstacleCells, nAccelCells, nFluidCells,
			minRho, maxRho, mass, sqrt(minU2), sqrt(maxU2));

}

/*############################################################################*/

static void storeValue(FILE* file, OUTPUT_PRECISION* v) {
	const int litteBigEndianTest = 1;
	if ((*((unsigned char*) &litteBigEndianTest)) == 0) { /* big endian */
		const char* vPtr = (char*) v;
		char buffer[sizeof(OUTPUT_PRECISION )];
		int i;

		for (i = 0; i < sizeof(OUTPUT_PRECISION ); i++)
			buffer[i] = vPtr[sizeof(OUTPUT_PRECISION ) - i - 1];

		fwrite(buffer, sizeof(OUTPUT_PRECISION ), 1, file);
	} else { /* little endian */
		fwrite(v, sizeof(OUTPUT_PRECISION ), 1, file);
	}
}

/*############################################################################*/

static void loadValue(FILE* file, OUTPUT_PRECISION* v) {
	const int litteBigEndianTest = 1;
	if ((*((unsigned char*) &litteBigEndianTest)) == 0) { /* big endian */
		char* vPtr = (char*) v;
		char buffer[sizeof(OUTPUT_PRECISION )];
		int i;

		fread(buffer, sizeof(OUTPUT_PRECISION ), 1, file);

		for (i = 0; i < sizeof(OUTPUT_PRECISION ); i++)
			vPtr[i] = buffer[sizeof(OUTPUT_PRECISION ) - i - 1];
	} else { /* little endian */
		fread(v, sizeof(OUTPUT_PRECISION ), 1, file);
	}
}

/*############################################################################*/

void LBM_storeVelocityField(LBM_Grid grid, const char* filename,
		const int binary) {
	int x, y, z;
	OUTPUT_PRECISION rho, ux, uy, uz;

	FILE* file = fopen(filename, (binary ? "wb" : "w"));

	for (z = 0; z < SIZE_Z; z++) {
		for (y = 0; y < SIZE_Y; y++) {
			for (x = 0; x < SIZE_X; x++) {
				rho = +GRID_ENTRY( grid, x, y, z, C )+ GRID_ENTRY( grid, x, y, z,N  )
				      + GRID_ENTRY( grid, x, y, z, S  ) + GRID_ENTRY( grid, x, y, z, E  )
				      + GRID_ENTRY( grid, x, y, z, W  ) + GRID_ENTRY( grid, x, y, z, T  )
				      + GRID_ENTRY( grid, x, y, z, B  ) + GRID_ENTRY( grid, x, y, z, NE )
				      + GRID_ENTRY( grid, x, y, z, NW ) + GRID_ENTRY( grid, x, y, z, SE )
				      + GRID_ENTRY( grid, x, y, z, SW ) + GRID_ENTRY( grid, x, y, z, NT )
				      + GRID_ENTRY( grid, x, y, z, NB ) + GRID_ENTRY( grid, x, y, z, ST )
				      + GRID_ENTRY( grid, x, y, z, SB ) + GRID_ENTRY( grid, x, y, z, ET )
				      + GRID_ENTRY( grid, x, y, z, EB ) + GRID_ENTRY( grid, x, y, z, WT )
				      + GRID_ENTRY( grid, x, y, z, WB );
				ux = + GRID_ENTRY( grid, x, y, z, E ) - GRID_ENTRY( grid, x, y, z, W )
				+ GRID_ENTRY( grid, x, y, z, NE ) - GRID_ENTRY( grid, x, y, z, NW )
				+ GRID_ENTRY( grid, x, y, z, SE ) - GRID_ENTRY( grid, x, y, z, SW )
				+ GRID_ENTRY( grid, x, y, z, ET ) + GRID_ENTRY( grid, x, y, z, EB )
				- GRID_ENTRY( grid, x, y, z, WT ) - GRID_ENTRY( grid, x, y, z, WB );
				uy = + GRID_ENTRY( grid, x, y, z, N ) - GRID_ENTRY( grid, x, y, z, S )
				+ GRID_ENTRY( grid, x, y, z, NE ) + GRID_ENTRY( grid, x, y, z, NW )
				- GRID_ENTRY( grid, x, y, z, SE ) - GRID_ENTRY( grid, x, y, z, SW )
				+ GRID_ENTRY( grid, x, y, z, NT ) + GRID_ENTRY( grid, x, y, z, NB )
				- GRID_ENTRY( grid, x, y, z, ST ) - GRID_ENTRY( grid, x, y, z, SB );
				uz = + GRID_ENTRY( grid, x, y, z, T ) - GRID_ENTRY( grid, x, y, z, B )
				+ GRID_ENTRY( grid, x, y, z, NT ) - GRID_ENTRY( grid, x, y, z, NB )
				+ GRID_ENTRY( grid, x, y, z, ST ) - GRID_ENTRY( grid, x, y, z, SB )
				+ GRID_ENTRY( grid, x, y, z, ET ) - GRID_ENTRY( grid, x, y, z, EB )
				+ GRID_ENTRY( grid, x, y, z, WT ) - GRID_ENTRY( grid, x, y, z, WB );
				ux /= rho;
				uy /= rho;
				uz /= rho;

				if( binary ) {
					/*
					 fwrite( &ux, sizeof( ux ), 1, file );
					 fwrite( &uy, sizeof( uy ), 1, file );
					 fwrite( &uz, sizeof( uz ), 1, file );
					 */
					storeValue( file, &ux );
					storeValue( file, &uy );
					storeValue( file, &uz );
				} else
				fprintf( file, "%e %e %e\n", ux, uy, uz );

			}
		}
	}

	fclose(file);
}

/*############################################################################*/

void LBM_compareVelocityField(LBM_Grid grid, const char* filename,
		const int binary) {
	int x, y, z;
	float rho, ux, uy, uz;
	OUTPUT_PRECISION fileUx, fileUy, fileUz, dUx, dUy, dUz, diff2, maxDiff2 =
			-1e+30;

	FILE* file = fopen(filename, (binary ? "rb" : "r"));

	for (z = 0; z < SIZE_Z; z++) {
		for (y = 0; y < SIZE_Y; y++) {
			for (x = 0; x < SIZE_X; x++) {
				rho = +GRID_ENTRY( grid, x, y, z, C )+ GRID_ENTRY( grid, x, y, z,N  )
				      + GRID_ENTRY( grid, x, y, z, S  ) + GRID_ENTRY( grid, x, y, z, E  )
				      + GRID_ENTRY( grid, x, y, z, W  ) + GRID_ENTRY( grid, x, y, z, T  )
				      + GRID_ENTRY( grid, x, y, z, B  ) + GRID_ENTRY( grid, x, y, z, NE )
				      + GRID_ENTRY( grid, x, y, z, NW ) + GRID_ENTRY( grid, x, y, z, SE )
				      + GRID_ENTRY( grid, x, y, z, SW ) + GRID_ENTRY( grid, x, y, z, NT )
				      + GRID_ENTRY( grid, x, y, z, NB ) + GRID_ENTRY( grid, x, y, z, ST )
				      + GRID_ENTRY( grid, x, y, z, SB ) + GRID_ENTRY( grid, x, y, z, ET )
				      + GRID_ENTRY( grid, x, y, z, EB ) + GRID_ENTRY( grid, x, y, z, WT )
				      + GRID_ENTRY( grid, x, y, z, WB );
				ux = + GRID_ENTRY( grid, x, y, z, E ) - GRID_ENTRY( grid, x, y, z, W )
				+ GRID_ENTRY( grid, x, y, z, NE ) - GRID_ENTRY( grid, x, y, z, NW )
				+ GRID_ENTRY( grid, x, y, z, SE ) - GRID_ENTRY( grid, x, y, z, SW )
				+ GRID_ENTRY( grid, x, y, z, ET ) + GRID_ENTRY( grid, x, y, z, EB )
				- GRID_ENTRY( grid, x, y, z, WT ) - GRID_ENTRY( grid, x, y, z, WB );
				uy = + GRID_ENTRY( grid, x, y, z, N ) - GRID_ENTRY( grid, x, y, z, S )
				+ GRID_ENTRY( grid, x, y, z, NE ) + GRID_ENTRY( grid, x, y, z, NW )
				- GRID_ENTRY( grid, x, y, z, SE ) - GRID_ENTRY( grid, x, y, z, SW )
				+ GRID_ENTRY( grid, x, y, z, NT ) + GRID_ENTRY( grid, x, y, z, NB )
				- GRID_ENTRY( grid, x, y, z, ST ) - GRID_ENTRY( grid, x, y, z, SB );
				uz = + GRID_ENTRY( grid, x, y, z, T ) - GRID_ENTRY( grid, x, y, z, B )
				+ GRID_ENTRY( grid, x, y, z, NT ) - GRID_ENTRY( grid, x, y, z, NB )
				+ GRID_ENTRY( grid, x, y, z, ST ) - GRID_ENTRY( grid, x, y, z, SB )
				+ GRID_ENTRY( grid, x, y, z, ET ) - GRID_ENTRY( grid, x, y, z, EB )
				+ GRID_ENTRY( grid, x, y, z, WT ) - GRID_ENTRY( grid, x, y, z, WB );
				ux /= rho;
				uy /= rho;
				uz /= rho;

				if( binary ) {
					loadValue( file, &fileUx );
					loadValue( file, &fileUy );
					loadValue( file, &fileUz );
				}
				else {
					if( sizeof( OUTPUT_PRECISION ) == sizeof( float )) {
						fscanf( file, "%lf %lf %lf\n", &fileUx, &fileUy, &fileUz );
					}
					else {
						fscanf( file, "%f %f %f\n", &fileUx, &fileUy, &fileUz );
					}
				}

				dUx = ux - fileUx;
				dUy = uy - fileUy;
				dUz = uz - fileUz;
				diff2 = dUx*dUx + dUy*dUy + dUz*dUz;
				if( diff2 > maxDiff2 ) maxDiff2 = diff2;
			}
		}
	}

	printf("LBM_compareVelocityField: maxDiff = %e  ==>  %s\n\n",
			sqrt(maxDiff2), sqrt(maxDiff2) > 1e-5 ? "##### ERROR #####" : "OK");
	fclose(file);
}

/*############################################################################*/
__global__ void LBM_gpu_performStreamCollide(LBM_Grid srcGrid,
		LBM_Grid dstGrid) {
	float rho, ux, uy, uz, u2;
	SWEEP_VAR
	int x = threadIdx.x;
	int y = blockIdx.x;
	int z = blockIdx.y;

	i = CALC_INDEX(x,y,z,0);
	if (TEST_FLAG_SWEEP( srcGrid, OBSTACLE )) {
		DST_C ( dstGrid )= SRC_C ( srcGrid ); //printf("\ngrid[%d], grid[%d]\n", CALC_INDEX( 0,  0,   0, C), CALC_INDEX(0, 0,  0, C));
		DST_S ( dstGrid ) = SRC_N ( srcGrid );//printf("\ngrid[%d], grid[%d]\n", CALC_INDEX( 0, -1,   0, S), CALC_INDEX(0, 0,  0, N));
		DST_N ( dstGrid ) = SRC_S ( srcGrid );//printf("\ngrid[%d], grid[%d]\n", CALC_INDEX( 0, +1,   0, N), CALC_INDEX(0, 0,  0, S));
		DST_W ( dstGrid ) = SRC_E ( srcGrid );//printf("\ngrid[%d], grid[%d]\n", CALC_INDEX(-1,  0,   0, W), CALC_INDEX(0, 0,  0, E));
		DST_E ( dstGrid ) = SRC_W ( srcGrid );//printf("\ngrid[%d], grid[%d]\n", CALC_INDEX(+1,  0,   0, E), CALC_INDEX(0, 0,  0, W));
		DST_B ( dstGrid ) = SRC_T ( srcGrid );//printf("\ngrid[%d], grid[%d]\n", CALC_INDEX( 0,  0,  -1, B), CALC_INDEX(0, 0,  0, T));
		DST_T ( dstGrid ) = SRC_B ( srcGrid );//printf("\ngrid[%d], grid[%d]\n", CALC_INDEX( 0,  0,  +1, T), CALC_INDEX(0, 0,  0, B));
		DST_SW( dstGrid ) = SRC_NE( srcGrid );//printf("\ngrid[%d], grid[%d]\n", CALC_INDEX(-1, -1,   0, SW), CALC_INDEX(0, 0,  0, NE));
		DST_SE( dstGrid ) = SRC_NW( srcGrid );//printf("\ngrid[%d], grid[%d]\n", CALC_INDEX(+1, -1,   0, SE), CALC_INDEX(0, 0,  0, NW));
		DST_NW( dstGrid ) = SRC_SE( srcGrid );//printf("\ngrid[%d], grid[%d]\n", CALC_INDEX(-1, +1,   0, NW), CALC_INDEX(0, 0,  0, SE));
		DST_NE( dstGrid ) = SRC_SW( srcGrid );//printf("\ngrid[%d], grid[%d]\n", CALC_INDEX(+1, +1,   0, NE), CALC_INDEX(0, 0,  0, SW));
		DST_SB( dstGrid ) = SRC_NT( srcGrid );//printf("\ngrid[%d], grid[%d]\n", CALC_INDEX( 0, -1,  -1, SB), CALC_INDEX(0, 0,  0, NT));
		DST_ST( dstGrid ) = SRC_NB( srcGrid );//printf("\ngrid[%d], grid[%d]\n", CALC_INDEX( 0, -1,  +1, ST), CALC_INDEX(0, 0,  0, NB));
		DST_NB( dstGrid ) = SRC_ST( srcGrid );//printf("\ngrid[%d], grid[%d]\n", CALC_INDEX( 0, +1,  -1, NB), CALC_INDEX(0, 0,  0, ST));
		DST_NT( dstGrid ) = SRC_SB( srcGrid );//printf("\ngrid[%d], grid[%d]\n", CALC_INDEX( 0, +1,  +1, NT), CALC_INDEX(0, 0,  0, SB));
		DST_WB( dstGrid ) = SRC_ET( srcGrid );//printf("\ngrid[%d], grid[%d]\n", CALC_INDEX(-1,  0,  -1, WB), CALC_INDEX(0, 0,  0, ET));
		DST_WT( dstGrid ) = SRC_EB( srcGrid );//printf("\ngrid[%d], grid[%d]\n", CALC_INDEX(-1,  0,  +1, WT), CALC_INDEX(0, 0,  0, EB));
		DST_EB( dstGrid ) = SRC_WT( srcGrid );//printf("\ngrid[%d], grid[%d]\n", CALC_INDEX(+1,  0,  -1, EB), CALC_INDEX(0, 0,  0, WT));
		DST_ET( dstGrid ) = SRC_WB( srcGrid );//printf("\ngrid[%d], grid[%d]\n", CALC_INDEX(+1,  0,  +1, ET), CALC_INDEX(0, 0,  0, WB));
		//system("pause");
	}
	//calculate the mass density rho
	rho = +SRC_C ( srcGrid )+ SRC_N ( srcGrid )
	+ SRC_S ( srcGrid ) + SRC_E ( srcGrid )
	+ SRC_W ( srcGrid ) + SRC_T ( srcGrid )
	+ SRC_B ( srcGrid ) + SRC_NE( srcGrid )
	+ SRC_NW( srcGrid ) + SRC_SE( srcGrid )
	+ SRC_SW( srcGrid ) + SRC_NT( srcGrid )
	+ SRC_NB( srcGrid ) + SRC_ST( srcGrid )
	+ SRC_SB( srcGrid ) + SRC_ET( srcGrid )
	+ SRC_EB( srcGrid ) + SRC_WT( srcGrid )
	+ SRC_WB( srcGrid );
	//calculate fluid velocity u
	ux = +SRC_E ( srcGrid )- SRC_W ( srcGrid )
	+ SRC_NE( srcGrid ) - SRC_NW( srcGrid )
	+ SRC_SE( srcGrid ) - SRC_SW( srcGrid )
	+ SRC_ET( srcGrid ) + SRC_EB( srcGrid )
	- SRC_WT( srcGrid ) - SRC_WB( srcGrid );
	uy = +SRC_N ( srcGrid )- SRC_S ( srcGrid )
	+ SRC_NE( srcGrid ) + SRC_NW( srcGrid )
	- SRC_SE( srcGrid ) - SRC_SW( srcGrid )
	+ SRC_NT( srcGrid ) + SRC_NB( srcGrid )
	- SRC_ST( srcGrid ) - SRC_SB( srcGrid );
	uz = +SRC_T ( srcGrid )- SRC_B ( srcGrid )
	+ SRC_NT( srcGrid ) - SRC_NB( srcGrid )
	+ SRC_ST( srcGrid ) - SRC_SB( srcGrid )
	+ SRC_ET( srcGrid ) - SRC_EB( srcGrid )
	+ SRC_WT( srcGrid ) - SRC_WB( srcGrid );

	ux /= rho;
	uy /= rho;
	uz /= rho;

	if (TEST_FLAG_SWEEP( srcGrid, ACCEL )) {
		ux = 0.005;
		uy = 0.002;
		uz = 0.000;
	}
	u2 = 1.5 * (ux * ux + uy * uy + uz * uz);
	//calculate the local equilibrium distribution function at C
	DST_C ( dstGrid )= (1.0-OMEGA)*SRC_C ( srcGrid ) + DFL1*OMEGA*rho*(1.0 - u2);
	//calculate the local equilibrium distribution function at N, S, E, W, T, B
	DST_N ( dstGrid )= (1.0-OMEGA)*SRC_N ( srcGrid ) + DFL2*OMEGA*rho*(1.0 + uy*(4.5*uy + 3.0) - u2);
	DST_S ( dstGrid )= (1.0-OMEGA)*SRC_S ( srcGrid ) + DFL2*OMEGA*rho*(1.0 + uy*(4.5*uy - 3.0) - u2);
	DST_E ( dstGrid )= (1.0-OMEGA)*SRC_E ( srcGrid ) + DFL2*OMEGA*rho*(1.0 + ux*(4.5*ux + 3.0) - u2);
	DST_W ( dstGrid )= (1.0-OMEGA)*SRC_W ( srcGrid ) + DFL2*OMEGA*rho*(1.0 + ux*(4.5*ux - 3.0) - u2);
	DST_T ( dstGrid )= (1.0-OMEGA)*SRC_T ( srcGrid ) + DFL2*OMEGA*rho*(1.0 + uz*(4.5*uz + 3.0) - u2);
	DST_B ( dstGrid )= (1.0-OMEGA)*SRC_B ( srcGrid ) + DFL2*OMEGA*rho*(1.0 + uz*(4.5*uz - 3.0) - u2);
	//calculate the local equilibrium distribution function at NE, NW, SE, SW, NT, NB, ST, SB, ET, EB, WT, WB
	DST_NE( dstGrid )= (1.0-OMEGA)*SRC_NE( srcGrid ) + DFL3*OMEGA*rho*(1.0 + (+ux+uy)*(4.5*(+ux+uy) + 3.0) - u2);
	DST_NW( dstGrid )= (1.0-OMEGA)*SRC_NW( srcGrid ) + DFL3*OMEGA*rho*(1.0 + (-ux+uy)*(4.5*(-ux+uy) + 3.0) - u2);
	DST_SE( dstGrid )= (1.0-OMEGA)*SRC_SE( srcGrid ) + DFL3*OMEGA*rho*(1.0 + (+ux-uy)*(4.5*(+ux-uy) + 3.0) - u2);
	DST_SW( dstGrid )= (1.0-OMEGA)*SRC_SW( srcGrid ) + DFL3*OMEGA*rho*(1.0 + (-ux-uy)*(4.5*(-ux-uy) + 3.0) - u2);
	DST_NT( dstGrid )= (1.0-OMEGA)*SRC_NT( srcGrid ) + DFL3*OMEGA*rho*(1.0 + (+uy+uz)*(4.5*(+uy+uz) + 3.0) - u2);
	DST_NB( dstGrid )= (1.0-OMEGA)*SRC_NB( srcGrid ) + DFL3*OMEGA*rho*(1.0 + (+uy-uz)*(4.5*(+uy-uz) + 3.0) - u2);
	DST_ST( dstGrid )= (1.0-OMEGA)*SRC_ST( srcGrid ) + DFL3*OMEGA*rho*(1.0 + (-uy+uz)*(4.5*(-uy+uz) + 3.0) - u2);
	DST_SB( dstGrid )= (1.0-OMEGA)*SRC_SB( srcGrid ) + DFL3*OMEGA*rho*(1.0 + (-uy-uz)*(4.5*(-uy-uz) + 3.0) - u2);
	DST_ET( dstGrid )= (1.0-OMEGA)*SRC_ET( srcGrid ) + DFL3*OMEGA*rho*(1.0 + (+ux+uz)*(4.5*(+ux+uz) + 3.0) - u2);
	DST_EB( dstGrid )= (1.0-OMEGA)*SRC_EB( srcGrid ) + DFL3*OMEGA*rho*(1.0 + (+ux-uz)*(4.5*(+ux-uz) + 3.0) - u2);
	DST_WT( dstGrid )= (1.0-OMEGA)*SRC_WT( srcGrid ) + DFL3*OMEGA*rho*(1.0 + (-ux+uz)*(4.5*(-ux+uz) + 3.0) - u2);
	DST_WB( dstGrid )= (1.0-OMEGA)*SRC_WB( srcGrid ) + DFL3*OMEGA*rho*(1.0 + (-ux-uz)*(4.5*(-ux-uz) + 3.0) - u2);

}
int checkSameGrid(float* grid1, float* grid2) {
	int count = 0;
	float sub;
	//int t;
	//float t1, t2;
	for (int i = 0; i < GRID_SIZE; i++) {
		/*t = i;
		t1 = grid1[i];
		t2 = grid2[i];
		if(i>=1900 && i<1930)
			printf("\n%.3f ,  %.3f", grid1[i], grid2[i]);*/
		sub = grid2[i] - grid1[i];
		sub = sub > 0 ? sub : sub*(-1);
		if (sub > 0.0001) {
			count++;
//			if (count < 100){
//				printf("\ni = %d (%d), %.10f %.10f %.10f", i, (i%20)* 1300000 + (i/20), grid1[i], grid2[i], sub);
//				//analyze(i);
//			}

		}
	}
	return count;
}
void LBM_showGridStatistics2(LBM_Grid grid, LBM_Grid d_grid) {
	int nObstacleCells = 0, nAccelCells = 0, nFluidCells = 0;
	float ux, uy, uz;
	float minU2 = 1e+30, maxU2 = -1e+30, u2;
	float minRho = 1e+30, maxRho = -1e+30, rho;
	float mass = 0;

	int d_nObstacleCells = 0, d_nAccelCells = 0, d_nFluidCells = 0;
	float d_ux, d_uy, d_uz;
	float d_minU2 = 1e+30, d_maxU2 = -1e+30, d_u2;
	float d_minRho = 1e+30, d_maxRho = -1e+30, d_rho;
	float d_mass = 0;
	int ta1[1300000], t1 = 0 ;
	int ta2[1300000], t2 = 0;
	SWEEP_VAR

	SWEEP_START( 0, 0, 0, 0, 0, SIZE_Z )
		rho = +LOCAL( grid, C )+ LOCAL( grid, N )
		+ LOCAL( grid, S ) + LOCAL( grid, E )
		+ LOCAL( grid, W ) + LOCAL( grid, T )
		+ LOCAL( grid, B ) + LOCAL( grid, NE )
		+ LOCAL( grid, NW ) + LOCAL( grid, SE )
		+ LOCAL( grid, SW ) + LOCAL( grid, NT )
		+ LOCAL( grid, NB ) + LOCAL( grid, ST )
		+ LOCAL( grid, SB ) + LOCAL( grid, ET )
		+ LOCAL( grid, EB ) + LOCAL( grid, WT )
		+ LOCAL( grid, WB );
		if( rho < minRho ) minRho = rho;
		if( rho > maxRho ) maxRho = rho;
		mass += rho;

		if( TEST_FLAG_SWEEP( grid, OBSTACLE )) {
			nObstacleCells++;
			ta1[t1] = 1;
			t1++;
		}
		else {
			if( TEST_FLAG_SWEEP( grid, ACCEL )){
				nAccelCells++;
				ta1[t1] = 2;
				t1++;
			}
			else{
			nFluidCells++;
			ta1[t1] = 0;
			t1++;
			}
			ux = + LOCAL( grid, E ) - LOCAL( grid, W )
			+ LOCAL( grid, NE ) - LOCAL( grid, NW )
			+ LOCAL( grid, SE ) - LOCAL( grid, SW )
			+ LOCAL( grid, ET ) + LOCAL( grid, EB )
			- LOCAL( grid, WT ) - LOCAL( grid, WB );
			uy = + LOCAL( grid, N ) - LOCAL( grid, S )
			+ LOCAL( grid, NE ) + LOCAL( grid, NW )
			- LOCAL( grid, SE ) - LOCAL( grid, SW )
			+ LOCAL( grid, NT ) + LOCAL( grid, NB )
			- LOCAL( grid, ST ) - LOCAL( grid, SB );
			uz = + LOCAL( grid, T ) - LOCAL( grid, B )
			+ LOCAL( grid, NT ) - LOCAL( grid, NB )
			+ LOCAL( grid, ST ) - LOCAL( grid, SB )
			+ LOCAL( grid, ET ) - LOCAL( grid, EB )
			+ LOCAL( grid, WT ) - LOCAL( grid, WB );

			u2 = (ux*ux + uy*uy + uz*uz) / (rho*rho);
			if( u2 < minU2 ) minU2 = u2;
			if( u2 > maxU2 ) maxU2 = u2;
		}
		//printf("\n%e",maxU2);
		d_rho = + LOCAL( d_grid, C ) + LOCAL( d_grid, N )
				+ LOCAL( d_grid, S ) + LOCAL( d_grid, E )
				+ LOCAL( d_grid, W ) + LOCAL( d_grid, T )
				+ LOCAL( d_grid, B ) + LOCAL( d_grid, NE )
				+ LOCAL( d_grid, NW ) + LOCAL( d_grid, SE )
				+ LOCAL( d_grid, SW ) + LOCAL( d_grid, NT )
				+ LOCAL( d_grid, NB ) + LOCAL( d_grid, ST )
				+ LOCAL( d_grid, SB ) + LOCAL( d_grid, ET )
				+ LOCAL( d_grid, EB ) + LOCAL( d_grid, WT )
				+ LOCAL( d_grid, WB );
		/*if(abs(d_rho - rho) > 0.00001 )
		{
		 printf("\nAt i = %d", i);
		 printf("\n%.10f %.10f",LOCAL( grid, C ), LOCAL(d_grid,C));
		 printf("\n%.10f %.10f",LOCAL( grid, N ), LOCAL(d_grid,N));
		 printf("\n%.10f %.10f",LOCAL( grid, S ), LOCAL(d_grid,S));
		 printf("\n%.10f %.10f",LOCAL( grid, E ), LOCAL(d_grid,E));
		 printf("\n%.10f %.10f",LOCAL( grid, W ), LOCAL(d_grid,W));
		 printf("\n%.10f %.10f",LOCAL( grid, T ), LOCAL(d_grid,T));
		 printf("\n%.10f %.10f",LOCAL( grid, B ), LOCAL(d_grid,B));
		 printf("\n%.10f %.10f",LOCAL( grid, NE ), LOCAL(d_grid,NE));
		 printf("\n%.10f %.10f",LOCAL( grid, NW ), LOCAL(d_grid,NW));
		 printf("\n%.10f %.10f",LOCAL( grid, SE ), LOCAL(d_grid,SE));
		 printf("\n%.10f %.10f",LOCAL( grid, SW ), LOCAL(d_grid,SW));
		 printf("\n%.10f %.10f",LOCAL( grid, NT ), LOCAL(d_grid,NT));
		 printf("\n%.10f %.10f",LOCAL( grid, NB ), LOCAL(d_grid,NB));
		 printf("\n%.10f %.10f",LOCAL( grid, ST ), LOCAL(d_grid,ST));
		 printf("\n%.10f %.10f",LOCAL( grid, SB ), LOCAL(d_grid,SB));
		 printf("\n%.10f %.10f",LOCAL( grid, ET ), LOCAL(d_grid,ET));
		 printf("\n%.10f %.10f",LOCAL( grid, EB ), LOCAL(d_grid,EB));
		 printf("\n%.10f %.10f",LOCAL( grid, WT ), LOCAL(d_grid,WT));
		 printf("\n%.10f %.10f",LOCAL( grid, WB ), LOCAL(d_grid,WB));

		 }*/
		if( d_rho < d_minRho ) d_minRho = d_rho;
		if( d_rho > d_maxRho ) d_maxRho = d_rho;
		d_mass += d_rho;

		if( TEST_FLAG_SWEEP( d_grid, OBSTACLE )) {
			//printf("\n%.10f",  LOCAL(d_grid,FLAGS));
			d_nObstacleCells++;
			ta2[t2] = 1;
			t2++;

		}
		else {
			if( TEST_FLAG_SWEEP( d_grid, ACCEL )){
			d_nAccelCells++;
			ta2[t2] = 2;
			t2++;
			}
			else{
			d_nFluidCells++;
			ta2[t2] = 0;
			t2++;
			}

			d_ux = + LOCAL( d_grid, E ) - LOCAL( d_grid, W )
			+ LOCAL( d_grid, NE ) - LOCAL( d_grid, NW )
			+ LOCAL( d_grid, SE ) - LOCAL( d_grid, SW )
			+ LOCAL( d_grid, ET ) + LOCAL( d_grid, EB )
			- LOCAL( d_grid, WT ) - LOCAL( d_grid, WB );
			d_uy = + LOCAL( d_grid, N ) - LOCAL( d_grid, S )
			+ LOCAL( d_grid, NE ) + LOCAL( d_grid, NW )
			- LOCAL( d_grid, SE ) - LOCAL( d_grid, SW )
			+ LOCAL( d_grid, NT ) + LOCAL( d_grid, NB )
			- LOCAL( d_grid, ST ) - LOCAL( d_grid, SB );
			d_uz = + LOCAL( d_grid, T ) - LOCAL( d_grid, B )
			+ LOCAL( d_grid, NT ) - LOCAL( d_grid, NB )
			+ LOCAL( d_grid, ST ) - LOCAL( d_grid, SB )
			+ LOCAL( d_grid, ET ) - LOCAL( d_grid, EB )
			+ LOCAL( d_grid, WT ) - LOCAL( d_grid, WB );
			d_u2 = (d_ux*d_ux + d_uy*d_uy + d_uz*d_uz) / (d_rho*d_rho);
			if( d_u2 < d_minU2 ) d_minU2 = d_u2;
			if( d_u2 > d_maxU2 ) d_maxU2 = d_u2;
		}
		SWEEP_END

	printf("Host - LBM_showGridStatistics:\n"
			"\tnObstacleCells: %7i nAccelCells: %7i nFluidCells: %7i\n"
			"\tminRho: %8.4f maxRho: %8.4f mass: %e\n"
			"\tminU: %e maxU: %e\n\n", nObstacleCells, nAccelCells, nFluidCells,
			minRho, maxRho, mass, sqrt(minU2), sqrt(maxU2));
	PRINT_LINE;
	printf("Device - LBM_showGridStatistics:\n"
			"\tnObstacleCells: %7i nAccelCells: %7i nFluidCells: %7i\n"
			"\tminRho: %8.4f maxRho: %8.4f mass: %e\n"
			"\tminU: %e maxU: %e\n\n", d_nObstacleCells, d_nAccelCells,
			d_nFluidCells, d_minRho, d_maxRho, d_mass, sqrt(d_minU2),
			sqrt(d_maxU2));

	/*for(int i=0;i<1300000; i++)
		if(ta1[i]!=ta2[i])
			if(i<5000)
				printf("\ni = %d, %d <> %d", i, ta1[i], ta2[i]);*/

}

int analyze(int i)
{
	//int d_c, d_n, d_s, d_e, d_w, d_t, d_b, d_ne, d_nw, d_se, d_sw, d_nt, d_nb, d_st, d_sb, d_et, d_eb, d_wt, d_wb;


	int d_arr[19];
	int idx;
	int e, new_i;
	e = i;
	while(e>=20)
	{
		e = e - 20;
	}
	new_i = i - e;

	d_arr[0] = CALC_INDEX( 0,  0,  0, 0 );
	d_arr[1] = CALC_INDEX( 0, +1,  0, 1 );
	d_arr[2] = CALC_INDEX( 0, -1,  0, 2 );
	d_arr[3] = CALC_INDEX(+1,  0,  0, 3 );
	d_arr[4] = CALC_INDEX(-1,  0,  0, 4 );
	d_arr[5] = CALC_INDEX( 0,  0, +1, 5 );
	d_arr[6] = CALC_INDEX( 0,  0, -1, 6 );
	d_arr[7] = CALC_INDEX(+1, +1,  0, 7 );
	d_arr[8] = CALC_INDEX(-1, +1,  0, 8 );
	d_arr[9] = CALC_INDEX(+1, -1,  0, 9 );
	d_arr[10]= CALC_INDEX(-1, -1,  0, 10 );
	d_arr[11]= CALC_INDEX( 0, +1, +1, 11 );
	d_arr[12]= CALC_INDEX( 0, +1, -1, 12 );
	d_arr[13]= CALC_INDEX( 0, -1, +1, 13 );
	d_arr[14]= CALC_INDEX( 0, -1, -1, 14 );
	d_arr[15]= CALC_INDEX(+1,  0, +1, 15 );
	d_arr[16]= CALC_INDEX(+1,  0, -1, 16 );
	d_arr[17]= CALC_INDEX(-1,  0, +1, 17 );
	d_arr[18]= CALC_INDEX(-1,  0, -1, 18 );


	if( new_i > d_arr[e] )
		idx = (new_i - d_arr[e])/20 + (new_i - d_arr[e] ) % 20;
	else
		idx = (d_arr[e] - new_i)/20 + (d_arr[e] - new_i) % 20;

	printf("\nAnalyze ---- d_arr[%d] = %d, i = %d, idx = %d", e, d_arr[e], new_i, idx);

	for(int z=0; z<130;z++)
		for(int y=0;y<100; y++)
			for(int x =0; x<100;x++)
				if((x + 100*y + 100*100*z) == idx)
				{
					printf("\nx = %d, y = %d, z = %d",x,y,z);
					return 1;
				}
	return 0;
}

void LBM_allocateGrid_SoA_w_Struct(Distributions *dist, int size)
{
	//dist->size = NUM_ELEMENTS_SOA;
	for(int i=0;i<QQ;i++)
	{
		cudaMallocManaged(&(dist->f[i]), sizeof(float)*size);
	}
	//cudaMallocManaged(&flags, sizeof(unsigned int)*NUM_ELEMENTS_SOA);

	printf("\nLBM_allocateGrid_SoA_w_Struct done.");
	printf("\n>>>Size of each distribution = %d", size);
}

void LBM_allocateGrid_SoA_w_Struct_arr(float **fc, float **fn, float **fs, float **fe,float **fw,float **ft,float **fb,float **fne,float **fnw,float **fse,float **fsw,float **fnt,float **fnb,float **fst,float **fsb,float **fet,float **feb,float **fwt,float **fwb, int size)
{
	cudaMallocManaged((fc), sizeof(float)*size);
	cudaMallocManaged((fn), sizeof(float)*size);
	cudaMallocManaged((fs), sizeof(float)*size);
	cudaMallocManaged((fe), sizeof(float)*size);
	cudaMallocManaged((fw), sizeof(float)*size);
	cudaMallocManaged((ft), sizeof(float)*size);
	cudaMallocManaged((fb), sizeof(float)*size);
	cudaMallocManaged((fne), sizeof(float)*size);
	cudaMallocManaged((fnw), sizeof(float)*size);
	cudaMallocManaged((fse), sizeof(float)*size);
	cudaMallocManaged((fsw), sizeof(float)*size);
	cudaMallocManaged((fnt), sizeof(float)*size);
	cudaMallocManaged((fnb), sizeof(float)*size);
	cudaMallocManaged((fst), sizeof(float)*size);
	cudaMallocManaged((fsb), sizeof(float)*size);
	cudaMallocManaged((fet), sizeof(float)*size);
	cudaMallocManaged((feb), sizeof(float)*size);
	cudaMallocManaged((fwt), sizeof(float)*size);
	cudaMallocManaged((fwb), sizeof(float)*size);

	printf("\nLBM_allocateGrid_SoA_w_Struct_arr done.");
	printf("\n>>>Size of each distribution = %d", size);

}
void LBM_freeGrid_SoA_w_Struct(Distributions *dist)
{
	if(dist!=NULL)
	{
		for(int i=0;i<QQ;i++)
			cudaFree(dist->f[i]);
	}
}
void LBM_freeGrid_SoA_w_Struct_arr(float *fc, float *fn, float *fs, float *fe,float *fw,float *ft,float *fb,float *fne,float *fnw,float *fse,float *fsw,float *fnt,float *fnb,float *fst,float *fsb,float *fet,float *feb,float *fwt,float *fwb)
{
	cudaFree((fc));
	cudaFree((fn));
	cudaFree((fs));
	cudaFree((fe));
	cudaFree((fw));
	cudaFree((ft));
	cudaFree((fb));
	cudaFree((fne));
	cudaFree((fnw));
	cudaFree((fse));
	cudaFree((fsw));
	cudaFree((fnt));
	cudaFree((fnb));
	cudaFree((fst));
	cudaFree((fsb));
	cudaFree((fet));
	cudaFree((feb));
	cudaFree((fwt));
	cudaFree((fwb));
}
void LBM_convertToSoA_w_Struct(LBM_Grid grid, int size, Distributions *dist, unsigned char * flags)
{
	int i, j;
	//int cur_pos = 0;
	int x,y,z;

	//FILE *f = fopen("soa_data.txt", "w");
	if(dist!=NULL){
	j = 0;
	for(z=1;z<SIZE_ZZ;z++){
		for(y=1;y<SIZE_YY;y++){
			for(x=0;x<SIZE_XX;x++){
				i = (z * SIZE_YY + y) * SIZE_XX + x;
				//fprintf(f, "\ni = %d(%d, %d, %d), j = %d, val = %.8f",i,x,y,z,j,grid[j]);
				dist->f[0][i] = grid[j++];
				//fprintf(f, "\ni = %d(%d, %d, %d), j = %d, val = %.8f",i,x,y,z,j,grid[j]);
				dist->f[1][i] = grid[j++];
				//fprintf(f, "\ni = %d(%d, %d, %d), j = %d, val = %.8f",i,x,y,z,j,grid[j]);
				dist->f[2][i] = grid[j++];
				//fprintf(f, "\ni = %d(%d, %d, %d), j = %d, val = %.8f",i,x,y,z,j,grid[j]);
				dist->f[3][i] = grid[j++];
				//fprintf(f, "\ni = %d(%d, %d, %d), j = %d, val = %.8f",i,x,y,z,j,grid[j]);
				dist->f[4][i] = grid[j++];
				//fprintf(f, "\ni = %d(%d, %d, %d), j = %d, val = %.8f",i,x,y,z,j,grid[j]);
				dist->f[5][i] = grid[j++];
				//fprintf(f, "\ni = %d(%d, %d, %d), j = %d, val = %.8f",i,x,y,z,j,grid[j]);
				dist->f[6][i] = grid[j++];
				//fprintf(f, "\ni = %d(%d, %d, %d), j = %d, val = %.8f",i,x,y,z,j,grid[j]);
				dist->f[7][i] = grid[j++];
				//fprintf(f, "\ni = %d(%d, %d, %d), j = %d, val = %.8f",i,x,y,z,j,grid[j]);
				dist->f[8][i] = grid[j++];
				//fprintf(f, "\ni = %d(%d, %d, %d), j = %d, val = %.8f",i,x,y,z,j,grid[j]);
				dist->f[9][i] = grid[j++];
				//fprintf(f, "\ni = %d(%d, %d, %d), j = %d, val = %.8f",i,x,y,z,j,grid[j]);
				dist->f[10][i] = grid[j++];
				//fprintf(f, "\ni = %d(%d, %d, %d), j = %d, val = %.8f",i,x,y,z,j,grid[j]);
				dist->f[11][i] = grid[j++];
				//fprintf(f, "\ni = %d(%d, %d, %d), j = %d, val = %.8f",i,x,y,z,j,grid[j]);
				dist->f[12][i] = grid[j++];
				//fprintf(f, "\ni = %d(%d, %d, %d), j = %d, val = %.8f",i,x,y,z,j,grid[j]);
				dist->f[13][i] = grid[j++];
				//fprintf(f, "\ni = %d(%d, %d, %d), j = %d, val = %.8f",i,x,y,z,j,grid[j]);
				dist->f[14][i] = grid[j++];
				//fprintf(f, "\ni = %d(%d, %d, %d), j = %d, val = %.8f",i,x,y,z,j,grid[j]);
				dist->f[15][i] = grid[j++];
				//fprintf(f, "\ni = %d(%d, %d, %d), j = %d, val = %.8f",i,x,y,z,j,grid[j]);
				dist->f[16][i] = grid[j++];
				//fprintf(f, "\ni = %d(%d, %d, %d), j = %d, val = %.8f",i,x,y,z,j,grid[j]);
				dist->f[17][i] = grid[j++];
				//fprintf(f, "\ni = %d(%d, %d, %d), j = %d, val = %.8f",i,x,y,z,j,grid[j]);
				dist->f[18][i] = grid[j++];
				if((*(unsigned int*) (void*) (&grid[j])) & OBSTACLE)
					flags[i] = OBSTACLE;
				else if((*(unsigned int*) (void*) (&grid[j])) & ACCEL)
					flags[i] = ACCEL;
				else
					flags[i] = 0;
				//fprintf(f, "\nFlag = %d",flags[i]);
				j++;
			}
		}
	}
	}

	//fclose(f);
	printf("\nLBM_convertToSoA done.");
}
void LBM_convertToSoA_w_Struct_arr(LBM_Grid grid, int size, float *fc, float *fn, float *fs, float *fe,float *fw,float *ft,float *fb,float *fne,float *fnw,float *fse,float *fsw,float *fnt,float *fnb,float *fst,float *fsb,float *fet,float *feb,float *fwt,float *fwb, unsigned char* flags)
{
	int i, j;
	//int cur_pos = 0;
	int x,y,z;

	//FILE *f = fopen("soa_data.txt", "w");
	j = 0;
	for(z=1;z<SIZE_ZZ;z++){
		for(y=1;y<SIZE_YY;y++){
			for(x=0;x<SIZE_XX;x++){
				i = (z * SIZE_YY + y) * SIZE_XX + x;
				//fprintf(f, "\ni = %d(%d, %d, %d), j = %d, val = %.8f",i,x,y,z,j,grid[j]);
				fc[i] = grid[j++];
				//fprintf(f, "\ni = %d(%d, %d, %d), j = %d, val = %.8f",i,x,y,z,j,grid[j]);
				fn[i] = grid[j++];
				//fprintf(f, "\ni = %d(%d, %d, %d), j = %d, val = %.8f",i,x,y,z,j,grid[j]);
				fs[i] = grid[j++];
				//fprintf(f, "\ni = %d(%d, %d, %d), j = %d, val = %.8f",i,x,y,z,j,grid[j]);
				fe[i] = grid[j++];
				//fprintf(f, "\ni = %d(%d, %d, %d), j = %d, val = %.8f",i,x,y,z,j,grid[j]);
				fw[i] = grid[j++];
				//fprintf(f, "\ni = %d(%d, %d, %d), j = %d, val = %.8f",i,x,y,z,j,grid[j]);
				ft[i] = grid[j++];
				//fprintf(f, "\ni = %d(%d, %d, %d), j = %d, val = %.8f",i,x,y,z,j,grid[j]);
				fb[i] = grid[j++];
				//fprintf(f, "\ni = %d(%d, %d, %d), j = %d, val = %.8f",i,x,y,z,j,grid[j]);
				fne[i] = grid[j++];
				//fprintf(f, "\ni = %d(%d, %d, %d), j = %d, val = %.8f",i,x,y,z,j,grid[j]);
				fnw[i] = grid[j++];
				//fprintf(f, "\ni = %d(%d, %d, %d), j = %d, val = %.8f",i,x,y,z,j,grid[j]);
				fse[i] = grid[j++];
				//fprintf(f, "\ni = %d(%d, %d, %d), j = %d, val = %.8f",i,x,y,z,j,grid[j]);
				fsw[i] = grid[j++];
				//fprintf(f, "\ni = %d(%d, %d, %d), j = %d, val = %.8f",i,x,y,z,j,grid[j]);
				fnt[i] = grid[j++];
				//fprintf(f, "\ni = %d(%d, %d, %d), j = %d, val = %.8f",i,x,y,z,j,grid[j]);
				fnb[i] = grid[j++];
				//fprintf(f, "\ni = %d(%d, %d, %d), j = %d, val = %.8f",i,x,y,z,j,grid[j]);
				fst[i] = grid[j++];
				//fprintf(f, "\ni = %d(%d, %d, %d), j = %d, val = %.8f",i,x,y,z,j,grid[j]);
				fsb[i] = grid[j++];
				//fprintf(f, "\ni = %d(%d, %d, %d), j = %d, val = %.8f",i,x,y,z,j,grid[j]);
				fet[i] = grid[j++];
				//fprintf(f, "\ni = %d(%d, %d, %d), j = %d, val = %.8f",i,x,y,z,j,grid[j]);
				feb[i] = grid[j++];
				//fprintf(f, "\ni = %d(%d, %d, %d), j = %d, val = %.8f",i,x,y,z,j,grid[j]);
				fwt[i] = grid[j++];
				//fprintf(f, "\ni = %d(%d, %d, %d), j = %d, val = %.8f",i,x,y,z,j,grid[j]);
				fwb[i] = grid[j++];
				if((*(unsigned int*) (void*) (&grid[j])) & OBSTACLE)
					flags[i] = OBSTACLE;
				else if((*(unsigned int*) (void*) (&grid[j])) & ACCEL)
					flags[i] = ACCEL;
				else
					flags[i] = 0;
				//fprintf(f, "\nFlag = %d",flags[i]);
				j++;
			}
		}
	}


	//fclose(f);
	printf("\nLBM_convertToSoA done.");
}
void LBM_displayFlags_SoA_w_Struct(unsigned char *flags)
{
	int i,x,y,z;
	printf("\n");
	for(z=1;z<SIZE_ZZ;z++){
			for(y=1;y<SIZE_YY;y++){
				for(x=0;x<SIZE_XX;x++){
					i = (z * SIZE_YY + y) * SIZE_XX + x;
					printf("%d ", flags[i]);
				}
			}
	}

}

int checkSameGrid_SoA_w_Struct(float* grid1, Distributions *dist) {
	int count = 0;
	float sub;
	int i, j=0;
	int c=0;
	printf("\n");
	//FILE* f = fopen("soa_checkSameGrid.txt", "w");
	for (int z = 1; z < SIZE_ZZ; z++) {
			for (int y = 1; y < SIZE_YY; y++) {
				for (int x = 0; x < SIZE_XX; x++) {
					c++;
					i = (z * SIZE_YY + y) * SIZE_XX + x;
					//fprintf(f,"\ni = %d", i);
					sub = dist->f[0][i] - grid1[j++];
					sub = sub > 0 ? sub : sub * (-1);
					if (sub > 0.0001) {
						//fprintf(f,"\n0. pos = %d(%d, %d, %d) <- %d, %.8f <- %.8f", i,x,y,z, j-1, dist->f[0][i], grid1[j-1]);
						count++;
					}

					sub = dist->f[1][i] - grid1[j++];
					sub = sub > 0 ? sub : sub * (-1);
					if (sub > 0.0001) {
						//fprintf(f,"\n1. pos = %d(%d, %d, %d) <- %d, %.8f <- %.8f", i,x,y,z, j-1, dist->f[1][i], grid1[j-1]);
						count++;
					}

					sub = dist->f[2][i] - grid1[j++];
					sub = sub > 0 ? sub : sub * (-1);
					if (sub > 0.0001) {
						//fprintf(f,"\n2. pos = %d(%d, %d, %d) <- %d, %.8f <- %.8f", i,x,y,z, j-1, dist->f[2][i], grid1[j-1]);
						count++;
					}

					sub = dist->f[3][i] - grid1[j++];
					sub = sub > 0 ? sub : sub * (-1);
					if (sub > 0.0001) {
						//fprintf(f,"\n3. pos = %d(%d, %d, %d) <- %d, %.8f <- %.8f", i,x,y,z, j-1, dist->f[3][i], grid1[j-1]);
						count++;
					}

					sub = dist->f[4][i] - grid1[j++];
					sub = sub > 0 ? sub : sub * (-1);
					if (sub > 0.0001) {
						//fprintf(f,"\n4. pos = %d(%d, %d, %d) <- %d, %.8f <- %.8f", i,x,y,z, j-1, dist->f[4][i], grid1[j-1]);
						count++;
					}

					sub = dist->f[5][i] - grid1[j++];
					sub = sub > 0 ? sub : sub * (-1);
					if (sub > 0.0001) {
						//fprintf(f,"\n5. pos = %d(%d, %d, %d) <- %d, %.8f <- %.8f", i,x,y,z, j-1, dist->f[5][i], grid1[j-1]);
						count++;
					}

					sub = dist->f[6][i] - grid1[j++];
					sub = sub > 0 ? sub : sub * (-1);
					if (sub > 0.0001) {
						//fprintf(f,"\n6. pos = %d(%d, %d, %d) <- %d, %.8f <- %.8f", i,x,y,z, j-1, dist->f[6][i], grid1[j-1]);
						count++;
					}

					sub = dist->f[7][i] - grid1[j++];
					sub = sub > 0 ? sub : sub * (-1);
					if (sub > 0.0001) {
						//fprintf(f,"\n7. pos = %d(%d, %d, %d) <- %d, %.8f <- %.8f", i,x,y,z, j-1, dist->f[7][i], grid1[j-1]);
						count++;
					}

					sub = dist->f[8][i] - grid1[j++];
					sub = sub > 0 ? sub : sub * (-1);
					if (sub > 0.0001) {
						//fprintf(f,"\n8. pos = %d(%d, %d, %d) <- %d, %.8f <- %.8f", i,x,y,z, j-1, dist->f[8][i], grid1[j-1]);
						count++;
					}

					sub = dist->f[9][i] - grid1[j++];
					sub = sub > 0 ? sub : sub * (-1);
					if (sub > 0.0001) {
						//fprintf(f,"\n9. pos = %d(%d, %d, %d) <- %d, %.8f <- %.8f", i,x,y,z, j-1, dist->f[9][i], grid1[j-1]);
						count++;
					}

					sub = dist->f[10][i] - grid1[j++];
					sub = sub > 0 ? sub : sub * (-1);
					if (sub > 0.0001) {
						//fprintf(f,"\n10. pos = %d(%d, %d, %d) <- %d, %.8f <- %.8f", i,x,y,z, j-1, dist->f[10][i], grid1[j-1]);
						count++;
					}

					sub = dist->f[11][i] - grid1[j++];
					sub = sub > 0 ? sub : sub * (-1);
					if (sub > 0.0001) {
						//fprintf(f,"\n11. pos = %d(%d, %d, %d) <- %d, %.8f <- %.8f", i,x,y,z, j-1, dist->f[11][i], grid1[j-1]);
						count++;
					}

					sub = dist->f[12][i] - grid1[j++];
					sub = sub > 0 ? sub : sub * (-1);
					if (sub > 0.0001) {
						//fprintf(f,"\n12. pos = %d(%d, %d, %d) <- %d, %.8f <- %.8f", i,x,y,z, j-1, dist->f[12][i], grid1[j-1]);
						count++;
					}

					sub = dist->f[13][i] - grid1[j++];
					sub = sub > 0 ? sub : sub * (-1);
					if (sub > 0.0001) {
						//fprintf(f,"\n13. pos = %d(%d, %d, %d) <- %d, %.8f <- %.8f", i,x,y,z, j-1, dist->f[13][i], grid1[j-1]);
						count++;
					}

					sub = dist->f[14][i] - grid1[j++];
					sub = sub > 0 ? sub : sub * (-1);
					if (sub > 0.0001) {
						//fprintf(f,"\n14. pos = %d(%d, %d, %d) <- %d, %.8f <- %.8f", i,x,y,z, j-1, dist->f[14][i], grid1[j-1]);
						count++;
					}

					sub = dist->f[15][i] - grid1[j++];
					sub = sub > 0 ? sub : sub * (-1);
					if (sub > 0.0001) {
						//fprintf(f,"\n15. pos = %d(%d, %d, %d) <- %d, %.8f <- %.8f", i,x,y,z, j-1, dist->f[15][i], grid1[j-1]);
						count++;
					}

					sub = dist->f[16][i] - grid1[j++];
					sub = sub > 0 ? sub : sub * (-1);
					if (sub > 0.0001) {
						//fprintf(f,"\n16. pos = %d(%d, %d, %d) <- %d, %.8f <- %.8f", i,x,y,z, j-1, dist->f[16][i], grid1[j-1]);
						count++;
					}

					sub = dist->f[17][i] - grid1[j++];
					sub = sub > 0 ? sub : sub * (-1);
					if (sub > 0.0001) {
						//fprintf(f,"\n17. pos = %d(%d, %d, %d) <- %d, %.8f <- %.8f", i,x,y,z, j-1, dist->f[17][i], grid1[j-1]);
						count++;
					}

					sub = dist->f[18][i] - grid1[j++];
					sub = sub > 0 ? sub : sub * (-1);
					if (sub > 0.0001) {
						//fprintf(f,"\n18. pos = %d(%d, %d, %d) <- %d, %.8f <- %.8f", i,x,y,z, j-1, dist->f[18][i], grid1[j-1]);
						count++;
					}
					 j++;
				}
			}
		}
		//printf("\nCount = %d", c);
		//fclose(f);
		return count;
}
int checkSameGrid_SoA_w_Struct_arr(float* grid1, float *fc, float *fn, float *fs, float *fe,float *fw,float *ft,float *fb,float *fne,float *fnw,float *fse,float *fsw,float *fnt,float *fnb,float *fst,float *fsb,float *fet,float *feb,float *fwt,float *fwb)
{
	int count = 0;
	float sub;
	int i, j=0;
	int c=0;
	printf("\n");
	//FILE* f = fopen("soa_checkSameGrid.txt", "w");
	for (int z = 1; z < SIZE_ZZ; z++) {
			for (int y = 1; y < SIZE_YY; y++) {
				for (int x = 0; x < SIZE_XX; x++) {
					c++;
					i = (z * SIZE_YY + y) * SIZE_XX + x;
					//fprintf(f,"\ni = %d", i);
					sub = fc[i] - grid1[j++];
					sub = sub > 0 ? sub : sub * (-1);
					if (sub > 0.0001) {
						//fprintf(f,"\n0. pos = %d(%d, %d, %d) <- %d, %.8f <- %.8f", i,x,y,z, j-1, dist->f[0][i], grid1[j-1]);
						count++;
					}

					sub = fn[i] - grid1[j++];
					sub = sub > 0 ? sub : sub * (-1);
					if (sub > 0.0001) {
						//fprintf(f,"\n1. pos = %d(%d, %d, %d) <- %d, %.8f <- %.8f", i,x,y,z, j-1, dist->f[1][i], grid1[j-1]);
						count++;
					}

					sub = fs[i] - grid1[j++];
					sub = sub > 0 ? sub : sub * (-1);
					if (sub > 0.0001) {
						//fprintf(f,"\n2. pos = %d(%d, %d, %d) <- %d, %.8f <- %.8f", i,x,y,z, j-1, dist->f[2][i], grid1[j-1]);
						count++;
					}

					sub = fe[i] - grid1[j++];
					sub = sub > 0 ? sub : sub * (-1);
					if (sub > 0.0001) {
						//fprintf(f,"\n3. pos = %d(%d, %d, %d) <- %d, %.8f <- %.8f", i,x,y,z, j-1, dist->f[3][i], grid1[j-1]);
						count++;
					}

					sub = fw[i] - grid1[j++];
					sub = sub > 0 ? sub : sub * (-1);
					if (sub > 0.0001) {
						//fprintf(f,"\n4. pos = %d(%d, %d, %d) <- %d, %.8f <- %.8f", i,x,y,z, j-1, dist->f[4][i], grid1[j-1]);
						count++;
					}

					sub = ft[i] - grid1[j++];
					sub = sub > 0 ? sub : sub * (-1);
					if (sub > 0.0001) {
						//fprintf(f,"\n5. pos = %d(%d, %d, %d) <- %d, %.8f <- %.8f", i,x,y,z, j-1, dist->f[5][i], grid1[j-1]);
						count++;
					}

					sub = fb[i] - grid1[j++];
					sub = sub > 0 ? sub : sub * (-1);
					if (sub > 0.0001) {
						//fprintf(f,"\n6. pos = %d(%d, %d, %d) <- %d, %.8f <- %.8f", i,x,y,z, j-1, dist->f[6][i], grid1[j-1]);
						count++;
					}

					sub = fne[i] - grid1[j++];
					sub = sub > 0 ? sub : sub * (-1);
					if (sub > 0.0001) {
						//fprintf(f,"\n7. pos = %d(%d, %d, %d) <- %d, %.8f <- %.8f", i,x,y,z, j-1, dist->f[7][i], grid1[j-1]);
						count++;
					}

					sub = fnw[i] - grid1[j++];
					sub = sub > 0 ? sub : sub * (-1);
					if (sub > 0.0001) {
						//fprintf(f,"\n8. pos = %d(%d, %d, %d) <- %d, %.8f <- %.8f", i,x,y,z, j-1, dist->f[8][i], grid1[j-1]);
						count++;
					}

					sub = fse[i] - grid1[j++];
					sub = sub > 0 ? sub : sub * (-1);
					if (sub > 0.0001) {
						//fprintf(f,"\n9. pos = %d(%d, %d, %d) <- %d, %.8f <- %.8f", i,x,y,z, j-1, dist->f[9][i], grid1[j-1]);
						count++;
					}

					sub = fsw[i] - grid1[j++];
					sub = sub > 0 ? sub : sub * (-1);
					if (sub > 0.0001) {
						//fprintf(f,"\n10. pos = %d(%d, %d, %d) <- %d, %.8f <- %.8f", i,x,y,z, j-1, dist->f[10][i], grid1[j-1]);
						count++;
					}

					sub = fnt[i] - grid1[j++];
					sub = sub > 0 ? sub : sub * (-1);
					if (sub > 0.0001) {
						//fprintf(f,"\n11. pos = %d(%d, %d, %d) <- %d, %.8f <- %.8f", i,x,y,z, j-1, dist->f[11][i], grid1[j-1]);
						count++;
					}

					sub = fnb[i] - grid1[j++];
					sub = sub > 0 ? sub : sub * (-1);
					if (sub > 0.0001) {
						//fprintf(f,"\n12. pos = %d(%d, %d, %d) <- %d, %.8f <- %.8f", i,x,y,z, j-1, dist->f[12][i], grid1[j-1]);
						count++;
					}

					sub = fst[i] - grid1[j++];
					sub = sub > 0 ? sub : sub * (-1);
					if (sub > 0.0001) {
						//fprintf(f,"\n13. pos = %d(%d, %d, %d) <- %d, %.8f <- %.8f", i,x,y,z, j-1, dist->f[13][i], grid1[j-1]);
						count++;
					}

					sub = fsb[i] - grid1[j++];
					sub = sub > 0 ? sub : sub * (-1);
					if (sub > 0.0001) {
						//fprintf(f,"\n14. pos = %d(%d, %d, %d) <- %d, %.8f <- %.8f", i,x,y,z, j-1, dist->f[14][i], grid1[j-1]);
						count++;
					}

					sub = fet[i] - grid1[j++];
					sub = sub > 0 ? sub : sub * (-1);
					if (sub > 0.0001) {
						//fprintf(f,"\n15. pos = %d(%d, %d, %d) <- %d, %.8f <- %.8f", i,x,y,z, j-1, dist->f[15][i], grid1[j-1]);
						count++;
					}

					sub = feb[i] - grid1[j++];
					sub = sub > 0 ? sub : sub * (-1);
					if (sub > 0.0001) {
						//fprintf(f,"\n16. pos = %d(%d, %d, %d) <- %d, %.8f <- %.8f", i,x,y,z, j-1, dist->f[16][i], grid1[j-1]);
						count++;
					}

					sub = fwt[i] - grid1[j++];
					sub = sub > 0 ? sub : sub * (-1);
					if (sub > 0.0001) {
						//fprintf(f,"\n17. pos = %d(%d, %d, %d) <- %d, %.8f <- %.8f", i,x,y,z, j-1, dist->f[17][i], grid1[j-1]);
						count++;
					}

					sub = fwb[i] - grid1[j++];
					sub = sub > 0 ? sub : sub * (-1);
					if (sub > 0.0001) {
						//fprintf(f,"\n18. pos = %d(%d, %d, %d) <- %d, %.8f <- %.8f", i,x,y,z, j-1, dist->f[18][i], grid1[j-1]);
						count++;
					}
					 j++;
				}
			}
		}
		//printf("\nCount = %d", c);
		//fclose(f);
		return count;
}
void LBM_showGridStatistics2_SoA_w_Struct( float* grid, Distributions *dist, unsigned char *flags)
{
	int nObstacleCells = 0, nAccelCells = 0, nFluidCells = 0;
	double ux, uy, uz;
	double minU2 = 1e+30, maxU2 = -1e+30, u2;
	double minRho = 1e+30, maxRho = -1e+30, rho;
	double mass = 0;

	int d_nObstacleCells = 0, d_nAccelCells = 0, d_nFluidCells = 0;
	double d_ux, d_uy, d_uz;
	double d_minU2 = 1e+30, d_maxU2 = -1e+30, d_u2;
	double d_minRho = 1e+30, d_maxRho = -1e+30, d_rho;
	double d_mass = 0;

	int x, y, z;
	SWEEP_VAR

	SWEEP_START( 0, 0, 0, 0, 0, SIZE_Z )
		rho = +LOCAL( grid, C )+ LOCAL( grid, N )
		+ LOCAL( grid, S ) + LOCAL( grid, E )
		+ LOCAL( grid, W ) + LOCAL( grid, T )
		+ LOCAL( grid, B ) + LOCAL( grid, NE )
		+ LOCAL( grid, NW ) + LOCAL( grid, SE )
		+ LOCAL( grid, SW ) + LOCAL( grid, NT )
		+ LOCAL( grid, NB ) + LOCAL( grid, ST )
		+ LOCAL( grid, SB ) + LOCAL( grid, ET )
		+ LOCAL( grid, EB ) + LOCAL( grid, WT )
		+ LOCAL( grid, WB );
		if( rho < minRho ) minRho = rho;
		if( rho > maxRho ) maxRho = rho;
		mass += rho;

		if( TEST_FLAG_SWEEP( grid, OBSTACLE )) {
			nObstacleCells++;
		}
		else {
			if( TEST_FLAG_SWEEP( grid, ACCEL )) {
				nAccelCells++;
		}
			else {
				nFluidCells++;
			}
		ux = + LOCAL( grid, E ) - LOCAL( grid, W )
			+ LOCAL( grid, NE ) - LOCAL( grid, NW )
			+ LOCAL( grid, SE ) - LOCAL( grid, SW )
			+ LOCAL( grid, ET ) + LOCAL( grid, EB )
			- LOCAL( grid, WT ) - LOCAL( grid, WB );
		uy = + LOCAL( grid, N ) - LOCAL( grid, S )
			+ LOCAL( grid, NE ) + LOCAL( grid, NW )
			- LOCAL( grid, SE ) - LOCAL( grid, SW )
			+ LOCAL( grid, NT ) + LOCAL( grid, NB )
			- LOCAL( grid, ST ) - LOCAL( grid, SB );
		uz = + LOCAL( grid, T ) - LOCAL( grid, B )
			+ LOCAL( grid, NT ) - LOCAL( grid, NB )
			+ LOCAL( grid, ST ) - LOCAL( grid, SB )
			+ LOCAL( grid, ET ) - LOCAL( grid, EB )
			+ LOCAL( grid, WT ) - LOCAL( grid, WB );

			u2 = (ux*ux + uy*uy + uz*uz) / (rho*rho);
			if( u2 < minU2 ) minU2 = u2;
			if( u2 > maxU2 ) maxU2 = u2;
		}
		SWEEP_END

	for (z = 1; z < SIZE_ZZ; z++) {
		for (y = 1; y < SIZE_YY; y++) {
			for (x = 0; x < SIZE_XX; x++) {
				i = (z * SIZE_YY + y) * SIZE_XX + x;
				d_rho = +dist->f[0][i] + dist->f[1][i] + dist->f[2][i]
						+ dist->f[3][i] + dist->f[4][i] + dist->f[5][i]
						+ dist->f[6][i] + dist->f[7][i] + dist->f[8][i]
						+ dist->f[9][i] + dist->f[10][i] + dist->f[11][i]
						+ dist->f[12][i] + dist->f[13][i] + dist->f[14][i]
						+ dist->f[15][i] + dist->f[16][i] + dist->f[17][i]
						+ dist->f[18][i];
				if (d_rho < d_minRho)
					d_minRho = d_rho;
				if (d_rho > d_maxRho)
					d_maxRho = d_rho;
				d_mass += d_rho;
				if (flags[i] == 1)
					d_nObstacleCells++;
				else if (flags[i] == 2)
					d_nAccelCells++;
				else
					d_nFluidCells++;
				d_ux = +dist->f[E][i] - dist->f[W][i] + dist->f[NE][i]
						- dist->f[NW][i] + dist->f[SE][i] - dist->f[SW][i]
						+ dist->f[ET][i] + dist->f[EB][i] - dist->f[WT][i]
						- dist->f[WB][i];
				d_uy = +dist->f[N][i] - dist->f[S][i] + dist->f[NE][i]
						+ dist->f[NW][i] - dist->f[SE][i] - dist->f[SW][i]
						+ dist->f[NT][i] + dist->f[NB][i] - dist->f[ST][i]
						- dist->f[SB][i];
				d_uz = +dist->f[T][i] - dist->f[B][i] + dist->f[NT][i]
						- dist->f[NB][i] + dist->f[ST][i] - dist->f[SB][i]
						+ dist->f[ET][i] - dist->f[EB][i] + dist->f[WT][i]
						- dist->f[WB][i];
				d_u2 = (d_ux * d_ux + d_uy * d_uy + d_uz * d_uz)
						/ (d_rho * d_rho);
				if (d_u2 < d_minU2)
					d_minU2 = d_u2;
				if (d_u2 > d_maxU2)
					d_maxU2 = d_u2;
			}
		}
	}

	printf("Host - LBM_showGridStatistics:\n"
			"\tnObstacleCells: %7i nAccelCells: %7i nFluidCells: %7i\n"
			"\tminRho: %8.4f maxRho: %8.4f mass: %e\n"
			"\tminU: %e maxU: %e\n\n", nObstacleCells, nAccelCells, nFluidCells,
			minRho, maxRho, mass, sqrt(minU2), sqrt(maxU2));
	PRINT_LINE;
	printf("Device - LBM_showGridStatistics:\n"
			"\tnObstacleCells: %7i nAccelCells: %7i nFluidCells: %7i\n"
			"\tminRho: %8.4f maxRho: %8.4f mass: %e\n"
			"\tminU: %e maxU: %e\n\n", d_nObstacleCells, d_nAccelCells,
			d_nFluidCells, d_minRho, d_maxRho, d_mass, sqrt(d_minU2),
			sqrt(d_maxU2));
}
/*void LBM_convertToSoA_wo_Struct(LBM_Grid grid, float *grid2, unsigned char* flags)
{
	int i, j, ii;
	int e;
	int x,y,z;
	int offset;
	//FILE *f = fopen("soa_data.txt", "w");
	if(grid2!=NULL){
	j = 0;
	for(z=1;z<SIZE_ZZ;z++){
		for(y=1;y<SIZE_YY;y++){
			for(x=0;x<SIZE_XX;x++){
				i = (z * SIZE_YY + y) * SIZE_XX + x;
				for(e=0;e<QQ;e++){
					offset = SIZE_XX*SIZE_YY*SIZE_ZZ*e;
					ii = i+ offset;
					grid2[ii] = grid[j];
					//fprintf(f,"\n%d (%d, %d, %d, %d)  %d %d %.8f", ii, x, y, z, e, i, j, grid2[ii]);
					j++;
				}
				if((*(unsigned int*) (void*) (&grid[j])) & OBSTACLE)
					flags[i] = OBSTACLE;
				else if((*(unsigned int*) (void*) (&grid[j])) & ACCEL)
					flags[i] = ACCEL;
				else
					flags[i] = 0;
				//fprintf(f, "\nFlag = %d",flags[i]);
				//printf("%d ", flags[i]);
				j++;
			}
		}
	}
	}

	//fclose(f);
	printf("\nLBM_convertToSoA done.");
}*/
void LBM_convertToSoA_wo_Struct(LBM_Grid grid, float *grid2, unsigned char* flags)
{
	int i, j;//, d = 0;
	int e;
	int offset;
	//FILE *f = fopen("soa_data.txt", "w");
	if(grid2!=NULL){
		j = 0;
		i = 0;
		offset = SIZE_X*SIZE_Y*SIZE_Z;
		while(j < SIZE_X*SIZE_Y*SIZE_Z*N_CELL_ENTRIES)
		{

			for(e = 0; e < 19; e++){
				grid2[MARGIN_L_SIZE + i + e*offset] = grid[j+e];
				//fprintf(f,"\n%d %d, %.15f %.15f", MARGIN_L_SIZE + i + e*offset, j+e, grid2[MARGIN_L_SIZE + i + e*offset], grid[j+e]);
			}
			if((*(unsigned int*) (void*) (&grid[j+e])) & OBSTACLE)
				flags[i] = OBSTACLE;
			else if((*(unsigned int*) (void*) (&grid[j+e])) & ACCEL)
				flags[i] = ACCEL;
			else
				flags[i] = 0;
			//fprintf(f,"%2d", flags[i]); d++;
			//if(d==16) {fprintf(f,"\n");d=0;}
			//if(j==360) printf("\nflag cell 18 = %d, %d", (*(unsigned int*) (void*) (&grid[j+19])) & OBSTACLE,(*(unsigned int*) (void*) (&grid[j+19])) & ACCEL );
			i++;
			j+=N_CELL_ENTRIES;
		}
	}
	//fclose(f);
	printf("\nLBM_convertToSoA done.");
}
inline int getTiledIndex(int x, int y, int z, int xx, int yy, int zz)
{
	int blockId = (x/TILED_WIDTH_X) + (y/TILED_WIDTH_Y)*(SIZE_X/TILED_WIDTH_X) + (z/TILED_WIDTH_Z)*(SIZE_X/TILED_WIDTH_X)*(SIZE_Y/TILED_WIDTH_Y);
	int idx = blockId * (TILED_WIDTH_X * TILED_WIDTH_Y * TILED_WIDTH_Z) + ((zz % TILED_WIDTH_Z) * TILED_WIDTH_X * TILED_WIDTH_Y) + ((yy%TILED_WIDTH_Y) * TILED_WIDTH_X) + (xx%TILED_WIDTH_X);
	return idx;
}
void LBM_convertToSoA_wo_Struct_New_Layout(LBM_Grid grid, float *grid2, unsigned char* flags) {
	int i, j;
	int e;
	int x, y, z, xx, yy, zz;
	int offset;
	FILE *f = fopen("soa_data.txt", "w");
	/*for(z = 0; z < SIZE_Z; z++)
		for(y = 0; y < SIZE_Y; y++)
			for(x = 0; x < SIZE_X; x++)
				fprintf(f,"%d, x = %d, y = %d, z = %d\n", x + y*SIZE_X + z*SIZE_X*SIZE_Y, x, y, z);
	fprintf(f,"=========================================================");
	*/
	if (grid2 != NULL) {
		j = 0;
		offset = SIZE_X * SIZE_Y * SIZE_Z;
		for (z = 0; z < SIZE_Z; z+=TILED_WIDTH_Z) {
			for (y = 0; y < SIZE_Y; y+=TILED_WIDTH_Y) {
				for (x = 0; x < SIZE_X; x+=TILED_WIDTH_X) {
					for(zz = z; zz < z + TILED_WIDTH_Z; zz++ ){
						for(yy = y; yy < y + TILED_WIDTH_Y; yy++){
							for(xx = x; xx < x + TILED_WIDTH_X; xx++){
								//calculate the transformed index in the new array
								//SIZE_X, SIZE_Y and SIZE_Z are size of x-dimension, y-dimension and z-dimension respectively.jj
								i  = xx + (yy)* SIZE_X + (zz)*SIZE_X*SIZE_Y;
								//int ii = (x+xx) + (y+yy)* SIZE_X + (z+ zz)*SIZE_X*SIZE_Y;
								fprintf(f,"%d => %d, x = %d, y = %d, z = %d, xx = %d, yy = %d, zz = %d\n",i, j ,x, y, z, xx, yy, zz);
								for (e = 0; e < QQ; e++) {
									//grid2[MARGIN_L_SIZE + i + e*offset] = grid[j+e];
									grid2[MARGIN_L_SIZE + j + e*offset] = grid[i+e];
								}
								if ((*(unsigned int*) (void*) (&grid[i])) & OBSTACLE)
									flags[j] = OBSTACLE;
								else if ((*(unsigned int*) (void*) (&grid[i])) & ACCEL)
									flags[j] = ACCEL;
								else
									flags[j] = 0;
								j++;
							}
						}

					}

				}
			}
		}
	}
	fclose(f);
	printf("\nLBM_convertToSoA_wo_Struct_New_Layout done.");
}

//LBM_convertToSoA_wo_Struct partitioned
void LBM_convertToSoA_partitioned(LBM_Grid grid, float *grid2, unsigned char* flags)
{
	int i, j, jj;
	int e;
	int offset;
	int partcount = 0;
	int partnum = 1;
	//FILE *f = fopen("soa_data.txt", "w");
	if(grid2!=NULL){
		j = 0;
		i = 0;
		offset = SIZE_X*SIZE_Y*(SIZE_Z + LAYERS_NUM);
		while(j < SIZE_X*SIZE_Y*(SIZE_Z+LAYERS_NUM)*N_CELL_ENTRIES)
		{

			for(e = 0; e < 19; e++){
				grid2[MARGIN_L_SIZE + i + e*offset] = grid[j+e];
		//		fprintf(f,"%d %d %.15f ", i, e, grid2[MARGIN_L_SIZE + i + e*offset]);
			}
			if((*(unsigned int*) (void*) (&grid[j+19])) & OBSTACLE)
				flags[i] = OBSTACLE;
			else if((*(unsigned int*) (void*) (&grid[j+19])) & ACCEL)
				flags[i] = ACCEL;
			else
				flags[i] = 0;
			//fprintf(f,"%2d\n", flags[i]);
			i++;
			j+=N_CELL_ENTRIES;
			partcount+=N_CELL_ENTRIES;
			if(partnum < PARTITIONS_NUM && partcount==SIZE_X*SIZE_Y*(SIZE_Z/PARTITIONS_NUM)*N_CELL_ENTRIES)
			{	//printf("\nstart of partition = %d, include left margin = %d", i, MARGIN_L_SIZE + i);
				//printf("\nDang tao partition...");
				jj = 0;
				while(jj < SIZE_X*SIZE_Y*N_CELL_ENTRIES)
				{
					for(e = 0; e < 19; e++){
						grid2[MARGIN_L_SIZE + i + e*offset] = grid[e];
					//	fprintf(f,"\t%d %d %.15f ", i, e, grid2[MARGIN_L_SIZE + i + e*offset]);
					}
					flags[i] = 0;
					//fprintf(f,"%2d\n", flags[i]);
					i++;
					jj+=N_CELL_ENTRIES;
					j+=N_CELL_ENTRIES;
				}
				partcount=0;
				partnum++;
			}
		}
	}
//	fclose(f);
	printf("\nLBM_convertToSoA done.");
}
void LBM_convertToSoA_partitioned_3dim(LBM_Grid grid, float *grid2, unsigned char* flags)
{
	int i, j;
	int e;
	int offset;
	int x_count, y_count, z_count;
	FILE *f = fopen("soa_data.txt", "w");
	if(grid2!=NULL){
		j = 0;
		i = 0;
		offset = (SIZE_X + LAYERS_NUM)*(SIZE_Y + LAYERS_NUM)*(SIZE_Z + LAYERS_NUM);
		z_count = 0;
		for(int z = 0; z < SIZE_Z; z++)
		{	z_count+=1;
			y_count = 0;
			fprintf(f,"\n");
			for(int y = 0; y < SIZE_Y; y++)
			{	y_count+=1;
				x_count = 0;
				fprintf(f,"\n");
				for(int x = 0; x < SIZE_X; x++)
				{	x_count+=1;
					//fprintf(f,"1 ");
					for(e = 0; e < 19; e++){
						grid2[MARGIN_L_SIZE + i + e*offset] = grid[j+e];
						if(e==0)
							fprintf(f,"%.3f(%d) ", grid2[MARGIN_L_SIZE + i + e*offset], i);
					}
					if((*(unsigned int*) (void*) (&grid[j+19])) & OBSTACLE)
						flags[i] = OBSTACLE;
					else if((*(unsigned int*) (void*) (&grid[j+19])) & ACCEL)
						flags[i] = ACCEL;
					else
						flags[i] = 0;

					if(x_count == CELLS_ONE_LAYER_X && (SIZE_X - x > 2))
					{	i+=1;
						for(e = 0; e < 19; e++){
							grid2[MARGIN_L_SIZE + i + e*offset] = grid[e];
							//if(e==0) fprintf(f," ", grid2[MARGIN_L_SIZE + i + e*offset]);
						}
						flags[i] = 0;
						x_count = 0;
						fprintf(f," ");
					}
					i+=1;
					j+= N_CELL_ENTRIES;
				}
				if((y_count == CELLS_ONE_LAYER_Y) && (SIZE_Y - y > 2))
				{	fprintf(f,"\n");
					for(int xx = 0; xx<SIZE_X+LAYERS_NUM; xx++)
					{
						for(e = 0; e < 19; e++){
							grid2[MARGIN_L_SIZE + i + e*offset] = grid[e];
							//if(e==0) fprintf(f,"%.3f ", grid2[MARGIN_L_SIZE + i + e*offset]);
						}
						flags[i] = 0;
						i+=1;
						fprintf(f," ");
					}
					y_count = 0;
				}
			}
			if((z_count == CELLS_ONE_LAYER_Z) && (SIZE_Z - z > 2))
			{	fprintf(f,"\n");
				for(int yy = 0; yy < SIZE_Y+LAYERS_NUM; yy++)
				{
					for(int xx = 0; xx<SIZE_X+LAYERS_NUM; xx++)
					{
						for(e = 0; e < 19; e++){
							grid2[MARGIN_L_SIZE + i + e*offset] = grid[e];
							///if(e==0) fprintf(f,"%.3f ", grid2[MARGIN_L_SIZE + i + e*offset]);
						}
						flags[i] = 0;
						i+=1;
						fprintf(f," ");
					}
				}
				z_count = 0;
			}
		}
	}
	fclose(f);
	printf("\nLBM_convertToSoA done.");
}
//void LBM_displayFlags_SoA_wo_Struct(int *flags);
int checkSameGrid_SoA_wo_Struct(float* grid1, float* grid2)
{
	int count = 0;
	float sub;
	int i, j=0;
	int offset, e;
	float a, b;
	printf("\n");
	//FILE* f = fopen("soa_checkSameGrid.txt", "w");
	if (grid2 != NULL) {
		j = 0;
		i = 0;
		offset = SIZE_X * SIZE_Y * SIZE_Z;
		while (j < SIZE_X * SIZE_Y * SIZE_Z * N_CELL_ENTRIES) {

			for (e = 0; e < 19; e++) {
				a = grid2[MARGIN_L_SIZE + i + e * offset];
				b = grid1[j + e];
				//fprintf(f, "\n%d %d %d %.15f %.15f", MARGIN_L_SIZE + i + e * offset,j + e ,e, a, b);
				sub = a - b;
				sub = sub > 0 ? sub : sub * (-1);
				//if (sub > 0.0001) {
				//	fprintf(f, "\n\t%d %d %d %.15f %.15f >>>>>>>>>> cell %d", MARGIN_L_SIZE + i + e * offset,j + e ,e, a, b, i);
				//	count++;
				//}
			}
			i++;
			j += N_CELL_ENTRIES;
		}
	}
	//fclose(f);
	return count;
}
//void LBM_showGridStatistics2_SoA_wo_Struct( float* grid1, float* grid2, int *flags);
