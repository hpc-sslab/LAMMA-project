
/*############################################################################*/

#ifndef _LBM_H_
#define _LBM_H_

/*############################################################################*/

#include "config.h"


/*############################################################################*/

typedef enum {C = 0,
              N, S, E, W, T, B,
              NE, NW, SE, SW,
              NT, NB, ST, SB,
              ET, EB, WT, WB,
              FLAGS, N_CELL_ENTRIES} CELL_ENTRIES;
#define N_DISTR_FUNCS FLAGS

typedef enum {OBSTACLE    = 1 << 0,
              ACCEL       = 1 << 1,
              IN_OUT_FLOW = 1 << 2} CELL_FLAGS;

#include "lbm_1d_array.h"

#define DFL1 (1.0f/3.0f)
#define DFL2 (1.0f/18.0f)
#define DFL3 (1.0f/36.0f)

#define DFL1_OMEGA (0.65f) //(DFL1*OMEGA)
#define DFL2_OMEGA (0.1083333333333333f) //(DFL2*OMEGA)
#define DFL3_OMEGA (0.0541666666666667f)//(DFL3*OMEGA)

#define ONEMINUSOMEGA (-0.95f) //(1.0-OMEGA)


////defined by myself
#define QQ 19

typedef __align__(8) struct {
	float *f[QQ];
	///unsigned int *flags;
	//unsigned int size;
} Distributions;


#define NUM_ELEMENTS_SOA SIZE_X*(SIZE_Y+1)*(SIZE_Z+1)
//#define GRID_SIZE_SOA (NUM_ELEMENTS_SOA * QQ)


#define PRINT_LINE (printf("\n=========================================================\n"))

/*############################################################################*/

void LBM_allocateGrid( float** ptr, float** org_ptr);
void LBM_freeGrid( float** ptr);
void LBM_initializeGrid( LBM_Grid grid);
void LBM_initializeSpecialCellsForLDC( LBM_Grid grid );
void LBM_loadObstacleFile( LBM_Grid grid, const char* filename );
void LBM_initializeSpecialCellsForChannel( LBM_Grid grid );
void LBM_swapGrids( LBM_GridPtr* grid1, LBM_GridPtr* grid2 );
void LBM_performStreamCollide( LBM_Grid srcGrid, LBM_Grid dstGrid );
void LBM_handleInOutFlow( LBM_Grid srcGrid );
void LBM_showGridStatistics( LBM_Grid Grid );
void LBM_storeVelocityField( LBM_Grid grid, const char* filename,
                           const BOOL binary );
void LBM_compareVelocityField( LBM_Grid grid, const char* filename,
                             const BOOL binary );
int checkSameGrid( float* grid1, float* grid2 );
void LBM_showGridStatistics2( LBM_Grid Grid, LBM_Grid d_Grid );
int analyze(int i);
/*############################################################################*/
__global__ void LBM_gpu_performStreamCollide(LBM_Grid srcGrid,LBM_Grid dstGrid );

//functions defined by myself
void LBM_allocateGrid_SoA_w_Struct(Distributions *dist, int size);
void LBM_allocateGrid_SoA_w_Struct_arr(float **fc, float **fn, float **fs, float **fe,float **fw,float **ft,float **fb,float **fne,float **fnw,float **fse,float **fsw,float **fnt,float **fnb,float **fst,float **fsb,float **fet,float **feb,float **fwt,float **fwb, int size);
void LBM_freeGrid_SoA_w_Struct(Distributions *dist);
void LBM_freeGrid_SoA_w_Struct_arr(float *fc, float *fn, float *fs, float *fe,float *fw,float *ft,float *fb,float *fne,float *fnw,float *fse,float *fsw,float *fnt,float *fnb,float *fst,float *fsb,float *fet,float *feb,float *fwt,float *fwb);
void LBM_convertToSoA_w_Struct(LBM_Grid grid, int size, Distributions *dist, unsigned char* flags);
void LBM_convertToSoA_w_Struct_arr(LBM_Grid grid, int size, float *fc, float *fn, float *fs, float *fe,float *fw,float *ft,float *fb,float *fne,float *fnw,float *fse,float *fsw,float *fnt,float *fnb,float *fst,float *fsb,float *fet,float *feb,float *fwt,float *fwb, unsigned char* flags);
void LBM_displayFlags_SoA_w_Struct(unsigned char *flags);
int checkSameGrid_SoA_w_Struct(float* grid1, Distributions *dist);
int checkSameGrid_SoA_w_Struct_arr(float* grid1, float *fc, float *fn, float *fs, float *fe,float *fw,float *ft,float *fb,float *fne,float *fnw,float *fse,float *fsw,float *fnt,float *fnb,float *fst,float *fsb,float *fet,float *feb,float *fwt,float *fwb);
void LBM_showGridStatistics2_SoA_w_Struct( float* grid, Distributions *dist, unsigned char *flags);


void LBM_convertToSoA_wo_Struct(LBM_Grid grid, float *grid2, unsigned char* flags);
void LBM_convertToSoA_partitioned(LBM_Grid grid, float *grid2, unsigned char* flags);
void LBM_convertToSoA_partitioned_3dim(LBM_Grid grid, float *grid2, unsigned char* flags);
void LBM_convertToSoA_wo_Struct_New_Layout(LBM_Grid grid, float *grid2, unsigned char* flags) ;
void LBM_displayFlags_SoA_wo_Struct(int *flags);
int checkSameGrid_SoA_wo_Struct(float* grid1, float* grid2);
void LBM_showGridStatistics2_SoA_wo_Struct( float* grid1, float* grid2, int *flags);

#endif /* _LBM_H_ */
