
/*############################################################################*/

#ifndef _CONFIG_H_
#define _CONFIG_H_

/*############################################################################*/
#define AoS 0
#define SoA_w_Struct 0
#define PARTITIONED 0

#define TOTAL_SIZE (64*64*64)

#define SIZE1 (64)
#define SIZE2 (64)
#define SIZE3 (64)

#define SIZE_X (SIZE1)//(1*SIZE)
#define SIZE_Y (SIZE2)//(1*SIZE)
#define SIZE_Z (SIZE3)//64//(130)

//add one ghost layer to y- and z-dimension
#define SIZE_XX (SIZE1)//(1*SIZE)
#define SIZE_YY (SIZE2+1)//(1*SIZE + 1)
#define SIZE_ZZ (SIZE3+1)//64//(130)

#define TILED_WIDTH_X 32
#define TILED_WIDTH_Y 16
#define TILED_WIDTH_Z 4
//definition for partition implementation

#define PARTITIONS_NUM 2
#define LAYERS_NUM (PARTITIONS_NUM-1)
#define PARTITION_SIZE ((SIZE_Z/PARTITIONS_NUM)*(SIZE_X*SIZE_Y*QQ)) //partitioned with z-dimension
#define ONE_LAYER_SIZE (SIZE_X*SIZE_Y*QQ)
#define ALL_LAYERS_SIZE (ONE_LAYER_SIZE*LAYERS_NUM)
#define PARTITION_SIZE_Z (SIZE_Z/PARTITIONS_NUM)
#define ONE_LAYER (SIZE_X*SIZE_Y)

#define CELLS_ONE_LAYER_X (SIZE_X/PARTITIONS_NUM)
#define CELLS_ONE_LAYER_Y (SIZE_Y/PARTITIONS_NUM)
#define CELLS_ONE_LAYER_Z (SIZE_Z/PARTITIONS_NUM)


//#define SIZE1 (128) //(128) //(100)
//#define SIZE2 (128) //(128)
//#define SIZE3 (128) //(TOTAL_SIZE/(SIZE1*SIZE2))


#define OMEGA (1.95f)

#define OUTPUT_PRECISION float

#define BOOL int
#define TRUE (-1)
#define FALSE (0)


#define GRID_SIZE (SIZE_Z*SIZE_Y*SIZE_X*N_CELL_ENTRIES)
#define MARGIN 	(2 * SIZE_X * SIZE_Y * N_CELL_ENTRIES)

#define DIST_SIZE (SIZE_X*SIZE_Y*SIZE_Z)

#define MARGIN_L (2*SIZE_X*SIZE_Y)
#define MARGIN_R (2*SIZE_X*SIZE_Y)

#define MARGIN_L_SIZE (2*SIZE_X*SIZE_Y*QQ)
#define MARGIN_R_SIZE (2*SIZE_X*SIZE_Y*QQ)
#define GRID_SIZE_SOA (SIZE_Z*SIZE_Y*SIZE_X*QQ)



/*############################################################################*/

#endif /* _CONFIG_H_ */
