#include "lbm_cuda.h"
#include <stdio.h>

#define iplus (i+1)
__device__ __constant__ int c_nbArray[19];
__device__ __constant__ int c_lcArray[19];
__device__ __constant__ int c_nb0, c_nb1, c_nb2, c_nb3, c_nb4, c_nb5, c_nb6, c_nb7, c_nb8, c_nb9, c_nb10,
							c_nb11, c_nb12, c_nb13, c_nb14, c_nb15, c_nb16, c_nb17, c_nb18;


/*__device__ __constant__ float cOneOmega;
__device__ __constant__ float cDFL1Omega;
__device__ __constant__ float cDFL2Omega;
__device__ __constant__ float cDFL3Omega;
__device__ __constant__ float3 c_cArray[19];
*/


/*############################################################################*/
__device__ int calc_idx(int x, int y, int z, int e) {
	//return e + 20 * (z * gridDim.x * blockDim.x + y * blockDim.x + x);
	return e + N_CELL_ENTRIES * (z * SIZE_X*SIZE_Y + y * SIZE_X + x);
}

__device__ int idx_SoA(int x, int y, int z, int e)
{
	return (x + ((y + z*SIZE_Y)*SIZE_X) + (SIZE_X*SIZE_Y)*SIZE_Z*e);
}
/*############################################################################*/
__device__ int lc(int e, int i) {
	//return calc_idx(0, 0, 0, e) + i;
	return CALC_INDEX(0, 0, 0, e) + i;
}
/*############################################################################*/
__device__ int nb_c(int i) {
	//return calc_idx(0, 0, 0, C) + i;
	//return calc_idx(0, 0, 0, 0) + i;
	//return CALC_INDEX(0, 0, 0, C) + i;
	return CALC_INDEX(0, 0, 0, 0) + i;
}
/*############################################################################*/
__device__ int nb_n(int i) {
	//return calc_idx(0, +1, 0, N) + i;
	//return calc_idx(0, +1, 0, 1) + i;
	//return CALC_INDEX(0, +1, 0, N) + i;
	return CALC_INDEX(0, +1, 0, 1) + i;
}
/*############################################################################*/
__device__ int nb_s(int i) {
	//return calc_idx(0, -1, 0, S) + i;
	//return calc_idx(0, -1, 0, 2) + i;
	//return CALC_INDEX(0, -1, 0, S) + i;
	return CALC_INDEX(0, -1, 0, 2) + i;
}
/*############################################################################*/
__device__ int nb_e(int i) {
	//return calc_idx(+1, 0, 0, E) + i;
	//return calc_idx(+1, 0, 0, 3) + i;
	//return CALC_INDEX(+1, 0, 0, E) + i;
	return CALC_INDEX(+1, 0, 0, 3) + i;
}
/*############################################################################*/
__device__ int nb_w(int i) {
	//return calc_idx(-1, 0, 0, W) + i;
	//return calc_idx(-1, 0, 0, 4) + i;
	//return CALC_INDEX(-1, 0, 0, W) + i;
	return CALC_INDEX(-1, 0, 0, 4) + i;
}
/*############################################################################*/
__device__ int nb_t(int i) {
	//return calc_idx(0, 0, +1, T) + i;
	//return calc_idx(0, 0, +1, 5) + i;
	//return CALC_INDEX(0, 0, +1, T) + i;
	return CALC_INDEX(0, 0, +1, 5) + i;
}
/*############################################################################*/
__device__ int nb_b(int i) {
	//return calc_idx(0, 0, -1, B) + i;
	//return calc_idx(0, 0, -1, 6) + i;
	//return CALC_INDEX(0, 0, -1, B) + i;
	return CALC_INDEX(0, 0, -1, 6) + i;
}
/*############################################################################*/
__device__ int nb_ne(int i) {
	//return calc_idx(+1, +1, 0, NE) + i;
	//return calc_idx(+1, +1, 0, 7) + i;
	//return CALC_INDEX(+1, +1, 0, NE) + i;
	return CALC_INDEX(+1, +1, 0, 7) + i;
}
/*############################################################################*/
__device__ int nb_nw(int i) {
	//return calc_idx(-1, +1, 0, NW) + i;
	//return calc_idx(-1, +1, 0, 8) + i;
	//return CALC_INDEX(-1, +1, 0, NW) + i;
	return CALC_INDEX(-1, +1, 0, 8) + i;
}
/*############################################################################*/
__device__ int nb_se(int i) {
	//return calc_idx(+1, -1, 0, SE) + i;
	//return calc_idx(+1, -1, 0, 9) + i;
	//return CALC_INDEX(+1, -1, 0, SE) + i;
	return CALC_INDEX(+1, -1, 0, 9) + i;
}
/*############################################################################*/
__device__ int nb_sw(int i) {
	//return calc_idx(-1, -1, 0, SW) + i;
	//return calc_idx(-1, -1, 0, 10) + i;
	//return CALC_INDEX(-1, -1, 0, SW) + i;
	return CALC_INDEX(-1, -1, 0, 10) + i;
}
/*############################################################################*/
__device__ int nb_nt(int i) {
	//return calc_idx(0, +1, +1, NT) + i;
	//return calc_idx(0, +1, +1, 11) + i;
	//return CALC_INDEX(0, +1, +1, NT) + i;
	return CALC_INDEX(0, +1, +1, 11) + i;
}
/*############################################################################*/
__device__ int nb_nb(int i) {
	//return calc_idx(0, +1, -1, NB) + i;
	//return calc_idx(0, +1, -1, 12) + i;
	//return CALC_INDEX(0, +1, -1, NB) + i;
	return CALC_INDEX(0, +1, -1, 12) + i;
}
/*############################################################################*/
__device__ int nb_st(int i) {
	//return calc_idx(0, -1, +1, ST) + i;
	//return calc_idx(0, -1, +1, 13) + i;
	//return CALC_INDEX(0, -1, +1, ST) + i;
	return CALC_INDEX(0, -1, +1, 13) + i;
}
/*############################################################################*/
__device__ int nb_sb(int i) {
	//return calc_idx(0, -1, -1, SB) + i;
	//return calc_idx(0, -1, -1, 14) + i;
	//return CALC_INDEX(0, -1, -1, SB) + i;
	return CALC_INDEX(0, -1, -1, 14) + i;
}
/*############################################################################*/
__device__ int nb_et(int i) {
	//return calc_idx(+1, 0, +1, ET) + i;
	//return calc_idx(+1, 0, +1, 15) + i;
	//return CALC_INDEX(+1, 0, +1, ET) + i;
	return CALC_INDEX(+1, 0, +1, 15) + i;
}
/*############################################################################*/
__device__ int nb_eb(int i) {
	//return calc_idx(+1, 0, -1, EB) + i;
	//return calc_idx(+1, 0, -1, 16) + i;
	//return CALC_INDEX(+1, 0, -1, EB) + i;
	return CALC_INDEX(+1, 0, -1, 16) + i;
}
/*############################################################################*/
__device__ int nb_wt(int i) {
	//return calc_idx(-1, 0, +1, WT) + i;
	//return calc_idx(-1, 0, +1, 17) + i;
	//return CALC_INDEX(-1, 0, +1, WT) + i;
	return CALC_INDEX(-1, 0, +1, 17) + i;
}
/*############################################################################*/
__device__ int nb_wb(int i) {
	//return calc_idx(-1, 0, -1, WB) + i;
	//return calc_idx(-1, 0, -1, 18) + i;
	//return CALC_INDEX(-1, 0, -1, WB) + i;
	return CALC_INDEX(-1, 0, -1, 18) + i;
}
/*############################################################################*/
__device__ int test_flag(int flag_value, int test_flag) {
	return flag_value & test_flag;
}

/*############################################################################*/
__device__ inline float u2_func(float ux, float uy, float uz)
{
	return 1.5*(ux*ux + uy*uy + uz*uz);
}
/*############################################################################*/

//this kernel function runs successfully
__global__ void lbm_kernel2(float *sGrid, float *dGrid) {
	int x = threadIdx.x;
	int y = blockIdx.x;
	int z = blockIdx.y;

	int idx;
	float rho, ux, uy, uz, u2;
	float c, n, s, e, w, t, b, ne, nw, se, sw, nt, nb, st, sb, et,	eb, wt, wb;
	//int d_c, d_n, d_s, d_e, d_w, d_t, d_b, d_ne, d_nw, d_se, d_sw, d_nt, d_nb,	d_st, d_sb, d_et, d_eb, d_wt, d_wb;
	int isobs = 0 , isacc = 0;

	idx = CALC_INDEX(x,y,z,0);// + 400000;
	//idx = calc_idx(x, y, z, 0); //+ 400000;

	idx = max(idx, 0);
	idx = min(idx, 26000000-1);

	int flag = *(unsigned int*) (void*) (&sGrid[lc(FLAGS, idx)]);
	int obs = test_flag(flag, OBSTACLE);
	int acc = test_flag(flag, ACCEL);

	/*c = sGrid[lc(0, idx)]; //sGrid[lc(C, idx)];
	n = sGrid[lc(1, idx)];//sGrid[lc(N, idx)];
	s = sGrid[lc(2, idx)];//sGrid[lc(S, idx)];
	e = sGrid[lc(3, idx)];//sGrid[lc(E, idx)];
	w = sGrid[lc(4, idx)];//sGrid[lc(W, idx)];
	t = sGrid[lc(5, idx)];//sGrid[lc(T, idx)];
	b = sGrid[lc(6, idx)];//sGrid[lc(B, idx)];
	ne = sGrid[lc(7, idx)];//sGrid[lc(NE, idx)];
	nw = sGrid[lc(8, idx)];//sGrid[lc(NW, idx)];
	se = sGrid[lc(9, idx)];//sGrid[lc(SE, idx)];
	sw = sGrid[lc(10, idx)];//sGrid[lc(SW, idx)];
	nt = sGrid[lc(11, idx)];//sGrid[lc(NT, idx)];
	nb = sGrid[lc(12, idx)];//sGrid[lc(NB, idx)];
	st = sGrid[lc(13, idx)];//sGrid[lc(ST, idx)];
	sb = sGrid[lc(14, idx)];//sGrid[lc(SB, idx)];
	et = sGrid[lc(15, idx)];//sGrid[lc(ET, idx)];
	eb = sGrid[lc(16, idx)];//sGrid[lc(EB, idx)];
	wt = sGrid[lc(17, idx)];//sGrid[lc(WT, idx)];
	wb = sGrid[lc(18, idx)];//sGrid[lc(WB, idx)];*/

	c = sGrid[lc(C, idx)];
	n = sGrid[lc(N, idx)];
	s = sGrid[lc(S, idx)];
	e = sGrid[lc(E, idx)];
	w = sGrid[lc(W, idx)];
	t = sGrid[lc(T, idx)];
	b = sGrid[lc(B, idx)];
	ne = sGrid[lc(NE, idx)];
	nw = sGrid[lc(NW, idx)];
	se = sGrid[lc(SE, idx)];
	sw = sGrid[lc(SW, idx)];
	nt = sGrid[lc(NT, idx)];
	nb = sGrid[lc(NB, idx)];
	st = sGrid[lc(ST, idx)];
	sb = sGrid[lc(SB, idx)];
	et = sGrid[lc(ET, idx)];
	eb = sGrid[lc(EB, idx)];
	wt = sGrid[lc(WT, idx)];
	wb = sGrid[lc(WB, idx)];

	isobs = (obs!=0);
	isacc = (acc!=0);

	rho = +c + n + s + e + w + t + b + ne + nw + se + sw + nt + nb + st + sb + et + eb + wt + wb;

	ux = +e - w + ne - nw + se - sw + et + eb - wt - wb;
	uy = +n - s + ne + nw - se - sw + nt + nb - st - sb;
	uz = +t - b + nt - nb + st - sb + et - eb + wt - wb;

	ux = (ux / rho) * (!isacc) + 0.005 * isacc;
	uy = (uy / rho) * (!isacc) + 0.002 * isacc;
	uz = (uz / rho) * (!isacc) + 0.000 * isacc;

	u2 = 1.5 * (ux * ux + uy * uy + uz * uz);

	dGrid[nb_c(idx)] = (float)(c * isobs) + ((1.0 - OMEGA)* c + DFL1 * OMEGA * rho * (1.0 - u2))* (!isobs);

	dGrid[nb_n(idx)] =
			(float) (s * isobs)
					+ ((1.0 - OMEGA)* n + DFL2 * OMEGA * rho * (1.0 + uy * (4.5 * uy + 3.0) - u2))* (!isobs);

	dGrid[nb_s(idx)] =
			(float) (n * isobs)
					+ ((1.0 - OMEGA)* s + DFL2 * OMEGA * rho * (1.0 + uy * (4.5 * uy - 3.0) - u2))* (!isobs);

	dGrid[nb_e(idx)] =
			(float) (w * isobs)
					+ ((1.0 - OMEGA)* e + DFL2 * OMEGA * rho * (1.0 + ux * (4.5 * ux + 3.0) - u2))* (!isobs);

	dGrid[nb_w(idx)] =
			(float) (e * isobs)
					+ ((1.0 - OMEGA)* w + DFL2 * OMEGA * rho * (1.0 + ux * (4.5 * ux - 3.0) - u2))* (!isobs);

	dGrid[nb_t(idx)] =
			(float) (b * isobs)
					+ ((1.0 - OMEGA)* t + DFL2 * OMEGA * rho * (1.0 + uz * (4.5 * uz + 3.0) - u2))* (!isobs);

	dGrid[nb_b(idx)] =
			(float) (t * isobs)
					+ ((1.0 - OMEGA)* b + DFL2 * OMEGA * rho * (1.0 + uz * (4.5 * uz - 3.0) - u2))* (!isobs);

	dGrid[nb_ne(idx)] =
			(float) (sw * isobs)
					+ ((1.0 - OMEGA)* ne + DFL3 * OMEGA * rho * (1.0 + (+ux + uy) * (4.5 * (+ux + uy) + 3.0) - u2))* (!isobs);

	dGrid[nb_nw(idx)] =
			(float) (se * isobs)
					+ ((1.0 - OMEGA)* nw + DFL3 * OMEGA * rho * (1.0 + (-ux + uy) * (4.5 * (-ux + uy) + 3.0) - u2))* (!isobs);

	dGrid[nb_se(idx)] =
			(float) (nw * isobs)
					+ ((1.0 - OMEGA)* se + DFL3 * OMEGA * rho * (1.0 + (+ux - uy) * (4.5 * (+ux - uy) + 3.0) - u2))* (!isobs);

	dGrid[nb_sw(idx)] =
			(float) (ne * isobs)
					+ ((1.0 - OMEGA)* sw + DFL3 * OMEGA * rho * (1.0 + (-ux - uy) * (4.5 * (-ux - uy) + 3.0) - u2))* (!isobs);

	dGrid[nb_nt(idx)] =
			(float) (sb * isobs)
					+ ((1.0 - OMEGA)* nt + DFL3 * OMEGA * rho * (1.0 + (+uy + uz) * (4.5 * (+uy + uz) + 3.0) - u2))* (!isobs);

	dGrid[nb_nb(idx)] =
			(float) (st * isobs)
					+ ((1.0 - OMEGA)* nb + DFL3 * OMEGA * rho * (1.0 + (+uy - uz) * (4.5 * (+uy - uz) + 3.0) - u2))* (!isobs);

	dGrid[nb_st(idx)] =
			(float) (nb * isobs)
					+ ((1.0 - OMEGA)* st + DFL3 * OMEGA * rho * (1.0 + (-uy + uz) * (4.5 * (-uy + uz) + 3.0) - u2))* (!isobs);

	dGrid[nb_sb(idx)] =
			(float) (nt * isobs)
					+ ((1.0 - OMEGA)* sb + DFL3 * OMEGA * rho * (1.0 + (-uy - uz) * (4.5 * (-uy - uz) + 3.0) - u2))* (!isobs);

	dGrid[nb_et(idx)] =
			(float) (wb * isobs)
					+ ((1.0 - OMEGA)* et + DFL3 * OMEGA * rho * (1.0 + (+ux + uz) * (4.5 * (+ux + uz) + 3.0) - u2))* (!isobs);

	dGrid[nb_eb(idx)] =
			(float) (wt * isobs)
					+ ((1.0 - OMEGA)* eb + DFL3 * OMEGA * rho * (1.0 + (+ux - uz) * (4.5 * (+ux - uz) + 3.0) - u2))* (!isobs);

	dGrid[nb_wt(idx)] =
			(float) (eb * isobs)
					+ ((1.0 - OMEGA)* wt + DFL3 * OMEGA * rho * (1.0 + (-ux + uz) * (4.5 * (-ux + uz) + 3.0) - u2))* (!isobs);

	dGrid[nb_wb(idx)] = (float)(et * isobs)+((1.0 - OMEGA)* wb + DFL3 * OMEGA * rho * (1.0 + (-ux - uz) * (4.5 * (-ux - uz) + 3.0) - u2))* (!isobs);
}
/*run well and faster than lbm_kernel2
 * this version uses the constant memory
*/
//__global__ void lbm_kernel_AoS(float *sGrid, float *dGrid, int *statusGrid) {
//
//	//__shared__ unsigned char isobs[100];
//	//__shared__ unsigned char isacc[100];
//
//	//int x = threadIdx.x;
//	//int y = blockIdx.x;
//	//int z = blockIdx.y;
//
//	int idx;
//	//float c, n, s, e, w, t, b, ne, nw, se, sw, nt, nb, st, sb, et,	eb, wt, wb;
//
//	idx = CALC_INDEX(threadIdx.x,blockIdx.x,blockIdx.y,0);
//	//idx = calc_idx(x, y, z, 0);
//
//	idx = max(idx, 0);
//	idx = min(idx, 26000000-1);
//
//	int flag = *(unsigned int*) (void*) (&sGrid[lc(FLAGS, idx)]); //uncoaleasced
//	//int obs = test_flag(flag, OBSTACLE);
//	//int acc = test_flag(flag, ACCEL);
//
//	unsigned char isobs = (test_flag(flag, OBSTACLE)!=0); //int
//	unsigned char isacc = (test_flag(flag, ACCEL)!=0); //int
//	/*c = sGrid[lc(0, idx)]; //sGrid[lc(C, idx)];
//	n = sGrid[lc(1, idx)];//sGrid[lc(N, idx)];
//	s = sGrid[lc(2, idx)];//sGrid[lc(S, idx)];
//	e = sGrid[lc(3, idx)];//sGrid[lc(E, idx)];
//	w = sGrid[lc(4, idx)];//sGrid[lc(W, idx)];
//	t = sGrid[lc(5, idx)];//sGrid[lc(T, idx)];
//	b = sGrid[lc(6, idx)];//sGrid[lc(B, idx)];
//	ne = sGrid[lc(7, idx)];//sGrid[lc(NE, idx)];
//	nw = sGrid[lc(8, idx)];//sGrid[lc(NW, idx)];
//	se = sGrid[lc(9, idx)];//sGrid[lc(SE, idx)];
//	sw = sGrid[lc(10, idx)];//sGrid[lc(SW, idx)];
//	nt = sGrid[lc(11, idx)];//sGrid[lc(NT, idx)];
//	nb = sGrid[lc(12, idx)];//sGrid[lc(NB, idx)];
//	st = sGrid[lc(13, idx)];//sGrid[lc(ST, idx)];
//	sb = sGrid[lc(14, idx)];//sGrid[lc(SB, idx)];
//	et = sGrid[lc(15, idx)];//sGrid[lc(ET, idx)];
//	eb = sGrid[lc(16, idx)];//sGrid[lc(EB, idx)];
//	wt = sGrid[lc(17, idx)];//sGrid[lc(WT, idx)];
//	wb = sGrid[lc(18, idx)];//sGrid[lc(WB, idx)];*/
//
//	float c = sGrid[c_lcArray[C] + idx];
//	float n = sGrid[c_lcArray[N] + idx];
//	float s = sGrid[c_lcArray[S] + idx];
//	float e = sGrid[c_lcArray[E] + idx];
//	float w = sGrid[c_lcArray[W] + idx];
//	float t = sGrid[c_lcArray[T] + idx];
//	float b = sGrid[c_lcArray[B] + idx];
//	float ne = sGrid[c_lcArray[NE]+ idx];
//	float nw = sGrid[c_lcArray[NW]+ idx];
//	float se = sGrid[c_lcArray[SE]+ idx];
//	float sw = sGrid[c_lcArray[SW]+ idx];
//	float nt = sGrid[c_lcArray[NT]+ idx];
//	float nb = sGrid[c_lcArray[NB]+ idx];
//	float st = sGrid[c_lcArray[ST]+ idx];
//	float sb = sGrid[c_lcArray[SB]+ idx];
//	float et = sGrid[c_lcArray[ET]+ idx];
//	float eb = sGrid[c_lcArray[EB]+ idx];
//	float wt = sGrid[c_lcArray[WT]+ idx];
//	float wb = sGrid[c_lcArray[WB]+ idx];
//
//
//
//	float rho = +c + n + s + e + w + t + b + ne + nw + se + sw + nt + nb + st + sb + et + eb + wt + wb;
//
//	float ux = ((+e - w + ne - nw + se - sw + et + eb - wt - wb)/rho)*(!isacc) + 0.005*isacc;
//	float uy = ((+n - s + ne + nw - se - sw + nt + nb - st - sb)/rho)*(!isacc) + 0.002*isacc;
//	float uz = ((+t - b + nt - nb + st - sb + et - eb + wt - wb)/rho)*(!isacc);
//
//
//
//	//ux = (ux / rho) * (!isacc) + 0.005 * isacc;
//	//uy = (uy / rho) * (!isacc) + 0.002 * isacc;
//	//uz = (uz / rho) * (!isacc) + 0.000 * isacc;
//
//	float u2 = 1.5 * (ux * ux + uy * uy + uz * uz);
//
//
//	dGrid[c_nbArray[0] +  idx] = (c * isobs) + ((1.0 - OMEGA)* c + DFL1 * OMEGA * rho * (1.0 - u2))* (!isobs);
//	//statusGrid[c_nbArray[0] +  idx]++;
//
//	dGrid[c_nbArray[1] +  idx] = (s * isobs)
//	+ ((1.0 - OMEGA)* n + DFL2 * OMEGA * rho * (1.0 + uy * (4.5 * uy + 3.0) - u2))* (!isobs);
//
//	//statusGrid[c_nbArray[1] +  idx]++;
//
//	dGrid[c_nbArray[2] +  idx] = (n * isobs)
//									+ ((1.0 - OMEGA)* s + DFL2 * OMEGA * rho * (1.0 + uy * (4.5 * uy - 3.0) - u2))* (!isobs);
//
//	//statusGrid[c_nbArray[2] +  idx]++;
//
//	dGrid[c_nbArray[3] +  idx] = (w * isobs)
//									+ ((1.0 - OMEGA)* e + DFL2 * OMEGA * rho * (1.0 + ux * (4.5 * ux + 3.0) - u2))* (!isobs);
//
//	//statusGrid[c_nbArray[3] +  idx]++;
//
//	dGrid[c_nbArray[4] +  idx] = (e * isobs)
//									+ ((1.0 - OMEGA)* w + DFL2 * OMEGA * rho * (1.0 + ux * (4.5 * ux - 3.0) - u2))* (!isobs);
//
//	//statusGrid[c_nbArray[4] +  idx]++;
//
//	dGrid[c_nbArray[5] +  idx] = (b * isobs)
//									+ ((1.0 - OMEGA)* t + DFL2 * OMEGA * rho * (1.0 + uz * (4.5 * uz + 3.0) - u2))* (!isobs);
//
//	//statusGrid[c_nbArray[5] +  idx]++;
//
//	dGrid[c_nbArray[6] +  idx] = (t * isobs)
//									+ ((1.0 - OMEGA)* b + DFL2 * OMEGA * rho * (1.0 + uz * (4.5 * uz - 3.0) - u2))* (!isobs);
//
//	//statusGrid[c_nbArray[6] +  idx]++;
//
//	dGrid[c_nbArray[7] +  idx] = (sw * isobs)
//									+ ((1.0 - OMEGA)* ne + DFL3 * OMEGA * rho * (1.0 + (+ux + uy) * (4.5 * (+ux + uy) + 3.0) - u2))* (!isobs);
//
//	//statusGrid[c_nbArray[7] +  idx]++;
//
//	dGrid[c_nbArray[8] +  idx] = (se * isobs)
//									+ ((1.0 - OMEGA)* nw + DFL3 * OMEGA * rho * (1.0 + (-ux + uy) * (4.5 * (-ux + uy) + 3.0) - u2))* (!isobs);
//
//	//statusGrid[c_nbArray[8] +  idx]++;
//
//	dGrid[c_nbArray[9] +  idx] = (nw * isobs)
//									+ ((1.0 - OMEGA)* se + DFL3 * OMEGA * rho * (1.0 + (+ux - uy) * (4.5 * (+ux - uy) + 3.0) - u2))* (!isobs);
//
//	//statusGrid[c_nbArray[9] +  idx]++;
//
//	dGrid[c_nbArray[10] +  idx] = (ne * isobs)
//									+ ((1.0 - OMEGA)* sw + DFL3 * OMEGA * rho * (1.0 + (-ux - uy) * (4.5 * (-ux - uy) + 3.0) - u2))* (!isobs);
//
//	//statusGrid[c_nbArray[10] +  idx]++;
//
//	dGrid[c_nbArray[11] +  idx] = (sb * isobs)
//									+ ((1.0 - OMEGA)* nt + DFL3 * OMEGA * rho * (1.0 + (+uy + uz) * (4.5 * (+uy + uz) + 3.0) - u2))* (!isobs);
//
//	//statusGrid[c_nbArray[11] +  idx]++;
//
//	dGrid[c_nbArray[12] +  idx] = (st * isobs)
//									+ ((1.0 - OMEGA)* nb + DFL3 * OMEGA * rho * (1.0 + (+uy - uz) * (4.5 * (+uy - uz) + 3.0) - u2))* (!isobs);
//
//	//statusGrid[c_nbArray[12] +  idx]++;
//
//	dGrid[c_nbArray[13] +  idx] = (nb * isobs)
//									+ ((1.0 - OMEGA)* st + DFL3 * OMEGA * rho * (1.0 + (-uy + uz) * (4.5 * (-uy + uz) + 3.0) - u2))* (!isobs);
//
//	//statusGrid[c_nbArray[13] +  idx]++;
//
//	dGrid[c_nbArray[14] +  idx] = (nt * isobs)
//									+ ((1.0 - OMEGA)* sb + DFL3 * OMEGA * rho * (1.0 + (-uy - uz) * (4.5 * (-uy - uz) + 3.0) - u2))* (!isobs);
//
//	//statusGrid[c_nbArray[14] +  idx]++;
//
//	dGrid[c_nbArray[15] +  idx] = (wb * isobs)
//									+ ((1.0 - OMEGA)* et + DFL3 * OMEGA * rho * (1.0 + (+ux + uz) * (4.5 * (+ux + uz) + 3.0) - u2))* (!isobs);
//
//	//statusGrid[c_nbArray[15] +  idx]++;
//
//	dGrid[c_nbArray[16] +  idx] = (wt * isobs)
//									+ ((1.0 - OMEGA)* eb + DFL3 * OMEGA * rho * (1.0 + (+ux - uz) * (4.5 * (+ux - uz) + 3.0) - u2))* (!isobs);
//
//	//statusGrid[c_nbArray[16] +  idx]++;
//
//	dGrid[c_nbArray[17] +  idx] = (eb * isobs)
//									+ ((1.0 - OMEGA)* wt + DFL3 * OMEGA * rho * (1.0 + (-ux + uz) * (4.5 * (-ux + uz) + 3.0) - u2))* (!isobs);
//
//	//statusGrid[c_nbArray[17] +  idx]++;
//
//	dGrid[c_nbArray[18] +  idx] = (et * isobs)+((1.0 - OMEGA)* wb + DFL3 * OMEGA * rho * (1.0 + (-ux - uz) * (4.5 * (-ux - uz) + 3.0) - u2))* (!isobs);
//	//statusGrid[c_nbArray[18] +  idx]++;
//}
__global__ void lbm_kernel_AoS(float *sGrid, float *dGrid) {

	int x = threadIdx.x;
	int y = blockIdx.x;
	int z = blockIdx.y;

	int idx;
	//float c, n, s, e, w, t, b, ne, nw, se, sw, nt, nb, st, sb, et,	eb, wt, wb;

	idx = CALC_INDEX(x,y,z,0);
	//idx = calc_idx(x, y, z, 0);

	idx = max(idx, 0);
	idx = min(idx, GRID_SIZE - 1); //khong nen dung hang so the nay

	int flag = *(unsigned int*) (void*) (&sGrid[lc(FLAGS, idx)]); //uncoaleasced
	//int obs = test_flag(flag, OBSTACLE);
	//int acc = test_flag(flag, ACCEL);

	unsigned char isobs = (test_flag(flag, OBSTACLE)!=0); //int
	unsigned char isacc = (test_flag(flag, ACCEL)!=0); //int

	float c = sGrid[c_lcArray[C] + idx];
	float n = sGrid[c_lcArray[N] + idx];
	float s = sGrid[c_lcArray[S] + idx];
	float e = sGrid[c_lcArray[E] + idx];
	float w = sGrid[c_lcArray[W] + idx];
	float t = sGrid[c_lcArray[T] + idx];
	float b = sGrid[c_lcArray[B] + idx];
	float ne = sGrid[c_lcArray[NE]+ idx];
	float nw = sGrid[c_lcArray[NW]+ idx];
	float se = sGrid[c_lcArray[SE]+ idx];
	float sw = sGrid[c_lcArray[SW]+ idx];
	float nt = sGrid[c_lcArray[NT]+ idx];
	float nb = sGrid[c_lcArray[NB]+ idx];
	float st = sGrid[c_lcArray[ST]+ idx];
	float sb = sGrid[c_lcArray[SB]+ idx];
	float et = sGrid[c_lcArray[ET]+ idx];
	float eb = sGrid[c_lcArray[EB]+ idx];
	float wt = sGrid[c_lcArray[WT]+ idx];
	float wb = sGrid[c_lcArray[WB]+ idx];



	float rho = +c + n + s + e + w + t + b + ne + nw + se + sw + nt + nb + st + sb + et + eb + wt + wb;

	float ux = ((+e - w + ne - nw + se - sw + et + eb - wt - wb)/rho)*(!isacc) + 0.005*isacc;
	float uy = ((+n - s + ne + nw - se - sw + nt + nb - st - sb)/rho)*(!isacc) + 0.002*isacc;
	float uz = ((+t - b + nt - nb + st - sb + et - eb + wt - wb)/rho)*(!isacc);

	//float ux = (+e - w + ne - nw + se - sw + et + eb - wt - wb);
	//float uy = (+n - s + ne + nw - se - sw + nt + nb - st - sb);
	//float uz = (+t - b + nt - nb + st - sb + et - eb + wt - wb);

	//ux = (ux / rho) * (!isacc) + 0.005 * isacc;
	//uy = (uy / rho) * (!isacc) + 0.002 * isacc;
	//uz = (uz / rho) * (!isacc) + 0.000 * isacc;

	float u2 = 1.5 * (ux * ux + uy * uy + uz * uz);

	float temp;

	temp = (c * isobs) + (ONEMINUSOMEGA* c + DFL1_OMEGA * rho * (1.0 - u2))* (!isobs);
	dGrid[c_nbArray[0] +  idx] = temp;

	temp = (s * isobs) + (ONEMINUSOMEGA* n + DFL2_OMEGA * rho * (1.0 + uy * (4.5 * uy + 3.0) - u2))* (!isobs);
	dGrid[c_nbArray[1] +  idx] = temp;

	temp  = (n * isobs)
								+ (ONEMINUSOMEGA* s + DFL2_OMEGA * rho * (1.0 + uy * (4.5 * uy - 3.0) - u2))* (!isobs);
	dGrid[c_nbArray[2] +  idx] = temp;


	temp = (w * isobs)
								+ (ONEMINUSOMEGA* e + DFL2_OMEGA * rho * (1.0 + ux * (4.5 * ux + 3.0) - u2))* (!isobs);
	dGrid[c_nbArray[3] +  idx] = temp;


	temp =  (e * isobs)
								+ (ONEMINUSOMEGA* w + DFL2_OMEGA * rho * (1.0 + ux * (4.5 * ux - 3.0) - u2))* (!isobs);
	dGrid[c_nbArray[4] +  idx] = temp;


	temp =  (b * isobs)
								+ (ONEMINUSOMEGA* t + DFL2_OMEGA * rho * (1.0 + uz * (4.5 * uz + 3.0) - u2))* (!isobs);
	dGrid[c_nbArray[5] +  idx] = temp;


	temp =  (t * isobs)
								+ (ONEMINUSOMEGA* b + DFL2_OMEGA * rho * (1.0 + uz * (4.5 * uz - 3.0) - u2))* (!isobs);
	dGrid[c_nbArray[6] +  idx] = temp;


	temp =  (sw * isobs)
								+ (ONEMINUSOMEGA* ne + DFL3_OMEGA * rho * (1.0 + (+ux + uy) * (4.5 * (+ux + uy) + 3.0) - u2))* (!isobs);
	dGrid[c_nbArray[7] +  idx] = temp;


	temp =   (se * isobs)
								+ (ONEMINUSOMEGA* nw + DFL3_OMEGA * rho * (1.0 + (-ux + uy) * (4.5 * (-ux + uy) + 3.0) - u2))* (!isobs);
	dGrid[c_nbArray[8] +  idx] = temp;


	temp =  (nw * isobs)
								+ (ONEMINUSOMEGA* se + DFL3_OMEGA * rho * (1.0 + (+ux - uy) * (4.5 * (+ux - uy) + 3.0) - u2))* (!isobs);
	dGrid[c_nbArray[9] +  idx] = temp;


	temp =  (ne * isobs)
								+ (ONEMINUSOMEGA* sw + DFL3_OMEGA * rho * (1.0 + (-ux - uy) * (4.5 * (-ux - uy) + 3.0) - u2))* (!isobs);
	dGrid[c_nbArray[10] +  idx] = temp;


	temp =  (sb * isobs)
								+ (ONEMINUSOMEGA* nt + DFL3_OMEGA * rho * (1.0 + (+uy + uz) * (4.5 * (+uy + uz) + 3.0) - u2))* (!isobs);
	dGrid[c_nbArray[11] +  idx] = temp;


	temp =  (st * isobs)
								+ (ONEMINUSOMEGA* nb + DFL3_OMEGA * rho * (1.0 + (+uy - uz) * (4.5 * (+uy - uz) + 3.0) - u2))* (!isobs);
	dGrid[c_nbArray[12] +  idx] = temp;


	temp =  (nb * isobs)
								+ (ONEMINUSOMEGA* st + DFL3_OMEGA * rho * (1.0 + (-uy + uz) * (4.5 * (-uy + uz) + 3.0) - u2))* (!isobs);
	dGrid[c_nbArray[13] +  idx] = temp;


	temp =  (nt * isobs)
								+ (ONEMINUSOMEGA* sb + DFL3_OMEGA * rho * (1.0 + (-uy - uz) * (4.5 * (-uy - uz) + 3.0) - u2))* (!isobs);
	dGrid[c_nbArray[14] +  idx] = temp;


	temp =  (wb * isobs)
								+ (ONEMINUSOMEGA* et + DFL3_OMEGA * rho * (1.0 + (+ux + uz) * (4.5 * (+ux + uz) + 3.0) - u2))* (!isobs);
	dGrid[c_nbArray[15] +  idx] = temp;


	temp =  (wt * isobs)
								+ (ONEMINUSOMEGA* eb + DFL3_OMEGA * rho * (1.0 + (+ux - uz) * (4.5 * (+ux - uz) + 3.0) - u2))* (!isobs);
	dGrid[c_nbArray[16] +  idx] = temp;


	temp =  (eb * isobs)
								+ (ONEMINUSOMEGA* wt + DFL3_OMEGA * rho * (1.0 + (-ux + uz) * (4.5 * (-ux + uz) + 3.0) - u2))* (!isobs);
	dGrid[c_nbArray[17] +  idx] = temp;


	temp = (et * isobs)
								+(ONEMINUSOMEGA* wb + DFL3_OMEGA * rho * (1.0 + (-ux - uz) * (4.5 * (-ux - uz) + 3.0) - u2))* (!isobs);
	dGrid[c_nbArray[18] +  idx] = temp;

//	temp = (c * isobs) + (ONEMINUSOMEGA* c + DFL1_OMEGA * rho * (1.0 - u2))* (!isobs);
//	dGrid[c_nbArray[0] +  idx] = temp;
//
//	temp = (s * isobs)
//							+ (ONEMINUSOMEGA* n + DFL2_OMEGA * rho * (1.0 + uy * (4.5 * uy + 3.0) - u2))* (!isobs);
//	dGrid[c_nbArray[1] +  idx] = temp;
//
//
//	temp  = (n * isobs)
//							+ (ONEMINUSOMEGA* s + DFL2_OMEGA * rho * (1.0 + uy * (4.5 * uy - 3.0) - u2))* (!isobs);
//	dGrid[c_nbArray[2] +  idx] = temp;
//
//
//	temp = (w * isobs)
//							+ (ONEMINUSOMEGA* e + DFL2_OMEGA * rho * (1.0 + ux * (4.5 * ux + 3.0) - u2))* (!isobs);
//	dGrid[c_nbArray[3] +  idx] = temp;
//
//
//	temp =  (e * isobs)
//							+ (ONEMINUSOMEGA* w + DFL2_OMEGA * rho * (1.0 + ux * (4.5 * ux - 3.0) - u2))* (!isobs);
//	dGrid[c_nbArray[4] +  idx] = temp;
//
//
//	temp =  (b * isobs)
//							+ (ONEMINUSOMEGA* t + DFL2_OMEGA * rho * (1.0 + uz * (4.5 * uz + 3.0) - u2))* (!isobs);
//	dGrid[c_nbArray[5] +  idx] = temp;
//
//
//	temp =  (t * isobs)
//							+ (ONEMINUSOMEGA* b + DFL2_OMEGA * rho * (1.0 + uz * (4.5 * uz - 3.0) - u2))* (!isobs);
//	dGrid[c_nbArray[6] +  idx] = temp;
//
//
//	temp =  (sw * isobs)
//							+ (ONEMINUSOMEGA* ne + DFL3_OMEGA * rho * (1.0 + (+ux + uy) * (4.5 * (+ux + uy) + 3.0) - u2))* (!isobs);
//	dGrid[c_nbArray[7] +  idx] = temp;
//
//
//	temp =   (se * isobs)
//							+ (ONEMINUSOMEGA* nw + DFL3_OMEGA * rho * (1.0 + (-ux + uy) * (4.5 * (-ux + uy) + 3.0) - u2))* (!isobs);
//	dGrid[c_nbArray[8] +  idx] = temp;
//
//
//	temp =  (nw * isobs)
//							+ (ONEMINUSOMEGA* se + DFL3_OMEGA * rho * (1.0 + (+ux - uy) * (4.5 * (+ux - uy) + 3.0) - u2))* (!isobs);
//	dGrid[c_nbArray[9] +  idx] = temp;
//
//
//	temp =  (ne * isobs)
//							+ (ONEMINUSOMEGA* sw + DFL3_OMEGA * rho * (1.0 + (-ux - uy) * (4.5 * (-ux - uy) + 3.0) - u2))* (!isobs);
//	dGrid[c_nbArray[10] +  idx] = temp;
//
//
//	temp =  (sb * isobs)
//							+ (ONEMINUSOMEGA* nt + DFL3_OMEGA * rho * (1.0 + (+uy + uz) * (4.5 * (+uy + uz) + 3.0) - u2))* (!isobs);
//	dGrid[c_nbArray[11] +  idx] = temp;
//
//
//	temp =  (st * isobs)
//							+ (ONEMINUSOMEGA* nb + DFL3_OMEGA * rho * (1.0 + (+uy - uz) * (4.5 * (+uy - uz) + 3.0) - u2))* (!isobs);
//	dGrid[c_nbArray[12] +  idx] = temp;
//
//
//	temp =  (nb * isobs)
//							+ (ONEMINUSOMEGA* st + DFL3_OMEGA * rho * (1.0 + (-uy + uz) * (4.5 * (-uy + uz) + 3.0) - u2))* (!isobs);
//	dGrid[c_nbArray[13] +  idx] = temp;
//
//
//	temp =  (nt * isobs)
//							+ (ONEMINUSOMEGA* sb + DFL3_OMEGA * rho * (1.0 + (-uy - uz) * (4.5 * (-uy - uz) + 3.0) - u2))* (!isobs);
//	dGrid[c_nbArray[14] +  idx] = temp;
//
//
//	temp =  (wb * isobs)
//							+ (ONEMINUSOMEGA* et + DFL3_OMEGA * rho * (1.0 + (+ux + uz) * (4.5 * (+ux + uz) + 3.0) - u2))* (!isobs);
//	dGrid[c_nbArray[15] +  idx] = temp;
//
//
//	temp =  (wt * isobs)
//							+ (ONEMINUSOMEGA* eb + DFL3_OMEGA * rho * (1.0 + (+ux - uz) * (4.5 * (+ux - uz) + 3.0) - u2))* (!isobs);
//	dGrid[c_nbArray[16] +  idx] = temp;
//
//
//	temp =  (eb * isobs)
//							+ (ONEMINUSOMEGA* wt + DFL3_OMEGA * rho * (1.0 + (-ux + uz) * (4.5 * (-ux + uz) + 3.0) - u2))* (!isobs);
//	dGrid[c_nbArray[17] +  idx] = temp;
//
//
//	temp = (et * isobs)+(ONEMINUSOMEGA* wb + DFL3_OMEGA * rho * (1.0 + (-ux - uz) * (4.5 * (-ux - uz) + 3.0) - u2))* (!isobs);
//	dGrid[c_nbArray[18] +  idx] = temp;

}

__global__ void lbm_kernel_AoS_with_branch(float *sGrid, float *dGrid){//, int *statusGrid) {
	int x = threadIdx.x;
	int y = blockIdx.x;
	int z = blockIdx.y;

	int idx;
	//float c, n, s, e, w, t, b, ne, nw, se, sw, nt, nb, st, sb, et,	eb, wt, wb;

	idx = CALC_INDEX(x,y,z,0);
	//idx = calc_idx(x, y, z, 0);

	idx = max(idx, 0);
	idx = min(idx, 26000000-1);

	int flag = *(unsigned int*) (void*) (&sGrid[lc(FLAGS, idx)]);
	int obs = test_flag(flag, OBSTACLE);
	int acc = test_flag(flag, ACCEL);

	float c = sGrid[c_lcArray[C] + idx];
	float n = sGrid[c_lcArray[N] + idx];
	float s = sGrid[c_lcArray[S] + idx];
	float e = sGrid[c_lcArray[E] + idx];
	float w = sGrid[c_lcArray[W] + idx];
	float t = sGrid[c_lcArray[T] + idx];
	float b = sGrid[c_lcArray[B] + idx];
	float ne = sGrid[c_lcArray[NE]+ idx];
	float nw = sGrid[c_lcArray[NW]+ idx];
	float se = sGrid[c_lcArray[SE]+ idx];
	float sw = sGrid[c_lcArray[SW]+ idx];
	float nt = sGrid[c_lcArray[NT]+ idx];
	float nb = sGrid[c_lcArray[NB]+ idx];
	float st = sGrid[c_lcArray[ST]+ idx];
	float sb = sGrid[c_lcArray[SB]+ idx];
	float et = sGrid[c_lcArray[ET]+ idx];
	float eb = sGrid[c_lcArray[EB]+ idx];
	float wt = sGrid[c_lcArray[WT]+ idx];
	float wb = sGrid[c_lcArray[WB]+ idx];

	int isobs = (obs!=0);
	if(isobs)
	{
		dGrid[c_nbArray[0] +  idx] = c;
		//statusGrid[c_nbArray[0] +  idx]++;
		dGrid[c_nbArray[1] +  idx] = s;
		//statusGrid[c_nbArray[1] +  idx]++;
		dGrid[c_nbArray[2] +  idx] = n;
		//statusGrid[c_nbArray[2] +  idx]++;
		dGrid[c_nbArray[3] +  idx] = w;
		//statusGrid[c_nbArray[3] +  idx]++;
		dGrid[c_nbArray[4] +  idx] = e;
		//statusGrid[c_nbArray[4] +  idx]++;
		dGrid[c_nbArray[5] +  idx] = b;
		//statusGrid[c_nbArray[5] +  idx]++;
		dGrid[c_nbArray[6] +  idx] = t;
		//statusGrid[c_nbArray[6] +  idx]++;
		dGrid[c_nbArray[7] +  idx] = sw;
		//statusGrid[c_nbArray[7] +  idx]++;
		dGrid[c_nbArray[8] +  idx] = se;
		//statusGrid[c_nbArray[8] +  idx]++;
		dGrid[c_nbArray[9] +  idx] = nw;
		//statusGrid[c_nbArray[9] +  idx]++;
		dGrid[c_nbArray[10] +  idx] =ne;
		//statusGrid[c_nbArray[10] +  idx]++;
		dGrid[c_nbArray[11] +  idx] =sb;
		//statusGrid[c_nbArray[11] +  idx]++;
		dGrid[c_nbArray[12] +  idx] =st;
		//statusGrid[c_nbArray[12] +  idx]++;
		dGrid[c_nbArray[13] +  idx] =nb;
		//statusGrid[c_nbArray[13] +  idx]++;
		dGrid[c_nbArray[14] +  idx] =nt;
		//statusGrid[c_nbArray[14] +  idx]++;
		dGrid[c_nbArray[15] +  idx] =wb;
		//statusGrid[c_nbArray[15] +  idx]++;
		dGrid[c_nbArray[16] +  idx] =wt;
		//statusGrid[c_nbArray[16] +  idx]++;
		dGrid[c_nbArray[17] +  idx] =eb;
		//statusGrid[c_nbArray[17] +  idx]++;
		dGrid[c_nbArray[18] +  idx] =et;
		//statusGrid[c_nbArray[18] +  idx]++;
	}
	else
	{
		float rho = +c + n + s + e + w + t + b + ne + nw + se + sw + nt + nb + st + sb + et + eb + wt + wb;

		float ux = +e - w + ne - nw + se - sw + et + eb - wt - wb;
		float uy = +n - s + ne + nw - se - sw + nt + nb - st - sb;
		float uz = +t - b + nt - nb + st - sb + et - eb + wt - wb;

		ux = (ux / rho);
		uy = (uy / rho);
		uz = (uz / rho);

		int isacc = (acc!=0);
		if (isacc)
		{
			ux = 0.005;
			uy = 0.002;
			uz = 0.000;
		}

		float u2 = 1.5 * (ux * ux + uy * uy + uz * uz);

		dGrid[c_nbArray[0] +  idx] =((1.0 - OMEGA)* c + DFL1 * OMEGA * rho * (1.0 - u2));
		//statusGrid[c_nbArray[0] +  idx]++;
		dGrid[c_nbArray[1] +  idx] =((1.0 - OMEGA)* n + DFL2 * OMEGA * rho * (1.0 + uy * (4.5 * uy + 3.0) - u2));
		//statusGrid[c_nbArray[1] +  idx]++;
		dGrid[c_nbArray[2] +  idx] =((1.0 - OMEGA)* s + DFL2 * OMEGA * rho * (1.0 + uy * (4.5 * uy - 3.0) - u2));
		//statusGrid[c_nbArray[2] +  idx]++;
		dGrid[c_nbArray[3] +  idx] =((1.0 - OMEGA)* e + DFL2 * OMEGA * rho * (1.0 + ux * (4.5 * ux + 3.0) - u2));
		//statusGrid[c_nbArray[3] +  idx]++;
		dGrid[c_nbArray[4] +  idx] =((1.0 - OMEGA)* w + DFL2 * OMEGA * rho * (1.0 + ux * (4.5 * ux - 3.0) - u2));
		//statusGrid[c_nbArray[4] +  idx]++;
		dGrid[c_nbArray[5] +  idx] =((1.0 - OMEGA)* t + DFL2 * OMEGA * rho * (1.0 + uz * (4.5 * uz + 3.0) - u2));
		//statusGrid[c_nbArray[5] +  idx]++;
		dGrid[c_nbArray[6] +  idx] =((1.0 - OMEGA)* b + DFL2 * OMEGA * rho * (1.0 + uz * (4.5 * uz - 3.0) - u2));
		//statusGrid[c_nbArray[6] +  idx]++;
		dGrid[c_nbArray[7] +  idx] =((1.0 - OMEGA)* ne + DFL3 * OMEGA * rho * (1.0 + (+ux + uy) * (4.5 * (+ux + uy) + 3.0) - u2));
		//statusGrid[c_nbArray[7] +  idx]++;
		dGrid[c_nbArray[8] +  idx] =((1.0 - OMEGA)* nw + DFL3 * OMEGA * rho * (1.0 + (-ux + uy) * (4.5 * (-ux + uy) + 3.0) - u2));
		//statusGrid[c_nbArray[8] +  idx]++;
		dGrid[c_nbArray[9] +  idx] =((1.0 - OMEGA)* se + DFL3 * OMEGA * rho * (1.0 + (+ux - uy) * (4.5 * (+ux - uy) + 3.0) - u2));
		//statusGrid[c_nbArray[9] +  idx]++;
		dGrid[c_nbArray[10] +  idx] =((1.0 - OMEGA)* sw + DFL3 * OMEGA * rho * (1.0 + (-ux - uy) * (4.5 * (-ux - uy) + 3.0) - u2));
		//statusGrid[c_nbArray[10] +  idx]++;
		dGrid[c_nbArray[11] +  idx] =((1.0 - OMEGA)* nt + DFL3 * OMEGA * rho * (1.0 + (+uy + uz) * (4.5 * (+uy + uz) + 3.0) - u2));
		//statusGrid[c_nbArray[11] +  idx]++;
		dGrid[c_nbArray[12] +  idx] =((1.0 - OMEGA)* nb + DFL3 * OMEGA * rho * (1.0 + (+uy - uz) * (4.5 * (+uy - uz) + 3.0) - u2));
		//statusGrid[c_nbArray[12] +  idx]++;
		dGrid[c_nbArray[13] +  idx] =((1.0 - OMEGA)* st + DFL3 * OMEGA * rho * (1.0 + (-uy + uz) * (4.5 * (-uy + uz) + 3.0) - u2));
		//statusGrid[c_nbArray[13] +  idx]++;
		dGrid[c_nbArray[14] +  idx] =((1.0 - OMEGA)* sb + DFL3 * OMEGA * rho * (1.0 + (-uy - uz) * (4.5 * (-uy - uz) + 3.0) - u2));
		//statusGrid[c_nbArray[14] +  idx]++;
		dGrid[c_nbArray[15] +  idx] =((1.0 - OMEGA)* et + DFL3 * OMEGA * rho * (1.0 + (+ux + uz) * (4.5 * (+ux + uz) + 3.0) - u2));
		//statusGrid[c_nbArray[15] +  idx]++;
		dGrid[c_nbArray[16] +  idx] =((1.0 - OMEGA)* eb + DFL3 * OMEGA * rho * (1.0 + (+ux - uz) * (4.5 * (+ux - uz) + 3.0) - u2));
		//statusGrid[c_nbArray[16] +  idx]++;
		dGrid[c_nbArray[17] +  idx] =((1.0 - OMEGA)* wt + DFL3 * OMEGA * rho * (1.0 + (-ux + uz) * (4.5 * (-ux + uz) + 3.0) - u2));
		//statusGrid[c_nbArray[17] +  idx]++;
		dGrid[c_nbArray[18] +  idx] = (et * isobs)+((1.0 - OMEGA)* wb + DFL3 * OMEGA * rho * (1.0 + (-ux - uz) * (4.5 * (-ux - uz) + 3.0) - u2));
		//statusGrid[c_nbArray[18] +  idx]++;
	}
}

__global__ void lbm_kernel3_1(float *sGrid, float *dGrid) {
	int x = threadIdx.x;
	int y = blockIdx.x;
	int z = blockIdx.y;

	int idx;
	float rho, ux, uy, uz, u2;
	float c, n, s, e, w, t, b, ne, nw, se, sw, nt, nb, st, sb, et,	eb, wt, wb;
	int isobs = 0 , isacc = 0;

	idx = CALC_INDEX(x,y,z,0);
	//idx = calc_idx(x, y, z, 0);

	idx = max(idx, 0);
	idx = min(idx, 26000000-1);

	int flag = *(unsigned int*) (void*) (&sGrid[lc(FLAGS, idx)]);
	int obs = test_flag(flag, OBSTACLE);
	int acc = test_flag(flag, ACCEL);

	/*c = sGrid[lc(0, idx)]; //sGrid[lc(C, idx)];
	n = sGrid[lc(1, idx)];//sGrid[lc(N, idx)];
	s = sGrid[lc(2, idx)];//sGrid[lc(S, idx)];
	e = sGrid[lc(3, idx)];//sGrid[lc(E, idx)];
	w = sGrid[lc(4, idx)];//sGrid[lc(W, idx)];
	t = sGrid[lc(5, idx)];//sGrid[lc(T, idx)];
	b = sGrid[lc(6, idx)];//sGrid[lc(B, idx)];
	ne = sGrid[lc(7, idx)];//sGrid[lc(NE, idx)];
	nw = sGrid[lc(8, idx)];//sGrid[lc(NW, idx)];
	se = sGrid[lc(9, idx)];//sGrid[lc(SE, idx)];
	sw = sGrid[lc(10, idx)];//sGrid[lc(SW, idx)];
	nt = sGrid[lc(11, idx)];//sGrid[lc(NT, idx)];
	nb = sGrid[lc(12, idx)];//sGrid[lc(NB, idx)];
	st = sGrid[lc(13, idx)];//sGrid[lc(ST, idx)];
	sb = sGrid[lc(14, idx)];//sGrid[lc(SB, idx)];
	et = sGrid[lc(15, idx)];//sGrid[lc(ET, idx)];
	eb = sGrid[lc(16, idx)];//sGrid[lc(EB, idx)];
	wt = sGrid[lc(17, idx)];//sGrid[lc(WT, idx)];
	wb = sGrid[lc(18, idx)];//sGrid[lc(WB, idx)];*/

	c = sGrid[c_lcArray[C] + idx];
	n = sGrid[c_lcArray[N] + idx];
	s = sGrid[c_lcArray[S] + idx];
	e = sGrid[c_lcArray[E] + idx];
	w = sGrid[c_lcArray[W] + idx];
	t = sGrid[c_lcArray[T] + idx];
	b = sGrid[c_lcArray[B] + idx];
	ne = sGrid[c_lcArray[NE]+ idx];
	nw = sGrid[c_lcArray[NW]+ idx];
	se = sGrid[c_lcArray[SE]+ idx];
	sw = sGrid[c_lcArray[SW]+ idx];
	nt = sGrid[c_lcArray[NT]+ idx];
	nb = sGrid[c_lcArray[NB]+ idx];
	st = sGrid[c_lcArray[ST]+ idx];
	sb = sGrid[c_lcArray[SB]+ idx];
	et = sGrid[c_lcArray[ET]+ idx];
	eb = sGrid[c_lcArray[EB]+ idx];
	wt = sGrid[c_lcArray[WT]+ idx];
	wb = sGrid[c_lcArray[WB]+ idx];

	isobs = (obs!=0);
	isacc = (acc!=0);

	rho = +c + n + s + e + w + t + b + ne + nw + se + sw + nt + nb + st + sb + et + eb + wt + wb;

	ux = +e - w + ne - nw + se - sw + et + eb - wt - wb;
	uy = +n - s + ne + nw - se - sw + nt + nb - st - sb;
	uz = +t - b + nt - nb + st - sb + et - eb + wt - wb;

	ux = (ux / rho) * (!isacc) + 0.005 * isacc;
	uy = (uy / rho) * (!isacc) + 0.002 * isacc;
	uz = (uz / rho) * (!isacc) + 0.000 * isacc;

	u2 = 1.5 * (ux * ux + uy * uy + uz * uz);

	dGrid[c_nbArray[0] +  idx] = (float)(c * isobs) + ((1.0 - OMEGA)* c + DFL1 * OMEGA * rho * (1.0 - u2))* (!isobs);

	dGrid[c_nbArray[1] +  idx] =
			(float) (s * isobs)
					+ ((1.0 - OMEGA)* n + DFL2 * OMEGA * rho * (1.0 + uy * (4.5 * uy + 3.0) - u2))* (!isobs);

	dGrid[c_nbArray[2] +  idx] =
			(float) (n * isobs)
					+ ((1.0 - OMEGA)* s + DFL2 * OMEGA * rho * (1.0 + uy * (4.5 * uy - 3.0) - u2))* (!isobs);

	dGrid[c_nbArray[3] +  idx] =
			(float) (w * isobs)
					+ ((1.0 - OMEGA)* e + DFL2 * OMEGA * rho * (1.0 + ux * (4.5 * ux + 3.0) - u2))* (!isobs);

	dGrid[c_nbArray[4] +  idx] =
			(float) (e * isobs)
					+ ((1.0 - OMEGA)* w + DFL2 * OMEGA * rho * (1.0 + ux * (4.5 * ux - 3.0) - u2))* (!isobs);

	dGrid[c_nbArray[5] +  idx] =
			(float) (b * isobs)
					+ ((1.0 - OMEGA)* t + DFL2 * OMEGA * rho * (1.0 + uz * (4.5 * uz + 3.0) - u2))* (!isobs);

	dGrid[c_nbArray[6] +  idx] =
			(float) (t * isobs)
					+ ((1.0 - OMEGA)* b + DFL2 * OMEGA * rho * (1.0 + uz * (4.5 * uz - 3.0) - u2))* (!isobs);

	dGrid[c_nbArray[7] +  idx] =
			(float) (sw * isobs)
					+ ((1.0 - OMEGA)* ne + DFL3 * OMEGA * rho * (1.0 + (+ux + uy) * (4.5 * (+ux + uy) + 3.0) - u2))* (!isobs);

	dGrid[c_nbArray[8] +  idx] =
			(float) (se * isobs)
					+ ((1.0 - OMEGA)* nw + DFL3 * OMEGA * rho * (1.0 + (-ux + uy) * (4.5 * (-ux + uy) + 3.0) - u2))* (!isobs);

	dGrid[c_nbArray[9] +  idx] =
			(float) (nw * isobs)
					+ ((1.0 - OMEGA)* se + DFL3 * OMEGA * rho * (1.0 + (+ux - uy) * (4.5 * (+ux - uy) + 3.0) - u2))* (!isobs);

	dGrid[c_nbArray[10] +  idx] =
			(float) (ne * isobs)
					+ ((1.0 - OMEGA)* sw + DFL3 * OMEGA * rho * (1.0 + (-ux - uy) * (4.5 * (-ux - uy) + 3.0) - u2))* (!isobs);

	dGrid[c_nbArray[11] +  idx] =
			(float) (sb * isobs)
					+ ((1.0 - OMEGA)* nt + DFL3 * OMEGA * rho * (1.0 + (+uy + uz) * (4.5 * (+uy + uz) + 3.0) - u2))* (!isobs);

	dGrid[c_nbArray[12] +  idx] =
			(float) (st * isobs)
					+ ((1.0 - OMEGA)* nb + DFL3 * OMEGA * rho * (1.0 + (+uy - uz) * (4.5 * (+uy - uz) + 3.0) - u2))* (!isobs);

	dGrid[c_nbArray[13] +  idx] =
			(float) (nb * isobs)
					+ ((1.0 - OMEGA)* st + DFL3 * OMEGA * rho * (1.0 + (-uy + uz) * (4.5 * (-uy + uz) + 3.0) - u2))* (!isobs);

	dGrid[c_nbArray[14] +  idx] =
			(float) (nt * isobs)
					+ ((1.0 - OMEGA)* sb + DFL3 * OMEGA * rho * (1.0 + (-uy - uz) * (4.5 * (-uy - uz) + 3.0) - u2))* (!isobs);

	dGrid[c_nbArray[15] +  idx] =
			(float) (wb * isobs)
					+ ((1.0 - OMEGA)* et + DFL3 * OMEGA * rho * (1.0 + (+ux + uz) * (4.5 * (+ux + uz) + 3.0) - u2))* (!isobs);

	dGrid[c_nbArray[16] +  idx] =
			(float) (wt * isobs)
					+ ((1.0 - OMEGA)* eb + DFL3 * OMEGA * rho * (1.0 + (+ux - uz) * (4.5 * (+ux - uz) + 3.0) - u2))* (!isobs);

	dGrid[c_nbArray[17] +  idx] =
			(float) (eb * isobs)
					+ ((1.0 - OMEGA)* wt + DFL3 * OMEGA * rho * (1.0 + (-ux + uz) * (4.5 * (-ux + uz) + 3.0) - u2))* (!isobs);

	dGrid[c_nbArray[18] +  idx] = (float)(et * isobs)+((1.0 - OMEGA)* wb + DFL3 * OMEGA * rho * (1.0 + (-ux - uz) * (4.5 * (-ux - uz) + 3.0) - u2))* (!isobs);
}
__global__ void child_kernel()
{
	//printf("ThreadId = %d", threadIdx.x);
}
//this version used SoA for source grid (means source grid was transposed to Soa before being passed to kernel
//but, the destination grid is still AoS. So, after get the destination grid out of the kernel we need to transpose destination to SoA before permuting
//this costs very expensive ==> do not use this way

//can reduce number of registers to 24. However, using array to store makes hihg local memory overhead (excessive register spilling)
__global__ void /*__launch_bounds__(128,8)*/ lbm_kernel_SoA_new(float *sGrid, float *dGrid, unsigned char *flags)
{
	__shared__ int offset;

	//__shared__ float one_minus_omega;
	offset = SIZE_XX*SIZE_YY*SIZE_ZZ;
	//one_minus_omega = 1.0 - OMEGA;
	int x = threadIdx.x;
	int y = blockIdx.x + 1;
	int z = blockIdx.y + 1;

	//int ElementsPerBlock = blockDim.x;
	//float c, n, s, e, w, t, b, ne, nw, se, sw, nt, nb, st, sb, et,	eb, wt, wb;

	/*the grid is organized as follows:
	 * c c c c c c . . . | n n n n n n . . . | s s s s s s . . . | e e e e e e . . . | . . .
	 */

	//calculate the index
	//int idx = (z * SIZE_YY + y)*SIZE_XX + x;
	//int idx = ((blockIdx.y +1) * SIZE_YY + (blockIdx.x+1))*SIZE_XX + threadIdx.x;
	int idx = CALC_INDEX_SOA_WO_STRUCT(x,y,z,0);

	//each direction has offset = 130*100*100 values, we need to jump one offset to get new value
	//int offset = SIZE_XX*SIZE_YY*SIZE_ZZ;
	//save index to address for operating

	float arr[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

	for(int i=0;i<QQ; i++)
		arr[i]= sGrid[idx + i*offset];

	//get the value final cell (20) to know the kind of cell: fluid or accel or obstacle
	//int flag = flags[idx];
	//unsigned short type ;
	//type = ((!(flags[idx] ^ 1)) << 8) & 'FF00'; //8 bits dau la isobs
	//type = (!(flags[idx] ^ 2)) &   'FFFF'; //8 bits sau la isacc
	unsigned short type = ((flags[idx] == 1) << 8) | ((flags[idx] == 2) & 0xff);
	//unsigned char isobs = !(flags[idx] ^ 1);
	//unsigned char isacc = !(flags[idx] ^ 2);

	float rho = +arr[0] + arr[1] + arr[2] + arr[3] + arr[4] + arr[5] + arr[6] + arr[7] + arr[8] + arr[9] + arr[10] + arr[11] + arr[12] + arr[13] + arr[14] + arr[15] + arr[16] + arr[17] + arr[18];


	/*float ux = ((+e - w + ne - nw + se - sw + et + eb - wt - wb)/rho)*(!isacc) + 0.005*isacc;
	float uy = ((+n - s + ne + nw - se - sw + nt + nb - st - sb)/rho)*(!isacc) + 0.002*isacc;
	float uz = ((+t - b + nt - nb + st - sb + et - eb + wt - wb)/rho)*(!isacc);
	*/
	float ux = ((+arr[3] - arr[4] + arr[7] - arr[8] + arr[9] - arr[10] + arr[15] + arr[16] - arr[17] - arr[18])/rho)*(!(type & 0xff)) + 0.005*(type & 0xff);
	float uy = ((+arr[1] - arr[2] + arr[7] + arr[8] - arr[9] - arr[10] + arr[11] + arr[12] - arr[13] - arr[14])/rho)*(!(type & 0xff)) + 0.002*(type & 0xff);
	float uz = ((+arr[5] - arr[6] + arr[11] - arr[12] + arr[13] - arr[14] + arr[15] - arr[16] + arr[17] - arr[18])/rho)*(!(type & 0xff));

	//	float ux = +e - w + ne - nw + se - sw + et + eb - wt - wb;
	//	float uy = +n - s + ne + nw - se - sw + nt + nb - st - sb;
	//	float uz = +t - b + nt - nb + st - sb + et - eb + wt - wb;


	float u2 = 1.5 * ux * ux + 1.5* uy * uy + 1.5* uz * uz;
	//float u2= U2(ux,uy,uz); //similar to regular usage
//	float u2 = u2_func(ux,uy,uz); //number of registers increases when using inline function

	//float opt1 = ONEMINUSOMEGA* (!(type >> 8));
	//float opt2 = DFL1_OMEGA * rho * (!(type >> 8));
	//float oneminusu2 = 1.0 - u2;

	dGrid[c_nbArray[ C] + idx] =  		  	(arr[0] * (type >> 8)) + (ONEMINUSOMEGA* arr[0] + DFL1_OMEGA * rho * (1.0 - u2))* (!(type >> 8));

	dGrid[c_nbArray[ N] + idx + offset] =   (arr[2] * (type >> 8))  + (ONEMINUSOMEGA* arr[1] + DFL2_OMEGA * rho * (1.0 + uy * (4.5 * uy + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nbArray[ S] + idx + 2*offset] = (arr[1] * (type >> 8))  + (ONEMINUSOMEGA* arr[2] + DFL2_OMEGA * rho * (1.0 + uy * (4.5 * uy - 3.0) - u2))* (!(type >> 8));

	dGrid[c_nbArray[ E] + idx + 3*offset] = (arr[4] * (type >> 8))	+ (ONEMINUSOMEGA* arr[3] + DFL2_OMEGA * rho * (1.0 + ux * (4.5 * ux + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nbArray[ W] + idx + 4*offset] = (arr[3] * (type >> 8))	+ (ONEMINUSOMEGA* arr[4] + DFL2_OMEGA * rho * (1.0 + ux * (4.5 * ux - 3.0) - u2))* (!(type >> 8));

	dGrid[c_nbArray[ T] + idx + 5*offset] = (arr[6] * (type >> 8))	+ (ONEMINUSOMEGA* arr[5] + DFL2_OMEGA * rho * (1.0 + uz * (4.5 * uz + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nbArray[ B] + idx + 6*offset] = (arr[5] * (type >> 8))	+ (ONEMINUSOMEGA* arr[6] + DFL2_OMEGA * rho * (1.0 + uz * (4.5 * uz - 3.0) - u2))* (!(type >> 8));

	dGrid[c_nbArray[NE] + idx + 7*offset] = (arr[10] * (type >> 8)) + (ONEMINUSOMEGA* arr[7] + DFL3_OMEGA * rho * (1.0 + (+ux + uy) * (4.5 * (+ux + uy) + 3.0) - u2))* (!(type >> 8)); //if(x==1 && y==1 && z==1) printf("\ndst_pos <-> sh_pos %d <-> %d", c_nbArray[NE] + idx + 7*offset, x);

	dGrid[c_nbArray[NW] + idx + 8*offset] = (arr[9] * (type >> 8)) + (ONEMINUSOMEGA* arr[8] + DFL3_OMEGA * rho * (1.0 + (-ux + uy) * (4.5 * (-ux + uy) + 3.0) - u2))* (!(type >> 8)); //if(x==1 && y==1 && z==1) printf("\ndst_pos <-> sh_pos %d <-> %d", c_nbArray[NW] + idx + 8*offset, x+1);

	dGrid[c_nbArray[SE] + idx + 9*offset] = (arr[8] * (type >> 8)) + (ONEMINUSOMEGA* arr[9] + DFL3_OMEGA * rho * (1.0 + (+ux - uy) * (4.5 * (+ux - uy) + 3.0) - u2))* (!(type >> 8)); //if(x==1 && y==1 && z==1) printf("\ndst_pos <-> sh_pos %d <-> %d", c_nbArray[SE] + idx + 9*offset, x);

	dGrid[c_nbArray[SW] + idx + 10*offset] = (arr[7] * (type >> 8))+ (ONEMINUSOMEGA* arr[10] + DFL3_OMEGA * rho * (1.0 + (-ux - uy) * (4.5 * (-ux - uy) + 3.0) - u2))* (!(type >> 8)); //if(x==1 && y==1 && z==1) printf("\ndst_pos <-> sh_pos %d <-> %d", c_nbArray[SW] + idx + 10*offset, x+1);

	dGrid[c_nbArray[NT] + idx + 11*offset] = (arr[14] * (type >> 8))+ (ONEMINUSOMEGA* arr[11] + DFL3_OMEGA * rho * (1.0 + (+uy + uz) * (4.5 * (+uy + uz) + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nbArray[NB] + idx + 12*offset] = (arr[13] * (type >> 8))+ (ONEMINUSOMEGA* arr[12] + DFL3_OMEGA * rho * (1.0 + (+uy - uz) * (4.5 * (+uy - uz) + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nbArray[ST] + idx + 13*offset] = (arr[12] * (type >> 8))+ (ONEMINUSOMEGA* arr[13] + DFL3_OMEGA * rho * (1.0 + (-uy + uz) * (4.5 * (-uy + uz) + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nbArray[SB] + idx + 14*offset] = (arr[11] * (type >> 8))+ (ONEMINUSOMEGA* arr[14] + DFL3_OMEGA * rho * (1.0 + (-uy - uz) * (4.5 * (-uy - uz) + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nbArray[ET] + idx + 15*offset] = (arr[18] * (type >> 8))+ (ONEMINUSOMEGA* arr[15] + DFL3_OMEGA * rho * (1.0 + (+ux + uz) * (4.5 * (+ux + uz) + 3.0) - u2))* (!(type >> 8)); //if(x==1 && y==1 && z==1) printf("\ndst_pos <-> sh_pos %d <-> %d", c_nbArray[ET] + idx + 15*offset, x);

	dGrid[c_nbArray[EB] + idx + 16*offset] = (arr[17] * (type >> 8))+ (ONEMINUSOMEGA* arr[16] + DFL3_OMEGA * rho * (1.0 + (+ux - uz) * (4.5 * (+ux - uz) + 3.0) - u2))* (!(type >> 8)); //if(x==1 && y==1 && z==1) printf("\ndst_pos <-> sh_pos %d <-> %d", c_nbArray[EB] + idx + 16*offset, x);

	dGrid[c_nbArray[WT] + idx + 17*offset] = (arr[16] * (type >> 8))+ (ONEMINUSOMEGA* arr[17] + DFL3_OMEGA * rho * (1.0 + (-ux + uz) * (4.5 * (-ux + uz) + 3.0) - u2))* (!(type >> 8)); //if(x==1 && y==1 && z==1) printf("\ndst_pos <-> sh_pos %d <-> %d", c_nbArray[WT] + idx + 17*offset, x+1);

	dGrid[c_nbArray[WB] + idx + 18*offset] = (arr[15] * (type >> 8))+ (ONEMINUSOMEGA* arr[18] + DFL3_OMEGA * rho * (1.0 + (-ux - uz) * (4.5 * (-ux - uz) + 3.0) - u2))* (!(type >> 8)); //if(x==1 && y==1 && z==1) printf("\ndst_pos <-> sh_pos %d <-> %d", c_nbArray[WB] + idx + 18*offset, x+1);
}
//use shuffle instruction to remove uncoalesced accesses
//can remove uncoalesced accesses completely but the runtime is not improved ????
__global__ void /*__launch_bounds__(128,8)*/ lbm_kernel_SoA_shuffle(float* __restrict__ const sGrid, float *dGrid, unsigned char* __restrict__ const flags)
{
	__shared__ int offset;
	//extern __shared__ float sval[];


	//__shared__ float one_minus_omega;
	offset = SIZE_XX*SIZE_YY*SIZE_ZZ;
	//one_minus_omega = 1.0 - OMEGA;
//	int x = threadIdx.x;
//	int y = blockIdx.x + 1;
//	int z = blockIdx.y + 1;

	//int ElementsPerBlock = blockDim.x;
	//float c, n, s, e, w, t, b, ne, nw, se, sw, nt, nb, st, sb, et,	eb, wt, wb;

	/*the grid is organized as follows:
	 * c c c c c c . . . | n n n n n n . . . | s s s s s s . . . | e e e e e e . . . | . . .
	 */

	//calculate the index
	//int idx = (z * SIZE_YY + y)*SIZE_XX + x;
	//int idx = ((blockIdx.y +1) * SIZE_YY + (blockIdx.x+1))*SIZE_XX + threadIdx.x;
	int idx = CALC_INDEX_SOA_WO_STRUCT(threadIdx.x , (blockIdx.x + 1), (blockIdx.y + 1),0);

	//each direction has offset = 130*100*100 values, we need to jump one offset to get new value
	//int offset = SIZE_XX*SIZE_YY*SIZE_ZZ;
	//save index to address for operating

	int address = idx; //if(y==1 && x==0 && z==1) printf("\nc src(%d)", address);
	float c = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nn src(%d)", address);
	float n = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\ns src(%d)", address);
	float s = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\ne src(%d)", address);
	float e = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nw src(%d)", address);
	float w = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nt src(%d)", address);
	float t = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nb src(%d)", address);
	float b = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nne src(%d)", address);
	float ne = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nnw src(%d)", address);
	float nw = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nse src(%d)", address);
	float se = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nsw src(%d)", address);
	float sw = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nnt src(%d)", address);
	float nt = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nnb src(%d)", address);
	float nb = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nst src(%d)", address);
	float st = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nsb src(%d)", address);
	float sb = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\net src(%d)", address);
	float et = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\neb src(%d)", address);
	float eb = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nwt src(%d)", address);
	float wt = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nwb src(%d)", address);
	float wb = sGrid[address];

	//get the value final cell (20) to know the kind of cell: fluid or accel or obstacle
	//int flag = flags[idx];
	unsigned short type = ((flags[idx] == 1) << 8) | ((flags[idx] == 2) & 0xff);
	//int isobs = (flags[idx] == 1);
	//int isacc = (flags[idx] == 2);

	float rho = +c + n + s + e + w + t + b + ne + nw + se + sw + nt + nb + st + sb + et + eb + wt + wb;

	float ux = ((+e - w + ne - nw + se - sw + et + eb - wt - wb)/rho)*(!(type & 0xff)) + 0.005*(type & 0xff);
	float uy = ((+n - s + ne + nw - se - sw + nt + nb - st - sb)/rho)*(!(type & 0xff)) + 0.002*(type & 0xff);
	float uz = ((+t - b + nt - nb + st - sb + et - eb + wt - wb)/rho)*(!(type & 0xff));

//	float ux = +e - w + ne - nw + se - sw + et + eb - wt - wb;
//	float uy = +n - s + ne + nw - se - sw + nt + nb - st - sb;
//	float uz = +t - b + nt - nb + st - sb + et - eb + wt - wb;


	float u2 = 1.5 * ux * ux + 1.5* uy * uy + 1.5* uz * uz;
	//float u2= U2(ux,uy,uz); //similar to regular usage
//	float u2 = u2_func(ux,uy,uz); //number of registers increases when using inline function

	dGrid[c_nbArray[0] + idx] = (c * (type >> 8))	+ ((1.0 - OMEGA)* c + DFL1 * OMEGA * rho * (1.0 - u2))* (!(type >> 8));

	dGrid[c_nbArray[1] + idx + offset] = (s * (type >> 8))  + ((1.0 - OMEGA)* n + DFL2 * OMEGA * rho * (1.0 + uy * (4.5 * uy + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nbArray[2] + idx + 2*offset] = (n * (type >> 8)) + ((1.0 - OMEGA)* s + DFL2 * OMEGA * rho * (1.0 + uy * (4.5 * uy - 3.0) - u2))* (!(type >> 8));


	dGrid[c_nbArray[5] + idx + 5*offset] = (b * (type >> 8)) + ((1.0 - OMEGA)* t + DFL2 * OMEGA * rho * (1.0 + uz * (4.5 * uz + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nbArray[6] + idx + 6*offset] = (t * (type >> 8)) + ((1.0 - OMEGA)* b + DFL2 * OMEGA * rho * (1.0 + uz * (4.5 * uz - 3.0) - u2))* (!(type >> 8));


	dGrid[c_nbArray[11] + idx + 11*offset] = (sb * (type >> 8))+ ((1.0 - OMEGA)* nt + DFL3 * OMEGA * rho * (1.0 + (+uy + uz) * (4.5 * (+uy + uz) + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nbArray[12] + idx + 12*offset] = (st * (type >> 8))+ ((1.0 - OMEGA)* nb + DFL3 * OMEGA * rho * (1.0 + (+uy - uz) * (4.5 * (+uy - uz) + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nbArray[13] + idx + 13*offset] = (nb * (type >> 8))+ ((1.0 - OMEGA)* st + DFL3 * OMEGA * rho * (1.0 + (-uy + uz) * (4.5 * (-uy + uz) + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nbArray[14] + idx + 14*offset] = (nt * (type >> 8))+ ((1.0 - OMEGA)* sb + DFL3 * OMEGA * rho * (1.0 + (-uy - uz) * (4.5 * (-uy - uz) + 3.0) - u2))* (!(type >> 8));


	float orgval1 = (w * (type >> 8)) + ((1.0 - OMEGA)* e + DFL2 * OMEGA * rho * (1.0 + ux * (4.5 * ux + 3.0) - u2))* (!(type >> 8));
	//float shuffledval1 = __shfl_up(orgval1,1);
	//float swapval = __shfl(orgval1, 31);
	dGrid[c_nbArray[ 3] + idx + 3*offset-1] = __shfl_up(orgval1,1);//shuffledval1;

	float orgval6 = (e * (type >> 8)) + ((1.0 - OMEGA)* w + DFL2 * OMEGA * rho * (1.0 + ux * (4.5 * ux - 3.0) - u2))* (!(type >> 8));
	//float shuffledval6 = __shfl_down(orgval6, 1);
	dGrid[c_nbArray[ 4] + idx + 4*offset+1] = __shfl_down(orgval6, 1);//shuffledval6;

	float orgval2 = (sw * (type >> 8)) + ((1.0 - OMEGA)* ne + DFL3 * OMEGA * rho * (1.0 + (+ux + uy) * (4.5 * (+ux + uy) + 3.0) - u2))* (!(type >> 8));
	//float shuffledval2 = __shfl_up(orgval2,1);
	dGrid[c_nbArray[ 7] + idx + 7*offset-1] = __shfl_up(orgval2,1);//shuffledval2;

	float orgval7 = (se * (type >> 8)) + ((1.0 - OMEGA)* nw + DFL3 * OMEGA * rho * (1.0 + (-ux + uy) * (4.5 * (-ux + uy) + 3.0) - u2))* (!(type >> 8));
	//float shuffledval7 = __shfl_down(orgval7, 1);
	dGrid[c_nbArray[ 8] + idx + 8*offset+1] = __shfl_down(orgval7, 1);//shuffledval7;

	float orgval3 = (nw * (type >> 8)) + ((1.0 - OMEGA)* se + DFL3 * OMEGA * rho * (1.0 + (+ux - uy) * (4.5 * (+ux - uy) + 3.0) - u2))* (!(type >> 8));
	//float shuffledval3 = __shfl_up(orgval3,1);
	dGrid[c_nbArray[ 9] + idx + 9*offset-1] =  __shfl_up(orgval3,1);//shuffledval3;

	float orgval8 = (ne * (type >> 8))+ ((1.0 - OMEGA)* sw + DFL3 * OMEGA * rho * (1.0 + (-ux - uy) * (4.5 * (-ux - uy) + 3.0) - u2))* (!(type >> 8));
	//float shuffledval8 = __shfl_down(orgval8, 1);
	dGrid[c_nbArray[10] + idx + 10*offset+1] = __shfl_down(orgval8, 1);//shuffledval8;

	float orgval4 = (wb * (type >> 8))+ ((1.0 - OMEGA)* et + DFL3 * OMEGA * rho * (1.0 + (+ux + uz) * (4.5 * (+ux + uz) + 3.0) - u2))* (!(type >> 8));
	//float shuffledval4 = __shfl_up(orgval4,1);
	dGrid[c_nbArray[15] + idx + 15*offset-1] = __shfl_up(orgval4,1);//shuffledval4;

	float orgval5 = (wt * (type >> 8))+ ((1.0 - OMEGA)* eb + DFL3 * OMEGA * rho * (1.0 + (+ux - uz) * (4.5 * (+ux - uz) + 3.0) - u2))* (!(type >> 8));
	//float shuffledval5 = __shfl_up(orgval5,1);
	dGrid[c_nbArray[16] + idx + 16*offset-1] = __shfl_up(orgval5,1);//shuffledval5;

	float orgval9 = (eb * (type >> 8))+ ((1.0 - OMEGA)* wt + DFL3 * OMEGA * rho * (1.0 + (-ux + uz) * (4.5 * (-ux + uz) + 3.0) - u2))* (!(type >> 8));
	//float shuffledval9 = __shfl_down(orgval9, 1);
	dGrid[c_nbArray[17] + idx + 17*offset+1] = __shfl_down(orgval9, 1);//shuffledval9;

	float orgval10 = (et * (type >> 8))+ ((1.0 - OMEGA)* wb + DFL3 * OMEGA * rho * (1.0 + (-ux - uz) * (4.5 * (-ux - uz) + 3.0) - u2))* (!(type >> 8));
	//float shuffledval10 = __shfl_down(orgval10, 1);
	dGrid[c_nbArray[18] + idx + 18*offset+1] = __shfl_down(orgval10, 1);//shuffledval10;

}
//move the pointer position to remove unalignment access
__global__ void /*__launch_bounds__(128,8)*/ lbm_kernel_SoA_move_pointer(float *sGrid, float *dGrid, unsigned char *flags)
{
	__shared__ int offset;

	//__shared__ float one_minus_omega;
	offset = SIZE_XX*SIZE_YY*SIZE_ZZ;
	//one_minus_omega = 1.0 - OMEGA;
	int x = threadIdx.x;
	int y = blockIdx.x + 1;
	int z = blockIdx.y + 1;

	//int ElementsPerBlock = blockDim.x;
	//float c, n, s, e, w, t, b, ne, nw, se, sw, nt, nb, st, sb, et,	eb, wt, wb;

	/*the grid is organized as follows:
	 * c c c c c c . . . | n n n n n n . . . | s s s s s s . . . | e e e e e e . . . | . . .
	 */

	//calculate the index
	//int idx = (z * SIZE_YY + y)*SIZE_XX + x;
	//int idx = ((blockIdx.y +1) * SIZE_YY + (blockIdx.x+1))*SIZE_XX + threadIdx.x;
	int idx = CALC_INDEX_SOA_WO_STRUCT(x,y,z,0);

	//each direction has offset = 130*100*100 values, we need to jump one offset to get new value
	//int offset = SIZE_XX*SIZE_YY*SIZE_ZZ;
	//save index to address for operating

	int address = idx; //if(y==1 && x==0 && z==1) printf("\nc src(%d)", address);
	float c = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nn src(%d)", address);
	float n = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\ns src(%d)", address);
	float s = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\ne src(%d)", address);
	float e = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nw src(%d)", address);
	float w = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nt src(%d)", address);
	float t = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nb src(%d)", address);
	float b = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nne src(%d)", address);
	float ne = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nnw src(%d)", address);
	float nw = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nse src(%d)", address);
	float se = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nsw src(%d)", address);
	float sw = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nnt src(%d)", address);
	float nt = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nnb src(%d)", address);
	float nb = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nst src(%d)", address);
	float st = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nsb src(%d)", address);
	float sb = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\net src(%d)", address);
	float et = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\neb src(%d)", address);
	float eb = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nwt src(%d)", address);
	float wt = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nwb src(%d)", address);
	float wb = sGrid[address];

	//get the value final cell (20) to know the kind of cell: fluid or accel or obstacle
	//int flag = flags[idx];
	unsigned short type = ((flags[idx] == 1) << 8) | ((flags[idx] == 2) & 0xff);
	//int isobs = (flags[idx] == 1);
	//int isacc = (flags[idx] == 2);

	float rho = +c + n + s + e + w + t + b + ne + nw + se + sw + nt + nb + st + sb + et + eb + wt + wb;

	float ux = ((+e - w + ne - nw + se - sw + et + eb - wt - wb)/rho)*(!(type & 0xff)) + 0.005*(type & 0xff);
	float uy = ((+n - s + ne + nw - se - sw + nt + nb - st - sb)/rho)*(!(type & 0xff)) + 0.002*(type & 0xff);
	float uz = ((+t - b + nt - nb + st - sb + et - eb + wt - wb)/rho)*(!(type & 0xff));

//	float ux = +e - w + ne - nw + se - sw + et + eb - wt - wb;
//	float uy = +n - s + ne + nw - se - sw + nt + nb - st - sb;
//	float uz = +t - b + nt - nb + st - sb + et - eb + wt - wb;


	float u2 = 1.5 * ux * ux + 1.5* uy * uy + 1.5* uz * uz;
	//float u2= U2(ux,uy,uz); //similar to regular usage
//	float u2 = u2_func(ux,uy,uz); //number of registers increases when using inline function

	dGrid[c_nbArray[0] + idx] = (c * (type >> 8))	+ ((1.0 - OMEGA)* c + DFL1 * OMEGA * rho * (1.0 - u2))* (!(type >> 8));

	dGrid[c_nbArray[1] + idx + offset] = (s * (type >> 8))  + ((1.0 - OMEGA)* n + DFL2 * OMEGA * rho * (1.0 + uy * (4.5 * uy + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nbArray[2] + idx + 2*offset] = (n * (type >> 8)) + ((1.0 - OMEGA)* s + DFL2 * OMEGA * rho * (1.0 + uy * (4.5 * uy - 3.0) - u2))* (!(type >> 8));

	dGrid[c_nbArray[5] + idx + 5*offset] = (b * (type >> 8)) + ((1.0 - OMEGA)* t + DFL2 * OMEGA * rho * (1.0 + uz * (4.5 * uz + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nbArray[6] + idx + 6*offset] = (t * (type >> 8)) + ((1.0 - OMEGA)* b + DFL2 * OMEGA * rho * (1.0 + uz * (4.5 * uz - 3.0) - u2))* (!(type >> 8));

	dGrid[c_nbArray[11] + idx + 11*offset] = (sb * (type >> 8))+ ((1.0 - OMEGA)* nt + DFL3 * OMEGA * rho * (1.0 + (+uy + uz) * (4.5 * (+uy + uz) + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nbArray[12] + idx + 12*offset] = (st * (type >> 8))+ ((1.0 - OMEGA)* nb + DFL3 * OMEGA * rho * (1.0 + (+uy - uz) * (4.5 * (+uy - uz) + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nbArray[13] + idx + 13*offset] = (nb * (type >> 8))+ ((1.0 - OMEGA)* st + DFL3 * OMEGA * rho * (1.0 + (-uy + uz) * (4.5 * (-uy + uz) + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nbArray[14] + idx + 14*offset] = (nt * (type >> 8))+ ((1.0 - OMEGA)* sb + DFL3 * OMEGA * rho * (1.0 + (-uy - uz) * (4.5 * (-uy - uz) + 3.0) - u2))* (!(type >> 8));


	dGrid+=1;
	dGrid[c_nbArray[3] + idx + 3*offset] = (w * (type >> 8)) + ((1.0 - OMEGA)* e + DFL2 * OMEGA * rho * (1.0 + ux * (4.5 * ux + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nbArray[7] + idx + 7*offset] = (sw * (type >> 8)) + ((1.0 - OMEGA)* ne + DFL3 * OMEGA * rho * (1.0 + (+ux + uy) * (4.5 * (+ux + uy) + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nbArray[9] + idx + 9*offset] = (nw * (type >> 8)) + ((1.0 - OMEGA)* se + DFL3 * OMEGA * rho * (1.0 + (+ux - uy) * (4.5 * (+ux - uy) + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nbArray[15] + idx + 15*offset] = (wb * (type >> 8))+ ((1.0 - OMEGA)* et + DFL3 * OMEGA * rho * (1.0 + (+ux + uz) * (4.5 * (+ux + uz) + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nbArray[16] + idx + 16*offset] = (wt * (type >> 8))+ ((1.0 - OMEGA)* eb + DFL3 * OMEGA * rho * (1.0 + (+ux - uz) * (4.5 * (+ux - uz) + 3.0) - u2))* (!(type >> 8));

	dGrid-=2;
	dGrid[c_nbArray[4] + idx + 4*offset] = (e * (type >> 8)) + ((1.0 - OMEGA)* w + DFL2 * OMEGA * rho * (1.0 + ux * (4.5 * ux - 3.0) - u2))* (!(type >> 8));

	dGrid[c_nbArray[8] + idx + 8*offset] = (se * (type >> 8)) + ((1.0 - OMEGA)* nw + DFL3 * OMEGA * rho * (1.0 + (-ux + uy) * (4.5 * (-ux + uy) + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nbArray[10] + idx + 10*offset] = (ne * (type >> 8))+ ((1.0 - OMEGA)* sw + DFL3 * OMEGA * rho * (1.0 + (-ux - uy) * (4.5 * (-ux - uy) + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nbArray[17] + idx + 17*offset] = (eb * (type >> 8))+ ((1.0 - OMEGA)* wt + DFL3 * OMEGA * rho * (1.0 + (-ux + uz) * (4.5 * (-ux + uz) + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nbArray[18] + idx + 18*offset] = (et * (type >> 8))+ ((1.0 - OMEGA)* wb + DFL3 * OMEGA * rho * (1.0 + (-ux - uz) * (4.5 * (-ux - uz) + 3.0) - u2))* (!(type >> 8));
	dGrid+=1;

/*
//	address = c_nbArray[ 0] + idx; if(y==1 && x==0 && z==1) printf("\n => c dst(%d)", address);
	dGrid[c_nbArray[ 0] + idx] =  		  	(c * (type>>8)) + (ONEMINUSOMEGA* c + DFL1_OMEGA * rho * (1.0 - u2))* (!(type>>8));
//	address = c_nbArray[ 1] + idx + offset; if(y==1 && x==0 && z==1) printf("\n => n dst(%d)", address);
	dGrid[c_nbArray[ 1] + idx + offset] =   (s * (type>>8))  + (ONEMINUSOMEGA* n + DFL2_OMEGA * rho * (1.0 + uy * (4.5 * uy + 3.0) - u2))* (!(type>>8));
//	address = c_nbArray[ 2] + idx + 2*offset; if(y==1 && x==0 && z==1) printf("\n => s dst(%d)", address);
	dGrid[c_nbArray[ 2] + idx + 2*offset] = (n * (type>>8))  + (ONEMINUSOMEGA* s + DFL2_OMEGA * rho * (1.0 + uy * (4.5 * uy - 3.0) - u2))* (!(type>>8));

//	address = c_nbArray[ 5] + idx + 5*offset; if(y==1 && x==0 && z==1) printf("\n => t dst(%d)", address);
	dGrid[c_nbArray[ 5] + idx + 5*offset] = (b * (type>>8))	+ (ONEMINUSOMEGA* t + DFL2_OMEGA * rho * (1.0 + uz * (4.5 * uz + 3.0) - u2))* (!(type>>8));
//	address = c_nbArray[ 6] + idx + 6*offset; if(y==1 && x==0 && z==1) printf("\n => b dst(%d)", address);
	dGrid[c_nbArray[ 6] + idx + 6*offset] = (t * (type>>8))	+ (ONEMINUSOMEGA* b + DFL2_OMEGA * rho * (1.0 + uz * (4.5 * uz - 3.0) - u2))* (!(type>>8));

//	address = c_nbArray[11] + idx + 11*offset; if(y==1 && x==0 && z==1) printf("\n => nt dst(%d)", address);
	dGrid[c_nbArray[11] + idx + 11*offset] = (sb * (type>>8))+ (ONEMINUSOMEGA* nt + DFL3_OMEGA * rho * (1.0 + (+uy + uz) * (4.5 * (+uy + uz) + 3.0) - u2))* (!(type>>8));
//	address = c_nbArray[12] + idx + 12*offset; if(y==1 && x==0 && z==1) printf("\n => nb dst(%d)", address);
	dGrid[c_nbArray[12] + idx + 12*offset] = (st * (type>>8))+ (ONEMINUSOMEGA* nb + DFL3_OMEGA * rho * (1.0 + (+uy - uz) * (4.5 * (+uy - uz) + 3.0) - u2))* (!(type>>8));
//	address = c_nbArray[13] + idx + 13*offset; if(y==1 && x==0 && z==1) printf("\n => st dst(%d)", address);
	dGrid[c_nbArray[13] + idx + 13*offset] = (nb * (type>>8))+ (ONEMINUSOMEGA* st + DFL3_OMEGA * rho * (1.0 + (-uy + uz) * (4.5 * (-uy + uz) + 3.0) - u2))* (!(type>>8));
//	address = c_nbArray[14] + idx + 14*offset; if(y==1 && x==0 && z==1) printf("\n => sb dst(%d)", address);
	dGrid[c_nbArray[14] + idx + 14*offset] = (nt * (type>>8))+ (ONEMINUSOMEGA* sb + DFL3_OMEGA * rho * (1.0 + (-uy - uz) * (4.5 * (-uy - uz) + 3.0) - u2))* (!(type>>8));


	dGrid+=1;
//	address = c_nbArray[ 3] + idx + 3*offset; if(y==1 && x==0 && z==1) printf("\n => e dst(%d)", address);
	dGrid[c_nbArray[ 3] + idx + 3*offset-1] = (w * (type>>8))	+ (ONEMINUSOMEGA* e + DFL2_OMEGA * rho * (1.0 + ux * (4.5 * ux + 3.0) - u2))* (!(type>>8));

//	address = c_nbArray[ 7] + idx + 7*offset; if(y==1 && x==0 && z==1) printf("\n => ne dst(%d)", address);
	dGrid[c_nbArray[ 7] + idx + 7*offset-1] = (sw * (type>>8)) + (ONEMINUSOMEGA* ne + DFL3_OMEGA * rho * (1.0 + (+ux + uy) * (4.5 * (+ux + uy) + 3.0) - u2))* (!(type>>8));;

//	address = c_nbArray[ 9] + idx + 9*offset; if(y==1 && x==0 && z==1) printf("\n => se dst(%d)", address);
	dGrid[c_nbArray[ 9] + idx + 9*offset-1] = (nw * (type>>8)) + (ONEMINUSOMEGA* se + DFL3_OMEGA * rho * (1.0 + (+ux - uy) * (4.5 * (+ux - uy) + 3.0) - u2))* (!(type>>8));;

//	address = c_nbArray[15] + idx + 15*offset; if(y==1 && x==0 && z==1) printf("\n => et dst(%d)", address);
	dGrid[c_nbArray[15] + idx + 15*offset-1] = (wb * (type>>8))+ (ONEMINUSOMEGA* et + DFL3_OMEGA * rho * (1.0 + (+ux + uz) * (4.5 * (+ux + uz) + 3.0) - u2))* (!(type>>8));;

//	address = c_nbArray[16] + idx + 16*offset; if(y==1 && x==0 && z==1) printf("\n => eb dst(%d)", address);
	dGrid[c_nbArray[16] + idx + 16*offset-1] = (wt * (type>>8))+ (ONEMINUSOMEGA* eb + DFL3_OMEGA * rho * (1.0 + (+ux - uz) * (4.5 * (+ux - uz) + 3.0) - u2))* (!(type>>8));;

	dGrid-=2;
//	address = c_nbArray[ 4] + idx + 4*offset; if(y==1 && x==0 && z==1) printf("\n => w dst(%d)", address);
	dGrid[c_nbArray[ 4] + idx + 4*offset+1] = (e * (type>>8))	+ (ONEMINUSOMEGA* w + DFL2_OMEGA * rho * (1.0 + ux * (4.5 * ux - 3.0) - u2))* (!(type>>8));;

//	address = c_nbArray[ 8] + idx + 8*offset; if(y==1 && x==0 && z==1) printf("\n => nw dst(%d)", address);
	dGrid[c_nbArray[ 8] + idx + 8*offset+1] = (se * (type>>8)) + (ONEMINUSOMEGA* nw + DFL3_OMEGA * rho * (1.0 + (-ux + uy) * (4.5 * (-ux + uy) + 3.0) - u2))* (!(type>>8));;

//	address = c_nbArray[10] + idx + 10*offset; if(y==1 && x==0 && z==1) printf("\n => sw dst(%d)", address);
	dGrid[c_nbArray[10] + idx + 10*offset+1] = (ne * (type>>8))+ (ONEMINUSOMEGA* sw + DFL3_OMEGA * rho * (1.0 + (-ux - uy) * (4.5 * (-ux - uy) + 3.0) - u2))* (!(type>>8));;

//	address = c_nbArray[17] + idx + 17*offset; if(y==1 && x==0 && z==1) printf("\n => wt dst(%d)", address);
	dGrid[c_nbArray[17] + idx + 17*offset+1] = (eb * (type>>8))+ (ONEMINUSOMEGA* wt + DFL3_OMEGA * rho * (1.0 + (-ux + uz) * (4.5 * (-ux + uz) + 3.0) - u2))* (!(type>>8));;

//	address = c_nbArray[18] + idx + 18*offset; if(y==1 && x==0 && z==1) printf("\n => wb dst(%d)", address);
	dGrid[c_nbArray[18] + idx + 18*offset+1] = (et * (type>>8))+ (ONEMINUSOMEGA* wb + DFL3_OMEGA * rho * (1.0 + (-ux - uz) * (4.5 * (-ux - uz) + 3.0) - u2))* (!(type>>8));;
	dGrid+=1;
	*/
}
//SoA backup
__global__ void /*__launch_bounds__(128,8)*/ lbm_kernel_SoA_bk(float* __restrict__ const sGrid, float *dGrid, unsigned char* __restrict__ const flags)
{
	__shared__ int offset;


	//__shared__ float one_minus_omega;
	offset = SIZE_XX*SIZE_YY*SIZE_ZZ;
	//one_minus_omega = 1.0 - OMEGA;
	int x = threadIdx.x;
	int y = blockIdx.x + 1;
	int z = blockIdx.y + 1;

	//int ElementsPerBlock = blockDim.x;
	//float c, n, s, e, w, t, b, ne, nw, se, sw, nt, nb, st, sb, et,	eb, wt, wb;

	/*the grid is organized as follows:
	 * c c c c c c . . . | n n n n n n . . . | s s s s s s . . . | e e e e e e . . . | . . .
	 */

	//calculate the index
	//int idx = (z * SIZE_YY + y)*SIZE_XX + x;
	//int idx = ((blockIdx.y +1) * SIZE_YY + (blockIdx.x+1))*SIZE_XX + threadIdx.x;
	int idx = CALC_INDEX_SOA_WO_STRUCT(x,y,z,0);

	//each direction has offset = 130*100*100 values, we need to jump one offset to get new value
	//int offset = SIZE_XX*SIZE_YY*SIZE_ZZ;
	//save index to address for operating

	int address = idx; //if(y==1 && x==0 && z==1) printf("\nc src(%d)", address);
	float c = sGrid[address]; //if(y==1 && x==0 && z==1) printf("\nc = %.5f", c);
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nn src(%d)", address);
	float n = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\ns src(%d)", address);
	float s = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\ne src(%d)", address);
	float e = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nw src(%d)", address);
	float w = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nt src(%d)", address);
	float t = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nb src(%d)", address);
	float b = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nne src(%d)", address);
	float ne = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nnw src(%d)", address);
	float nw = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nse src(%d)", address);
	float se = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nsw src(%d)", address);
	float sw = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nnt src(%d)", address);
	float nt = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nnb src(%d)", address);
	float nb = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nst src(%d)", address);
	float st = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nsb src(%d)", address);
	float sb = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\net src(%d)", address);
	float et = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\neb src(%d)", address);
	float eb = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nwt src(%d)", address);
	float wt = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nwb src(%d)", address);
	float wb = sGrid[address];

	//get the value final cell (20) to know the kind of cell: fluid or accel or obstacle
	//int flag = flags[idx];
	unsigned short type = ((flags[idx] == 1) << 8) | ((flags[idx] == 2) & 0xff);
	//int isobs = (flags[idx] == 1);
	//int isacc = (flags[idx] == 2);

	float rho = +c + n + s + e + w + t + b + ne + nw + se + sw + nt + nb + st + sb + et + eb + wt + wb;

	float ux = ((+e - w + ne - nw + se - sw + et + eb - wt - wb)/rho)*(!(type & 0xff)) + 0.005*(type & 0xff);
	float uy = ((+n - s + ne + nw - se - sw + nt + nb - st - sb)/rho)*(!(type & 0xff)) + 0.002*(type & 0xff);
	float uz = ((+t - b + nt - nb + st - sb + et - eb + wt - wb)/rho)*(!(type & 0xff));

//	float ux = +e - w + ne - nw + se - sw + et + eb - wt - wb;
//	float uy = +n - s + ne + nw - se - sw + nt + nb - st - sb;
//	float uz = +t - b + nt - nb + st - sb + et - eb + wt - wb;


	//float u2 = 1.5 * ux * ux + 1.5* uy * uy + 1.5* uz * uz;
	float u2 = 1.5 *( ux * ux +  uy * uy + uz * uz);
	//float u2= U2(ux,uy,uz); //similar to regular usage
//	float u2 = u2_func(ux,uy,uz); //number of registers increases when using inline function

	/*float opt1 = ONEMINUSOMEGA* (!(type >> 8));
	float opt2 = DFL1_OMEGA * rho * (!(type >> 8));
	float oneminusu2 = 1.0 - u2;

	//address = c_nbArray[ 0] + idx; if(y==1 && x==0 && z==1) printf("\n => c thread = %d, dst(%d)", x, address);
	dGrid[c_nbArray[ 0] + idx] =  		  	(c * (type >> 8)) + (opt1 * c  + opt2 * (oneminusu2));

	opt2 = DFL2_OMEGA * rho * (!(type >> 8));

	//address = c_nbArray[ 1] + idx + offset; if(y==1 && x==0 && z==1) printf("\n => n thread = %d, dst(%d)", x, address);
	dGrid[c_nbArray[ 1] + idx + offset] =   (s * (type >> 8))  + (opt1 * n + opt2 * (oneminusu2 + uy * (4.5 * uy + 3.0) ));

	//address = c_nbArray[ 2] + idx + 2*offset; if(y==1 && x==0 && z==1) printf("\n => s thread = %d, dst(%d)", x, address);
	dGrid[c_nbArray[ 2] + idx + 2*offset] = (n * (type >> 8))  + (opt1 * s + opt2 * (oneminusu2 + uy * (4.5 * uy - 3.0) ));

	//address = c_nbArray[ 3] + idx + 3*offset; if(y==1 && x==0 && z==1) printf("\n => e thread = %d, dst(%d)", x, address);
	dGrid[c_nbArray[ 3] + idx + 3*offset] = (w * (type >> 8))	+ (opt1 * e + opt2 * (oneminusu2 + ux * (4.5 * ux + 3.0) ));

	//address = c_nbArray[ 4] + idx + 4*offset; if(y==1 && x==0 && z==1) printf("\n => w thread = %d, dst(%d)", x, address);
	dGrid[c_nbArray[ 4] + idx + 4*offset] = (e * (type >> 8))	+ (opt1 * w + opt2 * (oneminusu2 + ux * (4.5 * ux - 3.0) ));

	//address = c_nbArray[ 5] + idx + 5*offset; if(y==1 && x==0 && z==1) printf("\n => t thread = %d, dst(%d)", x, address);
	dGrid[c_nbArray[ 5] + idx + 5*offset] = (b * (type >> 8))	+ (opt1 * t + opt2 * (oneminusu2 + uz * (4.5 * uz + 3.0) ));

	//address = c_nbArray[ 6] + idx + 6*offset; if(y==1 && x==0 && z==1) printf("\n => b thread = %d, dst(%d)", x, address);
	dGrid[c_nbArray[ 6] + idx + 6*offset] = (t * (type >> 8))	+ (opt1 * b + opt2 * (oneminusu2 + uz * (4.5 * uz - 3.0) ));

	opt2 = DFL3_OMEGA * rho * (!(type >> 8));
	//address = c_nbArray[ 7] + idx + 7*offset; if(y==1 && x==0 && z==1) printf("\n => ne thread = %d, dst(%d)", x, address);
	dGrid[c_nbArray[ 7] + idx + 7*offset] = (sw * (type >> 8)) + (opt1 * ne + opt2 * (oneminusu2 + (+ux + uy) * (4.5 * (+ux + uy) + 3.0) ));

	//address = c_nbArray[ 8] + idx + 8*offset; if(y==1 && x==0 && z==1) printf("\n => nw thread = %d, dst(%d)", x, address);
	dGrid[c_nbArray[ 8] + idx + 8*offset] = (se * (type >> 8)) + (opt1 * nw + opt2 * (oneminusu2 + (-ux + uy) * (4.5 * (-ux + uy) + 3.0) ));

	//address = c_nbArray[ 9] + idx + 9*offset; if(y==1 && x==0 && z==1) printf("\n => se thread = %d, dst(%d)", x, address);
	dGrid[c_nbArray[ 9] + idx + 9*offset] = (nw * (type >> 8)) + (opt1 * se + opt2 * (oneminusu2 + (+ux - uy) * (4.5 * (+ux - uy) + 3.0) ));

	//address = c_nbArray[10] + idx + 10*offset; if(y==1 && x==0 && z==1) printf("\n => sw thread = %d, dst(%d)", x, address);
	dGrid[c_nbArray[10] + idx + 10*offset] = (ne * (type >> 8))+ (opt1 * sw + opt2 * (oneminusu2 + (-ux - uy) * (4.5 * (-ux - uy) + 3.0) ));

	//address = c_nbArray[11] + idx + 11*offset; if(y==1 && x==0 && z==1) printf("\n => nt thread = %d, dst(%d)", x, address);
	dGrid[c_nbArray[11] + idx + 11*offset] = (sb * (type >> 8))+ (opt1 * nt + opt2 * (oneminusu2 + (+uy + uz) * (4.5 * (+uy + uz) + 3.0) ));

	//address = c_nbArray[12] + idx + 12*offset; if(y==1 && x==0 && z==1) printf("\n => nb thread = %d, dst(%d)", x, address);
	dGrid[c_nbArray[12] + idx + 12*offset] = (st * (type >> 8))+ (opt1 * nb + opt2 * (oneminusu2 + (+uy - uz) * (4.5 * (+uy - uz) + 3.0) ));

	//address = c_nbArray[13] + idx + 13*offset; if(y==1 && x==0 && z==1) printf("\n => st thread = %d, dst(%d)", x, address);
	dGrid[c_nbArray[13] + idx + 13*offset] = (nb * (type >> 8))+ (opt1 * st + opt2 * (oneminusu2 + (-uy + uz) * (4.5 * (-uy + uz) + 3.0) ));

	//address = c_nbArray[14] + idx + 14*offset; if(y==1 && x==0 && z==1) printf("\n => sb thread = %d, dst(%d)", x, address);
	dGrid[c_nbArray[14] + idx + 14*offset] = (nt * (type >> 8))+ (opt1 * sb + opt2 * (oneminusu2 + (-uy - uz) * (4.5 * (-uy - uz) + 3.0) ));

	//address = c_nbArray[15] + idx + 15*offset; if(y==1 && x==0 && z==1) printf("\n => et thread = %d, dst(%d)", x, address);
	dGrid[c_nbArray[15] + idx + 15*offset] = (wb * (type >> 8))+ (opt1 * et + opt2 * (oneminusu2 + (+ux + uz) * (4.5 * (+ux + uz) + 3.0) ));

	//address = c_nbArray[16] + idx + 16*offset; if(y==1 && x==0 && z==1) printf("\n => eb thread = %d, dst(%d)", x, address);
	dGrid[c_nbArray[16] + idx + 16*offset] = (wt * (type >> 8))+ (opt1 * eb + opt2 * (oneminusu2 + (+ux - uz) * (4.5 * (+ux - uz) + 3.0) ));

	//address = c_nbArray[17] + idx + 17*offset; if(y==1 && x==0 && z==1) printf("\n => wt thread = %d, dst(%d)", x, address);
	dGrid[c_nbArray[17] + idx + 17*offset] = (eb * (type >> 8))+ (opt1 * wt + opt2 * (oneminusu2 + (-ux + uz) * (4.5 * (-ux + uz) + 3.0) ));

	//address = c_nbArray[18] + idx + 18*offset; if(y==1 && x==0 && z==1) printf("\n => wb thread = %d, dst(%d)", x, address);
	dGrid[c_nbArray[18] + idx + 18*offset] = (et * (type >> 8))+ (opt1 * wb + opt2 * (oneminusu2 + (-ux - uz) * (4.5 * (-ux - uz) + 3.0) ));
	*/

/*
//	address = c_nbArray[ 0] + idx; if(y==1 && x==0 && z==1) printf("\n => c dst(%d)", address);
	dGrid[c_nbArray[ 0] + idx] =  		  	(c * isobs) + (ONEMINUSOMEGA* c + DFL1_OMEGA * rho * (1.0 - u2))* (!isobs);
//	address = c_nbArray[ 1] + idx + offset; if(y==1 && x==0 && z==1) printf("\n => n dst(%d)", address);
	dGrid[c_nbArray[ 1] + idx + offset] =   (s * isobs)  + (ONEMINUSOMEGA* n + DFL2_OMEGA * rho * (1.0 + uy * (4.5 * uy + 3.0) - u2))* (!isobs);
//	address = c_nbArray[ 2] + idx + 2*offset; if(y==1 && x==0 && z==1) printf("\n => s dst(%d)", address);
	dGrid[c_nbArray[ 2] + idx + 2*offset] = (n * isobs)  + (ONEMINUSOMEGA* s + DFL2_OMEGA * rho * (1.0 + uy * (4.5 * uy - 3.0) - u2))* (!isobs);

//	address = c_nbArray[ 5] + idx + 5*offset; if(y==1 && x==0 && z==1) printf("\n => t dst(%d)", address);
	dGrid[c_nbArray[ 5] + idx + 5*offset] = (b * isobs)	+ (ONEMINUSOMEGA* t + DFL2_OMEGA * rho * (1.0 + uz * (4.5 * uz + 3.0) - u2))* (!isobs);
//	address = c_nbArray[ 6] + idx + 6*offset; if(y==1 && x==0 && z==1) printf("\n => b dst(%d)", address);
	dGrid[c_nbArray[ 6] + idx + 6*offset] = (t * isobs)	+ (ONEMINUSOMEGA* b + DFL2_OMEGA * rho * (1.0 + uz * (4.5 * uz - 3.0) - u2))* (!isobs);

//	address = c_nbArray[11] + idx + 11*offset; if(y==1 && x==0 && z==1) printf("\n => nt dst(%d)", address);
	dGrid[c_nbArray[11] + idx + 11*offset] = (sb * isobs)+ (ONEMINUSOMEGA* nt + DFL3_OMEGA * rho * (1.0 + (+uy + uz) * (4.5 * (+uy + uz) + 3.0) - u2))* (!isobs);
//	address = c_nbArray[12] + idx + 12*offset; if(y==1 && x==0 && z==1) printf("\n => nb dst(%d)", address);
	dGrid[c_nbArray[12] + idx + 12*offset] = (st * isobs)+ (ONEMINUSOMEGA* nb + DFL3_OMEGA * rho * (1.0 + (+uy - uz) * (4.5 * (+uy - uz) + 3.0) - u2))* (!isobs);
//	address = c_nbArray[13] + idx + 13*offset; if(y==1 && x==0 && z==1) printf("\n => st dst(%d)", address);
	dGrid[c_nbArray[13] + idx + 13*offset] = (nb * isobs)+ (ONEMINUSOMEGA* st + DFL3_OMEGA * rho * (1.0 + (-uy + uz) * (4.5 * (-uy + uz) + 3.0) - u2))* (!isobs);
//	address = c_nbArray[14] + idx + 14*offset; if(y==1 && x==0 && z==1) printf("\n => sb dst(%d)", address);
	dGrid[c_nbArray[14] + idx + 14*offset] = (nt * isobs)+ (ONEMINUSOMEGA* sb + DFL3_OMEGA * rho * (1.0 + (-uy - uz) * (4.5 * (-uy - uz) + 3.0) - u2))* (!isobs);

//	address = c_nbArray[ 3] + idx + 3*offset; if(y==1 && x==0 && z==1) printf("\n => e dst(%d)", address);
	dGrid[c_nbArray[ 3] + idx + 3*offset] = (w * isobs)	+ (ONEMINUSOMEGA* e + DFL2_OMEGA * rho * (1.0 + ux * (4.5 * ux + 3.0) - u2))* (!isobs);
//	address = c_nbArray[ 4] + idx + 4*offset; if(y==1 && x==0 && z==1) printf("\n => w dst(%d)", address);
	dGrid[c_nbArray[ 4] + idx + 4*offset] = (e * isobs)	+ (ONEMINUSOMEGA* w + DFL2_OMEGA * rho * (1.0 + ux * (4.5 * ux - 3.0) - u2))* (!isobs);

//	address = c_nbArray[ 7] + idx + 7*offset; if(y==1 && x==0 && z==1) printf("\n => ne dst(%d)", address);
	dGrid[c_nbArray[ 7] + idx + 7*offset] = (sw * isobs) + (ONEMINUSOMEGA* ne + DFL3_OMEGA * rho * (1.0 + (+ux + uy) * (4.5 * (+ux + uy) + 3.0) - u2))* (!isobs);
//	address = c_nbArray[ 8] + idx + 8*offset; if(y==1 && x==0 && z==1) printf("\n => nw dst(%d)", address);
	dGrid[c_nbArray[ 8] + idx + 8*offset] = (se * isobs) + (ONEMINUSOMEGA* nw + DFL3_OMEGA * rho * (1.0 + (-ux + uy) * (4.5 * (-ux + uy) + 3.0) - u2))* (!isobs);
//	address = c_nbArray[ 9] + idx + 9*offset; if(y==1 && x==0 && z==1) printf("\n => se dst(%d)", address);
	dGrid[c_nbArray[ 9] + idx + 9*offset] = (nw * isobs) + (ONEMINUSOMEGA* se + DFL3_OMEGA * rho * (1.0 + (+ux - uy) * (4.5 * (+ux - uy) + 3.0) - u2))* (!isobs);
//	address = c_nbArray[10] + idx + 10*offset; if(y==1 && x==0 && z==1) printf("\n => sw dst(%d)", address);
	dGrid[c_nbArray[10] + idx + 10*offset] = (ne * isobs)+ (ONEMINUSOMEGA* sw + DFL3_OMEGA * rho * (1.0 + (-ux - uy) * (4.5 * (-ux - uy) + 3.0) - u2))* (!isobs);

//	address = c_nbArray[15] + idx + 15*offset; if(y==1 && x==0 && z==1) printf("\n => et dst(%d)", address);
	dGrid[c_nbArray[15] + idx + 15*offset] = (wb * isobs)+ (ONEMINUSOMEGA* et + DFL3_OMEGA * rho * (1.0 + (+ux + uz) * (4.5 * (+ux + uz) + 3.0) - u2))* (!isobs);
//	address = c_nbArray[16] + idx + 16*offset; if(y==1 && x==0 && z==1) printf("\n => eb dst(%d)", address);
	dGrid[c_nbArray[16] + idx + 16*offset] = (wt * isobs)+ (ONEMINUSOMEGA* eb + DFL3_OMEGA * rho * (1.0 + (+ux - uz) * (4.5 * (+ux - uz) + 3.0) - u2))* (!isobs);
//	address = c_nbArray[17] + idx + 17*offset; if(y==1 && x==0 && z==1) printf("\n => wt dst(%d)", address);
	dGrid[c_nbArray[17] + idx + 17*offset] = (eb * isobs)+ (ONEMINUSOMEGA* wt + DFL3_OMEGA * rho * (1.0 + (-ux + uz) * (4.5 * (-ux + uz) + 3.0) - u2))* (!isobs);
//	address = c_nbArray[18] + idx + 18*offset; if(y==1 && x==0 && z==1) printf("\n => wb dst(%d)", address);
	dGrid[c_nbArray[18] + idx + 18*offset] = (et * isobs)+ (ONEMINUSOMEGA* wb + DFL3_OMEGA * rho * (1.0 + (-ux - uz) * (4.5 * (-ux - uz) + 3.0) - u2))* (!isobs);
*/

	dGrid[c_nbArray[0] + idx] = (c * (type >> 8))	+ ((1.0 - OMEGA)* c + DFL1 * OMEGA * rho * (1.0 - u2))* (!(type >> 8));

	dGrid[c_nbArray[1] + idx + offset] = (s * (type >> 8))  + ((1.0 - OMEGA)* n + DFL2 * OMEGA * rho * (1.0 + uy * (4.5 * uy + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nbArray[2] + idx + 2*offset] = (n * (type >> 8)) + ((1.0 - OMEGA)* s + DFL2 * OMEGA * rho * (1.0 + uy * (4.5 * uy - 3.0) - u2))* (!(type >> 8));

	dGrid[c_nbArray[3] + idx + 3*offset] = (w * (type >> 8)) + ((1.0 - OMEGA)* e + DFL2 * OMEGA * rho * (1.0 + ux * (4.5 * ux + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nbArray[4] + idx + 4*offset] = (e * (type >> 8)) + ((1.0 - OMEGA)* w + DFL2 * OMEGA * rho * (1.0 + ux * (4.5 * ux - 3.0) - u2))* (!(type >> 8));

	dGrid[c_nbArray[5] + idx + 5*offset] = (b * (type >> 8)) + ((1.0 - OMEGA)* t + DFL2 * OMEGA * rho * (1.0 + uz * (4.5 * uz + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nbArray[6] + idx + 6*offset] = (t * (type >> 8)) + ((1.0 - OMEGA)* b + DFL2 * OMEGA * rho * (1.0 + uz * (4.5 * uz - 3.0) - u2))* (!(type >> 8));

	dGrid[c_nbArray[7] + idx + 7*offset] = (sw * (type >> 8)) + ((1.0 - OMEGA)* ne + DFL3 * OMEGA * rho * (1.0 + (+ux + uy) * (4.5 * (+ux + uy) + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nbArray[8] + idx + 8*offset] = (se * (type >> 8)) + ((1.0 - OMEGA)* nw + DFL3 * OMEGA * rho * (1.0 + (-ux + uy) * (4.5 * (-ux + uy) + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nbArray[9] + idx + 9*offset] = (nw * (type >> 8)) + ((1.0 - OMEGA)* se + DFL3 * OMEGA * rho * (1.0 + (+ux - uy) * (4.5 * (+ux - uy) + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nbArray[10] + idx + 10*offset] = (ne * (type >> 8))+ ((1.0 - OMEGA)* sw + DFL3 * OMEGA * rho * (1.0 + (-ux - uy) * (4.5 * (-ux - uy) + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nbArray[11] + idx + 11*offset] = (sb * (type >> 8))+ ((1.0 - OMEGA)* nt + DFL3 * OMEGA * rho * (1.0 + (+uy + uz) * (4.5 * (+uy + uz) + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nbArray[12] + idx + 12*offset] = (st * (type >> 8))+ ((1.0 - OMEGA)* nb + DFL3 * OMEGA * rho * (1.0 + (+uy - uz) * (4.5 * (+uy - uz) + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nbArray[13] + idx + 13*offset] = (nb * (type >> 8))+ ((1.0 - OMEGA)* st + DFL3 * OMEGA * rho * (1.0 + (-uy + uz) * (4.5 * (-uy + uz) + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nbArray[14] + idx + 14*offset] = (nt * (type >> 8))+ ((1.0 - OMEGA)* sb + DFL3 * OMEGA * rho * (1.0 + (-uy - uz) * (4.5 * (-uy - uz) + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nbArray[15] + idx + 15*offset] = (wb * (type >> 8))+ ((1.0 - OMEGA)* et + DFL3 * OMEGA * rho * (1.0 + (+ux + uz) * (4.5 * (+ux + uz) + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nbArray[16] + idx + 16*offset] = (wt * (type >> 8))+ ((1.0 - OMEGA)* eb + DFL3 * OMEGA * rho * (1.0 + (+ux - uz) * (4.5 * (+ux - uz) + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nbArray[17] + idx + 17*offset] = (eb * (type >> 8))+ ((1.0 - OMEGA)* wt + DFL3 * OMEGA * rho * (1.0 + (-ux + uz) * (4.5 * (-ux + uz) + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nbArray[18] + idx + 18*offset] = (et * (type >> 8))+ ((1.0 - OMEGA)* wb + DFL3 * OMEGA * rho * (1.0 + (-ux - uz) * (4.5 * (-ux - uz) + 3.0) - u2))* (!(type >> 8));

}
inline __device__ int getGlobalIdx_3D_3D(int z)
{
	int blockId = blockIdx.x
			 + blockIdx.y * gridDim.x
			 + gridDim.x * gridDim.y * (blockIdx.z+z) ;
	int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
			  + (threadIdx.z * (blockDim.x * blockDim.y))
			  + (threadIdx.y * blockDim.x)
			  + threadIdx.x;
	return threadId;
}
inline __device__ int getLayoutedPos()
{
	return (blockIdx.x * blockDim.x + threadIdx.x) +
		   (blockIdx.y * blockDim.y + threadIdx.y) * SIZE_X +
		   (blockIdx.z * blockDim.z + threadIdx.z) * SIZE_X * SIZE_Y;
}
__global__ void lbm_kernel_SoA(float* sGrid, float*  dGrid, unsigned char* flags)
{
	__shared__ int offset;

		//__shared__ float one_minus_omega;
	offset = SIZE_XX*SIZE_YY*SIZE_ZZ;
	//one_minus_omega = 1.0 - OMEGA;

	int x = threadIdx.x;
	int y = blockIdx.x;
	int z = blockIdx.y;


	//if(x ==0 && y==0 && z==0)
	//printf("\nblockDim.x = %d, blockDim.y = %d",blockDim.x, blockDim.y);

	//int ElementsPerBlock = blockDim.x;
	//float c, n, s, e, w, t, b, ne, nw, se, sw, nt, nb, st, sb, et,	eb, wt, wb;

	/*the grid is organized as follows:
	 * c c c c c c . . . | n n n n n n . . . | s s s s s s . . . | e e e e e e . . . | . . .
	 */

	//calculate the index
	//int idx = (z * SIZE_YY + y)*SIZE_XX + x;
	//int idx = ((blockIdx.y +1) * SIZE_YY + (blockIdx.x+1))*SIZE_XX + threadIdx.x;
	//int idx = CALC_INDEX_SOA_WO_STRUCT(x , y, z,0);

	int idx = CALC_INDEX_SOA_WO_STRUCT(x, y, z, 0)+ MARGIN_L_SIZE;

	//int idx  = getGlobalIdx_3D_3D() + MARGIN_L;
	/*int idx = (threadIdx.x + blockIdx.x * blockDim.x) +
			(threadIdx.y + blockIdx.y * blockDim.y) * SIZE_X +
			(threadIdx.z + blockIdx.z * blockDim.z) *SIZE_X *SIZE_Y + MARGIN_L;
	*/
	//int next_idx = getLayoutedPos() + MARGIN_L;
	//if(idx-MARGIN_L == 91)
	//printf("%d => %d, (%d, %d, %d) (%d, %d, %d)\n", idx - MARGIN_L, next_idx, threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);

	//each direction has offset = 130*100*100 values, we need to jump one offset to get new value
	//int offset = SIZE_XX*SIZE_YY*SIZE_ZZ;
	//save index to address for operating


	//int address = idx; //if(y==1 && x==0 && z==1) printf("\nc src(%d)", address);
	float c = sGrid[idx] ; //if(y==1 && x==0 && z==1) printf("\nc = %.5f",c);
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\nn src(%d)", address);
	float n = sGrid[idx + offset]; //if(y==1 && x==0 && z==1) printf("\nn = %.5f",n);
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\ns src(%d)", address);
	float s = sGrid[idx + 2*offset] ; //if(y==1 && x==0 && z==1) printf("\ns = %.5f",s);
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\ne src(%d)", address);
	float e = sGrid[idx + 3*offset] ; //if(y==1 && x==0 && z==1) printf("\ne = %.5f",e);
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\nw src(%d)", address);
	float w = sGrid[idx + 4*offset] ; //if(y==1 && x==0 && z==1) printf("\nw = %.5f",w);
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\nt src(%d)", address);
	float t = sGrid[idx + 5*offset] ; //if(y==1 && x==0 && z==1) printf("\nt = %.5f",t);
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\nb src(%d)", address);
	float b = sGrid[idx + 6*offset] ; //if(y==1 && x==0 && z==1) printf("\nb = %.5f",b);
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\nne src(%d)", address);
	float ne = sGrid[idx + 7*offset] ; //if(y==1 && x==0 && z==1) printf("\nne = %.5f",ne);
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\nnw src(%d)", address);
	float nw = sGrid[idx + 8*offset] ;//if(y==1 && x==0 && z==1) printf("\nnw = %.5f",nw);
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\nse src(%d)", address);
	float se = sGrid[idx + 9*offset] ;//if(y==1 && x==0 && z==1) printf("\nse = %.5f",se);
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\nsw src(%d)", address);
	float sw = sGrid[idx + 10*offset] ;//if(y==1 && x==0 && z==1) printf("\nsw = %.5f",sw);
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\nnt src(%d)", address);
	float nt = sGrid[idx + 11*offset] ;//if(y==1 && x==0 && z==1) printf("\nnt = %.5f",nt);
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\nnb src(%d)", address);
	float nb = sGrid[idx + 12*offset] ;//if(y==1 && x==0 && z==1) printf("\nnb = %.5f",nb);
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\nst src(%d)", address);
	float st = sGrid[idx + 13*offset] ;//if(y==1 && x==0 && z==1) printf("\nst = %.5f",st);
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\nsb src(%d)", address);
	float sb = sGrid[idx + 14*offset] ;//if(y==1 && x==0 && z==1) printf("\nsb = %.5f",sb);
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\net src(%d)", address);
	float et = sGrid[idx + 15*offset] ;//if(y==1 && x==0 && z==1) printf("\net = %.5f",et);
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\neb src(%d)", address);
	float eb = sGrid[idx + 16*offset] ;//if(y==1 && x==0 && z==1) printf("\neb = %.5f",eb);
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\nwt src(%d)", address);
	float wt = sGrid[idx + 17*offset] ;//if(y==1 && x==0 && z==1) printf("\nwt = %.5f",wt);
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\nwb src(%d)", address);
	float wb = sGrid[idx + 18*offset] ;//if(y==1 && x==0 && z==1) printf("\nwb = %.5f",wb);


	//get the value final cell (20) to know the kind of cell: fluid or accel or obstacle
	//int flag = flags[idx];
	unsigned short type = ((flags[idx-MARGIN_L_SIZE] == 227) << 8) | ((flags[idx-MARGIN_L_SIZE] == 228) & 0xff);

	float rho = +c + n + s + e + w + t + b + ne + nw + se + sw + nt + nb + st + sb + et + eb + wt + wb;
	float ux = ((+e - w + ne - nw + se - sw + et + eb - wt - wb)/rho)*(!(type & 0xff)) + 0.005*(type & 0xff);
	float uy = ((+n - s + ne + nw - se - sw + nt + nb - st - sb)/rho)*(!(type & 0xff)) + 0.002*(type & 0xff);
	float uz = ((+t - b + nt - nb + st - sb + et - eb + wt - wb)/rho)*(!(type & 0xff));

	float u2 = 1.5 *( ux * ux +  uy * uy + uz * uz);

	/*dGrid[idx] = (c * (type >> 8))	+ ((1.0 - OMEGA)* c + DFL1 * OMEGA * rho * (1.0 - u2))* (!(type >> 8));

	dGrid[c_nb1 + idx + offset] = (s * (type >> 8))  + ((1.0 - OMEGA)* n + DFL2 * OMEGA * rho * (1.0 + uy * (4.5 * uy + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nb2 + idx + 2*offset] = (n * (type >> 8)) + ((1.0 - OMEGA)* s + DFL2 * OMEGA * rho * (1.0 + uy * (4.5 * uy - 3.0) - u2))* (!(type >> 8));

	dGrid[c_nb3 + idx + 3*offset] = (w * (type >> 8)) + ((1.0 - OMEGA)* e + DFL2 * OMEGA * rho * (1.0 + ux * (4.5 * ux + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nb4 + idx + 4*offset] = (e * (type >> 8)) + ((1.0 - OMEGA)* w + DFL2 * OMEGA * rho * (1.0 + ux * (4.5 * ux - 3.0) - u2))* (!(type >> 8));

	dGrid[c_nb5 + idx + 5*offset] = (b * (type >> 8)) + ((1.0 - OMEGA)* t + DFL2 * OMEGA * rho * (1.0 + uz * (4.5 * uz + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nb6 + idx + 6*offset] = (t * (type >> 8)) + ((1.0 - OMEGA)* b + DFL2 * OMEGA * rho * (1.0 + uz * (4.5 * uz - 3.0) - u2))* (!(type >> 8));

	dGrid[c_nb7 + idx + 7*offset] = (sw * (type >> 8)) + ((1.0 - OMEGA)* ne + DFL3 * OMEGA * rho * (1.0 + (+ux + uy) * (4.5 * (+ux + uy) + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nb8 + idx + 8*offset] = (se * (type >> 8)) + ((1.0 - OMEGA)* nw + DFL3 * OMEGA * rho * (1.0 + (-ux + uy) * (4.5 * (-ux + uy) + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nb9 + idx + 9*offset] = (nw * (type >> 8)) + ((1.0 - OMEGA)* se + DFL3 * OMEGA * rho * (1.0 + (+ux - uy) * (4.5 * (+ux - uy) + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nb10 + idx + 10*offset] = (ne * (type >> 8))+ ((1.0 - OMEGA)* sw + DFL3 * OMEGA * rho * (1.0 + (-ux - uy) * (4.5 * (-ux - uy) + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nb11 + idx + 11*offset] = (sb * (type >> 8))+ ((1.0 - OMEGA)* nt + DFL3 * OMEGA * rho * (1.0 + (+uy + uz) * (4.5 * (+uy + uz) + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nb12 + idx + 12*offset] = (st * (type >> 8))+ ((1.0 - OMEGA)* nb + DFL3 * OMEGA * rho * (1.0 + (+uy - uz) * (4.5 * (+uy - uz) + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nb13 + idx + 13*offset] = (nb * (type >> 8))+ ((1.0 - OMEGA)* st + DFL3 * OMEGA * rho * (1.0 + (-uy + uz) * (4.5 * (-uy + uz) + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nb14 + idx + 14*offset] = (nt * (type >> 8))+ ((1.0 - OMEGA)* sb + DFL3 * OMEGA * rho * (1.0 + (-uy - uz) * (4.5 * (-uy - uz) + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nb15 + idx + 15*offset] = (wb * (type >> 8))+ ((1.0 - OMEGA)* et + DFL3 * OMEGA * rho * (1.0 + (+ux + uz) * (4.5 * (+ux + uz) + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nb16 + idx + 16*offset] = (wt * (type >> 8))+ ((1.0 - OMEGA)* eb + DFL3 * OMEGA * rho * (1.0 + (+ux - uz) * (4.5 * (+ux - uz) + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nb17 + idx + 17*offset] = (eb * (type >> 8))+ ((1.0 - OMEGA)* wt + DFL3 * OMEGA * rho * (1.0 + (-ux + uz) * (4.5 * (-ux + uz) + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nb18 + idx + 18*offset] = (et * (type >> 8))+ ((1.0 - OMEGA)* wb + DFL3 * OMEGA * rho * (1.0 + (-ux - uz) * (4.5 * (-ux - uz) + 3.0) - u2))* (!(type >> 8));
*/

	dGrid[idx] = (c * (type >> 8))	+ ((1.0 - OMEGA)* c + DFL1 * OMEGA * rho * (1.0 - u2))* (!(type >> 8));
	//printf("\n0. (%d, %d, %d) %d %.13f", x, y, z, idx, dGrid[idx]);
	c = (type >> 8); //resue variable c
	//address = offset;


	dGrid[c_nb1 + idx + offset] = (s * c)  + ((1.0 - OMEGA)* n + DFL2 * OMEGA * rho * (1.0 + uy * (4.5 * uy + 3.0) - u2))* (!c);
	//printf("\n1. (%d, %d, %d) %d %.13f", x, y, z, c_nb1 + idx + offset, dGrid[c_nb1 + idx + offset]);

	dGrid[c_nb2 + idx + 2*offset] = (n * c) + ((1.0 - OMEGA)* s + DFL2 * OMEGA * rho * (1.0 + uy * (4.5 * uy - 3.0) - u2))* (!c);
	//printf("\n2. (%d, %d, %d) %d %.13f", x, y, z, c_nb2 + idx + 2*offset, dGrid[c_nb2 + idx + 2*offset]);

	n = (1.0 - OMEGA); //resue variable n
	s = DFL2 * OMEGA * rho; //resue variable s

	dGrid[c_nb3 + idx + 3*offset] = (w * c) + (n* e + s * (1.0 + ux * (4.5 * ux + 3.0) - u2))* (!c);
	//printf("\n3. (%d, %d, %d) %d %.13f", x, y, z, c_nb3 + idx + 3*offset, dGrid[c_nb3 + idx + 3*offset]);

	dGrid[c_nb4 + idx + 4*offset] = (e * c) + (n* w + s * (1.0 + ux * (4.5 * ux - 3.0) - u2))* (!c);
	//printf("\n4. (%d, %d, %d) %d %.13f", x, y, z, c_nb4 + idx + 4*offset, dGrid[c_nb4 + idx + 4*offset]);

	dGrid[c_nb5 + idx + 5*offset] = (b * c) + (n* t + s * (1.0 + uz * (4.5 * uz + 3.0) - u2))* (!c);
	//printf("\n5. (%d, %d, %d) %d %.13f", x, y, z, c_nb5 + idx + 5*offset, dGrid[c_nb5 + idx + 5*offset]);

	dGrid[c_nb6 + idx + 6*offset] = (t * c) + (n* b + s * (1.0 + uz * (4.5 * uz - 3.0) - u2))* (!c);
	//printf("\n6. (%d, %d, %d) %d %.13f", x, y, z, c_nb6 + idx + 6*offset, dGrid[c_nb6 + idx + 6*offset]);

	b = DFL3 * OMEGA * rho; //resue variable b
	t = 1.0 - u2;
	e = ux + uy;
	w = b*(t+ 4.5*e*e);

	dGrid[c_nb7 + idx + 7*offset] = (sw * c) + (n* ne + w + 3*e*b)* (!c);
	//printf("\n7. (%d, %d, %d) %d %.13f", x, y, z, c_nb7 + idx + 7*offset, dGrid[c_nb7 + idx + 7*offset]);

	dGrid[c_nb10 + idx + 10*offset] = (ne * c)+ (n* sw + w - 3*e*b)* (!c);
	//printf("\n10. (%d, %d, %d) %d %.13f", x, y, z, c_nb10 + idx + 10*offset, dGrid[c_nb10 + idx + 10*offset]);

	e = -ux + uy;
	w = b*(t+ 4.5*e*e);

	dGrid[c_nb8 + idx + 8*offset] = (se * c) + (n* nw + w + 3*e*b)* (!c);
	//printf("\n8. (%d, %d, %d) %d %.13f", x, y, z, c_nb8 + idx + 8*offset, dGrid[c_nb8 + idx + 8*offset]);

	dGrid[c_nb9 + idx + 9*offset] = (nw * c) + (n* se + w - 3*e*b)* (!c);
	//printf("\n9. (%d, %d, %d) %d %.13f", x, y, z, c_nb9 + idx + 9*offset, dGrid[c_nb9 + idx + 9*offset]);

	e = uy + uz;
	w = b*(t+ 4.5*e*e);

	dGrid[c_nb11 + idx + 11*offset] = (sb * c)+ (n* nt + w + 3*e*b)* (!c);
	//printf("\n11. (%d, %d, %d) %d %.13f", x, y, z, c_nb11 + idx + 11*offset, dGrid[c_nb11 + idx + 11*offset]);

	dGrid[c_nb14 + idx + 14*offset] = (nt * c)+ (n* sb + w - 3*e*b)* (!c);
	//printf("\n14. (%d, %d, %d) %d %.13f", x, y, z, c_nb14 + idx + 14*offset, dGrid[c_nb14 + idx + 14*offset]);


	e = uy - uz;
	w = b*(t+ 4.5*e*e);

	dGrid[c_nb12 + idx + 12*offset] = (st * c)+ (n* nb + w + 3*e*b)* (!c);
	//printf("\n12. (%d, %d, %d) %d %.13f", x, y, z, c_nb12 + idx + 12*offset, dGrid[c_nb12 + idx + 12*offset]);

	dGrid[c_nb13 + idx + 13*offset] = (nb * c)+ (n* st + w - 3*e*b)* (!c);
	//printf("\n13. (%d, %d, %d) %d %.13f", x, y, z, c_nb13 + idx + 13*offset, dGrid[c_nb13 + idx + 13*offset]);


	e = ux + uz;
	w = b*(t+ 4.5*e*e);

	dGrid[c_nb15 + idx + 15*offset] = (wb * c)+ (n* et + w + 3*e*b)* (!c);
	//printf("\n15. (%d, %d, %d) %d %.13f", x, y, z, c_nb15 + idx + 15*offset, dGrid[c_nb15 + idx + 15*offset]);

	dGrid[c_nb18 + idx + 18*offset] = (et * c)+ (n* wb + w - 3*e*b)* (!c);
	//printf("\n18. (%d, %d, %d) %d %.13f", x, y, z, c_nb18 + idx + 18*offset, dGrid[c_nb18 + idx + 18*offset]);

	e = ux - uz;
	w = b*(t+ 4.5*e*e);

	dGrid[c_nb16 + idx + 16*offset] = (wt * c)+ (n* eb + w + 3*e*b)* (!c);
	//printf("\n16. (%d, %d, %d) %d %.13f", x, y, z, c_nb16 + idx + 16*offset, dGrid[c_nb16 + idx + 16*offset]);

	dGrid[c_nb17 + idx + 17*offset] = (eb * c)+ (n* wt + w - 3*e*b)* (!c);
	//printf("\n17. (%d, %d, %d) %d %.13f", x, y, z, c_nb17 + idx + 17*offset, dGrid[c_nb17 + idx + 17*offset]);

}

__inline__ __device__ int getThreadIdx()
{
	return threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
}
__inline__ __device__ int index3D(int nx, int ny, int x, int y, int z)
{
	return x + y*nx + z*nx*ny;
}
__global__ void lbm_kernel_SoA_CG(float* sGrid, float*  dGrid, unsigned char* __restrict__ const flags)
{
	__shared__ int offset;
	//__shared__ int startz, endz;
		//__shared__ float one_minus_omega;
	offset = SIZE_X*SIZE_Y*SIZE_Z;
//	__shared__ float  c[TILED_WIDTH_X + 1][TILED_WIDTH_Y + 1];
//	__shared__ float  n[TILED_WIDTH_X + 1][TILED_WIDTH_Y + 1];
//	__shared__ float  s[TILED_WIDTH_X + 1][TILED_WIDTH_Y + 1];
//	__shared__ float  e[TILED_WIDTH_X + 1][TILED_WIDTH_Y + 1];
//	__shared__ float  w[TILED_WIDTH_X + 1][TILED_WIDTH_Y + 1];
//	__shared__ float  t[TILED_WIDTH_X + 1][TILED_WIDTH_Y + 1];
//	__shared__ float  b[TILED_WIDTH_X + 1][TILED_WIDTH_Y + 1];
//	__shared__ float ne[TILED_WIDTH_X + 1][TILED_WIDTH_Y + 1];
//	__shared__ float nw[TILED_WIDTH_X + 1][TILED_WIDTH_Y + 1];
//	__shared__ float se[TILED_WIDTH_X + 1][TILED_WIDTH_Y + 1];
//	__shared__ float sw[TILED_WIDTH_X + 1][TILED_WIDTH_Y + 1];
//	__shared__ float nt[TILED_WIDTH_X + 1][TILED_WIDTH_Y + 1];
//	__shared__ float nb[TILED_WIDTH_X + 1][TILED_WIDTH_Y + 1];
//	__shared__ float st[TILED_WIDTH_X + 1][TILED_WIDTH_Y + 1];
//	__shared__ float sb[TILED_WIDTH_X + 1][TILED_WIDTH_Y + 1];
//	__shared__ float et[TILED_WIDTH_X + 1][TILED_WIDTH_Y + 1];
//	__shared__ float eb[TILED_WIDTH_X + 1][TILED_WIDTH_Y + 1];
//	__shared__ float wt[TILED_WIDTH_X + 1][TILED_WIDTH_Y + 1];
//	__shared__ float wb[TILED_WIDTH_X + 1][TILED_WIDTH_Y + 1];

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;


	//int x_tile = threadIdx.x + 1;
	//int y_tile = threadIdx.y + 1;


	float c, n, s, e, w, t, b, ne, nw, se, sw, nt, nb, st, sb, et, eb, wt, wb;
	unsigned short type;

	float rho;
	float ux;
	float uy;
	float uz;
	float u2;

	int startz = (blockIdx.z)*(TILED_WIDTH_Z);// (SIZE_Z/(gridDim.z));
	int endz = startz + TILED_WIDTH_Z; //((blockIdx.z) + 1) *(TILED_WIDTH_Z);// (SIZE_Z/(gridDim.z));

	int idx = MARGIN_L_SIZE + index3D(SIZE_X, SIZE_Y, x, y, startz);
	/*float c = sGrid[idx];
	float n = sGrid[idx +   offset];
	float s = sGrid[idx + 2*offset];
	float e = sGrid[idx + 3*offset];
	float w = sGrid[idx + 4*offset];

	float ne = sGrid[idx +  7*offset];
	float nw = sGrid[idx +  8*offset];
	float se = sGrid[idx +  9*offset];
	float sw = sGrid[idx + 10*offset];


	float b  = sGrid[idx +  6*offset];
	float nb = sGrid[idx + 12*offset];
	float sb = sGrid[idx + 14*offset];
	float eb = sGrid[idx + 16*offset];
	float wb = sGrid[idx + 18*offset];

	float t  = sGrid[idx +  5*offset];
	float nt = sGrid[idx + 11*offset];
	float st = sGrid[idx + 13*offset];
	float et = sGrid[idx + 15*offset];
	float wt = sGrid[idx + 17*offset];*/
//#pragma unroll
	for(int z = startz; z< endz; z++)
	{
		c = sGrid[idx]; //s_c;
		n = sGrid[idx +   offset]; //s_n;
		s = sGrid[idx + 2*offset]; //s_s;
		e = sGrid[idx + 3*offset]; //s_e;
		w = sGrid[idx + 4*offset]; //s_w;

		ne = sGrid[idx +  7*offset]; //s_ne;
		nw = sGrid[idx +  8*offset]; //s_nw;
		se = sGrid[idx +  9*offset]; //s_se;
		sw = sGrid[idx + 10*offset]; //s_sw;


		b  = sGrid[idx +  6*offset]; //s_b;
		nb = sGrid[idx + 12*offset]; //s_nb;
		sb = sGrid[idx + 14*offset]; //s_sb;
		eb = sGrid[idx + 16*offset]; //s_eb;
		wb = sGrid[idx + 18*offset]; //s_wb;

		t  = sGrid[idx +  5*offset]; //s_t;
		nt = sGrid[idx + 11*offset]; //s_nt;
		st = sGrid[idx + 13*offset]; //s_st;
		et = sGrid[idx + 15*offset]; //s_et;
		wt = sGrid[idx + 17*offset]; //s_wt;

		type = ((flags[idx-MARGIN_L_SIZE] == 227) << 8) | ((flags[idx-MARGIN_L_SIZE] == 228) & 0xff);


		rho = +c + n + s + e + w + t + b + ne + nw + se + sw + nt + nb + st + sb + et + eb + wt + wb;
		ux = ((+e - w + ne - nw + se - sw + et + eb - wt - wb)/rho)*(!(type & 0xff)) + 0.005*(type & 0xff);
		uy = ((+n - s + ne + nw - se - sw + nt + nb - st - sb)/rho)*(!(type & 0xff)) + 0.002*(type & 0xff);
		uz = ((+t - b + nt - nb + st - sb + et - eb + wt - wb)/rho)*(!(type & 0xff));

		u2 = 1.5 *( ux * ux +  uy * uy + uz * uz);

		dGrid[idx] = (c * (type >> 8))	+ ((1.0 - OMEGA)* c + DFL1 * OMEGA * rho * (1.0 - u2))* (!(type >> 8));

		c = (type >> 8); //resue variable c

		dGrid[c_nbArray[1] + idx + offset] = (s * c)  + ((1.0 - OMEGA)* n + DFL2 * OMEGA * rho * (1.0 + uy * (4.5 * uy + 3.0) - u2))* (!c);

		dGrid[c_nbArray[2] + idx + 2*offset] = (n * c) + ((1.0 - OMEGA)* s + DFL2 * OMEGA * rho * (1.0 + uy * (4.5 * uy - 3.0) - u2))* (!c);

		n = (1.0 - OMEGA); //resue variable n
		s = DFL2 * OMEGA * rho; //resue variable s

		dGrid[c_nbArray[3] + idx + 3*offset] = (w * c) + (n* e + s * (1.0 + ux * (4.5 * ux + 3.0) - u2))* (!c);

		dGrid[c_nbArray[4] + idx + 4*offset] = (e * c) + (n* w + s * (1.0 + ux * (4.5 * ux - 3.0) - u2))* (!c);

		dGrid[c_nbArray[5] + idx + 5*offset] = (b * c) + (n* t + s * (1.0 + uz * (4.5 * uz + 3.0) - u2))* (!c);

		dGrid[c_nbArray[6] + idx + 6*offset] = (t * c) + (n* b + s * (1.0 + uz * (4.5 * uz - 3.0) - u2))* (!c);

		b = DFL3 * OMEGA * rho; //resue variable b
		t = 1.0 - u2;
		e = ux + uy;
		w = b*(t+ 4.5*e*e);

		dGrid[c_nbArray[7] + idx + 7*offset] = (sw * c) + (n* ne + w + 3*e*b)* (!c);

		dGrid[c_nbArray[10] + idx + 10*offset] = (ne * c)+ (n* sw + w - 3*e*b)* (!c);

		e = -ux + uy;
		w = b*(t+ 4.5*e*e);

		dGrid[c_nbArray[8] + idx + 8*offset] = (se * c) + (n* nw + w + 3*e*b)* (!c);

		dGrid[c_nbArray[9] + idx + 9*offset] = (nw * c) + (n* se + w - 3*e*b)* (!c);

		e = uy + uz;
		w = b*(t+ 4.5*e*e);

		dGrid[c_nbArray[11] + idx + 11*offset] = (sb * c)+ (n* nt + w + 3*e*b)* (!c);

		dGrid[c_nbArray[14] + idx + 14*offset] = (nt * c)+ (n* sb + w - 3*e*b)* (!c);

		e = uy - uz;
		w = b*(t+ 4.5*e*e);

		dGrid[c_nbArray[12] + idx + 12*offset] = (st * c)+ (n* nb + w + 3*e*b)* (!c);

		dGrid[c_nbArray[13] + idx + 13*offset] = (nb * c)+ (n* st + w - 3*e*b)* (!c);

		e = ux + uz;
		w = b*(t+ 4.5*e*e);

		dGrid[c_nbArray[15] + idx + 15*offset] = (wb * c)+ (n* et + w + 3*e*b)* (!c);

		dGrid[c_nbArray[18] + idx + 18*offset] = (et * c)+ (n* wb + w - 3*e*b)* (!c);

		e = ux - uz;
		w = b*(t+ 4.5*e*e);

		dGrid[c_nbArray[16] + idx + 16*offset] = (wt * c)+ (n* eb + w + 3*e*b)* (!c);

		dGrid[c_nbArray[17] + idx + 17*offset] = (eb * c)+ (n* wt + w - 3*e*b)* (!c);

		idx = MARGIN_L_SIZE + index3D(SIZE_X, SIZE_Y, x, y, z);

		/*c = sGrid[idx]; //s_c;
		n = sGrid[idx +   offset]; //s_n;
		s = sGrid[idx + 2*offset]; //s_s;
		e = sGrid[idx + 3*offset]; //s_e;
		w = sGrid[idx + 4*offset]; //s_w;

		ne = sGrid[idx +  7*offset]; //s_ne;
		nw = sGrid[idx +  8*offset]; //s_nw;
		se = sGrid[idx +  9*offset]; //s_se;
		sw = sGrid[idx + 10*offset]; //s_sw;


		b  = sGrid[idx +  6*offset]; //s_b;
		nb = sGrid[idx + 12*offset]; //s_nb;
		sb = sGrid[idx + 14*offset]; //s_sb;
		eb = sGrid[idx + 16*offset]; //s_eb;
		wb = sGrid[idx + 18*offset]; //s_wb;

		t  = sGrid[idx +  5*offset]; //s_t;
		nt = sGrid[idx + 11*offset]; //s_nt;
		st = sGrid[idx + 13*offset]; //s_st;
		et = sGrid[idx + 15*offset]; //s_et;
		wt = sGrid[idx + 17*offset]; //s_wt;
	*/
		/*
		s_c = sGrid[idx];
		s_n = sGrid[idx +   offset];
		s_s = sGrid[idx + 2*offset];
		s_e = sGrid[idx + 3*offset];
		s_w = sGrid[idx + 4*offset];

		s_ne = sGrid[idx +  7*offset];
		s_nw = sGrid[idx +  8*offset];
		s_se = sGrid[idx +  9*offset];
		s_sw = sGrid[idx + 10*offset];

		s_b  = sGrid[idx +  6*offset];
		s_nb = sGrid[idx + 12*offset];
		s_sb = sGrid[idx + 14*offset];
		s_eb = sGrid[idx + 16*offset];
		s_wb = sGrid[idx + 18*offset];

		s_t  = sGrid[idx +  5*offset];
		s_nt = sGrid[idx + 11*offset];
		s_st = sGrid[idx + 13*offset];
		s_et = sGrid[idx + 15*offset];
		s_wt = sGrid[idx + 17*offset];
	*/


		/*if(threadIdx.x == 0)
		{	sIdx = MARGIN_L + index3D(SIZE_X, SIZE_Y, x -1, y, z);
			c[x_tile - 1][y_tile] = sGrid[sIdx];
			n[x_tile - 1][y_tile] = sGrid[sIdx + offset];
			s[x_tile - 1][y_tile] = sGrid[sIdx + 2*offset];
			e[x_tile - 1][y_tile] = sGrid[sIdx + 3*offset];
			w[x_tile - 1][y_tile] = sGrid[sIdx + 4*offset];
			//t[x_tile - 1][y_tile] = sGrid[sIdx + 5*offset];
			t = sGrid[sIdx + 5*offset];
			//b[x_tile - 1][y_tile] = sGrid[sIdx + 6*offset];
			b = sGrid[sIdx + 6*offset];
			ne[x_tile - 1][y_tile] = sGrid[sIdx + 7*offset];
			nw[x_tile - 1][y_tile] = sGrid[sIdx + 8*offset];
			se[x_tile - 1][y_tile] = sGrid[sIdx + 9*offset];
			sw[x_tile - 1][y_tile] = sGrid[sIdx + 10*offset];
			//nt[x_tile - 1][y_tile] = sGrid[sIdx + 11*offset];
			nt = sGrid[sIdx + 11*offset];
			//nb[x_tile - 1][y_tile] = sGrid[sIdx + 12*offset];
			nb = sGrid[sIdx + 13*offset];
			//st[x_tile - 1][y_tile] = sGrid[sIdx + 13*offset];
			st = sGrid[sIdx + 13*offset];
			//sb[x_tile - 1][y_tile] = sGrid[sIdx + 14*offset];
			sb = sGrid[sIdx + 14*offset];
			//et[x_tile - 1][y_tile] = sGrid[sIdx + 15*offset];
			et = sGrid[sIdx + 15*offset];
			//eb[x_tile - 1][y_tile] = sGrid[sIdx + 16*offset];
			eb = sGrid[sIdx + 16*offset];
			//wt[x_tile - 1][y_tile] = sGrid[sIdx + 17*offset];
			wt = sGrid[sIdx + 17*offset];
			//wb[x_tile - 1][y_tile] = sGrid[sIdx + 18*offset];
			wb = sGrid[sIdx + 18*offset];
		}
		if(threadIdx.x == blockDim.x - 1)
		{	sIdx = MARGIN_L + index3D(SIZE_X, SIZE_Y, x + 1, y, z);
			c[x_tile + 1][y_tile] = sGrid[sIdx];
			n[x_tile + 1][y_tile] = sGrid[sIdx + offset];
			s[x_tile + 1][y_tile] = sGrid[sIdx + 2*offset];
			e[x_tile + 1][y_tile] = sGrid[sIdx + 3*offset];
			w[x_tile + 1][y_tile] = sGrid[sIdx + 4*offset];
			//t[x_tile + 1][y_tile] = sGrid[sIdx + 5*offset];
			t = sGrid[sIdx + 5*offset];
			//b[x_tile + 1][y_tile] = sGrid[sIdx + 6*offset];
			b = sGrid[sIdx + 6*offset];
			ne[x_tile + 1][y_tile] = sGrid[sIdx + 7*offset];
			nw[x_tile + 1][y_tile] = sGrid[sIdx + 8*offset];
			se[x_tile + 1][y_tile] = sGrid[sIdx + 9*offset];
			sw[x_tile + 1][y_tile] = sGrid[sIdx + 10*offset];
			//nt[x_tile + 1][y_tile] = sGrid[sIdx + 11*offset];
			nt = sGrid[sIdx + 11*offset];
			//nb[x_tile + 1][y_tile] = sGrid[sIdx + 12*offset];
			nb = sGrid[sIdx + 13*offset];
			//st[x_tile + 1][y_tile] = sGrid[sIdx + 13*offset];
			st = sGrid[sIdx + 13*offset];
			//sb[x_tile + 1][y_tile] = sGrid[sIdx + 14*offset];
			sb = sGrid[sIdx + 14*offset];
			//et[x_tile + 1][y_tile] = sGrid[sIdx + 15*offset];
			et = sGrid[sIdx + 15*offset];
			//eb[x_tile + 1][y_tile] = sGrid[sIdx + 16*offset];
			eb = sGrid[sIdx + 16*offset];
			//wt[x_tile + 1][y_tile] = sGrid[sIdx + 17*offset];
			wt = sGrid[sIdx + 17*offset];
			//wb[x_tile + 1][y_tile] = sGrid[sIdx + 18*offset];
			wb = sGrid[sIdx + 18*offset];
		}
		if(threadIdx.y == 0)
		{	sIdx = MARGIN_L + index3D(SIZE_X, SIZE_Y, x, y - 1, z);
			c[x_tile][y_tile - 1] = sGrid[sIdx];
			n[x_tile][y_tile - 1] = sGrid[sIdx + offset];
			s[x_tile][y_tile - 1] = sGrid[sIdx + 2*offset];
			e[x_tile][y_tile - 1] = sGrid[sIdx + 3*offset];
			w[x_tile][y_tile - 1] = sGrid[sIdx + 4*offset];
			//t[x_tile][y_tile - 1] = sGrid[sIdx + 5*offset];
			t = sGrid[sIdx + 5*offset];
			//b[x_tile][y_tile - 1] = sGrid[sIdx + 6*offset];
			b = sGrid[sIdx + 6*offset];
			ne[x_tile][y_tile - 1] = sGrid[sIdx + 7*offset];
			nw[x_tile][y_tile - 1] = sGrid[sIdx + 8*offset];
			se[x_tile][y_tile - 1] = sGrid[sIdx + 9*offset];
			sw[x_tile][y_tile - 1] = sGrid[sIdx + 10*offset];
			//nt[x_tile][y_tile - 1] = sGrid[sIdx + 11*offset];
			nt = sGrid[sIdx + 11*offset];
			//nb[x_tile][y_tile - 1] = sGrid[sIdx + 12*offset];
			nb = sGrid[sIdx + 13*offset];
			//st[x_tile][y_tile - 1] = sGrid[sIdx + 13*offset];
			st = sGrid[sIdx + 13*offset];
			//sb[x_tile][y_tile - 1] = sGrid[sIdx + 14*offset];
			sb = sGrid[sIdx + 14*offset];
			//et[x_tile][y_tile - 1] = sGrid[sIdx + 15*offset];
			et = sGrid[sIdx + 15*offset];
			//eb[x_tile][y_tile - 1] = sGrid[sIdx + 16*offset];
			eb = sGrid[sIdx + 16*offset];
			//wt[x_tile][y_tile - 1] = sGrid[sIdx + 17*offset];
			wt = sGrid[sIdx + 17*offset];
			//wb[x_tile][y_tile - 1] = sGrid[sIdx + 18*offset];
			wb = sGrid[sIdx + 18*offset];
		}
		if(threadIdx.y == blockDim.y - 1)
		{	sIdx = MARGIN_L + index3D(SIZE_X, SIZE_Y, x, y + 1, z);
			c[x_tile][y_tile + 1] = sGrid[sIdx];
			n[x_tile][y_tile + 1] = sGrid[sIdx + offset];
			s[x_tile][y_tile + 1] = sGrid[sIdx + 2*offset];
			e[x_tile][y_tile + 1] = sGrid[sIdx + 3*offset];
			w[x_tile][y_tile + 1] = sGrid[sIdx + 4*offset];
			//t[x_tile][y_tile + 1] = sGrid[sIdx + 5*offset];
			t = sGrid[sIdx + 5*offset];
			//b[x_tile][y_tile + 1] = sGrid[sIdx + 6*offset];
			b = sGrid[sIdx + 6*offset];
			ne[x_tile][y_tile + 1] = sGrid[sIdx + 7*offset];
			nw[x_tile][y_tile + 1] = sGrid[sIdx + 8*offset];
			se[x_tile][y_tile + 1] = sGrid[sIdx + 9*offset];
			sw[x_tile][y_tile + 1] = sGrid[sIdx + 10*offset];
			//nt[x_tile][y_tile + 1] = sGrid[sIdx + 11*offset];
			nt = sGrid[sIdx + 11*offset];
			//nb[x_tile][y_tile + 1] = sGrid[sIdx + 12*offset];
			nb = sGrid[sIdx + 13*offset];
			//st[x_tile][y_tile + 1] = sGrid[sIdx + 13*offset];
			st = sGrid[sIdx + 13*offset];
			//sb[x_tile][y_tile + 1] = sGrid[sIdx + 14*offset];
			sb = sGrid[sIdx + 14*offset];
			//et[x_tile][y_tile + 1] = sGrid[sIdx + 15*offset];
			et = sGrid[sIdx + 15*offset];
			//eb[x_tile][y_tile + 1] = sGrid[sIdx + 16*offset];
			eb = sGrid[sIdx + 16*offset];
			//wt[x_tile][y_tile + 1] = sGrid[sIdx + 17*offset];
			wt = sGrid[sIdx + 17*offset];
			//wb[x_tile][y_tile + 1] = sGrid[sIdx + 18*offset];
			wb = sGrid[sIdx + 18*offset];
		}*/


	}


	/*


	//get the value final cell (20) to know the kind of cell: fluid or accel or obstacle
	//int flag = flags[idx];
	unsigned short type = ((flags[idx] == 227) << 8) | ((flags[idx] == 228) & 0xff);

	float rho = +c[localIdx] + n[localIdx] + s[localIdx] + e[localIdx] + w[localIdx] + t[localIdx] + b[localIdx] + ne[localIdx] + nw[localIdx] + se[localIdx] + sw[localIdx] + nt[localIdx] + nb[localIdx] + st[localIdx] + sb[localIdx] + et[localIdx] + eb[localIdx] + wt[localIdx] + wb[localIdx];
	float ux = ((+e[localIdx] - w[localIdx] + ne[localIdx] - nw[localIdx] + se[localIdx] - sw[localIdx] + et[localIdx] + eb[localIdx] - wt[localIdx] - wb[localIdx])/rho)*(!(type & 0xff)) + 0.005*(type & 0xff);
	float uy = ((+n[localIdx] - s[localIdx] + ne[localIdx] + nw[localIdx] - se[localIdx] - sw[localIdx] + nt[localIdx] + nb[localIdx] - st[localIdx] - sb[localIdx])/rho)*(!(type & 0xff)) + 0.002*(type & 0xff);
	float uz = ((+t[localIdx] - b[localIdx] + nt[localIdx] - nb[localIdx] + st[localIdx] - sb[localIdx] + et[localIdx] - eb[localIdx] + wt[localIdx] - wb[localIdx])/rho)*(!(type & 0xff));

	float u2 = 1.5 *( ux * ux +  uy * uy + uz * uz);

	dGrid[idx] = (c[localIdx] * (type >> 8))	+ ((1.0 - OMEGA)* c[localIdx] + DFL1 * OMEGA * rho * (1.0 - u2))* (!(type >> 8));

	dGrid[c_nb1 + idx + offset] = (s[localIdx] * (type >> 8))  + ((1.0 - OMEGA)* n[localIdx] + DFL2 * OMEGA * rho * (1.0 + uy * (4.5 * uy + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nb2 + idx + 2*offset] = (n[localIdx] * (type >> 8)) + ((1.0 - OMEGA)* s[localIdx] + DFL2 * OMEGA * rho * (1.0 + uy * (4.5 * uy - 3.0) - u2))* (!(type >> 8));

	dGrid[c_nb3 + idx + 3*offset] = (w[localIdx] * (type >> 8)) + ((1.0 - OMEGA)* e[localIdx] + DFL2 * OMEGA * rho * (1.0 + ux * (4.5 * ux + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nb4 + idx + 4*offset] = (e[localIdx] * (type >> 8)) + ((1.0 - OMEGA)* w[localIdx] + DFL2 * OMEGA * rho * (1.0 + ux * (4.5 * ux - 3.0) - u2))* (!(type >> 8));

	dGrid[c_nb5 + idx + 5*offset] = (b[localIdx] * (type >> 8)) + ((1.0 - OMEGA)* t[localIdx] + DFL2 * OMEGA * rho * (1.0 + uz * (4.5 * uz + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nb6 + idx + 6*offset] = (t[localIdx] * (type >> 8)) + ((1.0 - OMEGA)* b[localIdx] + DFL2 * OMEGA * rho * (1.0 + uz * (4.5 * uz - 3.0) - u2))* (!(type >> 8));

	dGrid[c_nb7 + idx + 7*offset] = (sw[localIdx] * (type >> 8)) + ((1.0 - OMEGA)* ne[localIdx] + DFL3 * OMEGA * rho * (1.0 + (+ux + uy) * (4.5 * (+ux + uy) + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nb8 + idx + 8*offset] = (se[localIdx] * (type >> 8)) + ((1.0 - OMEGA)* nw[localIdx] + DFL3 * OMEGA * rho * (1.0 + (-ux + uy) * (4.5 * (-ux + uy) + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nb9 + idx + 9*offset] = (nw[localIdx] * (type >> 8)) + ((1.0 - OMEGA)* se[localIdx] + DFL3 * OMEGA * rho * (1.0 + (+ux - uy) * (4.5 * (+ux - uy) + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nb10 + idx + 10*offset] = (ne[localIdx] * (type >> 8))+ ((1.0 - OMEGA)* sw[localIdx] + DFL3 * OMEGA * rho * (1.0 + (-ux - uy) * (4.5 * (-ux - uy) + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nb11 + idx + 11*offset] = (sb[localIdx] * (type >> 8))+ ((1.0 - OMEGA)* nt[localIdx] + DFL3 * OMEGA * rho * (1.0 + (+uy + uz) * (4.5 * (+uy + uz) + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nb12 + idx + 12*offset] = (st[localIdx] * (type >> 8))+ ((1.0 - OMEGA)* nb[localIdx] + DFL3 * OMEGA * rho * (1.0 + (+uy - uz) * (4.5 * (+uy - uz) + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nb13 + idx + 13*offset] = (nb[localIdx] * (type >> 8))+ ((1.0 - OMEGA)* st[localIdx] + DFL3 * OMEGA * rho * (1.0 + (-uy + uz) * (4.5 * (-uy + uz) + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nb14 + idx + 14*offset] = (nt[localIdx] * (type >> 8))+ ((1.0 - OMEGA)* sb[localIdx] + DFL3 * OMEGA * rho * (1.0 + (-uy - uz) * (4.5 * (-uy - uz) + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nb15 + idx + 15*offset] = (wb[localIdx] * (type >> 8))+ ((1.0 - OMEGA)* et[localIdx] + DFL3 * OMEGA * rho * (1.0 + (+ux + uz) * (4.5 * (+ux + uz) + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nb16 + idx + 16*offset] = (wt[localIdx] * (type >> 8))+ ((1.0 - OMEGA)* eb[localIdx] + DFL3 * OMEGA * rho * (1.0 + (+ux - uz) * (4.5 * (+ux - uz) + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nb17 + idx + 17*offset] = (eb[localIdx] * (type >> 8))+ ((1.0 - OMEGA)* wt[localIdx] + DFL3 * OMEGA * rho * (1.0 + (-ux + uz) * (4.5 * (-ux + uz) + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nb18 + idx + 18*offset] = (et[localIdx] * (type >> 8))+ ((1.0 - OMEGA)* wb[localIdx] + DFL3 * OMEGA * rho * (1.0 + (-ux - uz) * (4.5 * (-ux - uz) + 3.0) - u2))* (!(type >> 8));
*/
}
__global__ void lbm_kernel_SoA2(float* sGrid, float*  dGrid, unsigned char* __restrict__ const flags)
{
	__shared__ int offset;

		//__shared__ float one_minus_omega;
	offset = SIZE_X*SIZE_Y*SIZE_Z;
	//__shared__ float c[TILED_WIDTH_X + 1][TILED_WIDTH_Y + 1];
	//__shared__ float n[TILED_WIDTH_X + 1][TILED_WIDTH_Y + 1];
	//__shared__ float s[TILED_WIDTH_X + 1][TILED_WIDTH_Y + 1];
	//__shared__ float e[TILED_WIDTH_X + 1][TILED_WIDTH_Y + 1];
	//__shared__ float w[TILED_WIDTH_X + 1][TILED_WIDTH_Y + 1];
	//__shared__ float t[TILED_WIDTH_X + 1][TILED_WIDTH_Y + 1];
	//__shared__ float b[TILED_WIDTH_X + 1][TILED_WIDTH_Y + 1];
	//__shared__ float ne[TILED_WIDTH_X + 1][TILED_WIDTH_Y + 1];
	//__shared__ float nw[TILED_WIDTH_X + 1][TILED_WIDTH_Y + 1];
	//__shared__ float se[TILED_WIDTH_X + 1][TILED_WIDTH_Y + 1];
	//__shared__ float sw[TILED_WIDTH_X + 1][TILED_WIDTH_Y + 1];
	//__shared__ float nt[TILED_WIDTH_X + 1][TILED_WIDTH_Y + 1];
	//__shared__ float nb[TILED_WIDTH_X + 1][TILED_WIDTH_Y + 1];
	//__shared__ float st[TILED_WIDTH_X + 1][TILED_WIDTH_Y + 1];
	//__shared__ float sb[TILED_WIDTH_X + 1][TILED_WIDTH_Y + 1];
	//__shared__ float et[TILED_WIDTH_X + 1][TILED_WIDTH_Y + 1];
	//__shared__ float eb[TILED_WIDTH_X + 1][TILED_WIDTH_Y + 1];
	//__shared__ float wt[TILED_WIDTH_X + 1][TILED_WIDTH_Y + 1];
	//__shared__ float wb[TILED_WIDTH_X + 1][TILED_WIDTH_Y + 1];


	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;


	//int x_tile = threadIdx.x + 1;
	//int y_tile = threadIdx.y + 1;

	int idx;
	float c, n, s, e, w, t, b, ne, nw, se, sw, nt, nb, st, sb, et, eb, wt, wb;

	idx = MARGIN_L + index3D(SIZE_X, SIZE_Y, x, y, 0);

	b  = sGrid[idx +  6*offset];
	nb = sGrid[idx + 12*offset];
	sb = sGrid[idx + 14*offset];
	eb = sGrid[idx + 16*offset];
	wb = sGrid[idx + 18*offset];
	int startz = (blockIdx.z)*(SIZE_Z/(gridDim.z));
	int endz = ((blockIdx.z) + 1) * (SIZE_Z/(gridDim.z));
	for(int z = startz; z< endz; z++)
	{
		idx = MARGIN_L + index3D(SIZE_X, SIZE_Y, x, y, z);
		c = sGrid[idx];
		n = sGrid[idx +   offset];
		s = sGrid[idx + 2*offset];
		e = sGrid[idx + 3*offset];
		w = sGrid[idx + 4*offset];

		ne = sGrid[idx +  7*offset];
		nw = sGrid[idx +  8*offset];
		se = sGrid[idx +  9*offset];
		sw = sGrid[idx + 10*offset];

		//__syncthreads();

		t  = sGrid[idx +  5*offset];
		nt = sGrid[idx + 11*offset];
		st = sGrid[idx + 13*offset];
		et = sGrid[idx + 15*offset];
		wt = sGrid[idx + 17*offset];

		unsigned short type = ((flags[idx] == 227) << 8) | ((flags[idx] == 228) & 0xff);


		float rho = +c + n + s + e + w + t + b + ne + nw + se + sw + nt + nb + st + sb + et + eb + wt + wb;
		float ux = ((+e - w + ne - nw + se - sw + et + eb - wt - wb)/rho)*(!(type & 0xff)) + 0.005*(type & 0xff);
		float uy = ((+n - s + ne + nw - se - sw + nt + nb - st - sb)/rho)*(!(type & 0xff)) + 0.002*(type & 0xff);
		float uz = ((+t - b + nt - nb + st - sb + et - eb + wt - wb)/rho)*(!(type & 0xff));

		float u2 = 1.5 *( ux * ux +  uy * uy + uz * uz);

		dGrid[idx] = (c * (type >> 8))	+ ((1.0 - OMEGA)* c + DFL1 * OMEGA * rho * (1.0 - u2))* (!(type >> 8));

		dGrid[c_nb1 + idx + offset] = (s * (type >> 8))  + ((1.0 - OMEGA)* n + DFL2 * OMEGA * rho * (1.0 + uy * (4.5 * uy + 3.0) - u2))* (!(type >> 8));

		dGrid[c_nb2 + idx + 2*offset] = (n * (type >> 8)) + ((1.0 - OMEGA)* s + DFL2 * OMEGA * rho * (1.0 + uy * (4.5 * uy - 3.0) - u2))* (!(type >> 8));

		dGrid[c_nb3 + idx + 3*offset] = (w * (type >> 8)) + ((1.0 - OMEGA)* e + DFL2 * OMEGA * rho * (1.0 + ux * (4.5 * ux + 3.0) - u2))* (!(type >> 8));

		dGrid[c_nb4 + idx + 4*offset] = (e * (type >> 8)) + ((1.0 - OMEGA)* w + DFL2 * OMEGA * rho * (1.0 + ux * (4.5 * ux - 3.0) - u2))* (!(type >> 8));

		dGrid[c_nb5 + idx + 5*offset] = (b * (type >> 8)) + ((1.0 - OMEGA)* t + DFL2 * OMEGA * rho * (1.0 + uz * (4.5 * uz + 3.0) - u2))* (!(type >> 8));

		dGrid[c_nb6 + idx + 6*offset] = (t * (type >> 8)) + ((1.0 - OMEGA)* b + DFL2 * OMEGA * rho * (1.0 + uz * (4.5 * uz - 3.0) - u2))* (!(type >> 8));

		dGrid[c_nb7 + idx + 7*offset] = (sw * (type >> 8)) + ((1.0 - OMEGA)* ne + DFL3 * OMEGA * rho * (1.0 + (+ux + uy) * (4.5 * (+ux + uy) + 3.0) - u2))* (!(type >> 8));

		dGrid[c_nb8 + idx + 8*offset] = (se * (type >> 8)) + ((1.0 - OMEGA)* nw + DFL3 * OMEGA * rho * (1.0 + (-ux + uy) * (4.5 * (-ux + uy) + 3.0) - u2))* (!(type >> 8));

		dGrid[c_nb9 + idx + 9*offset] = (nw * (type >> 8)) + ((1.0 - OMEGA)* se + DFL3 * OMEGA * rho * (1.0 + (+ux - uy) * (4.5 * (+ux - uy) + 3.0) - u2))* (!(type >> 8));

		dGrid[c_nb10 + idx + 10*offset] = (ne * (type >> 8))+ ((1.0 - OMEGA)* sw + DFL3 * OMEGA * rho * (1.0 + (-ux - uy) * (4.5 * (-ux - uy) + 3.0) - u2))* (!(type >> 8));

		dGrid[c_nb11 + idx + 11*offset] = (sb * (type >> 8))+ ((1.0 - OMEGA)* nt + DFL3 * OMEGA * rho * (1.0 + (+uy + uz) * (4.5 * (+uy + uz) + 3.0) - u2))* (!(type >> 8));

		dGrid[c_nb12 + idx + 12*offset] = (st * (type >> 8))+ ((1.0 - OMEGA)* nb + DFL3 * OMEGA * rho * (1.0 + (+uy - uz) * (4.5 * (+uy - uz) + 3.0) - u2))* (!(type >> 8));

		dGrid[c_nb13 + idx + 13*offset] = (nb * (type >> 8))+ ((1.0 - OMEGA)* st + DFL3 * OMEGA * rho * (1.0 + (-uy + uz) * (4.5 * (-uy + uz) + 3.0) - u2))* (!(type >> 8));

		dGrid[c_nb14 + idx + 14*offset] = (nt * (type >> 8))+ ((1.0 - OMEGA)* sb + DFL3 * OMEGA * rho * (1.0 + (-uy - uz) * (4.5 * (-uy - uz) + 3.0) - u2))* (!(type >> 8));

		dGrid[c_nb15 + idx + 15*offset] = (wb * (type >> 8))+ ((1.0 - OMEGA)* et + DFL3 * OMEGA * rho * (1.0 + (+ux + uz) * (4.5 * (+ux + uz) + 3.0) - u2))* (!(type >> 8));

		dGrid[c_nb16 + idx + 16*offset] = (wt * (type >> 8))+ ((1.0 - OMEGA)* eb + DFL3 * OMEGA * rho * (1.0 + (+ux - uz) * (4.5 * (+ux - uz) + 3.0) - u2))* (!(type >> 8));

		dGrid[c_nb17 + idx + 17*offset] = (eb * (type >> 8))+ ((1.0 - OMEGA)* wt + DFL3 * OMEGA * rho * (1.0 + (-ux + uz) * (4.5 * (-ux + uz) + 3.0) - u2))* (!(type >> 8));

		dGrid[c_nb18 + idx + 18*offset] = (et * (type >> 8))+ ((1.0 - OMEGA)* wb + DFL3 * OMEGA * rho * (1.0 + (-ux - uz) * (4.5 * (-ux - uz) + 3.0) - u2))* (!(type >> 8));

		b = t;
		nb = nt;
		sb = st;
		eb = et;
		wb = wt;


		/*if(threadIdx.x == 0)
		{	sIdx = MARGIN_L + index3D(SIZE_X, SIZE_Y, x -1, y, z);
			c[x_tile - 1][y_tile] = sGrid[sIdx];
			n[x_tile - 1][y_tile] = sGrid[sIdx + offset];
			s[x_tile - 1][y_tile] = sGrid[sIdx + 2*offset];
			e[x_tile - 1][y_tile] = sGrid[sIdx + 3*offset];
			w[x_tile - 1][y_tile] = sGrid[sIdx + 4*offset];
			//t[x_tile - 1][y_tile] = sGrid[sIdx + 5*offset];
			t = sGrid[sIdx + 5*offset];
			//b[x_tile - 1][y_tile] = sGrid[sIdx + 6*offset];
			b = sGrid[sIdx + 6*offset];
			ne[x_tile - 1][y_tile] = sGrid[sIdx + 7*offset];
			nw[x_tile - 1][y_tile] = sGrid[sIdx + 8*offset];
			se[x_tile - 1][y_tile] = sGrid[sIdx + 9*offset];
			sw[x_tile - 1][y_tile] = sGrid[sIdx + 10*offset];
			//nt[x_tile - 1][y_tile] = sGrid[sIdx + 11*offset];
			nt = sGrid[sIdx + 11*offset];
			//nb[x_tile - 1][y_tile] = sGrid[sIdx + 12*offset];
			nb = sGrid[sIdx + 13*offset];
			//st[x_tile - 1][y_tile] = sGrid[sIdx + 13*offset];
			st = sGrid[sIdx + 13*offset];
			//sb[x_tile - 1][y_tile] = sGrid[sIdx + 14*offset];
			sb = sGrid[sIdx + 14*offset];
			//et[x_tile - 1][y_tile] = sGrid[sIdx + 15*offset];
			et = sGrid[sIdx + 15*offset];
			//eb[x_tile - 1][y_tile] = sGrid[sIdx + 16*offset];
			eb = sGrid[sIdx + 16*offset];
			//wt[x_tile - 1][y_tile] = sGrid[sIdx + 17*offset];
			wt = sGrid[sIdx + 17*offset];
			//wb[x_tile - 1][y_tile] = sGrid[sIdx + 18*offset];
			wb = sGrid[sIdx + 18*offset];
		}
		if(threadIdx.x == blockDim.x - 1)
		{	sIdx = MARGIN_L + index3D(SIZE_X, SIZE_Y, x + 1, y, z);
			c[x_tile + 1][y_tile] = sGrid[sIdx];
			n[x_tile + 1][y_tile] = sGrid[sIdx + offset];
			s[x_tile + 1][y_tile] = sGrid[sIdx + 2*offset];
			e[x_tile + 1][y_tile] = sGrid[sIdx + 3*offset];
			w[x_tile + 1][y_tile] = sGrid[sIdx + 4*offset];
			//t[x_tile + 1][y_tile] = sGrid[sIdx + 5*offset];
			t = sGrid[sIdx + 5*offset];
			//b[x_tile + 1][y_tile] = sGrid[sIdx + 6*offset];
			b = sGrid[sIdx + 6*offset];
			ne[x_tile + 1][y_tile] = sGrid[sIdx + 7*offset];
			nw[x_tile + 1][y_tile] = sGrid[sIdx + 8*offset];
			se[x_tile + 1][y_tile] = sGrid[sIdx + 9*offset];
			sw[x_tile + 1][y_tile] = sGrid[sIdx + 10*offset];
			//nt[x_tile + 1][y_tile] = sGrid[sIdx + 11*offset];
			nt = sGrid[sIdx + 11*offset];
			//nb[x_tile + 1][y_tile] = sGrid[sIdx + 12*offset];
			nb = sGrid[sIdx + 13*offset];
			//st[x_tile + 1][y_tile] = sGrid[sIdx + 13*offset];
			st = sGrid[sIdx + 13*offset];
			//sb[x_tile + 1][y_tile] = sGrid[sIdx + 14*offset];
			sb = sGrid[sIdx + 14*offset];
			//et[x_tile + 1][y_tile] = sGrid[sIdx + 15*offset];
			et = sGrid[sIdx + 15*offset];
			//eb[x_tile + 1][y_tile] = sGrid[sIdx + 16*offset];
			eb = sGrid[sIdx + 16*offset];
			//wt[x_tile + 1][y_tile] = sGrid[sIdx + 17*offset];
			wt = sGrid[sIdx + 17*offset];
			//wb[x_tile + 1][y_tile] = sGrid[sIdx + 18*offset];
			wb = sGrid[sIdx + 18*offset];
		}
		if(threadIdx.y == 0)
		{	sIdx = MARGIN_L + index3D(SIZE_X, SIZE_Y, x, y - 1, z);
			c[x_tile][y_tile - 1] = sGrid[sIdx];
			n[x_tile][y_tile - 1] = sGrid[sIdx + offset];
			s[x_tile][y_tile - 1] = sGrid[sIdx + 2*offset];
			e[x_tile][y_tile - 1] = sGrid[sIdx + 3*offset];
			w[x_tile][y_tile - 1] = sGrid[sIdx + 4*offset];
			//t[x_tile][y_tile - 1] = sGrid[sIdx + 5*offset];
			t = sGrid[sIdx + 5*offset];
			//b[x_tile][y_tile - 1] = sGrid[sIdx + 6*offset];
			b = sGrid[sIdx + 6*offset];
			ne[x_tile][y_tile - 1] = sGrid[sIdx + 7*offset];
			nw[x_tile][y_tile - 1] = sGrid[sIdx + 8*offset];
			se[x_tile][y_tile - 1] = sGrid[sIdx + 9*offset];
			sw[x_tile][y_tile - 1] = sGrid[sIdx + 10*offset];
			//nt[x_tile][y_tile - 1] = sGrid[sIdx + 11*offset];
			nt = sGrid[sIdx + 11*offset];
			//nb[x_tile][y_tile - 1] = sGrid[sIdx + 12*offset];
			nb = sGrid[sIdx + 13*offset];
			//st[x_tile][y_tile - 1] = sGrid[sIdx + 13*offset];
			st = sGrid[sIdx + 13*offset];
			//sb[x_tile][y_tile - 1] = sGrid[sIdx + 14*offset];
			sb = sGrid[sIdx + 14*offset];
			//et[x_tile][y_tile - 1] = sGrid[sIdx + 15*offset];
			et = sGrid[sIdx + 15*offset];
			//eb[x_tile][y_tile - 1] = sGrid[sIdx + 16*offset];
			eb = sGrid[sIdx + 16*offset];
			//wt[x_tile][y_tile - 1] = sGrid[sIdx + 17*offset];
			wt = sGrid[sIdx + 17*offset];
			//wb[x_tile][y_tile - 1] = sGrid[sIdx + 18*offset];
			wb = sGrid[sIdx + 18*offset];
		}
		if(threadIdx.y == blockDim.y - 1)
		{	sIdx = MARGIN_L + index3D(SIZE_X, SIZE_Y, x, y + 1, z);
			c[x_tile][y_tile + 1] = sGrid[sIdx];
			n[x_tile][y_tile + 1] = sGrid[sIdx + offset];
			s[x_tile][y_tile + 1] = sGrid[sIdx + 2*offset];
			e[x_tile][y_tile + 1] = sGrid[sIdx + 3*offset];
			w[x_tile][y_tile + 1] = sGrid[sIdx + 4*offset];
			//t[x_tile][y_tile + 1] = sGrid[sIdx + 5*offset];
			t = sGrid[sIdx + 5*offset];
			//b[x_tile][y_tile + 1] = sGrid[sIdx + 6*offset];
			b = sGrid[sIdx + 6*offset];
			ne[x_tile][y_tile + 1] = sGrid[sIdx + 7*offset];
			nw[x_tile][y_tile + 1] = sGrid[sIdx + 8*offset];
			se[x_tile][y_tile + 1] = sGrid[sIdx + 9*offset];
			sw[x_tile][y_tile + 1] = sGrid[sIdx + 10*offset];
			//nt[x_tile][y_tile + 1] = sGrid[sIdx + 11*offset];
			nt = sGrid[sIdx + 11*offset];
			//nb[x_tile][y_tile + 1] = sGrid[sIdx + 12*offset];
			nb = sGrid[sIdx + 13*offset];
			//st[x_tile][y_tile + 1] = sGrid[sIdx + 13*offset];
			st = sGrid[sIdx + 13*offset];
			//sb[x_tile][y_tile + 1] = sGrid[sIdx + 14*offset];
			sb = sGrid[sIdx + 14*offset];
			//et[x_tile][y_tile + 1] = sGrid[sIdx + 15*offset];
			et = sGrid[sIdx + 15*offset];
			//eb[x_tile][y_tile + 1] = sGrid[sIdx + 16*offset];
			eb = sGrid[sIdx + 16*offset];
			//wt[x_tile][y_tile + 1] = sGrid[sIdx + 17*offset];
			wt = sGrid[sIdx + 17*offset];
			//wb[x_tile][y_tile + 1] = sGrid[sIdx + 18*offset];
			wb = sGrid[sIdx + 18*offset];
		}*/


	}

	/*


	//get the value final cell (20) to know the kind of cell: fluid or accel or obstacle
	//int flag = flags[idx];
	unsigned short type = ((flags[idx] == 227) << 8) | ((flags[idx] == 228) & 0xff);

	float rho = +c[localIdx] + n[localIdx] + s[localIdx] + e[localIdx] + w[localIdx] + t[localIdx] + b[localIdx] + ne[localIdx] + nw[localIdx] + se[localIdx] + sw[localIdx] + nt[localIdx] + nb[localIdx] + st[localIdx] + sb[localIdx] + et[localIdx] + eb[localIdx] + wt[localIdx] + wb[localIdx];
	float ux = ((+e[localIdx] - w[localIdx] + ne[localIdx] - nw[localIdx] + se[localIdx] - sw[localIdx] + et[localIdx] + eb[localIdx] - wt[localIdx] - wb[localIdx])/rho)*(!(type & 0xff)) + 0.005*(type & 0xff);
	float uy = ((+n[localIdx] - s[localIdx] + ne[localIdx] + nw[localIdx] - se[localIdx] - sw[localIdx] + nt[localIdx] + nb[localIdx] - st[localIdx] - sb[localIdx])/rho)*(!(type & 0xff)) + 0.002*(type & 0xff);
	float uz = ((+t[localIdx] - b[localIdx] + nt[localIdx] - nb[localIdx] + st[localIdx] - sb[localIdx] + et[localIdx] - eb[localIdx] + wt[localIdx] - wb[localIdx])/rho)*(!(type & 0xff));

	float u2 = 1.5 *( ux * ux +  uy * uy + uz * uz);

	dGrid[idx] = (c[localIdx] * (type >> 8))	+ ((1.0 - OMEGA)* c[localIdx] + DFL1 * OMEGA * rho * (1.0 - u2))* (!(type >> 8));

	dGrid[c_nb1 + idx + offset] = (s[localIdx] * (type >> 8))  + ((1.0 - OMEGA)* n[localIdx] + DFL2 * OMEGA * rho * (1.0 + uy * (4.5 * uy + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nb2 + idx + 2*offset] = (n[localIdx] * (type >> 8)) + ((1.0 - OMEGA)* s[localIdx] + DFL2 * OMEGA * rho * (1.0 + uy * (4.5 * uy - 3.0) - u2))* (!(type >> 8));

	dGrid[c_nb3 + idx + 3*offset] = (w[localIdx] * (type >> 8)) + ((1.0 - OMEGA)* e[localIdx] + DFL2 * OMEGA * rho * (1.0 + ux * (4.5 * ux + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nb4 + idx + 4*offset] = (e[localIdx] * (type >> 8)) + ((1.0 - OMEGA)* w[localIdx] + DFL2 * OMEGA * rho * (1.0 + ux * (4.5 * ux - 3.0) - u2))* (!(type >> 8));

	dGrid[c_nb5 + idx + 5*offset] = (b[localIdx] * (type >> 8)) + ((1.0 - OMEGA)* t[localIdx] + DFL2 * OMEGA * rho * (1.0 + uz * (4.5 * uz + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nb6 + idx + 6*offset] = (t[localIdx] * (type >> 8)) + ((1.0 - OMEGA)* b[localIdx] + DFL2 * OMEGA * rho * (1.0 + uz * (4.5 * uz - 3.0) - u2))* (!(type >> 8));

	dGrid[c_nb7 + idx + 7*offset] = (sw[localIdx] * (type >> 8)) + ((1.0 - OMEGA)* ne[localIdx] + DFL3 * OMEGA * rho * (1.0 + (+ux + uy) * (4.5 * (+ux + uy) + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nb8 + idx + 8*offset] = (se[localIdx] * (type >> 8)) + ((1.0 - OMEGA)* nw[localIdx] + DFL3 * OMEGA * rho * (1.0 + (-ux + uy) * (4.5 * (-ux + uy) + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nb9 + idx + 9*offset] = (nw[localIdx] * (type >> 8)) + ((1.0 - OMEGA)* se[localIdx] + DFL3 * OMEGA * rho * (1.0 + (+ux - uy) * (4.5 * (+ux - uy) + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nb10 + idx + 10*offset] = (ne[localIdx] * (type >> 8))+ ((1.0 - OMEGA)* sw[localIdx] + DFL3 * OMEGA * rho * (1.0 + (-ux - uy) * (4.5 * (-ux - uy) + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nb11 + idx + 11*offset] = (sb[localIdx] * (type >> 8))+ ((1.0 - OMEGA)* nt[localIdx] + DFL3 * OMEGA * rho * (1.0 + (+uy + uz) * (4.5 * (+uy + uz) + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nb12 + idx + 12*offset] = (st[localIdx] * (type >> 8))+ ((1.0 - OMEGA)* nb[localIdx] + DFL3 * OMEGA * rho * (1.0 + (+uy - uz) * (4.5 * (+uy - uz) + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nb13 + idx + 13*offset] = (nb[localIdx] * (type >> 8))+ ((1.0 - OMEGA)* st[localIdx] + DFL3 * OMEGA * rho * (1.0 + (-uy + uz) * (4.5 * (-uy + uz) + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nb14 + idx + 14*offset] = (nt[localIdx] * (type >> 8))+ ((1.0 - OMEGA)* sb[localIdx] + DFL3 * OMEGA * rho * (1.0 + (-uy - uz) * (4.5 * (-uy - uz) + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nb15 + idx + 15*offset] = (wb[localIdx] * (type >> 8))+ ((1.0 - OMEGA)* et[localIdx] + DFL3 * OMEGA * rho * (1.0 + (+ux + uz) * (4.5 * (+ux + uz) + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nb16 + idx + 16*offset] = (wt[localIdx] * (type >> 8))+ ((1.0 - OMEGA)* eb[localIdx] + DFL3 * OMEGA * rho * (1.0 + (+ux - uz) * (4.5 * (+ux - uz) + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nb17 + idx + 17*offset] = (eb[localIdx] * (type >> 8))+ ((1.0 - OMEGA)* wt[localIdx] + DFL3 * OMEGA * rho * (1.0 + (-ux + uz) * (4.5 * (-ux + uz) + 3.0) - u2))* (!(type >> 8));

	dGrid[c_nb18 + idx + 18*offset] = (et[localIdx] * (type >> 8))+ ((1.0 - OMEGA)* wb[localIdx] + DFL3 * OMEGA * rho * (1.0 + (-ux - uz) * (4.5 * (-ux - uz) + 3.0) - u2))* (!(type >> 8));
*/
}
__global__ void lbm_kernel_partitioned(float* sGrid, float *dGrid, unsigned char* __restrict__ const flags, int sPos)
{
	__shared__ int offset;

		//__shared__ float one_minus_omega;
	offset = SIZE_XX*SIZE_YY*SIZE_ZZ;
	//one_minus_omega = 1.0 - OMEGA;
	//int x = threadIdx.x;
	//int y = blockIdx.x + 1;
	//int z = blockIdx.y + 1;

	//int ElementsPerBlock = blockDim.x;
	//float c, n, s, e, w, t, b, ne, nw, se, sw, nt, nb, st, sb, et,	eb, wt, wb;

	/*the grid is organized as follows:
	 * c c c c c c . . . | n n n n n n . . . | s s s s s s . . . | e e e e e e . . . | . . .
	 */

	//calculate the index
	//int idx = (z * SIZE_YY + y)*SIZE_XX + x;
	//int idx = ((blockIdx.y +1) * SIZE_YY + (blockIdx.x+1))*SIZE_XX + threadIdx.x;
	//int idx = CALC_INDEX_SOA_WO_STRUCT(x , y, z,0);
	int idx = CALC_INDEX_SOA_WO_STRUCT(threadIdx.x, blockIdx.x, blockIdx.y,0) + sPos;

	//each direction has offset = 130*100*100 values, we need to jump one offset to get new value
	//int offset = SIZE_XX*SIZE_YY*SIZE_ZZ;
	//save index to address for operating


	//int address = idx; //if(y==1 && x==0 && z==1) printf("\nc src(%d)", address);
	float c = sGrid[idx] ;
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\nn src(%d)", address);
	float n = sGrid[idx + offset] ;
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\ns src(%d)", address);
	float s = sGrid[idx + 2*offset] ;
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\ne src(%d)", address);
	float e = sGrid[idx + 3*offset] ;
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\nw src(%d)", address);
	float w = sGrid[idx + 4*offset] ;
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\nt src(%d)", address);
	float t = sGrid[idx + 5*offset] ;
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\nb src(%d)", address);
	float b = sGrid[idx + 6*offset] ;
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\nne src(%d)", address);
	float ne = sGrid[idx + 7*offset] ;
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\nnw src(%d)", address);
	float nw = sGrid[idx + 8*offset] ;
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\nse src(%d)", address);
	float se = sGrid[idx + 9*offset] ;
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\nsw src(%d)", address);
	float sw = sGrid[idx + 10*offset] ;
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\nnt src(%d)", address);
	float nt = sGrid[idx + 11*offset] ;
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\nnb src(%d)", address);
	float nb = sGrid[idx + 12*offset] ;
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\nst src(%d)", address);
	float st = sGrid[idx + 13*offset] ;
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\nsb src(%d)", address);
	float sb = sGrid[idx + 14*offset] ;
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\net src(%d)", address);
	float et = sGrid[idx + 15*offset] ;
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\neb src(%d)", address);
	float eb = sGrid[idx + 16*offset] ;
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\nwt src(%d)", address);
	float wt = sGrid[idx + 17*offset] ;
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\nwb src(%d)", address);
	float wb = sGrid[idx + 18*offset] ;


	//get the value final cell (20) to know the kind of cell: fluid or accel or obstacle
	//int flag = flags[idx];
	unsigned short type = ((flags[idx] == 227) << 8) | ((flags[idx] == 228) & 0xff);

	float rho = +c + n + s + e + w + t + b + ne + nw + se + sw + nt + nb + st + sb + et + eb + wt + wb;
	float ux = ((+e - w + ne - nw + se - sw + et + eb - wt - wb)/rho)*(!(type & 0xff)) + 0.005*(type & 0xff);
	float uy = ((+n - s + ne + nw - se - sw + nt + nb - st - sb)/rho)*(!(type & 0xff)) + 0.002*(type & 0xff);
	float uz = ((+t - b + nt - nb + st - sb + et - eb + wt - wb)/rho)*(!(type & 0xff));

	float u2 = 1.5 *( ux * ux +  uy * uy + uz * uz);

	dGrid[idx] = (c * (type >> 8))	+ ((1.0 - OMEGA)* c + DFL1 * OMEGA * rho * (1.0 - u2))* (!(type >> 8));

	c = (type >> 8); //resue variable c
	//address = offset;

	//dGrid[192 + idx + address] = (s * c)  + ((1.0 - OMEGA)* n + DFL2 * OMEGA * rho * (1.0 + uy * (4.5 * uy + 3.0) - u2))* (!c);
	dGrid[c_nb1 + idx + offset] = (s * c)  + ((1.0 - OMEGA)* n + DFL2 * OMEGA * rho * (1.0 + uy * (4.5 * uy + 3.0) - u2))* (!c);

	//dGrid[-192 + idx + 2*offset] = (n * c) + ((1.0 - OMEGA)* s + DFL2 * OMEGA * rho * (1.0 + uy * (4.5 * uy - 3.0) - u2))* (!c);
	dGrid[c_nb2 + idx + 2*offset] = (n * c) + ((1.0 - OMEGA)* s + DFL2 * OMEGA * rho * (1.0 + uy * (4.5 * uy - 3.0) - u2))* (!c);

	n = (1.0 - OMEGA); //resue variable n
	s = DFL2 * OMEGA * rho; //resue variable s

	//dGrid[1 + idx + 3*offset] = (w * c) + (n* e + s * (1.0 + ux * (4.5 * ux + 3.0) - u2))* (!c);
	dGrid[c_nb3 + idx + 3*offset] = (w * c) + (n* e + s * (1.0 + ux * (4.5 * ux + 3.0) - u2))* (!c);

	//dGrid[-1 + idx + 4*offset] = (e * c) + (n* w + s * (1.0 + ux * (4.5 * ux - 3.0) - u2))* (!c);
	dGrid[c_nb4 + idx + 4*offset] = (e * c) + (n* w + s * (1.0 + ux * (4.5 * ux - 3.0) - u2))* (!c);


	//dGrid[37056 + idx + 5*offset] = (b * c) + (n* t + s * (1.0 + uz * (4.5 * uz + 3.0) - u2))* (!c);
	dGrid[c_nb5 + idx + 5*offset] = (b * c) + (n* t + s * (1.0 + uz * (4.5 * uz + 3.0) - u2))* (!c);

	//dGrid[-37056 + idx + 6*offset] = (t * c) + (n* b + s * (1.0 + uz * (4.5 * uz - 3.0) - u2))* (!c);
	dGrid[c_nb6 + idx + 6*offset] = (t * c) + (n* b + s * (1.0 + uz * (4.5 * uz - 3.0) - u2))* (!c);

	b = DFL3 * OMEGA * rho; //resue variable b
	t = 1.0 - u2;
	e = ux + uy;
	w = b*(t+ 4.5*e*e);

	//dGrid[193 + idx + 7*offset] = (sw * c) + (n* ne + w + 3*e*b)* (!c);
	dGrid[c_nb7 + idx + 7*offset] = (sw * c) + (n* ne + w + 3*e*b)* (!c);

	//dGrid[-193 + idx + 10*offset] = (ne * c)+ (n* sw + w - 3*e*b)* (!c);
	dGrid[c_nb10 + idx + 10*offset] = (ne * c)+ (n* sw + w - 3*e*b)* (!c);

	e = -ux + uy;
	w = b*(t+ 4.5*e*e);

	//dGrid[191 + idx + 8*offset] = (se * c) + (n* nw + w + 3*e*b)* (!c);
	dGrid[c_nb8 + idx + 8*offset] = (se * c) + (n* nw + w + 3*e*b)* (!c);

	//dGrid[-191 + idx + 9*offset] = (nw * c) + (n* se + w - 3*e*b)* (!c);
	dGrid[c_nb9 + idx + 9*offset] = (nw * c) + (n* se + w - 3*e*b)* (!c);

	e = uy + uz;
	w = b*(t+ 4.5*e*e);
	//dGrid[37248 + idx + 11*offset] = (sb * c)+ (n* nt + w + 3*e*b)* (!c);
	dGrid[c_nb11 + idx + 11*offset] = (sb * c)+ (n* nt + w + 3*e*b)* (!c);

	//dGrid[-37248 + idx + 14*offset] = (nt * c)+ (n* sb + w - 3*e*b)* (!c);
	dGrid[c_nb14 + idx + 14*offset] = (nt * c)+ (n* sb + w - 3*e*b)* (!c);


	e = uy - uz;
	w = b*(t+ 4.5*e*e);
	//dGrid[-36864 + idx + 12*offset] = (st * c)+ (n* nb + w + 3*e*b)* (!c);
	dGrid[c_nb12 + idx + 12*offset] = (st * c)+ (n* nb + w + 3*e*b)* (!c);

	//dGrid[36864 + idx + 13*offset] = (nb * c)+ (n* st + w - 3*e*b)* (!c);
	dGrid[c_nb13 + idx + 13*offset] = (nb * c)+ (n* st + w - 3*e*b)* (!c);


	e = ux + uz;
	w = b*(t+ 4.5*e*e);

	//dGrid[37057 + idx + 15*offset] = (wb * c)+ (n* et + w + 3*e*b)* (!c);
	dGrid[c_nb15 + idx + 15*offset] = (wb * c)+ (n* et + w + 3*e*b)* (!c);

	//dGrid[-37057 + idx + 18*offset] = (et * c)+ (n* wb + w - 3*e*b)* (!c);
	dGrid[c_nb18 + idx + 18*offset] = (et * c)+ (n* wb + w - 3*e*b)* (!c);

	e = ux - uz;
	w = b*(t+ 4.5*e*e);

	//dGrid[-37055 + idx + 16*offset] = (wt * c)+ (n* eb + w + 3*e*b)* (!c);
	dGrid[c_nb16 + idx + 16*offset] = (wt * c)+ (n* eb + w + 3*e*b)* (!c);

	//dGrid[37055 + idx + 17*offset] = (eb * c)+ (n* wt + w - 3*e*b)* (!c);
	dGrid[c_nb17 + idx + 17*offset] = (eb * c)+ (n* wt + w - 3*e*b)* (!c);

}
__global__ void collide_cuda(float* sGrid, unsigned char* __restrict__ const flags)
{
	__shared__ int offset;

		//__shared__ float one_minus_omega;
	offset = SIZE_XX*SIZE_YY*SIZE_ZZ;
	//one_minus_omega = 1.0 - OMEGA;
	//int x = threadIdx.x;
	//int y = blockIdx.x + 1;
	//int z = blockIdx.y + 1;

	//int ElementsPerBlock = blockDim.x;
	//float c, n, s, e, w, t, b, ne, nw, se, sw, nt, nb, st, sb, et,	eb, wt, wb;

	/*the grid is organized as follows:
	 * c c c c c c . . . | n n n n n n . . . | s s s s s s . . . | e e e e e e . . . | . . .
	 */

	//calculate the index
	//int idx = (z * SIZE_YY + y)*SIZE_XX + x;
	//int idx = ((blockIdx.y +1) * SIZE_YY + (blockIdx.x+1))*SIZE_XX + threadIdx.x;
	//int idx = CALC_INDEX_SOA_WO_STRUCT(x , y, z,0);
	int idx = CALC_INDEX_SOA_WO_STRUCT(threadIdx.x, (blockIdx.x+1), (blockIdx.y+1),0);

	//each direction has offset = 130*100*100 values, we need to jump one offset to get new value
	//int offset = SIZE_XX*SIZE_YY*SIZE_ZZ;
	//save index to address for operating


	//int address = idx; //if(y==1 && x==0 && z==1) printf("\nc src(%d)", address);
	float c = sGrid[idx] ;
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\nn src(%d)", address);
	float n = sGrid[idx + offset] ;
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\ns src(%d)", address);
	float s = sGrid[idx + 2*offset] ;
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\ne src(%d)", address);
	float e = sGrid[idx + 3*offset] ;
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\nw src(%d)", address);
	float w = sGrid[idx + 4*offset] ;
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\nt src(%d)", address);
	float t = sGrid[idx + 5*offset] ;
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\nb src(%d)", address);
	float b = sGrid[idx + 6*offset] ;
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\nne src(%d)", address);
	float ne = sGrid[idx + 7*offset] ;
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\nnw src(%d)", address);
	float nw = sGrid[idx + 8*offset] ;
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\nse src(%d)", address);
	float se = sGrid[idx + 9*offset] ;
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\nsw src(%d)", address);
	float sw = sGrid[idx + 10*offset] ;
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\nnt src(%d)", address);
	float nt = sGrid[idx + 11*offset] ;
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\nnb src(%d)", address);
	float nb = sGrid[idx + 12*offset] ;
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\nst src(%d)", address);
	float st = sGrid[idx + 13*offset] ;
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\nsb src(%d)", address);
	float sb = sGrid[idx + 14*offset] ;
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\net src(%d)", address);
	float et = sGrid[idx + 15*offset] ;
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\neb src(%d)", address);
	float eb = sGrid[idx + 16*offset] ;
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\nwt src(%d)", address);
	float wt = sGrid[idx + 17*offset] ;
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\nwb src(%d)", address);
	float wb = sGrid[idx + 18*offset] ;


	//get the value final cell (20) to know the kind of cell: fluid or accel or obstacle
	//int flag = flags[idx];
	unsigned short type = ((flags[idx] == 1) << 8) | ((flags[idx] == 2) & 0xff);

	float rho = +c + n + s + e + w + t + b + ne + nw + se + sw + nt + nb + st + sb + et + eb + wt + wb;
	float ux = ((+e - w + ne - nw + se - sw + et + eb - wt - wb)/rho)*(!(type & 0xff)) + 0.005*(type & 0xff);
	float uy = ((+n - s + ne + nw - se - sw + nt + nb - st - sb)/rho)*(!(type & 0xff)) + 0.002*(type & 0xff);
	float uz = ((+t - b + nt - nb + st - sb + et - eb + wt - wb)/rho)*(!(type & 0xff));

	float u2 = 1.5 *( ux * ux +  uy * uy + uz * uz);

	sGrid[idx] = (c * (type >> 8))	+ ((1.0 - OMEGA)* c + DFL1 * OMEGA * rho * (1.0 - u2))* (!(type >> 8));

	c = (type >> 8); //resue variable c
	//address = offset;

	sGrid[idx + offset] = (s * c)  + ((1.0 - OMEGA)* n + DFL2 * OMEGA * rho * (1.0 + uy * (4.5 * uy + 3.0) - u2))* (!c);

	sGrid[idx + 2*offset] = (n * c) + ((1.0 - OMEGA)* s + DFL2 * OMEGA * rho * (1.0 + uy * (4.5 * uy - 3.0) - u2))* (!c);

	n = (1.0 - OMEGA); //resue variable n
	s = DFL2 * OMEGA * rho; //resue variable s

	sGrid[idx + 3*offset] = (w * c) + (n* e + s * (1.0 + ux * (4.5 * ux + 3.0) - u2))* (!c);

	sGrid[idx + 4*offset] = (e * c) + (n* w + s * (1.0 + ux * (4.5 * ux - 3.0) - u2))* (!c);

	sGrid[idx + 5*offset] = (b * c) + (n* t + s * (1.0 + uz * (4.5 * uz + 3.0) - u2))* (!c);

	sGrid[c_nb6 + idx + 6*offset] = (t * c) + (n* b + s * (1.0 + uz * (4.5 * uz - 3.0) - u2))* (!c);

	b = DFL3 * OMEGA * rho; //resue variable b
	t = 1.0 - u2;
	e = ux + uy;
	w = b*(t+ 4.5*e*e);

	sGrid[idx + 7*offset] = (sw * c) + (n* ne + w + 3*e*b)* (!c);

	sGrid[idx + 10*offset] = (ne * c)+ (n* sw + w - 3*e*b)* (!c);

	e = -ux + uy;
	w = b*(t+ 4.5*e*e);

	sGrid[idx + 8*offset] = (se * c) + (n* nw + w + 3*e*b)* (!c);

	sGrid[idx + 9*offset] = (nw * c) + (n* se + w - 3*e*b)* (!c);

	e = uy + uz;
	w = b*(t+ 4.5*e*e);

	sGrid[idx + 11*offset] = (sb * c)+ (n* nt + w + 3*e*b)* (!c);

	sGrid[idx + 14*offset] = (nt * c)+ (n* sb + w - 3*e*b)* (!c);

	e = uy - uz;
	w = b*(t+ 4.5*e*e);

	sGrid[idx + 12*offset] = (st * c)+ (n* nb + w + 3*e*b)* (!c);

	sGrid[idx + 13*offset] = (nb * c)+ (n* st + w - 3*e*b)* (!c);

	e = ux + uz;
	w = b*(t+ 4.5*e*e);

	sGrid[idx + 15*offset] = (wb * c)+ (n* et + w + 3*e*b)* (!c);

	sGrid[idx + 18*offset] = (et * c)+ (n* wb + w - 3*e*b)* (!c);

	e = ux - uz;
	w = b*(t+ 4.5*e*e);

	sGrid[idx + 16*offset] = (wt * c)+ (n* eb + w + 3*e*b)* (!c);

	sGrid[idx + 17*offset] = (eb * c)+ (n* wt + w - 3*e*b)* (!c);
}
__global__ void stream_cuda(float *sGrid, float *dGrid)
{
	__shared__ int offset ;
	offset = SIZE_XX*SIZE_YY*SIZE_ZZ;
	int idx = CALC_INDEX_SOA_WO_STRUCT(threadIdx.x, (blockIdx.x+1), (blockIdx.y+1),0);

	dGrid[idx] = sGrid[idx];

	dGrid[c_nb1 + idx + offset] = sGrid[idx + offset];

	dGrid[c_nb2 + idx + 2*offset] = sGrid[idx + 2*offset];

	dGrid[c_nb3 + idx + 3*offset] = sGrid[idx + 3*offset];

	dGrid[c_nb4 + idx + 4*offset] = sGrid[idx + 4*offset];

	dGrid[c_nb5 + idx + 5*offset] = sGrid[idx + 5*offset];

	dGrid[c_nb6 + idx + 6*offset] = sGrid[idx + 6*offset];

	dGrid[c_nb7 + idx + 7*offset] = sGrid[idx + 7*offset];

	dGrid[c_nb10 + idx + 10*offset] = sGrid[idx + 10*offset];

	dGrid[c_nb8 + idx + 8*offset] = sGrid[idx + 8*offset];

	dGrid[c_nb9 + idx + 9*offset] = sGrid[idx + 9*offset];

	dGrid[c_nb11 + idx + 11*offset] = sGrid[idx + 11*offset];

	dGrid[c_nb14 + idx + 14*offset] = sGrid[idx + 14*offset];

	dGrid[c_nb12 + idx + 12*offset] = sGrid[idx + 12*offset];

	dGrid[c_nb13 + idx + 13*offset] = sGrid[idx + 13*offset];

	dGrid[c_nb15 + idx + 15*offset] = sGrid[idx + 15*offset];

	dGrid[c_nb18 + idx + 18*offset] = sGrid[idx + 18*offset];

	dGrid[c_nb16 + idx + 16*offset] = sGrid[idx + 16*offset];

	dGrid[c_nb17 + idx + 17*offset] = sGrid[idx + 17*offset];
}

//reused registers => reduce fp_64 => reduce time
__global__ void /*__launch_bounds__(128,8)*/ lbm_kernel_SoA_reuse_reg(float* __restrict__ const sGrid, float *dGrid, unsigned char* __restrict__ const flags)
{
	__shared__ int offset;

		//__shared__ float one_minus_omega;
	offset = SIZE_XX*SIZE_YY*SIZE_ZZ;
	//one_minus_omega = 1.0 - OMEGA;
	//int x = threadIdx.x;
	//int y = blockIdx.x + 1;
	//int z = blockIdx.y + 1;

	//int ElementsPerBlock = blockDim.x;
	//float c, n, s, e, w, t, b, ne, nw, se, sw, nt, nb, st, sb, et,	eb, wt, wb;

	/*the grid is organized as follows:
	 * c c c c c c . . . | n n n n n n . . . | s s s s s s . . . | e e e e e e . . . | . . .
	 */

	//calculate the index
	//int idx = (z * SIZE_YY + y)*SIZE_XX + x;
	//int idx = ((blockIdx.y +1) * SIZE_YY + (blockIdx.x+1))*SIZE_XX + threadIdx.x;
	//int idx = CALC_INDEX_SOA_WO_STRUCT(x , y, z,0);
	int idx = CALC_INDEX_SOA_WO_STRUCT(threadIdx.x , (blockIdx.x+1), (blockIdx.y+1),0);

	//each direction has offset = 130*100*100 values, we need to jump one offset to get new value
	//int offset = SIZE_XX*SIZE_YY*SIZE_ZZ;
	//save index to address for operating


	//int address = idx; //if(y==1 && x==0 && z==1) printf("\nc src(%d)", address);
	float c = sGrid[idx] ;
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\nn src(%d)", address);
	float n = sGrid[idx + offset] ;
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\ns src(%d)", address);
	float s = sGrid[idx + 2*offset] ;
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\ne src(%d)", address);
	float e = sGrid[idx + 3*offset] ;
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\nw src(%d)", address);
	float w = sGrid[idx + 4*offset] ;
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\nt src(%d)", address);
	float t = sGrid[idx + 5*offset] ;
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\nb src(%d)", address);
	float b = sGrid[idx + 6*offset] ;
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\nne src(%d)", address);
	float ne = sGrid[idx + 7*offset] ;
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\nnw src(%d)", address);
	float nw = sGrid[idx + 8*offset] ;
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\nse src(%d)", address);
	float se = sGrid[idx + 9*offset] ;
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\nsw src(%d)", address);
	float sw = sGrid[idx + 10*offset] ;
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\nnt src(%d)", address);
	float nt = sGrid[idx + 11*offset] ;
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\nnb src(%d)", address);
	float nb = sGrid[idx + 12*offset] ;
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\nst src(%d)", address);
	float st = sGrid[idx + 13*offset] ;
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\nsb src(%d)", address);
	float sb = sGrid[idx + 14*offset] ;
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\net src(%d)", address);
	float et = sGrid[idx + 15*offset] ;
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\neb src(%d)", address);
	float eb = sGrid[idx + 16*offset] ;
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\nwt src(%d)", address);
	float wt = sGrid[idx + 17*offset] ;
	//address = address + offset; //if(y==1 && x==0 && z==1) printf("\nwb src(%d)", address);
	float wb = sGrid[idx + 18*offset] ;


	//get the value final cell (20) to know the kind of cell: fluid or accel or obstacle
	//int flag = flags[idx];
	unsigned short type = ((flags[idx] == 1) << 8) | ((flags[idx] == 2) & 0xff);

	float rho = +c + n + s + e + w + t + b + ne + nw + se + sw + nt + nb + st + sb + et + eb + wt + wb;
	float ux = ((+e - w + ne - nw + se - sw + et + eb - wt - wb)/rho)*(!(type & 0xff)) + 0.005*(type & 0xff);
	float uy = ((+n - s + ne + nw - se - sw + nt + nb - st - sb)/rho)*(!(type & 0xff)) + 0.002*(type & 0xff);
	float uz = ((+t - b + nt - nb + st - sb + et - eb + wt - wb)/rho)*(!(type & 0xff));

	float u2 = 1.5 *( ux * ux +  uy * uy + uz * uz);

	dGrid[idx] = (c * (type >> 8))	+ ((1.0 - OMEGA)* c + DFL1 * OMEGA * rho * (1.0 - u2))* (!(type >> 8));

	c = (type >> 8); //resue variable c
	//address = offset;

	//dGrid[192 + idx + address] = (s * c)  + ((1.0 - OMEGA)* n + DFL2 * OMEGA * rho * (1.0 + uy * (4.5 * uy + 3.0) - u2))* (!c);
	dGrid[c_nb1 + idx + offset] = (s * c)  + ((1.0 - OMEGA)* n + DFL2 * OMEGA * rho * (1.0 + uy * (4.5 * uy + 3.0) - u2))* (!c);

	//dGrid[-192 + idx + 2*offset] = (n * c) + ((1.0 - OMEGA)* s + DFL2 * OMEGA * rho * (1.0 + uy * (4.5 * uy - 3.0) - u2))* (!c);
	dGrid[c_nb2 + idx + 2*offset] = (n * c) + ((1.0 - OMEGA)* s + DFL2 * OMEGA * rho * (1.0 + uy * (4.5 * uy - 3.0) - u2))* (!c);

	n = (1.0 - OMEGA); //resue variable n
	s = DFL2 * OMEGA * rho; //resue variable s

	//dGrid[1 + idx + 3*offset] = (w * c) + (n* e + s * (1.0 + ux * (4.5 * ux + 3.0) - u2))* (!c);
	dGrid[c_nb3 + idx + 3*offset] = (w * c) + (n* e + s * (1.0 + ux * (4.5 * ux + 3.0) - u2))* (!c);

	//dGrid[-1 + idx + 4*offset] = (e * c) + (n* w + s * (1.0 + ux * (4.5 * ux - 3.0) - u2))* (!c);
	dGrid[c_nb4 + idx + 4*offset] = (e * c) + (n* w + s * (1.0 + ux * (4.5 * ux - 3.0) - u2))* (!c);


	//dGrid[37056 + idx + 5*offset] = (b * c) + (n* t + s * (1.0 + uz * (4.5 * uz + 3.0) - u2))* (!c);
	dGrid[c_nb5 + idx + 5*offset] = (b * c) + (n* t + s * (1.0 + uz * (4.5 * uz + 3.0) - u2))* (!c);

	//dGrid[-37056 + idx + 6*offset] = (t * c) + (n* b + s * (1.0 + uz * (4.5 * uz - 3.0) - u2))* (!c);
	dGrid[c_nb6 + idx + 6*offset] = (t * c) + (n* b + s * (1.0 + uz * (4.5 * uz - 3.0) - u2))* (!c);

	b = DFL3 * OMEGA * rho; //resue variable b
	t = 1.0 - u2;
	e = ux + uy;
	w = b*(t+ 4.5*e*e);

	//dGrid[193 + idx + 7*offset] = (sw * c) + (n* ne + w + 3*e*b)* (!c);
	dGrid[c_nb7 + idx + 7*offset] = (sw * c) + (n* ne + w + 3*e*b)* (!c);

	//dGrid[-193 + idx + 10*offset] = (ne * c)+ (n* sw + w - 3*e*b)* (!c);
	dGrid[c_nb10 + idx + 10*offset] = (ne * c)+ (n* sw + w - 3*e*b)* (!c);

	e = -ux + uy;
	w = b*(t+ 4.5*e*e);

	//dGrid[191 + idx + 8*offset] = (se * c) + (n* nw + w + 3*e*b)* (!c);
	dGrid[c_nb8 + idx + 8*offset] = (se * c) + (n* nw + w + 3*e*b)* (!c);

	//dGrid[-191 + idx + 9*offset] = (nw * c) + (n* se + w - 3*e*b)* (!c);
	dGrid[c_nb9 + idx + 9*offset] = (nw * c) + (n* se + w - 3*e*b)* (!c);

	e = uy + uz;
	w = b*(t+ 4.5*e*e);
	//dGrid[37248 + idx + 11*offset] = (sb * c)+ (n* nt + w + 3*e*b)* (!c);
	dGrid[c_nb11 + idx + 11*offset] = (sb * c)+ (n* nt + w + 3*e*b)* (!c);

	//dGrid[-37248 + idx + 14*offset] = (nt * c)+ (n* sb + w - 3*e*b)* (!c);
	dGrid[c_nb14 + idx + 14*offset] = (nt * c)+ (n* sb + w - 3*e*b)* (!c);


	e = uy - uz;
	w = b*(t+ 4.5*e*e);
	//dGrid[-36864 + idx + 12*offset] = (st * c)+ (n* nb + w + 3*e*b)* (!c);
	dGrid[c_nb12 + idx + 12*offset] = (st * c)+ (n* nb + w + 3*e*b)* (!c);

	//dGrid[36864 + idx + 13*offset] = (nb * c)+ (n* st + w - 3*e*b)* (!c);
	dGrid[c_nb13 + idx + 13*offset] = (nb * c)+ (n* st + w - 3*e*b)* (!c);


	e = ux + uz;
	w = b*(t+ 4.5*e*e);

	//dGrid[37057 + idx + 15*offset] = (wb * c)+ (n* et + w + 3*e*b)* (!c);
	dGrid[c_nb15 + idx + 15*offset] = (wb * c)+ (n* et + w + 3*e*b)* (!c);

	//dGrid[-37057 + idx + 18*offset] = (et * c)+ (n* wb + w - 3*e*b)* (!c);
	dGrid[c_nb18 + idx + 18*offset] = (et * c)+ (n* wb + w - 3*e*b)* (!c);

	e = ux - uz;
	w = b*(t+ 4.5*e*e);

	//dGrid[-37055 + idx + 16*offset] = (wt * c)+ (n* eb + w + 3*e*b)* (!c);
	dGrid[c_nb16 + idx + 16*offset] = (wt * c)+ (n* eb + w + 3*e*b)* (!c);

	//dGrid[37055 + idx + 17*offset] = (eb * c)+ (n* wt + w - 3*e*b)* (!c);
	dGrid[c_nb17 + idx + 17*offset] = (eb * c)+ (n* wt + w - 3*e*b)* (!c);

}
__device__ float mul(const float3 c, const float3 u)
{
	return (c.x)*(u.x) + (c.y)*(u.y) + (c.z)*(u.z);
}

//__device__ float mul(const float3 *c, const float3 *u)
//{
//	return (*c).x*(*u).x + (*c).y*(*u).y + (*c).z*(*u).z;
//}
//__device__ float operator*(const float3 *c, const float3 *u)
//{
//	return c.x*u.x + c.y*u.y + c.z*u.z;
//}
//__device__ float eq_func(int pos, float3 u, float u2)
//{
//	return 1.0 + 3.0*mul(c_cArray[1],u) + 4.5*mul(c_cArray[1],u)*mul(c_cArray[1],u) - u2;
//}

/*__device__ void stream(float *dGrid, float c, float n, float s, float e, float w, float t,
					   float b, float ne, float nw, float se, float sw, float nt,
					   float nb, float st, float sb, float et, float eb, float wt, float wb,
					   float rho, float u2, float3 u, int type, int idx, int offset)
{

	dGrid[c_nbArray[0] + idx] = (c * type)	+ ((cOneOmega)* c + cDFL1Omega * rho * (1.0 - u2))* (!type);
	float temp = mul(c_cArray[1],u);
	dGrid[c_nbArray[1] + idx + offset] = (s * type)  + ((cOneOmega)* n + cDFL2Omega * rho * (1.0 + (3.0 + 4.5*temp)*temp - u2))* (!type);
	temp = mul(c_cArray[2],u);
	dGrid[c_nbArray[2] + idx + 2*offset] = (n * type) + ((cOneOmega)* s + cDFL2Omega * rho * (1.0 + (3.0 + 4.5*temp)*temp - u2))* (!type);
	temp = mul(c_cArray[3],u);
	dGrid[c_nbArray[3] + idx + 3*offset] = (w * type) + ((cOneOmega)* e + cDFL2Omega * rho * (1.0 + (3.0 + 4.5*temp)*temp - u2))* (!type);
	temp = mul(c_cArray[4],u);
	dGrid[c_nbArray[4] + idx + 4*offset] = (e * type) + ((cOneOmega)* w + cDFL2Omega * rho * (1.0 + (3.0 + 4.5*temp)*temp - u2))* (!type);
	temp = mul(c_cArray[5],u);
	dGrid[c_nbArray[5] + idx + 5*offset] = (b * type) + ((cOneOmega)* t + cDFL2Omega * rho * (1.0 + (3.0 + 4.5*temp)*temp - u2))* (!type);
	temp = mul(c_cArray[6],u);
	dGrid[c_nbArray[6] + idx + 6*offset] = (t * type) + ((cOneOmega)* b + cDFL2Omega * rho * (1.0 + (3.0 + 4.5*temp)*temp - u2))* (!type);
	temp = mul(c_cArray[7],u);
	dGrid[c_nbArray[7] + idx + 7*offset] = (sw * type) + ((cOneOmega)* ne + cDFL3Omega * rho * (1.0 + (3.0 + 4.5*temp)*temp - u2))* (!type);
	temp = mul(c_cArray[8],u);
	dGrid[c_nbArray[8] + idx + 8*offset] = (se * type) + ((cOneOmega)* nw + cDFL3Omega * rho * (1.0 + (3.0 + 4.5*temp)*temp - u2))* (!type);
	temp = mul(c_cArray[9],u);
	dGrid[c_nbArray[9] + idx + 9*offset] = (nw * type) + ((cOneOmega)* se + cDFL3Omega * rho * (1.0 + (3.0 + 4.5*temp)*temp - u2))* (!type);
	temp = mul(c_cArray[10],u);
	dGrid[c_nbArray[10] + idx + 10*offset] = (ne * type)+ ((cOneOmega)* sw + cDFL3Omega * rho * (1.0 + (3.0 + 4.5*temp)*temp - u2))* (!type);
	temp = mul(c_cArray[11],u);
	dGrid[c_nbArray[11] + idx + 11*offset] = (sb * type)+ ((cOneOmega)* nt + cDFL3Omega * rho * (1.0 + (3.0 + 4.5*temp)*temp - u2))* (!type);
	temp = mul(c_cArray[12],u);
	dGrid[c_nbArray[12] + idx + 12*offset] = (st * type)+ ((cOneOmega)* nb + cDFL3Omega * rho * (1.0 + (3.0 + 4.5*temp)*temp - u2))* (!type);
	temp = mul(c_cArray[13],u);
	dGrid[c_nbArray[13] + idx + 13*offset] = (nb * type)+ ((cOneOmega)* st + cDFL3Omega * rho * (1.0 + (3.0 + 4.5*temp)*temp - u2))* (!type);
	temp = mul(c_cArray[14],u);
	dGrid[c_nbArray[14] + idx + 14*offset] = (nt * type)+ ((cOneOmega)* sb + cDFL3Omega * rho * (1.0 + (3.0 + 4.5*temp)*temp - u2))* (!type);
	temp = mul(c_cArray[15],u);
	dGrid[c_nbArray[15] + idx + 15*offset] = (wb * type)+ ((cOneOmega)* et + cDFL3Omega * rho * (1.0 + (3.0 + 4.5*temp)*temp - u2))* (!type);
	temp = mul(c_cArray[16],u);
	dGrid[c_nbArray[16] + idx + 16*offset] = (wt * type)+ ((cOneOmega)* eb + cDFL3Omega * rho * (1.0 + (3.0 + 4.5*temp)*temp - u2))* (!type);
	temp = mul(c_cArray[17],u);
	dGrid[c_nbArray[17] + idx + 17*offset] = (eb * type)+ ((cOneOmega)* wt + cDFL3Omega * rho * (1.0 + (3.0 + 4.5*temp)*temp - u2))* (!type);
	temp = mul(c_cArray[18],u);
	dGrid[c_nbArray[18] + idx + 18*offset] = (et * type)+ ((cOneOmega)* wb + cDFL3Omega * rho * (1.0 + (3.0 + 4.5*temp)*temp - u2))* (!type);
}*/
//move some values to constant memory. However, it looks slower than normal way
//__global__ void /*__launch_bounds__(128,8)*/ lbm_kernel_SoA_const(float* __restrict__ const sGrid, float *dGrid, unsigned char* __restrict__ const flags)
//{
//	__shared__ int offset;
//
//	//__shared__ float one_minus_omega;
//	offset = SIZE_XX*SIZE_YY*SIZE_ZZ;
//	//one_minus_omega = 1.0 - OMEGA;
//	int x = threadIdx.x;
//	int y = blockIdx.x + 1;
//	int z = blockIdx.y + 1;
//
//	//int ElementsPerBlock = blockDim.x;
//	//float c, n, s, e, w, t, b, ne, nw, se, sw, nt, nb, st, sb, et,	eb, wt, wb;
//
//	/*the grid is organized as follows:
//	 * c c c c c c . . . | n n n n n n . . . | s s s s s s . . . | e e e e e e . . . | . . .
//	 */
//
//	//calculate the index
//	//int idx = (z * SIZE_YY + y)*SIZE_XX + x;
//	//int idx = ((blockIdx.y +1) * SIZE_YY + (blockIdx.x+1))*SIZE_XX + threadIdx.x;
//	int idx = CALC_INDEX_SOA_WO_STRUCT(x,y,z,0);
//
//	//each direction has offset = 130*100*100 values, we need to jump one offset to get new value
//	//int offset = SIZE_XX*SIZE_YY*SIZE_ZZ;
//	//save index to address for operating
//
//	int address = idx; //if(y==1 && x==0 && z==1) printf("\nc src(%d)", address);
//	float c = sGrid[address];
//	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nn src(%d)", address);
//	float n = sGrid[address];
//	address = address + offset; //if(y==1 && x==0 && z==1) printf("\ns src(%d)", address);
//	float s = sGrid[address];
//	address = address + offset; //if(y==1 && x==0 && z==1) printf("\ne src(%d)", address);
//	float e = sGrid[address];
//	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nw src(%d)", address);
//	float w = sGrid[address];
//	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nt src(%d)", address);
//	float t = sGrid[address];
//	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nb src(%d)", address);
//	float b = sGrid[address];
//	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nne src(%d)", address);
//	float ne = sGrid[address];
//	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nnw src(%d)", address);
//	float nw = sGrid[address];
//	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nse src(%d)", address);
//	float se = sGrid[address];
//	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nsw src(%d)", address);
//	float sw = sGrid[address];
//	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nnt src(%d)", address);
//	float nt = sGrid[address];
//	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nnb src(%d)", address);
//	float nb = sGrid[address];
//	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nst src(%d)", address);
//	float st = sGrid[address];
//	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nsb src(%d)", address);
//	float sb = sGrid[address];
//	address = address + offset; //if(y==1 && x==0 && z==1) printf("\net src(%d)", address);
//	float et = sGrid[address];
//	address = address + offset; //if(y==1 && x==0 && z==1) printf("\neb src(%d)", address);
//	float eb = sGrid[address];
//	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nwt src(%d)", address);
//	float wt = sGrid[address];
//	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nwb src(%d)", address);
//	float wb = sGrid[address];
//
//	//get the value final cell (20) to know the kind of cell: fluid or accel or obstacle
//	//int flag = flags[idx];
//	unsigned short type = ((flags[idx] == 1) << 8) | ((flags[idx] == 2) & 0xff);
//	//int isobs = (flags[idx] == 1);
//	//int isacc = (flags[idx] == 2);
//
//	float rho = +c + n + s + e + w + t + b + ne + nw + se + sw + nt + nb + st + sb + et + eb + wt + wb;
//
//	float ux = ((+e - w + ne - nw + se - sw + et + eb - wt - wb)/rho)*(!(type & 0xff)) + 0.005*(type & 0xff);
//	float uy = ((+n - s + ne + nw - se - sw + nt + nb - st - sb)/rho)*(!(type & 0xff)) + 0.002*(type & 0xff);
//	float uz = ((+t - b + nt - nb + st - sb + et - eb + wt - wb)/rho)*(!(type & 0xff));
//
////	float ux = +e - w + ne - nw + se - sw + et + eb - wt - wb;
////	float uy = +n - s + ne + nw - se - sw + nt + nb - st - sb;
////	float uz = +t - b + nt - nb + st - sb + et - eb + wt - wb;
//
//	/*float3 u = make_float3(((+e - w + ne - nw + se - sw + et + eb - wt - wb)/rho)*(!(type & 0xff)) + 0.005*(type & 0xff),
//				((+n - s + ne + nw - se - sw + nt + nb - st - sb)/rho)*(!(type & 0xff)) + 0.002*(type & 0xff),
//				((+t - b + nt - nb + st - sb + et - eb + wt - wb)/rho)*(!(type & 0xff)));
//	*/
//	float u2 = 1.5 * ux * ux + 1.5* uy * uy + 1.5* uz * uz;
//
//	//float u2 = 1.5*u.x*u.x + 1.5*u.y*u.y + 1.5*u.z*u.z;
//	//float u2= U2(ux,uy,uz); //similar to regular usage
////	float u2 = u2_func(ux,uy,uz); //number of registers increases when using inline function
//
//
//	dGrid[c_nbArray[0] + idx] = (c * (type >> 8))	+ ((cOneOmega)* c + cDFL1Omega * rho * (1.0 - u2))* (!(type >> 8));
//
//	dGrid[c_nbArray[1] + idx + offset] = (s * (type >> 8))  + ((cOneOmega)* n + cDFL2Omega * rho * (1.0 + uy * (4.5 * uy + 3.0) - u2))* (!(type >> 8));
//
//	dGrid[c_nbArray[2] + idx + 2*offset] = (n * (type >> 8)) + ((cOneOmega)* s + cDFL2Omega * rho * (1.0 + uy * (4.5 * uy - 3.0) - u2))* (!(type >> 8));
//
//	dGrid[c_nbArray[3] + idx + 3*offset] = (w * (type >> 8)) + ((cOneOmega)* e + cDFL2Omega * rho * (1.0 + ux * (4.5 * ux + 3.0) - u2))* (!(type >> 8));
//
//	dGrid[c_nbArray[4] + idx + 4*offset] = (e * (type >> 8)) + ((cOneOmega)* w + cDFL2Omega * rho * (1.0 + ux * (4.5 * ux - 3.0) - u2))* (!(type >> 8));
//
//	dGrid[c_nbArray[5] + idx + 5*offset] = (b * (type >> 8)) + ((cOneOmega)* t + cDFL2Omega * rho * (1.0 + uz * (4.5 * uz + 3.0) - u2))* (!(type >> 8));
//
//	dGrid[c_nbArray[6] + idx + 6*offset] = (t * (type >> 8)) + ((cOneOmega)* b + cDFL2Omega * rho * (1.0 + uz * (4.5 * uz - 3.0) - u2))* (!(type >> 8));
//
//	dGrid[c_nbArray[7] + idx + 7*offset] = (sw * (type >> 8)) + ((cOneOmega)* ne + cDFL3Omega * rho * (1.0 + (+ux + uy) * (4.5 * (+ux + uy) + 3.0) - u2))* (!(type >> 8));
//
//	dGrid[c_nbArray[8] + idx + 8*offset] = (se * (type >> 8)) + ((cOneOmega)* nw + cDFL3Omega * rho * (1.0 + (-ux + uy) * (4.5 * (-ux + uy) + 3.0) - u2))* (!(type >> 8));
//
//	dGrid[c_nbArray[9] + idx + 9*offset] = (nw * (type >> 8)) + ((cOneOmega)* se + cDFL3Omega * rho * (1.0 + (+ux - uy) * (4.5 * (+ux - uy) + 3.0) - u2))* (!(type >> 8));
//
//	dGrid[c_nbArray[10] + idx + 10*offset] = (ne * (type >> 8))+ ((cOneOmega)* sw + cDFL3Omega * rho * (1.0 + (-ux - uy) * (4.5 * (-ux - uy) + 3.0) - u2))* (!(type >> 8));
//
//	dGrid[c_nbArray[11] + idx + 11*offset] = (sb * (type >> 8))+ ((cOneOmega)* nt + cDFL3Omega * rho * (1.0 + (+uy + uz) * (4.5 * (+uy + uz) + 3.0) - u2))* (!(type >> 8));
//
//	dGrid[c_nbArray[12] + idx + 12*offset] = (st * (type >> 8))+ ((cOneOmega)* nb + cDFL3Omega * rho * (1.0 + (+uy - uz) * (4.5 * (+uy - uz) + 3.0) - u2))* (!(type >> 8));
//
//	dGrid[c_nbArray[13] + idx + 13*offset] = (nb * (type >> 8))+ ((cOneOmega)* st + cDFL3Omega * rho * (1.0 + (-uy + uz) * (4.5 * (-uy + uz) + 3.0) - u2))* (!(type >> 8));
//
//	dGrid[c_nbArray[14] + idx + 14*offset] = (nt * (type >> 8))+ ((cOneOmega)* sb + cDFL3Omega * rho * (1.0 + (-uy - uz) * (4.5 * (-uy - uz) + 3.0) - u2))* (!(type >> 8));
//
//	dGrid[c_nbArray[15] + idx + 15*offset] = (wb * (type >> 8))+ ((cOneOmega)* et + cDFL3Omega * rho * (1.0 + (+ux + uz) * (4.5 * (+ux + uz) + 3.0) - u2))* (!(type >> 8));
//
//	dGrid[c_nbArray[16] + idx + 16*offset] = (wt * (type >> 8))+ ((cOneOmega)* eb + cDFL3Omega * rho * (1.0 + (+ux - uz) * (4.5 * (+ux - uz) + 3.0) - u2))* (!(type >> 8));
//
//	dGrid[c_nbArray[17] + idx + 17*offset] = (eb * (type >> 8))+ ((cOneOmega)* wt + cDFL3Omega * rho * (1.0 + (-ux + uz) * (4.5 * (-ux + uz) + 3.0) - u2))* (!(type >> 8));
//
//	dGrid[c_nbArray[18] + idx + 18*offset] = (et * (type >> 8))+ ((cOneOmega)* wb + cDFL3Omega * rho * (1.0 + (-ux - uz) * (4.5 * (-ux - uz) + 3.0) - u2))* (!(type >> 8));
//
//
//	/*stream(dGrid,c, n,  s,  e,  w,  t,
//						    b,  ne,  nw,  se,  sw,  nt,
//						    nb,  st,  sb,  et,  eb,  wt,  wb,
//						    rho,  u2, u, (type >> 8), idx, offset);
//	*/
//}


__global__ void /*__launch_bounds__(128,8)*/ lbm_kernel_SoA_Branch(float *sGrid, float *dGrid, unsigned char *flags)
{
	__shared__ int offset;

	//__shared__ float one_minus_omega;
	offset = SIZE_XX*SIZE_YY*SIZE_ZZ;
	//one_minus_omega = 1.0 - OMEGA;
	int x = threadIdx.x;
	int y = blockIdx.x + 1;
	int z = blockIdx.y + 1;

	//int ElementsPerBlock = blockDim.x;
	//float c, n, s, e, w, t, b, ne, nw, se, sw, nt, nb, st, sb, et,	eb, wt, wb;

	/*the grid is organized as follows:
	 * c c c c c c . . . | n n n n n n . . . | s s s s s s . . . | e e e e e e . . . | . . .
	 */

	//calculate the index
	//int idx = (z * SIZE_YY + y)*SIZE_XX + x;
	//int idx = ((blockIdx.y +1) * SIZE_YY + (blockIdx.x+1))*SIZE_XX + threadIdx.x;
	int idx = CALC_INDEX_SOA_WO_STRUCT(x,y,z,0);

	//each direction has offset = 130*100*100 values, we need to jump one offset to get new value
	//int offset = SIZE_XX*SIZE_YY*SIZE_ZZ;
	//save index to address for operating

	int address = idx; //if(y==1 && x==0 && z==1) printf("\nc src(%d)", address);
	float c = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nn src(%d)", address);
	float n = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\ns src(%d)", address);
	float s = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\ne src(%d)", address);
	float e = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nw src(%d)", address);
	float w = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nt src(%d)", address);
	float t = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nb src(%d)", address);
	float b = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nne src(%d)", address);
	float ne = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nnw src(%d)", address);
	float nw = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nse src(%d)", address);
	float se = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nsw src(%d)", address);
	float sw = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nnt src(%d)", address);
	float nt = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nnb src(%d)", address);
	float nb = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nst src(%d)", address);
	float st = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nsb src(%d)", address);
	float sb = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\net src(%d)", address);
	float et = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\neb src(%d)", address);
	float eb = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nwt src(%d)", address);
	float wt = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nwb src(%d)", address);
	float wb = sGrid[address];

	//get the value final cell (20) to know the kind of cell: fluid or accel or obstacle
	//int flag = flags[idx];
	unsigned short type = ((flags[idx] == 1) << 8) | ((flags[idx] == 2) & 0xff);
	//int isobs = (flags[idx] == 1);
	//int isacc = (flags[idx] == 2);

	if((type >> 8)==1)
	{
		dGrid[c_nbArray[ 0] + idx] =  		  	c;

		dGrid[c_nbArray[ 1] + idx + offset] =   s;

		dGrid[c_nbArray[ 2] + idx + 2*offset] = n;

		dGrid[c_nbArray[ 3] + idx + 3*offset] = w;

		dGrid[c_nbArray[ 4] + idx + 4*offset] = e;

		dGrid[c_nbArray[ 5] + idx + 5*offset] = b;

		dGrid[c_nbArray[ 6] + idx + 6*offset] = t;

		dGrid[c_nbArray[ 7] + idx + 7*offset] = sw;

		dGrid[c_nbArray[ 8] + idx + 8*offset] = se;

		dGrid[c_nbArray[ 9] + idx + 9*offset] = nw;

		dGrid[c_nbArray[10] + idx + 10*offset] = ne;

		dGrid[c_nbArray[11] + idx + 11*offset] = sb;

		dGrid[c_nbArray[12] + idx + 12*offset] = st;

		dGrid[c_nbArray[13] + idx + 13*offset] = nb;

		dGrid[c_nbArray[14] + idx + 14*offset] = nt;

		dGrid[c_nbArray[15] + idx + 15*offset] = wb;

		dGrid[c_nbArray[16] + idx + 16*offset] = wt;

		dGrid[c_nbArray[17] + idx + 17*offset] = eb;

		dGrid[c_nbArray[18] + idx + 18*offset] = et;
	}
	else
	{
		float rho = +c + n + s + e + w + t + b + ne + nw + se + sw + nt + nb + st + sb + et + eb + wt + wb;

		float ux = ((+e - w + ne - nw + se - sw + et + eb - wt - wb)/rho)*(!(type & 0xff)) + 0.005*(type & 0xff);
		float uy = ((+n - s + ne + nw - se - sw + nt + nb - st - sb)/rho)*(!(type & 0xff)) + 0.002*(type & 0xff);
		float uz = ((+t - b + nt - nb + st - sb + et - eb + wt - wb)/rho)*(!(type & 0xff));

		//	float ux = +e - w + ne - nw + se - sw + et + eb - wt - wb;
		//	float uy = +n - s + ne + nw - se - sw + nt + nb - st - sb;
		//	float uz = +t - b + nt - nb + st - sb + et - eb + wt - wb;


		float u2 = 1.5 * ux * ux + 1.5* uy * uy + 1.5* uz * uz;
		//float u2= U2(ux,uy,uz); //similar to regular usage
		//	float u2 = u2_func(ux,uy,uz); //number of registers increases when using inline function

		dGrid[c_nbArray[ 0] + idx] =  		  	(ONEMINUSOMEGA* c + DFL1_OMEGA * rho * (1.0 - u2));

		dGrid[c_nbArray[ 1] + idx + offset] =   (ONEMINUSOMEGA* n + DFL2_OMEGA * rho * (1.0 + uy * (4.5 * uy + 3.0) - u2));

		dGrid[c_nbArray[ 2] + idx + 2*offset] = (ONEMINUSOMEGA* s + DFL2_OMEGA * rho * (1.0 + uy * (4.5 * uy - 3.0) - u2));

		dGrid[c_nbArray[ 3] + idx + 3*offset] = (ONEMINUSOMEGA* e + DFL2_OMEGA * rho * (1.0 + ux * (4.5 * ux + 3.0) - u2));

		dGrid[c_nbArray[ 4] + idx + 4*offset] = (ONEMINUSOMEGA* w + DFL2_OMEGA * rho * (1.0 + ux * (4.5 * ux - 3.0) - u2));

		dGrid[c_nbArray[ 5] + idx + 5*offset] = (ONEMINUSOMEGA* t + DFL2_OMEGA * rho * (1.0 + uz * (4.5 * uz + 3.0) - u2));

		dGrid[c_nbArray[ 6] + idx + 6*offset] = (ONEMINUSOMEGA* b + DFL2_OMEGA * rho * (1.0 + uz * (4.5 * uz - 3.0) - u2));

		dGrid[c_nbArray[ 7] + idx + 7*offset] = (ONEMINUSOMEGA* ne + DFL3_OMEGA * rho * (1.0 + (+ux + uy) * (4.5 * (+ux + uy) + 3.0) - u2));

		dGrid[c_nbArray[ 8] + idx + 8*offset] = (ONEMINUSOMEGA* nw + DFL3_OMEGA * rho * (1.0 + (-ux + uy) * (4.5 * (-ux + uy) + 3.0) - u2));

		dGrid[c_nbArray[ 9] + idx + 9*offset] = (ONEMINUSOMEGA* se + DFL3_OMEGA * rho * (1.0 + (+ux - uy) * (4.5 * (+ux - uy) + 3.0) - u2));

		dGrid[c_nbArray[10] + idx + 10*offset] = (ONEMINUSOMEGA* sw + DFL3_OMEGA * rho * (1.0 + (-ux - uy) * (4.5 * (-ux - uy) + 3.0) - u2));

		dGrid[c_nbArray[11] + idx + 11*offset] = (ONEMINUSOMEGA* nt + DFL3_OMEGA * rho * (1.0 + (+uy + uz) * (4.5 * (+uy + uz) + 3.0) - u2));

		dGrid[c_nbArray[12] + idx + 12*offset] = (ONEMINUSOMEGA* nb + DFL3_OMEGA * rho * (1.0 + (+uy - uz) * (4.5 * (+uy - uz) + 3.0) - u2));

		dGrid[c_nbArray[13] + idx + 13*offset] = (ONEMINUSOMEGA* st + DFL3_OMEGA * rho * (1.0 + (-uy + uz) * (4.5 * (-uy + uz) + 3.0) - u2));

		dGrid[c_nbArray[14] + idx + 14*offset] = (ONEMINUSOMEGA* sb + DFL3_OMEGA * rho * (1.0 + (-uy - uz) * (4.5 * (-uy - uz) + 3.0) - u2));

		dGrid[c_nbArray[15] + idx + 15*offset] = (ONEMINUSOMEGA* et + DFL3_OMEGA * rho * (1.0 + (+ux + uz) * (4.5 * (+ux + uz) + 3.0) - u2));

		dGrid[c_nbArray[16] + idx + 16*offset] = (ONEMINUSOMEGA* eb + DFL3_OMEGA * rho * (1.0 + (+ux - uz) * (4.5 * (+ux - uz) + 3.0) - u2));

		dGrid[c_nbArray[17] + idx + 17*offset] = (ONEMINUSOMEGA* wt + DFL3_OMEGA * rho * (1.0 + (-ux + uz) * (4.5 * (-ux + uz) + 3.0) - u2));

		dGrid[c_nbArray[18] + idx + 18*offset] = (ONEMINUSOMEGA* wb + DFL3_OMEGA * rho * (1.0 + (-ux - uz) * (4.5 * (-ux - uz) + 3.0) - u2));


	}



//	dGrid[c_nbArray[ 0] + idx] =  		  	(c * isobs) + (ONEMINUSOMEGA* c + DFL1_OMEGA * rho * (1.0 - u2))* (!isobs);
//	address = c_nbArray[ 1] + idx + offset; if(y==1 && x==0 && z==1) printf("\n => n dst(%d)", address);
//	dGrid[c_nbArray[ 1] + idx + offset] =   (s * isobs)  + (ONEMINUSOMEGA* n + DFL2_OMEGA * rho * (1.0 + uy * (4.5 * uy + 3.0) - u2))* (!isobs);
//	address = c_nbArray[ 2] + idx + 2*offset; if(y==1 && x==0 && z==1) printf("\n => s dst(%d)", address);
//	dGrid[c_nbArray[ 2] + idx + 2*offset] = (n * isobs)  + (ONEMINUSOMEGA* s + DFL2_OMEGA * rho * (1.0 + uy * (4.5 * uy - 3.0) - u2))* (!isobs);
//	address = c_nbArray[ 3] + idx + 3*offset; if(y==1 && x==0 && z==1) printf("\n => e dst(%d)", address);
//	dGrid[c_nbArray[ 3] + idx + 3*offset] = (w * isobs)	+ (ONEMINUSOMEGA* e + DFL2_OMEGA * rho * (1.0 + ux * (4.5 * ux + 3.0) - u2))* (!isobs);
//	address = c_nbArray[ 4] + idx + 4*offset; if(y==1 && x==0 && z==1) printf("\n => w dst(%d)", address);
//	dGrid[c_nbArray[ 4] + idx + 4*offset] = (e * isobs)	+ (ONEMINUSOMEGA* w + DFL2_OMEGA * rho * (1.0 + ux * (4.5 * ux - 3.0) - u2))* (!isobs);
//	address = c_nbArray[ 5] + idx + 5*offset; if(y==1 && x==0 && z==1) printf("\n => t dst(%d)", address);
//	dGrid[c_nbArray[ 5] + idx + 5*offset] = (b * isobs)	+ (ONEMINUSOMEGA* t + DFL2_OMEGA * rho * (1.0 + uz * (4.5 * uz + 3.0) - u2))* (!isobs);
//	address = c_nbArray[ 6] + idx + 6*offset; if(y==1 && x==0 && z==1) printf("\n => b dst(%d)", address);
//	dGrid[c_nbArray[ 6] + idx + 6*offset] = (t * isobs)	+ (ONEMINUSOMEGA* b + DFL2_OMEGA * rho * (1.0 + uz * (4.5 * uz - 3.0) - u2))* (!isobs);
//	address = c_nbArray[ 7] + idx + 7*offset; if(y==1 && x==0 && z==1) printf("\n => ne dst(%d)", address);
//	dGrid[c_nbArray[ 7] + idx + 7*offset] = (sw * isobs) + (ONEMINUSOMEGA* ne + DFL3_OMEGA * rho * (1.0 + (+ux + uy) * (4.5 * (+ux + uy) + 3.0) - u2))* (!isobs);
//	address = c_nbArray[ 8] + idx + 8*offset; if(y==1 && x==0 && z==1) printf("\n => nw dst(%d)", address);
//	dGrid[c_nbArray[ 8] + idx + 8*offset] = (se * isobs) + (ONEMINUSOMEGA* nw + DFL3_OMEGA * rho * (1.0 + (-ux + uy) * (4.5 * (-ux + uy) + 3.0) - u2))* (!isobs);
//	address = c_nbArray[ 9] + idx + 9*offset; if(y==1 && x==0 && z==1) printf("\n => se dst(%d)", address);
//	dGrid[c_nbArray[ 9] + idx + 9*offset] = (nw * isobs) + (ONEMINUSOMEGA* se + DFL3_OMEGA * rho * (1.0 + (+ux - uy) * (4.5 * (+ux - uy) + 3.0) - u2))* (!isobs);
//	address = c_nbArray[10] + idx + 10*offset; if(y==1 && x==0 && z==1) printf("\n => sw dst(%d)", address);
//	dGrid[c_nbArray[10] + idx + 10*offset] = (ne * isobs)+ (ONEMINUSOMEGA* sw + DFL3_OMEGA * rho * (1.0 + (-ux - uy) * (4.5 * (-ux - uy) + 3.0) - u2))* (!isobs);
//	address = c_nbArray[11] + idx + 11*offset; if(y==1 && x==0 && z==1) printf("\n => nt dst(%d)", address);
//	dGrid[c_nbArray[11] + idx + 11*offset] = (sb * isobs)+ (ONEMINUSOMEGA* nt + DFL3_OMEGA * rho * (1.0 + (+uy + uz) * (4.5 * (+uy + uz) + 3.0) - u2))* (!isobs);
//	address = c_nbArray[12] + idx + 12*offset; if(y==1 && x==0 && z==1) printf("\n => nb dst(%d)", address);
//	dGrid[c_nbArray[12] + idx + 12*offset] = (st * isobs)+ (ONEMINUSOMEGA* nb + DFL3_OMEGA * rho * (1.0 + (+uy - uz) * (4.5 * (+uy - uz) + 3.0) - u2))* (!isobs);
//	address = c_nbArray[13] + idx + 13*offset; if(y==1 && x==0 && z==1) printf("\n => st dst(%d)", address);
//	dGrid[c_nbArray[13] + idx + 13*offset] = (nb * isobs)+ (ONEMINUSOMEGA* st + DFL3_OMEGA * rho * (1.0 + (-uy + uz) * (4.5 * (-uy + uz) + 3.0) - u2))* (!isobs);
//	address = c_nbArray[14] + idx + 14*offset; if(y==1 && x==0 && z==1) printf("\n => sb dst(%d)", address);
//	dGrid[c_nbArray[14] + idx + 14*offset] = (nt * isobs)+ (ONEMINUSOMEGA* sb + DFL3_OMEGA * rho * (1.0 + (-uy - uz) * (4.5 * (-uy - uz) + 3.0) - u2))* (!isobs);
//	address = c_nbArray[15] + idx + 15*offset; if(y==1 && x==0 && z==1) printf("\n => et dst(%d)", address);
//	dGrid[c_nbArray[15] + idx + 15*offset] = (wb * isobs)+ (ONEMINUSOMEGA* et + DFL3_OMEGA * rho * (1.0 + (+ux + uz) * (4.5 * (+ux + uz) + 3.0) - u2))* (!isobs);
//	address = c_nbArray[16] + idx + 16*offset; if(y==1 && x==0 && z==1) printf("\n => eb dst(%d)", address);
//	dGrid[c_nbArray[16] + idx + 16*offset] = (wt * isobs)+ (ONEMINUSOMEGA* eb + DFL3_OMEGA * rho * (1.0 + (+ux - uz) * (4.5 * (+ux - uz) + 3.0) - u2))* (!isobs);
//	address = c_nbArray[17] + idx + 17*offset; if(y==1 && x==0 && z==1) printf("\n => wt dst(%d)", address);
//	dGrid[c_nbArray[17] + idx + 17*offset] = (eb * isobs)+ (ONEMINUSOMEGA* wt + DFL3_OMEGA * rho * (1.0 + (-ux + uz) * (4.5 * (-ux + uz) + 3.0) - u2))* (!isobs);
//	address = c_nbArray[18] + idx + 18*offset; if(y==1 && x==0 && z==1) printf("\n => wb dst(%d)", address);
//	dGrid[c_nbArray[18] + idx + 18*offset] = (et * isobs)+ (ONEMINUSOMEGA* wb + DFL3_OMEGA * rho * (1.0 + (-ux - uz) * (4.5 * (-ux - uz) + 3.0) - u2))* (!isobs);

}

//__global__ void /*__launch_bounds__(128,8)*/ lbm_kernel_SoA_sm_new_wrong(float *sGrid, float *dGrid, unsigned char *flags)
//{
//	__shared__ float s_E[SIZE_X +2];
//	__shared__ float s_W[SIZE_X +2];
//	__shared__ float s_SE[SIZE_X+2];
//	__shared__ float s_NE[SIZE_X+2];
//	__shared__ float s_SW[SIZE_X+2];
//	__shared__ float s_NW[SIZE_X+2];
//	__shared__ float s_EB[SIZE_X+2];
//	__shared__ float s_ET[SIZE_X+2];
//	__shared__ float s_WB[SIZE_X+2];
//	__shared__ float s_WT[SIZE_X+2];
//
//	//float *shPropPointer = (float*)array;
//	__shared__ int offset;
//
//	//__shared__ float one_minus_omega;
//	offset = SIZE_XX*SIZE_YY*SIZE_ZZ;
//	//one_minus_omega = 1.0 - OMEGA;
//	int x = threadIdx.x;
//	int y = blockIdx.x + 1;
//	int z = blockIdx.y + 1;
//
//	//int ElementsPerBlock = blockDim.x;
//	//float c, n, s, e, w, t, b, ne, nw, se, sw, nt, nb, st, sb, et,	eb, wt, wb;
//
//	/*the grid is organized as follows:
//	 * c c c c c c . . . | n n n n n n . . . | s s s s s s . . . | e e e e e e . . . | . . .
//	 */
//
//	//calculate the index
//	//int idx = (z * SIZE_YY + y)*SIZE_XX + x;
//	//int idx = ((blockIdx.y +1) * SIZE_YY + (blockIdx.x+1))*SIZE_XX + threadIdx.x;
//	int idx = CALC_INDEX_SOA_WO_STRUCT(x,y,z,0);
//
//	//each direction has offset = 130*100*100 values, we need to jump one offset to get new value
//	//int offset = SIZE_XX*SIZE_YY*SIZE_ZZ;
//	//save index to address for operating
//	float arr[] = {0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0 };
//	for(int i=0; i <19; i++)
//	{
//		arr[i] = sGrid[idx + i*offset];
//	}
//	//int address = idx;
////	float c = sGrid[idx];
////	//address = address + offset;
////	float n = sGrid[idx+offset];
////	//address = address + offset;
////	float s = sGrid[idx+2*offset];
////	//address = address + offset;
////	float e = sGrid[idx+3*offset];
////	//address = address + offset;
////	float w = sGrid[idx+4*offset];
////	//address = address + offset;
////	float t = sGrid[idx+5*offset];
////	//address = address + offset;
////	float b = sGrid[idx+6*offset];
////	//address = address + offset;
////	float ne = sGrid[idx+7*offset];
////	//address = address + offset;
////	float nw = sGrid[idx+8*offset];
////	//address = address + offset;
////	float se = sGrid[idx+9*offset];
////	//address = address + offset;
////	float sw = sGrid[idx+10*offset];
////	//address = address + offset;
////	float nt = sGrid[idx+11*offset];
////	//address = address + offset;
////	float nb = sGrid[idx+12*offset];
////	//address = address + offset;
////	float st = sGrid[idx+13*offset];
////	//address = address + offset;
////	float sb = sGrid[idx+14*offset];
////	//address = address + offset;
////	float et = sGrid[idx+15*offset];
////	//address = address + offset;
////	float eb = sGrid[idx+16*offset];
////	//address = address + offset;
////	float wt = sGrid[idx+17*offset];
////	//address = address + offset;
////	float wb = sGrid[idx+18*offset];
//
//	//get the value final cell (20) to know the kind of cell: fluid or accel or obstacle
//	//int flag = flags[idx];
//	//unsigned short type ;
//	//type = ((!(flags[idx] ^ 1)) << 8) & 'FF00'; //8 bits dau la isobs
//	//type = (!(flags[idx] ^ 2)) &   'FFFF'; //8 bits sau la isacc
//	unsigned short type = ((flags[idx] == 1) << 8) | ((flags[idx] == 2) & 0xff);
//	//unsigned char isobs = !(flags[idx] ^ 1);
//	//unsigned char isacc = !(flags[idx] ^ 2);
//
//	float rho = +arr[0] + arr[1] + arr[2] + arr[3] + arr[4] + arr[5] + arr[6] + arr[7] + arr[8] + arr[9] + arr[10] + arr[11] + arr[12] + arr[13] + arr[14] + arr[15] + arr[16] + arr[17] + arr[18];
//
//	/*float ux = ((+e - w + ne - nw + se - sw + et + eb - wt - wb)/rho)*(!isacc) + 0.005*isacc;
//	float uy = ((+n - s + ne + nw - se - sw + nt + nb - st - sb)/rho)*(!isacc) + 0.002*isacc;
//	float uz = ((+t - b + nt - nb + st - sb + et - eb + wt - wb)/rho)*(!isacc);
//	*/
//	float ux = ((+arr[3] - arr[4] + arr[7] - arr[8] + arr[9] - arr[10] + arr[15] + arr[16] - arr[17] - arr[18])/rho)*(!(type & 0xff)) + 0.005*(type & 0xff);
//	float uy = ((+arr[1] - arr[2] + arr[7] + arr[8] - arr[9] - arr[10] + arr[11] + arr[12] - arr[13] - arr[14])/rho)*(!(type & 0xff)) + 0.002*(type & 0xff);
//	float uz = ((+arr[5] - arr[6] + arr[11] - arr[12] + arr[13] - arr[14] + arr[15] - arr[16] + arr[17] - arr[18])/rho)*(!(type & 0xff));
//
//	//float ux = +e - w + ne - nw + se - sw + et + eb - wt - wb;
//	//float uy = +n - s + ne + nw - se - sw + nt + nb - st - sb;
//	//float uz = +t - b + nt - nb + st - sb + et - eb + wt - wb;
//
//	//ux = (ux / rho) * (!isacc) + 0.005 * isacc;
//	//uy = (uy / rho) * (!isacc) + 0.002 * isacc;
//	//uz = (uz / rho) * (!isacc) + 0.000 * isacc;
//
//	float u2 = 1.5 * (ux * ux + uy * uy + uz * uz); //U2(ux,uy,uz);//
////	float u2 = u2_func(ux,uy,uz);
//
//	int shiftE = ((y-1)&0x1)^((z-1)&0x1); //if(x==1 && y==1 && z==1) printf("\nshiftE = %d", shiftE);
//	int shiftW = 0x1 & (~shiftE); //if(x==1 && y==1 && z==1) printf("\nshiftW = %d", shiftW);
//	int txE = x+shiftE; //x;
//	int txW = x-shiftW; //x;
//
//
//	s_E[txE] = (arr[4] * (type >> 8))	+ (ONEMINUSOMEGA* arr[3] + DFL2_OMEGA * rho * (1.0 + ux * (4.5 * ux + 3.0) - u2))* (!(type >> 8));
//	//if(x==1 && y==1 && z==1) printf("\n(%d, %d, %d) s_E[%d] %.8f", x, y, z, txE, s_E[txE]);
//	s_SE[txE] = (arr[8] * (type >> 8)) + (ONEMINUSOMEGA* arr[9] + DFL3_OMEGA * rho * (1.0 + (+ux - uy) * (4.5 * (+ux - uy) + 3.0) - u2))* (!(type >> 8));
//	//if(x==1 && y==1 && z==1) printf("\n(%d, %d, %d) s_SE[%d] %.8f", x, y, z, txE, s_SE[txE]);
//	s_NE[txE] = (arr[10] * (type >> 8)) + (ONEMINUSOMEGA* arr[7] + DFL3_OMEGA * rho * (1.0 + (+ux + uy) * (4.5 * (+ux + uy) + 3.0) - u2))* (!(type >> 8));
//	//if(x==1 && y==1 && z==1) printf("\n(%d, %d, %d) s_NE[%d] %.8f", x, y, z, txE, s_NE[txE]);
//	s_EB[txE] = (arr[17] * (type >> 8))+ (ONEMINUSOMEGA* arr[16] + DFL3_OMEGA * rho * (1.0 + (+ux - uz) * (4.5 * (+ux - uz) + 3.0) - u2))* (!(type >> 8));
//	//if(x==1 && y==1 && z==1) printf("\n(%d, %d, %d) s_EB[%d] %.8f", x, y, z, txE, s_EB[txE]);
//	s_ET[txE] = (arr[18] * (type >> 8))+ (ONEMINUSOMEGA* arr[15] + DFL3_OMEGA * rho * (1.0 + (+ux + uz) * (4.5 * (+ux + uz) + 3.0) - u2))* (!(type >> 8));
//	//if(x==1 && y==1 && z==1) printf("\n(%d, %d, %d) s_ET[%d] %.8f", x, y, z, txE, s_ET[txE]);
//	s_W[txW+1] = (arr[3] * (type >> 8))	+ (ONEMINUSOMEGA* arr[4] + DFL2_OMEGA * rho * (1.0 + ux * (4.5 * ux - 3.0) - u2))* (!(type >> 8));
//	//if(x==1 && y==1 && z==1) printf("\n(%d, %d, %d) s_W[%d] %.8f %d", x, y, z, txW+1, s_W[txW+1], txW);
//	s_SW[txW+1] = (arr[7] * (type >> 8))+ (ONEMINUSOMEGA* arr[10] + DFL3_OMEGA * rho * (1.0 + (-ux - uy) * (4.5 * (-ux - uy) + 3.0) - u2))* (!(type >> 8));
//	//if(x==1 && y==1 && z==1) printf("\n(%d, %d, %d) s_SW[%d] %.8f", x, y, z, txW+1, s_SW[txW+1]);
//	s_NW[txW+1] = (arr[9] * (type >> 8)) + (ONEMINUSOMEGA* arr[8] + DFL3_OMEGA * rho * (1.0 + (-ux + uy) * (4.5 * (-ux + uy) + 3.0) - u2))* (!(type >> 8));
//	//if(x==1 && y==1 && z==1) printf("\n(%d, %d, %d) s_NW[%d] %.8f", x, y, z, txW+1, s_NW[txW+1]);
//	s_WB[txW+1] = (arr[15] * (type >> 8))+ (ONEMINUSOMEGA* arr[18] + DFL3_OMEGA * rho * (1.0 + (-ux - uz) * (4.5 * (-ux - uz) + 3.0) - u2))* (!(type >> 8));
//	//if(x==1 && y==1 && z==1) printf("\n(%d, %d, %d) s_WB[%d] %.8f", x, y, z, txW+1, s_WB[txW+1]);
//	s_WT[txW+1] = (arr[16] * (type >> 8))+ (ONEMINUSOMEGA* arr[17] + DFL3_OMEGA * rho * (1.0 + (-ux + uz) * (4.5 * (-ux + uz) + 3.0) - u2))* (!(type >> 8));
//	//if(x==1 && y==1 && z==1) printf("\n(%d, %d, %d) s_WT[%d] %.8f", x, y, z, txW+1, s_WT[txW+1]);
//	__syncthreads();
//
//	dGrid[c_nbArray[ C] + idx] =  		  	(arr[0] * (type >> 8)) + (ONEMINUSOMEGA* arr[0] + DFL1_OMEGA * rho * (1.0 - u2))* (!(type >> 8));
//
//	dGrid[c_nbArray[ N] + idx + offset] =   (arr[2] * (type >> 8))  + (ONEMINUSOMEGA* arr[1] + DFL2_OMEGA * rho * (1.0 + uy * (4.5 * uy + 3.0) - u2))* (!(type >> 8));
//
//	dGrid[c_nbArray[ S] + idx + 2*offset] = (arr[1] * (type >> 8))  + (ONEMINUSOMEGA* arr[2] + DFL2_OMEGA * rho * (1.0 + uy * (4.5 * uy - 3.0) - u2))* (!(type >> 8));
//
//	dGrid[c_nbArray[ E] + idx + 3*offset] = s_E[x];
//	//if(x==1 && y==1 && z==1) printf("\ndst_pos <-> sh_pos %d <-> %d", c_nbArray[ E] + idx + 3*offset, x);
//	dGrid[c_nbArray[ W] + idx + 4*offset] = s_W[x+1];
//	//if(x==1 && y==1 && z==1) printf("\ndst_pos <-> sh_pos %d <-> %d", c_nbArray[ W] + idx + 4*offset, x+1);
//	dGrid[c_nbArray[ T] + idx + 5*offset] = (arr[6] * (type >> 8))	+ (ONEMINUSOMEGA* arr[5] + DFL2_OMEGA * rho * (1.0 + uz * (4.5 * uz + 3.0) - u2))* (!(type >> 8));
//
//	dGrid[c_nbArray[ B] + idx + 6*offset] = (arr[5] * (type >> 8))	+ (ONEMINUSOMEGA* arr[6] + DFL2_OMEGA * rho * (1.0 + uz * (4.5 * uz - 3.0) - u2))* (!(type >> 8));
//
//	dGrid[c_nbArray[NE] + idx + 7*offset] = s_NE[x]; //if(x==1 && y==1 && z==1) printf("\ndst_pos <-> sh_pos %d <-> %d", c_nbArray[NE] + idx + 7*offset, x);
//
//	dGrid[c_nbArray[NW] + idx + 8*offset] = s_NW[x+1]; //if(x==1 && y==1 && z==1) printf("\ndst_pos <-> sh_pos %d <-> %d", c_nbArray[NW] + idx + 8*offset, x+1);
//
//	dGrid[c_nbArray[SE] + idx + 9*offset] = s_SE[x]; //if(x==1 && y==1 && z==1) printf("\ndst_pos <-> sh_pos %d <-> %d", c_nbArray[SE] + idx + 9*offset, x);
//
//	dGrid[c_nbArray[SW] + idx + 10*offset] = s_SW[x+1]; //if(x==1 && y==1 && z==1) printf("\ndst_pos <-> sh_pos %d <-> %d", c_nbArray[SW] + idx + 10*offset, x+1);
//
//	dGrid[c_nbArray[NT] + idx + 11*offset] = (arr[14] * (type >> 8))+ (ONEMINUSOMEGA* arr[11] + DFL3_OMEGA * rho * (1.0 + (+uy + uz) * (4.5 * (+uy + uz) + 3.0) - u2))* (!(type >> 8));
//
//	dGrid[c_nbArray[NB] + idx + 12*offset] = (arr[13] * (type >> 8))+ (ONEMINUSOMEGA* arr[12] + DFL3_OMEGA * rho * (1.0 + (+uy - uz) * (4.5 * (+uy - uz) + 3.0) - u2))* (!(type >> 8));
//
//	dGrid[c_nbArray[ST] + idx + 13*offset] = (arr[12] * (type >> 8))+ (ONEMINUSOMEGA* arr[13] + DFL3_OMEGA * rho * (1.0 + (-uy + uz) * (4.5 * (-uy + uz) + 3.0) - u2))* (!(type >> 8));
//
//	dGrid[c_nbArray[SB] + idx + 14*offset] = (arr[11] * (type >> 8))+ (ONEMINUSOMEGA* arr[14] + DFL3_OMEGA * rho * (1.0 + (-uy - uz) * (4.5 * (-uy - uz) + 3.0) - u2))* (!(type >> 8));
//
//	dGrid[c_nbArray[ET] + idx + 15*offset] = s_ET[x]; //if(x==1 && y==1 && z==1) printf("\ndst_pos <-> sh_pos %d <-> %d", c_nbArray[ET] + idx + 15*offset, x);
//
//	dGrid[c_nbArray[EB] + idx + 16*offset] = s_EB[x]; //if(x==1 && y==1 && z==1) printf("\ndst_pos <-> sh_pos %d <-> %d", c_nbArray[EB] + idx + 16*offset, x);
//
//	dGrid[c_nbArray[WT] + idx + 17*offset] = s_WT[x+1]; //if(x==1 && y==1 && z==1) printf("\ndst_pos <-> sh_pos %d <-> %d", c_nbArray[WT] + idx + 17*offset, x+1);
//
//	dGrid[c_nbArray[WB] + idx + 18*offset] = s_WB[x+1]; //if(x==1 && y==1 && z==1) printf("\ndst_pos <-> sh_pos %d <-> %d", c_nbArray[WB] + idx + 18*offset, x+1);
//
//}
//__global__ void /*__launch_bounds__(128,8)*/ lbm_kernel_SoA_sm_arr(float *sGrid, float *dGrid, unsigned char *flags)
//{
//	__shared__ float s_E[SIZE_X +2];
//	__shared__ float s_W[SIZE_X +2];
//	__shared__ float s_SE[SIZE_X+2];
//	__shared__ float s_NE[SIZE_X+2];
//	__shared__ float s_SW[SIZE_X+2];
//	__shared__ float s_NW[SIZE_X+2];
//	__shared__ float s_EB[SIZE_X+2];
//	__shared__ float s_ET[SIZE_X+2];
//	__shared__ float s_WB[SIZE_X+2];
//	__shared__ float s_WT[SIZE_X+2];
//
//	//float *shPropPointer = (float*)array;
//	__shared__ int offset;
//
//	//__shared__ float one_minus_omega;
//	offset = SIZE_XX*SIZE_YY*SIZE_ZZ;
//	//one_minus_omega = 1.0 - OMEGA;
//	int x = threadIdx.x;
//	int y = blockIdx.x + 1;
//	int z = blockIdx.y + 1;
//
//	//int ElementsPerBlock = blockDim.x;
//	//float c, n, s, e, w, t, b, ne, nw, se, sw, nt, nb, st, sb, et,	eb, wt, wb;
//
//	/*the grid is organized as follows:
//	 * c c c c c c . . . | n n n n n n . . . | s s s s s s . . . | e e e e e e . . . | . . .
//	 */
//
//	//calculate the index
//	//int idx = (z * SIZE_YY + y)*SIZE_XX + x;
//	//int idx = ((blockIdx.y +1) * SIZE_YY + (blockIdx.x+1))*SIZE_XX + threadIdx.x;
//	int idx = CALC_INDEX_SOA_WO_STRUCT(x,y,z,0);
//
//	//each direction has offset = 130*100*100 values, we need to jump one offset to get new value
//	//int offset = SIZE_XX*SIZE_YY*SIZE_ZZ;
//	//save index to address for operating
//	float arr[] = {0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0 };
//	for(int i=0; i <19; i++)
//	{
//		arr[i] = sGrid[idx + i*offset];
//	}
//
//	//get the value final cell (20) to know the kind of cell: fluid or accel or obstacle
//	//int flag = flags[idx];
//	//unsigned short type ;
//	//type = ((!(flags[idx] ^ 1)) << 8) & 'FF00'; //8 bits dau la isobs
//	//type = (!(flags[idx] ^ 2)) &   'FFFF'; //8 bits sau la isacc
//	unsigned short type = ((flags[idx] == 1) << 8) | ((flags[idx] == 2) & 0xff);
//	//unsigned char isobs = !(flags[idx] ^ 1);
//	//unsigned char isacc = !(flags[idx] ^ 2);
//
//	float rho = +arr[0] + arr[1] + arr[2] + arr[3] + arr[4] + arr[5] + arr[6] + arr[7] + arr[8] + arr[9] + arr[10] + arr[11] + arr[12] + arr[13] + arr[14] + arr[15] + arr[16] + arr[17] + arr[18];
//
//	/*float ux = ((+e - w + ne - nw + se - sw + et + eb - wt - wb)/rho)*(!isacc) + 0.005*isacc;
//	float uy = ((+n - s + ne + nw - se - sw + nt + nb - st - sb)/rho)*(!isacc) + 0.002*isacc;
//	float uz = ((+t - b + nt - nb + st - sb + et - eb + wt - wb)/rho)*(!isacc);
//	*/
//	float ux = ((+arr[3] - arr[4] + arr[7] - arr[8] + arr[9] - arr[10] + arr[15] + arr[16] - arr[17] - arr[18])/rho)*(!(type & 0xff)) + 0.005*(type & 0xff);
//	float uy = ((+arr[1] - arr[2] + arr[7] + arr[8] - arr[9] - arr[10] + arr[11] + arr[12] - arr[13] - arr[14])/rho)*(!(type & 0xff)) + 0.002*(type & 0xff);
//	float uz = ((+arr[5] - arr[6] + arr[11] - arr[12] + arr[13] - arr[14] + arr[15] - arr[16] + arr[17] - arr[18])/rho)*(!(type & 0xff));
//
//	//float ux = +e - w + ne - nw + se - sw + et + eb - wt - wb;
//	//float uy = +n - s + ne + nw - se - sw + nt + nb - st - sb;
//	//float uz = +t - b + nt - nb + st - sb + et - eb + wt - wb;
//
//	//ux = (ux / rho) * (!isacc) + 0.005 * isacc;
//	//uy = (uy / rho) * (!isacc) + 0.002 * isacc;
//	//uz = (uz / rho) * (!isacc) + 0.000 * isacc;
//
//	float u2 = 1.5 * (ux * ux + uy * uy + uz * uz); //U2(ux,uy,uz);//
////	float u2 = u2_func(ux,uy,uz);
//
////	int shiftE = ((y-1)&0x1)^((z-1)&0x1); //if(x==1 && y==1 && z==1) printf("\nshiftE = %d", shiftE);
////	int shiftW = 0x1 & (~shiftE); //if(x==1 && y==1 && z==1) printf("\nshiftW = %d", shiftW);
////	int txE = x+shiftE; //x;
////	int txW = x-shiftW; //x;
//
//
//	s_E[x] = (arr[4] * (type >> 8))	+ (ONEMINUSOMEGA* arr[3] + DFL2_OMEGA * rho * (1.0 + ux * (4.5 * ux + 3.0) - u2))* (!(type >> 8));
//	//if(x==1 && y==1 && z==1) printf("\n(%d, %d, %d) s_E[%d] %.8f", x, y, z, x, s_E[x]);
//	s_SE[x] = (arr[8] * (type >> 8)) + (ONEMINUSOMEGA* arr[9] + DFL3_OMEGA * rho * (1.0 + (+ux - uy) * (4.5 * (+ux - uy) + 3.0) - u2))* (!(type >> 8));
//	//if(x==1 && y==1 && z==1) printf("\n(%d, %d, %d) s_SE[%d] %.8f", x, y, z, x, s_SE[x]);
//	s_NE[x] = (arr[10] * (type >> 8)) + (ONEMINUSOMEGA* arr[7] + DFL3_OMEGA * rho * (1.0 + (+ux + uy) * (4.5 * (+ux + uy) + 3.0) - u2))* (!(type >> 8));
//	//if(x==1 && y==1 && z==1) printf("\n(%d, %d, %d) s_NE[%d] %.8f", x, y, z, x, s_NE[x]);
//	s_EB[x] = (arr[17] * (type >> 8))+ (ONEMINUSOMEGA* arr[16] + DFL3_OMEGA * rho * (1.0 + (+ux - uz) * (4.5 * (+ux - uz) + 3.0) - u2))* (!(type >> 8));
//	//if(x==1 && y==1 && z==1) printf("\n(%d, %d, %d) s_EB[%d] %.8f", x, y, z, x, s_EB[x]);
//	s_ET[x] = (arr[18] * (type >> 8))+ (ONEMINUSOMEGA* arr[15] + DFL3_OMEGA * rho * (1.0 + (+ux + uz) * (4.5 * (+ux + uz) + 3.0) - u2))* (!(type >> 8));
//	//if(x==1 && y==1 && z==1) printf("\n(%d, %d, %d) s_ET[%d] %.8f", x, y, z, x, s_ET[x]);
//	s_W[x+1] = (arr[3] * (type >> 8))	+ (ONEMINUSOMEGA* arr[4] + DFL2_OMEGA * rho * (1.0 + ux * (4.5 * ux - 3.0) - u2))* (!(type >> 8));
//	//if(x==1 && y==1 && z==1) printf("\n(%d, %d, %d) s_W[%d] %.8f %d", x, y, z, x+1, s_W[x+1], txW);
//	s_SW[x+1] = (arr[7] * (type >> 8))+ (ONEMINUSOMEGA* arr[10] + DFL3_OMEGA * rho * (1.0 + (-ux - uy) * (4.5 * (-ux - uy) + 3.0) - u2))* (!(type >> 8));
//	//if(x==1 && y==1 && z==1) printf("\n(%d, %d, %d) s_SW[%d] %.8f", x, y, z, x+1, s_SW[x+1]);
//	s_NW[x+1] = (arr[9] * (type >> 8)) + (ONEMINUSOMEGA* arr[8] + DFL3_OMEGA * rho * (1.0 + (-ux + uy) * (4.5 * (-ux + uy) + 3.0) - u2))* (!(type >> 8));
//	//if(x==1 && y==1 && z==1) printf("\n(%d, %d, %d) s_NW[%d] %.8f", x, y, z, x+1, s_NW[x+1]);
//	s_WB[x+1] = (arr[15] * (type >> 8))+ (ONEMINUSOMEGA* arr[18] + DFL3_OMEGA * rho * (1.0 + (-ux - uz) * (4.5 * (-ux - uz) + 3.0) - u2))* (!(type >> 8));
//	//if(x==1 && y==1 && z==1) printf("\n(%d, %d, %d) s_WB[%d] %.8f", x, y, z, x+1, s_WB[x+1]);
//	s_WT[x+1] = (arr[16] * (type >> 8))+ (ONEMINUSOMEGA* arr[17] + DFL3_OMEGA * rho * (1.0 + (-ux + uz) * (4.5 * (-ux + uz) + 3.0) - u2))* (!(type >> 8));
//	//if(x==1 && y==1 && z==1) printf("\n(%d, %d, %d) s_WT[%d] %.8f", x, y, z, txW+1, s_WT[txW+1]);
//	__syncthreads();
//
//	dGrid[c_nbArray[ C] + idx] =  		  	(arr[0] * (type >> 8)) + (ONEMINUSOMEGA* arr[0] + DFL1_OMEGA * rho * (1.0 - u2))* (!(type >> 8));
//
//	dGrid[c_nbArray[ N] + idx + offset] =   (arr[2] * (type >> 8))  + (ONEMINUSOMEGA* arr[1] + DFL2_OMEGA * rho * (1.0 + uy * (4.5 * uy + 3.0) - u2))* (!(type >> 8));
//
//	dGrid[c_nbArray[ S] + idx + 2*offset] = (arr[1] * (type >> 8))  + (ONEMINUSOMEGA* arr[2] + DFL2_OMEGA * rho * (1.0 + uy * (4.5 * uy - 3.0) - u2))* (!(type >> 8));
//
//	dGrid[c_nbArray[ E] + idx + 3*offset] = s_E[x];
//	//if(x==1 && y==1 && z==1) printf("\ndst_pos <-> sh_pos %d <-> %d", c_nbArray[ E] + idx + 3*offset, x);
//	dGrid[c_nbArray[ W] + idx + 4*offset] = s_W[x+1];
//	//if(x==1 && y==1 && z==1) printf("\ndst_pos <-> sh_pos %d <-> %d", c_nbArray[ W] + idx + 4*offset, x+1);
//	dGrid[c_nbArray[ T] + idx + 5*offset] = (arr[6] * (type >> 8))	+ (ONEMINUSOMEGA* arr[5] + DFL2_OMEGA * rho * (1.0 + uz * (4.5 * uz + 3.0) - u2))* (!(type >> 8));
//
//	dGrid[c_nbArray[ B] + idx + 6*offset] = (arr[5] * (type >> 8))	+ (ONEMINUSOMEGA* arr[6] + DFL2_OMEGA * rho * (1.0 + uz * (4.5 * uz - 3.0) - u2))* (!(type >> 8));
//
//	dGrid[c_nbArray[NE] + idx + 7*offset] = s_NE[x]; //if(x==1 && y==1 && z==1) printf("\ndst_pos <-> sh_pos %d <-> %d", c_nbArray[NE] + idx + 7*offset, x);
//
//	dGrid[c_nbArray[NW] + idx + 8*offset] = s_NW[x+1]; //if(x==1 && y==1 && z==1) printf("\ndst_pos <-> sh_pos %d <-> %d", c_nbArray[NW] + idx + 8*offset, x+1);
//
//	dGrid[c_nbArray[SE] + idx + 9*offset] = s_SE[x]; //if(x==1 && y==1 && z==1) printf("\ndst_pos <-> sh_pos %d <-> %d", c_nbArray[SE] + idx + 9*offset, x);
//
//	dGrid[c_nbArray[SW] + idx + 10*offset] = s_SW[x+1]; //if(x==1 && y==1 && z==1) printf("\ndst_pos <-> sh_pos %d <-> %d", c_nbArray[SW] + idx + 10*offset, x+1);
//
//	dGrid[c_nbArray[NT] + idx + 11*offset] = (arr[14] * (type >> 8))+ (ONEMINUSOMEGA* arr[11] + DFL3_OMEGA * rho * (1.0 + (+uy + uz) * (4.5 * (+uy + uz) + 3.0) - u2))* (!(type >> 8));
//
//	dGrid[c_nbArray[NB] + idx + 12*offset] = (arr[13] * (type >> 8))+ (ONEMINUSOMEGA* arr[12] + DFL3_OMEGA * rho * (1.0 + (+uy - uz) * (4.5 * (+uy - uz) + 3.0) - u2))* (!(type >> 8));
//
//	dGrid[c_nbArray[ST] + idx + 13*offset] = (arr[12] * (type >> 8))+ (ONEMINUSOMEGA* arr[13] + DFL3_OMEGA * rho * (1.0 + (-uy + uz) * (4.5 * (-uy + uz) + 3.0) - u2))* (!(type >> 8));
//
//	dGrid[c_nbArray[SB] + idx + 14*offset] = (arr[11] * (type >> 8))+ (ONEMINUSOMEGA* arr[14] + DFL3_OMEGA * rho * (1.0 + (-uy - uz) * (4.5 * (-uy - uz) + 3.0) - u2))* (!(type >> 8));
//
//	dGrid[c_nbArray[ET] + idx + 15*offset] = s_ET[x]; //if(x==1 && y==1 && z==1) printf("\ndst_pos <-> sh_pos %d <-> %d", c_nbArray[ET] + idx + 15*offset, x);
//
//	dGrid[c_nbArray[EB] + idx + 16*offset] = s_EB[x]; //if(x==1 && y==1 && z==1) printf("\ndst_pos <-> sh_pos %d <-> %d", c_nbArray[EB] + idx + 16*offset, x);
//
//	dGrid[c_nbArray[WT] + idx + 17*offset] = s_WT[x+1]; //if(x==1 && y==1 && z==1) printf("\ndst_pos <-> sh_pos %d <-> %d", c_nbArray[WT] + idx + 17*offset, x+1);
//
//	dGrid[c_nbArray[WB] + idx + 18*offset] = s_WB[x+1]; //if(x==1 && y==1 && z==1) printf("\ndst_pos <-> sh_pos %d <-> %d", c_nbArray[WB] + idx + 18*offset, x+1);
//
//}
//backup sm
//__global__ void /*__launch_bounds__(128,8)*/ lbm_kernel_SoA_sm(float *sGrid, float *dGrid, unsigned char *flags)
//{
//	__shared__ float s_E[SIZE_X +2];
//	__shared__ float s_W[SIZE_X +2];
//	__shared__ float s_SE[SIZE_X+2];
//	__shared__ float s_NE[SIZE_X+2];
//	__shared__ float s_SW[SIZE_X+2];
//	__shared__ float s_NW[SIZE_X+2];
//	__shared__ float s_EB[SIZE_X+2];
//	__shared__ float s_ET[SIZE_X+2];
//	__shared__ float s_WB[SIZE_X+2];
//	__shared__ float s_WT[SIZE_X+2];
//
//	//float *shPropPointer = (float*)array;
//	__shared__ int offset;
//
//	//__shared__ float one_minus_omega;
//	offset = SIZE_XX*SIZE_YY*SIZE_ZZ;
//	//one_minus_omega = 1.0 - OMEGA;
//	int x = threadIdx.x;
//	int y = blockIdx.x + 1;
//	int z = blockIdx.y + 1;
//
//	//int ElementsPerBlock = blockDim.x;
//	//float c, n, s, e, w, t, b, ne, nw, se, sw, nt, nb, st, sb, et,	eb, wt, wb;
//
//	/*the grid is organized as follows:
//	 * c c c c c c . . . | n n n n n n . . . | s s s s s s . . . | e e e e e e . . . | . . .
//	 */
//
//	//calculate the index
//	//int idx = (z * SIZE_YY + y)*SIZE_XX + x;
//	//int idx = ((blockIdx.y +1) * SIZE_YY + (blockIdx.x+1))*SIZE_XX + threadIdx.x;
//	int idx = CALC_INDEX_SOA_WO_STRUCT(x,y,z,0);
//
//	//each direction has offset = 130*100*100 values, we need to jump one offset to get new value
//	//int offset = SIZE_XX*SIZE_YY*SIZE_ZZ;
//	//save index to address for operating
//
//	int address = idx;
//	float c = sGrid[address];
//	address = address + offset;
//	float n = sGrid[address];
//	address = address + offset;
//	float s = sGrid[address];
//	address = address + offset;
//	float e = sGrid[address];
//	address = address + offset;
//	float w = sGrid[address];
//	address = address + offset;
//	float t = sGrid[address];
//	address = address + offset;
//	float b = sGrid[address];
//	address = address + offset;
//	float ne = sGrid[address];
//	address = address + offset;
//	float nw = sGrid[address];
//	address = address + offset;
//	float se = sGrid[address];
//	address = address + offset;
//	float sw = sGrid[address];
//	address = address + offset;
//	float nt = sGrid[address];
//	address = address + offset;
//	float nb = sGrid[address];
//	address = address + offset;
//	float st = sGrid[address];
//	address = address + offset;
//	float sb = sGrid[address];
//	address = address + offset;
//	float et = sGrid[address];
//	address = address + offset;
//	float eb = sGrid[address];
//	address = address + offset;
//	float wt = sGrid[address];
//	address = address + offset;
//	float wb = sGrid[address];
//
//	//get the value final cell (20) to know the kind of cell: fluid or accel or obstacle
//	//int flag = flags[idx];
//	//unsigned short type ;
//	//type = ((!(flags[idx] ^ 1)) << 8) & 'FF00'; //8 bits dau la isobs
//	//type = (!(flags[idx] ^ 2)) &   'FFFF'; //8 bits sau la isacc
//	unsigned short type = ((flags[idx] == 1) << 8) | ((flags[idx] == 2) & 0xff);
//	//unsigned char isobs = !(flags[idx] ^ 1);
//	//unsigned char isacc = !(flags[idx] ^ 2);
//
//	float rho = +c + n + s + e + w + t + b + ne + nw + se + sw + nt + nb + st + sb + et + eb + wt + wb;
//
//	/*float ux = ((+e - w + ne - nw + se - sw + et + eb - wt - wb)/rho)*(!isacc) + 0.005*isacc;
//	float uy = ((+n - s + ne + nw - se - sw + nt + nb - st - sb)/rho)*(!isacc) + 0.002*isacc;
//	float uz = ((+t - b + nt - nb + st - sb + et - eb + wt - wb)/rho)*(!isacc);
//	*/
//	float ux = ((+e - w + ne - nw + se - sw + et + eb - wt - wb)/rho)*(!(type & 0xff)) + 0.005*(type & 0xff);
//	float uy = ((+n - s + ne + nw - se - sw + nt + nb - st - sb)/rho)*(!(type & 0xff)) + 0.002*(type & 0xff);
//	float uz = ((+t - b + nt - nb + st - sb + et - eb + wt - wb)/rho)*(!(type & 0xff));
//
//	//float ux = +e - w + ne - nw + se - sw + et + eb - wt - wb;
//	//float uy = +n - s + ne + nw - se - sw + nt + nb - st - sb;
//	//float uz = +t - b + nt - nb + st - sb + et - eb + wt - wb;
//
//	//ux = (ux / rho) * (!isacc) + 0.005 * isacc;
//	//uy = (uy / rho) * (!isacc) + 0.002 * isacc;
//	//uz = (uz / rho) * (!isacc) + 0.000 * isacc;
//
//	float u2 = 1.5 * (ux * ux + uy * uy + uz * uz); //U2(ux,uy,uz);//
////	float u2 = u2_func(ux,uy,uz);
//
//	int shiftE = ((y-1)&0x1)^((z-1)&0x1); //0
//	int shiftW = 0x1 & (~shiftE); //0
//	int txE = x+shiftE;
//	int txW = x-shiftW;
//
//
//	s_E[txE] = (w * (type >> 8))	+ (ONEMINUSOMEGA* e + DFL2_OMEGA * rho * (1.0 + ux * (4.5 * ux + 3.0) - u2))* (!(type >> 8));
//	//printf("\n(%d, %d, %d) s_E[%d] %.8f", x, y, z, txE, s_E[txE]);
//	s_SE[txE] = (nw * (type >> 8)) + (ONEMINUSOMEGA* se + DFL3_OMEGA * rho * (1.0 + (+ux - uy) * (4.5 * (+ux - uy) + 3.0) - u2))* (!(type >> 8));
//	//printf("\n(%d, %d, %d) s_SE[%d] %.8f", x, y, z, txE, s_SE[txE]);
//	s_NE[txE] = (sw * (type >> 8)) + (ONEMINUSOMEGA* ne + DFL3_OMEGA * rho * (1.0 + (+ux + uy) * (4.5 * (+ux + uy) + 3.0) - u2))* (!(type >> 8));
//	//printf("\n(%d, %d, %d) s_NE[%d] %.8f", x, y, z, txE, s_NE[txE]);
//	s_EB[txE] = (wt * (type >> 8))+ (ONEMINUSOMEGA* eb + DFL3_OMEGA * rho * (1.0 + (+ux - uz) * (4.5 * (+ux - uz) + 3.0) - u2))* (!(type >> 8));
//	//printf("\n(%d, %d, %d) s_EB[%d] %.8f", x, y, z, txE, s_EB[txE]);
//	s_ET[txE] = (wb * (type >> 8))+ (ONEMINUSOMEGA* et + DFL3_OMEGA * rho * (1.0 + (+ux + uz) * (4.5 * (+ux + uz) + 3.0) - u2))* (!(type >> 8));
//	//printf("\n(%d, %d, %d) s_ET[%d] %.8f", x, y, z, txE, s_ET[txE]);
//	s_W[txW+1] = (e * (type >> 8))	+ (ONEMINUSOMEGA* w + DFL2_OMEGA * rho * (1.0 + ux * (4.5 * ux - 3.0) - u2))* (!(type >> 8));
//	//printf("\n(%d, %d, %d) s_W[%d] %.8f", x, y, z, txW+1, s_W[txW+1]);
//	s_SW[txW+1] = (ne * (type >> 8))+ (ONEMINUSOMEGA* sw + DFL3_OMEGA * rho * (1.0 + (-ux - uy) * (4.5 * (-ux - uy) + 3.0) - u2))* (!(type >> 8));
//	//printf("\n(%d, %d, %d) s_SW[%d] %.8f", x, y, z, txW+1, s_SW[txW+1]);
//	s_NW[txW+1] = (se * (type >> 8)) + (ONEMINUSOMEGA* nw + DFL3_OMEGA * rho * (1.0 + (-ux + uy) * (4.5 * (-ux + uy) + 3.0) - u2))* (!(type >> 8));
//	//printf("\n(%d, %d, %d) s_NW[%d] %.8f", x, y, z, txW+1, s_NW[txW+1]);
//	s_WB[txW+1] = (et * (type >> 8))+ (ONEMINUSOMEGA* wb + DFL3_OMEGA * rho * (1.0 + (-ux - uz) * (4.5 * (-ux - uz) + 3.0) - u2))* (!(type >> 8));
//	//printf("\n(%d, %d, %d) s_WB[%d] %.8f", x, y, z, txW+1, s_WB[txW+1]);
//	s_WT[txW+1] = (eb * (type >> 8))+ (ONEMINUSOMEGA* wt + DFL3_OMEGA * rho * (1.0 + (-ux + uz) * (4.5 * (-ux + uz) + 3.0) - u2))* (!(type >> 8));
//	//printf("\n(%d, %d, %d) s_WT[%d] %.8f", x, y, z, txW+1, s_WT[txW+1]);
//	__syncthreads();
//
//	//if(x==1 && y==1 && z==1) printf("\nidx = %d, offset = %d", idx, offset);
//	dGrid[c_nbArray[ C] + idx] =  		  	(c * (type >> 8)) + (ONEMINUSOMEGA* c + DFL1_OMEGA * rho * (1.0 - u2))* (!(type >> 8));
//	//if(x==1 && y==1 && z==1) printf("\nC  | thread=%d, writeto=%d, sm_pos=%d", x, c_nbArray[ C] + idx, 100);
//	//if(c_nbArray[ C] + idx == 18733) printf("\nC  | x=%d, y=%d, z=%d", x, y, z);
//
//	dGrid[c_nbArray[ N] + idx + offset] =   (s * (type >> 8))  + (ONEMINUSOMEGA* n + DFL2_OMEGA * rho * (1.0 + uy * (4.5 * uy + 3.0) - u2))* (!(type >> 8));
//	//if(x==1 && y==1 && z==1) printf("\nN  | thread=%d, writeto=%d, sm_pos=%d", x, c_nbArray[ N] + idx + offset, 100);
//	//if(c_nbArray[ N] + idx + offset == 18733) printf("\nN  | x=%d, y=%d, z=%d", x, y, z);
//
//	dGrid[c_nbArray[ S] + idx + 2*offset] = (n * (type >> 8))  + (ONEMINUSOMEGA* s + DFL2_OMEGA * rho * (1.0 + uy * (4.5 * uy - 3.0) - u2))* (!(type >> 8));
//	//if(x==1 && y==1 && z==1) printf("\nS  | thread=%d, writeto=%d, sm_pos=%d", x, c_nbArray[ S] + idx + 2*offset, 100);
//	//if(c_nbArray[ S] + idx + 2*offset == 18733) printf("\nS  | x=%d, y=%d, z=%d", x, y, z);
//
//	dGrid[c_nbArray[ T] + idx + 5*offset] = (b * (type >> 8))	+ (ONEMINUSOMEGA* t + DFL2_OMEGA * rho * (1.0 + uz * (4.5 * uz + 3.0) - u2))* (!(type >> 8));
//	//if(x==1 && y==1 && z==1) printf("\nT  | thread=%d, writeto=%d, sm_pos=%d", x, c_nbArray[ T] + idx + 5*offset, 100);
//	//if(c_nbArray[ T] + idx + 5*offset == 14161) printf("\nT  | x=%d, y=%d, z=%d", x, y, z);
//
//	dGrid[c_nbArray[ B] + idx + 6*offset] = (t * (type >> 8))	+ (ONEMINUSOMEGA* b + DFL2_OMEGA * rho * (1.0 + uz * (4.5 * uz - 3.0) - u2))* (!(type >> 8));
//	//if(x==1 && y==1 && z==1) printf("\nB  | thread=%d, writeto=%d, sm_pos=%d", x, c_nbArray[ B] + idx + 6*offset, 100);
//	//if(c_nbArray[ B] + idx + 6*offset == 18733) printf("\nB  | x=%d, y=%d, z=%d", x, y, z);
//
//	dGrid[c_nbArray[NT] + idx + 11*offset] = (sb * (type >> 8))+ (ONEMINUSOMEGA* nt + DFL3_OMEGA * rho * (1.0 + (+uy + uz) * (4.5 * (+uy + uz) + 3.0) - u2))* (!(type >> 8));
//	//if(x==1 && y==1 && z==1) printf("\nNT  | thread=%d, writeto=%d, sm_pos=%d", x, c_nbArray[NT] + idx + 11*offset, 100);
//	//if(c_nbArray[NT] + idx + 11*offset == 18733) printf("\nNT  | x=%d, y=%d, z=%d", x, y, z);
//
//	dGrid[c_nbArray[NB] + idx + 12*offset] = (st * (type >> 8))+ (ONEMINUSOMEGA* nb + DFL3_OMEGA * rho * (1.0 + (+uy - uz) * (4.5 * (+uy - uz) + 3.0) - u2))* (!(type >> 8));
//	//if(x==1 && y==1 && z==1) printf("\nNB  | thread=%d, writeto=%d, sm_pos=%d", x, c_nbArray[NB] + idx + 12*offset, 100);
//	//if(c_nbArray[NB] + idx + offset == 18733) printf("\nNB  | x=%d, y=%d, z=%d", x, y, z);
//
//	dGrid[c_nbArray[ST] + idx + 13*offset] = (nb * (type >> 8))+ (ONEMINUSOMEGA* st + DFL3_OMEGA * rho * (1.0 + (-uy + uz) * (4.5 * (-uy + uz) + 3.0) - u2))* (!(type >> 8));
//	//if(x==1 && y==1 && z==1) printf("\nST  | thread=%d, writeto=%d, sm_pos=%d", x, c_nbArray[ST] + idx + 13*offset, 100);
//	//if(c_nbArray[ST] + idx + 13*offset == 18733) printf("\nST  | x=%d, y=%d, z=%d", x, y, z);
//
//	dGrid[c_nbArray[SB] + idx + 14*offset] = (nt * (type >> 8))+ (ONEMINUSOMEGA* sb + DFL3_OMEGA * rho * (1.0 + (-uy - uz) * (4.5 * (-uy - uz) + 3.0) - u2))* (!(type >> 8));
//	//if(x==1 && y==1 && z==1) printf("\nSB  | thread=%d, writeto=%d, sm_pos=%d", x, c_nbArray[SB] + idx + 14*offset, 100);
//	//if(c_nbArray[SB] + idx + 14*offset == 18733) printf("\nSB  | x=%d, y=%d, z=%d", x, y, z);
//
//	dGrid+=1;
//	dGrid[c_nbArray[ E] + idx + 3*offset - 1] = s_E[txE];
//	//if(x==1 && y==1 && z==1) printf("\nE  | thread=%d, writeto=%d, sm_pos=%d <--", x, c_nbArray[ E] + idx + 3*offset, x);
//	//if(c_nbArray[ E] + idx + 3*offset == 18733) printf("\nE  | x=%d, y=%d, z=%d", x, y, z);
//
//	dGrid[c_nbArray[NE] + idx + 7*offset - 1] = s_NE[txE];
//	//if(x==1 && y==1 && z==1) printf("\nNE  | thread=%d, writeto=%d, sm_pos=%d <--", x, c_nbArray[NE] + idx + 7*offset, x);
//	//if(c_nbArray[NE] + idx + 7*offset == 18733) printf("\nNE  | x=%d, y=%d, z=%d", x, y, z);
//
//	dGrid[c_nbArray[SE] + idx + 9*offset - 1] = s_SE[txE];
//	//if(x==1 && y==1 && z==1) printf("\nSE  | thread=%d, writeto=%d, sm_pos=%d <--", x, c_nbArray[SE] + idx + 9*offset, x);
//	//if(c_nbArray[SE] + idx + 9*offset == 18733) printf("\nSE  | x=%d, y=%d, z=%d", x, y, z);
//
//	dGrid[c_nbArray[ET] + idx + 15*offset - 1] = s_ET[txE];
//	//if(x==1 && y==1 && z==1) printf("\nET  | thread=%d, writeto=%d, sm_pos=%d <--", x, c_nbArray[ET] + idx + 15*offset, x);
//	//if(c_nbArray[ET] + idx + 15*offset == 18733) printf("\nET  | x=%d, y=%d, z=%d", x, y, z);
//
//	dGrid[c_nbArray[EB] + idx + 16*offset - 1] = s_EB[txE];
//	//if(x==1 && y==1 && z==1) printf("\nEB  | thread=%d, writeto=%d, sm_pos=%d <--", x, c_nbArray[EB] + idx + 16*offset, x);
//	//if(c_nbArray[EB] + idx + 16*offset == 18733) printf("\nEB  | x=%d, y=%d, z=%d", x, y, z);
//
//	dGrid-=2;
//	dGrid[c_nbArray[ W] + idx + 4*offset + 1] = s_W[txW+1];
//	//if(x==1 && y==1 && z==1) printf("\nW  | thread=%d, writeto=%d, sm_pos=%d <--", x, c_nbArray[ W] + idx + 4*offset, x+1);
//	//if(c_nbArray[ W] + idx + 4*offset == 18733) printf("\nW  | x=%d, y=%d, z=%d", x, y, z);
//
//	dGrid[c_nbArray[NW] + idx + 8*offset + 1] = s_NW[txW+1];
//	//if(x==1 && y==1 && z==1) printf("\nNW  | thread=%d, writeto=%d, sm_pos=%d <--", x, c_nbArray[NW] + idx + 8*offset, x+1);
//	//if(c_nbArray[NW] + idx + 8*offset == 18733) printf("\nNW  | x=%d, y=%d, z=%d", x, y, z);
//
//	dGrid[c_nbArray[SW] + idx + 10*offset + 1] = s_SW[txW+1];
//	//if(x==1 && y==1 && z==1) printf("\nSW  | thread=%d, writeto=%d, sm_pos=%d <--", x, c_nbArray[SW] + idx + 10*offset, x+1);
//	//if(c_nbArray[SW] + idx + 10*offset == 18733) printf("\nSW  | x=%d, y=%d, z=%d", x, y, z);
//
//	dGrid[c_nbArray[WT] + idx + 17*offset + 1] = s_WT[txW+1];
//	//if(x==1 && y==1 && z==1) printf("\nWT  | thread=%d, writeto=%d, sm_pos=%d <--", x, c_nbArray[WT] + idx + 17*offset, x+1);
//	//if(c_nbArray[WT] + idx + 17*offset == 18733) printf("\nWT  | x=%d, y=%d, z=%d", x, y, z);
//
//	dGrid[c_nbArray[WB] + idx + 18*offset + 1] = s_WB[txW+1];
//	//if(x==1 && y==1 && z==1) printf("\nWB  | thread=%d, writeto=%d, sm_pos=%d <--", x, c_nbArray[WB] + idx + 18*offset, x+1);
//	//if(c_nbArray[WB] + idx + 18*offset == 18733) printf("\nWB  | x=%d, y=%d, z=%d", x, y, z);
//	dGrid+=1;
//}
//backup
__global__ void /*__launch_bounds__(128,8)*/ lbm_kernel_SoA31(float *sGrid, float *dGrid, unsigned char *flags)
{
//	__shared__ float s_E[NUM_THREADS +2];
//	__shared__ float s_W[NUM_THREADS +2];
//	__shared__ float s_SE[NUM_THREADS+2];
//	__shared__ float s_NE[NUM_THREADS+2];
//	__shared__ float s_SW[NUM_THREADS+2];
//	__shared__ float s_NE[NUM_THREADS+2];
//	__shared__ float s_BE[NUM_THREADS+2];
//	__shared__ float s_TE[NUM_THREADS+2];
//	__shared__ float s_BW[NUM_THREADS+2];
//	__shared__ float s_TW[NUM_THREADS+2];

	//float *shPropPointer = (float*)array;
	__shared__ int offset;

	//__shared__ float one_minus_omega;
	offset = SIZE_XX*SIZE_YY*SIZE_ZZ;
	//one_minus_omega = 1.0 - OMEGA;
	int x = threadIdx.x;
	int y = blockIdx.x + 1;
	int z = blockIdx.y + 1;

	//int ElementsPerBlock = blockDim.x;
	//float c, n, s, e, w, t, b, ne, nw, se, sw, nt, nb, st, sb, et,	eb, wt, wb;

	/*the grid is organized as follows:
	 * c c c c c c . . . | n n n n n n . . . | s s s s s s . . . | e e e e e e . . . | . . .
	 */

	//calculate the index
	//int idx = (z * SIZE_YY + y)*SIZE_XX + x;
	//int idx = ((blockIdx.y +1) * SIZE_YY + (blockIdx.x+1))*SIZE_XX + threadIdx.x;
	int idx = CALC_INDEX_SOA_WO_STRUCT(x,y,z,0);

	//each direction has offset = 130*100*100 values, we need to jump one offset to get new value
	//int offset = SIZE_XX*SIZE_YY*SIZE_ZZ;
	//save index to address for operating

	int address = idx; //if(y==1 && x==0 && z==1) printf("\nc src(%d)", address);
	float c = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nn src(%d)", address);
	float n = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\ns src(%d)", address);
	float s = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\ne src(%d)", address);
	float e = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nw src(%d)", address);
	float w = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nt src(%d)", address);
	float t = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nb src(%d)", address);
	float b = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nne src(%d)", address);
	float ne = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nnw src(%d)", address);
	float nw = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nse src(%d)", address);
	float se = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nsw src(%d)", address);
	float sw = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nnt src(%d)", address);
	float nt = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nnb src(%d)", address);
	float nb = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nst src(%d)", address);
	float st = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nsb src(%d)", address);
	float sb = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\net src(%d)", address);
	float et = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\neb src(%d)", address);
	float eb = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nwt src(%d)", address);
	float wt = sGrid[address];
	address = address + offset; //if(y==1 && x==0 && z==1) printf("\nwb src(%d)", address);
	float wb = sGrid[address];

	//get the value final cell (20) to know the kind of cell: fluid or accel or obstacle
	//int flag = flags[idx];

	unsigned char isobs = (flags[idx] == 1);
	unsigned char isacc = (flags[idx] == 2);

	float rho = +c + n + s + e + w + t + b + ne + nw + se + sw + nt + nb + st + sb + et + eb + wt + wb;

	float ux = ((+e - w + ne - nw + se - sw + et + eb - wt - wb)/rho)*(!isacc) + 0.005*isacc;
	float uy = ((+n - s + ne + nw - se - sw + nt + nb - st - sb)/rho)*(!isacc) + 0.002*isacc;
	float uz = ((+t - b + nt - nb + st - sb + et - eb + wt - wb)/rho)*(!isacc);

	//float ux = +e - w + ne - nw + se - sw + et + eb - wt - wb;
	//float uy = +n - s + ne + nw - se - sw + nt + nb - st - sb;
	//float uz = +t - b + nt - nb + st - sb + et - eb + wt - wb;

	//ux = (ux / rho) * (!isacc) + 0.005 * isacc;
	//uy = (uy / rho) * (!isacc) + 0.002 * isacc;
	//uz = (uz / rho) * (!isacc) + 0.000 * isacc;

	float u2 = 1.5 * (ux * ux + uy * uy + uz * uz); //U2(ux,uy,uz);//
//	float u2 = u2_func(ux,uy,uz);


	address = c_nbArray[ 0] + idx; //if(y==1 && x==0 && z==1) printf("\n => c dst(%d)", address);
	dGrid[c_nbArray[ 0] + idx] =  		  	(c * isobs) + (ONEMINUSOMEGA* c + DFL1_OMEGA * rho * (1.0 - u2))* (!isobs);
	address = c_nbArray[ 1] + idx + offset; //if(y==1 && x==0 && z==1) printf("\n => n dst(%d)", address);
	dGrid[c_nbArray[ 1] + idx + offset] =   (s * isobs)  + (ONEMINUSOMEGA* n + DFL2_OMEGA * rho * (1.0 + uy * (4.5 * uy + 3.0) - u2))* (!isobs);
	address = c_nbArray[ 2] + idx + 2*offset; //if(y==1 && x==0 && z==1) printf("\n => s dst(%d)", address);
	dGrid[c_nbArray[ 2] + idx + 2*offset] = (n * isobs)  + (ONEMINUSOMEGA* s + DFL2_OMEGA * rho * (1.0 + uy * (4.5 * uy - 3.0) - u2))* (!isobs);
	address = c_nbArray[ 3] + idx + 3*offset; //if(y==1 && x==0 && z==1) printf("\n => e dst(%d)", address);
	dGrid[c_nbArray[ 3] + idx + 3*offset] = (w * isobs)	+ (ONEMINUSOMEGA* e + DFL2_OMEGA * rho * (1.0 + ux * (4.5 * ux + 3.0) - u2))* (!isobs);
	address = c_nbArray[ 4] + idx + 4*offset; //if(y==1 && x==0 && z==1) printf("\n => w dst(%d)", address);
	dGrid[c_nbArray[ 4] + idx + 4*offset] = (e * isobs)	+ (ONEMINUSOMEGA* w + DFL2_OMEGA * rho * (1.0 + ux * (4.5 * ux - 3.0) - u2))* (!isobs);
	address = c_nbArray[ 5] + idx + 5*offset; //if(y==1 && x==0 && z==1) printf("\n => t dst(%d)", address);
	dGrid[c_nbArray[ 5] + idx + 5*offset] = (b * isobs)	+ (ONEMINUSOMEGA* t + DFL2_OMEGA * rho * (1.0 + uz * (4.5 * uz + 3.0) - u2))* (!isobs);
	address = c_nbArray[ 6] + idx + 6*offset; //if(y==1 && x==0 && z==1) printf("\n => b dst(%d)", address);
	dGrid[c_nbArray[ 6] + idx + 6*offset] = (t * isobs)	+ (ONEMINUSOMEGA* b + DFL2_OMEGA * rho * (1.0 + uz * (4.5 * uz - 3.0) - u2))* (!isobs);
	address = c_nbArray[ 7] + idx + 7*offset; //if(y==1 && x==0 && z==1) printf("\n => ne dst(%d)", address);
	dGrid[c_nbArray[ 7] + idx + 7*offset] = (sw * isobs) + (ONEMINUSOMEGA* ne + DFL3_OMEGA * rho * (1.0 + (+ux + uy) * (4.5 * (+ux + uy) + 3.0) - u2))* (!isobs);
	address = c_nbArray[ 8] + idx + 8*offset; //if(y==1 && x==0 && z==1) printf("\n => nw dst(%d)", address);
	dGrid[c_nbArray[ 8] + idx + 8*offset] = (se * isobs) + (ONEMINUSOMEGA* nw + DFL3_OMEGA * rho * (1.0 + (-ux + uy) * (4.5 * (-ux + uy) + 3.0) - u2))* (!isobs);
	address = c_nbArray[ 9] + idx + 9*offset; //if(y==1 && x==0 && z==1) printf("\n => se dst(%d)", address);
	dGrid[c_nbArray[ 9] + idx + 9*offset] = (nw * isobs) + (ONEMINUSOMEGA* se + DFL3_OMEGA * rho * (1.0 + (+ux - uy) * (4.5 * (+ux - uy) + 3.0) - u2))* (!isobs);
	address = c_nbArray[10] + idx + 10*offset; //if(y==1 && x==0 && z==1) printf("\n => sw dst(%d)", address);
	dGrid[c_nbArray[10] + idx + 10*offset] = (ne * isobs)+ (ONEMINUSOMEGA* sw + DFL3_OMEGA * rho * (1.0 + (-ux - uy) * (4.5 * (-ux - uy) + 3.0) - u2))* (!isobs);
	address = c_nbArray[11] + idx + 11*offset; //if(y==1 && x==0 && z==1) printf("\n => nt dst(%d)", address);
	dGrid[c_nbArray[11] + idx + 11*offset] = (sb * isobs)+ (ONEMINUSOMEGA* nt + DFL3_OMEGA * rho * (1.0 + (+uy + uz) * (4.5 * (+uy + uz) + 3.0) - u2))* (!isobs);
	address = c_nbArray[12] + idx + 12*offset; //if(y==1 && x==0 && z==1) printf("\n => nb dst(%d)", address);
	dGrid[c_nbArray[12] + idx + 12*offset] = (st * isobs)+ (ONEMINUSOMEGA* nb + DFL3_OMEGA * rho * (1.0 + (+uy - uz) * (4.5 * (+uy - uz) + 3.0) - u2))* (!isobs);
	address = c_nbArray[13] + idx + 13*offset; //if(y==1 && x==0 && z==1) printf("\n => st dst(%d)", address);
	dGrid[c_nbArray[13] + idx + 13*offset] = (nb * isobs)+ (ONEMINUSOMEGA* st + DFL3_OMEGA * rho * (1.0 + (-uy + uz) * (4.5 * (-uy + uz) + 3.0) - u2))* (!isobs);
	address = c_nbArray[14] + idx + 14*offset; //if(y==1 && x==0 && z==1) printf("\n => sb dst(%d)", address);
	dGrid[c_nbArray[14] + idx + 14*offset] = (nt * isobs)+ (ONEMINUSOMEGA* sb + DFL3_OMEGA * rho * (1.0 + (-uy - uz) * (4.5 * (-uy - uz) + 3.0) - u2))* (!isobs);
	address = c_nbArray[15] + idx + 15*offset; //if(y==1 && x==0 && z==1) printf("\n => et dst(%d)", address);
	dGrid[c_nbArray[15] + idx + 15*offset] = (wb * isobs)+ (ONEMINUSOMEGA* et + DFL3_OMEGA * rho * (1.0 + (+ux + uz) * (4.5 * (+ux + uz) + 3.0) - u2))* (!isobs);
	address = c_nbArray[16] + idx + 16*offset; //if(y==1 && x==0 && z==1) printf("\n => eb dst(%d)", address);
	dGrid[c_nbArray[16] + idx + 16*offset] = (wt * isobs)+ (ONEMINUSOMEGA* eb + DFL3_OMEGA * rho * (1.0 + (+ux - uz) * (4.5 * (+ux - uz) + 3.0) - u2))* (!isobs);
	address = c_nbArray[17] + idx + 17*offset; //if(y==1 && x==0 && z==1) printf("\n => wt dst(%d)", address);
	dGrid[c_nbArray[17] + idx + 17*offset] = (eb * isobs)+ (ONEMINUSOMEGA* wt + DFL3_OMEGA * rho * (1.0 + (-ux + uz) * (4.5 * (-ux + uz) + 3.0) - u2))* (!isobs);
	address = c_nbArray[18] + idx + 18*offset; //if(y==1 && x==0 && z==1) printf("\n => wb dst(%d)", address);
	dGrid[c_nbArray[18] + idx + 18*offset] = (et * isobs)+ (ONEMINUSOMEGA* wb + DFL3_OMEGA * rho * (1.0 + (-ux - uz) * (4.5 * (-ux - uz) + 3.0) - u2))* (!isobs);

}
//SoA Pull
__global__ void lbm_kernel_SoA_Pull(float* sGrid, float* dGrid, const unsigned char* __restrict__ flags)
{
	__shared__ int offset;

	//__shared__ float one_minus_omega;
	offset = SIZE_XX*SIZE_YY*SIZE_ZZ;
	//one_minus_omega = 1.0 - OMEGA;
	int x = threadIdx.x;
	int y = blockIdx.x;// + 1;
	int z = blockIdx.y;// + 1;

	//int ElementsPerBlock = blockDim.x;
	//float c, n, s, e, w, t, b, ne, nw, se, sw, nt, nb, st, sb, et,	eb, wt, wb;

	/*the grid is organized as follows:
	 * c c c c c c . . . | n n n n n n . . . | s s s s s s . . . | e e e e e e . . . | . . .
	 */

	//calculate the index
	//int idx = (z * SIZE_YY + y)*SIZE_XX + x;
	//int idx = ((blockIdx.y +1) * SIZE_YY + (blockIdx.x+1))*SIZE_XX + threadIdx.x;
	int idx = CALC_INDEX_SOA_WO_STRUCT(x, y, z, 0) + MARGIN_L_SIZE;

	//each direction has offset = 130*100*100 values, we need to jump one offset to get new value
	//int offset = SIZE_XX*SIZE_YY*SIZE_ZZ;
	//save index to address for operating
	//int address = offset;
	//int address = -c_nbArray[ 0] + idx; //if(y==1 && x==0 && z==1) printf("\nc src(%d)", address);
	float c = dGrid[-c_nbArray[ 0] + idx]; //if(y==1 && x==0 && z==1) printf("\nc = %.5f", c);
	//address = -c_nbArray[ 1] + idx + offset; ////if(y==1 && x==0 && z==1) printf("\nn src(%d)", address);
	float n = dGrid[-c_nbArray[ 1] + idx + offset];//if(y==1 && x==0 && z==1) printf("\nn = %.5f", n);
	//address = -c_nbArray[ 2] + idx + 2*offset; //if(y==1 && x==0 && z==1) printf("\ns src(%d)", address);
	float s = dGrid[-c_nbArray[ 2] + idx + 2*offset]; //if(y==1 && x==0 && z==1) printf("\ns = %.5f", s);
	//address = -c_nbArray[ 3] + idx + 3*offset; //if(y==1 && x==0 && z==1) printf("\ne src(%d)", address);
	float e = dGrid[-c_nbArray[ 3] + idx + 3*offset]; //if(y==1 && x==0 && z==1) printf("\ne = %.5f", e);
	//address = -c_nbArray[ 4] + idx + 4*offset; //if(y==1 && x==0 && z==1) printf("\nw src(%d)", address);
	float w = dGrid[-c_nbArray[ 4] + idx + 4*offset]; //if(y==1 && x==0 && z==1) printf("\nw = %.5f", w);
	//address = -c_nbArray[ 5] + idx + 5*offset; //if(y==1 && x==0 && z==1) printf("\nt src(%d)", address);
	float t = dGrid[-c_nbArray[ 5] + idx + 5*offset]; //if(y==1 && x==0 && z==1) printf("\nt = %.5f", t);
	//address = -c_nbArray[ 6] + idx + 6*offset; //if(y==1 && x==0 && z==1) printf("\nb src(%d)", address);
	float b = dGrid[-c_nbArray[ 6] + idx + 6*offset]; //if(y==1 && x==0 && z==1) printf("\nb = %.5f", b);
	//address = -c_nbArray[ 7] + idx + 7*offset; //if(y==1 && x==0 && z==1) printf("\nne src(%d)", address);
	float ne = dGrid[-c_nbArray[ 7] + idx + 7*offset]; //if(y==1 && x==0 && z==1) printf("\nne = %.5f", ne);
	//address = -c_nbArray[ 8] + idx + 8*offset; //if(y==1 && x==0 && z==1) printf("\nnw src(%d)", address);
	float nw = dGrid[-c_nbArray[ 8] + idx + 8*offset]; //if(y==1 && x==0 && z==1) printf("\nnw = %.5f", nw);
	//address = -c_nbArray[ 9] + idx + 9*offset; //if(y==1 && x==0 && z==1) printf("\nse src(%d)", address);
	float se = dGrid[-c_nbArray[ 9] + idx + 9*offset]; //if(y==1 && x==0 && z==1) printf("\nse = %.5f", se);
	//address = -c_nbArray[10] + idx + 10*offset; //if(y==1 && x==0 && z==1) printf("\nsw src(%d)", address);
	float sw = dGrid[-c_nbArray[10] + idx + 10*offset]; //if(y==1 && x==0 && z==1) printf("\nsw = %.5f", sw);
	//address = -c_nbArray[11] + idx + 11*offset; //if(y==1 && x==0 && z==1) printf("\nnt src(%d)", address);
	float nt = dGrid[-c_nbArray[11] + idx + 11*offset]; //if(y==1 && x==0 && z==1) printf("\nnt = %.5f", nt);
	//address = -c_nbArray[12] + idx + 12*offset; //if(y==1 && x==0 && z==1) printf("\nnb src(%d)", address);
	float nb = dGrid[-c_nbArray[12] + idx + 12*offset]; //if(y==1 && x==0 && z==1) printf("\nnb = %.5f", nb);
	//address = -c_nbArray[13] + idx + 13*offset; //if(y==1 && x==0 && z==1) printf("\nst src(%d)", address);
	float st = dGrid[-c_nbArray[13] + idx + 13*offset]; //if(y==1 && x==0 && z==1) printf("\nst = %.5f", st);
	//address = -c_nbArray[14] + idx + 14*offset; //if(y==1 && x==0 && z==1) printf("\nsb src(%d)", address);
	float sb = dGrid[-c_nbArray[14] + idx + 14*offset]; //if(y==1 && x==0 && z==1) printf("\nsb = %.5f", sb);
	//address = -c_nbArray[15] + idx + 15*offset; //if(y==1 && x==0 && z==1) printf("\net src(%d)", address);
	float et = dGrid[-c_nbArray[15] + idx + 15*offset]; //if(y==1 && x==0 && z==1) printf("\net = %.5f", et);
	//address = -c_nbArray[16] + idx + 16*offset; //if(y==1 && x==0 && z==1) printf("\neb src(%d)", address);
	float eb = dGrid[-c_nbArray[16] + idx + 16*offset]; //if(y==1 && x==0 && z==1) printf("\neb = %.5f", eb);
	//address = -c_nbArray[17] + idx + 17*offset; //if(y==1 && x==0 && z==1) printf("\nwt src(%d)", address);
	float wt = dGrid[-c_nbArray[17] + idx + 17*offset]; //if(y==1 && x==0 && z==1) printf("\nwt = %.5f", wt);
	//address = -c_nbArray[18] + idx + 18*offset; //if(y==1 && x==0 && z==1) printf("\nwb src(%d)", address);
	float wb = dGrid[-c_nbArray[18] + idx + 18*offset]; //if(y==1 && x==0 && z==1) printf("\nwb = %.5f", wb);

	//get the value final cell (20) to know the kind of cell: fluid or accel or obstacle
	//int flag = flags[idx];
	unsigned short type = ((flags[idx-MARGIN_L_SIZE] == 1) << 8) | ((flags[idx-MARGIN_L_SIZE] == 2) & 0xff);
	//int isobs = (flags[idx] == 1);
	//int isacc = (flags[idx] == 2);

	float rho = +c + n + s + e + w + t + b + ne + nw + se + sw + nt + nb + st + sb + et + eb + wt + wb;

	float ux = ((+e - w + ne - nw + se - sw + et + eb - wt - wb)/rho)*(!(type & 0xff)) + 0.005*(type & 0xff);
	float uy = ((+n - s + ne + nw - se - sw + nt + nb - st - sb)/rho)*(!(type & 0xff)) + 0.002*(type & 0xff);
	float uz = ((+t - b + nt - nb + st - sb + et - eb + wt - wb)/rho)*(!(type & 0xff));


	float u2 = 1.5 * (ux * ux + uy * uy + uz * uz);


	sGrid[idx] = (c * (type >> 8))	+ ((1.0 - OMEGA)* c + DFL1 * OMEGA * rho * (1.0 - u2))* (!(type >> 8));

	c = (type >> 8);
	//address = offset;
	sGrid[idx + offset] = (s * c)  + ((1.0 - OMEGA)* n + DFL2 * OMEGA * rho * (1.0 + uy * (4.5 * uy + 3.0) - u2))* (!c);

	sGrid[idx + 2*offset] = (n * c)  + ((1.0 - OMEGA)* s + DFL2 * OMEGA * rho * (1.0 + uy * (4.5 * uy - 3.0) - u2))* (!c);

	n = (1.0 - OMEGA); //resue variable n
	s = DFL2 * OMEGA * rho; //resue variable s

	sGrid[idx + 3*offset] = (w * c)	+ (n* e + s * (1.0 + ux * (4.5 * ux + 3.0) - u2))* (!c);

	sGrid[idx + 4*offset] = (e * c)	+ (n* w + s * (1.0 + ux * (4.5 * ux - 3.0) - u2))* (!c);

	sGrid[idx + 5*offset] = (b * c)	+ (n* t + s * (1.0 + uz * (4.5 * uz + 3.0) - u2))* (!c);

	sGrid[idx + 6*offset] = (t * c)	+ (n* b + s  * (1.0 + uz * (4.5 * uz - 3.0) - u2))* (!c);

	b = DFL3 * OMEGA * rho; //resue variable b
	t = 1.0 - u2;
	e = ux + uy;
	w = b*(t+ 4.5*e*e);

	sGrid[idx + 7*offset] = (sw * c) + (n* ne + w + 3*e*b)* (!c);

	sGrid[idx + 10*offset] = (ne * c)+ (n* sw + w - 3*e*b)* (!c);

	e = -ux + uy;
	w = b*(t+ 4.5*e*e);

	sGrid[idx + 8*offset] = (se * c) + (n* nw + w + 3*e*b)* (!c);

	sGrid[idx + 9*offset] = (nw * c) + (n* se + w - 3*e*b)* (!c);

	e = uy + uz;
	w = b*(t+ 4.5*e*e);
	sGrid[idx + 11*offset] = (sb * c)+ (n* nt + w + 3*e*b)* (!c);
	sGrid[idx + 14*offset] = (nt * c)+ (n* sb + w - 3*e*b)* (!c);

	e = uy - uz;
	w = b*(t+ 4.5*e*e);
	//address += offset;
	sGrid[idx + 12*offset] = (st * c)+ (n* nb + w + 3*e*b)* (!c);

	sGrid[idx + 13*offset] = (nb * c)+ (n* st + w - 3*e*b)* (!c);

	e = ux + uz;
	w = b*(t+ 4.5*e*e);
	sGrid[idx + 15*offset] = (wb * c)+ (n* et + w + 3*e*b)* (!c);

	sGrid[idx + 18*offset] = (et * c)+ (n* wb + w - 3*e*b)* (!c);

	e = ux - uz;
	w = b*(t+ 4.5*e*e);

	sGrid[idx + 16*offset] = (wt * c)+ (n* eb + w + 3*e*b)* (!c);

	sGrid[idx + 17*offset] = (eb * c)+ (n* wt + w - 3*e*b)* (!c);


}
__global__ void lbm_kernel_SoA_Pull_CG(float* sGrid, float* dGrid, const unsigned char* __restrict__ flags)
{
	__shared__ int offset;

	offset = SIZE_X*SIZE_Y*SIZE_Z;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	//int x_tile = threadIdx.x + 1;
	//int y_tile = threadIdx.y + 1;

	float c, n, s, e, w, t, b, ne, nw, se, sw, nt, nb, st, sb, et, eb, wt, wb;
	unsigned short type;
	float rho;
	float ux;
	float uy;
	float uz;
	float u2;

	int startz = (blockIdx.z)*(TILED_WIDTH_Z);// (SIZE_Z/(gridDim.z));
	int endz = startz + TILED_WIDTH_Z; //((blockIdx.z) + 1) *(TILED_WIDTH_Z);// (SIZE_Z/(gridDim.z));

	int idx = MARGIN_L_SIZE + index3D(SIZE_X, SIZE_Y, x, y, startz);

	//each direction has offset = 130*100*100 values, we need to jump one offset to get new value
	//int offset = SIZE_XX*SIZE_YY*SIZE_ZZ;
	//save index to address for operating
	//int address = offset;
	for(int z=startz; z<endz; z++)
	{
		//int address = -c_nbArray[ 0] + idx; //if(y==1 && x==0 && z==1) printf("\nc src(%d)", address);
		c = dGrid[-c_nbArray[ 0] + idx]; //if(y==1 && x==0 && z==1) printf("\nc = %.5f", c);
		//address = -c_nbArray[ 1] + idx + offset; ////if(y==1 && x==0 && z==1) printf("\nn src(%d)", address);
		n = dGrid[-c_nbArray[ 1] + idx + offset];//if(y==1 && x==0 && z==1) printf("\nn = %.5f", n);
		//address = -c_nbArray[ 2] + idx + 2*offset; //if(y==1 && x==0 && z==1) printf("\ns src(%d)", address);
		s = dGrid[-c_nbArray[ 2] + idx + 2*offset]; //if(y==1 && x==0 && z==1) printf("\ns = %.5f", s);
		//address = -c_nbArray[ 3] + idx + 3*offset; //if(y==1 && x==0 && z==1) printf("\ne src(%d)", address);
		e = dGrid[-c_nbArray[ 3] + idx + 3*offset]; //if(y==1 && x==0 && z==1) printf("\ne = %.5f", e);
		//address = -c_nbArray[ 4] + idx + 4*offset; //if(y==1 && x==0 && z==1) printf("\nw src(%d)", address);
		w = dGrid[-c_nbArray[ 4] + idx + 4*offset]; //if(y==1 && x==0 && z==1) printf("\nw = %.5f", w);
		//address = -c_nbArray[ 5] + idx + 5*offset; //if(y==1 && x==0 && z==1) printf("\nt src(%d)", address);
		t = dGrid[-c_nbArray[ 5] + idx + 5*offset]; //if(y==1 && x==0 && z==1) printf("\nt = %.5f", t);
		//address = -c_nbArray[ 6] + idx + 6*offset; //if(y==1 && x==0 && z==1) printf("\nb src(%d)", address);
		b = dGrid[-c_nbArray[ 6] + idx + 6*offset]; //if(y==1 && x==0 && z==1) printf("\nb = %.5f", b);
		//address = -c_nbArray[ 7] + idx + 7*offset; //if(y==1 && x==0 && z==1) printf("\nne src(%d)", address);
		ne = dGrid[-c_nbArray[ 7] + idx + 7*offset]; //if(y==1 && x==0 && z==1) printf("\nne = %.5f", ne);
		//address = -c_nbArray[ 8] + idx + 8*offset; //if(y==1 && x==0 && z==1) printf("\nnw src(%d)", address);
		nw = dGrid[-c_nbArray[ 8] + idx + 8*offset]; //if(y==1 && x==0 && z==1) printf("\nnw = %.5f", nw);
		//address = -c_nbArray[ 9] + idx + 9*offset; //if(y==1 && x==0 && z==1) printf("\nse src(%d)", address);
		se = dGrid[-c_nbArray[ 9] + idx + 9*offset]; //if(y==1 && x==0 && z==1) printf("\nse = %.5f", se);
		//address = -c_nbArray[10] + idx + 10*offset; //if(y==1 && x==0 && z==1) printf("\nsw src(%d)", address);
		sw = dGrid[-c_nbArray[10] + idx + 10*offset]; //if(y==1 && x==0 && z==1) printf("\nsw = %.5f", sw);
		//address = -c_nbArray[11] + idx + 11*offset; //if(y==1 && x==0 && z==1) printf("\nnt src(%d)", address);
		nt = dGrid[-c_nbArray[11] + idx + 11*offset]; //if(y==1 && x==0 && z==1) printf("\nnt = %.5f", nt);
		//address = -c_nbArray[12] + idx + 12*offset; //if(y==1 && x==0 && z==1) printf("\nnb src(%d)", address);
		nb = dGrid[-c_nbArray[12] + idx + 12*offset]; //if(y==1 && x==0 && z==1) printf("\nnb = %.5f", nb);
		//address = -c_nbArray[13] + idx + 13*offset; //if(y==1 && x==0 && z==1) printf("\nst src(%d)", address);
		st = dGrid[-c_nbArray[13] + idx + 13*offset]; //if(y==1 && x==0 && z==1) printf("\nst = %.5f", st);
		//address = -c_nbArray[14] + idx + 14*offset; //if(y==1 && x==0 && z==1) printf("\nsb src(%d)", address);
		sb = dGrid[-c_nbArray[14] + idx + 14*offset]; //if(y==1 && x==0 && z==1) printf("\nsb = %.5f", sb);
		//address = -c_nbArray[15] + idx + 15*offset; //if(y==1 && x==0 && z==1) printf("\net src(%d)", address);
		et = dGrid[-c_nbArray[15] + idx + 15*offset]; //if(y==1 && x==0 && z==1) printf("\net = %.5f", et);
		//address = -c_nbArray[16] + idx + 16*offset; //if(y==1 && x==0 && z==1) printf("\neb src(%d)", address);
		eb = dGrid[-c_nbArray[16] + idx + 16*offset]; //if(y==1 && x==0 && z==1) printf("\neb = %.5f", eb);
		//address = -c_nbArray[17] + idx + 17*offset; //if(y==1 && x==0 && z==1) printf("\nwt src(%d)", address);
		wt = dGrid[-c_nbArray[17] + idx + 17*offset]; //if(y==1 && x==0 && z==1) printf("\nwt = %.5f", wt);
		//address = -c_nbArray[18] + idx + 18*offset; //if(y==1 && x==0 && z==1) printf("\nwb src(%d)", address);
		wb = dGrid[-c_nbArray[18] + idx + 18*offset]; //if(y==1 && x==0 && z==1) printf("\nwb = %.5f", wb);

		//get the value final cell (20) to know the kind of cell: fluid or accel or obstacle
		//int flag = flags[idx];
		type = ((flags[idx-MARGIN_L_SIZE] == 1) << 8) | ((flags[idx-MARGIN_L_SIZE] == 2) & 0xff);
		//int isobs = (flags[idx] == 1);
		//int isacc = (flags[idx] == 2);

		rho = +c + n + s + e + w + t + b + ne + nw + se + sw + nt + nb + st + sb + et + eb + wt + wb;

		ux = ((+e - w + ne - nw + se - sw + et + eb - wt - wb)/rho)*(!(type & 0xff)) + 0.005*(type & 0xff);
		uy = ((+n - s + ne + nw - se - sw + nt + nb - st - sb)/rho)*(!(type & 0xff)) + 0.002*(type & 0xff);
		uz = ((+t - b + nt - nb + st - sb + et - eb + wt - wb)/rho)*(!(type & 0xff));


		u2 = 1.5 * (ux * ux + uy * uy + uz * uz);


		sGrid[idx] = (c * (type >> 8))	+ ((1.0 - OMEGA)* c + DFL1 * OMEGA * rho * (1.0 - u2))* (!(type >> 8));

		c = (type >> 8);
		//address = offset;
		sGrid[idx + offset] = (s * c)  + ((1.0 - OMEGA)* n + DFL2 * OMEGA * rho * (1.0 + uy * (4.5 * uy + 3.0) - u2))* (!c);

		sGrid[idx + 2*offset] = (n * c)  + ((1.0 - OMEGA)* s + DFL2 * OMEGA * rho * (1.0 + uy * (4.5 * uy - 3.0) - u2))* (!c);

		n = (1.0 - OMEGA); //resue variable n
		s = DFL2 * OMEGA * rho; //resue variable s

		sGrid[idx + 3*offset] = (w * c)	+ (n* e + s * (1.0 + ux * (4.5 * ux + 3.0) - u2))* (!c);

		sGrid[idx + 4*offset] = (e * c)	+ (n* w + s * (1.0 + ux * (4.5 * ux - 3.0) - u2))* (!c);

		sGrid[idx + 5*offset] = (b * c)	+ (n* t + s * (1.0 + uz * (4.5 * uz + 3.0) - u2))* (!c);

		sGrid[idx + 6*offset] = (t * c)	+ (n* b + s  * (1.0 + uz * (4.5 * uz - 3.0) - u2))* (!c);

		b = DFL3 * OMEGA * rho; //resue variable b
		t = 1.0 - u2;
		e = ux + uy;
		w = b*(t+ 4.5*e*e);

		sGrid[idx + 7*offset] = (sw * c) + (n* ne + w + 3*e*b)* (!c);

		sGrid[idx + 10*offset] = (ne * c)+ (n* sw + w - 3*e*b)* (!c);

		e = -ux + uy;
		w = b*(t+ 4.5*e*e);

		sGrid[idx + 8*offset] = (se * c) + (n* nw + w + 3*e*b)* (!c);

		sGrid[idx + 9*offset] = (nw * c) + (n* se + w - 3*e*b)* (!c);

		e = uy + uz;
		w = b*(t+ 4.5*e*e);
		sGrid[idx + 11*offset] = (sb * c)+ (n* nt + w + 3*e*b)* (!c);
		sGrid[idx + 14*offset] = (nt * c)+ (n* sb + w - 3*e*b)* (!c);

		e = uy - uz;
		w = b*(t+ 4.5*e*e);
		//address += offset;
		sGrid[idx + 12*offset] = (st * c)+ (n* nb + w + 3*e*b)* (!c);

		sGrid[idx + 13*offset] = (nb * c)+ (n* st + w - 3*e*b)* (!c);

		e = ux + uz;
		w = b*(t+ 4.5*e*e);
		sGrid[idx + 15*offset] = (wb * c)+ (n* et + w + 3*e*b)* (!c);

		sGrid[idx + 18*offset] = (et * c)+ (n* wb + w - 3*e*b)* (!c);

		e = ux - uz;
		w = b*(t+ 4.5*e*e);

		sGrid[idx + 16*offset] = (wt * c)+ (n* eb + w + 3*e*b)* (!c);

		sGrid[idx + 17*offset] = (eb * c)+ (n* wt + w - 3*e*b)* (!c);

		idx = MARGIN_L_SIZE + index3D(SIZE_X, SIZE_Y, x, y, z);
	}
}
__global__ void __launch_bounds__(128,8) lbm_kernel_SoA32(float *sGrid, float *dGrid, int *flags)
{
//	__shared__ float s_E[NUM_THREADS +2];
//	__shared__ float s_W[NUM_THREADS +2];
//	__shared__ float s_SE[NUM_THREADS+2];
//	__shared__ float s_NE[NUM_THREADS+2];
//	__shared__ float s_SW[NUM_THREADS+2];
//	__shared__ float s_NE[NUM_THREADS+2];
//	__shared__ float s_BE[NUM_THREADS+2];
//	__shared__ float s_TE[NUM_THREADS+2];
//	__shared__ float s_BW[NUM_THREADS+2];
//	__shared__ float s_TW[NUM_THREADS+2];

	//float *shPropPointer = (float*)array;
	__shared__ int offset;
	__shared__ float uvalues[128*5];

	//__shared__ float one_minus_omega;
	offset = SIZE_XX*SIZE_YY*SIZE_ZZ;
	//one_minus_omega = 1.0 - OMEGA;
	int x = threadIdx.x;
	int y = blockIdx.x + 1;
	int z = blockIdx.y + 1;

	//int ElementsPerBlock = blockDim.x;
	//float c, n, s, e, w, t, b, ne, nw, se, sw, nt, nb, st, sb, et,	eb, wt, wb;

	/*the grid is organized as follows:
	 * c c c c c c . . . | n n n n n n . . . | s s s s s s . . . | e e e e e e . . . | . . .
	 */

	//calculate the index
	//int idx = (z * SIZE_YY + y)*SIZE_XX + x;
	//int idx = ((blockIdx.y +1) * SIZE_YY + (blockIdx.x+1))*SIZE_XX + threadIdx.x;
	int idx = CALC_INDEX_SOA_WO_STRUCT(x,y,z,0);

	//each direction has offset = 130*100*100 values, we need to jump one offset to get new value
	//int offset = SIZE_XX*SIZE_YY*SIZE_ZZ;
	//save index to address for operating

	int address = idx;
	float c = sGrid[address];
	address = address + offset;
	float n = sGrid[address];
	address = address + offset;
	float s = sGrid[address];
	address = address + offset;
	float e = sGrid[address];
	address = address + offset;
	float w = sGrid[address];
	address = address + offset;
	float t = sGrid[address];
	address = address + offset;
	float b = sGrid[address];
	address = address + offset;
	float ne = sGrid[address];
	address = address + offset;
	float nw = sGrid[address];
	address = address + offset;
	float se = sGrid[address];
	address = address + offset;
	float sw = sGrid[address];
	address = address + offset;
	float nt = sGrid[address];
	address = address + offset;
	float nb = sGrid[address];
	address = address + offset;
	float st = sGrid[address];
	address = address + offset;
	float sb = sGrid[address];
	address = address + offset;
	float et = sGrid[address];
	address = address + offset;
	float eb = sGrid[address];
	address = address + offset;
	float wt = sGrid[address];
	address = address + offset;
	float wb = sGrid[address];

	//get the value final cell (20) to know the kind of cell: fluid or accel or obstacle
	//int flag = flags[idx];

	int isobs = (flags[idx] == 1);
	int isacc = (flags[idx] == 2);

	//float rho = +c + n + s + e + w + t + b + ne + nw + se + sw + nt + nb + st + sb + et + eb + wt + wb;
	uvalues[x*5] = +c + n + s + e + w + t + b + ne + nw + se + sw + nt + nb + st + sb + et + eb + wt + wb;

	//float ux = ((+e - w + ne - nw + se - sw + et + eb - wt - wb)/rho)*(!isacc) + 0.005*isacc;
	//float uy = ((+n - s + ne + nw - se - sw + nt + nb - st - sb)/rho)*(!isacc) + 0.002*isacc;
	//float uz = ((+t - b + nt - nb + st - sb + et - eb + wt - wb)/rho)*(!isacc);

	uvalues[x*5 +1] = ((+e - w + ne - nw + se - sw + et + eb - wt - wb)/uvalues[x*5])*(!isacc) + 0.005*isacc;
	uvalues[x*5 +2] = ((+n - s + ne + nw - se - sw + nt + nb - st - sb)/uvalues[x*5])*(!isacc) + 0.002*isacc;
	uvalues[x*5 +3] = ((+t - b + nt - nb + st - sb + et - eb + wt - wb)/uvalues[x*5])*(!isacc);

	//float ux = +e - w + ne - nw + se - sw + et + eb - wt - wb;
	//float uy = +n - s + ne + nw - se - sw + nt + nb - st - sb;
	//float uz = +t - b + nt - nb + st - sb + et - eb + wt - wb;

	//ux = (ux / rho) * (!isacc) + 0.005 * isacc;
	//uy = (uy / rho) * (!isacc) + 0.002 * isacc;
	//uz = (uz / rho) * (!isacc) + 0.000 * isacc;

	//float u2 = 1.5 * (ux * ux + uy * uy + uz * uz); //U2(ux,uy,uz);//
	uvalues[x*5 + 4] = 1.5 * (uvalues[x*5 +1] * uvalues[x*5 +1] + uvalues[x*5 +2] * uvalues[x*5 +2] + uvalues[x*5 +3] * uvalues[x*5 +3]);
	//uvalues[x*5 + 4] = u2_func(uvalues[x*5 +1], uvalues[x*5 +2], uvalues[x*5 +3]);


	//address = c_nbArray[ 0] + idx;
	dGrid[c_nbArray[ 0] + idx] =  		  	(c * isobs) + (ONEMINUSOMEGA* c + DFL1_OMEGA * (uvalues[x*5]) * (1.0 - (uvalues[x*5 + 4])))* (!isobs);
	//address = c_nbArray[ 1] + idx + offset;
	dGrid[c_nbArray[ 1] + idx + offset] =   (s * isobs)  + (ONEMINUSOMEGA* n + DFL2_OMEGA * (uvalues[x*5]) * (1.0 + (uvalues[x*5 + 2]) * (4.5 * (uvalues[x*5 + 2]) + 3.0) - (uvalues[x*5 + 4])))* (!isobs);
	//address = c_nbArray[ 2] + idx + 2*offset;
	dGrid[c_nbArray[ 2] + idx + 2*offset] = (n * isobs)  + (ONEMINUSOMEGA* s + DFL2_OMEGA * (uvalues[x*5]) * (1.0 + (uvalues[x*5 + 2]) * (4.5 * (uvalues[x*5 + 2]) - 3.0) - (uvalues[x*5 + 4])))* (!isobs);
	//address = c_nbArray[ 3] + idx + 3*offset;
	dGrid[c_nbArray[ 3] + idx + 3*offset] = (w * isobs)	+ (ONEMINUSOMEGA* e + DFL2_OMEGA * (uvalues[x*5]) * (1.0 + (uvalues[x*5 + 1]) * (4.5 * (uvalues[x*5 + 1]) + 3.0) - (uvalues[x*5 + 4])))* (!isobs);
	//address = c_nbArray[ 4] + idx + 4*offset;
	dGrid[c_nbArray[ 4] + idx + 4*offset] = (e * isobs)	+ (ONEMINUSOMEGA* w + DFL2_OMEGA * (uvalues[x*5]) * (1.0 + (uvalues[x*5 + 1]) * (4.5 * (uvalues[x*5 + 1]) - 3.0) - (uvalues[x*5 + 4])))* (!isobs);
	//address = c_nbArray[ 5] + idx + 5*offset;
	dGrid[c_nbArray[ 5] + idx + 5*offset] = (b * isobs)	+ (ONEMINUSOMEGA* t + DFL2_OMEGA * (uvalues[x*5]) * (1.0 + (uvalues[x*5 + 3]) * (4.5 * (uvalues[x*5 + 3]) + 3.0) - (uvalues[x*5 + 4])))* (!isobs);
	//address = c_nbArray[ 6] + idx + 6*offset;
	dGrid[c_nbArray[ 6] + idx + 6*offset] = (t * isobs)	+ (ONEMINUSOMEGA* b + DFL2_OMEGA * (uvalues[x*5]) * (1.0 + (uvalues[x*5 + 3]) * (4.5 * (uvalues[x*5 + 3]) - 3.0) - (uvalues[x*5 + 4])))* (!isobs);
	//address = c_nbArray[ 7] + idx + 7*offset;
	dGrid[c_nbArray[ 7] + idx + 7*offset] = (sw * isobs) + (ONEMINUSOMEGA* ne + DFL3_OMEGA * (uvalues[x*5]) * (1.0 + (+(uvalues[x*5 + 1]) + (uvalues[x*5 + 2])) * (4.5 * (+(uvalues[x*5 + 1]) + (uvalues[x*5 + 2])) + 3.0) - (uvalues[x*5 + 4])))* (!isobs);
	//address = c_nbArray[ 8] + idx + 8*offset;
	dGrid[c_nbArray[ 8] + idx + 8*offset] = (se * isobs) + (ONEMINUSOMEGA* nw + DFL3_OMEGA * (uvalues[x*5]) * (1.0 + (-(uvalues[x*5 + 1]) + (uvalues[x*5 + 2])) * (4.5 * (-(uvalues[x*5 + 1]) + (uvalues[x*5 + 2])) + 3.0) - (uvalues[x*5 + 4])))* (!isobs);
	//address = c_nbArray[ 9] + idx + 9*offset;
	dGrid[c_nbArray[ 9] + idx + 9*offset] = (nw * isobs) + (ONEMINUSOMEGA* se + DFL3_OMEGA * (uvalues[x*5]) * (1.0 + (+(uvalues[x*5 + 1]) - (uvalues[x*5 + 2])) * (4.5 * (+(uvalues[x*5 + 1]) - (uvalues[x*5 + 2])) + 3.0) - (uvalues[x*5 + 4])))* (!isobs);
	//address = c_nbArray[10] + idx + 10*offset;
	dGrid[c_nbArray[10] + idx + 10*offset] = (ne * isobs)+ (ONEMINUSOMEGA* sw + DFL3_OMEGA * (uvalues[x*5]) * (1.0 + (-(uvalues[x*5 + 1]) - (uvalues[x*5 + 2])) * (4.5 * (-(uvalues[x*5 + 1]) - (uvalues[x*5 + 2])) + 3.0) - (uvalues[x*5 + 4])))* (!isobs);
	//address = c_nbArray[11] + idx + 11*offset;
	dGrid[c_nbArray[11] + idx + 11*offset] = (sb * isobs)+ (ONEMINUSOMEGA* nt + DFL3_OMEGA * (uvalues[x*5]) * (1.0 + (+(uvalues[x*5 + 2]) + (uvalues[x*5 + 3])) * (4.5 * (+(uvalues[x*5 + 2]) + (uvalues[x*5 + 3])) + 3.0) - (uvalues[x*5 + 4])))* (!isobs);
	//address = c_nbArray[12] + idx + 12*offset;
	dGrid[c_nbArray[12] + idx + 12*offset] = (st * isobs)+ (ONEMINUSOMEGA* nb + DFL3_OMEGA * (uvalues[x*5]) * (1.0 + (+(uvalues[x*5 + 2]) - (uvalues[x*5 + 3])) * (4.5 * (+(uvalues[x*5 + 2]) - (uvalues[x*5 + 3])) + 3.0) - (uvalues[x*5 + 4])))* (!isobs);
	//address = c_nbArray[13] + idx + 13*offset;
	dGrid[c_nbArray[13] + idx + 13*offset] = (nb * isobs)+ (ONEMINUSOMEGA* st + DFL3_OMEGA * (uvalues[x*5]) * (1.0 + (-(uvalues[x*5 + 2]) + (uvalues[x*5 + 3])) * (4.5 * (-(uvalues[x*5 + 2]) + (uvalues[x*5 + 3])) + 3.0) - (uvalues[x*5 + 4])))* (!isobs);
	//address = c_nbArray[14] + idx + 14*offset;
	dGrid[c_nbArray[14] + idx + 14*offset] = (nt * isobs)+ (ONEMINUSOMEGA* sb + DFL3_OMEGA * (uvalues[x*5]) * (1.0 + (-(uvalues[x*5 + 2]) - (uvalues[x*5 + 3])) * (4.5 * (-(uvalues[x*5 + 2]) - (uvalues[x*5 + 3])) + 3.0) - (uvalues[x*5 + 4])))* (!isobs);
	//address = c_nbArray[15] + idx + 15*offset;
	dGrid[c_nbArray[15] + idx + 15*offset] = (wb * isobs)+ (ONEMINUSOMEGA* et + DFL3_OMEGA * (uvalues[x*5]) * (1.0 + (+(uvalues[x*5 + 1]) + (uvalues[x*5 + 3])) * (4.5 * (+(uvalues[x*5 + 1]) + (uvalues[x*5 + 3])) + 3.0) - (uvalues[x*5 + 4])))* (!isobs);
	//address = c_nbArray[16] + idx + 16*offset;
	dGrid[c_nbArray[16] + idx + 16*offset] = (wt * isobs)+ (ONEMINUSOMEGA* eb + DFL3_OMEGA * (uvalues[x*5]) * (1.0 + (+(uvalues[x*5 + 1]) - (uvalues[x*5 + 3])) * (4.5 * (+(uvalues[x*5 + 1]) - (uvalues[x*5 + 3])) + 3.0) - (uvalues[x*5 + 4])))* (!isobs);
	//address = c_nbArray[17] + idx + 17*offset;
	dGrid[c_nbArray[17] + idx + 17*offset] = (eb * isobs)+ (ONEMINUSOMEGA* wt + DFL3_OMEGA * (uvalues[x*5]) * (1.0 + (-(uvalues[x*5 + 1]) + (uvalues[x*5 + 3])) * (4.5 * (-(uvalues[x*5 + 1]) + (uvalues[x*5 + 3])) + 3.0) - (uvalues[x*5 + 4])))* (!isobs);
	//address = c_nbArray[18] + idx + 18*offset;
	dGrid[c_nbArray[18] + idx + 18*offset] = (et * isobs)+ (ONEMINUSOMEGA* wb + DFL3_OMEGA * (uvalues[x*5]) * (1.0 + (-(uvalues[x*5 + 1]) - (uvalues[x*5 + 3])) * (4.5 * (-(uvalues[x*5 + 1]) - (uvalues[x*5 + 3])) + 3.0) - (uvalues[x*5 + 4])))* (!isobs);

}
__global__ void lbm_kernel_SoA_2(float *sGrid, float *dGrid)
{
	//extern __shared__ unsigned char array[];
	//float *shPropPointer = (float*)array;
	//int x = threadIdx.x;
	//int y = blockIdx.x;
	//int z = blockIdx.y;

	int idx;
	//int ElementsPerBlock = blockDim.x;
	//float c, n, s, e, w, t, b, ne, nw, se, sw, nt, nb, st, sb, et,	eb, wt, wb;

	/*the grid is organized as follows:
	 * c c c c c c . . . | n n n n n n . . . | s s s s s s . . . | e e e e e e . . . | . . .
	 */

	idx = (blockIdx.y * gridDim.x + blockIdx.x)*blockDim.x + threadIdx.x;
	//calculate the index
	//idx = x + y*100 + z*100*100;
	//idx = calc_idx(x, y, z, 0);

	//limit the index in the right value range
	idx = max(idx, 0);
	idx = min(idx, 1300000-1);

	//each direction has offset = 130*100*100 values, we need to jump one offset to get new value
	int offset = 1300000;
	//save index to address for operating
	int address = idx;
	float c = sGrid[address];
	address = address + offset;
	float n = sGrid[address];
	address = address + offset;
	float s = sGrid[address];
	address = address + offset;
	float e = sGrid[address];
	address = address + offset;
	float w = sGrid[address];
	address = address + offset;
	float t = sGrid[address];
	address = address + offset;
	float b = sGrid[address];
	address = address + offset;
	float ne = sGrid[address];
	address = address + offset;
	float nw = sGrid[address];
	address = address + offset;
	float se = sGrid[address];
	address = address + offset;
	float sw = sGrid[address];
	address = address + offset;
	float nt = sGrid[address];
	address = address + offset;
	float nb = sGrid[address];
	address = address + offset;
	float st = sGrid[address];
	address = address + offset;
	float sb = sGrid[address];
	address = address + offset;
	float et = sGrid[address];
	address = address + offset;
	float eb = sGrid[address];
	address = address + offset;
	float wt = sGrid[address];
	address = address + offset;
	float wb = sGrid[address];
	address = address + offset;
	//get the value final cell (20) to know the kind of cell: fluid or accel or obstacle
	int flag = *(unsigned int*) (void*) (&sGrid[address]);
	int obs = test_flag(flag, OBSTACLE);
	int acc = test_flag(flag, ACCEL);

	int isobs = (obs != 0);
	int isacc = (acc != 0);

	float rho = +c + n + s + e + w + t + b + ne + nw + se + sw + nt + nb + st
			+ sb + et + eb + wt + wb;

	float ux = +e - w + ne - nw + se - sw + et + eb - wt - wb;
	float uy = +n - s + ne + nw - se - sw + nt + nb - st - sb;
	float uz = +t - b + nt - nb + st - sb + et - eb + wt - wb;

	ux = (ux / rho) * (!isacc) + 0.005 * isacc;
	uy = (uy / rho) * (!isacc) + 0.002 * isacc;
	uz = (uz / rho) * (!isacc) + 0.000 * isacc;

	float u2 = 1.5 * (ux * ux + uy * uy + uz * uz);

	//int prop_addr;
	//addr = x;
	//shPropPointer[addr] = (c * isobs) + ((1.0 - OMEGA)* c + DFL1 * OMEGA * rho * (1.0 - u2))* (!isobs);
	int n_pos;
	//float t_val;
	n_pos = c_nbArray[0] + idx*20; //convert to pos which equivalent to pos in the old array
	//prop_addr = (n_pos>=0)*((n_pos%20)*1300000 + n_pos/20) + (n_pos<0)*n_pos ;
	//if(idx==0) printf("\n0.%d", addr);
	//t_val = (c * isobs)	+ ((1.0 - OMEGA)* c + DFL1 * OMEGA * rho * (1.0 - u2))* (!isobs);
	dGrid[n_pos] = (c * isobs)	+ ((1.0 - OMEGA)* c + DFL1 * OMEGA * rho * (1.0 - u2))* (!isobs);

	n_pos = c_nbArray[1] + idx*20;
	//prop_addr = (n_pos>=0)*((n_pos%20)*1300000 + n_pos/20) + (n_pos<0)*n_pos ;
	//if(idx==0) printf("\n1.%d", prop_addr);
	//t_val =	(s * isobs) + ((1.0 - OMEGA)* n + DFL2 * OMEGA * rho * (1.0 + uy * (4.5 * uy + 3.0) - u2))* (!isobs);
	dGrid[n_pos] = (s * isobs) + ((1.0 - OMEGA)* n + DFL2 * OMEGA * rho * (1.0 + uy * (4.5 * uy + 3.0) - u2))* (!isobs);
	//prop_addr +=ElementsPerBlock;
	//shPropPointer[prop_addr] = (s * isobs) + ((1.0 - OMEGA)* n + DFL2 * OMEGA * rho * (1.0 + uy * (4.5 * uy + 3.0) - u2))* (!isobs);

	n_pos = c_nbArray[2] + idx*20;
	//prop_addr = (n_pos>=0)*((n_pos%20)*1300000 + n_pos/20) + (n_pos<0)*n_pos ;
	//if(idx==0) printf("\n2.%d", prop_addr);
	//t_val =	(n * isobs) + ((1.0 - OMEGA)* s + DFL2 * OMEGA * rho * (1.0 + uy * (4.5 * uy - 3.0) - u2))* (!isobs);
	dGrid[n_pos] = (n * isobs) + ((1.0 - OMEGA)* s + DFL2 * OMEGA * rho * (1.0 + uy * (4.5 * uy - 3.0) - u2))* (!isobs);
	//prop_addr +=ElementsPerBlock;
	//shPropPointer[prop_addr] = (n * isobs) + ((1.0 - OMEGA)* s + DFL2 * OMEGA * rho * (1.0 + uy * (4.5 * uy - 3.0) - u2))* (!isobs);

	n_pos = c_nbArray[3] + idx*20;
	//prop_addr = (n_pos>=0)*((n_pos%20)*1300000 + n_pos/20) + (n_pos<0)*n_pos ;
	//if(idx==0) printf("\n3.%d", prop_addr);
	//t_val =	(w * isobs)	+ ((1.0 - OMEGA)* e + DFL2 * OMEGA * rho * (1.0 + ux * (4.5 * ux + 3.0) - u2))* (!isobs);
	dGrid[n_pos] = (w * isobs)	+ ((1.0 - OMEGA)* e + DFL2 * OMEGA * rho * (1.0 + ux * (4.5 * ux + 3.0) - u2))* (!isobs);
	//prop_addr +=ElementsPerBlock;
	//shPropPointer[prop_addr] = (w * isobs) + ((1.0 - OMEGA)* e + DFL2 * OMEGA * rho * (1.0 + ux * (4.5 * ux + 3.0) - u2))* (!isobs);

	n_pos = c_nbArray[4] + idx*20;
	//prop_addr = (n_pos>=0)*((n_pos%20)*1300000 + n_pos/20) + (n_pos<0)*n_pos ;
	//if(idx==0) printf("\n4.%d", prop_addr);
	//t_val =	(e * isobs)	+ ((1.0 - OMEGA)* w + DFL2 * OMEGA * rho * (1.0 + ux * (4.5 * ux - 3.0) - u2))* (!isobs);
	dGrid[n_pos] = (e * isobs)	+ ((1.0 - OMEGA)* w + DFL2 * OMEGA * rho * (1.0 + ux * (4.5 * ux - 3.0) - u2))* (!isobs);
	//prop_addr +=ElementsPerBlock;
	//shPropPointer[prop_addr] = (e * isobs) + ((1.0 - OMEGA)* w + DFL2 * OMEGA * rho * (1.0 + ux * (4.5 * ux - 3.0) - u2))* (!isobs);

	n_pos = c_nbArray[5] + idx*20;
	//prop_addr = (n_pos>=0)*((n_pos%20)*1300000 + n_pos/20) + (n_pos<0)*n_pos ;
	//if(idx==0) printf("\n5.%d", prop_addr);
	//t_val = (b * isobs)	+ ((1.0 - OMEGA)* t + DFL2 * OMEGA * rho * (1.0 + uz * (4.5 * uz + 3.0) - u2))* (!isobs);
	dGrid[n_pos] = (b * isobs)	+ ((1.0 - OMEGA)* t + DFL2 * OMEGA * rho * (1.0 + uz * (4.5 * uz + 3.0) - u2))* (!isobs);
	//prop_addr +=ElementsPerBlock;
	//shPropPointer[prop_addr] = (b * isobs)	+ ((1.0 - OMEGA)* t + DFL2 * OMEGA * rho * (1.0 + uz * (4.5 * uz + 3.0) - u2))* (!isobs);

	n_pos = c_nbArray[6] + idx*20;
	//prop_addr = (n_pos>=0)*((n_pos%20)*1300000 + n_pos/20) + (n_pos<0)*n_pos ;
	//if(idx==0) printf("\n6.%d", prop_addr);
	//t_val =	(t * isobs)	+ ((1.0 - OMEGA)* b + DFL2 * OMEGA * rho * (1.0 + uz * (4.5 * uz - 3.0) - u2))* (!isobs);
	dGrid[n_pos] = (t * isobs)	+ ((1.0 - OMEGA)* b + DFL2 * OMEGA * rho * (1.0 + uz * (4.5 * uz - 3.0) - u2))* (!isobs);
	//prop_addr +=ElementsPerBlock;
	//shPropPointer[prop_addr] = (t * isobs)	+ ((1.0 - OMEGA)* b + DFL2 * OMEGA * rho * (1.0 + uz * (4.5 * uz - 3.0) - u2))* (!isobs);

	n_pos = c_nbArray[7] + idx*20;
	//prop_addr = (n_pos>=0)*((n_pos%20)*1300000 + n_pos/20) + (n_pos<0)*n_pos ;
	//if(idx==0) printf("\n7.%d", prop_addr);
	//t_val =	(sw * isobs)+ ((1.0 - OMEGA)* ne + DFL3 * OMEGA * rho * (1.0 + (+ux + uy) * (4.5 * (+ux + uy) + 3.0) - u2))* (!isobs);
	dGrid[n_pos] = (sw * isobs)+ ((1.0 - OMEGA)* ne + DFL3 * OMEGA * rho * (1.0 + (+ux + uy) * (4.5 * (+ux + uy) + 3.0) - u2))* (!isobs);
	//prop_addr +=ElementsPerBlock;
	//shPropPointer[prop_addr] = (sw * isobs)+ ((1.0 - OMEGA)* ne + DFL3 * OMEGA * rho * (1.0 + (+ux + uy) * (4.5 * (+ux + uy) + 3.0) - u2))* (!isobs);

	n_pos = c_nbArray[8] + idx*20;
	//prop_addr = (n_pos>=0)*((n_pos%20)*1300000 + n_pos/20) + (n_pos<0)*n_pos ;
	//if(idx==0) printf("\n8.%d", prop_addr);
	//t_val = (se * isobs)+ ((1.0 - OMEGA)* nw + DFL3 * OMEGA * rho * (1.0 + (-ux + uy) * (4.5 * (-ux + uy) + 3.0) - u2))* (!isobs);
	dGrid[n_pos] = (se * isobs)+ ((1.0 - OMEGA)* nw + DFL3 * OMEGA * rho * (1.0 + (-ux + uy) * (4.5 * (-ux + uy) + 3.0) - u2))* (!isobs);
	//prop_addr +=ElementsPerBlock;
	//shPropPointer[prop_addr] = (se * isobs)+ ((1.0 - OMEGA)* nw + DFL3 * OMEGA * rho * (1.0 + (-ux + uy) * (4.5 * (-ux + uy) + 3.0) - u2))* (!isobs);

	n_pos = c_nbArray[9] + idx*20;
	//prop_addr = (n_pos>=0)*((n_pos%20)*1300000 + n_pos/20) + (n_pos<0)*n_pos ;
	//if(idx==0) printf("\n9.%d", prop_addr);
	//t_val =	(nw * isobs)+ ((1.0 - OMEGA)* se + DFL3 * OMEGA * rho * (1.0 + (+ux - uy) * (4.5 * (+ux - uy) + 3.0) - u2))* (!isobs);
	dGrid[n_pos] = (nw * isobs)+ ((1.0 - OMEGA)* se + DFL3 * OMEGA * rho * (1.0 + (+ux - uy) * (4.5 * (+ux - uy) + 3.0) - u2))* (!isobs);
	//prop_addr +=ElementsPerBlock;
	//shPropPointer[prop_addr] = (nw * isobs)+ ((1.0 - OMEGA)* se + DFL3 * OMEGA * rho * (1.0 + (+ux - uy) * (4.5 * (+ux - uy) + 3.0) - u2))* (!isobs);

	n_pos = c_nbArray[10] + idx*20;
	//prop_addr = (n_pos>=0)*((n_pos%20)*1300000 + n_pos/20) + (n_pos<0)*n_pos ;
	//if(idx==0) printf("\n10.%d", prop_addr);
	//t_val = (ne * isobs)+ ((1.0 - OMEGA)* sw + DFL3 * OMEGA * rho * (1.0 + (-ux - uy) * (4.5 * (-ux - uy) + 3.0) - u2))* (!isobs);
	dGrid[n_pos] = (ne * isobs)+ ((1.0 - OMEGA)* sw + DFL3 * OMEGA * rho * (1.0 + (-ux - uy) * (4.5 * (-ux - uy) + 3.0) - u2))* (!isobs);
	//prop_addr +=ElementsPerBlock;
	//shPropPointer[prop_addr] = (ne * isobs)+ ((1.0 - OMEGA)* sw + DFL3 * OMEGA * rho * (1.0 + (-ux - uy) * (4.5 * (-ux - uy) + 3.0) - u2))* (!isobs);

	n_pos = c_nbArray[11] + idx*20;
	//prop_addr = (n_pos>=0)*((n_pos%20)*1300000 + n_pos/20) + (n_pos<0)*n_pos ;
	//if(idx==0) printf("\n11.%d", prop_addr);
	//t_val =	(sb * isobs)+ ((1.0 - OMEGA)* nt + DFL3 * OMEGA * rho * (1.0 + (+uy + uz) * (4.5 * (+uy + uz) + 3.0) - u2))* (!isobs);
	dGrid[n_pos] = (sb * isobs)+ ((1.0 - OMEGA)* nt + DFL3 * OMEGA * rho * (1.0 + (+uy + uz) * (4.5 * (+uy + uz) + 3.0) - u2))* (!isobs);
	//prop_addr +=ElementsPerBlock;
	//shPropPointer[prop_addr] = (sb * isobs)+ ((1.0 - OMEGA)* nt + DFL3 * OMEGA * rho * (1.0 + (+uy + uz) * (4.5 * (+uy + uz) + 3.0) - u2))* (!isobs);

	n_pos = c_nbArray[12] + idx*20;
	//prop_addr = (n_pos>=0)*((n_pos%20)*1300000 + n_pos/20) + (n_pos<0)*n_pos ;
	//if(idx==0) printf("\n12.%d", prop_addr);
	//t_val =	(st * isobs)+ ((1.0 - OMEGA)* nb + DFL3 * OMEGA * rho * (1.0 + (+uy - uz) * (4.5 * (+uy - uz) + 3.0) - u2))* (!isobs);
	dGrid[n_pos] = (st * isobs)+ ((1.0 - OMEGA)* nb + DFL3 * OMEGA * rho * (1.0 + (+uy - uz) * (4.5 * (+uy - uz) + 3.0) - u2))* (!isobs);
	//prop_addr +=ElementsPerBlock;
	//shPropPointer[prop_addr] = (st * isobs)+ ((1.0 - OMEGA)* nb + DFL3 * OMEGA * rho * (1.0 + (+uy - uz) * (4.5 * (+uy - uz) + 3.0) - u2))* (!isobs);

	n_pos = c_nbArray[13] + idx*20;
	//prop_addr = (n_pos>=0)*((n_pos%20)*1300000 + n_pos/20) + (n_pos<0)*n_pos ;
	//if(idx==0) printf("\n13.%d", prop_addr);
	//t_val =	(nb * isobs)+ ((1.0 - OMEGA)* st + DFL3 * OMEGA * rho * (1.0 + (-uy + uz) * (4.5 * (-uy + uz) + 3.0) - u2))* (!isobs);
	dGrid[n_pos] = (nb * isobs)+ ((1.0 - OMEGA)* st + DFL3 * OMEGA * rho * (1.0 + (-uy + uz) * (4.5 * (-uy + uz) + 3.0) - u2))* (!isobs);
	//prop_addr +=ElementsPerBlock;
	//shPropPointer[prop_addr] = (nb * isobs)+ ((1.0 - OMEGA)* st + DFL3 * OMEGA * rho * (1.0 + (-uy + uz) * (4.5 * (-uy + uz) + 3.0) - u2))* (!isobs);

	n_pos = c_nbArray[14] + idx*20;
	//prop_addr = (n_pos>=0)*((n_pos%20)*1300000 + n_pos/20) + (n_pos<0)*n_pos ;
	//if(idx==0) printf("\n14.%d", prop_addr);
	//t_val =	(nt * isobs)+ ((1.0 - OMEGA)* sb + DFL3 * OMEGA * rho * (1.0 + (-uy - uz) * (4.5 * (-uy - uz) + 3.0) - u2))* (!isobs);
	dGrid[n_pos] = (nt * isobs)+ ((1.0 - OMEGA)* sb + DFL3 * OMEGA * rho * (1.0 + (-uy - uz) * (4.5 * (-uy - uz) + 3.0) - u2))* (!isobs);
	//prop_addr +=ElementsPerBlock;
	//shPropPointer[prop_addr] = (nt * isobs)+ ((1.0 - OMEGA)* sb + DFL3 * OMEGA * rho * (1.0 + (-uy - uz) * (4.5 * (-uy - uz) + 3.0) - u2))* (!isobs);

	n_pos = c_nbArray[15] + idx*20;
	//prop_addr = (n_pos>=0)*((n_pos%20)*1300000 + n_pos/20) + (n_pos<0)*n_pos ;
	//if(idx==0) printf("\n15.%d", prop_addr);
	//t_val = (wb * isobs) + ((1.0 - OMEGA)* et + DFL3 * OMEGA * rho * (1.0 + (+ux + uz) * (4.5 * (+ux + uz) + 3.0) - u2))* (!isobs);
	dGrid[n_pos] = (wb * isobs) + ((1.0 - OMEGA)* et + DFL3 * OMEGA * rho * (1.0 + (+ux + uz) * (4.5 * (+ux + uz) + 3.0) - u2))* (!isobs);
	//prop_addr +=ElementsPerBlock;
	//shPropPointer[prop_addr] = (wb * isobs)+ ((1.0 - OMEGA)* et + DFL3 * OMEGA * rho * (1.0 + (+ux + uz) * (4.5 * (+ux + uz) + 3.0) - u2))* (!isobs);

	n_pos = c_nbArray[16] + idx*20;
	//prop_addr = (n_pos>=0)*((n_pos%20)*1300000 + n_pos/20) + (n_pos<0)*n_pos ;
	//if(idx==0) printf("\n16.%d", prop_addr);
	//t_val = (wt * isobs) + ((1.0 - OMEGA)* eb + DFL3 * OMEGA * rho * (1.0 + (+ux - uz) * (4.5 * (+ux - uz) + 3.0) - u2))* (!isobs);
	dGrid[n_pos] = (wt * isobs) + ((1.0 - OMEGA)* eb + DFL3 * OMEGA * rho * (1.0 + (+ux - uz) * (4.5 * (+ux - uz) + 3.0) - u2))* (!isobs);
	//prop_addr +=ElementsPerBlock;
	//shPropPointer[prop_addr] = (wt * isobs) + ((1.0 - OMEGA)* eb + DFL3 * OMEGA * rho * (1.0 + (+ux - uz) * (4.5 * (+ux - uz) + 3.0) - u2))* (!isobs);

	n_pos = c_nbArray[17] + idx*20;
	//prop_addr = (n_pos>=0)*((n_pos%20)*1300000 + n_pos/20) + (n_pos<0)*n_pos ;
	//if(idx==0) printf("\n17.%d", prop_addr);
	//t_val =	(eb * isobs) + ((1.0 - OMEGA)* wt + DFL3 * OMEGA * rho * (1.0 + (-ux + uz) * (4.5 * (-ux + uz) + 3.0) - u2))* (!isobs);
	dGrid[n_pos] = (eb * isobs) + ((1.0 - OMEGA)* wt + DFL3 * OMEGA * rho * (1.0 + (-ux + uz) * (4.5 * (-ux + uz) + 3.0) - u2))* (!isobs);
	//prop_addr +=ElementsPerBlock;
	//shPropPointer[prop_addr] = (eb * isobs) + ((1.0 - OMEGA)* wt + DFL3 * OMEGA * rho * (1.0 + (-ux + uz) * (4.5 * (-ux + uz) + 3.0) - u2))* (!isobs);

	n_pos = c_nbArray[18] + idx*20;
	//prop_addr = (n_pos>=0)*((n_pos%20)*1300000 + n_pos/20) + (n_pos<0)*n_pos ;
	//if(idx==0) printf("\n18.%d", prop_addr);
	//t_val = (et * isobs)+((1.0 - OMEGA)* wb + DFL3 * OMEGA * rho * (1.0 + (-ux - uz) * (4.5 * (-ux - uz) + 3.0) - u2))* (!isobs);
	dGrid[n_pos] =(et * isobs)+((1.0 - OMEGA)* wb + DFL3 * OMEGA * rho * (1.0 + (-ux - uz) * (4.5 * (-ux - uz) + 3.0) - u2))* (!isobs);
	//prop_addr +=ElementsPerBlock;
	//shPropPointer[prop_addr] = (et * isobs)+((1.0 - OMEGA)* wb + DFL3 * OMEGA * rho * (1.0 + (-ux - uz) * (4.5 * (-ux - uz) + 3.0) - u2))* (!isobs);
	//__syncthreads();

}
//su dung shared memory nhu noi luu tru tam
//lay du lieu tu sGrid -> tinh toan-> luu xuong shared memory -> luu tro lai dGrid
//size of shared memory = 20 * 8 * number of threads (384) ~ 64KB
__global__ void lbm_kernel_SoA_3(float *sGrid, float *dGrid)
{
	//extern __shared__ unsigned char array[];
	//float *shPropPointer = (float*)array;
	//int x = threadIdx.x;
	//int y = blockIdx.x;
	//int z = blockIdx.y;

	int idx;
	//int ElementsPerBlock = blockDim.x;
	//float c, n, s, e, w, t, b, ne, nw, se, sw, nt, nb, st, sb, et,	eb, wt, wb;

	/*the grid is organized as follows:
	 * c c c c c c . . . | n n n n n n . . . | s s s s s s . . . | e e e e e e . . . | . . .
	 */

	//calculate the index
	idx = (blockIdx.y * gridDim.x + blockIdx.x)*blockDim.x + threadIdx.x;
	//idx = calc_idx(x, y, z, 0);

	//limit the index in the right value range
	idx = max(idx, 0);
	idx = min(idx, 1300000-1);

	//each direction has offset = 130*100*100 values, we need to jump one offset to get new value
	/*int offset = 1300000;
	//save index to address for operating
	int address = idx;
	float c = sGrid[address];
	address = address + offset;
	float n = sGrid[address];
	address = address + offset;
	float s = sGrid[address];
	address = address + offset;
	float e = sGrid[address];
	address = address + offset;
	float w = sGrid[address];
	address = address + offset;
	float t = sGrid[address];
	address = address + offset;
	float b = sGrid[address];
	address = address + offset;
	float ne = sGrid[address];
	address = address + offset;
	float nw = sGrid[address];
	address = address + offset;
	float se = sGrid[address];
	address = address + offset;
	float sw = sGrid[address];
	address = address + offset;
	float nt = sGrid[address];
	address = address + offset;
	float nb = sGrid[address];
	address = address + offset;
	float st = sGrid[address];
	address = address + offset;
	float sb = sGrid[address];
	address = address + offset;
	float et = sGrid[address];
	address = address + offset;
	float eb = sGrid[address];
	address = address + offset;
	float wt = sGrid[address];
	address = address + offset;
	float wb = sGrid[address];
	address = address + offset;*/


	float c = sGrid[idx];

	float n = sGrid[idx + 1300000];

	float s = sGrid[idx + 2*1300000];

	float e = sGrid[idx + 3*1300000];

	float w = sGrid[idx + 4*1300000];

	float t = sGrid[idx + 5*1300000];

	float b = sGrid[idx + 6*1300000];

	float ne = sGrid[idx +7*1300000];

	float nw = sGrid[idx + 8*1300000];

	float se = sGrid[idx + 9*1300000];

	float sw = sGrid[idx + 10*1300000];

	float nt = sGrid[idx + 11*1300000];

	float nb = sGrid[idx + 12*1300000];

	float st = sGrid[idx + 13*1300000];

	float sb = sGrid[idx + 14*1300000];

	float et = sGrid[idx + 15*1300000];

	float eb = sGrid[idx + 16*1300000];

	float wt = sGrid[idx + 17*1300000];

	float wb = sGrid[idx + 18*1300000];

	//get the value final cell (20) to know the kind of cell: fluid or accel or obstacle
	int flag = *(unsigned int*) (void*) (&sGrid[idx + 19*1300000]);
	int obs = test_flag(flag, OBSTACLE);
	int acc = test_flag(flag, ACCEL);

	int isobs = (obs != 0);
	int isacc = (acc != 0);

	float rho = +c + n + s + e + w + t + b + ne + nw + se + sw + nt + nb + st
			+ sb + et + eb + wt + wb;

	float ux = +e - w + ne - nw + se - sw + et + eb - wt - wb;
	float uy = +n - s + ne + nw - se - sw + nt + nb - st - sb;
	float uz = +t - b + nt - nb + st - sb + et - eb + wt - wb;

	ux = (ux / rho) * (!isacc) + 0.005 * isacc;
	uy = (uy / rho) * (!isacc) + 0.002 * isacc;
	uz = (uz / rho) * (!isacc) + 0.000 * isacc;

	float u2 = 1.5 * (ux * ux + uy * uy + uz * uz);

	dGrid[c_nbArray[0] + idx*20] = (c * isobs)	+ ((1.0 - OMEGA)* c + DFL1 * OMEGA * rho * (1.0 - u2))* (!isobs);

	dGrid[c_nbArray[1] + idx*20] = (s * isobs)  + ((1.0 - OMEGA)* n + DFL2 * OMEGA * rho * (1.0 + uy * (4.5 * uy + 3.0) - u2))* (!isobs);

	dGrid[c_nbArray[2] + idx*20] = (n * isobs)  + ((1.0 - OMEGA)* s + DFL2 * OMEGA * rho * (1.0 + uy * (4.5 * uy - 3.0) - u2))* (!isobs);

	dGrid[c_nbArray[3] + idx*20] = (w * isobs)	+ ((1.0 - OMEGA)* e + DFL2 * OMEGA * rho * (1.0 + ux * (4.5 * ux + 3.0) - u2))* (!isobs);

	dGrid[c_nbArray[4] + idx*20] = (e * isobs)	+ ((1.0 - OMEGA)* w + DFL2 * OMEGA * rho * (1.0 + ux * (4.5 * ux - 3.0) - u2))* (!isobs);

	dGrid[c_nbArray[5] + idx*20] = (b * isobs)	+ ((1.0 - OMEGA)* t + DFL2 * OMEGA * rho * (1.0 + uz * (4.5 * uz + 3.0) - u2))* (!isobs);

	dGrid[c_nbArray[6] + idx*20] = (t * isobs)	+ ((1.0 - OMEGA)* b + DFL2 * OMEGA * rho * (1.0 + uz * (4.5 * uz - 3.0) - u2))* (!isobs);

	dGrid[c_nbArray[7] + idx*20] = (sw * isobs) + ((1.0 - OMEGA)* ne + DFL3 * OMEGA * rho * (1.0 + (+ux + uy) * (4.5 * (+ux + uy) + 3.0) - u2))* (!isobs);

	dGrid[c_nbArray[8] + idx*20] = (se * isobs) + ((1.0 - OMEGA)* nw + DFL3 * OMEGA * rho * (1.0 + (-ux + uy) * (4.5 * (-ux + uy) + 3.0) - u2))* (!isobs);

	dGrid[c_nbArray[9] + idx*20] = (nw * isobs) + ((1.0 - OMEGA)* se + DFL3 * OMEGA * rho * (1.0 + (+ux - uy) * (4.5 * (+ux - uy) + 3.0) - u2))* (!isobs);

	dGrid[c_nbArray[10] + idx*20] = (ne * isobs)+ ((1.0 - OMEGA)* sw + DFL3 * OMEGA * rho * (1.0 + (-ux - uy) * (4.5 * (-ux - uy) + 3.0) - u2))* (!isobs);

	dGrid[c_nbArray[11] + idx*20] = (sb * isobs)+ ((1.0 - OMEGA)* nt + DFL3 * OMEGA * rho * (1.0 + (+uy + uz) * (4.5 * (+uy + uz) + 3.0) - u2))* (!isobs);

	dGrid[c_nbArray[12] + idx*20] = (st * isobs)+ ((1.0 - OMEGA)* nb + DFL3 * OMEGA * rho * (1.0 + (+uy - uz) * (4.5 * (+uy - uz) + 3.0) - u2))* (!isobs);

	dGrid[c_nbArray[13] + idx*20] = (nb * isobs)+ ((1.0 - OMEGA)* st + DFL3 * OMEGA * rho * (1.0 + (-uy + uz) * (4.5 * (-uy + uz) + 3.0) - u2))* (!isobs);

	dGrid[c_nbArray[14] + idx*20] = (nt * isobs)+ ((1.0 - OMEGA)* sb + DFL3 * OMEGA * rho * (1.0 + (-uy - uz) * (4.5 * (-uy - uz) + 3.0) - u2))* (!isobs);

	dGrid[c_nbArray[15] + idx*20] = (wb * isobs)+ ((1.0 - OMEGA)* et + DFL3 * OMEGA * rho * (1.0 + (+ux + uz) * (4.5 * (+ux + uz) + 3.0) - u2))* (!isobs);

	dGrid[c_nbArray[16] + idx*20] = (wt * isobs)+ ((1.0 - OMEGA)* eb + DFL3 * OMEGA * rho * (1.0 + (+ux - uz) * (4.5 * (+ux - uz) + 3.0) - u2))* (!isobs);

	dGrid[c_nbArray[17] + idx*20] = (eb * isobs)+ ((1.0 - OMEGA)* wt + DFL3 * OMEGA * rho * (1.0 + (-ux + uz) * (4.5 * (-ux + uz) + 3.0) - u2))* (!isobs);

	dGrid[c_nbArray[18] + idx*20] = (et * isobs)+ ((1.0 - OMEGA)* wb + DFL3 * OMEGA * rho * (1.0 + (-ux - uz) * (4.5 * (-ux - uz) + 3.0) - u2))* (!isobs);

}

__global__ void test_sm(float *test_arr)
{
	//extern __shared__ unsigned char array[];
	extern __shared__ float array[];
	//float *shPropPointer = (float*)array;
	//int x = threadIdx.x;
	//int y = blockIdx.x;
	//int z = blockIdx.y;

	int idx;
	//calculate the index
	idx = (blockIdx.y * gridDim.x + blockIdx.x)*blockDim.x + threadIdx.x;

	//limit the index in the right value range
	idx = max(idx, 0);
	idx = min(idx, 1300000-1);

	//shPropPointer[x] = idx*1.0;
	array[threadIdx.x] = idx*1.0;

	__syncthreads();

	test_arr[idx] = array[threadIdx.x];

}
void copyToConstantMem(int *h_lcArray, int *h_nbArray, float one_minus_o, float dfl1_o, float dfl2_o, float dfl3_o)
{
	cudaError_t err;
	err = cudaMemcpyToSymbol(c_nbArray, h_nbArray, 19 * sizeof(int));
	if(err!=0) printf("\nCopy data to constant memory:  %s\n",cudaGetErrorString(cudaGetLastError()));
	err = cudaMemcpyToSymbol(c_nb0, &h_nbArray[0], sizeof(int));
	err = cudaMemcpyToSymbol(c_nb1, &h_nbArray[1], sizeof(int));
	err = cudaMemcpyToSymbol(c_nb2, &h_nbArray[2], sizeof(int));
	err = cudaMemcpyToSymbol(c_nb3, &h_nbArray[3], sizeof(int));
	err = cudaMemcpyToSymbol(c_nb4, &h_nbArray[4], sizeof(int));
	err = cudaMemcpyToSymbol(c_nb5, &h_nbArray[5], sizeof(int));
	err = cudaMemcpyToSymbol(c_nb6, &h_nbArray[6], sizeof(int));
	err = cudaMemcpyToSymbol(c_nb7, &h_nbArray[7], sizeof(int));
	err = cudaMemcpyToSymbol(c_nb8, &h_nbArray[8], sizeof(int));
	err = cudaMemcpyToSymbol(c_nb9, &h_nbArray[9], sizeof(int));
	err = cudaMemcpyToSymbol(c_nb10, &h_nbArray[10], sizeof(int));
	err = cudaMemcpyToSymbol(c_nb11, &h_nbArray[11], sizeof(int));
	err = cudaMemcpyToSymbol(c_nb12, &h_nbArray[12], sizeof(int));
	err = cudaMemcpyToSymbol(c_nb13, &h_nbArray[13], sizeof(int));
	err = cudaMemcpyToSymbol(c_nb14, &h_nbArray[14], sizeof(int));
	err = cudaMemcpyToSymbol(c_nb15, &h_nbArray[15], sizeof(int));
	err = cudaMemcpyToSymbol(c_nb16, &h_nbArray[16], sizeof(int));
	err = cudaMemcpyToSymbol(c_nb17, &h_nbArray[17], sizeof(int));
	err = cudaMemcpyToSymbol(c_nb18, &h_nbArray[18], sizeof(int));
	if(err!=0) printf("\nCopy data to constant memory:  %s\n",cudaGetErrorString(cudaGetLastError()));
	err = cudaMemcpyToSymbol(c_lcArray, h_lcArray, 19 * sizeof(int));
	if(err!=0) printf("\nCopy data to constant memory:  %s\n",cudaGetErrorString(cudaGetLastError()));

	//err = cudaMemcpyToSymbol(cOneOmega, &one_minus_o, sizeof(float));
	//if(err!=0) printf("\nCopy data to constant memory:  %s\n",cudaGetErrorString(cudaGetLastError()));
	/*err = cudaMemcpyToSymbol(cDFL1Omega, &dfl1_o, sizeof(float));
	if(err!=0) printf("\nCopy data to constant memory:  %s\n",cudaGetErrorString(cudaGetLastError()));
	err = cudaMemcpyToSymbol(cDFL2Omega, &dfl2_o, sizeof(float));
	if(err!=0) printf("\nCopy data to constant memory:  %s\n",cudaGetErrorString(cudaGetLastError()));
	err = cudaMemcpyToSymbol(cDFL3Omega, &dfl3_o, sizeof(float));
	if(err!=0) printf("\nCopy data to constant memory:  %s\n",cudaGetErrorString(cudaGetLastError()));


	float3 cArray[19];
	cArray[ 0] = make_float3( 0, 0, 0);
	cArray[ 1] = make_float3( 0, 1, 0);
	cArray[ 2] = make_float3( 0,-1, 0);
	cArray[ 3] = make_float3( 1, 0, 0);
	cArray[ 4] = make_float3(-1, 0, 0);
	cArray[ 5] = make_float3( 0, 0, 1);
	cArray[ 6] = make_float3( 0, 0,-1);
	cArray[	7] = make_float3( 1, 1, 0);
	cArray[ 8] = make_float3(-1, 1, 0);
	cArray[ 9] = make_float3( 1,-1, 0);
	cArray[10] = make_float3(-1,-1, 0);
	cArray[11] = make_float3( 0, 1, 1);
	cArray[12] = make_float3( 0, 1,-1);
	cArray[13] = make_float3( 0,-1, 1);
	cArray[14] = make_float3( 0,-1,-1);
	cArray[15] = make_float3( 1, 0, 1);
	cArray[16] = make_float3( 1, 0,-1);
	cArray[17] = make_float3(-1, 0, 1);
	cArray[18] = make_float3(-1, 0,-1);
	err = cudaMemcpyToSymbol(c_cArray, cArray,  19*sizeof(float3));
	if(err!=0) printf("\nCopy data to constant memory:  %s\n",cudaGetErrorString(cudaGetLastError()));
	*/

}
__device__ void streaming(Distributions dst, int idx, float c, float n, float s, float e,
		float w, float t, float b, float ne, float nw, float se, float sw,
		float nt, float nb, float st, float sb, float et, float eb, float wt, float wb,
		float ux, float uy, float uz, float u2, float rho, int isobs, int isacc)
{
	(dst.f[C])[c_nbArray[ 0] + idx]  = (c * isobs)	+ (ONEMINUSOMEGA* c + DFL1_OMEGA * rho * (1.0 - u2))* (!isobs);
	//printf("\n(0. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 0] + idx, (dst.f[C])[c_nbArray[ 0] + idx]);
	(dst.f[N])[c_nbArray[ 1] + idx]  = (s * isobs)  + (ONEMINUSOMEGA* n + DFL2_OMEGA * rho * (1.0 + uy * (4.5 * uy + 3.0) - u2))* (!isobs);
	//printf("\n(1. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 1] + idx, (dst.f[N])[c_nbArray[ 1] + idx]);
	(dst.f[S])[c_nbArray[ 2] + idx]  = (n * isobs)  + (ONEMINUSOMEGA* s + DFL2_OMEGA * rho * (1.0 + uy * (4.5 * uy - 3.0) - u2))* (!isobs);
	//printf("\n(2. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 2] + idx, (dst.f[S])[c_nbArray[ 2] + idx]);
	(dst.f[E])[c_nbArray[ 3] + idx]  = (w * isobs)	+ (ONEMINUSOMEGA* e + DFL2_OMEGA * rho * (1.0 + ux * (4.5 * ux + 3.0) - u2))* (!isobs);
	//printf("\n(3. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 3] + idx, (dst.f[E])[c_nbArray[ 3] + idx]);
	(dst.f[W])[c_nbArray[ 4] + idx]  = (e * isobs)	+ (ONEMINUSOMEGA* w + DFL2_OMEGA * rho * (1.0 + ux * (4.5 * ux - 3.0) - u2))* (!isobs);
	//printf("\n(4. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 4] + idx, (dst.f[W])[c_nbArray[ 4] + idx]);
	(dst.f[T])[c_nbArray[ 5] + idx]  = (b * isobs)	+ (ONEMINUSOMEGA* t + DFL2_OMEGA * rho * (1.0 + uz * (4.5 * uz + 3.0) - u2))* (!isobs);
	//printf("\n(5. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 5] + idx, (dst.f[T])[c_nbArray[ 5] + idx]);
	(dst.f[B])[c_nbArray[ 6] + idx]  = (t * isobs)	+ (ONEMINUSOMEGA* b + DFL2_OMEGA * rho * (1.0 + uz * (4.5 * uz - 3.0) - u2))* (!isobs);
	//printf("\n(6. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 6] + idx, (dst.f[B])[c_nbArray[ 6] + idx]);
	(dst.f[NE])[c_nbArray[7] + idx]  = (sw * isobs) + (ONEMINUSOMEGA* ne + DFL3_OMEGA * rho * (1.0 + (+ux + uy) * (4.5 * (+ux + uy) + 3.0) - u2))* (!isobs);
	//printf("\n(7. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 7] + idx, (dst.f[NE])[c_nbArray[7] + idx]);
	(dst.f[NW])[c_nbArray[8] + idx]  = (se * isobs) + (ONEMINUSOMEGA* nw + DFL3_OMEGA * rho * (1.0 + (-ux + uy) * (4.5 * (-ux + uy) + 3.0) - u2))* (!isobs);
	//printf("\n(8. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 8] + idx, (dst.f[NW])[c_nbArray[8] + idx]);
	(dst.f[SE])[c_nbArray[9] + idx]  = (nw * isobs) + (ONEMINUSOMEGA* se + DFL3_OMEGA * rho * (1.0 + (+ux - uy) * (4.5 * (+ux - uy) + 3.0) - u2))* (!isobs);
	//printf("\n(9. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 9] + idx, (dst.f[SE])[c_nbArray[9] + idx]);
	(dst.f[SW])[c_nbArray[10] + idx] = (ne * isobs)+ (ONEMINUSOMEGA* sw + DFL3_OMEGA * rho * (1.0 + (-ux - uy) * (4.5 * (-ux - uy) + 3.0) - u2))* (!isobs);
	//printf("\n(10. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 10] + idx, (dst.f[SW])[c_nbArray[10] + idx]);
	(dst.f[NT])[c_nbArray[11] + idx] = (sb * isobs)+ (ONEMINUSOMEGA* nt + DFL3_OMEGA * rho * (1.0 + (+uy + uz) * (4.5 * (+uy + uz) + 3.0) - u2))* (!isobs);
	//printf("\n(11. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 11] + idx, (dst.f[NT])[c_nbArray[11] + idx]);
	(dst.f[NB])[c_nbArray[12] + idx] = (st * isobs)+ (ONEMINUSOMEGA* nb + DFL3_OMEGA * rho * (1.0 + (+uy - uz) * (4.5 * (+uy - uz) + 3.0) - u2))* (!isobs);
	//printf("\n(12. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 12] + idx, (dst.f[NB])[c_nbArray[12] + idx]);
	(dst.f[ST])[c_nbArray[13] + idx] = (nb * isobs)+ (ONEMINUSOMEGA* st + DFL3_OMEGA * rho * (1.0 + (-uy + uz) * (4.5 * (-uy + uz) + 3.0) - u2))* (!isobs);
	//printf("\n(13. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 13] + idx, (dst.f[ST])[c_nbArray[13] + idx]);
	(dst.f[SB])[c_nbArray[14] + idx] = (nt * isobs)+ (ONEMINUSOMEGA* sb + DFL3_OMEGA * rho * (1.0 + (-uy - uz) * (4.5 * (-uy - uz) + 3.0) - u2))* (!isobs);
	//printf("\n(14. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 14] + idx, (dst.f[SB])[c_nbArray[14] + idx]);
	(dst.f[ET])[c_nbArray[15] + idx] = (wb * isobs)+ (ONEMINUSOMEGA* et + DFL3_OMEGA * rho * (1.0 + (+ux + uz) * (4.5 * (+ux + uz) + 3.0) - u2))* (!isobs);
	//printf("\n(15. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 15] + idx, (dst.f[ET])[c_nbArray[15] + idx]);
	(dst.f[EB])[c_nbArray[16] + idx] = (wt * isobs)+ (ONEMINUSOMEGA* eb + DFL3_OMEGA * rho * (1.0 + (+ux - uz) * (4.5 * (+ux - uz) + 3.0) - u2))* (!isobs);
	//printf("\n(16. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 16] + idx, (dst.f[EB])[c_nbArray[16] + idx]);
	(dst.f[WT])[c_nbArray[17] + idx] = (eb * isobs)+ (ONEMINUSOMEGA* wt + DFL3_OMEGA * rho * (1.0 + (-ux + uz) * (4.5 * (-ux + uz) + 3.0) - u2))* (!isobs);
	//printf("\n(17. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 17] + idx, (dst.f[WT])[c_nbArray[17] + idx]);
	(dst.f[WB])[c_nbArray[18] + idx] = (et * isobs)+ (ONEMINUSOMEGA* wb + DFL3_OMEGA * rho * (1.0 + (-ux - uz) * (4.5 * (-ux - uz) + 3.0) - u2))* (!isobs);
	//printf("\n(18. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 18] + idx, (dst.f[WB])[c_nbArray[18] + idx]);
}
__global__ void /*__launch_bounds__(128,8)*/ lbm_kernel_SoA_Struct(Distributions src, Distributions dst, unsigned char *flags)
{

		//float *shPropPointer = (float*)array;
		int x = threadIdx.x;
		int y = blockIdx.x+1;
		int z = blockIdx.y+1;

		//int ElementsPerBlock = blockDim.x;
		//float c, n, s, e, w, t, b, ne, nw, se, sw, nt, nb, st, sb, et,	eb, wt, wb;

		/*the grid is organized as follows:
		 * c c c c c c . . . | n n n n n n . . . | s s s s s s . . . | e e e e e e . . . | . . .
		 */

		//calculate the index
		//int idx = (z * gridDim.x + y)*blockDim.x + threadIdx.x;
		int idx = (z * SIZE_YY + y)*SIZE_XX + x;
		//idx = calc_idx(x, y, z, 0);

		float c = (src.f[C])[idx];

		float n = (src.f[N])[idx];

		float s = (src.f[S])[idx];

		float e = (src.f[E])[idx];

		float w = (src.f[W])[idx];

		float t = (src.f[T])[idx];

		float b = (src.f[B])[idx];

		float ne = (src.f[NE])[idx];

		float nw = (src.f[NW])[idx];

		float se = (src.f[SE])[idx];

		float sw = (src.f[SW])[idx];

		float nt = (src.f[NT])[idx];

		float nb = (src.f[NB])[idx];

		float st = (src.f[ST])[idx];

		float sb = (src.f[SB])[idx];

		float et = (src.f[ET])[idx];

		float eb = (src.f[EB])[idx];

		float wt = (src.f[WT])[idx];

		float wb = (src.f[WB])[idx];

		//get the value final cell (20) to know the kind of cell: fluid or accel or obstacle
	    //unsigned int flag = flags[idx];
		unsigned short type = ((flags[idx] == 1) << 8) | ((flags[idx] == 2) & 0xff);
		//int isobs = (flags[idx] == 1);
		//int isacc = (flags[idx] == 2);

		float rho = +c + n + s + e + w + t + b + ne + nw + se + sw + nt + nb + st + sb + et + eb + wt + wb;

		float ux = ((+e - w + ne - nw + se - sw + et + eb - wt - wb)/rho)*(!(type & 0xff)) + 0.005*(type & 0xff);
		float uy = ((+n - s + ne + nw - se - sw + nt + nb - st - sb)/rho)*(!(type & 0xff)) + 0.002*(type & 0xff);
		float uz = ((+t - b + nt - nb + st - sb + et - eb + wt - wb)/rho)*(!(type & 0xff));


		/*float ux = +e - w + ne - nw + se - sw + et + eb - wt - wb;
		float uy = +n - s + ne + nw - se - sw + nt + nb - st - sb;
		float uz = +t - b + nt - nb + st - sb + et - eb + wt - wb;

		ux = ((ux / rho) * (!isacc) + 0.005 * isacc);
		uy = ((uy / rho) * (!isacc) + 0.002 * isacc);
		uz = ((uz / rho) * (!isacc) + 0.000 * isacc);

		*/
		float u2 =  1.5 * (ux * ux + uy * uy + uz * uz); //until this row, number of registers are 88 but why this row is used, number of registers are up to 177 ????

		//streaming(dst, idx, c, n, s, e, w,  t,  b,  ne,  nw,  se,  sw, nt,  nb,  st,  sb,  et,  eb,  wt,  wb, ux,  uy,  uz,  u2,  rho, isobs, isacc);

		(dst.f[C])[c_nbArray[ 0] + idx]  = (c * (type >> 8))	+ (ONEMINUSOMEGA* c + DFL1_OMEGA * rho * (1.0 - u2))* (!(type >> 8));
		//printf("\n(0. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 0] + idx, (dst.f[C])[c_nbArray[ 0] + idx]);
		(dst.f[N])[c_nbArray[ 1] + idx]  =  (s * (type >> 8))  + (ONEMINUSOMEGA* n + DFL2_OMEGA * rho * (1.0 + uy * (4.5 * uy + 3.0) - u2))* (!(type >> 8));
		//printf("\n(1. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 1] + idx, (dst.f[N])[c_nbArray[ 1] + idx]);
		(dst.f[S])[c_nbArray[ 2] + idx]  =  (n * (type >> 8))  + (ONEMINUSOMEGA* s + DFL2_OMEGA * rho * (1.0 + uy * (4.5 * uy - 3.0) - u2))* (!(type >> 8));
		//printf("\n(2. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 2] + idx, (dst.f[S])[c_nbArray[ 2] + idx]);
		(dst.f[E])[c_nbArray[ 3] + idx]  =  (w * (type >> 8))	+ (ONEMINUSOMEGA* e + DFL2_OMEGA * rho * (1.0 + ux * (4.5 * ux + 3.0) - u2))* (!(type >> 8));
		//printf("\n(3. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 3] + idx, (dst.f[E])[c_nbArray[ 3] + idx]);
		(dst.f[W])[c_nbArray[ 4] + idx]  =  (e * (type >> 8))	+ (ONEMINUSOMEGA* w + DFL2_OMEGA * rho * (1.0 + ux * (4.5 * ux - 3.0) - u2))* (!(type >> 8));
		//printf("\n(4. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 4] + idx, (dst.f[W])[c_nbArray[ 4] + idx]);
		(dst.f[T])[c_nbArray[ 5] + idx]  =  (b * (type >> 8))	+ (ONEMINUSOMEGA* t + DFL2_OMEGA * rho * (1.0 + uz * (4.5 * uz + 3.0) - u2))* (!(type >> 8));
		//printf("\n(5. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 5] + idx, (dst.f[T])[c_nbArray[ 5] + idx]);
		(dst.f[B])[c_nbArray[ 6] + idx]  =  (t * (type >> 8))	+ (ONEMINUSOMEGA* b + DFL2_OMEGA * rho * (1.0 + uz * (4.5 * uz - 3.0) - u2))* (!(type >> 8));
		//printf("\n(6. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 6] + idx, (dst.f[B])[c_nbArray[ 6] + idx]);
		(dst.f[NE])[c_nbArray[7] + idx]  =  (sw * (type >> 8)) + (ONEMINUSOMEGA* ne + DFL3_OMEGA * rho * (1.0 + (+ux + uy) * (4.5 * (+ux + uy) + 3.0) - u2))* (!(type >> 8));
		//printf("\n(7. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 7] + idx, (dst.f[NE])[c_nbArray[7] + idx]);
		(dst.f[NW])[c_nbArray[8] + idx]  =  (se * (type >> 8)) + (ONEMINUSOMEGA* nw + DFL3_OMEGA * rho * (1.0 + (-ux + uy) * (4.5 * (-ux + uy) + 3.0) - u2))* (!(type >> 8));
		//printf("\n(8. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 8] + idx, (dst.f[NW])[c_nbArray[8] + idx]);
		(dst.f[SE])[c_nbArray[9] + idx]  =  (nw * (type >> 8)) + (ONEMINUSOMEGA* se + DFL3_OMEGA * rho * (1.0 + (+ux - uy) * (4.5 * (+ux - uy) + 3.0) - u2))* (!(type >> 8));
		//printf("\n(9. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 9] + idx, (dst.f[SE])[c_nbArray[9] + idx]);
		(dst.f[SW])[c_nbArray[10] + idx] =  (ne * (type >> 8)) + (ONEMINUSOMEGA* sw + DFL3_OMEGA * rho * (1.0 + (-ux - uy) * (4.5 * (-ux - uy) + 3.0) - u2))* (!(type >> 8));
		//printf("\n(10. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 10] + idx, (dst.f[SW])[c_nbArray[10] + idx]);
		(dst.f[NT])[c_nbArray[11] + idx] =  (sb * (type >> 8))+ (ONEMINUSOMEGA* nt + DFL3_OMEGA * rho * (1.0 + (+uy + uz) * (4.5 * (+uy + uz) + 3.0) - u2))* (!(type >> 8));
		//printf("\n(11. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 11] + idx, (dst.f[NT])[c_nbArray[11] + idx]);
		(dst.f[NB])[c_nbArray[12] + idx] =  (st * (type >> 8))+ (ONEMINUSOMEGA* nb + DFL3_OMEGA * rho * (1.0 + (+uy - uz) * (4.5 * (+uy - uz) + 3.0) - u2))* (!(type >> 8));
		//printf("\n(12. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 12] + idx, (dst.f[NB])[c_nbArray[12] + idx]);
		(dst.f[ST])[c_nbArray[13] + idx] =  (nb * (type >> 8))+ (ONEMINUSOMEGA* st + DFL3_OMEGA * rho * (1.0 + (-uy + uz) * (4.5 * (-uy + uz) + 3.0) - u2))* (!(type >> 8));
		//printf("\n(13. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 13] + idx, (dst.f[ST])[c_nbArray[13] + idx]);
		(dst.f[SB])[c_nbArray[14] + idx] =  (nt * (type >> 8))+ (ONEMINUSOMEGA* sb + DFL3_OMEGA * rho * (1.0 + (-uy - uz) * (4.5 * (-uy - uz) + 3.0) - u2))* (!(type >> 8));
		//printf("\n(14. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 14] + idx, (dst.f[SB])[c_nbArray[14] + idx]);
		(dst.f[ET])[c_nbArray[15] + idx] =  (wb * (type >> 8))+ (ONEMINUSOMEGA* et + DFL3_OMEGA * rho * (1.0 + (+ux + uz) * (4.5 * (+ux + uz) + 3.0) - u2))* (!(type >> 8));
		//printf("\n(15. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 15] + idx, (dst.f[ET])[c_nbArray[15] + idx]);
		(dst.f[EB])[c_nbArray[16] + idx] =  (wt * (type >> 8))+ (ONEMINUSOMEGA* eb + DFL3_OMEGA * rho * (1.0 + (+ux - uz) * (4.5 * (+ux - uz) + 3.0) - u2))* (!(type >> 8));
		//printf("\n(16. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 16] + idx, (dst.f[EB])[c_nbArray[16] + idx]);
		(dst.f[WT])[c_nbArray[17] + idx] =  (eb * (type >> 8))+ (ONEMINUSOMEGA* wt + DFL3_OMEGA * rho * (1.0 + (-ux + uz) * (4.5 * (-ux + uz) + 3.0) - u2))* (!(type >> 8));
		//printf("\n(17. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 17] + idx, (dst.f[WT])[c_nbArray[17] + idx]);
		(dst.f[WB])[c_nbArray[18] + idx] =  (et * (type >> 8))+ (ONEMINUSOMEGA* wb + DFL3_OMEGA * rho * (1.0 + (-ux - uz) * (4.5 * (-ux - uz) + 3.0) - u2))* (!(type >> 8));
		//printf("\n(18. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 18] + idx, (dst.f[WB])[c_nbArray[18] + idx]);
}

__global__ void __launch_bounds__(128,8) lbm_kernel_SoA_Struct_arr(Distributions src, Distributions dst, int *flags)
{

		//float *shPropPointer = (float*)array;
		int x = threadIdx.x;
		int y = blockIdx.x+1;
		int z = blockIdx.y+1;

		//int ElementsPerBlock = blockDim.x;
		//float c, n, s, e, w, t, b, ne, nw, se, sw, nt, nb, st, sb, et,	eb, wt, wb;

		/*the grid is organized as follows:
		 * c c c c c c . . . | n n n n n n . . . | s s s s s s . . . | e e e e e e . . . | . . .
		 */

		//calculate the index
		//int idx = (z * gridDim.x + y)*blockDim.x + threadIdx.x;
		int idx = (z * SIZE_YY + y)*SIZE_XX + x;
		//idx = calc_idx(x, y, z, 0);
		float dir[19];

		dir[0] = (src.f[C])[idx];

		dir[1] = (src.f[N])[idx];

		dir[2] = (src.f[S])[idx];

		dir[3] = (src.f[E])[idx];

		dir[4] = (src.f[W])[idx];

		dir[5] = (src.f[T])[idx];

		dir[6] = (src.f[B])[idx];

		dir[7] = (src.f[NE])[idx];

		dir[8] = (src.f[NW])[idx];

		dir[9] = (src.f[SE])[idx];

		dir[10] = (src.f[SW])[idx];

		dir[11] = (src.f[NT])[idx];

		dir[12] = (src.f[NB])[idx];

		dir[13] = (src.f[ST])[idx];

		dir[14] = (src.f[SB])[idx];

		dir[15] = (src.f[ET])[idx];

		dir[16] = (src.f[EB])[idx];

		dir[17] = (src.f[WT])[idx];

		dir[18] = (src.f[WB])[idx];

//		float c = (src.f[C])[idx]; //0
//
//		float n = (src.f[N])[idx]; //1
//
//		float s = (src.f[S])[idx]; //2
//
//		float e = (src.f[E])[idx]; //3
//
//		float w = (src.f[W])[idx]; //4
//
//		float t = (src.f[T])[idx]; //5
//
//		float b = (src.f[B])[idx]; //6
//
//		float ne = (src.f[NE])[idx]; //7
//
//		float nw = (src.f[NW])[idx]; //8
//
//		float se = (src.f[SE])[idx]; //9
//
//		float sw = (src.f[SW])[idx]; //10
//
//		float nt = (src.f[NT])[idx]; //11
//
//		float nb = (src.f[NB])[idx]; //12
//
//		float st = (src.f[ST])[idx]; //13
//
//		float sb = (src.f[SB])[idx]; //14
//
//		float et = (src.f[ET])[idx]; //15
//
//		float eb = (src.f[EB])[idx]; //16
//
//		float wt = (src.f[WT])[idx]; //17
//
//		float wb = (src.f[WB])[idx]; //18

		//get the value final cell (20) to know the kind of cell: fluid or accel or obstacle
	    //unsigned int flag = flags[idx];

		int isobs = (flags[idx] == 1);
		int isacc = (flags[idx] == 2);

		float rho;
		for(int i=0;i<19;i++)
			rho += dir[i];
		//float rho = +c + n + s + e + w + t + b + ne + nw + se + sw + nt + nb + st + sb + et + eb + wt + wb;

		//float ux = ((+e - w + ne - nw + se - sw + et + eb - wt - wb)/rho)*(!isacc) + 0.005*isacc;
		//float uy = ((+n - s + ne + nw - se - sw + nt + nb - st - sb)/rho)*(!isacc) + 0.002*isacc;
		//float uz = ((+t - b + nt - nb + st - sb + et - eb + wt - wb)/rho)*(!isacc);

		float ux = ((+dir[3] - dir[4] + dir[7] - dir[8] + dir[9] - dir[10] + dir[15] + dir[16] - dir[17] - dir[18])/rho)*(!isacc) + 0.005*isacc;
		float uy = ((+dir[1] - dir[2] + dir[7] + dir[8] - dir[9] - dir[10] + dir[11] + dir[12] - dir[13] - dir[14])/rho)*(!isacc) + 0.002*isacc;
		float uz = ((+dir[5] - dir[6] + dir[11] - dir[12] + dir[13] - dir[14] + dir[15] - dir[16] + dir[17] - dir[18])/rho)*(!isacc);
		float u2 =  1.5 * (ux * ux + uy * uy + uz * uz); //until this row, number of registers are 88 but why this row is used, number of registers are up to 177 ????

		//streaming(dst, idx, c, n, s, e, w,  t,  b,  ne,  nw,  se,  sw, nt,  nb,  st,  sb,  et,  eb,  wt,  wb, ux,  uy,  uz,  u2,  rho, isobs, isacc);

		(dst.f[C])[c_nbArray[ 0] + idx]  = (dir[0] * isobs)	+ (ONEMINUSOMEGA* dir[0] + DFL1_OMEGA * rho * (1.0 - u2))* (!isobs);
		//printf("\n(0. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 0] + idx, (dst.f[C])[c_nbArray[ 0] + idx]);
		(dst.f[N])[c_nbArray[ 1] + idx]  = (dir[2] * isobs)  + (ONEMINUSOMEGA* dir[1] + DFL2_OMEGA * rho * (1.0 + uy * (4.5 * uy + 3.0) - u2))* (!isobs);
		//printf("\n(1. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 1] + idx, (dst.f[N])[c_nbArray[ 1] + idx]);
		(dst.f[S])[c_nbArray[ 2] + idx]  = (dir[1] * isobs)  + (ONEMINUSOMEGA* dir[2] + DFL2_OMEGA * rho * (1.0 + uy * (4.5 * uy - 3.0) - u2))* (!isobs);
		//printf("\n(2. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 2] + idx, (dst.f[S])[c_nbArray[ 2] + idx]);
		(dst.f[E])[c_nbArray[ 3] + idx]  = (dir[4] * isobs)	+ (ONEMINUSOMEGA* dir[3] + DFL2_OMEGA * rho * (1.0 + ux * (4.5 * ux + 3.0) - u2))* (!isobs);
		//printf("\n(3. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 3] + idx, (dst.f[E])[c_nbArray[ 3] + idx]);
		(dst.f[W])[c_nbArray[ 4] + idx]  = (dir[3] * isobs)	+ (ONEMINUSOMEGA* dir[4] + DFL2_OMEGA * rho * (1.0 + ux * (4.5 * ux - 3.0) - u2))* (!isobs);
		//printf("\n(4. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 4] + idx, (dst.f[W])[c_nbArray[ 4] + idx]);
		(dst.f[T])[c_nbArray[ 5] + idx]  = (dir[6] * isobs)	+ (ONEMINUSOMEGA* dir[5] + DFL2_OMEGA * rho * (1.0 + uz * (4.5 * uz + 3.0) - u2))* (!isobs);
		//printf("\n(5. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 5] + idx, (dst.f[T])[c_nbArray[ 5] + idx]);
		(dst.f[B])[c_nbArray[ 6] + idx]  = (dir[5] * isobs)	+ (ONEMINUSOMEGA* dir[6] + DFL2_OMEGA * rho * (1.0 + uz * (4.5 * uz - 3.0) - u2))* (!isobs);
		//printf("\n(6. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 6] + idx, (dst.f[B])[c_nbArray[ 6] + idx]);
		(dst.f[NE])[c_nbArray[7] + idx]  = (dir[10] * isobs) + (ONEMINUSOMEGA* dir[7] + DFL3_OMEGA * rho * (1.0 + (+ux + uy) * (4.5 * (+ux + uy) + 3.0) - u2))* (!isobs);
		//printf("\n(7. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 7] + idx, (dst.f[NE])[c_nbArray[7] + idx]);
		(dst.f[NW])[c_nbArray[8] + idx]  = (dir[9] * isobs) + (ONEMINUSOMEGA* dir[8] + DFL3_OMEGA * rho * (1.0 + (-ux + uy) * (4.5 * (-ux + uy) + 3.0) - u2))* (!isobs);
		//printf("\n(8. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 8] + idx, (dst.f[NW])[c_nbArray[8] + idx]);
		(dst.f[SE])[c_nbArray[9] + idx]  = (dir[8] * isobs) + (ONEMINUSOMEGA* dir[9] + DFL3_OMEGA * rho * (1.0 + (+ux - uy) * (4.5 * (+ux - uy) + 3.0) - u2))* (!isobs);
		//printf("\n(9. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 9] + idx, (dst.f[SE])[c_nbArray[9] + idx]);
		(dst.f[SW])[c_nbArray[10] + idx] = (dir[7] * isobs)+ (ONEMINUSOMEGA* dir[10] + DFL3_OMEGA * rho * (1.0 + (-ux - uy) * (4.5 * (-ux - uy) + 3.0) - u2))* (!isobs);
		//printf("\n(10. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 10] + idx, (dst.f[SW])[c_nbArray[10] + idx]);
		(dst.f[NT])[c_nbArray[11] + idx] = (dir[14] * isobs)+ (ONEMINUSOMEGA* dir[11] + DFL3_OMEGA * rho * (1.0 + (+uy + uz) * (4.5 * (+uy + uz) + 3.0) - u2))* (!isobs);
		//printf("\n(11. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 11] + idx, (dst.f[NT])[c_nbArray[11] + idx]);
		(dst.f[NB])[c_nbArray[12] + idx] = (dir[13] * isobs)+ (ONEMINUSOMEGA* dir[12] + DFL3_OMEGA * rho * (1.0 + (+uy - uz) * (4.5 * (+uy - uz) + 3.0) - u2))* (!isobs);
		//printf("\n(12. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 12] + idx, (dst.f[NB])[c_nbArray[12] + idx]);
		(dst.f[ST])[c_nbArray[13] + idx] = (dir[12] * isobs)+ (ONEMINUSOMEGA* dir[13] + DFL3_OMEGA * rho * (1.0 + (-uy + uz) * (4.5 * (-uy + uz) + 3.0) - u2))* (!isobs);
		//printf("\n(13. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 13] + idx, (dst.f[ST])[c_nbArray[13] + idx]);
		(dst.f[SB])[c_nbArray[14] + idx] = (dir[11] * isobs)+ (ONEMINUSOMEGA* dir[14] + DFL3_OMEGA * rho * (1.0 + (-uy - uz) * (4.5 * (-uy - uz) + 3.0) - u2))* (!isobs);
		//printf("\n(14. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 14] + idx, (dst.f[SB])[c_nbArray[14] + idx]);
		(dst.f[ET])[c_nbArray[15] + idx] = (dir[18] * isobs)+ (ONEMINUSOMEGA* dir[15] + DFL3_OMEGA * rho * (1.0 + (+ux + uz) * (4.5 * (+ux + uz) + 3.0) - u2))* (!isobs);
		//printf("\n(15. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 15] + idx, (dst.f[ET])[c_nbArray[15] + idx]);
		(dst.f[EB])[c_nbArray[16] + idx] = (dir[17] * isobs)+ (ONEMINUSOMEGA* dir[16] + DFL3_OMEGA * rho * (1.0 + (+ux - uz) * (4.5 * (+ux - uz) + 3.0) - u2))* (!isobs);
		//printf("\n(16. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 16] + idx, (dst.f[EB])[c_nbArray[16] + idx]);
		(dst.f[WT])[c_nbArray[17] + idx] = (dir[16] * isobs)+ (ONEMINUSOMEGA* dir[17] + DFL3_OMEGA * rho * (1.0 + (-ux + uz) * (4.5 * (-ux + uz) + 3.0) - u2))* (!isobs);
		//printf("\n(17. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 17] + idx, (dst.f[WT])[c_nbArray[17] + idx]);
		(dst.f[WB])[c_nbArray[18] + idx] = (dir[15] * isobs)+ (ONEMINUSOMEGA* dir[18] + DFL3_OMEGA * rho * (1.0 + (-ux - uz) * (4.5 * (-ux - uz) + 3.0) - u2))* (!isobs);
		//printf("\n(18. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 18] + idx, (dst.f[WB])[c_nbArray[18] + idx]);
}
//__global__ void lbm_kernel_SoA_Struct_sm(Distributions src, Distributions dst, int *flags)
//{
//	__shared__ float s_E[SIZE_X +1];
//	__shared__ float s_W[SIZE_X +1];
//	__shared__ float s_SE[SIZE_X+1];
//	__shared__ float s_NE[SIZE_X+1];
//	__shared__ float s_SW[SIZE_X+1];
//	__shared__ float s_NW[SIZE_X+1];
//	__shared__ float s_EB[SIZE_X+1];
//	__shared__ float s_ET[SIZE_X+1];
//	__shared__ float s_WB[SIZE_X+1];
//	__shared__ float s_WT[SIZE_X+1];
//
//	//float *shPropPointer = (float*)array;
//	int x = threadIdx.x;
//	int y = blockIdx.x+1;
//	int z = blockIdx.y+1;
//
//	//int ElementsPerBlock = blockDim.x;
//	//float c, n, s, e, w, t, b, ne, nw, se, sw, nt, nb, st, sb, et,	eb, wt, wb;
//
//	/*the grid is organized as follows:
//	 * c c c c c c . . . | n n n n n n . . . | s s s s s s . . . | e e e e e e . . . | . . .
//	 */
//
//	//calculate the index
//	//int idx = (z * gridDim.x + y)*blockDim.x + threadIdx.x;
//	int idx = (z * SIZE_YY + y)*SIZE_XX + x;
//	//idx = calc_idx(x, y, z, 0);
//
//	float c = (src.f[C])[idx];
//
//	float n = (src.f[N])[idx];
//
//	float s = (src.f[S])[idx];
//
//	float e = (src.f[E])[idx];
//
//	float w = (src.f[W])[idx];
//
//	float t = (src.f[T])[idx];
//
//	float b = (src.f[B])[idx];
//
//	float ne = (src.f[NE])[idx];
//
//	float nw = (src.f[NW])[idx];
//
//	float se = (src.f[SE])[idx];
//
//	float sw = (src.f[SW])[idx];
//
//	float nt = (src.f[NT])[idx];
//
//	float nb = (src.f[NB])[idx];
//
//	float st = (src.f[ST])[idx];
//
//	float sb = (src.f[SB])[idx];
//
//	float et = (src.f[ET])[idx];
//
//	float eb = (src.f[EB])[idx];
//
//	float wt = (src.f[WT])[idx];
//
//	float wb = (src.f[WB])[idx];
//
//	//get the value final cell (20) to know the kind of cell: fluid or accel or obstacle
//	//unsigned int flag = flags[idx];
//
//	int isobs = (flags[idx] == 1);
//	int isacc = (flags[idx] == 2);
//
//	float rho = +c + n + s + e + w + t + b + ne + nw + se + sw + nt + nb + st
//			+ sb + et + eb + wt + wb;
//
//	float ux = ((+e - w + ne - nw + se - sw + et + eb - wt - wb)/rho)*(!isacc) + 0.005*isacc;
//	float uy = ((+n - s + ne + nw - se - sw + nt + nb - st - sb)/rho)*(!isacc) + 0.002*isacc;
//	float uz = ((+t - b + nt - nb + st - sb + et - eb + wt - wb)/rho)*(!isacc);
//
////		float ux = +e - w + ne - nw + se - sw + et + eb - wt - wb;
////		float uy = +n - s + ne + nw - se - sw + nt + nb - st - sb;
////		float uz = +t - b + nt - nb + st - sb + et - eb + wt - wb;
////
////		ux = ((ux / rho) * (!isacc) + 0.005 * isacc);
////		uy = ((uy / rho) * (!isacc) + 0.002 * isacc);
////		uz = ((uz / rho) * (!isacc) + 0.000 * isacc);
//
//
//	float u2 =  1.5 * (ux * ux + uy * uy + uz * uz); //until this row, number of registers are 88 but why this row is used, number of registers are up to 177 ????
//
//
//	(dst.f[C])[c_nbArray[ 0] + idx]  = (c * isobs)	+ (ONEMINUSOMEGA* c + DFL1_OMEGA * rho * (1.0 - u2))* (!isobs);
//	//printf("\n(%d, %d, %d) C %d", x,y,z, c_nbArray[ 0] + idx);
//	//printf("\n(0. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 0] + idx, (dst.f[C])[c_nbArray[ 0] + idx]);
//	(dst.f[N])[c_nbArray[ 1] + idx]  = (s * isobs)  + (ONEMINUSOMEGA* n + DFL2_OMEGA * rho * (1.0 + uy * (4.5 * uy + 3.0) - u2))* (!isobs);
//	//printf("\n(%d, %d, %d) N %d", x,y,z, c_nbArray[ 1] + idx);
//	//printf("\n(1. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 1] + idx, (dst.f[N])[c_nbArray[ 1] + idx]);
//	(dst.f[S])[c_nbArray[ 2] + idx]  = (n * isobs)  + (ONEMINUSOMEGA* s + DFL2_OMEGA * rho * (1.0 + uy * (4.5 * uy - 3.0) - u2))* (!isobs);
//	//printf("\n(%d, %d, %d) S %d", x,y,z, c_nbArray[ 2] + idx);
//	//printf("\n(2. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 2] + idx, (dst.f[S])[c_nbArray[ 2] + idx]);
//
//	(dst.f[T])[c_nbArray[ 5] + idx]  = (b * isobs)	+ (ONEMINUSOMEGA* t + DFL2_OMEGA * rho * (1.0 + uz * (4.5 * uz + 3.0) - u2))* (!isobs);
//	//printf("\n(%d, %d, %d) T %d", x,y,z, c_nbArray[ 5] + idx);
//	//printf("\n(5. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 5] + idx, (dst.f[T])[c_nbArray[ 5] + idx]);
//	(dst.f[B])[c_nbArray[ 6] + idx]  = (t * isobs)	+ (ONEMINUSOMEGA* b + DFL2_OMEGA * rho * (1.0 + uz * (4.5 * uz - 3.0) - u2))* (!isobs);
//	//printf("\n(%d, %d, %d) B %d", x,y,z, c_nbArray[ 6] + idx);
//	//printf("\n(6. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 6] + idx, (dst.f[B])[c_nbArray[ 6] + idx]);
//
//	(dst.f[NT])[c_nbArray[11] + idx] = (sb * isobs)+ (ONEMINUSOMEGA* nt + DFL3_OMEGA * rho * (1.0 + (+uy + uz) * (4.5 * (+uy + uz) + 3.0) - u2))* (!isobs);
//	//printf("\n(%d, %d, %d) NT %d", x,y,z, c_nbArray[ 11] + idx);
//	//printf("\n(11. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 11] + idx, (dst.f[NT])[c_nbArray[11] + idx]);
//	(dst.f[NB])[c_nbArray[12] + idx] = (st * isobs)+ (ONEMINUSOMEGA* nb + DFL3_OMEGA * rho * (1.0 + (+uy - uz) * (4.5 * (+uy - uz) + 3.0) - u2))* (!isobs);
//	//printf("\n(%d, %d, %d) NB %d", x,y,z, c_nbArray[ 12] + idx);
//	//printf("\n(12. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 12] + idx, (dst.f[NB])[c_nbArray[12] + idx]);
//	(dst.f[ST])[c_nbArray[13] + idx] = (nb * isobs)+ (ONEMINUSOMEGA* st + DFL3_OMEGA * rho * (1.0 + (-uy + uz) * (4.5 * (-uy + uz) + 3.0) - u2))* (!isobs);
//	//printf("\n(%d, %d, %d) ST %d", x,y,z, c_nbArray[ 13] + idx);
//	//printf("\n(13. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 13] + idx, (dst.f[ST])[c_nbArray[13] + idx]);
//	(dst.f[SB])[c_nbArray[14] + idx] = (nt * isobs)+ (ONEMINUSOMEGA* sb + DFL3_OMEGA * rho * (1.0 + (-uy - uz) * (4.5 * (-uy - uz) + 3.0) - u2))* (!isobs);
//	//printf("\n(%d, %d, %d) SB %d", x,y,z, c_nbArray[ 14] + idx);
//	//printf("\n(14. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 14] + idx, (dst.f[SB])[c_nbArray[14] + idx]);
//
//
//	//int shiftE = 0;//((y-1)&0x1)^((z-1)&0x1);
//	int shiftW = 0;//0x1 & (~shiftE);
//	int txE = x;//+shiftE;
//	int txW = x;//-shiftW;
//
//	s_E[txE] = (w * isobs)	+ (ONEMINUSOMEGA* e + DFL2_OMEGA * rho * (1.0 + ux * (4.5 * ux + 3.0) - u2))* (!isobs);
//	//printf("\n(%d, %d, %d) s_E[%d] %.8f", x, y, z, txE, s_E[txE]);
//	s_SE[txE] = (nw * isobs) + (ONEMINUSOMEGA* se + DFL3_OMEGA * rho * (1.0 + (+ux - uy) * (4.5 * (+ux - uy) + 3.0) - u2))* (!isobs);
//	//printf("\n(%d, %d, %d) s_SE[%d] %.8f", x, y, z, txE, s_SE[txE]);
//	s_NE[txE] = (sw * isobs) + (ONEMINUSOMEGA* ne + DFL3_OMEGA * rho * (1.0 + (+ux + uy) * (4.5 * (+ux + uy) + 3.0) - u2))* (!isobs);
//	//printf("\n(%d, %d, %d) s_NE[%d] %.8f", x, y, z, txE, s_NE[txE]);
//	s_EB[txE] = (wt * isobs)+ (ONEMINUSOMEGA* eb + DFL3_OMEGA * rho * (1.0 + (+ux - uz) * (4.5 * (+ux - uz) + 3.0) - u2))* (!isobs);
//	//printf("\n(%d, %d, %d) s_EB[%d] %.8f", x, y, z, txE, s_EB[txE]);
//	s_ET[txE] = (wb * isobs)+ (ONEMINUSOMEGA* et + DFL3_OMEGA * rho * (1.0 + (+ux + uz) * (4.5 * (+ux + uz) + 3.0) - u2))* (!isobs);
//	//printf("\n(%d, %d, %d) s_ET[%d] %.8f", x, y, z, txE, s_ET[txE]);
//
//	s_W[txW+1] = (e * isobs)	+ (ONEMINUSOMEGA* w + DFL2_OMEGA * rho * (1.0 + ux * (4.5 * ux - 3.0) - u2))* (!isobs);
//	////printf("\n(%d, %d, %d) s_W[%d] %.8f", x, y, z, txW+1, s_W[txW+1]);
//	s_SW[txW+1] = (ne * isobs)+ (ONEMINUSOMEGA* sw + DFL3_OMEGA * rho * (1.0 + (-ux - uy) * (4.5 * (-ux - uy) + 3.0) - u2))* (!isobs);
//	//printf("\n(%d, %d, %d) s_SW[%d] %.8f", x, y, z, txW+1, s_SW[txW+1]);
//	s_NW[txW+1] = (se * isobs) + (ONEMINUSOMEGA* nw + DFL3_OMEGA * rho * (1.0 + (-ux + uy) * (4.5 * (-ux + uy) + 3.0) - u2))* (!isobs);
//	//printf("\n(%d, %d, %d) s_NW[%d] %.8f", x, y, z, txW+1, s_NW[txW+1]);
//	s_WB[txW+1] = (et * isobs)+ (ONEMINUSOMEGA* wb + DFL3_OMEGA * rho * (1.0 + (-ux - uz) * (4.5 * (-ux - uz) + 3.0) - u2))* (!isobs);
//	//printf("\n(%d, %d, %d) s_WB[%d] %.8f", x, y, z, txW+1, s_WB[txW+1]);
//	s_WT[txW+1] = (eb * isobs)+ (ONEMINUSOMEGA* wt + DFL3_OMEGA * rho * (1.0 + (-ux + uz) * (4.5 * (-ux + uz) + 3.0) - u2))* (!isobs);
//	//printf("\n(%d, %d, %d) s_WT[%d] %.8f", x, y, z, txW+1, s_WT[txW+1]);
//
//	__syncthreads();
//
//	(dst.f[E])[c_nbArray[ 3] + idx]  = s_E[x];
//			//(w * isobs)	+ (ONEMINUSOMEGA* e + DFL2_OMEGA * rho * (1.0 + ux * (4.5 * ux + 3.0) - u2))* (!isobs);
//
//				//printf("\n(%d, %d, %d) E %d", x,y,z, c_nbArray[ 3] + idx);
//				//printf("\n(3. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 3] + idx, (dst.f[E])[c_nbArray[ 3] + idx]);
//	(dst.f[W])[c_nbArray[ 4] + idx]  = s_W[x+1];
//			//(e * isobs)	+ (ONEMINUSOMEGA* w + DFL2_OMEGA * rho * (1.0 + ux * (4.5 * ux - 3.0) - u2))* (!isobs);
//
//				//printf("\n(%d, %d, %d) W %d", x,y,z, c_nbArray[ 4] + idx);
//				//printf("\n(4. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 4] + idx, (dst.f[W])[c_nbArray[ 4] + idx]);
//	(dst.f[NE])[c_nbArray[7] + idx]  = s_NE[x];
//	//printf("\n(%d, %d, %d) NE %d", x,y,z, c_nbArray[ 7] + idx);
//			//(sw * isobs) + (ONEMINUSOMEGA* ne + DFL3_OMEGA * rho * (1.0 + (+ux + uy) * (4.5 * (+ux + uy) + 3.0) - u2))* (!isobs);
//	//printf("\n(7. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 7] + idx, (dst.f[NE])[c_nbArray[7] + idx]);
//	(dst.f[NW])[c_nbArray[8] + idx]  = s_NW[x+1];
//	//printf("\n(%d, %d, %d) NW %d", x,y,z, c_nbArray[ 8] + idx);
//			//(se * isobs) + (ONEMINUSOMEGA* nw + DFL3_OMEGA * rho * (1.0 + (-ux + uy) * (4.5 * (-ux + uy) + 3.0) - u2))* (!isobs);
//	//printf("\n(8. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 8] + idx, (dst.f[NW])[c_nbArray[8] + idx]);
//	(dst.f[SE])[c_nbArray[9] + idx]  =  s_SE[x];
//	//printf("\n(%d, %d, %d) SE %d", x,y,z, c_nbArray[ 9] + idx);
//			//(nw * isobs) + (ONEMINUSOMEGA* se + DFL3_OMEGA * rho * (1.0 + (+ux - uy) * (4.5 * (+ux - uy) + 3.0) - u2))* (!isobs);
//	//printf("\n(9. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 9] + idx, (dst.f[SE])[c_nbArray[9] + idx]);
//	(dst.f[SW])[c_nbArray[10] + idx] = s_SW[x+1];
//	//printf("\n(%d, %d, %d) SW %d", x,y,z, c_nbArray[ 10] + idx);
//			//(ne * isobs)+ (ONEMINUSOMEGA* sw + DFL3_OMEGA * rho * (1.0 + (-ux - uy) * (4.5 * (-ux - uy) + 3.0) - u2))* (!isobs);
//	//printf("\n(10. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 10] + idx, (dst.f[SW])[c_nbArray[10] + idx]);
//
//	(dst.f[ET])[c_nbArray[15] + idx] = s_ET[x];
//	//printf("\n(%d, %d, %d) ET %d", x,y,z, c_nbArray[ 15] + idx);
//			//(wb * isobs)+ (ONEMINUSOMEGA* et + DFL3_OMEGA * rho * (1.0 + (+ux + uz) * (4.5 * (+ux + uz) + 3.0) - u2))* (!isobs);
//	//printf("\n(15. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 15] + idx, (dst.f[ET])[c_nbArray[15] + idx]);
//	(dst.f[EB])[c_nbArray[16] + idx] = s_EB[x];
//	//printf("\n(%d, %d, %d) EB %d", x,y,z, c_nbArray[ 16] + idx);
//			//(wt * isobs)+ (ONEMINUSOMEGA* eb + DFL3_OMEGA * rho * (1.0 + (+ux - uz) * (4.5 * (+ux - uz) + 3.0) - u2))* (!isobs);
//	//printf("\n(16. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 16] + idx, (dst.f[EB])[c_nbArray[16] + idx]);
//	(dst.f[WT])[c_nbArray[17] + idx] = s_WT[x+1];
//	//printf("\n(%d, %d, %d) WT %d", x,y,z, c_nbArray[ 17] + idx);
//			//(eb * isobs)+ (ONEMINUSOMEGA* wt + DFL3_OMEGA * rho * (1.0 + (-ux + uz) * (4.5 * (-ux + uz) + 3.0) - u2))* (!isobs);
//	//printf("\n(17. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 17] + idx, (dst.f[WT])[c_nbArray[17] + idx]);
//	(dst.f[WB])[c_nbArray[18] + idx] =  s_WB[x+1];
//	//printf("\n(%d, %d, %d) WB %d", x,y,z, c_nbArray[ 18] + idx);
//			//(et * isobs)+ (ONEMINUSOMEGA* wb + DFL3_OMEGA * rho * (1.0 + (-ux - uz) * (4.5 * (-ux - uz) + 3.0) - u2))* (!isobs);
//	//printf("\n(18. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 18] + idx, (dst.f[WB])[c_nbArray[18] + idx]);
//}
//__global__ void lbm_kernel_SoA_Struct_sm_arr(float *sfc, float *sfn, float *sfs, float *sfe,float *sfw,float *sft,float *sfb,float *sfne,float *sfnw,float *sfse,float *sfsw,float *sfnt,float *sfnb,float *sfst,float *sfsb,float *sfet,float *sfeb,float *sfwt,float *sfwb,
//											float *dfc, float *dfn, float *dfs, float *dfe,float *dfw,float *dft,float *dfb,float *dfne,float *dfnw,float *dfse,float *dfsw,float *dfnt,float *dfnb,float *dfst,float *dfsb,float *dfet,float *dfeb,float *dfwt,float *dfwb,
//											int *flags)
//{
//	__shared__ float s_E[SIZE_X +1];
//	__shared__ float s_W[SIZE_X +1];
//	__shared__ float s_SE[SIZE_X+1];
//	__shared__ float s_NE[SIZE_X+1];
//	__shared__ float s_SW[SIZE_X+1];
//	__shared__ float s_NW[SIZE_X+1];
//	__shared__ float s_EB[SIZE_X+1];
//	__shared__ float s_ET[SIZE_X+1];
//	__shared__ float s_WB[SIZE_X+1];
//	__shared__ float s_WT[SIZE_X+1];
//
//	//float *shPropPointer = (float*)array;
//	int x = threadIdx.x;
//	int y = blockIdx.x+1;
//	int z = blockIdx.y+1;
//
//	//int ElementsPerBlock = blockDim.x;
//	//float c, n, s, e, w, t, b, ne, nw, se, sw, nt, nb, st, sb, et,	eb, wt, wb;
//
//	/*the grid is organized as follows:
//	 * c c c c c c . . . | n n n n n n . . . | s s s s s s . . . | e e e e e e . . . | . . .
//	 */
//
//	//calculate the index
//	//int idx = (z * gridDim.x + y)*blockDim.x + threadIdx.x;
//	int idx = (z * SIZE_YY + y)*SIZE_XX + x;
//	//idx = calc_idx(x, y, z, 0);
//
//	float c = sfc[idx];
//
//	float n = sfn[idx];
//
//	float s = sfs[idx];
//
//	float e = sfe[idx];
//
//	float w = sfw[idx];
//
//	float t = sft[idx];
//
//	float b = sfb[idx];
//
//	float ne = sfne[idx];
//
//	float nw = sfnw[idx];
//
//	float se = sfse[idx];
//
//	float sw = sfsw[idx];
//
//	float nt = sfnt[idx];
//
//	float nb = sfnb[idx];
//
//	float st = sfst[idx];
//
//	float sb = sfsb[idx];
//
//	float et = sfet[idx];
//
//	float eb = sfeb[idx];
//
//	float wt = sfwt[idx];
//
//	float wb = sfwb[idx];
//
//	//get the value final cell (20) to know the kind of cell: fluid or accel or obstacle
//	//unsigned int flag = flags[idx];
//
//	int isobs = (flags[idx] == 1);
//	int isacc = (flags[idx] == 2);
//
//	float rho = +c + n + s + e + w + t + b + ne + nw + se + sw + nt + nb + st
//			+ sb + et + eb + wt + wb;
//
//	float ux = ((+e - w + ne - nw + se - sw + et + eb - wt - wb)/rho)*(!isacc) + 0.005*isacc;
//	float uy = ((+n - s + ne + nw - se - sw + nt + nb - st - sb)/rho)*(!isacc) + 0.002*isacc;
//	float uz = ((+t - b + nt - nb + st - sb + et - eb + wt - wb)/rho)*(!isacc);
//
////		float ux = +e - w + ne - nw + se - sw + et + eb - wt - wb;
////		float uy = +n - s + ne + nw - se - sw + nt + nb - st - sb;
////		float uz = +t - b + nt - nb + st - sb + et - eb + wt - wb;
////
////		ux = ((ux / rho) * (!isacc) + 0.005 * isacc);
////		uy = ((uy / rho) * (!isacc) + 0.002 * isacc);
////		uz = ((uz / rho) * (!isacc) + 0.000 * isacc);
//
//
//	float u2 =  1.5 * (ux * ux + uy * uy + uz * uz); //until this row, number of registers are 88 but why this row is used, number of registers are up to 177 ????
//
//
//	dfc[c_nbArray[ 0] + idx]  = (c * isobs)	+ (ONEMINUSOMEGA* c + DFL1_OMEGA * rho * (1.0 - u2))* (!isobs);
//	//printf("\n(%d, %d, %d) C %d", x,y,z, c_nbArray[ 0] + idx);
//	//printf("\n(0. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 0] + idx, (dst.f[C])[c_nbArray[ 0] + idx]);
//	dfn[c_nbArray[ 1] + idx]  = (s * isobs)  + (ONEMINUSOMEGA* n + DFL2_OMEGA * rho * (1.0 + uy * (4.5 * uy + 3.0) - u2))* (!isobs);
//	//printf("\n(%d, %d, %d) N %d", x,y,z, c_nbArray[ 1] + idx);
//	//printf("\n(1. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 1] + idx, (dst.f[N])[c_nbArray[ 1] + idx]);
//	dfs[c_nbArray[ 2] + idx]  = (n * isobs)  + (ONEMINUSOMEGA* s + DFL2_OMEGA * rho * (1.0 + uy * (4.5 * uy - 3.0) - u2))* (!isobs);
//	//printf("\n(%d, %d, %d) S %d", x,y,z, c_nbArray[ 2] + idx);
//	//printf("\n(2. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 2] + idx, (dst.f[S])[c_nbArray[ 2] + idx]);
//
//	dft[c_nbArray[ 5] + idx]  = (b * isobs)	+ (ONEMINUSOMEGA* t + DFL2_OMEGA * rho * (1.0 + uz * (4.5 * uz + 3.0) - u2))* (!isobs);
//	//printf("\n(%d, %d, %d) T %d", x,y,z, c_nbArray[ 5] + idx);
//	//printf("\n(5. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 5] + idx, (dst.f[T])[c_nbArray[ 5] + idx]);
//	dfb[c_nbArray[ 6] + idx]  = (t * isobs)	+ (ONEMINUSOMEGA* b + DFL2_OMEGA * rho * (1.0 + uz * (4.5 * uz - 3.0) - u2))* (!isobs);
//	//printf("\n(%d, %d, %d) B %d", x,y,z, c_nbArray[ 6] + idx);
//	//printf("\n(6. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 6] + idx, (dst.f[B])[c_nbArray[ 6] + idx]);
//
//	dfnt[c_nbArray[11] + idx] = (sb * isobs)+ (ONEMINUSOMEGA* nt + DFL3_OMEGA * rho * (1.0 + (+uy + uz) * (4.5 * (+uy + uz) + 3.0) - u2))* (!isobs);
//	//printf("\n(%d, %d, %d) NT %d", x,y,z, c_nbArray[ 11] + idx);
//	//printf("\n(11. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 11] + idx, (dst.f[NT])[c_nbArray[11] + idx]);
//	dfnb[c_nbArray[12] + idx] = (st * isobs)+ (ONEMINUSOMEGA* nb + DFL3_OMEGA * rho * (1.0 + (+uy - uz) * (4.5 * (+uy - uz) + 3.0) - u2))* (!isobs);
//	//printf("\n(%d, %d, %d) NB %d", x,y,z, c_nbArray[ 12] + idx);
//	//printf("\n(12. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 12] + idx, (dst.f[NB])[c_nbArray[12] + idx]);
//	dfst[c_nbArray[13] + idx] = (nb * isobs)+ (ONEMINUSOMEGA* st + DFL3_OMEGA * rho * (1.0 + (-uy + uz) * (4.5 * (-uy + uz) + 3.0) - u2))* (!isobs);
//	//printf("\n(%d, %d, %d) ST %d", x,y,z, c_nbArray[ 13] + idx);
//	//printf("\n(13. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 13] + idx, (dst.f[ST])[c_nbArray[13] + idx]);
//	dfsb[c_nbArray[14] + idx] = (nt * isobs)+ (ONEMINUSOMEGA* sb + DFL3_OMEGA * rho * (1.0 + (-uy - uz) * (4.5 * (-uy - uz) + 3.0) - u2))* (!isobs);
//	//printf("\n(%d, %d, %d) SB %d", x,y,z, c_nbArray[ 14] + idx);
//	//printf("\n(14. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 14] + idx, (dst.f[SB])[c_nbArray[14] + idx]);
//
//
//	//int shiftE = 0;//((y-1)&0x1)^((z-1)&0x1);
//	int shiftW = 0;//0x1 & (~shiftE);
//	int txE = x;//+shiftE;
//	int txW = x;//-shiftW;
//
//	s_E[txE] = (w * isobs)	+ (ONEMINUSOMEGA* e + DFL2_OMEGA * rho * (1.0 + ux * (4.5 * ux + 3.0) - u2))* (!isobs);
//	//printf("\n(%d, %d, %d) s_E[%d] %.8f", x, y, z, txE, s_E[txE]);
//	s_SE[txE] = (nw * isobs) + (ONEMINUSOMEGA* se + DFL3_OMEGA * rho * (1.0 + (+ux - uy) * (4.5 * (+ux - uy) + 3.0) - u2))* (!isobs);
//	//printf("\n(%d, %d, %d) s_SE[%d] %.8f", x, y, z, txE, s_SE[txE]);
//	s_NE[txE] = (sw * isobs) + (ONEMINUSOMEGA* ne + DFL3_OMEGA * rho * (1.0 + (+ux + uy) * (4.5 * (+ux + uy) + 3.0) - u2))* (!isobs);
//	//printf("\n(%d, %d, %d) s_NE[%d] %.8f", x, y, z, txE, s_NE[txE]);
//	s_EB[txE] = (wt * isobs)+ (ONEMINUSOMEGA* eb + DFL3_OMEGA * rho * (1.0 + (+ux - uz) * (4.5 * (+ux - uz) + 3.0) - u2))* (!isobs);
//	//printf("\n(%d, %d, %d) s_EB[%d] %.8f", x, y, z, txE, s_EB[txE]);
//	s_ET[txE] = (wb * isobs)+ (ONEMINUSOMEGA* et + DFL3_OMEGA * rho * (1.0 + (+ux + uz) * (4.5 * (+ux + uz) + 3.0) - u2))* (!isobs);
//	//printf("\n(%d, %d, %d) s_ET[%d] %.8f", x, y, z, txE, s_ET[txE]);
//
//	s_W[txW+1] = (e * isobs)	+ (ONEMINUSOMEGA* w + DFL2_OMEGA * rho * (1.0 + ux * (4.5 * ux - 3.0) - u2))* (!isobs);
//	////printf("\n(%d, %d, %d) s_W[%d] %.8f", x, y, z, txW+1, s_W[txW+1]);
//	s_SW[txW+1] = (ne * isobs)+ (ONEMINUSOMEGA* sw + DFL3_OMEGA * rho * (1.0 + (-ux - uy) * (4.5 * (-ux - uy) + 3.0) - u2))* (!isobs);
//	//printf("\n(%d, %d, %d) s_SW[%d] %.8f", x, y, z, txW+1, s_SW[txW+1]);
//	s_NW[txW+1] = (se * isobs) + (ONEMINUSOMEGA* nw + DFL3_OMEGA * rho * (1.0 + (-ux + uy) * (4.5 * (-ux + uy) + 3.0) - u2))* (!isobs);
//	//printf("\n(%d, %d, %d) s_NW[%d] %.8f", x, y, z, txW+1, s_NW[txW+1]);
//	s_WB[txW+1] = (et * isobs)+ (ONEMINUSOMEGA* wb + DFL3_OMEGA * rho * (1.0 + (-ux - uz) * (4.5 * (-ux - uz) + 3.0) - u2))* (!isobs);
//	//printf("\n(%d, %d, %d) s_WB[%d] %.8f", x, y, z, txW+1, s_WB[txW+1]);
//	s_WT[txW+1] = (eb * isobs)+ (ONEMINUSOMEGA* wt + DFL3_OMEGA * rho * (1.0 + (-ux + uz) * (4.5 * (-ux + uz) + 3.0) - u2))* (!isobs);
//	//printf("\n(%d, %d, %d) s_WT[%d] %.8f", x, y, z, txW+1, s_WT[txW+1]);
//
//	__syncthreads();
//
//	dfe[c_nbArray[ 3] + idx]  = s_E[x];
//			//(w * isobs)	+ (ONEMINUSOMEGA* e + DFL2_OMEGA * rho * (1.0 + ux * (4.5 * ux + 3.0) - u2))* (!isobs);
//
//				//printf("\n(%d, %d, %d) E %d", x,y,z, c_nbArray[ 3] + idx);
//				//printf("\n(3. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 3] + idx, (dst.f[E])[c_nbArray[ 3] + idx]);
//	dfw[c_nbArray[ 4] + idx]  = s_W[x+1];
//			//(e * isobs)	+ (ONEMINUSOMEGA* w + DFL2_OMEGA * rho * (1.0 + ux * (4.5 * ux - 3.0) - u2))* (!isobs);
//
//				//printf("\n(%d, %d, %d) W %d", x,y,z, c_nbArray[ 4] + idx);
//				//printf("\n(4. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 4] + idx, (dst.f[W])[c_nbArray[ 4] + idx]);
//	dfne[c_nbArray[7] + idx]  = s_NE[x];
//	//printf("\n(%d, %d, %d) NE %d", x,y,z, c_nbArray[ 7] + idx);
//			//(sw * isobs) + (ONEMINUSOMEGA* ne + DFL3_OMEGA * rho * (1.0 + (+ux + uy) * (4.5 * (+ux + uy) + 3.0) - u2))* (!isobs);
//	//printf("\n(7. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 7] + idx, (dst.f[NE])[c_nbArray[7] + idx]);
//	dfnw[c_nbArray[8] + idx]  = s_NW[x+1];
//	//printf("\n(%d, %d, %d) NW %d", x,y,z, c_nbArray[ 8] + idx);
//			//(se * isobs) + (ONEMINUSOMEGA* nw + DFL3_OMEGA * rho * (1.0 + (-ux + uy) * (4.5 * (-ux + uy) + 3.0) - u2))* (!isobs);
//	//printf("\n(8. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 8] + idx, (dst.f[NW])[c_nbArray[8] + idx]);
//	dfse[c_nbArray[9] + idx]  =  s_SE[x];
//	//printf("\n(%d, %d, %d) SE %d", x,y,z, c_nbArray[ 9] + idx);
//			//(nw * isobs) + (ONEMINUSOMEGA* se + DFL3_OMEGA * rho * (1.0 + (+ux - uy) * (4.5 * (+ux - uy) + 3.0) - u2))* (!isobs);
//	//printf("\n(9. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 9] + idx, (dst.f[SE])[c_nbArray[9] + idx]);
//	dfsw[c_nbArray[10] + idx] = s_SW[x+1];
//	//printf("\n(%d, %d, %d) SW %d", x,y,z, c_nbArray[ 10] + idx);
//			//(ne * isobs)+ (ONEMINUSOMEGA* sw + DFL3_OMEGA * rho * (1.0 + (-ux - uy) * (4.5 * (-ux - uy) + 3.0) - u2))* (!isobs);
//	//printf("\n(10. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 10] + idx, (dst.f[SW])[c_nbArray[10] + idx]);
//
//	dfet[c_nbArray[15] + idx] = s_ET[x];
//	//printf("\n(%d, %d, %d) ET %d", x,y,z, c_nbArray[ 15] + idx);
//			//(wb * isobs)+ (ONEMINUSOMEGA* et + DFL3_OMEGA * rho * (1.0 + (+ux + uz) * (4.5 * (+ux + uz) + 3.0) - u2))* (!isobs);
//	//printf("\n(15. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 15] + idx, (dst.f[ET])[c_nbArray[15] + idx]);
//	dfeb[c_nbArray[16] + idx] = s_EB[x];
//	//printf("\n(%d, %d, %d) EB %d", x,y,z, c_nbArray[ 16] + idx);
//			//(wt * isobs)+ (ONEMINUSOMEGA* eb + DFL3_OMEGA * rho * (1.0 + (+ux - uz) * (4.5 * (+ux - uz) + 3.0) - u2))* (!isobs);
//	//printf("\n(16. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 16] + idx, (dst.f[EB])[c_nbArray[16] + idx]);
//	dfwt[c_nbArray[17] + idx] = s_WT[x+1];
//	//printf("\n(%d, %d, %d) WT %d", x,y,z, c_nbArray[ 17] + idx);
//			//(eb * isobs)+ (ONEMINUSOMEGA* wt + DFL3_OMEGA * rho * (1.0 + (-ux + uz) * (4.5 * (-ux + uz) + 3.0) - u2))* (!isobs);
//	//printf("\n(17. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 17] + idx, (dst.f[WT])[c_nbArray[17] + idx]);
//	dfwb[c_nbArray[18] + idx] =  s_WB[x+1];
//	//printf("\n(%d, %d, %d) WB %d", x,y,z, c_nbArray[ 18] + idx);
//			//(et * isobs)+ (ONEMINUSOMEGA* wb + DFL3_OMEGA * rho * (1.0 + (-ux - uz) * (4.5 * (-ux - uz) + 3.0) - u2))* (!isobs);
//	//printf("\n(18. (%d, %d, %d), %d, %d, %.8f",x,y,z, idx ,c_nbArray[ 18] + idx, (dst.f[WB])[c_nbArray[18] + idx]);
//}
__global__ void lbm_kernel_SoA_19_Arrays(float* __restrict__ sfc, float* __restrict__ sfn, float* __restrict__ sfs, float* __restrict__ sfe,float* __restrict__ sfw,float* __restrict__ sft,float* __restrict__ sfb,float* __restrict__ sfne,float* __restrict__ sfnw,float* __restrict__ sfse,float* __restrict__ sfsw,float* __restrict__ sfnt,float* __restrict__ sfnb,float* __restrict__ sfst,float* __restrict__ sfsb,float* __restrict__ sfet,float* __restrict__ sfeb,float* __restrict__ sfwt,float* __restrict__ sfwb,
											float *dfc, float *dfn, float *dfs, float *dfe,float *dfw,float *dft,float *dfb,float *dfne,float *dfnw,float *dfse,float *dfsw,float *dfnt,float *dfnb,float *dfst,float *dfsb,float *dfet,float *dfeb,float *dfwt,float *dfwb,
											unsigned char* __restrict__ flags)

{

//	int x = threadIdx.x;
//	int y = blockIdx.x + 1;
//	int z = blockIdx.y + 1;

	//int ElementsPerBlock = blockDim.x;
	//float c, n, s, e, w, t, b, ne, nw, se, sw, nt, nb, st, sb, et,	eb, wt, wb;

	/*the grid is organized as follows:
	 * c c c c c c . . . | n n n n n n . . . | s s s s s s . . . | e e e e e e . . . | . . .
	 */

	//calculate the index
	//int idx = (z * SIZE_YY + y)*SIZE_XX + x;
	//int idx = ((blockIdx.y +1) * SIZE_YY + (blockIdx.x+1))*SIZE_XX + threadIdx.x;
	int idx = CALC_INDEX_SOA_WO_STRUCT(threadIdx.x, (blockIdx.x + 1), (blockIdx.y + 1),0);

	float c = sfc[idx];

	float n = sfn[idx];

	float s = sfs[idx];

	float e = sfe[idx];

	float w = sfw[idx];

	float t = sft[idx];

	float b = sfb[idx];

	float ne = sfne[idx];

	float nw = sfnw[idx];

	float se = sfse[idx];

	float sw = sfsw[idx];

	float nt = sfnt[idx];

	float nb = sfnb[idx];

	float st = sfst[idx];

	float sb = sfsb[idx];

	float et = sfet[idx];

	float eb = sfeb[idx];

	float wt = sfwt[idx];

	float wb = sfwb[idx];

//	//get the value final cell (20) to know the kind of cell: fluid or accel or obstacle
//	//int flag = flags[idx];
	unsigned short type = ((flags[idx] == 1) << 8) | ((flags[idx] == 2) & 0xff);


	float rho = +c + n + s + e + w + t + b + ne + nw + se + sw + nt + nb + st + sb + et + eb + wt + wb;

	float ux = ((+e - w + ne - nw + se - sw + et + eb - wt - wb)/rho)*(!(type & 0xff)) + 0.005*(type & 0xff);
	float uy = ((+n - s + ne + nw - se - sw + nt + nb - st - sb)/rho)*(!(type & 0xff)) + 0.002*(type & 0xff);
	float uz = ((+t - b + nt - nb + st - sb + et - eb + wt - wb)/rho)*(!(type & 0xff));

	float u2 = 1.5 *( ux * ux +  uy * uy + uz * uz);


	dfc[c_nb0 + idx] = (c * (type >> 8))	+ ((1.0 - OMEGA)* c + DFL1 * OMEGA * rho * (1.0 - u2))* (!(type >> 8));

	c = (type >> 8); //resue variable c

	dfn[c_nb1 + idx] = (s * c)  + ((1.0 - OMEGA)* n + DFL2 * OMEGA * rho * (1.0 + uy * (4.5 * uy + 3.0) - u2))* (!c);

	dfs[c_nb2 + idx] = (n * c) + ((1.0 - OMEGA)* s + DFL2 * OMEGA * rho * (1.0 + uy * (4.5 * uy - 3.0) - u2))* (!c);

	n = (1.0 - OMEGA); //resue variable n
	s = DFL2 * OMEGA * rho; //resue variable s

	dfe-=1;
	dfe[c_nb3 + idx] = (w * c) + (n* e + s * (1.0 + ux * (4.5 * ux + 3.0) - u2))* (!c);
	dfe+=1;

	dfw+=1;
	dfw[c_nb4 + idx] = (e * c) + (n* w + s * (1.0 + ux * (4.5 * ux - 3.0) - u2))* (!c);
	dfw-=1;

	dft[c_nb5 + idx] = (b * c) + (n* t + s * (1.0 + uz * (4.5 * uz + 3.0) - u2))* (!c);

	dfb[c_nb6 + idx] = (t * c) + (n* b + s * (1.0 + uz * (4.5 * uz - 3.0) - u2))* (!c);

	b = DFL3 * OMEGA * rho; //resue variable b
	t = 1.0 - u2;
	e = ux + uy;
	w = b*(t+ 4.5*e*e);

	dfne-=1;
	dfne[c_nb7 + idx] = (sw * c) + (n* ne + w + 3*e*b)* (!c);
	dfne+=1;

	dfsw+=1;
	dfsw[c_nb10 + idx] = (ne * c)+ (n* sw + w - 3*e*b)* (!c);
	dfsw-=1;

	e = -ux + uy;
	w = b*(t+ 4.5*e*e);

	dfnw+=1;
	dfnw[c_nb8 + idx] = (se * c) + (n* nw + w + 3*e*b)* (!c);
	dfnw-=1;

	dfse-=1;
	dfse[c_nb9 + idx] = (nw * c) + (n* se + w - 3*e*b)* (!c);
	dfse+=1;

	e = uy + uz;
	w = b*(t+ 4.5*e*e);

	dfnt[c_nb11 + idx] = (sb * c)+ (n* nt + w + 3*e*b)* (!c);

	dfsb[c_nb14 + idx] = (nt * c)+ (n* sb + w - 3*e*b)* (!c);

	e = uy - uz;
	w = b*(t+ 4.5*e*e);

	dfnb[c_nb12 + idx] = (st * c)+ (n* nb + w + 3*e*b)* (!c);

	dfst[c_nb13 + idx] = (nb * c)+ (n* st + w - 3*e*b)* (!c);

	e = ux + uz;
	w = b*(t+ 4.5*e*e);

	dfet-=1;
	dfet[c_nb15 + idx] = (wb * c)+ (n* et + w + 3*e*b)* (!c);
	dfet+=1;

	dfwb+=1;
	dfwb[c_nb18 + idx] = (et * c)+ (n* wb + w - 3*e*b)* (!c);
	dfwb-=1;

	e = ux - uz;
	w = b*(t+ 4.5*e*e);

	dfeb-=1;
	dfeb[c_nb16 + idx] = (wt * c)+ (n* eb + w + 3*e*b)* (!c);
	dfeb+=1;

	dfwt+=1;
	dfwt[c_nb17 + idx] = (eb * c)+ (n* wt + w - 3*e*b)* (!c);
	dfwt-=1;
}
__global__ void lbm_kernel_SoA_dp(float *sGrid, float *dGrid, int *flags)
{
	//float *shPropPointer = (float*)array;
	__shared__ int offset;
	//__shared__ float one_minus_omega;
	offset = SIZE_XX*SIZE_YY*SIZE_ZZ;
	//one_minus_omega = 1.0 - OMEGA;
	int x = threadIdx.x;
	int y = blockIdx.x + 1;
	int z = blockIdx.y + 1;

	//int ElementsPerBlock = blockDim.x;
	//float c, n, s, e, w, t, b, ne, nw, se, sw, nt, nb, st, sb, et,	eb, wt, wb;

	/*the grid is organized as follows:
	 * c c c c c c . . . | n n n n n n . . . | s s s s s s . . . | e e e e e e . . . | . . .
	 */

	//calculate the index
	//int idx = (z * SIZE_YY + y)*SIZE_XX + x;
	//int idx = ((blockIdx.y +1) * SIZE_YY + (blockIdx.x+1))*SIZE_XX + threadIdx.x;
	int idx = CALC_INDEX_SOA_WO_STRUCT(x,y,z,0);

	//each direction has offset = 130*100*100 values, we need to jump one offset to get new value
	//int offset = SIZE_XX*SIZE_YY*SIZE_ZZ;
	//save index to address for operating
	int address = idx;
	float c = sGrid[address];
	address = address + offset;
	float n = sGrid[address];
	address = address + offset;
	float s = sGrid[address];
	address = address + offset;
	float e = sGrid[address];
	address = address + offset;
	float w = sGrid[address];
	address = address + offset;
	float t = sGrid[address];
	address = address + offset;
	float b = sGrid[address];
	address = address + offset;
	float ne = sGrid[address];
	address = address + offset;
	float nw = sGrid[address];
	address = address + offset;
	float se = sGrid[address];
	address = address + offset;
	float sw = sGrid[address];
	address = address + offset;
	float nt = sGrid[address];
	address = address + offset;
	float nb = sGrid[address];
	address = address + offset;
	float st = sGrid[address];
	address = address + offset;
	float sb = sGrid[address];
	address = address + offset;
	float et = sGrid[address];
	address = address + offset;
	float eb = sGrid[address];
	address = address + offset;
	float wt = sGrid[address];
	address = address + offset;
	float wb = sGrid[address];

	//get the value final cell (20) to know the kind of cell: fluid or accel or obstacle
	//int flag = flags[idx];

	int isobs = (flags[idx] == 1);
	int isacc = (flags[idx] == 2);

	float rho = +c + n + s + e + w + t + b + ne + nw + se + sw + nt + nb + st
			+ sb + et + eb + wt + wb;

	float ux = ((+e - w + ne - nw + se - sw + et + eb - wt - wb)/rho)*(!isacc) + 0.005*isacc;
	float uy = ((+n - s + ne + nw - se - sw + nt + nb - st - sb)/rho)*(!isacc) + 0.002*isacc;
	float uz = ((+t - b + nt - nb + st - sb + et - eb + wt - wb)/rho)*(!isacc);

//	float ux = +e - w + ne - nw + se - sw + et + eb - wt - wb;
//	float uy = +n - s + ne + nw - se - sw + nt + nb - st - sb;
//	float uz = +t - b + nt - nb + st - sb + et - eb + wt - wb;
//
//	ux = (ux / rho) * (!isacc) + 0.005 * isacc;
//	uy = (uy / rho) * (!isacc) + 0.002 * isacc;
//	uz = (uz / rho) * (!isacc) + 0.000 * isacc;

	float u2 = 1.5 * (ux * ux + uy * uy + uz * uz); //U2(ux,uy,uz);//

	//address = c_nbArray[ 0] + idx;
	dGrid[c_nbArray[ 0] + idx] =  		  	(c * isobs) + (ONEMINUSOMEGA* c + DFL1_OMEGA * rho * (1.0 - u2))* (!isobs);
	//address = c_nbArray[ 1] + idx + offset;
	dGrid[c_nbArray[ 1] + idx + offset] =   (s * isobs)  + (ONEMINUSOMEGA* n + DFL2_OMEGA * rho * (1.0 + uy * (4.5 * uy + 3.0) - u2))* (!isobs);
	//address = c_nbArray[ 2] + idx + 2*offset;
	dGrid[c_nbArray[ 2] + idx + 2*offset] = (n * isobs)  + (ONEMINUSOMEGA* s + DFL2_OMEGA * rho * (1.0 + uy * (4.5 * uy - 3.0) - u2))* (!isobs);
	//address = c_nbArray[ 3] + idx + 3*offset;
	dGrid[c_nbArray[ 3] + idx + 3*offset] = (w * isobs)	+ (ONEMINUSOMEGA* e + DFL2_OMEGA * rho * (1.0 + ux * (4.5 * ux + 3.0) - u2))* (!isobs);
	//address = c_nbArray[ 4] + idx + 4*offset;
	dGrid[c_nbArray[ 4] + idx + 4*offset] = (e * isobs)	+ (ONEMINUSOMEGA* w + DFL2_OMEGA * rho * (1.0 + ux * (4.5 * ux - 3.0) - u2))* (!isobs);
	//address = c_nbArray[ 5] + idx + 5*offset;
	dGrid[c_nbArray[ 5] + idx + 5*offset] = (b * isobs)	+ (ONEMINUSOMEGA* t + DFL2_OMEGA * rho * (1.0 + uz * (4.5 * uz + 3.0) - u2))* (!isobs);
	//address = c_nbArray[ 6] + idx + 6*offset;
	dGrid[c_nbArray[ 6] + idx + 6*offset] = (t * isobs)	+ (ONEMINUSOMEGA* b + DFL2_OMEGA * rho * (1.0 + uz * (4.5 * uz - 3.0) - u2))* (!isobs);
	//address = c_nbArray[ 7] + idx + 7*offset;
	dGrid[c_nbArray[ 7] + idx + 7*offset] = (sw * isobs) + (ONEMINUSOMEGA* ne + DFL3_OMEGA * rho * (1.0 + (+ux + uy) * (4.5 * (+ux + uy) + 3.0) - u2))* (!isobs);
	//address = c_nbArray[ 8] + idx + 8*offset;
	dGrid[c_nbArray[ 8] + idx + 8*offset] = (se * isobs) + (ONEMINUSOMEGA* nw + DFL3_OMEGA * rho * (1.0 + (-ux + uy) * (4.5 * (-ux + uy) + 3.0) - u2))* (!isobs);
	//address = c_nbArray[ 9] + idx + 9*offset;
	dGrid[c_nbArray[ 9] + idx + 9*offset] = (nw * isobs) + (ONEMINUSOMEGA* se + DFL3_OMEGA * rho * (1.0 + (+ux - uy) * (4.5 * (+ux - uy) + 3.0) - u2))* (!isobs);
	//address = c_nbArray[10] + idx + 10*offset;
	dGrid[c_nbArray[10] + idx + 10*offset] = (ne * isobs)+ (ONEMINUSOMEGA* sw + DFL3_OMEGA * rho * (1.0 + (-ux - uy) * (4.5 * (-ux - uy) + 3.0) - u2))* (!isobs);
	//address = c_nbArray[11] + idx + 11*offset;
	dGrid[c_nbArray[11] + idx + 11*offset] = (sb * isobs)+ (ONEMINUSOMEGA* nt + DFL3_OMEGA * rho * (1.0 + (+uy + uz) * (4.5 * (+uy + uz) + 3.0) - u2))* (!isobs);
	//address = c_nbArray[12] + idx + 12*offset;
	dGrid[c_nbArray[12] + idx + 12*offset] = (st * isobs)+ (ONEMINUSOMEGA* nb + DFL3_OMEGA * rho * (1.0 + (+uy - uz) * (4.5 * (+uy - uz) + 3.0) - u2))* (!isobs);
	//address = c_nbArray[13] + idx + 13*offset;
	dGrid[c_nbArray[13] + idx + 13*offset] = (nb * isobs)+ (ONEMINUSOMEGA* st + DFL3_OMEGA * rho * (1.0 + (-uy + uz) * (4.5 * (-uy + uz) + 3.0) - u2))* (!isobs);
	//address = c_nbArray[14] + idx + 14*offset;
	dGrid[c_nbArray[14] + idx + 14*offset] = (nt * isobs)+ (ONEMINUSOMEGA* sb + DFL3_OMEGA * rho * (1.0 + (-uy - uz) * (4.5 * (-uy - uz) + 3.0) - u2))* (!isobs);
	//address = c_nbArray[15] + idx + 15*offset;
	dGrid[c_nbArray[15] + idx + 15*offset] = (wb * isobs)+ (ONEMINUSOMEGA* et + DFL3_OMEGA * rho * (1.0 + (+ux + uz) * (4.5 * (+ux + uz) + 3.0) - u2))* (!isobs);
	//address = c_nbArray[16] + idx + 16*offset;
	dGrid[c_nbArray[16] + idx + 16*offset] = (wt * isobs)+ (ONEMINUSOMEGA* eb + DFL3_OMEGA * rho * (1.0 + (+ux - uz) * (4.5 * (+ux - uz) + 3.0) - u2))* (!isobs);
	//address = c_nbArray[17] + idx + 17*offset;
	dGrid[c_nbArray[17] + idx + 17*offset] = (eb * isobs)+ (ONEMINUSOMEGA* wt + DFL3_OMEGA * rho * (1.0 + (-ux + uz) * (4.5 * (-ux + uz) + 3.0) - u2))* (!isobs);
	//address = c_nbArray[18] + idx + 18*offset;
	dGrid[c_nbArray[18] + idx + 18*offset] = (et * isobs)+ (ONEMINUSOMEGA* wb + DFL3_OMEGA * rho * (1.0 + (-ux - uz) * (4.5 * (-ux - uz) + 3.0) - u2))* (!isobs);

}
/* Only use SoA scheme
 * NO: branch divergence removal
 * NO: register use reduction
 * NO: Double precision Floating Point Reduction
*/
__global__ void SoA_Push_Only(float *sGrid, float *dGrid, unsigned char *flags)
{
	int offset;

	//__shared__ float one_minus_omega;
	offset = SIZE_XX*SIZE_YY*SIZE_ZZ;
	//one_minus_omega = 1.0 - OMEGA;
	int x = threadIdx.x;
	int y = blockIdx.x + 1;
	int z = blockIdx.y + 1;

	//int ElementsPerBlock = blockDim.x;
	//float c, n, s, e, w, t, b, ne, nw, se, sw, nt, nb, st, sb, et,	eb, wt, wb;

	/*the grid is organized as follows:
	 * c c c c c c . . . | n n n n n n . . . | s s s s s s . . . | e e e e e e . . . | . . .
	 */

	//calculate the index
	//int idx = (z * SIZE_YY + y)*SIZE_XX + x;
	//int idx = ((blockIdx.y +1) * SIZE_YY + (blockIdx.x+1))*SIZE_XX + threadIdx.x;
	int idx = CALC_INDEX_SOA_WO_STRUCT(x,y,z,0) + MARGIN_L_SIZE;

	//each direction has offset = 130*100*100 values, we need to jump one offset to get new value
	//int offset = SIZE_XX*SIZE_YY*SIZE_ZZ;
	//save index to address for operating

	int address0 = idx; //if(y==1 && x==0 && z==1) printf("\nc src(%d)", address);
	float c = sGrid[address0];
	int address1 = address0 + offset; //if(y==1 && x==0 && z==1) printf("\nn src(%d)", address);
	float n = sGrid[address1];
	int address2 = address1 + offset; //if(y==1 && x==0 && z==1) printf("\ns src(%d)", address);
	float s = sGrid[address2];
	int address3 = address2 + offset; //if(y==1 && x==0 && z==1) printf("\ne src(%d)", address);
	float e = sGrid[address3];
	int address4 = address3 + offset; //if(y==1 && x==0 && z==1) printf("\nw src(%d)", address);
	float w = sGrid[address4];
	int address5 = address4 + offset; //if(y==1 && x==0 && z==1) printf("\nt src(%d)", address);
	float t = sGrid[address5];
	int address6 = address5 + offset; //if(y==1 && x==0 && z==1) printf("\nb src(%d)", address);
	float b = sGrid[address6];
	int address7 = address6 + offset; //if(y==1 && x==0 && z==1) printf("\nne src(%d)", address);
	float ne = sGrid[address7];
	int address8 = address7 + offset; //if(y==1 && x==0 && z==1) printf("\nnw src(%d)", address);
	float nw = sGrid[address8];
	int address9 = address8 + offset; //if(y==1 && x==0 && z==1) printf("\nse src(%d)", address);
	float se = sGrid[address9];
	int address10 = address9 + offset; //if(y==1 && x==0 && z==1) printf("\nsw src(%d)", address);
	float sw = sGrid[address10];
	int address11 = address10 + offset; //if(y==1 && x==0 && z==1) printf("\nnt src(%d)", address);
	float nt = sGrid[address11];
	int address12 = address11 + offset; //if(y==1 && x==0 && z==1) printf("\nnb src(%d)", address);
	float nb = sGrid[address12];
	int address13 = address12 + offset; //if(y==1 && x==0 && z==1) printf("\nst src(%d)", address);
	float st = sGrid[address13];
	int address14 = address13 + offset; //if(y==1 && x==0 && z==1) printf("\nsb src(%d)", address);
	float sb = sGrid[address14];
	int address15 = address14 + offset; //if(y==1 && x==0 && z==1) printf("\net src(%d)", address);
	float et = sGrid[address15];
	int address16 = address15 + offset; //if(y==1 && x==0 && z==1) printf("\neb src(%d)", address);
	float eb = sGrid[address16];
	int address17 = address16 + offset; //if(y==1 && x==0 && z==1) printf("\nwt src(%d)", address);
	float wt = sGrid[address17];
	int address18 = address17 + offset; //if(y==1 && x==0 && z==1) printf("\nwb src(%d)", address);
	float wb = sGrid[address18];

	//get the value final cell (20) to know the kind of cell: fluid or accel or obstacle
	//int flag = flags[idx];
	unsigned short type = ((flags[idx-MARGIN_L_SIZE] == 1) << 8) | ((flags[idx-MARGIN_L_SIZE] == 2) & 0xff);

	if((type >> 8)==1)
	{	int idx0 = idx;
		dGrid[c_nbArray[ 0] + idx0] =  	c;
		int idx1 = idx + offset;
		dGrid[c_nbArray[ 1] + idx1] =   s;
		int idx2 = idx + 2*offset;
		dGrid[c_nbArray[ 2] + idx2] = n;
		int idx3 = idx + 3*offset;
		dGrid[c_nbArray[ 3] + idx3] = w;
		int idx4 = idx + 4*offset;
		dGrid[c_nbArray[ 4] + idx4] = e;
		int idx5 = idx + 5*offset;
		dGrid[c_nbArray[ 5] + idx5] = b;
		int idx6 = idx + 6*offset;
		dGrid[c_nbArray[ 6] + idx6] = t;
		int idx7 = idx + 7*offset;
		dGrid[c_nbArray[ 7] + idx7] = sw;
		int idx8 = idx + 8*offset;
		dGrid[c_nbArray[ 8] + idx8] = se;
		int idx9 = idx + 9*offset;
		dGrid[c_nbArray[ 9] + idx9] = nw;
		int idx10 = idx + 10*offset;
		dGrid[c_nbArray[10] + idx10] = ne;
		int idx11 = idx + 11*offset;
		dGrid[c_nbArray[11] + idx11] = sb;
		int idx12 = idx + 12*offset;
		dGrid[c_nbArray[12] + idx12] = st;
		int idx13 = idx + 13*offset;
		dGrid[c_nbArray[13] + idx13] = nb;
		int idx14 = idx + 14*offset;
		dGrid[c_nbArray[14] + idx14] = nt;
		int idx15 = idx + 15*offset;
		dGrid[c_nbArray[15] + idx15] = wb;
		int idx16 = idx + 16*offset;
		dGrid[c_nbArray[16] + idx16] = wt;
		int idx17 = idx + 17*offset;
		dGrid[c_nbArray[17] + idx17] = eb;
		int idx18 = idx + 18*offset;
		dGrid[c_nbArray[18] + idx18] = et;
	}
	else
	{
		float rho = +c + n + s + e + w + t + b + ne + nw + se + sw + nt + nb + st + sb + et + eb + wt + wb;

		float ux = ((+e - w + ne - nw + se - sw + et + eb - wt - wb)/rho)*(!(type & 0xff)) + 0.005*(type & 0xff);
		float uy = ((+n - s + ne + nw - se - sw + nt + nb - st - sb)/rho)*(!(type & 0xff)) + 0.002*(type & 0xff);
		float uz = ((+t - b + nt - nb + st - sb + et - eb + wt - wb)/rho)*(!(type & 0xff));

		//	float ux = +e - w + ne - nw + se - sw + et + eb - wt - wb;
		//	float uy = +n - s + ne + nw - se - sw + nt + nb - st - sb;
		//	float uz = +t - b + nt - nb + st - sb + et - eb + wt - wb;


		float u2 = 1.5 * ux * ux + 1.5* uy * uy + 1.5* uz * uz;
		int idx0 = idx;
		dGrid[c_nbArray[ 0] + idx0] =  		  	(ONEMINUSOMEGA* c + DFL1_OMEGA * rho * (1.0 - u2));
		int idx1 = idx + offset;
		dGrid[c_nbArray[ 1] + idx1] =   (ONEMINUSOMEGA* n + DFL2_OMEGA * rho * (1.0 + uy * (4.5 * uy + 3.0) - u2));
		int idx2 = idx + 2*offset;
		dGrid[c_nbArray[ 2] + idx2] = (ONEMINUSOMEGA* s + DFL2_OMEGA * rho * (1.0 + uy * (4.5 * uy - 3.0) - u2));
		int idx3 = idx + 3*offset;
		dGrid[c_nbArray[ 3] + idx3] = (ONEMINUSOMEGA* e + DFL2_OMEGA * rho * (1.0 + ux * (4.5 * ux + 3.0) - u2));
		int idx4 = idx + 4*offset;
		dGrid[c_nbArray[ 4] + idx4] = (ONEMINUSOMEGA* w + DFL2_OMEGA * rho * (1.0 + ux * (4.5 * ux - 3.0) - u2));
		int idx5 = idx + 5*offset;
		dGrid[c_nbArray[ 5] + idx5] = (ONEMINUSOMEGA* t + DFL2_OMEGA * rho * (1.0 + uz * (4.5 * uz + 3.0) - u2));
		int idx6 = idx + 6*offset;
		dGrid[c_nbArray[ 6] + idx6] = (ONEMINUSOMEGA* b + DFL2_OMEGA * rho * (1.0 + uz * (4.5 * uz - 3.0) - u2));
		int idx7 = idx + 7*offset;
		dGrid[c_nbArray[ 7] + idx7] = (ONEMINUSOMEGA* ne + DFL3_OMEGA * rho * (1.0 + (+ux + uy) * (4.5 * (+ux + uy) + 3.0) - u2));
		int idx8 = idx + 8*offset;
		dGrid[c_nbArray[ 8] + idx8] = (ONEMINUSOMEGA* nw + DFL3_OMEGA * rho * (1.0 + (-ux + uy) * (4.5 * (-ux + uy) + 3.0) - u2));
		int idx9 = idx + 9*offset;
		dGrid[c_nbArray[ 9] + idx9] = (ONEMINUSOMEGA* se + DFL3_OMEGA * rho * (1.0 + (+ux - uy) * (4.5 * (+ux - uy) + 3.0) - u2));
		int idx10 = idx + 10*offset;
		dGrid[c_nbArray[10] + idx10] = (ONEMINUSOMEGA* sw + DFL3_OMEGA * rho * (1.0 + (-ux - uy) * (4.5 * (-ux - uy) + 3.0) - u2));
		int idx11 = idx + 11*offset;
		dGrid[c_nbArray[11] + idx11] = (ONEMINUSOMEGA* nt + DFL3_OMEGA * rho * (1.0 + (+uy + uz) * (4.5 * (+uy + uz) + 3.0) - u2));
		int idx12 = idx + 12*offset;
		dGrid[c_nbArray[12] + idx12] = (ONEMINUSOMEGA* nb + DFL3_OMEGA * rho * (1.0 + (+uy - uz) * (4.5 * (+uy - uz) + 3.0) - u2));
		int idx13 = idx + 13*offset;
		dGrid[c_nbArray[13] + idx13] = (ONEMINUSOMEGA* st + DFL3_OMEGA * rho * (1.0 + (-uy + uz) * (4.5 * (-uy + uz) + 3.0) - u2));
		int idx14 = idx + 14*offset;
		dGrid[c_nbArray[14] + idx14] = (ONEMINUSOMEGA* sb + DFL3_OMEGA * rho * (1.0 + (-uy - uz) * (4.5 * (-uy - uz) + 3.0) - u2));
		int idx15 = idx + 15*offset;
		dGrid[c_nbArray[15] + idx15] = (ONEMINUSOMEGA* et + DFL3_OMEGA * rho * (1.0 + (+ux + uz) * (4.5 * (+ux + uz) + 3.0) - u2));
		int idx16 = idx + 16*offset;
		dGrid[c_nbArray[16] + idx16] = (ONEMINUSOMEGA* eb + DFL3_OMEGA * rho * (1.0 + (+ux - uz) * (4.5 * (+ux - uz) + 3.0) - u2));
		int idx17 = idx + 17*offset;
		dGrid[c_nbArray[17] + idx17] = (ONEMINUSOMEGA* wt + DFL3_OMEGA * rho * (1.0 + (-ux + uz) * (4.5 * (-ux + uz) + 3.0) - u2));
		int idx18 = idx + 18*offset;
		dGrid[c_nbArray[18] + idx18] = (ONEMINUSOMEGA* wb + DFL3_OMEGA * rho * (1.0 + (-ux - uz) * (4.5 * (-ux - uz) + 3.0) - u2));
	}
}
__global__ void SoA_Pull_Only(float* sGrid, float* dGrid, unsigned char* flags)
{
	int offset;

	//__shared__ float one_minus_omega;
	offset = SIZE_XX*SIZE_YY*SIZE_ZZ;
	//one_minus_omega = 1.0 - OMEGA;
	int x = threadIdx.x;
	int y = blockIdx.x;// + 1;
	int z = blockIdx.y;// + 1;

	//int ElementsPerBlock = blockDim.x;
	//float c, n, s, e, w, t, b, ne, nw, se, sw, nt, nb, st, sb, et,	eb, wt, wb;

	/*the grid is organized as follows:
	 * c c c c c c . . . | n n n n n n . . . | s s s s s s . . . | e e e e e e . . . | . . .
	 */

	//calculate the index
	//int idx = (z * SIZE_YY + y)*SIZE_XX + x;
	//int idx = ((blockIdx.y +1) * SIZE_YY + (blockIdx.x+1))*SIZE_XX + threadIdx.x;
	int idx = CALC_INDEX_SOA_WO_STRUCT(x, y, z, 0) + MARGIN_L_SIZE;

	//each direction has offset = 130*100*100 values, we need to jump one offset to get new value
	//int offset = SIZE_XX*SIZE_YY*SIZE_ZZ;
	//save index to address for operating
	//int address = offset;
	int address0 = -c_nbArray[ 0] + idx; //if(y==1 && x==0 && z==1) printf("\nc src(%d)", address);
	float c = dGrid[address0]; //if(y==1 && x==0 && z==1) printf("\nc = %.5f", c);
	int address1 = -c_nbArray[ 1] + idx + offset; ////if(y==1 && x==0 && z==1) printf("\nn src(%d)", address);
	float n = dGrid[address1];//if(y==1 && x==0 && z==1) printf("\nn = %.5f", n);
	int address2 = -c_nbArray[ 2] + idx + 2*offset; //if(y==1 && x==0 && z==1) printf("\ns src(%d)", address);
	float s = dGrid[address2]; //if(y==1 && x==0 && z==1) printf("\ns = %.5f", s);
	int address3 = -c_nbArray[ 3] + idx + 3*offset; //if(y==1 && x==0 && z==1) printf("\ne src(%d)", address);
	float e = dGrid[address3]; //if(y==1 && x==0 && z==1) printf("\ne = %.5f", e);
	int address4 = -c_nbArray[ 4] + idx + 4*offset; //if(y==1 && x==0 && z==1) printf("\nw src(%d)", address);
	float w = dGrid[address4]; //if(y==1 && x==0 && z==1) printf("\nw = %.5f", w);
	int address5 = -c_nbArray[ 5] + idx + 5*offset; //if(y==1 && x==0 && z==1) printf("\nt src(%d)", address);
	float t = dGrid[address5]; //if(y==1 && x==0 && z==1) printf("\nt = %.5f", t);
	int address6 = -c_nbArray[ 6] + idx + 6*offset; //if(y==1 && x==0 && z==1) printf("\nb src(%d)", address);
	float b = dGrid[address6]; //if(y==1 && x==0 && z==1) printf("\nb = %.5f", b);
	int address7 = -c_nbArray[ 7] + idx + 7*offset; //if(y==1 && x==0 && z==1) printf("\nne src(%d)", address);
	float ne = dGrid[address7]; //if(y==1 && x==0 && z==1) printf("\nne = %.5f", ne);
	int address8 = -c_nbArray[ 8] + idx + 8*offset; //if(y==1 && x==0 && z==1) printf("\nnw src(%d)", address);
	float nw = dGrid[address8]; //if(y==1 && x==0 && z==1) printf("\nnw = %.5f", nw);
	int address9 = -c_nbArray[ 9] + idx + 9*offset; //if(y==1 && x==0 && z==1) printf("\nse src(%d)", address);
	float se = dGrid[address9]; //if(y==1 && x==0 && z==1) printf("\nse = %.5f", se);
	int address10 = -c_nbArray[10] + idx + 10*offset; //if(y==1 && x==0 && z==1) printf("\nsw src(%d)", address);
	float sw = dGrid[address10]; //if(y==1 && x==0 && z==1) printf("\nsw = %.5f", sw);
	int address11 = -c_nbArray[11] + idx + 11*offset; //if(y==1 && x==0 && z==1) printf("\nnt src(%d)", address);
	float nt = dGrid[address11]; //if(y==1 && x==0 && z==1) printf("\nnt = %.5f", nt);
	int address12 = -c_nbArray[12] + idx + 12*offset; //if(y==1 && x==0 && z==1) printf("\nnb src(%d)", address);
	float nb = dGrid[address12]; //if(y==1 && x==0 && z==1) printf("\nnb = %.5f", nb);
	int address13 = -c_nbArray[13] + idx + 13*offset; //if(y==1 && x==0 && z==1) printf("\nst src(%d)", address);
	float st = dGrid[address13]; //if(y==1 && x==0 && z==1) printf("\nst = %.5f", st);
	int address14 = -c_nbArray[14] + idx + 14*offset; //if(y==1 && x==0 && z==1) printf("\nsb src(%d)", address);
	float sb = dGrid[address14]; //if(y==1 && x==0 && z==1) printf("\nsb = %.5f", sb);
	int address15 = -c_nbArray[15] + idx + 15*offset; //if(y==1 && x==0 && z==1) printf("\net src(%d)", address);
	float et = dGrid[address15]; //if(y==1 && x==0 && z==1) printf("\net = %.5f", et);
	int address16 = -c_nbArray[16] + idx + 16*offset; //if(y==1 && x==0 && z==1) printf("\neb src(%d)", address);
	float eb = dGrid[address16]; //if(y==1 && x==0 && z==1) printf("\neb = %.5f", eb);
	int address17 = -c_nbArray[17] + idx + 17*offset; //if(y==1 && x==0 && z==1) printf("\nwt src(%d)", address);
	float wt = dGrid[address17]; //if(y==1 && x==0 && z==1) printf("\nwt = %.5f", wt);
	int address18 = -c_nbArray[18] + idx + 18*offset; //if(y==1 && x==0 && z==1) printf("\nwb src(%d)", address);
	float wb = dGrid[address18]; //if(y==1 && x==0 && z==1) printf("\nwb = %.5f", wb);

	//get the value final cell (20) to know the kind of cell: fluid or accel or obstacle
	//int flag = flags[idx];
	unsigned short type = ((flags[idx-MARGIN_L_SIZE] == 1) << 8) | ((flags[idx-MARGIN_L_SIZE] == 2) & 0xff);
	//int isobs = (flags[idx] == 1);
	//int isacc = (flags[idx] == 2);

	if((type >> 8)==1)
	{	int idx0 = idx;
		sGrid[idx0] = c;
		int idx1 = idx + offset;
		sGrid[idx1] = s;
		int idx2 = idx + 2*offset;
		sGrid[idx2] = n;
		int idx3 = idx + 3*offset;
		sGrid[idx3] = w;
		int idx4 = idx + 4*offset;
		sGrid[idx4] = e;
		int idx5 = idx + 5*offset;
		sGrid[idx5] = b;
		int idx6 = idx + 6*offset;
		sGrid[idx6] = t;
		int idx7 = idx + 7*offset;
		sGrid[idx7] = sw;
		int idx8 = idx + 8*offset;
		sGrid[idx8] = se;
		int idx9 = idx + 9*offset;
		sGrid[idx9] = nw;
		int idx10 = idx + 10*offset;
		sGrid[idx10] = ne;
		int idx11 = idx + 11*offset;
		sGrid[idx11] = sb;
		int idx12 = idx + 12*offset;
		sGrid[idx12] = st;
		int idx13 = idx + 13*offset;
		sGrid[idx13] = nb;
		int idx14 = idx + 14*offset;
		sGrid[idx14] = nt;
		int idx15 = idx + 15*offset;
		sGrid[idx15] = wb;
		int idx16 = idx + 16*offset;
		sGrid[idx16] = wt;
		int idx17 = idx + 17*offset;
		sGrid[idx17] = eb;
		int idx18 = idx + 18*offset;
		sGrid[idx18] = et;

	}
	else
	{

		float rho = +c + n + s + e + w + t + b + ne + nw + se + sw + nt + nb + st + sb + et + eb + wt + wb;

		float ux = ((+e - w + ne - nw + se - sw + et + eb - wt - wb)/rho)*(!(type & 0xff)) + 0.005*(type & 0xff);
		float uy = ((+n - s + ne + nw - se - sw + nt + nb - st - sb)/rho)*(!(type & 0xff)) + 0.002*(type & 0xff);
		float uz = ((+t - b + nt - nb + st - sb + et - eb + wt - wb)/rho)*(!(type & 0xff));


		float u2 = 1.5 * (ux * ux + uy * uy + uz * uz);

		int idx0 = idx;
		sGrid[idx0] =  		  	(ONEMINUSOMEGA* c + DFL1_OMEGA * rho * (1.0 - u2));
		int idx1 = idx + offset;
		sGrid[idx1] =   (ONEMINUSOMEGA* n + DFL2_OMEGA * rho * (1.0 + uy * (4.5 * uy + 3.0) - u2));
		int idx2 = idx + 2*offset;
		sGrid[idx2] = (ONEMINUSOMEGA* s + DFL2_OMEGA * rho * (1.0 + uy * (4.5 * uy - 3.0) - u2));
		int idx3 = idx + 3*offset;
		sGrid[idx3] = (ONEMINUSOMEGA* e + DFL2_OMEGA * rho * (1.0 + ux * (4.5 * ux + 3.0) - u2));
		int idx4 = idx + 4*offset;
		sGrid[idx4] = (ONEMINUSOMEGA* w + DFL2_OMEGA * rho * (1.0 + ux * (4.5 * ux - 3.0) - u2));
		int idx5 = idx + 5*offset;
		sGrid[idx5] = (ONEMINUSOMEGA* t + DFL2_OMEGA * rho * (1.0 + uz * (4.5 * uz + 3.0) - u2));
		int idx6 = idx + 6*offset;
		sGrid[idx6] = (ONEMINUSOMEGA* b + DFL2_OMEGA * rho * (1.0 + uz * (4.5 * uz - 3.0) - u2));
		int idx7 = idx + 7*offset;
		sGrid[idx7] = (ONEMINUSOMEGA* ne + DFL3_OMEGA * rho * (1.0 + (+ux + uy) * (4.5 * (+ux + uy) + 3.0) - u2));
		int idx8 = idx + 8*offset;
		sGrid[idx8] = (ONEMINUSOMEGA* nw + DFL3_OMEGA * rho * (1.0 + (-ux + uy) * (4.5 * (-ux + uy) + 3.0) - u2));
		int idx9 = idx + 9*offset;
		sGrid[idx9] = (ONEMINUSOMEGA* se + DFL3_OMEGA * rho * (1.0 + (+ux - uy) * (4.5 * (+ux - uy) + 3.0) - u2));
		int idx10 = idx + 10*offset;
		sGrid[idx10] = (ONEMINUSOMEGA* sw + DFL3_OMEGA * rho * (1.0 + (-ux - uy) * (4.5 * (-ux - uy) + 3.0) - u2));
		int idx11 = idx + 11*offset;
		sGrid[idx11] = (ONEMINUSOMEGA* nt + DFL3_OMEGA * rho * (1.0 + (+uy + uz) * (4.5 * (+uy + uz) + 3.0) - u2));
		int idx12 = idx + 12*offset;
		sGrid[idx12] = (ONEMINUSOMEGA* nb + DFL3_OMEGA * rho * (1.0 + (+uy - uz) * (4.5 * (+uy - uz) + 3.0) - u2));
		int idx13 = idx + 13*offset;
		sGrid[idx13] = (ONEMINUSOMEGA* st + DFL3_OMEGA * rho * (1.0 + (-uy + uz) * (4.5 * (-uy + uz) + 3.0) - u2));
		int idx14 = idx + 14*offset;
		sGrid[idx14] = (ONEMINUSOMEGA* sb + DFL3_OMEGA * rho * (1.0 + (-uy - uz) * (4.5 * (-uy - uz) + 3.0) - u2));
		int idx15 = idx + 15*offset;
		sGrid[idx15] = (ONEMINUSOMEGA* et + DFL3_OMEGA * rho * (1.0 + (+ux + uz) * (4.5 * (+ux + uz) + 3.0) - u2));
		int idx16 = idx + 16*offset;
		sGrid[idx16] = (ONEMINUSOMEGA* eb + DFL3_OMEGA * rho * (1.0 + (+ux - uz) * (4.5 * (+ux - uz) + 3.0) - u2));
		int idx17 = idx + 17*offset;
		sGrid[idx17] = (ONEMINUSOMEGA* wt + DFL3_OMEGA * rho * (1.0 + (-ux + uz) * (4.5 * (-ux + uz) + 3.0) - u2));
		int idx18 = idx + 18*offset;
		sGrid[idx18] = (ONEMINUSOMEGA* wb + DFL3_OMEGA * rho * (1.0 + (-ux - uz) * (4.5 * (-ux - uz) + 3.0) - u2));

	}
}
__global__ void SoA_Pull_Branch_Removal(float *sGrid, float *dGrid, unsigned char *flags)
{
	__shared__ int offset;

		//__shared__ float one_minus_omega;
		offset = SIZE_XX*SIZE_YY*SIZE_ZZ;
		//one_minus_omega = 1.0 - OMEGA;
		int x = threadIdx.x;
		int y = blockIdx.x;// + 1;
		int z = blockIdx.y;// + 1;

		//int ElementsPerBlock = blockDim.x;
		//float c, n, s, e, w, t, b, ne, nw, se, sw, nt, nb, st, sb, et,	eb, wt, wb;

		/*the grid is organized as follows:
		 * c c c c c c . . . | n n n n n n . . . | s s s s s s . . . | e e e e e e . . . | . . .
		 */

		//calculate the index
		//int idx = (z * SIZE_YY + y)*SIZE_XX + x;
		//int idx = ((blockIdx.y +1) * SIZE_YY + (blockIdx.x+1))*SIZE_XX + threadIdx.x;
		int idx = CALC_INDEX_SOA_WO_STRUCT(x, y, z, 0) + MARGIN_L_SIZE;

		//each direction has offset = 130*100*100 values, we need to jump one offset to get new value
		//int offset = SIZE_XX*SIZE_YY*SIZE_ZZ;
		//save index to address for operating
		//int address = offset;
		//int address = -c_nbArray[ 0] + idx; //if(y==1 && x==0 && z==1) printf("\nc src(%d)", address);
		int address0 = -c_nbArray[ 0] + idx; //if(y==1 && x==0 && z==1) printf("\nc src(%d)", address);
		float c = dGrid[address0]; //if(y==1 && x==0 && z==1) printf("\nc = %.5f", c);
		int address1 = -c_nbArray[ 1] + idx + offset; ////if(y==1 && x==0 && z==1) printf("\nn src(%d)", address);
		float n = dGrid[address1];//if(y==1 && x==0 && z==1) printf("\nn = %.5f", n);
		int address2 = -c_nbArray[ 2] + idx + 2*offset; //if(y==1 && x==0 && z==1) printf("\ns src(%d)", address);
		float s = dGrid[address2]; //if(y==1 && x==0 && z==1) printf("\ns = %.5f", s);
		int address3 = -c_nbArray[ 3] + idx + 3*offset; //if(y==1 && x==0 && z==1) printf("\ne src(%d)", address);
		float e = dGrid[address3]; //if(y==1 && x==0 && z==1) printf("\ne = %.5f", e);
		int address4 = -c_nbArray[ 4] + idx + 4*offset; //if(y==1 && x==0 && z==1) printf("\nw src(%d)", address);
		float w = dGrid[address4]; //if(y==1 && x==0 && z==1) printf("\nw = %.5f", w);
		int address5 = -c_nbArray[ 5] + idx + 5*offset; //if(y==1 && x==0 && z==1) printf("\nt src(%d)", address);
		float t = dGrid[address5]; //if(y==1 && x==0 && z==1) printf("\nt = %.5f", t);
		int address6 = -c_nbArray[ 6] + idx + 6*offset; //if(y==1 && x==0 && z==1) printf("\nb src(%d)", address);
		float b = dGrid[address6]; //if(y==1 && x==0 && z==1) printf("\nb = %.5f", b);
		int address7 = -c_nbArray[ 7] + idx + 7*offset; //if(y==1 && x==0 && z==1) printf("\nne src(%d)", address);
		float ne = dGrid[address7]; //if(y==1 && x==0 && z==1) printf("\nne = %.5f", ne);
		int address8 = -c_nbArray[ 8] + idx + 8*offset; //if(y==1 && x==0 && z==1) printf("\nnw src(%d)", address);
		float nw = dGrid[address8]; //if(y==1 && x==0 && z==1) printf("\nnw = %.5f", nw);
		int address9 = -c_nbArray[ 9] + idx + 9*offset; //if(y==1 && x==0 && z==1) printf("\nse src(%d)", address);
		float se = dGrid[address9]; //if(y==1 && x==0 && z==1) printf("\nse = %.5f", se);
		int address10 = -c_nbArray[10] + idx + 10*offset; //if(y==1 && x==0 && z==1) printf("\nsw src(%d)", address);
		float sw = dGrid[address10]; //if(y==1 && x==0 && z==1) printf("\nsw = %.5f", sw);
		int address11 = -c_nbArray[11] + idx + 11*offset; //if(y==1 && x==0 && z==1) printf("\nnt src(%d)", address);
		float nt = dGrid[address11]; //if(y==1 && x==0 && z==1) printf("\nnt = %.5f", nt);
		int address12 = -c_nbArray[12] + idx + 12*offset; //if(y==1 && x==0 && z==1) printf("\nnb src(%d)", address);
		float nb = dGrid[address12]; //if(y==1 && x==0 && z==1) printf("\nnb = %.5f", nb);
		int address13 = -c_nbArray[13] + idx + 13*offset; //if(y==1 && x==0 && z==1) printf("\nst src(%d)", address);
		float st = dGrid[address13]; //if(y==1 && x==0 && z==1) printf("\nst = %.5f", st);
		int address14 = -c_nbArray[14] + idx + 14*offset; //if(y==1 && x==0 && z==1) printf("\nsb src(%d)", address);
		float sb = dGrid[address14]; //if(y==1 && x==0 && z==1) printf("\nsb = %.5f", sb);
		int address15 = -c_nbArray[15] + idx + 15*offset; //if(y==1 && x==0 && z==1) printf("\net src(%d)", address);
		float et = dGrid[address15]; //if(y==1 && x==0 && z==1) printf("\net = %.5f", et);
		int address16 = -c_nbArray[16] + idx + 16*offset; //if(y==1 && x==0 && z==1) printf("\neb src(%d)", address);
		float eb = dGrid[address16]; //if(y==1 && x==0 && z==1) printf("\neb = %.5f", eb);
		int address17 = -c_nbArray[17] + idx + 17*offset; //if(y==1 && x==0 && z==1) printf("\nwt src(%d)", address);
		float wt = dGrid[address17]; //if(y==1 && x==0 && z==1) printf("\nwt = %.5f", wt);
		int address18 = -c_nbArray[18] + idx + 18*offset; //if(y==1 && x==0 && z==1) printf("\nwb src(%d)", address);
		float wb = dGrid[address18]; //if(y==1 && x==0 && z==1) printf("\nwb = %.5f", wb);

		//get the value final cell (20) to know the kind of cell: fluid or accel or obstacle
		//int flag = flags[idx];
		unsigned short type = ((flags[idx-MARGIN_L_SIZE] == 1) << 8) | ((flags[idx-MARGIN_L_SIZE] == 2) & 0xff);
		//int isobs = (flags[idx] == 1);
		//int isacc = (flags[idx] == 2);

		float rho = +c + n + s + e + w + t + b + ne + nw + se + sw + nt + nb + st + sb + et + eb + wt + wb;

		float ux = ((+e - w + ne - nw + se - sw + et + eb - wt - wb)/rho)*(!(type & 0xff)) + 0.005*(type & 0xff);
		float uy = ((+n - s + ne + nw - se - sw + nt + nb - st - sb)/rho)*(!(type & 0xff)) + 0.002*(type & 0xff);
		float uz = ((+t - b + nt - nb + st - sb + et - eb + wt - wb)/rho)*(!(type & 0xff));


		float u2 = 1.5 * (ux * ux + uy * uy + uz * uz);

		int idx0 = idx;
		sGrid[idx0] =  		  	(c * (type >> 8)) + (ONEMINUSOMEGA* c + DFL1_OMEGA * rho * (1.0 - u2))*(!(type >> 8));
		int idx1 = idx + offset;
		sGrid[idx1] =   (s * (type >> 8)) + (ONEMINUSOMEGA* n + DFL2_OMEGA * rho * (1.0 + uy * (4.5 * uy + 3.0) - u2))*(!(type >> 8));
		int idx2 = idx + 2*offset;
		sGrid[idx2] = (n * (type >> 8)) + (ONEMINUSOMEGA* s + DFL2_OMEGA * rho * (1.0 + uy * (4.5 * uy - 3.0) - u2))*(!(type >> 8));
		int idx3 = idx + 3*offset;
		sGrid[idx3] = (w * (type >> 8)) + (ONEMINUSOMEGA* e + DFL2_OMEGA * rho * (1.0 + ux * (4.5 * ux + 3.0) - u2))*(!(type >> 8));
		int idx4 = idx + 4*offset;
		sGrid[idx4] = (e * (type >> 8)) + (ONEMINUSOMEGA* w + DFL2_OMEGA * rho * (1.0 + ux * (4.5 * ux - 3.0) - u2))*(!(type >> 8));
		int idx5 = idx + 5*offset;
		sGrid[idx5] = (b * (type >> 8)) + (ONEMINUSOMEGA* t + DFL2_OMEGA * rho * (1.0 + uz * (4.5 * uz + 3.0) - u2))*(!(type >> 8));
		int idx6 = idx + 6*offset;
		sGrid[idx6] = (t * (type >> 8)) + (ONEMINUSOMEGA* b + DFL2_OMEGA * rho * (1.0 + uz * (4.5 * uz - 3.0) - u2))*(!(type >> 8));
		int idx7 = idx + 7*offset;
		sGrid[idx7] = (sw * (type >> 8)) + (ONEMINUSOMEGA* ne + DFL3_OMEGA * rho * (1.0 + (+ux + uy) * (4.5 * (+ux + uy) + 3.0) - u2))*(!(type >> 8));
		int idx8 = idx + 8*offset;
		sGrid[idx8] = (se * (type >> 8)) + (ONEMINUSOMEGA* nw + DFL3_OMEGA * rho * (1.0 + (-ux + uy) * (4.5 * (-ux + uy) + 3.0) - u2))*(!(type >> 8));
		int idx9 = idx + 9*offset;
		sGrid[idx9] = (nw * (type >> 8)) + (ONEMINUSOMEGA* se + DFL3_OMEGA * rho * (1.0 + (+ux - uy) * (4.5 * (+ux - uy) + 3.0) - u2))*(!(type >> 8));
		int idx10 = idx + 10*offset;
		sGrid[idx10] = (ne * (type >> 8)) + (ONEMINUSOMEGA* sw + DFL3_OMEGA * rho * (1.0 + (-ux - uy) * (4.5 * (-ux - uy) + 3.0) - u2))*(!(type >> 8));
		int idx11 = idx + 11*offset;
		sGrid[idx11] = (sb * (type >> 8)) + (ONEMINUSOMEGA* nt + DFL3_OMEGA * rho * (1.0 + (+uy + uz) * (4.5 * (+uy + uz) + 3.0) - u2))*(!(type >> 8));
		int idx12 = idx + 12*offset;
		sGrid[idx12] = (st * (type >> 8)) + (ONEMINUSOMEGA* nb + DFL3_OMEGA * rho * (1.0 + (+uy - uz) * (4.5 * (+uy - uz) + 3.0) - u2))*(!(type >> 8));
		int idx13 = idx + 13*offset;
		sGrid[idx13] = (nb * (type >> 8)) + (ONEMINUSOMEGA* st + DFL3_OMEGA * rho * (1.0 + (-uy + uz) * (4.5 * (-uy + uz) + 3.0) - u2))*(!(type >> 8));
		int idx14 = idx + 14*offset;
		sGrid[idx14] = (nt * (type >> 8)) + (ONEMINUSOMEGA* sb + DFL3_OMEGA * rho * (1.0 + (-uy - uz) * (4.5 * (-uy - uz) + 3.0) - u2))*(!(type >> 8));
		int idx15 = idx + 15*offset;
		sGrid[idx15] = (wb * (type >> 8)) + (ONEMINUSOMEGA* et + DFL3_OMEGA * rho * (1.0 + (+ux + uz) * (4.5 * (+ux + uz) + 3.0) - u2))*(!(type >> 8));
		int idx16 = idx + 16*offset;
		sGrid[idx16] = (wt * (type >> 8)) + (ONEMINUSOMEGA* eb + DFL3_OMEGA * rho * (1.0 + (+ux - uz) * (4.5 * (+ux - uz) + 3.0) - u2))*(!(type >> 8));
		int idx17 = idx + 17*offset;
		sGrid[idx17] = (eb * (type >> 8)) + (ONEMINUSOMEGA* wt + DFL3_OMEGA * rho * (1.0 + (-ux + uz) * (4.5 * (-ux + uz) + 3.0) - u2))*(!(type >> 8));
		int idx18 = idx + 18*offset;
		sGrid[idx18] = (et * (type >> 8)) + (ONEMINUSOMEGA* wb + DFL3_OMEGA * rho * (1.0 + (-ux - uz) * (4.5 * (-ux - uz) + 3.0) - u2))*(!(type >> 8));
}

__global__ void SoA_Pull_Register_Reduction(float *sGrid, float *dGrid, unsigned char *flags)
{
	__shared__ int offset;

	//__shared__ float one_minus_omega;
	offset = SIZE_XX*SIZE_YY*SIZE_ZZ;
	//one_minus_omega = 1.0 - OMEGA;
	int x = threadIdx.x;
	int y = blockIdx.x;// + 1;
	int z = blockIdx.y;// + 1;

	//int ElementsPerBlock = blockDim.x;
	//float c, n, s, e, w, t, b, ne, nw, se, sw, nt, nb, st, sb, et,	eb, wt, wb;

	/*the grid is organized as follows:
	 * c c c c c c . . . | n n n n n n . . . | s s s s s s . . . | e e e e e e . . . | . . .
	 */

	//calculate the index
	//int idx = (z * SIZE_YY + y)*SIZE_XX + x;
	//int idx = ((blockIdx.y +1) * SIZE_YY + (blockIdx.x+1))*SIZE_XX + threadIdx.x;
	int idx = CALC_INDEX_SOA_WO_STRUCT(x, y, z, 0) + MARGIN_L_SIZE;

	//each direction has offset = 130*100*100 values, we need to jump one offset to get new value
	//int offset = SIZE_XX*SIZE_YY*SIZE_ZZ;
	//save index to address for operating
	//int address = offset;
	int address = -c_nbArray[ 0] + idx; //if(y==1 && x==0 && z==1) printf("\nc src(%d)", address);
	float c = dGrid[address]; //if(y==1 && x==0 && z==1) printf("\nc = %.5f", c);
	address = -c_nbArray[ 1] + idx + offset; ////if(y==1 && x==0 && z==1) printf("\nn src(%d)", address);
	float n = dGrid[address];//if(y==1 && x==0 && z==1) printf("\nn = %.5f", n);
	address = -c_nbArray[ 2] + idx + 2*offset; //if(y==1 && x==0 && z==1) printf("\ns src(%d)", address);
	float s = dGrid[address]; //if(y==1 && x==0 && z==1) printf("\ns = %.5f", s);
	address = -c_nbArray[ 3] + idx + 3*offset; //if(y==1 && x==0 && z==1) printf("\ne src(%d)", address);
	float e = dGrid[address]; //if(y==1 && x==0 && z==1) printf("\ne = %.5f", e);
	address = -c_nbArray[ 4] + idx + 4*offset; //if(y==1 && x==0 && z==1) printf("\nw src(%d)", address);
	float w = dGrid[address]; //if(y==1 && x==0 && z==1) printf("\nw = %.5f", w);
	address = -c_nbArray[ 5] + idx + 5*offset; //if(y==1 && x==0 && z==1) printf("\nt src(%d)", address);
	float t = dGrid[address]; //if(y==1 && x==0 && z==1) printf("\nt = %.5f", t);
	address = -c_nbArray[ 6] + idx + 6*offset; //if(y==1 && x==0 && z==1) printf("\nb src(%d)", address);
	float b = dGrid[address]; //if(y==1 && x==0 && z==1) printf("\nb = %.5f", b);
	address = -c_nbArray[ 7] + idx + 7*offset; //if(y==1 && x==0 && z==1) printf("\nne src(%d)", address);
	float ne = dGrid[address]; //if(y==1 && x==0 && z==1) printf("\nne = %.5f", ne);
	address = -c_nbArray[ 8] + idx + 8*offset; //if(y==1 && x==0 && z==1) printf("\nnw src(%d)", address);
	float nw = dGrid[address]; //if(y==1 && x==0 && z==1) printf("\nnw = %.5f", nw);
	address = -c_nbArray[ 9] + idx + 9*offset; //if(y==1 && x==0 && z==1) printf("\nse src(%d)", address);
	float se = dGrid[address]; //if(y==1 && x==0 && z==1) printf("\nse = %.5f", se);
	address = -c_nbArray[10] + idx + 10*offset; //if(y==1 && x==0 && z==1) printf("\nsw src(%d)", address);
	float sw = dGrid[address]; //if(y==1 && x==0 && z==1) printf("\nsw = %.5f", sw);
	address = -c_nbArray[11] + idx + 11*offset; //if(y==1 && x==0 && z==1) printf("\nnt src(%d)", address);
	float nt = dGrid[address]; //if(y==1 && x==0 && z==1) printf("\nnt = %.5f", nt);
	address = -c_nbArray[12] + idx + 12*offset; //if(y==1 && x==0 && z==1) printf("\nnb src(%d)", address);
	float nb = dGrid[address]; //if(y==1 && x==0 && z==1) printf("\nnb = %.5f", nb);
	address = -c_nbArray[13] + idx + 13*offset; //if(y==1 && x==0 && z==1) printf("\nst src(%d)", address);
	float st = dGrid[address]; //if(y==1 && x==0 && z==1) printf("\nst = %.5f", st);
	address = -c_nbArray[14] + idx + 14*offset; //if(y==1 && x==0 && z==1) printf("\nsb src(%d)", address);
	float sb = dGrid[address]; //if(y==1 && x==0 && z==1) printf("\nsb = %.5f", sb);
	address = -c_nbArray[15] + idx + 15*offset; //if(y==1 && x==0 && z==1) printf("\net src(%d)", address);
	float et = dGrid[address]; //if(y==1 && x==0 && z==1) printf("\net = %.5f", et);
	address = -c_nbArray[16] + idx + 16*offset; //if(y==1 && x==0 && z==1) printf("\neb src(%d)", address);
	float eb = dGrid[address]; //if(y==1 && x==0 && z==1) printf("\neb = %.5f", eb);
	address = -c_nbArray[17] + idx + 17*offset; //if(y==1 && x==0 && z==1) printf("\nwt src(%d)", address);
	float wt = dGrid[address]; //if(y==1 && x==0 && z==1) printf("\nwt = %.5f", wt);
	address = -c_nbArray[18] + idx + 18*offset; //if(y==1 && x==0 && z==1) printf("\nwb src(%d)", address);
	float wb = dGrid[address]; //if(y==1 && x==0 && z==1) printf("\nwb = %.5f", wb);

	//get the value final cell (20) to know the kind of cell: fluid or accel or obstacle
	//int flag = flags[idx];
	unsigned short type = ((flags[idx-MARGIN_L_SIZE] == 1) << 8) | ((flags[idx-MARGIN_L_SIZE] == 2) & 0xff);
	//int isobs = (flags[idx] == 1);
	//int isacc = (flags[idx] == 2);

	if((type >> 8)==1)
	{
		sGrid[idx] = c;

		sGrid[idx + offset] = s;

		sGrid[idx + 2*offset] = n;

		sGrid[idx + 3*offset] = w;

		sGrid[idx + 4*offset] = e;

		sGrid[idx + 5*offset] = b;

		sGrid[idx + 6*offset] = t;

		sGrid[idx + 7*offset] = sw;

		sGrid[idx + 8*offset] = se;

		sGrid[idx + 9*offset] = nw;

		sGrid[idx + 10*offset] = ne;

		sGrid[idx + 11*offset] = sb;

		sGrid[idx + 12*offset] = st;

		sGrid[idx + 13*offset] = nb;

		sGrid[idx + 14*offset] = nt;

		sGrid[idx + 15*offset] = wb;

		sGrid[idx + 16*offset] = wt;

		sGrid[idx + 17*offset] = eb;

		sGrid[idx + 18*offset] = et;

	}
	else
	{

		float rho = +c + n + s + e + w + t + b + ne + nw + se + sw + nt + nb + st + sb + et + eb + wt + wb;

		float ux = ((+e - w + ne - nw + se - sw + et + eb - wt - wb)/rho)*(!(type & 0xff)) + 0.005*(type & 0xff);
		float uy = ((+n - s + ne + nw - se - sw + nt + nb - st - sb)/rho)*(!(type & 0xff)) + 0.002*(type & 0xff);
		float uz = ((+t - b + nt - nb + st - sb + et - eb + wt - wb)/rho)*(!(type & 0xff));


		float u2 = 1.5 * (ux * ux + uy * uy + uz * uz);


		sGrid[idx] =  		  	(ONEMINUSOMEGA* c + DFL1_OMEGA * rho * (1.0 - u2));

		sGrid[idx + offset] =   (ONEMINUSOMEGA* n + DFL2_OMEGA * rho * (1.0 + uy * (4.5 * uy + 3.0) - u2));

		sGrid[idx + 2*offset] = (ONEMINUSOMEGA* s + DFL2_OMEGA * rho * (1.0 + uy * (4.5 * uy - 3.0) - u2));

		sGrid[idx + 3*offset] = (ONEMINUSOMEGA* e + DFL2_OMEGA * rho * (1.0 + ux * (4.5 * ux + 3.0) - u2));

		sGrid[idx + 4*offset] = (ONEMINUSOMEGA* w + DFL2_OMEGA * rho * (1.0 + ux * (4.5 * ux - 3.0) - u2));

		sGrid[idx + 5*offset] = (ONEMINUSOMEGA* t + DFL2_OMEGA * rho * (1.0 + uz * (4.5 * uz + 3.0) - u2));

		sGrid[idx + 6*offset] = (ONEMINUSOMEGA* b + DFL2_OMEGA * rho * (1.0 + uz * (4.5 * uz - 3.0) - u2));

		sGrid[idx + 7*offset] = (ONEMINUSOMEGA* ne + DFL3_OMEGA * rho * (1.0 + (+ux + uy) * (4.5 * (+ux + uy) + 3.0) - u2));

		sGrid[idx + 8*offset] = (ONEMINUSOMEGA* nw + DFL3_OMEGA * rho * (1.0 + (-ux + uy) * (4.5 * (-ux + uy) + 3.0) - u2));

		sGrid[idx + 9*offset] = (ONEMINUSOMEGA* se + DFL3_OMEGA * rho * (1.0 + (+ux - uy) * (4.5 * (+ux - uy) + 3.0) - u2));

		sGrid[idx + 10*offset] = (ONEMINUSOMEGA* sw + DFL3_OMEGA * rho * (1.0 + (-ux - uy) * (4.5 * (-ux - uy) + 3.0) - u2));

		sGrid[idx + 11*offset] = (ONEMINUSOMEGA* nt + DFL3_OMEGA * rho * (1.0 + (+uy + uz) * (4.5 * (+uy + uz) + 3.0) - u2));

		sGrid[idx + 12*offset] = (ONEMINUSOMEGA* nb + DFL3_OMEGA * rho * (1.0 + (+uy - uz) * (4.5 * (+uy - uz) + 3.0) - u2));

		sGrid[idx + 13*offset] = (ONEMINUSOMEGA* st + DFL3_OMEGA * rho * (1.0 + (-uy + uz) * (4.5 * (-uy + uz) + 3.0) - u2));

		sGrid[idx + 14*offset] = (ONEMINUSOMEGA* sb + DFL3_OMEGA * rho * (1.0 + (-uy - uz) * (4.5 * (-uy - uz) + 3.0) - u2));

		sGrid[idx + 15*offset] = (ONEMINUSOMEGA* et + DFL3_OMEGA * rho * (1.0 + (+ux + uz) * (4.5 * (+ux + uz) + 3.0) - u2));

		sGrid[idx + 16*offset] = (ONEMINUSOMEGA* eb + DFL3_OMEGA * rho * (1.0 + (+ux - uz) * (4.5 * (+ux - uz) + 3.0) - u2));

		sGrid[idx + 17*offset] = (ONEMINUSOMEGA* wt + DFL3_OMEGA * rho * (1.0 + (-ux + uz) * (4.5 * (-ux + uz) + 3.0) - u2));

		sGrid[idx + 18*offset] = (ONEMINUSOMEGA* wb + DFL3_OMEGA * rho * (1.0 + (-ux - uz) * (4.5 * (-ux - uz) + 3.0) - u2));

	}
}
__global__ void SoA_Pull_DPFP_Reduction(float *sGrid, float *dGrid, unsigned char *flags)
{
	int offset;

	//__shared__ float one_minus_omega;
	offset = SIZE_XX*SIZE_YY*SIZE_ZZ;
	//one_minus_omega = 1.0 - OMEGA;
	int x = threadIdx.x;
	int y = blockIdx.x;// + 1;
	int z = blockIdx.y;// + 1;

	//int ElementsPerBlock = blockDim.x;
	//float c, n, s, e, w, t, b, ne, nw, se, sw, nt, nb, st, sb, et,	eb, wt, wb;

	/*the grid is organized as follows:
	 * c c c c c c . . . | n n n n n n . . . | s s s s s s . . . | e e e e e e . . . | . . .
	 */

	//calculate the index
	//int idx = (z * SIZE_YY + y)*SIZE_XX + x;
	//int idx = ((blockIdx.y +1) * SIZE_YY + (blockIdx.x+1))*SIZE_XX + threadIdx.x;
	int idx = CALC_INDEX_SOA_WO_STRUCT(x, y, z, 0) + MARGIN_L_SIZE;

	//each direction has offset = 130*100*100 values, we need to jump one offset to get new value
	//int offset = SIZE_XX*SIZE_YY*SIZE_ZZ;
	//save index to address for operating
	//int address = offset;
	int address0 = -c_nbArray[ 0] + idx; //if(y==1 && x==0 && z==1) printf("\nc src(%d)", address);
	float c = dGrid[address0]*1.f; //if(y==1 && x==0 && z==1) printf("\nc = %.5f", c);
	int address1 = -c_nbArray[ 1] + idx + offset; ////if(y==1 && x==0 && z==1) printf("\nn src(%d)", address);
	float n = dGrid[address1]*1.f;//if(y==1 && x==0 && z==1) printf("\nn = %.5f", n);
	int address2 = -c_nbArray[ 2] + idx + 2*offset; //if(y==1 && x==0 && z==1) printf("\ns src(%d)", address);
	float s = dGrid[address2]*1.f; //if(y==1 && x==0 && z==1) printf("\ns = %.5f", s);
	int address3 = -c_nbArray[ 3] + idx + 3*offset; //if(y==1 && x==0 && z==1) printf("\ne src(%d)", address);
	float e = dGrid[address3]*1.f; //if(y==1 && x==0 && z==1) printf("\ne = %.5f", e);
	int address4 = -c_nbArray[ 4] + idx + 4*offset; //if(y==1 && x==0 && z==1) printf("\nw src(%d)", address);
	float w = dGrid[address4]*1.f; //if(y==1 && x==0 && z==1) printf("\nw = %.5f", w);
	int address5 = -c_nbArray[ 5] + idx + 5*offset; //if(y==1 && x==0 && z==1) printf("\nt src(%d)", address);
	float t = dGrid[address5]*1.f; //if(y==1 && x==0 && z==1) printf("\nt = %.5f", t);
	int address6 = -c_nbArray[ 6] + idx + 6*offset; //if(y==1 && x==0 && z==1) printf("\nb src(%d)", address);
	float b = dGrid[address6]*1.f; //if(y==1 && x==0 && z==1) printf("\nb = %.5f", b);
	int address7 = -c_nbArray[ 7] + idx + 7*offset; //if(y==1 && x==0 && z==1) printf("\nne src(%d)", address);
	float ne = dGrid[address7]*1.f; //if(y==1 && x==0 && z==1) printf("\nne = %.5f", ne);
	int address8 = -c_nbArray[ 8] + idx + 8*offset; //if(y==1 && x==0 && z==1) printf("\nnw src(%d)", address);
	float nw = dGrid[address8]*1.f; //if(y==1 && x==0 && z==1) printf("\nnw = %.5f", nw);
	int address9 = -c_nbArray[ 9] + idx + 9*offset; //if(y==1 && x==0 && z==1) printf("\nse src(%d)", address);
	float se = dGrid[address9]*1.f; //if(y==1 && x==0 && z==1) printf("\nse = %.5f", se);
	int address10 = -c_nbArray[10] + idx + 10*offset; //if(y==1 && x==0 && z==1) printf("\nsw src(%d)", address);
	float sw = dGrid[address10]*1.f; //if(y==1 && x==0 && z==1) printf("\nsw = %.5f", sw);
	int address11 = -c_nbArray[11] + idx + 11*offset; //if(y==1 && x==0 && z==1) printf("\nnt src(%d)", address);
	float nt = dGrid[address11]*1.f; //if(y==1 && x==0 && z==1) printf("\nnt = %.5f", nt);
	int address12 = -c_nbArray[12] + idx + 12*offset; //if(y==1 && x==0 && z==1) printf("\nnb src(%d)", address);
	float nb = dGrid[address12]*1.f; //if(y==1 && x==0 && z==1) printf("\nnb = %.5f", nb);
	int address13 = -c_nbArray[13] + idx + 13*offset; //if(y==1 && x==0 && z==1) printf("\nst src(%d)", address);
	float st = dGrid[address13]*1.f; //if(y==1 && x==0 && z==1) printf("\nst = %.5f", st);
	int address14 = -c_nbArray[14] + idx + 14*offset; //if(y==1 && x==0 && z==1) printf("\nsb src(%d)", address);
	float sb = dGrid[address14]*1.f; //if(y==1 && x==0 && z==1) printf("\nsb = %.5f", sb);
	int address15 = -c_nbArray[15] + idx + 15*offset; //if(y==1 && x==0 && z==1) printf("\net src(%d)", address);
	float et = dGrid[address15]*1.f; //if(y==1 && x==0 && z==1) printf("\net = %.5f", et);
	int address16 = -c_nbArray[16] + idx + 16*offset; //if(y==1 && x==0 && z==1) printf("\neb src(%d)", address);
	float eb = dGrid[address16]*1.f; //if(y==1 && x==0 && z==1) printf("\neb = %.5f", eb);
	int address17 = -c_nbArray[17] + idx + 17*offset; //if(y==1 && x==0 && z==1) printf("\nwt src(%d)", address);
	float wt = dGrid[address17]*1.f; //if(y==1 && x==0 && z==1) printf("\nwt = %.5f", wt);
	int address18 = -c_nbArray[18] + idx + 18*offset; //if(y==1 && x==0 && z==1) printf("\nwb src(%d)", address);
	float wb = dGrid[address18]*1.f; //if(y==1 && x==0 && z==1) printf("\nwb = %.5f", wb);

	//get the value final cell (20) to know the kind of cell: fluid or accel or obstacle
	//int flag = flags[idx];
	unsigned short type = ((flags[idx-MARGIN_L_SIZE] == 1) << 8) | ((flags[idx-MARGIN_L_SIZE] == 2) & 0xff);
	//int isobs = (flags[idx] == 1);
	//int isacc = (flags[idx] == 2);

	if((type >> 8)==1)
	{	int idx0 = idx;
		sGrid[idx0] = c*1.f;
		int idx1 = idx + offset;
		sGrid[idx1] = s*1.f;
		int idx2 = idx + 2*offset;
		sGrid[idx2] = n*1.f;
		int idx3 = idx + 3*offset;
		sGrid[idx3] = w*1.f;
		int idx4 = idx + 4*offset;
		sGrid[idx4] = e*1.f;
		int idx5 = idx + 5*offset;
		sGrid[idx5] = b*1.f;
		int idx6 = idx + 6*offset;
		sGrid[idx6] = t*1.f;
		int idx7 = idx + 7*offset;
		sGrid[idx7] = sw*1.f;
		int idx8 = idx + 8*offset;
		sGrid[idx8] = se*1.f;
		int idx9 = idx + 9*offset;
		sGrid[idx9] = nw*1.f;
		int idx10 = idx + 10*offset;
		sGrid[idx10] = ne*1.f;
		int idx11 = idx + 11*offset;
		sGrid[idx11] = sb*1.f;
		int idx12 = idx + 12*offset;
		sGrid[idx12] = st*1.f;
		int idx13 = idx + 13*offset;
		sGrid[idx13] = nb*1.f;
		int idx14 = idx + 14*offset;
		sGrid[idx14] = nt*1.f;
		int idx15 = idx + 15*offset;
		sGrid[idx15] = wb*1.f;
		int idx16 = idx + 16*offset;
		sGrid[idx16] = wt*1.f;
		int idx17 = idx + 17*offset;
		sGrid[idx17] = eb*1.f;
		int idx18 = idx + 18*offset;
		sGrid[idx18] = et*1.f;

	}
	else
	{

		float rho = (+c + n + s + e + w + t + b + ne + nw + se + sw + nt + nb + st + sb + et + eb + wt + wb)*1.f;

		float ux = ((+e - w + ne - nw + se - sw + et + eb - wt - wb)/rho)*(!(type & 0xff)) + 0.005f*(type & 0xff);
		float uy = ((+n - s + ne + nw - se - sw + nt + nb - st - sb)/rho)*(!(type & 0xff)) + 0.002f*(type & 0xff);
		float uz = ((+t - b + nt - nb + st - sb + et - eb + wt - wb)/rho)*(!(type & 0xff));


		float u2 = 1.5f * (ux * ux + uy * uy + uz * uz)*1.f;

		int idx0 = idx;
		sGrid[idx0] = (1.0f - OMEGA)* c + DFL1 * OMEGA * rho * (1.0f- u2)*1.f;

		c = (type >> 8);
		int idx1 = idx + offset;
		sGrid[idx1] = (1.0f - OMEGA)* n + DFL2 * OMEGA * rho * (1.0f + uy * (4.5 * uy + 3.0) - u2)*1.f;
		int idx2 = idx + 2*offset;
		sGrid[idx2] = (1.0f - OMEGA)* s + DFL2 * OMEGA * rho * (1.0f + uy * (4.5 * uy - 3.0) - u2)*1.f;

		n = (1.0f - OMEGA); //resue variable n
		s = DFL2 * OMEGA * rho; //resue variable s
		int idx3 = idx + 3*offset;
		sGrid[idx3] = (n* e + s * (1.0f + ux * (4.5f * ux + 3.0f) - u2));
		int idx4 = idx + 4*offset;
		sGrid[idx4] = (n* w + s * (1.0f + ux * (4.5*1.f * ux - 3.0f) - u2));
		int idx5 = idx + 5*offset;
		sGrid[idx5] = (n* t + s * (1.0f + uz * (4.5*1.f * uz + 3.0f) - u2));
		int idx6 = idx + 6*offset;
		sGrid[idx6] = (n* b + s  * (1.0f + uz * (4.5*1.f * uz - 3.0f) - u2));

		b = DFL3 * OMEGA * rho; //resue variable b
		t = 1.0f - u2;
		e = ux + uy;
		w = b*(t+ 4.5*e*e);
		int idx7 = idx + 7*offset;
		sGrid[idx7] = (n* ne + w + 3*e*b);
		int idx10 = idx + 10*offset;
		sGrid[idx10] = (n* sw + w - 3*e*b);

		e = -ux + uy;
		w = b*(t+ 4.5*e*e);
		int idx8 = idx + 8*offset;
		sGrid[idx8] = (n* nw + w + 3*e*b);
		int idx9 = idx + 9*offset;
		sGrid[idx9] = (n* se + w - 3*e*b);

		e = uy + uz;
		w = b*(t+ 4.5*e*e);
		int idx11 = idx + 11*offset;
		sGrid[idx11] = (n* nt + w + 3*e*b);
		int idx14 = idx + 14*offset;
		sGrid[idx14] = (n* sb + w - 3*e*b);

		e = uy - uz;
		w = b*(t+ 4.5*e*e);
		int idx12 = idx + 12*offset;
		sGrid[idx12] = (n* nb + w + 3*e*b);
		int idx13 = idx + 13*offset;
		sGrid[idx13] = (n* st + w - 3*e*b);

		e = ux + uz;
		w = b*(t+ 4.5*e*e);
		int idx15 = idx + 15*offset;
		sGrid[idx15] = (n* et + w + 3*e*b);
		int idx18 = idx + 18*offset;
		sGrid[idx18] = (n* wb + w - 3*e*b);

		e = ux - uz;
		w = b*(t+ 4.5*e*e);
		int idx16 = idx + 16*offset;
		sGrid[idx16] = (n* eb + w + 3*e*b);
		int idx17 = idx + 17*offset;
		sGrid[idx17] = (n* wt + w - 3*e*b);


	}
}
