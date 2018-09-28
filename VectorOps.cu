#include "VectorOps.cuh"

void __device__  vvaddDev(int i2d, real alpha, real* x, real* y, int totpoints) {
	if(i2d>=totpoints) return;
	y[i2d] += alpha * x[i2d];
	//if (i2d == printv) printf("vm: %.31f\n~~~~~~~~~~~~~~~~\n", g_dev.vm[i2d]);
}