#include <math.h>
#include <cuda.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "./common/CudaSafeCall.h"

#include "typedefSparse.h"
#include "sparsePrototypes.cuh"
#include "parseInput.cuh"
#include "VectorOps.cuh"

#include "mainwrapper.cuh"

#include "inputs.cuh"

#include "./CRN/CRNMethods.cu"

#include "./Stim/stimSimple.cu"


void main_wrapper(int pitch, real beta, real Cm, real t, real dt, int totpoints, real rx, gateType g_dev, gateType g_devF, stimulus stim, cudasparse cudaMatrixINT) {
	int grid = (int)ceil(totpoints / (real)BLOCK_DIM_1D);
	int block = BLOCK_DIM_1D;

	TstepEM<<<grid, block>>>(pitch, beta, Cm, t, dt, totpoints, rx, g_dev, g_devF, cudaMatrixINT);
	cudaDeviceSynchronize();

	stimSimpleDev<<<1, 1>>>(stim, t, g_devF);
	cudaDeviceSynchronize();

	TstepEMExit<<<grid, block>>>(Cm, dt, totpoints, g_dev, g_devF);
	cudaDeviceSynchronize();
	
}

void __global__ TstepEM(int pitch, real beta, real Cm, real t, real dt, int totpoints, real rx, gateType g_dev, gateType g_devF, cudasparse cudaMatrixINT) {
	const int i2d = blockIdx.x*BLOCK_DIM_1D + threadIdx.x;
	if(i2d < totpoints) {
		g_devF.vm[i2d] = 0;
		sprDiff(g_devF.vm, &cudaMatrixINT, i2d, totpoints, g_dev.vm, beta);

		GetFDev_CRN(i2d, pitch, beta, Cm, t, dt, totpoints, rx, g_dev, g_devF);

	}
}
void __global__ TstepEMExit(real Cm, real dt, int totpoints, gateType g_dev, gateType g_devF) {
	const int i2d = blockIdx.x*BLOCK_DIM_1D + threadIdx.x;
	if(i2d < totpoints){
		vvaddDev(i2d, dt / Cm, g_devF.vm, g_dev.vm, totpoints);
		for (int i = 0; i < g_dev.qpl; i++) vvaddDev(i2d, dt, g_devF.qp[i], g_dev.qp[i], totpoints);
	}
}