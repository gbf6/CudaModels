#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include "../common/CudaSafeCall.h"

#include "globalVariables.cuh"
#include "typedefSparse.h"
#include "sparsePrototypes.cuh"
#include "typedef.h"

#include "hostPrototypes.h"
#include "devicePrototypes.cuh"

void _gateinit(int memSize, size_t* pitch, gateType* gate_h, gateType* gate_dev, gateType* gate_devF) {

	cudaHostAlloc((void**)&(gate_h->), memSize, 0);
	

	// Allocate device memory arrays
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->, pitch,
		memSize, 1));

	// Allocate device forward memory arrays
	CudaSafeCall(cudaMallocPitch((void **)&gate_devF->, pitch,
		memSize, 1));

	puts("\nFinished allocating device arrays\n");

	int totpoints = (int)memSize / sizeof(real);

	for (int idx = 0; idx < totpoints; idx++) {
		gate_h->[idx] = 0.0;
	}

	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->, *pitch, (void *)gate_h->,
		memSize, memSize, 1, cudaMemcpyHostToDevice));

	CudaSafeCall(cudaMemcpy2D((void *)gate_devF->, *pitch, (void *)gate_h->,
		memSize, memSize, 1, cudaMemcpyHostToDevice));

	real** qpH = (real**)malloc(sizeof(real *)*gate_h->qpl);
	int i = 0;
	qpH[i++] = gate_devF->;

	CudaSafeCall(cudaMemcpy((void *)gate_devF->qp, (void*)qpH, sizeof(real *)*gate_h->qpl, cudaMemcpyHostToDevice));

	i = 0;
	qpH[i++] = gate_dev->;

	CudaSafeCall(cudaMemcpy((void *)gate_dev->qp, (void*)qpH, sizeof(real *)*gate_h->qpl, cudaMemcpyHostToDevice));
	

	CudaCheckError();

	puts("\nFinished initializing device arrays\n");

}

void _sync(int memSize, size_t pitch, gateType* gate_h, gateType* gate_dev) {
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->, memSize, (void *)gate_dev->,
		pitch, memSize, 1, cudaMemcpyDeviceToHost));
}

void _exit(int memSize, size_t pitch, gateType* gate_h, gateType* gate_dev, gateType* gate_devF, sparse* MatrixINT, cudasparse* cudaMatrixINT){
	// Free gate host and device memory
	cudaFreeHost(gate_h->); 

	cudaFree(gate_dev->); 

	cudaFree(gate_devF->); 

	cudaFree(cudaMatrixINT->type);
	cudaFree(cudaMatrixINT->rows);
	cudaFree(cudaMatrixINT->maxnz);
	cudaFree(cudaMatrixINT->csep);
	cudaFree(cudaMatrixINT->jcoef);
	cudaFree(cudaMatrixINT->coef);
}


void __device__ GetFDev_(int i2d, int pitch, real beta, real Cm, real t, real dt, int totpoints, real rx, gateType g_dev, gateType g_devF) {

	/*------------------------------------------------------------------------
	* return if outside domain
	*------------------------------------------------------------------------
	*/
	if (i2d >= totpoints) {
		return;
	}

	/*------------------------------------------------------------------------
	* setting local variables
	*------------------------------------------------------------------------
	*/
	real  = g_dev.[i2d];

	real f = g_devF.[i2d];



	f -= dt * I_sum / Cm;

	g_devF.[i2d] = ;
}