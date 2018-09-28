#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include "../common/CudaSafeCall.h"

#include "LRDglobalVariables.cuh"
#include "typedefSparse.h"
#include "sparsePrototypes.cuh"
#include "typedefLRD.h"

#include "LRDhostPrototypes.h"
#include "LRDdevicePrototypes.cuh"

void LRD_gateinit(int memSize, size_t* pitch, gateType* gate_h, gateType* gate_dev, gateType* gate_devF) {

	cudaHostAlloc((void**)&(gate_h->vm), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->m), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->h), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->j), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->d), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->f), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->b), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->g), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->xr), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->xs1), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->xs2), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->zdv), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->ydv), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->nai), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->ki), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->nsr), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->nao), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->ko), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->cao), memSize, 0);

	cudaHostAlloc((void**)&(gate_h->cai), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->jsr), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->caiont), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->BOOL), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->tcicr), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->tjsrol), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->dcaiont), memSize, 0);

	// Allocate device memory arrays
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->vm, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->m, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->h, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->j, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->d, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->f, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->b, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->g, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->xr, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->xs1, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->xs2, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->zdv, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->ydv, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->nai, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->ki, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->nsr, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->nao, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->ko, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->cao, pitch,
		memSize, 1));

	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->cai, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->jsr, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->caiont, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->BOOL, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->tcicr, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->tjsrol, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->dcaiont, pitch,
		memSize, 1));

	// Allocate device forward memory arrays
	CudaSafeCall(cudaMallocPitch((void **)&gate_devF->vm, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_devF->m, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_devF->h, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_devF->j, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_devF->d, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_devF->f, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_devF->b, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_devF->g, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_devF->xr, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_devF->xs1, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_devF->xs2, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_devF->zdv, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_devF->ydv, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_devF->nai, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_devF->ki, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_devF->nsr, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_devF->nao, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_devF->ko, pitch,
		memSize, 1));

	CudaSafeCall(cudaMallocPitch((void **)&gate_devF->cai, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_devF->jsr, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_devF->caiont, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_devF->BOOL, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_devF->tcicr, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_devF->tjsrol, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_devF->dcaiont, pitch,
		memSize, 1));

	puts("\nFinished allocating device arrays\n");

	int totpoints = (int)memSize / sizeof(real);

	for (int idx = 0; idx < totpoints; idx++) {
		gate_h->vm[idx] = LRD_RestVoltage;

		gate_h->m[idx] = 0.0008;
		gate_h->h[idx] = 0.993771;
		gate_h->j[idx] = 0.995727;
		gate_h->d[idx] = 3.210618e-06;
		gate_h->f[idx] = 0.999837;
		gate_h->b[idx] = 0.000970231;
		gate_h->g[idx] = 0.994305;
		gate_h->xr[idx] = 0.000124042;
		gate_h->xs1[idx] = 0.00445683;
		gate_h->xs2[idx] = 0.00445683;
		gate_h->zdv[idx] = 0.0120892;
		gate_h->ydv[idx] = 0.999978;
		gate_h->nai[idx] = 9.0;
		gate_h->ki[idx] = 141.2;
		gate_h->nsr[idx] = 1.838;
		gate_h->nao[idx] = 140;
		gate_h->ko[idx] = 4.5;
		gate_h->cao[idx] = 1.8;

		gate_h->cai[idx] = 0.00006;
		gate_h->jsr[idx] = 1.838;
		gate_h->caiont[idx] = 0;
		gate_h->BOOL[idx] = 0;
		gate_h->tcicr[idx] = -25;
		gate_h->tjsrol[idx] = -25;
		gate_h->dcaiont[idx] = 0;
	}

	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->vm, *pitch, (void *)gate_h->vm,
		memSize, memSize, 1, cudaMemcpyHostToDevice));

	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->m, *pitch, (void *)gate_h->m,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->h, *pitch, (void *)gate_h->h,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->j, *pitch, (void *)gate_h->j,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->d, *pitch, (void *)gate_h->d,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->f, *pitch, (void *)gate_h->f,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->b, *pitch, (void *)gate_h->b,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->g, *pitch, (void *)gate_h->g,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->xr, *pitch, (void *)gate_h->xr,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->xs1, *pitch, (void *)gate_h->xs1,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->xs2, *pitch, (void *)gate_h->xs2,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->zdv, *pitch, (void *)gate_h->zdv,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->ydv, *pitch, (void *)gate_h->ydv,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->nai, *pitch, (void *)gate_h->nai,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->ki, *pitch, (void *)gate_h->ki,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->nsr, *pitch, (void *)gate_h->nsr,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->nao, *pitch, (void *)gate_h->nao,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->ko, *pitch, (void *)gate_h->ko,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->cao, *pitch, (void *)gate_h->cao,
		memSize, memSize, 1, cudaMemcpyHostToDevice));

	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->cai, *pitch, (void *)gate_h->cai,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->jsr, *pitch, (void *)gate_h->jsr,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->caiont, *pitch, (void *)gate_h->caiont,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->BOOL, *pitch, (void *)gate_h->BOOL,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->tcicr, *pitch, (void *)gate_h->tcicr,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->tjsrol, *pitch, (void *)gate_h->tjsrol,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->dcaiont, *pitch, (void *)gate_h->dcaiont,
		memSize, memSize, 1, cudaMemcpyHostToDevice));


	CudaSafeCall(cudaMemcpy2D((void *)gate_devF->vm, *pitch, (void *)gate_h->vm,
		memSize, memSize, 1, cudaMemcpyHostToDevice));

	CudaSafeCall(cudaMemcpy2D((void *)gate_devF->m, *pitch, (void *)gate_h->m,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_devF->h, *pitch, (void *)gate_h->h,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_devF->j, *pitch, (void *)gate_h->j,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_devF->d, *pitch, (void *)gate_h->d,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_devF->f, *pitch, (void *)gate_h->f,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_devF->b, *pitch, (void *)gate_h->b,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_devF->g, *pitch, (void *)gate_h->g,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_devF->xr, *pitch, (void *)gate_h->xr,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_devF->xs1, *pitch, (void *)gate_h->xs1,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_devF->xs2, *pitch, (void *)gate_h->xs2,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_devF->zdv, *pitch, (void *)gate_h->zdv,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_devF->ydv, *pitch, (void *)gate_h->ydv,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_devF->nai, *pitch, (void *)gate_h->nai,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_devF->ki, *pitch, (void *)gate_h->ki,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_devF->nsr, *pitch, (void *)gate_h->nsr,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_devF->nao, *pitch, (void *)gate_h->nao,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_devF->ko, *pitch, (void *)gate_h->ko,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_devF->cao, *pitch, (void *)gate_h->cao,
		memSize, memSize, 1, cudaMemcpyHostToDevice));

	CudaSafeCall(cudaMemcpy2D((void *)gate_devF->cai, *pitch, (void *)gate_h->cai,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_devF->jsr, *pitch, (void *)gate_h->jsr,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_devF->caiont, *pitch, (void *)gate_h->caiont,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_devF->BOOL, *pitch, (void *)gate_h->BOOL,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_devF->tcicr, *pitch, (void *)gate_h->tcicr,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_devF->tjsrol, *pitch, (void *)gate_h->tjsrol,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_devF->dcaiont, *pitch, (void *)gate_h->dcaiont,
		memSize, memSize, 1, cudaMemcpyHostToDevice));

	real** qpH = (real**)malloc(sizeof(real *)*gate_h->qpl);
	int i = 0;
	qpH[i++] = gate_devF->m;
	qpH[i++] = gate_devF->h;
	qpH[i++] = gate_devF->j;
	qpH[i++] = gate_devF->;
	qpH[i++] = gate_devF->f;
	qpH[i++] = gate_devF->b;
	qpH[i++] = gate_devF->g;
	qpH[i++] = gate_devF->xr;
	qpH[i++] = gate_devF->xs1;
	qpH[i++] = gate_devF->xs2;
	qpH[i++] = gate_devF->zdv;
	qpH[i++] = gate_devF->ydv;
	qpH[i++] = gate_devF->nai;
	qpH[i++] = gate_devF->ki;
	qpH[i++] = gate_devF->nsr;
	qpH[i++] = gate_devF->nao;
	qpH[i++] = gate_devF->ko;
	qpH[i++] = gate_devF->cao;

	CudaSafeCall(cudaMemcpy((void *)gate_devF->qp, (void*)qpH, sizeof(real *)*gate_h->qpl, cudaMemcpyHostToDevice));

	i = 0;
	qpH[i++] = gate_dev->m;
	qpH[i++] = gate_dev->h;
	qpH[i++] = gate_dev->j;
	qpH[i++] = gate_dev->;
	qpH[i++] = gate_dev->f;
	qpH[i++] = gate_dev->b;
	qpH[i++] = gate_dev->g;
	qpH[i++] = gate_dev->xr;
	qpH[i++] = gate_dev->xs1;
	qpH[i++] = gate_dev->xs2;
	qpH[i++] = gate_dev->zdv;
	qpH[i++] = gate_dev->ydv;
	qpH[i++] = gate_dev->nai;
	qpH[i++] = gate_dev->ki;
	qpH[i++] = gate_dev->nsr;
	qpH[i++] = gate_dev->nao;
	qpH[i++] = gate_dev->ko;
	qpH[i++] = gate_dev->cao;

	CudaSafeCall(cudaMemcpy((void *)gate_dev->qp, (void*)qpH, sizeof(real *)*gate_h->qpl, cudaMemcpyHostToDevice));
	

	CudaCheckError();

	puts("\nFinished initializing device arrays\n");

}

void LRD_sync(int memSize, size_t pitch, gateType* gate_h, gateType* gate_dev) {

	CudaSafeCall(cudaMemcpy2D((void *)gate_h->vm, *pitch, (void *)gate_dev->vm,
		memSize, memSize, 1, cudaMemcpyDeviceToHost));

	CudaSafeCall(cudaMemcpy2D((void *)gate_h->m, *pitch, (void *)gate_dev->m,
		memSize, memSize, 1, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->h, *pitch, (void *)gate_dev->h,
		memSize, memSize, 1, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->j, *pitch, (void *)gate_dev->j,
		memSize, memSize, 1, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->d, *pitch, (void *)gate_dev->d,
		memSize, memSize, 1, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->f, *pitch, (void *)gate_dev->f,
		memSize, memSize, 1, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->b, *pitch, (void *)gate_dev->b,
		memSize, memSize, 1, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->g, *pitch, (void *)gate_dev->g,
		memSize, memSize, 1, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->xr, *pitch, (void *)gate_dev->xr,
		memSize, memSize, 1, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->xs1, *pitch, (void *)gate_dev->xs1,
		memSize, memSize, 1, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->xs2, *pitch, (void *)gate_dev->xs2,
		memSize, memSize, 1, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->zdv, *pitch, (void *)gate_dev->zdv,
		memSize, memSize, 1, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->ydv, *pitch, (void *)gate_dev->ydv,
		memSize, memSize, 1, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->nai, *pitch, (void *)gate_dev->nai,
		memSize, memSize, 1, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->ki, *pitch, (void *)gate_dev->ki,
		memSize, memSize, 1, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->nsr, *pitch, (void *)gate_dev->nsr,
		memSize, memSize, 1, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->nao, *pitch, (void *)gate_dev->nao,
		memSize, memSize, 1, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->ko, *pitch, (void *)gate_dev->ko,
		memSize, memSize, 1, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->cao, *pitch, (void *)gate_dev->cao,
		memSize, memSize, 1, cudaMemcpyDeviceToHost));

	CudaSafeCall(cudaMemcpy2D((void *)gate_h->cai, *pitch, (void *)gate_dev->cai,
		memSize, memSize, 1, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->jsr, *pitch, (void *)gate_dev->jsr,
		memSize, memSize, 1, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->caiont, *pitch, (void *)gate_dev->caiont,
		memSize, memSize, 1, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->BOOL, *pitch, (void *)gate_dev->BOOL,
		memSize, memSize, 1, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->tcicr, *pitch, (void *)gate_dev->tcicr,
		memSize, memSize, 1, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->tjsrol, *pitch, (void *)gate_dev->tjsrol,
		memSize, memSize, 1, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->dcaiont, *pitch, (void *)gate_dev->dcaiont,
		memSize, memSize, 1, cudaMemcpyDeviceToHost));
}

void LRD_exit(int memSize, size_t pitch, gateType* gate_h, gateType* gate_dev, gateType* gate_devF, sparse* MatrixINT, cudasparse* cudaMatrixINT){
	// Free gate host and device memory
	cudaFreeHost(gate_h->vm);

	cudaFreeHost(gate_h->m);
	cudaFreeHost(gate_h->h);
	cudaFreeHost(gate_h->j);
	cudaFreeHost(gate_h->d);
	cudaFreeHost(gate_h->f);
	cudaFreeHost(gate_h->b);
	cudaFreeHost(gate_h->g);
	cudaFreeHost(gate_h->xr);
	cudaFreeHost(gate_h->xs1);
	cudaFreeHost(gate_h->xs2);
	cudaFreeHost(gate_h->zdv);
	cudaFreeHost(gate_h->ydv);
	cudaFreeHost(gate_h->nai);
	cudaFreeHost(gate_h->ki);
	cudaFreeHost(gate_h->nsr);
	cudaFreeHost(gate_h->nao);
	cudaFreeHost(gate_h->ko);
	cudaFreeHost(gate_h->cao);

	cudaFreeHost(gate_h->cai);
	cudaFreeHost(gate_h->jsr);
	cudaFreeHost(gate_h->caiont);
	cudaFreeHost(gate_h->BOOL);
	cudaFreeHost(gate_h->tcicr);
	cudaFreeHost(gate_h->tjsrol);
	cudaFreeHost(gate_h->dcaiont);
	cudaFreeHost(gate_h->qp);


	cudaFree(gate_dev->vm);

	cudaFree(gate_dev->m);
	cudaFree(gate_dev->h);
	cudaFree(gate_dev->j);
	cudaFree(gate_dev->d);
	cudaFree(gate_dev->f);
	cudaFree(gate_dev->b);
	cudaFree(gate_dev->g);
	cudaFree(gate_dev->xr);
	cudaFree(gate_dev->xs1);
	cudaFree(gate_dev->xs2);
	cudaFree(gate_dev->zdv);
	cudaFree(gate_dev->ydv);
	cudaFree(gate_dev->nai);
	cudaFree(gate_dev->ki);
	cudaFree(gate_dev->nsr);
	cudaFree(gate_dev->nao);
	cudaFree(gate_dev->ko);
	cudaFree(gate_dev->cao);

	cudaFree(gate_dev->cai);
	cudaFree(gate_dev->jsr);
	cudaFree(gate_dev->caiont);
	cudaFree(gate_dev->BOOL);
	cudaFree(gate_dev->tcicr);
	cudaFree(gate_dev->tjsrol);
	cudaFree(gate_dev->dcaiont);
	cudaFree(gate_dev->qp);


	cudaFree(gate_devF->vm);

	cudaFree(gate_devF->m);
	cudaFree(gate_devF->h);
	cudaFree(gate_devF->j);
	cudaFree(gate_devF->d);
	cudaFree(gate_devF->f);
	cudaFree(gate_devF->b);
	cudaFree(gate_devF->g);
	cudaFree(gate_devF->xr);
	cudaFree(gate_devF->xs1);
	cudaFree(gate_devF->xs2);
	cudaFree(gate_devF->zdv);
	cudaFree(gate_devF->ydv);
	cudaFree(gate_devF->nai);
	cudaFree(gate_devF->ki);
	cudaFree(gate_devF->nsr);
	cudaFree(gate_devF->nao);
	cudaFree(gate_devF->ko);
	cudaFree(gate_devF->cao);

	cudaFree(gate_devF->cai);
	cudaFree(gate_devF->jsr);
	cudaFree(gate_devF->caiont);
	cudaFree(gate_devF->BOOL);
	cudaFree(gate_devF->tcicr);
	cudaFree(gate_devF->tjsrol);
	cudaFree(gate_devF->dcaiont);	
	cudaFree(gate_devF->qp);



	cudaFree(cudaMatrixINT->type);
	cudaFree(cudaMatrixINT->rows);
	cudaFree(cudaMatrixINT->maxnz);
	cudaFree(cudaMatrixINT->csep);
	cudaFree(cudaMatrixINT->jcoef);
	cudaFree(cudaMatrixINT->coef);
}


void __device__ GetFDev_LRD(int i2d, int pitch, real beta, real Cm, real t, real dt, int totpoints, real rx, gateType g_dev, gateType g_devF) {

	/*------------------------------------------------------------------------
	* return if outside domain
	*------------------------------------------------------------------------
	*/
	if (i2d >= totpoints) {
		return;
	}

	/* declare variables */ 	                                 
    real vm,BOOL,tcicr,tjsrol,csqn;
    real LRD_RTF;
    real LRD_Gkr, LRD_Gks,LRD_Gki,LRD_Gkatp;
    real LRD_Ena,LRD_Etca,LRD_Ekr,LRD_Eks,LRD_Eki,LRD_Ekp;
    real LRD_Ekna,LRD_Ekatp,LRD_Ekdv,LRD_Ecan,LRD_Enan;
    real m,h,j,am,bm,ah,bh,aj,bj,Ina;  
    real d,f,dss,taud,fss,tauf,Ibarca,Ibarna,Ibark;
    real fca,Ilca,Ilcana,Ilcak,Ilcatot;
    real b,g,bss,taub,gss,taug,Itca;
    real xr,r,xrss,tauxr,Ikr;
    real xs1,xs2,xs1ss,xs2ss,tauxs1,tauxs2,Iks; 
    real aki,bki,kin,Ikti;
    real kp,Ikp;
    real Inaca;
    real sigma,fnak,Inak;
    real Ipca;
    real Icab;
    real Inab;
    real pona,pov,Ikna;
    real patp,gkbaratp,Ikatp;
    real Ibarnsna,Ibarnsk,Insna,Insk;
    real rvdv,Ito;
    real azdv,bzdv,tauzdv,zdvss,zdv;
    real aydv,bydv,tauydv,ydvss,ydv;
    real naiont,kiont,caiont,Itotal;
    /*ions*/
    real nao,ko,cao;
    real dnao,dko,dcao;
 
	real nai,ki;
	real dnai,dki;

    real itr;
    real nsr,kleak,ileak,iup,dnsr;
    /* JSR CICR */
    real dcaiont,caiontold;
    real magrel,on,off,irelcicr;
    real greljsrol,ireljsrol;
    real trpn,cmdn;
    real jsr,bjsr,cjsr,djsr; 
    /* cai update here */
    real cai,catotal,bmyo,cmyo,dmyo,gpig,dcai;
    real vcell,ageo,acap,vmyo,vnsr,vjsr,vcleft;
    LRD_RTF = LRD_R*LRD_temp/LRD_frdy;


	vm        = g_dev.vm[i2d];

	m 		  = g_dev.m[i2d];
	h 		  = g_dev.h[i2d];
	j 		  = g_dev.j[i2d];
	d 		  = g_dev.d[i2d];
	f 		  = g_dev.f[i2d];
	b 		  = g_dev.b[i2d];
	g 	      = g_dev.g[i2d];
	xr 	      = g_dev.xr[i2d];
	xs1 	  = g_dev.xs1[i2d];
	xs2	      = g_dev.xs2[i2d];
	zdv       = g_dev.zdv[i2d];
	ydv	      = g_dev.ydv[i2d];
	nai 	  = g_dev.nai[i2d];
	ki 	      = g_dev.ki[i2d];
	nsr	      = g_dev.nsr[i2d];
	nao	      = g_dev.nao[i2d];
	ko 	      = g_dev.ko[i2d];
	cao       = g_dev.cao[i2d];

	cai       = g_dev.cai[i2d];
	jsr       = g_dev.jsr[i2d];
	caiont    = g_dev.caiont[i2d];
	BOOL      = g_dev.BOOL[i2d];
	tcicr     = g_dev.tcicr[i2d];
	tjsrol    = g_dev.tjsrol[i2d];
	dcaiont   = g_dev.dcaiont[i2d];

	/*------------------------------------------------------------------------
	* setting local variables
	*------------------------------------------------------------------------
	*/

	real fv = g_devF.vm[i2d];

	/* Declare varying G's and E's */
    LRD_Gkr   = 0.02614*sqrt(ko/5.4); 
    LRD_Gks   = 0.433*(1+0.6/(1+pow((0.000038/cai),1.4)));
    LRD_Gki   = 0.75*(sqrt(ko/5.4));
    LRD_Gkatp = 0.000195/nicholsarea;
    
    LRD_Ena   = (LRD_RTF)*log(nao/nai);
    LRD_Etca  = 0.5*(LRD_RTF)*log(cao/cai);  
    LRD_Ekr   = (LRD_RTF)*log(ko/ki);
    LRD_Eks   = (LRD_RTF)*log((ko+prnak*nao)/(ki+prnak*nai));
    LRD_Eki   = (LRD_RTF)*log(ko/ki);
    LRD_Ekp   = LRD_Eki;
    LRD_Ekna  = LRD_Ekr; 
    LRD_Ekatp = LRD_Ekr; 
    LRD_Ekdv  = LRD_Ekr;
    LRD_Ecan  = LRD_Etca;
    LRD_Enan  = LRD_Ena;

	/* Na current [15] */
    am = 0.32*(vm+47.13)/(1-exp(-0.1*(vm+47.13)));
    bm = 0.08*exp(-vm/11);
    if (vm < -40) {
		ah = 0.135*exp((80+vm)/-6.8);
		bh = 3.56*exp(0.079*vm)+310000*exp(0.35*vm);
		aj = (-127140*exp(0.2444*vm)-0.00003474*exp(-0.04391*vm))*((vm+37.78)/(1+exp(0.311*(vm+79.23))));
		bj = (0.1212*exp(-0.01052*vm))/(1+exp(-0.1378*(vm+40.14)));
     } else {
		ah = 0;
		bh = 1/(0.13*(1+exp((vm+10.66)/-11.1)));
		aj = 0;
		bj = (0.3*exp(-0.0000002535*vm))/(1+exp(-0.1*(vm+32)));
    }        

      Ina=LRD_Gna*(m*m*m*h*j)*(vm-LRD_Ena);

	/* L-type Calcium current [14,15] */
	dss  = 1/(1+exp(-(vm+10)/6.24));
	taud = dss*(1-exp(-(vm+10)/6.24))/(0.035*(vm+10));
	fss  = (1/(1+exp((vm+32)/8)))+(0.6/(1+exp((50-vm)/20)));
	tauf = 1/(0.0197*exp(-0.0337*0.0337*(vm+10)*(vm+10))+0.02);

	Ibarca = pca*zca*zca*((vm*LRD_frdy)/(LRD_RTF))*((gacai*cai*exp((zca*vm)/(LRD_RTF))-gacao*cao)/(exp((zca*vm)/(LRD_RTF))-1));
	Ibarna = pna*zna*zna*((vm*LRD_frdy)/(LRD_RTF))*((ganai*nai*exp((zna*vm)/(LRD_RTF))-ganao*nao)/(exp((zna*vm)/(LRD_RTF))-1));
	Ibark  = pk*zk*zk*((vm*LRD_frdy)/(LRD_RTF))*((gaki*ki*exp((zk*vm)/(LRD_RTF))-gako*ko)/(exp((zk*vm)/(LRD_RTF))-1));

	fca = 1/(1+cai/kmca);

	Ilca    = d*f*fca*Ibarca;
	Ilcana  = d*f*fca*Ibarna;
	Ilcak   = d*f*fca*Ibark;	
	Ilcatot = Ilca+Ilcana+Ilcak;

	/* T-type Calcium current [13] */
	bss  = 1/(1+exp(-(vm+14)/10.8));
	taub = 3.7+6.1/(1+exp((vm+25)/4.5));
	gss  = 1/(1+exp((vm+60)/5.6));
	if (vm<=0) {
		taug = -0.875*vm+12;
	} else {
		taug = 12;
	}
	Itca = LRD_Gtca*b*b*g*(vm-LRD_Etca);

	/* K current - Rapid [13] */
	xrss  = 1/(1+exp(-(vm+21.5)/7.5));
	tauxr = 1/(0.00138*(vm+14.2)/(1-exp(-0.123*(vm+14.2)))+0.00061*(vm+38.9)/(exp(0.145*(vm+38.9))-1));

	r = 1/(1+exp((vm+9)/22.4));

	Ikr = LRD_Gkr*xr*r*(vm-LRD_Ekr);

	/* K current - Slow [10,13] */
	xs1ss  = 1/(1+exp(-(vm-1.5)/16.7));
	xs2ss  = xs1ss;
	tauxs1 = 1/(0.0000719*(vm+30)/(1-exp(-0.148*(vm+30)))+0.000131*(vm+30)/(exp(0.0687*(vm+30))-1));
	tauxs2 = 4*tauxs1;

	Iks = LRD_Gks*xs1*xs2*(vm-LRD_Eks);

	/* K current -  Time independent [15] */
	aki = 1.02/(1+exp(0.2385*(vm-LRD_Eki-59.215)));
	bki = (0.49124*exp(0.08032*(vm-LRD_Eki+5.476))+exp(0.06175*(vm-LRD_Eki-594.31)))/(1+exp(-0.5143*(vm-LRD_Eki+4.753)));	

	kin = aki/(aki+bki);

	Ikti = LRD_Gki*kin*(vm-LRD_Eki);

	/* K current - Plateau [15] */
	kp  = 1/(1+exp((7.488-vm)/5.98));	

	Ikp = LRD_Gkp*kp*(vm-LRD_Ekp);
      
	/* Na-Ca exchanger [6,14,15] */
	Inaca = c1*exp((gammas-1)*vm/(LRD_RTF))*((exp(vm/(LRD_RTF))*nai*nai*nai*cao-nao*nao*nao*cai)/(1+c2*exp((gammas-1)*vm/(LRD_RTF))*(exp(vm/(LRD_RTF))*nai*nai*nai*cao+nao*nao*nao*cai)));

	/* Na-K pump [15] */
	sigma = (exp(nao/67.3)-1)/7;
	fnak  = 1/(1+0.1245*exp((-0.1*vm)/(LRD_RTF))+0.0365*sigma*exp((-vm)/(LRD_RTF)));

	Inak = Ibarnak*fnak*(1/(1+kmnai*kmnai/(nai*nai)))*(ko/(ko+kmko));

	/* Sarcolemmal Ca pump [15] */
	Ipca = (Ibarpca*cai)/(kmpca+cai);

	/* Ca background current [15] */
	Icab = LRD_Gcab*(vm-LRD_Ecan);

	/* Na background current [15] */
	Inab = LRD_Gnab*(vm-LRD_Enan);          
      
	/* Na activated K current [6] */
	pona = 0.85/(1+pow((kdkna/nai),2.8));
	pov  = 0.8-(0.65/(1+exp((vm+125)/15)));

	Ikna = LRD_ikna*LRD_Gkna*pona*pov*(vm-LRD_Ekna);

	/* ATP sensitive K current [11] */
	patp = 1/(1+(pow((atpi/katp),hatp)));
	gkbaratp = LRD_Gkatp*patp*(pow((ko/4),natp));

	Ikatp = LRD_ikatp*gkbaratp*(vm-LRD_Ekatp);     

	/* Non-specific Ca-activated current [14,15] */
	Ibarnsna = pnsca*zna*zna*((vm*LRD_frdy)/(LRD_RTF))*((ganai*nai*exp((zna*vm)/(LRD_RTF))-ganao*nao)/(exp((zna*vm)/(LRD_RTF))-1));
	Ibarnsk  = pnsca*zk*zk*((vm*LRD_frdy)/(LRD_RTF))*((gaki*ki*exp((zk*vm)/(LRD_RTF))-gako*ko)/(exp((zk*vm)/(LRD_RTF))-1));

	Insna = LRD_insna*Ibarnsna/(1+kmnsca*kmnsca*kmnsca/(cai*cai*cai)); 
	Insk = LRD_insk*Ibarnsk/(1+kmnsca*kmnsca*kmnsca/(cai*cai*cai));   

	/* Transient outward current */
	rvdv = exp(vm/100);

	azdv = (10*exp((vm-40)/25))/(1+exp((vm-40)/25));
	bzdv = (10*exp(-(vm+90)/25))/(1+exp(-(vm+90)/25));
	tauzdv = 1/(azdv+bzdv);
	zdvss = azdv/(azdv+bzdv);

	aydv = 0.015/(1+exp((vm+60)/5));
	bydv = (0.1*exp((vm+25)/5))/(1+exp((vm+25)/5));
	tauydv = 1/(aydv+bydv);
	ydvss = aydv/(aydv+bydv);

	Ito = LRD_ito*LRD_Gitodv*zdv*zdv*zdv*ydv*rvdv*(vm-LRD_Ekdv);

	/* Summing currents (inactive currents are set to zero with activation variables) */
	naiont = Ina+Inab+Ilcana+3*Inak+3*Inaca+Insna;
	kiont  = Ikr+Iks+Ikti+Ikp+Ilcak+-2*Inak+Insk+Ito+Ikna+Ikatp;  
	caiont = Ilca+Icab+Ipca-2*Inaca+Itca;

	Itotal = LRD_cm*(naiont+kiont+caiont);     /* uA/cm2 */
 
	if (((t-tcicr)>80) && (vm<-30)) {
		BOOL = 0;  
		g_dev.BOOL[i2d] = BOOL;                                                                  
	}

	/* Put voltage update here  */
	fv += -Itotal;
	g_devF.vm[i2d] = fv;


	/* change in cleft concentration */
	dnao = LRD_cleft*((nabm-nao)/taudiff+naiont*acap*LRD_cm/(vcleft*LRD_frdy));
	dko = LRD_cleft*((kbm-ko)/taudiff+kiont*acap*LRD_cm/(vcleft*LRD_frdy));
	dcao = LRD_cleft*((cabm-cao)/taudiff+caiont*acap*LRD_cm/(vcleft*LRD_frdy*2));

	/* change in nai and ki concentration */
	dnai = -LRD_cm*(naiont*acap)/(vmyo*zna*LRD_frdy);     /* dnai/dt */
	dki = -LRD_cm*(kiont*acap)/(vmyo*zk*LRD_frdy);        /* dki/dt */

	/* change in itr [14] */ 
	itr = (nsr-jsr)/tautr;                           

	/* change in nsr [14] */
	kleak = iupbar/nsrbar;
	ileak = kleak*nsr;
	iup   = iupbar*cai/(cai+kmup);

	dnsr = (iup-ileak-itr*vjsr/vnsr);          /* dnsr/dt */

	/* Calcium-induced-calcium-release (CICR) criteia [6] */
	if ((vm>-35) && (((caiont-caiontold)/dt)<dcaiont) && (BOOL==0)){
		BOOL  = 1;
		tcicr = t;
		g_dev.BOOL[i2d] = BOOL;
		g_dev.tcicr[i2d] = tcicr;   /* changes reference time */
	}   

	on  = 1/(1+exp((-(t-tcicr)+4)/.5));
	off = 1-on;
	magrel = 1/(1+exp(((Ilca+Icab+Ipca-2*Inaca+Itca)+5)/0.9));

	irelcicr = gmaxrel*on*off*magrel*(jsr-cai);

	/* JSR Calciium overload [13] */
	greljsrol = grelbarjsrol*(1-exp(-(t-tjsrol)/tauon))*exp(-(t-tjsrol)/tauoff);
	ireljsrol = greljsrol*(jsr-cai);

	csqn = csqnbar*(jsr/(jsr+kmcsqn));

	djsr = dt*(itr-irelcicr-ireljsrol);

	bjsr = csqnbar-csqn-djsr-jsr+kmcsqn;
	cjsr = kmcsqn*(csqn+djsr+jsr);

	jsr =(sqrt(bjsr*bjsr+4*cjsr)-bjsr)/2; 

	/* Calcium buffers in myoplasm [15] */
	trpn = trpnbar*(cai/(cai+kmtrpn));
	cmdn = cmdnbar*(cai/(cai+kmcmdn));
     
	/* change in cai concentration [13] */
	dcai = -dt*(((LRD_cm*caiont*acap)/(vmyo*zca*LRD_frdy))+((iup-ileak)*vnsr/vmyo)-(irelcicr*vjsr/vmyo)-(ireljsrol*vjsr/vmyo));

	catotal = trpn+cmdn+dcai+cai;

	bmyo = cmdnbar+trpnbar-catotal+kmtrpn+kmcmdn;
	cmyo = (kmcmdn*kmtrpn)-(catotal*(kmtrpn+kmcmdn))+(trpnbar*kmcmdn)+(cmdnbar*kmtrpn);
	dmyo = -kmtrpn*kmcmdn*catotal;
	gpig = sqrt(bmyo*bmyo-3*cmyo);

	cai = ((2*gpig/3)*cos(acos((9*bmyo*cmyo-2*bmyo*bmyo*bmyo-27*dmyo)/(2*pow((bmyo*bmyo-3*cmyo),1.5)))/3)-(bmyo/3)); 

	/* Calcium overload criteria [15] */
	if((csqn>=csqnth) && ((t-tjsrol)>50)){
		printf("Spontaneous Release occured at time %lf at node %i\n",t,i);
		tjsrol = t;  	       
		g_dev.tjsrol[i2d] = tjsrol;                /* changes reference time */
	}

    g_devF.m[i2d]    = am*(1.0-m) - bm*m;
    g_devF.h[i2d]    = ah*(1.0-h) - bh*h;
    g_devF.j[i2d]    = aj*(1.0-j) - bj*j;
    g_devF.d[i2d]    = (dss/taud)*(1-d)-(1-dss)*(d/taud);
    g_devF.f[i2d]    = (fss/tauf)*(1-f)-(1-fss)*(f/tauf);
    g_devF.b[i2d]    = (bss/taub)*(1-b)-(1-bss)*(b/taub);
    g_devF.g[i2d]    = (gss/taug)*(1-g)-(1-gss)*(g/taug);
    g_devF.xr[i2d]   = (xrss/tauxr)*(1-xr)-(1-xrss)*(xr/tauxr);
    g_devF.xs1[i2d]  = (xs1ss/tauxs1)*(1-xs1)-(1-xs1ss)*(xs1/tauxs1);
    g_devF.xs2[i2d]  = (xs2ss/tauxs2)*(1-xs2)-(1-xs2ss)*(xs2/tauxs2);
    g_devF.zdv[i2d]  = (zdvss/tauzdv)*(1-zdv)-(1-zdvss)*(zdv/tauzdv);
    g_devF.ydv[i2d]  = (ydvss/tauydv)*(1-ydv)-(1-ydvss)*(ydv/tauydv);
    g_devF.nai[i2d]  = dnai;
    g_devF.ki[i2d]   = dki;
    g_devF.nsr[i2d]  = dnsr;
    g_devF.nao[i2d]  = dnao; 
    g_devF.ko[i2d]   = dko; 
    g_devF.cao[i2d]  = dcao;


  /* assign Temp variables to memory */    
    g_devF.caiont[i2d]  = caiont;
    g_devF.cai[i2d]     = cai; 
    g_devF.jsr[i2d]     = jsr;
    g_devF.dcaiont[i2d] = (caiont-caiontold)/dt;

}