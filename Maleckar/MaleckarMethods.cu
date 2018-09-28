#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include "../common/CudaSafeCall.h"

#include "MaleckarglobalVariables.cuh"
#include "typedefSparse.h"
#include "sparsePrototypes.cuh"
#include "typedefMaleckar.h"

#include "MaleckarhostPrototypes.h"
#include "MaleckardevicePrototypes.cuh"

void Maleckar_gateinit(int memSize, size_t* pitch, gateType* gate_h, gateType* gate_dev, gateType* gate_devF) {

	cudaHostAlloc((void**)&(gate_h->vm), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->r), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->s), memSize, 0);
	

	// Allocate device memory arrays
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->vm, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->r, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->x, pitch,
		memSize, 1));

	// Allocate device forward memory arrays
	CudaSafeCall(cudaMallocPitch((void **)&gate_devF->vm, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_devF->r, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_devF->s, pitch,
		memSize, 1));

	puts("\nFinished allocating device arrays\n");

	int totpoints = (int)memSize / sizeof(real);

	for (int idx = 0; idx < totpoints; idx++) {
		gate_h->vm[idx] = Maleckar_RestVoltage;
		gate_h->r[idx] = 5.454e-02;
		gate_h->s[idx] = 9.814e-01;
	}

	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->vm, *pitch, (void *)gate_h->vm,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->r, *pitch, (void *)gate_h->r,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->s, *pitch, (void *)gate_h->s,
		memSize, memSize, 1, cudaMemcpyHostToDevice));


	CudaSafeCall(cudaMemcpy2D((void *)gate_devF->vm, *pitch, (void *)gate_h->vm,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_devF->r, *pitch, (void *)gate_h->r,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_devF->s, *pitch, (void *)gate_h->s,
		memSize, memSize, 1, cudaMemcpyHostToDevice));

	real** qpH = (real**)malloc(sizeof(real *)*gate_h->qpl);
	int i = 0;
	qpH[i++] = gate_devF->r;
	qpH[i++] = gate_devF->s;


	CudaSafeCall(cudaMemcpy((void *)gate_devF->qp, (void*)qpH, sizeof(real *)*gate_h->qpl, cudaMemcpyHostToDevice));

	i = 0;
	qpH[i++] = gate_dev->r;
	qpH[i++] = gate_dev->s;

	CudaSafeCall(cudaMemcpy((void *)gate_dev->qp, (void*)qpH, sizeof(real *)*gate_h->qpl, cudaMemcpyHostToDevice));
	

	CudaCheckError();

	puts("\nFinished initializing device arrays\n");

}

void Maleckar_sync(int memSize, size_t pitch, gateType* gate_h, gateType* gate_dev) {
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->vm, memSize, (void *)gate_dev->vm,
		pitch, memSize, 1, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->r, memSize, (void *)gate_dev->r,
		pitch, memSize, 1, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->s, memSize, (void *)gate_dev->s,
		pitch, memSize, 1, cudaMemcpyDeviceToHost));
}

void Maleckar_exit(int memSize, size_t pitch, gateType* gate_h, gateType* gate_dev, gateType* gate_devF, sparse* MatrixINT, cudasparse* cudaMatrixINT){
	// Free gate host and device memory
	cudaFreeHost(gate_h->vm); cudaFreeHost(gate_h->r); cudaFreeHost(gate_h->s); cudaFree(gate_dev->qp); 

	cudaFree(gate_dev->vm); cudaFree(gate_dev->r); cudaFree(gate_dev->s); cudaFree(gate_dev->qp); 

	cudaFree(gate_devF->vm); cudaFree(gate_devF->r); cudaFree(gate_devF->s); cudaFree(gate_devF->qp); 

	cudaFree(cudaMatrixINT->type);
	cudaFree(cudaMatrixINT->rows);
	cudaFree(cudaMatrixINT->maxnz);
	cudaFree(cudaMatrixINT->csep);
	cudaFree(cudaMatrixINT->jcoef);
	cudaFree(cudaMatrixINT->coef);
}


void __device__ GetFDev_Maleckar(int i2d, int pitch, real beta, real Cm, real t, real dt, int totpoints, real rx, gateType g_dev, gateType g_devF) {

	/*------------------------------------------------------------------------
	* return if outside domain
	*------------------------------------------------------------------------
	*/
	if (i2d >= totpoints) {
		return;
	}

	real   vm;
	real   r, s;
	real   r_inf, s_inf, tau_r, tau_s, aK1, bK1, ENa, EK;
	real   Ikv, Ik1, Inak, Ibna, Iion;

	real vm = g_dev.vm[i2d];
	real r  = g_dev.r[i2d];
	real s  = g_dev.s[i2d];

	/*------------------------------------------------------------------------
	* setting local variables
	*------------------------------------------------------------------------
	*/
	
	real fv = g_devF.vm[i2d];

	/* gating variables */
	r_inf = 1/(1+exp(-(vm)/11));
	tau_r = 20.3 + 138 * exp( -sqr((vm+20)/25.9) );
	s_inf = 1/(1+exp((vm+3)/7));
	tau_s = 1574 + 5268 * exp( -sqr((vm+23)/22.7) );
	
	/* I_Kv */
	EK = R*T/F * log(Ko/Ki);
	Ikv = gkv * r * s * (vm-EK);
	
	/* I_K1 */
	aK1 = 0.1/(1+exp(0.06*(vm-EK-200)));
	bK1 = ( 3*exp(0.0002*(vm-EK+100)) + exp(0.1*(vm-EK-10)) )
	    / ( 1+exp(-0.5*(vm-EK)) );
	Ik1 = gK1 * aK1/(aK1+bK1) * (vm-EK);
	
	/* I_NaK */
	Inak = INaKbar / (1+KmK/Ko) / (1+pow((KmNa/Nai),1.5)) * (vm-Vrev) / (vm-B);
	
	/* I_bNa */
	ENa =  R*T/F * log(Nao/Nai);
	Ibna = GbNa * (vm-ENa);

	/* I_ion */
	Iion = Ikv + Ik1 + Inak + Ibna; 
	
	/* differential equations */
	fv -= Cm*Iion;
	g_devF.r[i2d]  = (r_inf - r) / tau_r;
	g_devF.s[i2d]  = (s_inf - s) / tau_s;


	g_devF.vm[i2d] = fv;
}