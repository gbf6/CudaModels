#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include "../common/CudaSafeCall.h"

#include "MaleckarBacNavglobalVariables.cuh"
#include "typedefSparse.h"
#include "sparsePrototypes.cuh"
#include "typedefMaleckar.h"

#include "parseInput.h"

#include "MaleckarBacNavhostPrototypes.h"
#include "MaleckarBacNavdevicePrototypes.cuh"

real MaleckarBacNav_RestVoltage = MaleckarBacNav_RestVoltage_0;
__device__ real gkv = gkv_0;
__device__ real shiftrs = shiftrs_0;
__device__ real GbNa = GbNa_0;
__device__ real GNA = GNA_0;


void Maleckar_init(char** res) {
	rword resources[] = {
	{ "MaleckarBacNav_Node",    1100 },
	{ "MaleckarBacNav_Nodetype",1100 },
	{ "MaleckarBacNav_Type",    1100 },
	{ "MaleckarBacNav_patch",   1102 },
	{ "MaleckarBacNav_Vr",	1007 },
	{ "MaleckarBacNav_Vrest",	1007 },
	{ "MaleckarBacNav_gkv",	1008 },
	{ "MaleckarBacNav_shiftrs",	1009 },
	{ "MaleckarBacNav_gbna",	1010 },
	{ "MaleckarBacNav_gna",         1011 },
	{ NULL, 0 }
	};

	int i, j, c;
	int cmd;
	real temp;

	i = 0;
	while( res[i] != NULL ) {
		cmd = FindCommand( resources, res[i] );
		switch( cmd ) {
			case 1007:
				MaleckarBacNav_RestVoltage = GetRealValue( res[i] );
				break;
			case 1008:
				temp = GetRealValue(res[i]);
				cudaMemcpyToSymbol(gkv, (void *)&temp, sizeof(real), 0, cudaMemcpyHostToDevice);
				break;
			case 1009:
				temp = GetRealValue(res[i]);
				cudaMemcpyToSymbol(shiftrs, (void *)&temp, sizeof(real), 0, cudaMemcpyHostToDevice);
				break;
			case 1010:
				temp = GetRealValue(res[i]);
				cudaMemcpyToSymbol(GbNa, (void *)&temp, sizeof(real), 0, cudaMemcpyHostToDevice);
				break;
			case 1011:
				temp = GetRealValue(res[i]);
				cudaMemcpyToSymbol(GNA, (void *)&temp, sizeof(real), 0, cudaMemcpyHostToDevice);
				break;
			case 1100:
				//MaleckarBacNav_NodeType = GetByteValue( res[i] );
				break;
			case 1102:
				/*iv = GetRealArray( res[i] );
				p  = (real*)(&MaleckarBacNav_RestPatch);
				c  = GetNumValues( res[i] );
				if( c > MaleckarBacNav_PatchSize ) {
					c = MaleckarBacNav_PatchSize;
				}
				for(j=0;j<c;j++) {
					p[j] = iv[j];
				}
				break;*/
		}
		i++;
	}
}

void Maleckar_gateinit(int memSize, size_t* pitch, gateType* gate_h, gateType* gate_dev, gateType* gate_devF) {

	cudaHostAlloc((void**)&(gate_h->vm), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->r), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->s), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->m), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->h), memSize, 0);
	

	// Allocate device memory arrays
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->vm, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->r, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->s, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->m, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->h, pitch,
		memSize, 1));

	// Allocate device forward memory arrays
	CudaSafeCall(cudaMallocPitch((void **)&gate_devF->vm, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_devF->r, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_devF->s, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_devF->m, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_devF->h, pitch,
		memSize, 1));

	puts("\nFinished allocating device arrays\n");

	int totpoints = (int)memSize / sizeof(real);

	for (int idx = 0; idx < totpoints; idx++) {
		gate_h->vm[idx] = MaleckarBacNav_RestVoltage;
		gate_h->r[idx] = 5.454e-02;
		gate_h->s[idx] = 9.814e-01;
		gate_h->m[idx] = 0.00012037;
		gate_h->h[idx] = .4769;
	}

	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->vm, *pitch, (void *)gate_h->vm,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->r, *pitch, (void *)gate_h->r,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->s, *pitch, (void *)gate_h->s,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->m, *pitch, (void *)gate_h->m,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->h, *pitch, (void *)gate_h->h,
		memSize, memSize, 1, cudaMemcpyHostToDevice));


	CudaSafeCall(cudaMemcpy2D((void *)gate_devF->vm, *pitch, (void *)gate_h->vm,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_devF->r, *pitch, (void *)gate_h->r,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_devF->s, *pitch, (void *)gate_h->s,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_devF->m, *pitch, (void *)gate_h->m,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_devF->h, *pitch, (void *)gate_h->h,
		memSize, memSize, 1, cudaMemcpyHostToDevice));

	real** qpH = (real**)malloc(sizeof(real *)*gate_h->qpl);
	int i = 0;
	qpH[i++] = gate_devF->r;
	qpH[i++] = gate_devF->s;
	qpH[i++] = gate_devF->m;
	qpH[i++] = gate_devF->h;


	CudaSafeCall(cudaMemcpy((void *)gate_devF->qp, (void*)qpH, sizeof(real *)*gate_h->qpl, cudaMemcpyHostToDevice));

	i = 0;
	qpH[i++] = gate_dev->r;
	qpH[i++] = gate_dev->s;
	qpH[i++] = gate_dev->m;
	qpH[i++] = gate_dev->h;

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
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->m, memSize, (void *)gate_dev->m,
		pitch, memSize, 1, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->h, memSize, (void *)gate_dev->h,
		pitch, memSize, 1, cudaMemcpyDeviceToHost));
}

void Maleckar_exit(int memSize, size_t pitch, gateType* gate_h, gateType* gate_dev, gateType* gate_devF, sparse* MatrixINT, cudasparse* cudaMatrixINT){
	// Free gate host and device memory
	cudaFreeHost(gate_h->vm); cudaFreeHost(gate_h->r); cudaFreeHost(gate_h->s); cudaFreeHost(gate_h->m); cudaFreeHost(gate_h->h); cudaFree(gate_dev->qp); 

	cudaFree(gate_dev->vm); cudaFree(gate_dev->r); cudaFree(gate_dev->s); cudaFree(gate_dev->m); cudaFree(gate_dev->h); cudaFree(gate_dev->qp); 

	cudaFree(gate_devF->vm); cudaFree(gate_devF->r); cudaFree(gate_devF->s); cudaFree(gate_devF->m); cudaFree(gate_devF->h); cudaFree(gate_devF->qp); 

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
	real   r, s, m, h;
	real   r_inf, s_inf, tau_r, tau_s, minf, hinf, taum, tauh, aK1, bK1, ENa, EK;
	real   Ikv, Ik1, Inak, Ibna, Iion, INaBacNav;

	real vm = g_dev.vm[i2d];
	real r  = g_dev.r[i2d];
	real s  = g_dev.s[i2d];
	real m  = g_dev.m[i2d];
	real h  = g_dev.h[i2d];

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
    minf = (1.0/(1.0+exp((vm+28.34)/(-5.33))));
	hinf = (1.0-1.0/(1.0+exp((-77.21-vm)/8.32)));
	taum = (96.37/(exp((vm+82.74)/17.64) + exp(-(vm+6.008)/3.337)) + .4844);
	tauh = (96.17 - (96.17 - 10.45)/(1.0+exp((-23.26-vm)/2.529)));

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
	
	/* INaBacNav */
	INaBacNav = GNA*m*m*m*h*(vm-ENa);	

	/* I_ion */
	Iion = Ikv + Ik1 + Inak + INaBacNav + Ibna; 
	
	/* differential equations */
	fv -= Cm*Iion;
	g_devF.r[i2d]  = (r_inf - r) / tau_r;
	g_devF.s[i2d]  = (s_inf - s) / tau_s;
	g_devF.m[i2d]  = (m_inf - m) / tau_m;
	g_devF.h[i2d]  = (h_inf - h) / tau_h;


	g_devF.vm[i2d] = fv;
}