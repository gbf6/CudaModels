#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../common/CudaSafeCall.h"

#include "CRNglobalVariables.cuh"
#include "typedefSparse.h"
#include "sparsePrototypes.cuh"
#include "typedefCRN.h"

#include "CRNhostPrototypes.h"
#include "CRNdevicePrototypes.cuh"

void __device__ stimSimple(real t, int i2d, gateType* g_dev, real* fv);

void CRN_gateinit(int memSize, size_t* pitch, gateType* gate_h, gateType* gate_dev) {

	cudaHostAlloc((void**)&(gate_h->vm), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->vmp), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->fv), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->m), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->h), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->j), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->sa), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->si), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->ua), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->ui), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->xr), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->xs), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->d), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->f), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->fca), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->u), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->v), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->w), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->Nai), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->Ki), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->Cai), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->Caup), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->Carel), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->start), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->finish), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->strength), memSize, 0);



	// Allocate device memory arrays
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->vm, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->vmp, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->fv, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->m, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->h, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->j, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->sa, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->si, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->ua, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->ui, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->xr, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->xs, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->d, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->f, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->fca, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->u, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->v, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->w, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->Nai, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->Ki, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->Cai, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->Caup, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->Carel, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->start, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->finish, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->strength, pitch,
		memSize, 1));

	puts("\nFinished allocating device arrays\n");

	int totpoints = (int)memSize / sizeof(real);

	for (int idx = 0; idx < totpoints; idx++) {
		gate_h->vm[idx]       = -81.2;
		gate_h->vmp[idx]	  = -81.2;
		gate_h->fv[idx]       = 0;
		gate_h->m[idx]        = 2.91e-3;
		gate_h->h[idx]        = 0.965;
		gate_h->j[idx]        = 0.978;
		gate_h->sa[idx]       = 3.04e-2;
		gate_h->si[idx]       = 0.999;
		gate_h->ua[idx]       = 4.96e-3;
		gate_h->ui[idx]       = 0.999;
		gate_h->xr[idx]       = 3.29e-5;
		gate_h->xs[idx]       = 1.87e-2;
		gate_h->d[idx]        = 1.37e-4;
		gate_h->f[idx]        = 0.999;
		gate_h->fca[idx]      = 0.775;
		gate_h->u[idx]        = 0.0;
		gate_h->v[idx]        = 1.0;
		gate_h->w[idx]        = 0.999;
		gate_h->Nai[idx]      = 11.2;
		gate_h->Ki[idx]       = 139.0;
		gate_h->Cai[idx]      = 1.02e-4;
		gate_h->Caup[idx]     = 1.49;
		gate_h->Carel[idx]    = 1.49;
		gate_h->start[idx]    = -1;
		gate_h->finish[idx]   = -1;
		gate_h->strength[idx] = 0;
	}

	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->vm, *pitch, (void *)gate_h->vm,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->vmp, *pitch, (void *)gate_h->vmp,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->fv, *pitch, (void *)gate_h->fv,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->m, *pitch, (void *)gate_h->m,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->h, *pitch, (void *)gate_h->h,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->j, *pitch, (void *)gate_h->j,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->sa, *pitch, (void *)gate_h->sa,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->si, *pitch, (void *)gate_h->si,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->ua, *pitch, (void *)gate_h->ua,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->ui, *pitch, (void *)gate_h->ui,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->xr, *pitch, (void *)gate_h->xr,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->xs, *pitch, (void *)gate_h->xs,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->d, *pitch, (void *)gate_h->d,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->f, *pitch, (void *)gate_h->f,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->fca, *pitch, (void *)gate_h->fca,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->u, *pitch, (void *)gate_h->u,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->v, *pitch, (void *)gate_h->v,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->w, *pitch, (void *)gate_h->w,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->Nai, *pitch, (void *)gate_h->Nai,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->Ki, *pitch, (void *)gate_h->Ki,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->Cai, *pitch, (void *)gate_h->Cai,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->Caup, *pitch, (void *)gate_h->Caup,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->Carel, *pitch, (void *)gate_h->Carel,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	
	CudaCheckError();

	puts("\nFinished initializing device arrays\n");

}

void CRN_sync(int memSize, size_t pitch, gateType* gate_h, gateType* gate_dev) {
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->vm, memSize, (void *)gate_dev->vm,
		pitch, memSize, 1, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->vmp, memSize, (void *)gate_dev->vmp,
		pitch, memSize, 1, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->fv, memSize, (void *)gate_dev->fv,
		pitch, memSize, 1, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->m, memSize, (void *)gate_dev->m,
		pitch, memSize, 1, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->h, memSize, (void *)gate_dev->h,
		pitch, memSize, 1, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->j, memSize, (void *)gate_dev->j,
		pitch, memSize, 1, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->sa, memSize, (void *)gate_dev->sa,
		pitch, memSize, 1, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->si, memSize, (void *)gate_dev->si,
		pitch, memSize, 1, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->ua, memSize, (void *)gate_dev->ua,
		pitch, memSize, 1, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->ui, memSize, (void *)gate_dev->ui,
		pitch, memSize, 1, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->xr, memSize, (void *)gate_dev->xr,
		pitch, memSize, 1, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->xs, memSize, (void *)gate_dev->xs,
		pitch, memSize, 1, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->d, memSize, (void *)gate_dev->d,
		pitch, memSize, 1, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->f, memSize, (void *)gate_dev->f,
		pitch, memSize, 1, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->fca, memSize, (void *)gate_dev->fca,
		pitch, memSize, 1, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->u, memSize, (void *)gate_dev->u,
		pitch, memSize, 1, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->v, memSize, (void *)gate_dev->v,
		pitch, memSize, 1, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->w, memSize, (void *)gate_dev->w,
		pitch, memSize, 1, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->Nai, memSize, (void *)gate_dev->Nai,
		pitch, memSize, 1, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->Ki, memSize, (void *)gate_dev->Ki,
		pitch, memSize, 1, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->Cai, memSize, (void *)gate_dev->Cai,
		pitch, memSize, 1, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->Caup, memSize, (void *)gate_dev->Caup,
		pitch, memSize, 1, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->Carel, memSize, (void *)gate_dev->Carel,
		pitch, memSize, 1, cudaMemcpyDeviceToHost));
}

void CRN_exit(int memSize, size_t pitch, gateType* gate_h, gateType* gate_dev, sparse* MatrixINT, cudasparse* cudaMatrixINT) {
	CRN_sync(memSize, pitch, gate_h, gate_dev);

	// Free gate host and device memory
	cudaFreeHost(gate_h->vm); cudaFreeHost(gate_h->m); cudaFreeHost(gate_h->j); cudaFreeHost(gate_h->sa);
	cudaFreeHost(gate_h->si); cudaFreeHost(gate_h->ua); cudaFreeHost(gate_h->ui); cudaFreeHost(gate_h->xr);
	cudaFreeHost(gate_h->xs); cudaFreeHost(gate_h->d); cudaFreeHost(gate_h->f); cudaFreeHost(gate_h->fca);
	cudaFreeHost(gate_h->u); cudaFreeHost(gate_h->v); cudaFreeHost(gate_h->w); cudaFreeHost(gate_h->Nai);
	cudaFreeHost(gate_h->Ki); cudaFreeHost(gate_h->Cai); cudaFreeHost(gate_h->Caup); cudaFreeHost(gate_h->Carel);
	cudaFreeHost(gate_h->start); cudaFreeHost(gate_h->finish); cudaFreeHost(gate_h->strength);
	cudaFreeHost(gate_h->vmp); cudaFreeHost(gate_h->fv);

	cudaFree(gate_dev->vm); cudaFree(gate_dev->m); cudaFree(gate_dev->j); cudaFree(gate_dev->sa);
	cudaFree(gate_dev->si); cudaFree(gate_dev->ua); cudaFree(gate_dev->ui); cudaFree(gate_dev->xr);
	cudaFree(gate_dev->xs); cudaFree(gate_dev->d); cudaFree(gate_dev->f); cudaFree(gate_dev->fca);
	cudaFree(gate_dev->u); cudaFree(gate_dev->v); cudaFree(gate_dev->w); cudaFree(gate_dev->Nai);
	cudaFree(gate_dev->Ki); cudaFree(gate_dev->Cai); cudaFree(gate_dev->Caup); cudaFree(gate_h->Carel);
	cudaFree(gate_dev->start); cudaFree(gate_dev->finish); cudaFree(gate_dev->strength);
	cudaFree(gate_dev->vmp); cudaFree(gate_dev->fv);
	cudaFree(cudaMatrixINT->type);
	cudaFree(cudaMatrixINT->rows);
	cudaFree(cudaMatrixINT->maxnz);
	cudaFree(cudaMatrixINT->csep);
	cudaFree(cudaMatrixINT->jcoef);
	cudaFree(cudaMatrixINT->coef);

}

 void __global__ CRN_Kernel(int pitch, real beta, real Cm, real t, real dt, int totpoints, real rx, gateType g_dev, cudasparse cudaMatrixINT) {

	 const int i2d = blockIdx.x*BLOCK_DIM_1D + threadIdx.x;


	/*------------------------------------------------------------------------
	* return if outside domain
	*------------------------------------------------------------------------
	*/
	if (i2d >= totpoints) {
		//printf("Aborting");
		return;
	}

	/*------------------------------------------------------------------------
	* setting local variables
	*------------------------------------------------------------------------
	*/

	real vm    = g_dev.vm[i2d];
	real vmp   = g_dev.vmp[i2d];
	real fv	   = g_dev.fv[i2d];
	real m     = g_dev.m[i2d];
	real h     = g_dev.h[i2d];
	real j     = g_dev.j[i2d];
	real sa    = g_dev.sa[i2d];
	real si    = g_dev.si[i2d];
	real ua    = g_dev.ua[i2d];
	real ui    = g_dev.ui[i2d];
	real xr    = g_dev.xr[i2d];
	real xs    = g_dev.xs[i2d];
	real d     = g_dev.d[i2d];
	real f     = g_dev.f[i2d];
	real fca   = g_dev.fca[i2d];
	real u     = g_dev.u[i2d];
	real v     = g_dev.v[i2d];
	real w     = g_dev.w[i2d];
	real Nai   = g_dev.Nai[i2d];
	real Ki    = g_dev.Ki[i2d];
	real Cai   = g_dev.Cai[i2d];
	real Caup  = g_dev.Caup[i2d];
	real Carel = g_dev.Carel[i2d];

	/*------------------------------------------------------------------------
	* Laplacian Calculation
	*
	* No flux boundary condition is applied on all boundaries through
	* the Laplacian operator definition
	*------------------------------------------------------------------------
	*/
	//int printv = 10200;

	fv = 0;
	//sprDiff(&fv, cudaMatrixINT, i2d, totpoints, g_dev.vmp, beta);
	for (int i = 0; i < *cudaMatrixINT.maxnz; i++) {
		fv += (1.0/beta)*cudaMatrixINT.coef[i2d + i * totpoints] * g_dev.vmp[cudaMatrixINT.jcoef[i2d + i * totpoints] - 1];
		//if (i2d == printv) printf("%d: %.31f, %.31f, %.31f, %.31f\n", cudaMatrixINT.jcoef[i2d + i * totpoints], cudaMatrixINT.coef[i2d + i * totpoints], g_dev.vmp[cudaMatrixINT.jcoef[i2d + i * totpoints] - 1], (1.0 / beta)*cudaMatrixINT.coef[i2d + i * totpoints] * g_dev.vmp[cudaMatrixINT.jcoef[i2d + i * totpoints] - 1], fv);
	}

	//if (i2d == printv) printf("t: %.31f\ndiff: %.31f\n", t, fv);

	/*------------------------------------------------------------------------
	* Currents
	*------------------------------------------------------------------------
	*/

	real Ena = (R*T / F)*log(Nao / Nai);
	real Ek = (R*T / F)*log(Ko / Ki);
	real Eca = (R*T / (2.0*F))*log(Cao / Cai);
	real gkur = 1.0e-4*(0.005 + 0.05 / (1.0 + exp(-1.0*(vm - 15.0) / 13.0)));
	real sigma = (1.0 / 7.0)*(exp(Nao / 67.3) - 1.0);
	real fnak = 1.0 / (1.0 + 0.1245*exp(-0.1*F*vm / (R*T)) + 0.0365*sigma*exp(-1.0*F*vm / (R*T)));

	real Ina = gna * m*m*m*h*j*(vm - Ena);
	real Ik1 = gk1 * (vm - Ek) / (1.0 + exp(0.07*(vm + 80.0)));
	real Ito = gto * sa*sa*sa*si*(vm - Ek);
	real Ikur = gkur * ua*ua*ua*ui*(vm - Ek);
	real Ikr = gkr * xr*(vm - Ek) / (1.0 + exp((vm + 15.0) / 22.4));
	real Iks = gks * xs*xs*(vm - Ek);
	real Ical = gcal * d*f*fca*(vm - 65.0);
	real Inak = Inakmax * fnak*(1.0 / (1.0 + pow(Kmnai / Nai, 1.5))) * Ko / (Ko + Kmko);
	real Inaca = Inacamax * (exp(lambda*F*vm / (R*T))*Nai*Nai*Nai*Cao -
		exp((lambda - 1)*F*vm / (R*T))*Nao*Nao*Nao*Cai) /
		((Kmna*Kmna*Kmna + Nao * Nao*Nao)*(Kmca + Cao)*(1.0 + ksat * exp((lambda - 1)*F*vm / (R*T))));
	real Ibca = gbca * (vm - Eca);
	real Ibna = gbna * (vm - Ena);
	real Ipca = Ipcamax * Cai / (0.0005 + Cai);

	real Irel = krel * u*u*v*w*(Carel - Cai);
	real Itr = (Caup - Carel) / tautr;
	real Iup = Iupmax / (1.0 + (kup / Cai));
	real Iupleak = Caup * Iupmax / Caupmax;

	real Iion = Ina + Ik1 + Ito + Ikur + Ikr + Iks + Ical + Ipca + Inak + Inaca + Ibna + Ibca;
	Iion /= (pi*cellDiameter*cellLength*1.0e-8);
	
	/*------------------------------------------------------------------------
	* Gating variables
	*------------------------------------------------------------------------
	*/

	real am;
	if (vm == -47.13) {
		am = 3.2;
	}
	else {
		am = 0.32*(vm + 47.13) / (1.0 - exp(-0.1*(vm + 47.13)));
	}
	real bm = 0.08*exp(-vm / 11.0);
	real taum = 1.0 / (am + bm);
	real infm = am * taum;

	real ah, bh, aj, bj;
	if (vm >= -40.0) {
		ah = 0.0;
		bh = 1.0 / (0.13*(1.0 + exp(-1.0*(vm + 10.66) / 11.1)));
		aj = 0.0;
		bj = 0.3*exp((-2.535e-7)*vm) / (1.0 + exp(-0.1*(vm + 32.0)));
	}
	else {
		ah = 0.135*exp(-1.0*(vm + 80.0) / 6.8);
		bh = 3.56*exp(0.079*vm) + (3.1e5)*exp(0.35*vm);
		aj = (-127140.0*exp(0.2444*vm) - (3.474e-5)*exp(-0.04391*vm))*(vm + 37.78) /
			(1.0 + exp(0.311*(vm + 79.23)));
		bj = 0.1212*exp(-0.01052*vm) / (1.0 + exp(-0.1378*(vm + 40.14)));
	}
	real tauh = 1.0 / (ah + bh);
	real infh = ah * tauh;
	real tauj = 1.0 / (aj + bj);
	real infj = aj * tauj;

	real asa = 0.65 / ((exp(-1.0*(vm + 10.0) / 8.5)) + (exp(-1.0*(vm - 30.0) / 59.0)));
	real bsa = 0.65 / (2.5 + exp((vm + 82.0) / 17.0));
	real infsa = 1.0 / (1.0 + exp(-1.0*(vm + 20.47) / 17.54));
	real tausa = (1.0 / (asa + bsa)) / Kq10;

	real asi = 1.0 / (18.53 + exp((vm + 113.7) / 10.95));
	real bsi = 1.0 / (35.56 + exp(-1.0*(vm + 1.26) / 7.44));
	real infsi = 1.0 / (1.0 + exp((vm + 43.1) / 5.3));
	real tausi = (1.0 / (asi + bsi)) / Kq10;

	real aua = 0.65*1.0 / (exp(-1.0*(vm + 10.0) / 8.5) + exp(-1.0*(vm - 30.0) / 59.0));
	real bua = 0.65*1.0 / (2.5 + exp((vm + 82.0) / 17.0));
	real infua = 1.0 / (1.0 + exp(-1.0*(vm + 30.3) / 9.6));
	real tauua = (1.0 / (aua + bua)) / Kq10;

	real aui = 1.0 / (21.0 + exp(-1.0*(vm - 185.0) / 28.0));
	real bui = exp((vm - 158.0) / 16.0);
	real infui = 1.0 / (1.0 + exp((vm - 99.45) / 27.48));
	real tauui = (1.0 / (aui + bui)) / Kq10;

	real axr = 0.0003*(vm + 14.1) / (1.0 - exp(-1.0*(vm + 14.1) / 5.0));
	real bxr = (7.3898e-5)*(vm - 3.3328) / (exp((vm - 3.3328) / 5.1237) - 1.0);
	real infxr = 1.0 / (1.0 + exp(-1.0*(vm + 14.1) / 6.5));
	real tauxr = (1.0 / (axr + bxr));

	real axs = (4.0e-5)*(vm - 19.9) / (1.0 - exp(-1.0*(vm - 19.9) / 17.0));
	real bxs = (3.5e-5)*(vm - 19.9) / (exp((vm - 19.9) / 9.0) - 1.0);
	real tauxs = 0.5 / (axs + bxs);
	real infxs = 1.0 / sqrt(1.0 + exp(-1.0*(vm - 19.9) / 12.7));

	real taud = (1.0 - exp(-1.0*(vm + 10.0) / 6.24)) / (0.035*(vm + 10.0)*(1.0 + exp(-1.0*(vm + 10.0) / 6.24)));
	real infd = 1.0 / (1.0 + exp(-1.0*(vm + 10.0) / 8.0));

	real tauf = 9.0 / (0.0197*exp(-1.0*0.0337*0.0337*(vm + 10.0)*(vm + 10.0)) + 0.02);
	real inff = 1.0 / (1.0 + exp((vm + 28.0) / 6.9));

	real inffca = 1.0 / (1.0 + Cai / 0.00035);

	real Fn = 1.0e-12*Vrel*Irel - (5.0e-7 / F)*(0.5*Ical - 0.2*Inaca);

	real infu = 1.0 / (1.0 + exp(-1.0*(Fn - 3.4175e-13) / 13.67e-16));

	real tauv = 1.91 + 2.09 / (1.0 + exp(-1.0*(Fn - 3.4175e-13) / 13.67e-16));
	real infv = 1.0 - 1.0 / (1.0 + exp(-1.0*(Fn - 6.835e-14) / 13.67e-16));

	real tauw = 6.0*(1.0 - exp(-1.0*(vm - 7.9) / 5.0)) / ((1.0 + 0.3*exp(-1.0*(vm - 7.9) / 5.0))*(vm - 7.9));
	real infw = 1.0 - 1.0 / (1.0 + exp(-1.0*(vm - 40.0) / 17.0));

	/*------------------------------------------------------------------------
	* Update variables
	*------------------------------------------------------------------------
	*/
	fv += -1.0*Iion;
	g_dev.fv[i2d] = fv;
	//if (i2d == printv) printf("eqns: %.31f\n", fv);

	if (t >= g_dev.start[i2d] && t <= g_dev.finish[i2d]) fv += g_dev.strength[i2d];

	g_dev.vm[i2d] += dt / Cm * fv;

	g_dev.Nai[i2d] += dt * ( 1.0e6*((-3.0*Inak - 3.0*Inaca - Ibna - Ina) / (F*Vi)) );
	g_dev.Ki[i2d] += dt * ( 1.0e6*((2.0*Inak - Ik1 - Ito - Ikur - Ikr - Iks) / (F*Vi)) );
	real B1 = 1.0e6*((2.0*Inaca - Ipca - Ical - Ibca) / (2.0*F*Vi)) +
		(Vup*(Iupleak - Iup) + Irel * Vrel) / Vi;
	real B2 = 1.0 + Trpnmax * KmTrpn / ((Cai + KmTrpn)*(Cai + KmTrpn)) + Cmdnmax * KmCmdn /
		((Cai + KmCmdn)*(Cai + KmCmdn));
	g_dev.Cai[i2d] += dt * ( B1 / B2 );
	g_dev.Caup[i2d] += dt * ( Iup - Iupleak - Itr * Vrel / Vup );
	g_dev.Carel[i2d] += dt * ( (Itr - Irel)*(1.0 / (1.0 + (Csqnmax*KmCsqn) /
		((Carel + KmCsqn)*(Carel + KmCsqn)))) );

	g_dev.m[i2d]   += dt * ( (infm - m) / taum );
	g_dev.h[i2d]   += dt * ( (infh - h) / tauh );
	g_dev.j[i2d]   += dt * ( (infj - j) / tauj );
	g_dev.sa[i2d]  += dt * ( (infsa - sa) / tausa );
	g_dev.si[i2d]  += dt * ( (infsi - si) / tausi );
	g_dev.ua[i2d]  += dt * ( (infua - ua) / tauua );
	g_dev.ui[i2d]  += dt * ( (infui - ui) / tauui );
	g_dev.xr[i2d]  += dt * ( (infxr - xr) / tauxr );
	g_dev.xs[i2d]  += dt * ( (infxs - xs) / tauxs );
	g_dev.d[i2d]   += dt * ( (infd - d) / taud );
	g_dev.f[i2d]   += dt * ( (inff - f) / tauf );
	g_dev.fca[i2d] += dt * ( (inffca - fca) / taufca );
	g_dev.u[i2d]   += dt * ( (infu - u) / tauu );
	g_dev.v[i2d]   += dt * ( (infv - v) / tauv );
	g_dev.w[i2d]   += dt * ( (infw - w) / tauw );

}


void __host__ CRN_print(int totpoints, gateType* gate_h, FILE* fp) {
	fwrite(gate_h->vm, sizeof(real), totpoints, fp);
	fflush(fp);	
}