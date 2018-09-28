#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include "../common/CudaSafeCall.h"

#include "TNNPglobalVariables.cuh"
#include "typedefSparse.h"
#include "sparsePrototypes.cuh"
#include "typedefTNNP.h"

#include "TNNPhostPrototypes.h"
#include "TNNPdevicePrototypes.cuh"

void TNNP_gateinit(int memSize, size_t* pitch, gateType* gate_h, gateType* gate_dev, gateType* gate_devF) {

	cudaHostAlloc((void**)&(gate_h->vm), memSize, 0);

	cudaHostAlloc((void**)&(gate_h->Cai), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->CaSR), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->Nai), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->Ki), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->m), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->h), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->j), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->xr1), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->xr2), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->xs), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->r), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->s), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->d), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->f), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->fca), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->g), memSize, 0);
	

	// Allocate device memory arrays
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->vm, pitch,
		memSize, 1));

	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->Cai, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->CaSR, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->Nai, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->Ki, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->m, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->h, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->j, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->xr1, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->xr2, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->xs, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->r, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->s, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->d, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->f, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->fca, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->g, pitch,
		memSize, 1));

	// Allocate device forward memory arrays
	CudaSafeCall(cudaMallocPitch((void **)&gate_devF->vm, pitch,
		memSize, 1));

	CudaSafeCall(cudaMallocPitch((void **)&gate_devF->Cai, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_devF->CaSR, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_devF->Nai, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_devF->Ki, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_devF->m, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_devF->h, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_devF->j, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_devF->xr1, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_devF->xr2, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_devF->xs, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_devF->r, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_devF->s, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_devF->d, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_devF->f, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_devF->fca, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_devF->g, pitch,
		memSize, 1));

	puts("\nFinished allocating device arrays\n");

	int totpoints = (int)memSize / sizeof(real);

	for (int idx = 0; idx < totpoints; idx++) {
		gate_h->vm[idx] = TNNP_RestVoltage;

		gate_h->Cai[idx] = 0.0002;
		gate_h->CaSR[idx] = 0.2;
		gate_h->Nai[idx] = 11.6;
		gate_h->Ki[idx] = 138.3;
		gate_h->m[idx] = 0.0;
		gate_h->h[idx] = 0.75;
		gate_h->j[idx] = 0.75;
		gate_h->xr1[idx] = 0.0;
		gate_h->xr2[idx] = 1.0;
		gate_h->xs[idx] = 0.0;
		gate_h->r[idx] = 0.0;
		gate_h->s[idx] = 1.0;
		gate_h->d[idx] = 0.0;
		gate_h->f[idx] = 1.0;
		gate_h->fca[idx] = 1.0;
		gate_h->g)[idx] = 1.0;
	}

	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->vm, *pitch, (void *)gate_h->vm,
		memSize, memSize, 1, cudaMemcpyHostToDevice));

	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->Cai, *pitch, (void *)gate_h->Cai,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->CaSR, *pitch, (void *)gate_h->CaSR,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->Nai, *pitch, (void *)gate_h->Nai,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->Ki, *pitch, (void *)gate_h->Ki,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->m, *pitch, (void *)gate_h->m,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->h, *pitch, (void *)gate_h->h,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->j, *pitch, (void *)gate_h->j,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->xr1, *pitch, (void *)gate_h->xr1,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->xr2, *pitch, (void *)gate_h->xr2,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->xs, *pitch, (void *)gate_h->xs,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->r, *pitch, (void *)gate_h->r,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->s, *pitch, (void *)gate_h->s,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->d, *pitch, (void *)gate_h->d,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->,f *pitch, (void *)gate_h->f,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->fca, *pitch, (void *)gate_h->fca,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->g, *pitch, (void *)gate_h->g,
		memSize, memSize, 1, cudaMemcpyHostToDevice));


	CudaSafeCall(cudaMemcpy2D((void *)gate_devF->vm, *pitch, (void *)gate_h->vm,
		memSize, memSize, 1, cudaMemcpyHostToDevice));

	CudaSafeCall(cudaMemcpy2D((void *)gate_devF->Cai, *pitch, (void *)gate_h->Cai,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_devF->CaSR, *pitch, (void *)gate_h->CaSR,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_devF->Nai, *pitch, (void *)gate_h->Nai,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_devF->Ki, *pitch, (void *)gate_h->Ki,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_devF->m, *pitch, (void *)gate_h->m,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_devF->h, *pitch, (void *)gate_h->h,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_devF->j, *pitch, (void *)gate_h->j,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_devF->xr1, *pitch, (void *)gate_h->xr1,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_devF->xr2, *pitch, (void *)gate_h->xr2,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_devF->xs, *pitch, (void *)gate_h->xs,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_devF->r, *pitch, (void *)gate_h->r,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_devF->s, *pitch, (void *)gate_h->s,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_devF->d, *pitch, (void *)gate_h->d,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_devF->,f *pitch, (void *)gate_h->f,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_devF->fca, *pitch, (void *)gate_h->fca,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_devF->g, *pitch, (void *)gate_h->g,
		memSize, memSize, 1, cudaMemcpyHostToDevice));

	real** qpH = (real**)malloc(sizeof(real *)*gate_h->qpl);
	int i = 0;
	qpH[i++] = gate_devF->Cai;
	qpH[i++] = gate_devF->CaSR;
	qpH[i++] = gate_devF->Nai;
	qpH[i++] = gate_devF->Ki;
	qpH[i++] = gate_devF->m;
	qpH[i++] = gate_devF->h;
	qpH[i++] = gate_devF->j;
	qpH[i++] = gate_devF->xr1;
	qpH[i++] = gate_devF->xr2;
	qpH[i++] = gate_devF->xs;
	qpH[i++] = gate_devF->r;
	qpH[i++] = gate_devF->s;
	qpH[i++] = gate_devF->d;
	qpH[i++] = gate_devF->f;
	qpH[i++] = gate_devF->fca;
	qpH[i++] = gate_devF->g;

	CudaSafeCall(cudaMemcpy((void *)gate_devF->qp, (void*)qpH, sizeof(real *)*gate_h->qpl, cudaMemcpyHostToDevice));

	i = 0;
	qpH[i++] = gate_dev->Cai;
	qpH[i++] = gate_dev->CaSR;
	qpH[i++] = gate_dev->Nai;
	qpH[i++] = gate_dev->Ki;
	qpH[i++] = gate_dev->m;
	qpH[i++] = gate_dev->h;
	qpH[i++] = gate_dev->j;
	qpH[i++] = gate_dev->xr1;
	qpH[i++] = gate_dev->xr2;
	qpH[i++] = gate_dev->xs;
	qpH[i++] = gate_dev->r;
	qpH[i++] = gate_dev->s;
	qpH[i++] = gate_dev->d;
	qpH[i++] = gate_dev->f;
	qpH[i++] = gate_dev->fca;
	qpH[i++] = gate_dev->g;

	CudaSafeCall(cudaMemcpy((void *)gate_dev->qp, (void*)qpH, sizeof(real *)*gate_h->qpl, cudaMemcpyHostToDevice));
	

	CudaCheckError();

	puts("\nFinished initializing device arrays\n");

}

void TNNP_sync(int memSize, size_t pitch, gateType* gate_h, gateType* gate_dev) {

	CudaSafeCall(cudaMemcpy2D((void *)gate_h->vm, *pitch, (void *)gate_dev->vm,
		memSize, memSize, 1, cudaMemcpyDeviceToHost));

	CudaSafeCall(cudaMemcpy2D((void *)gate_h->Cai, *pitch, (void *)gate_dev->Cai,
		memSize, memSize, 1, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->CaSR, *pitch, (void *)gate_dev->CaSR,
		memSize, memSize, 1, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->Nai, *pitch, (void *)gate_dev->Nai,
		memSize, memSize, 1, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->Ki, *pitch, (void *)gate_dev->Ki,
		memSize, memSize, 1, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->m, *pitch, (void *)gate_dev->m,
		memSize, memSize, 1, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->h, *pitch, (void *)gate_dev->h,
		memSize, memSize, 1, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->j, *pitch, (void *)gate_dev->j,
		memSize, memSize, 1, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->xr1, *pitch, (void *)gate_dev->xr1,
		memSize, memSize, 1, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->xr2, *pitch, (void *)gate_dev->xr2,
		memSize, memSize, 1, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->xs, *pitch, (void *)gate_dev->xs,
		memSize, memSize, 1, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->r, *pitch, (void *)gate_dev->r,
		memSize, memSize, 1, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->s, *pitch, (void *)gate_dev->s,
		memSize, memSize, 1, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->d, *pitch, (void *)gate_dev->d,
		memSize, memSize, 1, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->,f *pitch, (void *)gate_dev->f,
		memSize, memSize, 1, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->fca, *pitch, (void *)gate_dev->fca,
		memSize, memSize, 1, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->g, *pitch, (void *)gate_dev->g,
		memSize, memSize, 1, cudaMemcpyDeviceToHost));
}

void TNNP_exit(int memSize, size_t pitch, gateType* gate_h, gateType* gate_dev, gateType* gate_devF, sparse* MatrixINT, cudasparse* cudaMatrixINT){
	// Free gate host and device memory
	cudaFreeHost(gate_h->vm); cudaFreeHost(gate_h->Cai); cudaFreeHost(gate_h->CaSR); cudaFreeHost(gate_h->Nai); cudaFreeHost(gate_h->Ki); cudaFreeHost(gate_h->m); cudaFreeHost(gate_h->h); cudaFreeHost(gate_h->j); cudaFreeHost(gate_h->xr1); cudaFreeHost(gate_h->xr2); cudaFreeHost(gate_h->xs); cudaFreeHost(gate_h->r); cudaFreeHost(gate_h->s); cudaFreeHost(gate_h->d); cudaFreeHost(gate_h->f); cudaFreeHost(gate_h->fca); cudaFreeHost(gate_h->g); cudaFreeHost(gate_h->qp);

	cudaFree(gate_dev->vm); cudaFree(gate_dev->Cai); cudaFree(gate_dev->CaSR); cudaFree(gate_dev->Nai); cudaFree(gate_dev->Ki); cudaFree(gate_dev->m); cudaFree(gate_dev->h); cudaFree(gate_dev->j); cudaFree(gate_dev->xr1); cudaFree(gate_dev->xr2); cudaFree(gate_dev->xs); cudaFree(gate_dev->r); cudaFree(gate_dev->s); cudaFree(gate_dev->d); cudaFree(gate_dev->f); cudaFree(gate_dev->fca); cudaFree(gate_dev->g); cudaFree(gate_dev->qp);

	cudaFree(gate_devF->vm); cudaFree(gate_devF->Cai); cudaFree(gate_devF->CaSR); cudaFree(gate_devF->Nai); cudaFree(gate_devF->Ki); cudaFree(gate_devF->m); cudaFree(gate_devF->h); cudaFree(gate_devF->j); cudaFree(gate_devF->xr1); cudaFree(gate_devF->xr2); cudaFree(gate_devF->xs); cudaFree(gate_devF->r); cudaFree(gate_devF->s); cudaFree(gate_devF->d); cudaFree(gate_devF->f); cudaFree(gate_devF->fca); cudaFree(gate_devF->g); cudaFree(gate_devF->qp);

	cudaFree(cudaMatrixINT->type);
	cudaFree(cudaMatrixINT->rows);
	cudaFree(cudaMatrixINT->maxnz);
	cudaFree(cudaMatrixINT->csep);
	cudaFree(cudaMatrixINT->jcoef);
	cudaFree(cudaMatrixINT->coef);
}


void __device__ GetFDev_TNNP(int i2d, int pitch, real beta, real Cm, real t, real dt, int totpoints, real rx, gateType g_dev, gateType g_devF) {

	/*------------------------------------------------------------------------
	* return if outside domain
	*------------------------------------------------------------------------
	*/
	if (i2d >= totpoints) {
		return;
	}

	real V,Ek,Ena,Eks,Eca;
	real Iion,INa,ICaL,Ito,IKr,IKs,IK1,INaCa,INaK,IpCa,IpK,IbNa,IbCa;
	real Nai,Ki,m,h,j,xr1,xr2,xs,r,s,d,f,fca,g;
	real Caisquare,Cai,CaSRsquare,CaCurrent,A,Irel,Ileak,SERCA,CaSRCurrent;
	real CaSR,CaCSQN,dCaSR,bjsr,cjsr,CaBuf,dCai,bc,cc;
	real AM,BM,TAU_M,M_INF,AH_1,BH_1,TAU_H,AH_2,BH_2,H_INF,AJ_1,BJ_1;
	real TAU_J,AJ_2,BJ_2,J_INF,Xr1_INF,axr1,bxr1,TAU_Xr1,Xr2_INF,axr2,bxr2;
	real TAU_Xr2,Xs_INF,Axs,Bxs,TAU_Xs,R_INF,S_INF,TAU_R,TAU_S,D_INF,Ad,Bd,Cd;
	real TAU_D,F_INF,TAU_F,FCa_INF,G_INF,fcaold,gold;
	real rec_iK1,rec_ipK,rec_iNaK,Ak1,Bk1;
	real CaiNew,CaSRNew,fcatemp,gtemp;

	real V    = g_dev.vm[i2d];
	real Cai  = g_dev.Cai[i2d];
	real CaSR = g_dev.CaSR[i2d];
	real Nai  = g_dev.Nai[i2d];
	real Ki   = g_dev.Ki[i2d];
	real m    = g_dev.m[i2d];
	real h    = g_dev.h[i2d];
	real j    = g_dev.j[i2d];
	real xr1  = g_dev.xr1[i2d];
	real xr2  = g_dev.xr2[i2d];
	real xs   = g_dev.xs[i2d];
	real r    = g_dev.r[i2d];
	real s    = g_dev.s[i2d];
	real d    = g_dev.d[i2d];
	real f    = g_dev.f[i2d];
	real fca  = g_dev.fca[i2d];
	real g    = g_dev.g[i2d];

	/*------------------------------------------------------------------------
	* setting local variables
	*------------------------------------------------------------------------
	*/
	real fv = g_devF.vm[i2d];


	Ek=RTONF*(log((Ko/Ki)));
	Ena=RTONF*(log((Nao/Nai)));
	Eks=RTONF*(log((Ko+pKNa*Nao)/(Ki+pKNa*Nai)));
	Eca=0.5*RTONF*(log((Cao/Cai)));
	Ak1=0.1/(1.+exp(0.06*(V-Ek-200)));
	Bk1=(3.*exp(0.0002*(V-Ek+100))+exp(0.1*(V-Ek-10)))/(1.+exp(-0.5*(V-Ek)));
	rec_iK1=Ak1/(Ak1+Bk1);
	rec_iNaK=(1./(1.+0.1245*exp(-0.1*V*F/(R*T))+0.0353*exp(-V*F/(R*T))));
	rec_ipK=1./(1.+exp((25-V)/5.98));


	INa=TNNP_GNa*m*m*m*h*j*(V-Ena);
	ICaL=TNNP_GCaL*d*f*fca*4*V*(F*F/(R*T))*
	   (exp(2*V*F/(R*T))*Cai-0.341*Cao)/(exp(2*V*F/(R*T))-1.);
	Ito=TNNP_Gto*r*s*(V-Ek);
	IKr=TNNP_Gkr*sqrt(Ko/5.4)*xr1*xr2*(V-Ek);
	IKs=TNNP_Gks*xs*xs*(V-Eks);
	IK1=TNNP_GK1*rec_iK1*(V-Ek);
	INaCa=knaca*(1./(KmNai*KmNai*KmNai+Nao*Nao*Nao))*(1./(KmCa+Cao))*
	      (1./(1+ksat*exp((n-1)*V*F/(R*T))))*
	      (exp(n*V*F/(R*T))*Nai*Nai*Nai*Cao-
	       exp((n-1)*V*F/(R*T))*Nao*Nao*Nao*Cai*2.5);
	INaK=knak*(Ko/(Ko+KmK))*(Nai/(Nai+KmNa))*rec_iNaK;
	IpCa=TNNP_GpCa*Cai/(KpCa+Cai);
	IpK=TNNP_GpK*rec_ipK*(V-Ek);
	IbNa=TNNP_GbNa*(V-Ena);
	IbCa=TNNP_GbCa*(V-Eca);
	Iion = IKr+IKs+IK1+Ito+INa+IbNa+ICaL+IbCa+INaK+INaCa+IpCa+IpK;

	/* Update Auxilliary Variables */
    Caisquare=Cai*Cai;
    CaSRsquare=CaSR*CaSR;
    CaCurrent=-(ICaL+IbCa+IpCa-2*INaCa)*inverseVcF2*CAPACITANCE;
    A=0.016464*CaSRsquare/(0.0625+CaSRsquare)+0.008232;
    Irel=A*d*g;
    Ileak=0.00008*(CaSR-Cai);
    SERCA=Vmaxup/(1.+(Kupsquare/Caisquare));
    CaSRCurrent=SERCA-Irel-Ileak;
    CaCSQN=Bufsr*CaSR/(CaSR+Kbufsr);
    dCaSR=dt*(Vc/Vsr)*CaSRCurrent;
    bjsr=Bufsr-CaCSQN-dCaSR-CaSR+Kbufsr;
    cjsr=Kbufsr*(CaCSQN+dCaSR+CaSR);
    CaSRNew=(sqrt(bjsr*bjsr+4*cjsr)-bjsr)/2;
    CaBuf=Bufc*Cai/(Cai+Kbufc);
    dCai=dt*(CaCurrent-CaSRCurrent);
    bc=Bufc-CaBuf-dCai-Cai+Kbufc;
    cc=Kbufc*(CaBuf+dCai+Cai);
    CaiNew=(sqrt(bc*bc+4*cc)-bc)/2;

   	AM=1./(1.+exp((-60.-V)/5.));
    BM=0.1/(1.+exp((V+35.)/5.))+0.10/(1.+exp((V-50.)/200.));
    TAU_M=AM*BM;
    M_INF=1./((1.+exp((-56.86-V)/9.03))*(1.+exp((-56.86-V)/9.03)));
    if (V>=-40.){
		AH_1=0.; 
		BH_1=(0.77/(0.13*(1.+exp(-(V+10.66)/11.1))));
		TAU_H= 1.0/(AH_1+BH_1);
    }
    else{
		AH_2=(0.057*exp(-(V+80.)/6.8));
		BH_2=(2.7*exp(0.079*V)+(3.1e5)*exp(0.3485*V));
		TAU_H=1.0/(AH_2+BH_2);
    }
    H_INF=1./((1.+exp((V+71.55)/7.43))*(1.+exp((V+71.55)/7.43)));
    if(V>=-40.){
		AJ_1=0.;      
		BJ_1=(0.6*exp((0.057)*V)/(1.+exp(-0.1*(V+32.))));
		TAU_J= 1.0/(AJ_1+BJ_1);
    }
    else{
		AJ_2=(((-2.5428e4)*exp(0.2444*V)-(6.948e-6)*
		exp(-0.04391*V))*(V+37.78)/
		   (1.+exp(0.311*(V+79.23))));    
		BJ_2=(0.02424*exp(-0.01052*V)/(1.+exp(-0.1378*(V+40.14))));
		TAU_J= 1.0/(AJ_2+BJ_2);
    }
    J_INF=H_INF;

    Xr1_INF=1./(1.+exp((-26.-V)/7.));
    axr1=450./(1.+exp((-45.-V)/10.));
    bxr1=6./(1.+exp((V-(-30.))/11.5));
    TAU_Xr1=axr1*bxr1;
    Xr2_INF=1./(1.+exp((V-(-88.))/24.));
    axr2=3./(1.+exp((-60.-V)/20.));
    bxr2=1.12/(1.+exp((V-60.)/20.));
    TAU_Xr2=axr2*bxr2;

    Xs_INF=1./(1.+exp((-5.-V)/14.));
    Axs=1100./(sqrt(1.+exp((-10.-V)/6)));
    Bxs=1./(1.+exp((V-60.)/20.));
    TAU_Xs=Axs*Bxs;
    
    R_INF=1./(1.+exp((20-V)/6.));
    S_INF=1./(1.+exp((V+20)/5.));
    TAU_R=9.5*exp(-(V+40.)*(V+40.)/1800.)+0.8;
    TAU_S=85.*exp(-(V+45.)*(V+45.)/320.)+5./(1.+exp((V-20.)/5.))+3.;

	D_INF=1./(1.+exp((-5-V)/7.5));
	Ad=1.4/(1.+exp((-35-V)/13))+0.25;
	Bd=1.4/(1.+exp((V+5)/5));
	Cd=1./(1.+exp((50-V)/20));
	TAU_D=Ad*Bd+Cd;
	F_INF=1./(1.+exp((V+20)/7));
	TAU_F=1125*exp(-(V+27)*(V+27)/240)+80+165/(1.+exp((25-V)/10));

	FCa_INF=(1./(1.+pow((CaiNew/0.000325),8))+
	 0.1/(1.+exp((CaiNew-0.0005)/0.0001))+
	 0.20/(1.+exp((CaiNew-0.00075)/0.0008))+
	 0.23 )/1.46;
	if(CaiNew<0.00035) G_INF=1./(1.+pow((CaiNew/0.00035),6));
	else G_INF=1./(1.+pow((CaiNew/0.00035),16));


	fv -= Iion;
	g_devF.vm[i2d] = fv;

	g_devF.Cai[i2d] = (CaiNew-Cai)/dt;
	g_devF.CaSR[i2d] = (CaSRNew-CaSR)/dt;
	g_devF.Nai[i2d] = -(INa+IbNa+3*INaK+3*INaCa)*inverseVcF*CAPACITANCE;
	g_devF.Ki[i2d] =  -(IK1+Ito+IKr+IKs-2*INaK+IpK)*inverseVcF*CAPACITANCE;
	g_devF.m[i2d] = (M_INF-(M_INF-m)*exp(-dt/TAU_M)-m)/dt;
	g_devF.h[i2d] = (H_INF-(H_INF-h)*exp(-dt/TAU_H)-h)/dt;
	g_devF.j[i2d] = (J_INF-(J_INF-j)*exp(-dt/TAU_J)-j)/dt;
	g_devF.xr1[i2d] = (Xr1_INF-(Xr1_INF-xr1)*exp(-dt/TAU_Xr1)-xr1)/dt;
	g_devF.xr2[i2d] = (Xr2_INF-(Xr2_INF-xr2)*exp(-dt/TAU_Xr2)-xr2)/dt;
	g_devF.xs[i2d] = (Xs_INF-(Xs_INF-xs)*exp(-dt/TAU_Xs)-xs)/dt;
	g_devF.s[i2d] = (S_INF-(S_INF-s)*exp(-dt/TAU_S)-s)/dt;
	g_devF.r[i2d] = (R_INF-(R_INF-r)*exp(-dt/TAU_R)-r)/dt;
	g_devF.d[i2d] = (D_INF-(D_INF-d)*exp(-dt/TAU_D)-d)/dt;
	g_devF.f[i2d] = (F_INF-(F_INF-f)*exp(-dt/TAU_F)-f)/dt;

	fcaold = fca;
	fcatemp = FCa_INF-(FCa_INF-fca)*exp(-dt/taufca);
    if(fcatemp>fcaold && (V)>-60){
		g_devF.fca[i2d] =(fcaold-fca)/dt;
   	} 
   	else{
		g_devF.fca[i2d] =(fcatemp-fca)/dt;
	}


	gold = g;
	gtemp = G_INF-(G_INF-g)*exp(-dt/taug);
    if(gtemp>gold && (V)>-60){
   		g_devF.g[i2d] = (gold-g)/dt;
    }
    else{
		g_devF.g[i2d]= (gtemp-g)/dt;
	}
}