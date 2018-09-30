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

#include "parseInput.h"

#include "LRDhostPrototypes.h"
#include "LRDdevicePrototypes.cuh"

real LRDBacNav_RestVoltage = LRDBacNav_RestVoltage_0;
__device__ real LRDBacNav_cm = LRD_cm_0;
__device__ real LRDBacNav_Gna = LRD_Gna_0;
__device__ real LRDBacNav_Gtca = LRD_Gtca_0;
__device__ real LRDBacNav_Gkp = LRD_Gkp_0;
__device__ real LRDBacNav_Gitodv = LRD_Gitodv_0;
__device__ real LRDBacNav_Gcab = LRD_Gcab_0;
__device__ real LRDBacNav_Gnab = LRD_Gnab_0;
__device__ real LRDBacNav_ito = LRD_ito_0;
__device__ real LRDBacNav_ikna = LRD_ikna_0;
__device__ real LRDBacNav_ikatp = LRD_ikatp_0;
__device__ real LRDBacNav_insna = LRD_insna_0;
__device__ real LRDBacNav_insk = LRD_insk_0;
__device__ real LRDBacNav_cleft = LRD_cleft_0;
__device__ real BacNav_Gna = BacNav_Gna_0;


void LRD_init(char** res) {
	rword resources[] = {
	{ "LRDBacNav_IV",	  1007 },
	{ "LRDBacNav_Node",	  1100 },
	{ "LRDBacNav_Nodetype", 1100 },
	{ "LRDBacNav_Patch",	  1007 },
	{ "LRDBacNav_Type",	  1100 },
	{ "LRDBacNav_Vr",	      1008 },
	{ "LRDBacNav_Vrest",	  1008 },
	{ "LRDBacNav_Cm",       1009 },
    { "LRDBacNav_Gna",      1112 },
    { "LRDBacNav_Gtca",     1113 },
    { "LRDBacNav_Gkp",      1114 },
    { "LRDBacNav_Gitodv",   1115 },
    { "LRDBacNav_Gcab",     1116 },
    { "LRDBacNav_Gnab",     1117 },
    { "LRDBacNav_ito",      1118 },
    { "LRDBacNav_ikna",     1119 },
    { "LRDBacNav_ikatp",    1120 },
    { "LRDBacNav_insna",    1121 }, 
    { "LRDBacNav_insk",     1122 },
    { "LRDBacNav_cleft",    1123 },
    { "LRDBacNav_BacNavFactor", 1124},
	{ NULL, 0 }
	};

	int i, j, c, r;
	int cmd;
	real temp;

	temp = GetRealValue(res[i]);
	cudaMemcpyToSymbol(LRD_cm, (void *)&temp, sizeof(real), 0, cudaMemcpyHostToDevice);
			

	i = 0;
	while( res[i] != NULL ) {
		cmd = FindCommand( resources, res[i] );
		switch( cmd ) {
			case 1007:
				/*iv = GetRealArray( res[i] );
				p = (real*)(&LRDBacNav_RestPatch);
				c  = GetNumValues( res[i] );
				if( c > LRDBacNav_PatchSize ) {
					c = LRDBacNav_PatchSize;
				}
				for(j=0;j<c;j++) {
					p[j] = iv[j];
				}*/
				break;
			case 1008:
				LRDBacNav_RestVoltage = GetRealValue( res[i] );
				break;
            case 1009:
				temp = GetRealValue(res[i]);
				cudaMemcpyToSymbol(LRDBacNav_cm, (void *)&temp, sizeof(real), 0, cudaMemcpyHostToDevice);
                break; 
			case 1100:
				//LRDBacNav_NodeType = GetByteValue( res[i] );
				break;
			case 1112:
				temp = GetRealValue(res[i]);
				cudaMemcpyToSymbol(LRDBacNav_Gna, (void *)&temp, sizeof(real), 0, cudaMemcpyHostToDevice);
				break;
			case 1113:
				temp = GetRealValue(res[i]);
				cudaMemcpyToSymbol(LRDBacNav_Gtca, (void *)&temp, sizeof(real), 0, cudaMemcpyHostToDevice);
				break;
			case 1114:
				temp = GetRealValue(res[i]);
				cudaMemcpyToSymbol(LRDBacNav_Gkp, (void *)&temp, sizeof(real), 0, cudaMemcpyHostToDevice);
				break;
			case 1115:
				temp = GetRealValue(res[i]);
				cudaMemcpyToSymbol(LRDBacNav_Gitodv, (void *)&temp, sizeof(real), 0, cudaMemcpyHostToDevice);
				break;
			case 1116:
				temp = GetRealValue(res[i]);
				cudaMemcpyToSymbol(LRDBacNav_Gcab, (void *)&temp, sizeof(real), 0, cudaMemcpyHostToDevice);
				break;
			case 1117:
				temp = GetRealValue(res[i]);
				cudaMemcpyToSymbol(LRDBacNav_Gnab, (void *)&temp, sizeof(real), 0, cudaMemcpyHostToDevice);
				break;
            case 1118:
				temp = GetRealValue(res[i]);
				cudaMemcpyToSymbol(LRDBacNav_ito, (void *)&temp, sizeof(real), 0, cudaMemcpyHostToDevice);
	            break;
            case 1119:
				temp = GetRealValue(res[i]);
				cudaMemcpyToSymbol(LRDBacNav_ikna, (void *)&temp, sizeof(real), 0, cudaMemcpyHostToDevice);
                break;
            case 1120: 
				temp = GetRealValue(res[i]);
				cudaMemcpyToSymbol(LRDBacNav_ikatp, (void *)&temp, sizeof(real), 0, cudaMemcpyHostToDevice);
                break;
            case 1121:
				temp = GetRealValue(res[i]);
				cudaMemcpyToSymbol(LRDBacNav_insna, (void *)&temp, sizeof(real), 0, cudaMemcpyHostToDevice);
                break;
            case 1122:
				temp = GetRealValue(res[i]);
				cudaMemcpyToSymbol(LRDBacNav_insk, (void *)&temp, sizeof(real), 0, cudaMemcpyHostToDevice);
                break;
            case 1123:
				temp = GetRealValue(res[i]);
				cudaMemcpyToSymbol(LRDBacNav_cleft, (void *)&temp, sizeof(real), 0, cudaMemcpyHostToDevice);
                break;
		    case 1124:
				temp = GetRealValue(res[i]);
				cudaMemcpyToSymbol(BacNav_Gna, (void *)&temp, sizeof(real), 0, cudaMemcpyHostToDevice);
		}
		i++;
	}
}

void LRD_gateinit(int memSize, size_t* pitch, gateType* gate_h, gateType* gate_dev, gateType* gate_devF) {

	cudaHostAlloc((void**)&(gate_h->vm), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->m), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->h), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->j), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->mb), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->hb), memSize, 0);
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
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->mb, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->hb, pitch,
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
	CudaSafeCall(cudaMallocPitch((void **)&gate_devF->mb, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_devF->hb, pitch,
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
		gate_h->mb[idx] = 0.000094;
		gate_h->hb[idx] = 0.8231;
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
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->mb, *pitch, (void *)gate_h->mb,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->hb, *pitch, (void *)gate_h->hb,
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
	CudaSafeCall(cudaMemcpy2D((void *)gate_devF->mb, *pitch, (void *)gate_h->mb,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_devF->hb, *pitch, (void *)gate_h->hb,
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
	qpH[i++] = gate_devF->mb;
	qpH[i++] = gate_devF->hb;
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
	qpH[i++] = gate_dev->mb;
	qpH[i++] = gate_dev->hb;
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
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->mb, *pitch, (void *)gate_dev->mb,
		memSize, memSize, 1, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->hb, *pitch, (void *)gate_dev->hb,
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
	cudaFreeHost(gate_h->mb);
	cudaFreeHost(gate_h->hb);
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
	cudaFree(gate_dev->mb);
	cudaFree(gate_dev->hb);
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
	cudaFree(gate_devF->mb);
	cudaFree(gate_devF->hb);
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
    real LRDBacNav_RTF;
    real LRDBacNav_Gkr, LRDBacNav_Gks,LRDBacNav_Gki,LRDBacNav_Gkatp;
    real LRDBacNav_Ena,LRDBacNav_Etca,LRDBacNav_Ekr,LRDBacNav_Eks,LRDBacNav_Eki,LRDBacNav_Ekp;
    real LRDBacNav_Ekna,LRDBacNav_Ekatp,LRDBacNav_Ekdv,LRDBacNav_Ecan,LRDBacNav_Enan;
    real m,h,j,aproto,aproto2,am,bm,ah,bh,aj,bj,mb,hb,taumb, tauhb, minfb, hinfb, Ina;  
    real d,f,dss,dss1,taud,fss,tauf,Ibarca,Ibarna,Ibark;
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
    LRDBacNav_RTF = LRDBacNav_R*LRDBacNav_temp/LRDBacNav_frdy;


	vm        = g_dev.vm[i2d];

	m 		  = g_dev.m[i2d];
	h 		  = g_dev.h[i2d];
	j 		  = g_dev.j[i2d];
	mb 		  = g_dev.mb[i2d];
	hb 		  = g_dev.hb[i2d];
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
	LRDBacNav_Gkr   = 0.02614*sqrt(ko/5.4); 
	LRDBacNav_Gks   = 0.433*(1+0.6/(1+pow((0.000038/cai),1.4)));
	LRDBacNav_Gki   = 0.75*(sqrt(ko/5.4));
	LRDBacNav_Gkatp = 0.000195/nicholsarea;

	LRDBacNav_Ena   = (LRDBacNav_RTF)*log(nao/nai);
	LRDBacNav_Etca  = 0.5*(LRDBacNav_RTF)*log(cao/cai);  
	LRDBacNav_Ekr   = (LRDBacNav_RTF)*log(ko/ki);
	LRDBacNav_Eks   = (LRDBacNav_RTF)*log((ko+prnak*nao)/(ki+prnak*nai));
	LRDBacNav_Eki   = (LRDBacNav_RTF)*log(ko/ki);
	LRDBacNav_Ekp   = LRDBacNav_Eki;
	LRDBacNav_Ekna  = LRDBacNav_Ekr; 
	LRDBacNav_Ekatp = LRDBacNav_Ekr; 
	LRDBacNav_Ekdv  = LRDBacNav_Ekr;
	LRDBacNav_Ecan  = LRDBacNav_Etca;
	LRDBacNav_Enan  = LRDBacNav_Ena;

	/* Na current [15] */
	aproto  = 1-1.0/(1+exp(-(vm+40)/0.024));
	am = 0.32*(vm+47.13)/(1-exp(-0.1*(vm+47.13)));
	bm = 0.08*exp(-vm/11);
	ah = aproto*0.135*exp((80+vm)/-6.8);
	bh = (1-aproto)/(0.13*(1+exp((vm+10.66)/(-11.1)))) + aproto*(3.56*exp(0.079*vm)+3.1*pow(10,5)*exp(0.35*vm));
	aj = aproto*(-127140*exp(0.2444*vm)-0.00003474*exp(-0.04391*vm))*((vm+37.78)/(1+exp(0.311*(vm+79.23))));
	bj = (1-aproto)*(0.3*exp(-2.535*pow(10,-7)*vm)/(1+exp(-0.1*(vm+32))))+aproto*(0.1212*exp(-0.01052*vm))/(1+exp(-0.1378*(vm+40.14)));        
	/* BacNav component */
	minfb = (1.0/(1.0+exp((vm+28.34)/(-5.33))));
	hinfb = (1.0-1.0/(1.0+exp((-77.21-vm)/8.32)));
	taumb = (86.37/(exp((vm+82.74)/17.64) + exp(-(vm+ 6.008)/3.337)) + .4844);
	tauhb = (96.17-(96.17-10.45)/(1.0+exp((-23.26-vm)/2.529)));
	Ina=(LRDBacNav_Gna*(m*m*m*h*j)+BacNav_Gna*(mb*mb*mb*hb))*(vm-LRDBacNav_Ena);

	/* L-type Calcium current [14,15] */
	dss  = 1/(1+exp(-(vm+10)/6.24));
	dss1 = 1/(1+exp(-(vm+60)/0.024));
	taud = dss*(1-exp(-(vm+10)/6.24))/(0.035*(vm+10));
	dss = dss * dss1;
	fss  = (1/(1+exp((vm+32)/8)))+(0.6/(1+exp((50-vm)/20)));
	tauf = 1/(0.0197*exp(-0.0337*0.0337*(vm+10)*(vm+10))+0.02);

	Ibarca = pca*zca*zca*((vm*LRDBacNav_frdy)/(LRDBacNav_RTF))*((gacai*cai*exp((zca*vm)/(LRDBacNav_RTF))-gacao*cao)/(exp((zca*vm)/(LRDBacNav_RTF))-1));
	Ibarna = pna*zna*zna*((vm*LRDBacNav_frdy)/(LRDBacNav_RTF))*((ganai*nai*exp((zna*vm)/(LRDBacNav_RTF))-ganao*nao)/(exp((zna*vm)/(LRDBacNav_RTF))-1));
	Ibark  = pk*zk*zk*((vm*LRDBacNav_frdy)/(LRDBacNav_RTF))*((gaki*ki*exp((zk*vm)/(LRDBacNav_RTF))-gako*ko)/(exp((zk*vm)/(LRDBacNav_RTF))-1));

	fca = 1/(1+cai/kmca);

	Ilca    = d*f*fca*Ibarca;
	Ilcana  = d*f*fca*Ibarna;
	Ilcak   = d*f*fca*Ibark;	
	Ilcatot = Ilca+Ilcana+Ilcak;

	/* T-type Calcium current [13] */
	bss  = 1/(1+exp(-(vm+14)/10.8));
	taub = 3.7+6.1/(1+exp((vm+25)/4.5));
	gss  = 1/(1+exp((vm+60)/5.6));
	aproto2 = 1-1/(1+exp(-vm/0.0024));
	taug = aproto2*(-0.875*vm+12.0)+12.0*(1-aproto2);
	Itca = LRDBacNav_Gtca*b*b*g*(vm-LRDBacNav_Etca);

	/* K current - Rapid [13] */
	xrss  = 1/(1+exp(-(vm+21.5)/7.5));
	tauxr = 1/(0.00138*(vm+14.2)/(1-exp(-0.123*(vm+14.2)))+0.00061*(vm+38.9)/(exp(0.145*(vm+38.9))-1));

	r = 1/(1+exp((vm+9)/22.4));

	Ikr = LRDBacNav_Gkr*xr*r*(vm-LRDBacNav_Ekr);

	/* K current - Slow [10,13] */
	xs1ss  = 1/(1+exp(-(vm-1.5)/16.7));
	xs2ss  = xs1ss;
	tauxs1 = 1/(0.0000719*(vm+30)/(1-exp(-0.148*(vm+30)))+0.000131*(vm+30)/(exp(0.0687*(vm+30))-1));
	tauxs2 = 4*tauxs1;

	Iks = LRDBacNav_Gks*xs1*xs2*(vm-LRDBacNav_Eks);

	/* K current -  Time independent [15] */
	aki = 1.02/(1+exp(0.2385*(vm-LRDBacNav_Eki-59.215)));
	bki = (0.49124*exp(0.08032*(vm-LRDBacNav_Eki+5.476))+exp(0.06175*(vm-LRDBacNav_Eki-594.31)))/(1+exp(-0.5143*(vm-LRDBacNav_Eki+4.753)));	

	kin = aki/(aki+bki);

	Ikti = LRDBacNav_Gki*kin*(vm-LRDBacNav_Eki);

	/* K current - Plateau [15] */
	kp  = 1/(1+exp((7.488-vm)/5.98));	

	Ikp = LRDBacNav_Gkp*kp*(vm-LRDBacNav_Ekp);

	/* Na-Ca exchanger [6,14,15] */
	Inaca = c1*exp((gammas-1)*vm/(LRDBacNav_RTF))*((exp(vm/(LRDBacNav_RTF))*nai*nai*nai*cao-nao*nao*nao*cai)/(1+c2*exp((gammas-1)*vm/(LRDBacNav_RTF))*(exp(vm/(LRDBacNav_RTF))*nai*nai*nai*cao+nao*nao*nao*cai)));

	/* Na-K pump [15] */
	sigma = (exp(nao/67.3)-1)/7;
	fnak  = 1/(1+0.1245*exp((-0.1*vm)/(LRDBacNav_RTF))+0.0365*sigma*exp((-vm)/(LRDBacNav_RTF)));

	Inak = Ibarnak*fnak*(1/(1+kmnai*kmnai/(nai*nai)))*(ko/(ko+kmko));

	/* Sarcolemmal Ca pump [15] */
	Ipca = (Ibarpca*cai)/(kmpca+cai);

	/* Ca background current [15] */
	Icab = LRDBacNav_Gcab*(vm-LRDBacNav_Ecan);

	/* Na background current [15] */
	Inab = LRDBacNav_Gnab*(vm-LRDBacNav_Enan);          

	/* Na activated K current [6] */
	pona = 0.85/(1+pow((kdkna/nai),2.8));
	pov  = 0.8-(0.65/(1+exp((vm+125)/15)));

	Ikna = LRDBacNav_ikna*LRDBacNav_Gkna*pona*pov*(vm-LRDBacNav_Ekna);

	/* ATP sensitive K current [11] */
	patp = 1/(1+(pow((atpi/katp),hatp)));
	gkbaratp = LRDBacNav_Gkatp*patp*(pow((ko/4),natp));

	Ikatp = LRDBacNav_ikatp*gkbaratp*(vm-LRDBacNav_Ekatp);     

	/* Non-specific Ca-activated current [14,15] */
	Ibarnsna = pnsca*zna*zna*((vm*LRDBacNav_frdy)/(LRDBacNav_RTF))*((ganai*nai*exp((zna*vm)/(LRDBacNav_RTF))-ganao*nao)/(exp((zna*vm)/(LRDBacNav_RTF))-1));
	Ibarnsk  = pnsca*zk*zk*((vm*LRDBacNav_frdy)/(LRDBacNav_RTF))*((gaki*ki*exp((zk*vm)/(LRDBacNav_RTF))-gako*ko)/(exp((zk*vm)/(LRDBacNav_RTF))-1));

	Insna = LRDBacNav_insna*Ibarnsna/(1+kmnsca*kmnsca*kmnsca/(cai*cai*cai)); 
	Insk = LRDBacNav_insk*Ibarnsk/(1+kmnsca*kmnsca*kmnsca/(cai*cai*cai));   

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

	Ito = LRDBacNav_ito*LRDBacNav_Gitodv*zdv*zdv*zdv*ydv*rvdv*(vm-LRDBacNav_Ekdv);

	/* Summing currents (inactive currents are set to zero with activation variables) */
	naiont = Ina+Inab+Ilcana+3*Inak+3*Inaca+Insna;
	kiont  = Ikr+Iks+Ikti+Ikp+Ilcak+-2*Inak+Insk+Ito+Ikna+Ikatp;  
	caiont = Ilca+Icab+Ipca-2*Inaca+Itca;

	Itotal = LRDBacNav_cm*(naiont+kiont+caiont);     /* uA/cm2 */

	if (((t-tcicr)>80) && (vm<-30)) {
		BOOL = 0;  
		g_dev.BOOL[i2d] = BOOL;                                                                  
	}

	/* Put voltage update here  */
	fv += -Itotal;
	g_devF.vm[i2d] = fv;

	/* change in cleft concentration */
	dnao = LRDBacNav_cleft*((nabm-nao)/taudiff+naiont*acap*LRDBacNav_cm/(vcleft*LRDBacNav_frdy));
	dko = LRDBacNav_cleft*((kbm-ko)/taudiff+kiont*acap*LRDBacNav_cm/(vcleft*LRDBacNav_frdy));
	dcao = LRDBacNav_cleft*((cabm-cao)/taudiff+caiont*acap*LRDBacNav_cm/(vcleft*LRDBacNav_frdy*2));

	/* change in nai and ki concentration */
	dnai = -LRDBacNav_cm*(naiont*acap)/(vmyo*zna*LRDBacNav_frdy);     /* dnai/dt */
	dki = -LRDBacNav_cm*(kiont*acap)/(vmyo*zk*LRDBacNav_frdy);        /* dki/dt */

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
	dcai = -dt*(((LRDBacNav_cm*caiont*acap)/(vmyo*zca*LRDBacNav_frdy))+((iup-ileak)*vnsr/vmyo)-(irelcicr*vjsr/vmyo)-(ireljsrol*vjsr/vmyo));

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
    g_devF.mb[i2d]   = (minfb - mb)/taumb;
	g_devF.hb[i2d]   = (hinfb - hb)/tauhb;
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