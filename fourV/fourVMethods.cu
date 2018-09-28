#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include "../common/CudaSafeCall.h"

#include "fourVglobalVariables.cuh"
#include "typedefSparse.h"
#include "sparsePrototypes.cuh"
#include "typedeffourV.h"

#include "fourVhostPrototypes.h"
#include "fourVdevicePrototypes.cuh"

void fourV_gateinit(int memSize, size_t* pitch, gateType* gate_h, gateType* gate_dev, gateType* gate_devF) {

	cudaHostAlloc((void**)&(gate_h->u), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->v), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->w), memSize, 0);
	cudaHostAlloc((void**)&(gate_h->s), memSize, 0);

	// Allocate device memory arrays
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->u, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->v, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->w, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_dev->s, pitch,
		memSize, 1));

	// Allocate device forward memory arrays
	CudaSafeCall(cudaMallocPitch((void **)&gate_devF->u, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_devF->v, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_devF->w, pitch,
		memSize, 1));
	CudaSafeCall(cudaMallocPitch((void **)&gate_devF->s, pitch,
		memSize, 1));

	puts("\nFinished allocating device arrays\n");

	int totpoints = (int)memSize / sizeof(real);

	for (int idx = 0; idx < totpoints; idx++) {
		gate_h->u[idx] = 0.0;
		gate_h->v[idx] = 1.0;
		gate_h->w[idx] = 1.0;
		gate_h->s[idx] = 0.0;
	}

	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->u, *pitch, (void *)gate_h->u,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->v, *pitch, (void *)gate_h->v,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->w, *pitch, (void *)gate_h->w,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_dev->s, *pitch, (void *)gate_h->s,
		memSize, memSize, 1, cudaMemcpyHostToDevice));

	CudaSafeCall(cudaMemcpy2D((void *)gate_devF->u, *pitch, (void *)gate_h->u,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_devF->v, *pitch, (void *)gate_h->v,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_devF->w, *pitch, (void *)gate_h->w,
		memSize, memSize, 1, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy2D((void *)gate_devF->s, *pitch, (void *)gate_h->s,
		memSize, memSize, 1, cudaMemcpyHostToDevice));

	real** qpH = (real**)malloc(sizeof(real *)*gate_h->qpl);
	int i = 0;
	qpH[i++] = gate_devF->v;
	qpH[i++] = gate_devF->w;
	qpH[i++] = gate_devF->s;

	CudaSafeCall(cudaMemcpy((void *)gate_devF->qp, (void*)qpH, sizeof(real *)*gate_h->qpl, cudaMemcpyHostToDevice));

	i = 0;
	qpH[i++] = gate_dev->v;
	qpH[i++] = gate_dev->w;
	qpH[i++] = gate_dev->s;

	CudaSafeCall(cudaMemcpy((void *)gate_dev->qp, (void*)qpH, sizeof(real *)*gate_h->qpl, cudaMemcpyHostToDevice));

	gate_h->vm = (real*)malloc(sizeof(real*));
	gate_h->vm = gate_h->u;

	real* point;

	CudaSafeCall(cudaMalloc((void **)&gate_dev->vm, sizeof(gate_dev->v)));
	point = gate_dev->v;
	CudaSafeCall(cudaMemcpy((void *)gate_dev->vm, (void *)point, sizeof(real *), cudaMemcpyHostToDevice));

	CudaSafeCall(cudaMalloc((void **)&gate_devF->vm, sizeof(gate_devF->v)));
	point = gate_devF->v;
	CudaSafeCall(cudaMemcpy((void *)gate_devF->vm, (void *)point, sizeof(real *), cudaMemcpyHostToDevice));
	

	CudaCheckError();

	puts("\nFinished initializing device arrays\n");

}

void fourV_sync(int memSize, size_t pitch, gateType* gate_h, gateType* gate_dev) {
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->u, memSize, (void *)gate_dev->u,
		pitch, memSize, 1, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->v, memSize, (void *)gate_dev->v,
		pitch, memSize, 1, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->w, memSize, (void *)gate_dev->w,
		pitch, memSize, 1, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy2D((void *)gate_h->s, memSize, (void *)gate_dev->s,
		pitch, memSize, 1, cudaMemcpyDeviceToHost));
}

void fourV_exit(int memSize, size_t pitch, gateType* gate_h, gateType* gate_dev, gateType* gate_devF, sparse* MatrixINT, cudasparse* cudaMatrixINT){
	// Free gate host and device memory
	cudaFreeHost(gate_h->u); cudaFreeHost(gate_h->v); cudaFreeHost(gate_h->w); cudaFreeHost(gate_h->s);
	cudaFreeHost(gate_h->vm);
	
	cudaFree(gate_dev->u); cudaFree(gate_dev->v); cudaFree(gate_dev->w); cudaFree(gate_dev->s);
	cudaFree(gate_dev->vm); cudaFree(gate_h->qp);

	cudaFree(gate_devF->u); cudaFree(gate_devF->v); cudaFree(gate_devF->w); cudaFree(gate_devF->s);
	cudaFree(gate_devF->vm); cudaFree(gate_h->qp);

	cudaFree(cudaMatrixINT->type);
	cudaFree(cudaMatrixINT->rows);
	cudaFree(cudaMatrixINT->maxnz);
	cudaFree(cudaMatrixINT->csep);
	cudaFree(cudaMatrixINT->jcoef);
	cudaFree(cudaMatrixINT->coef);
}

/*========================================================================
* 2D : 4-Variable Model Time Integrator
*========================================================================
*/

void __device__ GetFDev_fourV(int i2d, int pitch, real beta, real Cm, real t, real dt, int totpoints, real rx, gateType g_dev, gateType g_devF) {

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
	real u = g_dev.u[i2d];
	real v = g_dev.v[i2d];
	real w = g_dev.w[i2d];
	real s = g_dev.s[i2d];

	real fu = g_devF.u[i2d];

	/*------------------------------------------------------------------------
	* Additional heaviside functions
	*------------------------------------------------------------------------
	*/
	real  H_theta_v = (u > theta_v) ? 1.0 : 0.0;
	real  H_theta_m_v = (u > theta_m_v) ? 1.0 : 0.0;
	real  H_theta_w = (u > theta_w) ? 1.0 : 0.0;
	real  H_theta_so = (u > theta_so) ? 1.0 : 0.0;
	real  H_theta_si = (u > theta_si) ? 1.0 : 0.0;
	real  H_theta_s = (u > theta_s) ? 1.0 : 0.0;
	real  H_theta_o = (u > theta_o) ? 1.0 : 0.0;
	real  H_theta_vinf = (u > theta_vinf) ? 1.0 : 0.0;
	real  H_theta_winf = (u > theta_winf) ? 1.0 : 0.0;
	/*------------------------------------------------------------------------
	* Calculating dependant tau's
	*------------------------------------------------------------------------
	*/
	real tau_m_v = (1.0 - H_theta_m_v)*tau_m_v1
		+ H_theta_m_v * tau_m_v2;
	real tau_m_w = tau_m_w1
		+ (tau_m_w2 - tau_m_w1)*(1. + tanh(k_m_w*(u - u_m_w)))*0.5;
	real  tau_p_w = tau_p_w1
		+ (tau_p_w2 - tau_p_w1)*(1. + tanh(k_p_w*(
			delta_w*(w - w_p_c) + (1. - delta_w)*(u - u_p_w))))*0.5;
	real tau_s = (1. - H_theta_s)*tau_s1 + H_theta_s * tau_s2;
	real tau_o = (1. - H_theta_o)*tau_o1 + H_theta_o * tau_o2;
	real tau_so = tau_so1
		+ (tau_so2 - tau_so1)*(1. + tanh(k_so*(u - u_so)))*0.5;
	real  tau_si = tau_si1
		+ (tau_si2 - tau_si1)*(1. + tanh(k_si*(s - s_c)))*0.5;
	real  tau_p_si = alpha_si * (1. + exp(k_si1*(u - theta_p_si))) /
		(1. - tanh(k_si2*(u - theta_p_si)));

	real v_inf = 1. - H_theta_vinf;
	real w_inf = (1. - H_theta_winf)*(1. - u / tau_winf)
		+ H_theta_winf * w_sinf;

	/*------------------------------------------------------------------------
	* v
	*------------------------------------------------------------------------
	*/
	real  dv2dt = (1. - H_theta_v)*(v_inf - v) / tau_m_v
		- H_theta_v * v / tau_p_v;
	v += dv2dt * dt;
	g_devF.v[i2d] = v;

	/*------------------------------------------------------------------------
	* w
	*------------------------------------------------------------------------
	*/
	real  wx = (2. - alpha_w)*(3. - alpha_w)*(4. - alpha_w)*w / 6.0
		+ (alpha_w - 1.)*(3. - alpha_w)*(4. - alpha_w)*0.5*w*w
		+ (alpha_w - 1.)*(alpha_w - 2.)*(4. - alpha_w)*0.5*w*w*w
		+ (alpha_w - 1.)*(alpha_w - 2.)*(alpha_w - 3.)*w*w*w*w / 6.;

	real dw2dt = (1. - H_theta_w)*(w_inf - wx) / tau_m_w
		- H_theta_w * w / tau_p_w;
	w += dw2dt * dt;
	g_devF.w[i2d] = w;

	/*------------------------------------------------------------------------
	* s
	*------------------------------------------------------------------------
	*/
	real   ds2dt = ((1. + tanh(k_s*(u - u_s)))*0.5 - s) / tau_s;

	s += ds2dt * dt;
	g_devF.s[i2d] = s;

	/*------------------------------------------------------------------------
	* I_sum
	*------------------------------------------------------------------------
	*/
	real  J_fi = -v * H_theta_v*(u - theta_p_v)*(u_u - u) / tau_fi;
	real  J_so = (u - u_o)*(1. - H_theta_so)*(1. - beta_v * v) / tau_o
		+ H_theta_so / tau_so;
	real  J_si;
	if (gamma_si > 0.5)
		J_si = -H_theta_si * w*s / tau_si;
	else
		J_si = -(1. + tanh(k_si_c*(u - theta_si_c)))*w / tau_p_si;

	real  I_sum = J_fi + J_so + J_si;


	/*------------------------------------------------------------------------
	* Time integration
	*------------------------------------------------------------------------
	*/
	fu -= dt * I_sum / Cm;

	g_devF.u[i2d] = u;
}