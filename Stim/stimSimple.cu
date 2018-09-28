#include "typedefstimSimple.h"
#include "stimSimple.cuh"

void __global__ stimSimpleDev(stimulus stim, real t, gateType g_devF) {
	for (int i = 0; i < stim.numStim[0]; i++) {
		if (t >= stim.start[i] && t <= stim.finish[i]) g_devF.vm[stim.node[i]] += stim.strength[i];
	}
}

void setupStim(stimulus* stim, int* nodes, real* start, real* dur, real* strength, int numStim) {
	CudaSafeCall(cudaMalloc((void **)&stim->start, sizeof(real)*numStim));
	CudaSafeCall(cudaMalloc((void **)&stim->finish, sizeof(real)*numStim));
	CudaSafeCall(cudaMalloc((void **)&stim->strength, sizeof(real)*numStim));
	CudaSafeCall(cudaMalloc((void **)&stim->node, sizeof(int)*numStim));
	CudaSafeCall(cudaMalloc((void **)&stim->numStim, sizeof(int)));
	

	for (int i = 0; i < numStim; i++) {
		dur[i] = dur[i] + start[i];
	}

	CudaSafeCall(cudaMemcpy((void*)stim->start, start, sizeof(real)*numStim, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy((void*)stim->finish, dur, sizeof(real)*numStim, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy((void*)stim->strength, strength, sizeof(real)*numStim, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy((void*)stim->node, nodes, sizeof(int)*numStim, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy((void*)stim->numStim, &numStim, sizeof(int), cudaMemcpyHostToDevice));
}