#ifndef _MAINWRAPPERCUH_
#define _MAINWRAPPERCUH_

#include "inputs.cuh"

void main_wrapper(int pitch, real beta, real Cm, real t, real dt, int totpoints, real rx, gateType g_dev, gateType g_devF, stimulus stim, cudasparse cudaMatrixINT);
void __global__ TstepEM(int pitch, real beta, real Cm, real t, real dt, int totpoints, real rx, gateType g_dev, gateType g_devF, cudasparse cudaMatrixINT);
void __global__ TstepEMExit(real Cm, real dt, int totpoints, gateType g_dev, gateType g_devF);

#endif