#ifndef _STIMSIMPLECUH_
#define _STIMSIMPLECUH_

void setupStim(stimulus* stim, int* nodes, real* start, real* dur, real* strength, int numStim);
void __global__ stimSimpleDev(stimulus stim, real t, gateType g_dev);

#endif