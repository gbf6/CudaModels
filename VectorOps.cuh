#ifndef _VECTOROPSCUH_
#define _VECTOROPSCUH_

typedef double real;

void __device__  vvaddDev(int i2d, real alpha, real* x, real* y, int totpoints);

#endif