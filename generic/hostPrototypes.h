#ifndef _HOSTPROTOTYPESH_
#define _HOSTPROTOTYPESH_

#include <stdio.h>

void _sync(int memSize, size_t pitch, gateType* gate_h, gateType* gate_dev);
void _gateinit(int memSize, size_t* pitch, gateType* gate_h, gateType* gate_dev, gateType* gate_devF);
void _exit(int memSize, size_t pitch, gateType* gate_h, gateType* gate_dev, gateType* gate_devF, sparse* MatrixINT, cudasparse* cudaMatrixINT);

#endif