#ifndef _LRDHOSTPROTOTYPESH_
#define _LRDHOSTPROTOTYPESH_

#include <stdio.h>

void LRD_sync(int memSize, size_t pitch, gateType* gate_h, gateType* gate_dev);
void LRD_gateinit(int memSize, size_t* pitch, gateType* gate_h, gateType* gate_dev, gateType* gate_devF);
void LRD_exit(int memSize, size_t pitch, gateType* gate_h, gateType* gate_dev, gateType* gate_devF, sparse* MatrixINT, cudasparse* cudaMatrixINT);

#endif