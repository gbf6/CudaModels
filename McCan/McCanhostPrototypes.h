#ifndef _MCCANHOSTPROTOTYPESH_
#define _MCCANHOSTPROTOTYPESH_

#include <stdio.h>

void McCan_sync(int memSize, size_t pitch, gateType* gate_h, gateType* gate_dev);
void McCan_gateinit(int memSize, size_t* pitch, gateType* gate_h, gateType* gate_dev, gateType* gate_devF);
void McCan_exit(int memSize, size_t pitch, gateType* gate_h, gateType* gate_dev, gateType* gate_devF, sparse* MatrixINT, cudasparse* cudaMatrixINT);

#endif