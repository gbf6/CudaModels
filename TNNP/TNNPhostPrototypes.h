#ifndef _TNNPHOSTPROTOTYPESH_
#define _TNNPHOSTPROTOTYPESH_

#include <stdio.h>

void TNNP_sync(int memSize, size_t pitch, gateType* gate_h, gateType* gate_dev);
void TNNP_gateinit(int memSize, size_t* pitch, gateType* gate_h, gateType* gate_dev, gateType* gate_devF);
void TNNP_exit(int memSize, size_t pitch, gateType* gate_h, gateType* gate_dev, gateType* gate_devF, sparse* MatrixINT, cudasparse* cudaMatrixINT);

#endif