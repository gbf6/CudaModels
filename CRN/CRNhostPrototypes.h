#ifndef _CRNHOSTPROTOTYPESH_
#define _CRNHOSTPROTOTYPESH_

#include <stdio.h>

void CRN_init(char** res);
void CRN_sync(int memSize, size_t pitch, gateType* gate_h, gateType* gate_dev);
void CRN_gateinit(int memSize, size_t* pitch, gateType* gate_h, gateType* gate_dev, gateType* gate_devF);
void CRN_exit(int memSize, size_t pitch, gateType* gate_h, gateType* gate_dev, gateType* gate_devF, sparse* MatrixINT, cudasparse* cudaMatrixINT);

#endif