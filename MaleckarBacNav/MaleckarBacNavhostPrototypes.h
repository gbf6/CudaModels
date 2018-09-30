#ifndef _MALECKARBACNAVHOSTPROTOTYPESH_
#define _MALECKARBACNAVHOSTPROTOTYPESH_

#include <stdio.h>

void MaleckarBacNav_init(char** res);
void MaleckarBacNav_sync(int memSize, size_t pitch, gateType* gate_h, gateType* gate_dev);
void MaleckarBacNav_gateinit(int memSize, size_t* pitch, gateType* gate_h, gateType* gate_dev, gateType* gate_devF);
void MaleckarBacNav_exit(int memSize, size_t pitch, gateType* gate_h, gateType* gate_dev, gateType* gate_devF, sparse* MatrixINT, cudasparse* cudaMatrixINT);

#endif