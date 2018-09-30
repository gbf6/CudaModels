#ifndef _FOURVHOSTPROTOTYPESH_
#define _FOURVHOSTPROTOTYPESH_

#include <stdio.h>

void fourV_init(char** res);
void fourV_sync(int memSize, size_t pitch, gateType* gate_h, gateType* gate_dev);
void fourV_gateinit(int memSize, size_t* pitch, gateType* gate_h, gateType* gate_dev, gateType* gate_devF);
void fourV_exit(int memSize, size_t pitch, gateType* gate_h, gateType* gate_dev, gateType* gate_devF, sparse* MatrixINT, cudasparse* cudaMatrixINT);

#endif