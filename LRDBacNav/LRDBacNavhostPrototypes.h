#ifndef _LRDBACNAVHOSTPROTOTYPESH_
#define _LRDBACNAVHOSTPROTOTYPESH_

#include <stdio.h>

void LRDBacNav_init(char** res);
void LRDBacNav_sync(int memSize, size_t pitch, gateType* gate_h, gateType* gate_dev);
void LRDBacNav_gateinit(int memSize, size_t* pitch, gateType* gate_h, gateType* gate_dev, gateType* gate_devF);
void LRDBacNav_exit(int memSize, size_t pitch, gateType* gate_h, gateType* gate_dev, gateType* gate_devF, sparse* MatrixINT, cudasparse* cudaMatrixINT);

#endif