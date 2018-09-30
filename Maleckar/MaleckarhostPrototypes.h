#ifndef _MALECKARHOSTPROTOTYPESH_
#define _MALECKARHOSTPROTOTYPESH_

#include <stdio.h>

void Maleckar_init(char** res);
void Maleckar_sync(int memSize, size_t pitch, gateType* gate_h, gateType* gate_dev);
void Maleckar_gateinit(int memSize, size_t* pitch, gateType* gate_h, gateType* gate_dev, gateType* gate_devF);
void Maleckar_exit(int memSize, size_t pitch, gateType* gate_h, gateType* gate_dev, gateType* gate_devF, sparse* MatrixINT, cudasparse* cudaMatrixINT);

#endif