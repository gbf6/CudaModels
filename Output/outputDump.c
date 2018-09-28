
#include "outputDump.h"
#include "inputs.cuh"

void outputDump(int totpoints, gateType* gate_h, FILE* fp) {
	fwrite(gate_h->vm, sizeof(real), totpoints, fp);
	fflush(fp);
}