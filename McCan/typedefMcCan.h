
#ifndef TYPEDEFMCCAN_H
#define TYPEDEFMCCAN_H

#include "typedefSparse.h"

typedef struct gateType {
	real *vm;

	real *r, *s;
	real **qp;
	int qpl = 2;
	} gateType;

#endif