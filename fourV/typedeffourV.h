
#ifndef TYPEDEFFOURV_H
#define TYPEDEFFOURV_H

#include "typedefSparse.h"

typedef struct gateType {
	real *u, *v, *w, *s;
	real *vm;
	real **qp;
	int qpl = 3;
	} gateType;

#endif