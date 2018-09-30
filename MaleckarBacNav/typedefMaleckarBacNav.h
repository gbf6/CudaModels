
#ifndef TYPEDEFMALECKARBACNAV_H
#define TYPEDEFMALECKARBACNAV_H

#include "typedefSparse.h"

typedef struct gateType {
	real *vm;
	real *r, *s, *m, *h;
	real **qp;
	int qpl = 4;
	} gateType;

#endif