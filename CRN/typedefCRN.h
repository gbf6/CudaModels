
#ifndef _TYPEDEFCRNH_
#define _TYPEDEFCRNH_

#include "typedefSparse.h"

typedef struct gateType {
	real *m, *h, *j, *sa, *si, *ua, *ui, *xr, *xs, *d, *f, *fca, *u, *v, *w;
	real *Nai, *Ki, *Cai, *Caup, *Carel;
	real **qp;
	int qpl = 20;
	real *vm;
	} gateType;

#endif