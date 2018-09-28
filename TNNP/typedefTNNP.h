
#ifndef TYPEDEFTNNP_H
#define TYPEDEFTNNP_H

#include "typedefSparse.h"

typedef struct gateType {
	real *vm;
	real *Cai,*CaSR,*Nai,*Ki,*m,*h,*j,*xr1,*xr2,*xs,*r,*s,*d,*f,*fca,*g;
	real **qp;
	int qpl = 16;
	} gateType;

#endif