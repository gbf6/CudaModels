
#ifndef TYPEDEF_H
#define TYPEDEF_H

#include "typedefSparse.h"

typedef struct gateType {
	real *vm;

	real *m,*h,*j,*d,*f,*b,*g,*xr,*xs1,*xs2,*zdv,*ydv;
	real *nai,*ki,*nsr;
    real *nao,*ko,*cao;

    real *cai,*jsr,*caiont,*BOOL,*tcicr,*tjsrol,dcaitont;
	real **qp;
	int qpl = 17;
	} gateType;

#endif