
#ifndef _TYPEDEFSPARSEH_
#define _TYPEDEFSPARSEH_

#define tokensep " \t\n"
#define _PRIMARY 1
#define _INIT_SPARSE {0,0,0,0,0,Undefd,0,NULL,NULL}


typedef double real;
typedef unsigned char byte;

typedef enum domain_t {
	Undefd = -1,
	Tissue = 0,
	Intracellular = 1,
	Intra = 1,
	Extracellular = 2,
	Extra = 2,
	Bath = 3,
	Solver = 4,
	State = 5,
	StateVar = 5,
	Aux = 6,
	AuxVar = 6,
	Memparam = 7,
	Param = 7
} domain_t;

/* sparse matrix structure */
typedef struct sparse {
	int   type;
	int   rows, cols;
	int   maxnz;
	int   csep;
	domain_t dtype;
	int   msgtag;
	int*  jcoef;
	real* coef;
} sparse;

typedef struct cudasparse {
	int*   type;
	int*   rows, cols;
	int*   maxnz;
	int*   csep;
	domain_t* dtype;
	int*   msgtag;
	int*  jcoef;
	real* coef;
} cudasparse;

#endif