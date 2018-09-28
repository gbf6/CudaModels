
#ifndef _TYPEDEFSTIMSIMPLEH_
#define _TYPEDEFSTIMSIMPLEH_

typedef struct stimulus {
	real *start, *finish, *strength;
	int *node, *numStim;
} stimulus;

#endif