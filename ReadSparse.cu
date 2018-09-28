
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

#include "typedefSparse.h"
#include "sparsePrototypes.cuh"
#include "./common/CudaSafeCall.h"

int bvecreadinfo(char* filename, int* ssz) {
	int   sz;
	FILE* fp;
	char  buffer[128];

	fp = fopen(filename, "r");
	if (fp == NULL) {
		return(-1);
	}
	fgets(buffer, 128, fp);
	fclose(fp);

	/* check text IDs */
	if (toupper(buffer[0]) != 'B') {
		/* not a vector file */
		return(-2);
	}
	if ((buffer[2] - '0') != sizeof(byte)) {
		/* wrong data-type size - could convert! */
		return(-4);
	}
	sz = atoi(buffer + 4);

	*ssz = sz;

	return(0);
}


/* allocate a sparse matrix of the given type */
int spralloc(sparse* M, domain_t dtp, int type, int rows,
	int cols, int maxnz) {
	int m, n;
	int* ip;

	M->type = type;
	M->rows = rows;
	M->cols = cols;
	M->maxnz = maxnz;
	M->csep = rows;
	M->dtype = dtp;
	/* grab 2 message tags */
	//M->msgtag = NextMsgTag();
	//NextMsgTag();

	/*if (type == _ZERO) {
		return(0);
	}*/

	/* jcoef array is of size = rows*maxnz */
	M->jcoef = (int*)malloc(rows*maxnz * sizeof(int));
	if (M->jcoef == NULL) {
		return(-1);
	}
	/* set up jcoef area for SPARSE type */
	ip = M->jcoef;
	for (m = 0; m<maxnz; m++) {
		for (n = 0; n<rows; n++) {
			ip[n] = n + 1;
		}
		ip += rows;
	}

	/* coef array is of size = rows*maxnz */
	M->coef = (real*)malloc(rows*maxnz * sizeof(real));
	if (M->coef == NULL) {
		return(-2);
	}
	memset((void*)(M->coef), 0, rows*maxnz * sizeof(real));

	return(0);
}


int sprreadinfo( char* filename, int* type, int* rows, int* cols, 
		int* bwp, int* bwm, int* maxnz ) {
	int   tp,rrr,ccc,bbp,bbm,mmm,m;
	FILE* fp;
	char  buffer[64];
	char* ctmp;

	fp = fopen( filename, "r" );
	if( fp == NULL ) {
		return( -1 );
	}

	fgets( buffer, 64, fp );

	/* check text IDs */
	if( (buffer[0]!='S') && (buffer[0]!='s') ) {
		/* not a sparse matrix file */
		return( -2 );
	}
	if( (buffer[2]-'0') != sizeof(real) ) {
		/* wrong real data-type size - could convert! */
		return( -4 );
	}
	ctmp  = strtok( buffer, tokensep );
	ctmp  = strtok( NULL, tokensep );
	rrr   = atoi( ctmp );
	ctmp  = strtok( NULL, tokensep );
	ccc   = atoi( ctmp );
	ctmp  = strtok( NULL, tokensep );
	bbp   = atoi( ctmp );
	ctmp  = strtok( NULL, tokensep );
	bbm   = atoi( ctmp );
	ctmp  = strtok( NULL, tokensep );
	tp    = atoi( ctmp );
	ctmp  = strtok( NULL, tokensep );
	mmm   = atoi( ctmp );

	fclose( fp );

	*type  = tp;
	*rows  = rrr;
	*cols  = ccc;
	*bwp   = bbp;
	*bwm   = bbm;
	*maxnz = mmm;

	return( 0 );
}

int sprfree(sparse* M) {
	if (M->jcoef != NULL) {
		free(M->jcoef);
	}
	if (M->coef != NULL) {
		free(M->coef);
	}

	M->type = 0;
	M->rows = 0;
	M->cols = 0;
	M->maxnz = 0;
	M->csep = 0;
	M->jcoef = NULL;
	M->coef = NULL;
	M->dtype = Undefd;
	//M->msgtag = MPI_ANY_TAG;

	return(0);
}

/* works for sequential programs only!! */
int sprread( char* filename, sparse* m1 ) {
	int   tp,rrr,ccc,bbp,bbm,mnz,r,c,q,i;
	FILE* fp;
	char  buffer[64];
	char* ctmp;
	real* rtmp;
	int*  itmp;
	real* rptr;
	int*  iptr;

	sprfree( m1 );

	fp = fopen( filename, "r" );
	if( fp == NULL ) {
		return( -1 );
	}

	fgets( buffer, 64, fp );

	/* check text IDs */
	if( (buffer[0]!='S') && (buffer[0]!='s') ) {
		/* not a sparse matrix file */
		return( -2 );
	}

	q = 0x1234;
	ctmp = (char*)(&q);
	if( ctmp[0] == 0x34 ) {
		/* this machine is little endian */
		if( (buffer[1]!='L') && (buffer[1]!='l') ) {
			/* wrong endian-ness - could convert! */
			return( -3 );
		}
	} else {
		/* this machine is big endian */
		if( (buffer[1]!='B') && (buffer[1]!='b') ) {
			/* wrong endian-ness - could convert! */
			return( -3 );
		}
	}

	if( (buffer[2]-'0') != sizeof(real) ) {
		/* wrong real data-type size - could convert! */
		return( -4 );
	}
	ctmp = strtok( buffer, tokensep );
	ctmp = strtok( NULL, tokensep );
	rrr  = atoi( ctmp );
	ctmp = strtok( NULL, tokensep );
	ccc  = atoi( ctmp );
	ctmp = strtok( NULL, tokensep );
	bbp  = atoi( ctmp );
	ctmp = strtok( NULL, tokensep );
	bbm  = atoi( ctmp );
	ctmp = strtok( NULL, tokensep );
	tp   = atoi( ctmp );
	ctmp = strtok( NULL, tokensep );
	mnz  = atoi( ctmp );

	/* set up memory space */
	r = spralloc( m1, Tissue, _PRIMARY, rrr, ccc, mnz );
	if( r < 0 ) {
		return( r-6 );
	}

	/* get jcoef entries */
	q = buffer[3] - '0';
	if( q == sizeof(int) ) {
		fseek( fp, 128, SEEK_SET );
		itmp = m1->jcoef;
		/*for(c=0;c<mnz;c++) {
			fread( itmp, sizeof(int), rrr, fp );
			itmp += rrr;
		}*/
		fread(itmp, sizeof(int), rrr*mnz, fp);
		fseek( fp, 128+mnz*rrr*sizeof(int), SEEK_SET );
	} else {
		fclose( fp );
		return( -5 );
	}

	/* get coef entries */
	rtmp = m1->coef;
	for(c=0;c<mnz;c++) {
		fread( rtmp, sizeof(real), rrr, fp );
		rtmp += rrr;
	}
	fclose(fp);

	return( 0 );
}

void swapendian(void* ptr, int dsize, int num) {
	char* cp = (char*)ptr;
	char b0, b1, b2, b3, b4, b5, b6, b7;
	int i, j;

	switch (dsize) {
	case 8:
		for (i = 0; i<num; i++) {
			b0 = cp[0];
			b1 = cp[1];
			b2 = cp[2];
			b3 = cp[3];
			b4 = cp[4];
			b5 = cp[5];
			b6 = cp[6];
			b7 = cp[7];
			cp[0] = b7;
			cp[1] = b6;
			cp[2] = b5;
			cp[3] = b4;
			cp[4] = b3;
			cp[5] = b2;
			cp[6] = b1;
			cp[7] = b0;
			cp += 8;
		}
		break;
	case 4:
		for (i = 0; i<num; i++) {
			b0 = cp[0];
			b1 = cp[1];
			b2 = cp[2];
			b3 = cp[3];
			cp[0] = b3;
			cp[1] = b2;
			cp[2] = b1;
			cp[3] = b0;
			cp += 4;
		}
		break;
	case 2:
		for (i = 0; i<num; i++) {
			b0 = cp[0];
			b1 = cp[1];
			cp[0] = b1;
			cp[1] = b0;
			cp += 2;
		}
		break;
	case 1:
		/* no work to do */
		break;
	}

	return;
}

void cudasprinit(cudasparse* C, sparse* M) {
	CudaSafeCall(cudaMalloc((void **)&C->type, sizeof(M->type)));
	CudaSafeCall(cudaMalloc((void **)&C->rows, sizeof(M->rows)));
	CudaSafeCall(cudaMalloc((void **)&C->maxnz, sizeof(M->maxnz)));
	CudaSafeCall(cudaMalloc((void **)&C->csep, sizeof(M->csep)));
	CudaSafeCall(cudaMalloc((void **)&C->jcoef, sizeof(int)*M->rows*M->maxnz));
	CudaSafeCall(cudaMalloc((void **)&C->coef, sizeof(real)*M->rows*M->maxnz));

	CudaSafeCall(cudaMemcpy((void *)C->type, &M->type, sizeof(M->type), cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy((void *)C->rows, &M->rows, sizeof(M->rows), cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy((void *)C->maxnz, &M->maxnz, sizeof(M->maxnz), cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy((void *)C->csep, &M->csep, sizeof(M->csep), cudaMemcpyHostToDevice));

	CudaSafeCall(cudaMemcpy((void *)C->jcoef, (void *)M->jcoef, sizeof(int)*M->rows*M->maxnz, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy((void *)C->coef, (void *)M->coef, sizeof(real)*M->rows*M->maxnz, cudaMemcpyHostToDevice));

}

void __device__ sprDiff(real* fv, cudasparse* cudaMatrixINT, int i2d, int totpoints, real* v, real beta) {
	for (int i = 0; i < *cudaMatrixINT->maxnz; i++) {
		fv[i2d] += (1.0 / beta)*cudaMatrixINT->coef[i2d + i * totpoints] * v[cudaMatrixINT->jcoef[i2d + i * totpoints] - 1];
		//if (i2d == printv) printf("%d: %.31f, %.31f, %.31f, %.31f\n", cudaMatrixINT.jcoef[i2d + i * totpoints], cudaMatrixINT.coef[i2d + i * totpoints], g_dev.vmp[cudaMatrixINT.jcoef[i2d + i * totpoints] - 1], (1.0 / beta)*cudaMatrixINT.coef[i2d + i * totpoints] * g_dev.vmp[cudaMatrixINT.jcoef[i2d + i * totpoints] - 1], fv);
	}
	return;
}