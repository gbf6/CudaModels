int bvecreadinfo(char* filename, int* ssz);
int spralloc(sparse* M, domain_t dtp, int type, int rows);
int sprreadinfo(char* filename, int* type, int* rows, int* cols, int* bwp, int* bwm, int* maxnz);
int sprfree(sparse* M);
int sprread(char* filename, sparse* m1);
void swapendian(void* pt, int dsize, int num);
void cudasprinit(cudasparse* C, sparse* M);
void __device__ sprDiff(real* fv, cudasparse* cudaMatrixINT, int i2d, int totpoints, real* v, real beta);