#ifndef _PARSEINPUTCUH_
#define _PARSEINPUTCUH_

#define MAX_RESOURCES 256
#define MAX_STIM      256

typedef double real;

static char* inpBuffer[MAX_RESOURCES] = {
	"grid=default",  "dt=0.001",  "t0=0.0",  "tf=5.0",
	"outputfile=default.out",  "beta=2000.0",  "debug=0",  "loadstate=no",
	"savestate=no",  "checkpoint=no",  "defaultnodetype=0",
	NULL
};

typedef struct {
	char* txt;
	int   cmd;
} rword;

rword cases[];

int AppendResources(char** res, int ac, char** av);
int NumResources(char** res);
int FindCommand(rword* list, char* txt);
char* GetStringValue(char* txt);
real GetRealValue(char* txt);
int GetIntValue(char* txt);
int FindNum(char* txt);
int GetNumValues(char* txt);
int* GetIntArray(char* txt);
real* GetRealArray(char* txt);
bool GetTFValue(char* txt);

#endif