#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#include <cuda.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

typedef double real;

#include "typedefSparse.h"
#include "sparsePrototypes.cuh"
#include "parseInput.cuh"
#include "VectorOps.cuh"
#include "mainwrapper.cuh"

#include "inputs.cuh"

#include "./Output/outputDump.c"


int memSize, totpoints;

size_t pitch;
int count;

// Physical parameters
real Lx, Ly, Lz, hx, hy, hz, CFL_max, t, tlim;
real dt, diff, rx, ry, rz;

// Voltage and gate arrays
gateType gate_h, gate_dev, gate_devF;

int i, tp, r, c, bwp, bwm, mnz;
real TissueLocalSize;
sparse MatrixINT = _INIT_SPARSE;
cudasparse cudaMatrixINT;

/////// MAIN ///////
static char* int_file = NULL;
static char* out_file = NULL;
real Tspacing, Tnext, beta, Cm;

//Stim
static int numStim = 0;
static int nodes[MAX_STIM];
static real start[MAX_STIM];
static real dur[MAX_STIM];
static real strength[MAX_STIM];

stimulus stim;

int main(int argc, char **argv) {
	int err = AppendResources(inpBuffer, argc, argv);

	int cmd, k;
	int* vi;
	real* vr;
	for (int i = 0; inpBuffer[i] != NULL; i++) {
		cmd = FindCommand(cases, inpBuffer[i]);
		switch (cmd) {
			case 4001:
				int_file = GetStringValue(inpBuffer[i]);
				break;
			case 2002:
				tlim = GetRealValue(inpBuffer[i]);
				break;
			case 2001:
				dt = GetRealValue(inpBuffer[i]);
				break;
			case 3001:
				numStim = GetIntValue(inpBuffer[i]);
				break;
			case 3002:
				k = FindNum(inpBuffer[i]);
				vi = GetIntArray(inpBuffer[i]);
				for(int j = 0; j < numStim; j++) nodes[j+k] = vi[j];
				break;
			case 3003:
				k = FindNum(inpBuffer[i]);
				vr = GetRealArray(inpBuffer[i]);
				for (int j = 0; j < numStim; j++) strength[j + k] = vr[j];
				break;
			case 3004:
				k = FindNum(inpBuffer[i]);
				vr = GetRealArray(inpBuffer[i]);
				for (int j = 0; j < numStim; j++) start[j + k] = vr[j];
				break;
			case 3005:
				k = FindNum(inpBuffer[i]);
				vr = GetRealArray(inpBuffer[i]);
				for (int j = 0; j < numStim; j++) dur[j + k] = vr[j];
				break;
			case 6003:
				out_file = GetStringValue(inpBuffer[i]);
				break;
			case 6001:
				Tspacing = GetRealValue(inpBuffer[i]);
				break;
			case 9010:
				beta = GetRealValue(inpBuffer[i]);
				break;
			case 9011:
				Cm = GetRealValue(inpBuffer[i]);
				break;
		}
	}

	sprreadinfo(int_file, &tp, &r, &c, &bwp, &bwm, &mnz);
	TissueLocalSize = r;
	sprread(int_file, &MatrixINT);
	cudasprinit(&cudaMatrixINT, &MatrixINT);
	
	// Memory size declaration
	memSize = TissueLocalSize*sizeof(real);
	totpoints = (int)TissueLocalSize;
	CFL_max = 0.2; // Courant stability condition
	
	// Physical length (cm)
	Lx = 12.0;
	// Physical spacing between nodes (cm)
	if(MatrixINT.maxnz <= 5) hx = Lx/(sqrt(totpoints)-1.0);
	else hx = Lx / (cbrt(totpoints) - 1.0);
	//dt = (1.0/diff)*(CFL_max-0.1)*(hx*hx*hy*hy)/(hx*hx+hy*hy);
	diff = 0.001;
	rx = dt*diff/(hx*hx);
	t = 0.0;
	Tnext = t;

	printf("\n********Grid dimensions*********\n");
	printf("Total number of nodes: %d\n", totpoints);
	printf("\n********Spatial dimensions*********\n");
	printf("Physical dx %f cm \n", hx);
	printf("\n********Diffusion*********\n");
	printf("Diffusion parallel component: %f cm^2/ms\n", diff);
	printf("\n********Time*********\n");
	printf("Time step: %f ms\n", dt);

	// Array allocation and init

	//CRN_gateinit(memSize, &pitch, &gate_h, &gate_dev, &gate_devF);
	fourV_gateinit(memSize, &pitch, &gate_h, &gate_dev, &gate_devF);

	setupStim(&stim, nodes, start, dur, strength, numStim);
	puts("\nStarting simulation\n");

	static FILE *fp1;
	fp1 = fopen(out_file, "wb+");
	// main loop
	outputDump(totpoints, &gate_h, fp1);
	Tnext += Tspacing;

	while (t < tlim) {

		main_wrapper(pitch, beta, Cm, t, dt, totpoints, rx, gate_dev, gate_devF, stim, cudaMatrixINT);
		t += dt;

		if (t >= Tnext) {

			//CRN_sync(memSize, pitch, &gate_h, &gate_dev);
			fourV_sync(memSize, pitch, &gate_h, &gate_dev);

			outputDump(totpoints, &gate_h, fp1);
			Tnext += Tspacing;
		}
	}

	if (t >= Tnext) {
		outputDump(totpoints, &gate_h, fp1);
		Tnext += Tspacing;
	}
	
	fclose(fp1);
	puts("\nSimulation ended\n");

	//CRN_exit(memSize, pitch, &gate_h, &gate_dev, &gate_devF, &MatrixINT, &cudaMatrixINT);
	fourV_exit(memSize, pitch, &gate_h, &gate_dev, &gate_devF, &MatrixINT, &cudaMatrixINT);

	return 0;
}