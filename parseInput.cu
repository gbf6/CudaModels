#include <stdio.h>

#include "parseInput.cuh"

#include "inputs.cuh"

rword cases[] = {
	{ "defaultnodetype", 4201 },
{ "default_nodetype", 4201 },
{ "grid",	4000 },
{ "grid_a",	4004 },
{ "grid_area",	4004 },
{ "grid_b",	4005 },
{ "grid_bath",	4005 },
{ "grid_e",	4002 },
{ "grid_ext",	4002 },
{ "grid_i",	4001 },
{ "grid_int",	4001 },
{ "grid_v",	4004 },
{ "grid_vol",	4004 },
{ "grid_volume",4004 },
{ "gridname",	4000 },
{ "gridname_a",	4004 },
{ "gridname_area",4004 },
{ "gridname_e",	4002 },
{ "gridname_ext",4002 },
{ "gridname_i",	4001 },
{ "gridname_int",4001 },
{ "gridname_v",	4004 },
{ "gridname_vol",4004 },
{ "nodefile",	4200 },
{ "scale_a",	4012 },
{ "scale_area",	4012 },
{ "scale_b",	4013 },
{ "scale_bath",	4013 },
{ "scale_e",	4011 },
{ "scale_ext",	4011 },
{ "scale_i",	4010 },
{ "scale_int",	4010 },
{ "scale_v",	4012 },
{ "scale_vol",	4012 },
{ "scale_volume",4012 },
{ "dur",	3005 },
{ "duration",	3005 },
{ "is",		3003 },
{ "istim",	3003 },
{ "list",	3002 },
{ "nlist",	3002 },
{ "node",	3002 },
{ "nodelist",	3002 },
{ "nodes",	3002 },
{ "nstim",	3001 },
{ "numstim",	3001 },
{ "onset",	3004 },
{ "start",	3004 },
{ "strength",	3003 },
{ "tdur",	3005 },
{ "ts",		3004 },
{ "tstart",	3004 },
{ "type",	3006 },
{ "auxvar",	9044 },
{ "auxvars",	9044 },
{ "beta",	9010 },
{ "catchsig",	9099 },
{ "catchsignal",9099 },
{ "checkpoint",	9100 },
{ "checkpointfile", 9101 },
{ "checkpointinterval", 9102 },
{ "ckpt",	9100 },
{ "ckptfile",	9101 },
{ "ckptint",	9102 },
{ "ckptinterval", 9102 },
{ "checkpointtime", 9103 },
{ "ckpttime",	9103 },
{ "ckptinttime", 9103 },
{ "ckptintervaltime", 9103 },
{ "cm",		9011 },
{ "debug",	9002 },
{ "debuglevel",	9002 },
{ "debugflush",	9003 },
{ "debug_flush",9003 },
{ "loadfile",   9040 },
{ "loadstate",	9030 },
{ "memfile",	9033 },
{ "memparam",	9033 },
{ "memparamfile", 9033 },
{ "savefile",	9041 },
{ "savestate",  9031 },
{ "statefile",	9021 },
{ "statfile",	9021 },
{ "statusfile",	9021 },
{ "forceauxvars",9045 },
{ "forceauxvar",9045 },
{ "forceaux",	9045 },
{ "useauxvar",	9044 },
{ "useauxvars",	9044 },
{ "useaux",	9044 },
{ "initvm",	9201 },
{ "initialvm",	9201 },
{ "initcondvm",	9201 },
{ "initcondfilevm", 9201 },
{ "initcondvmfile", 9201 },
{ "initialconditionvm", 9201 },
{ "initcond",	9200 },
{ "initcondfile", 9200 },
{ "initialcondition", 9200 },
{ "showversion", 9300 },
{ "show_version", 9300 },
{ "showver", 9300 },
{ "show_ver", 9300 },
{ "showrev", 9300 },
{ "show_rev", 9300 },
{ "deltat",	2001 },
{ "dt",		2001 },
{ "t0",		2003 },
{ "tf",		2002 },
{ "tfinal",	2002 },
{ "tinit",	2003 },
{ "tinitial",	2003 },
{ "appendout",	6040 },
{ "appendoutput",6040 },
{ "dump_a",	6011 },
{ "dump_aux",	6011 },
{ "dump_b",	6024 },
{ "dump_bath",	6024 },
{ "dump_q",	6010 },
{ "dump_spacing",6001 },
{ "dump_save",	6030 },
{ "dump_savepe",6030 },
{ "dump_time",	6025 },
{ "dump_tnext",	6099 },
{ "dump_vb",	6024 },
{ "dump_ve",	6023 },
{ "dump_vi",	6022 },
{ "dump_vm",	6021 },
{ "dump_wt",	6026 },
{ "dumpa",	6011 },
{ "dumpaux",	6011 },
{ "dumpb",	6024 },
{ "dumpbath",	6024 },
{ "dumpq",	6010 },
{ "dumpspacing",6001 },
{ "dumptime",	6025 },
{ "dumptnext",	6099 },
{ "dumpvb",	6024 },
{ "dumpve",	6023 },
{ "dumpvi",	6022 },
{ "dumpvm",	6021 },
{ "dumpwt",	6026 },
{ "dump_hdr",	6050 },
{ "dumphdr",	6050 },
{ "dump_header",6050 },
{ "dumpheader",	6050 },
{ "outfile",	6003 },
{ "outfilename",6003 },
{ "outputfile",	6003 },
{ "outputfilename",6003 },
{ "outputspacing",6001 },
{ "outspacing",	6001 },
{ NULL, 0 }
};

int AppendResources(char** res, int ac, char** av) {
	FILE* fp;
	int   i, k, c;
	long  l;
	char* buffer;
	char* ptr;

	c = NumResources(res);
	for (i = 1; i<ac; i++) {
		if ((av[i][0] == '-') || (av[i][0] == '+')) {
			/* input file */
			fp = fopen(av[i] + 1, "r");
			if (fp != NULL) {
				/* get file size */
				l = 0;
				fseek(fp, l, SEEK_END);
				l = ftell(fp);
				/* allocate memory to hold the file */
				buffer = (char*)malloc(l + 1);
				/* load file into malloc'd memory */
				rewind(fp);
				fread(buffer, 1, l, fp);
				fclose(fp);
				buffer[l] = 0;
				/* parse each line */
				k = 0;
				ptr = buffer;
				while (k < l) {
					if (buffer[k] == '\n') {
						buffer[k] = 0;
						if ((ptr[0] != '#')
							&& (ptr[0] != '%')
							&& (ptr[0] != '!')
							&& (ptr[0] != '/')
							&& (ptr[0] != 0)) {
							res[c] = ptr;
							c++;
						}
						/* set up for next one */
						ptr = buffer + k + 1;
					}
					else if ((buffer[k] == '\\')
						&& (buffer[k + 1] == '\n')) {
						/* catenate with */
						/* previous line */
						buffer[k] = ' ';
						buffer[k + 1] = ' ';
					}
					k++;
				}
				/* check to see if we forgot to put a return */
				/* after the last line of the input file     */
				if (ptr[0] != 0) {
					res[c] = ptr;
					c++;
					if (c >= MAX_RESOURCES) {
						return(-1);
					}
				}
			}
			else {
				printf("Error opening file %s\n", av[i] + 1);
			}
		}
		else {
			/* resource spec:  stuff=value */
			res[c] = av[i];
			c++;
			if (c >= MAX_RESOURCES) {
				return(-1);
			}
		}
	}
	res[c] = NULL;

	return(0);
}

int NumResources(char** res) {
	int i;

	i = 0;
	while (res[i] != NULL) {
		i++;
	}

	return(i);
}

int FindCommand(rword* list, char* txt) {
	int    i;
	int    cmd = -1;
	rword* wf;
	static char keytxt[128];
	rword v1, v2;

	i = 0;
	while ((txt[i] != '(') && (txt[i] != '[')
		&& (txt[i] != ':') && (txt[i] != '=')
		&& (txt[i] != ' ') && (txt[i] != '\t')) {
		keytxt[i] = txt[i];
		i++;
	}
	keytxt[i] = 0;

	wf = list;
	while ((wf->txt) != NULL) {
		i = stricmp(wf->txt, keytxt);
		if (i == 0) {
			/* match found */
			cmd = wf->cmd;
			break;
		}
		else {
			wf++;
		}
	}

	return(cmd);
}

char* GetStringValue(char* txt) {
	int i;

	i = 0;
	while (txt[i] != '=') {
		i++;
	}

	if (txt[i + 1] == 0) {
		return(0);
	}

	return(txt + i + 1);
}

real GetRealValue(char* txt) {
	int i;
	real r;

	/* move to the equals sign */
	i = 0;
	while (txt[i] != '=') {
		i++;
	}

	if (txt[i] == 0) {
		return(-9.9e9);
	}

	return(atof(txt + i + 1));
}

int GetIntValue(char* txt) {
	int i;

	/* move to the equals sign */
	i = 0;
	while (txt[i] != '=') {
		i++;
	}

	if (txt[i] == 0) {
		return(-1);
	}

	return(atoi(txt + i + 1));
}


int FindNum(char* txt) {
	int i;

	/* move forward to a bracket/paren/colon */
	i = 0;
	while ((txt[i] != '(') && (txt[i] != '[')
		&& (txt[i] != ':') && (txt[i] != 0)) {
		i++;
	}
	if (txt[i] == 0) {
		return(-1);
	}
	i++;

	if ((txt[i]<'0') || (txt[i]>'9')) {
		if (txt[i] < 'a') {
			return((int)txt[i] + 32);
		}
		else {
			return((int)txt[i]);
		}
	}
	return(atoi(txt + i));
}


int GetNumValues(char* txt) {
	int i, j, k, s, nc;
	char* p;
	char* cpy;
	int count;
	static char* lasttxt = NULL;
	static int   lastcount = -1;

	if (txt == lasttxt) {
		return(lastcount);
	}

	/* go to the equals sign */
	i = 0;
	while (txt[i] != '=') {
		i++;
	}
	i++;

	/* go thru each token in the text */
	count = 0;
	cpy = strdup(txt + i);
	p = strtok(cpy, ",");
	while (p != NULL) {
		/* look for commas and colons */
		i = 0;
		nc = 0;
		while (p[i] != 0) {
			if (p[i] == ':') {
				if (nc == 0) {
					j = i;
				}
				else if (nc == 1) {
					k = i;
				}
				nc++;
			}
			i++;
		}
		switch (nc) {
		case 0: /* no colons, just a single number */
			count++;
			break;
		case 1: /* 1 colon, assume unit stride */
			k = atoi(p + j + 1);
			j = atoi(p);
			count += (k - j) + 1;
			break;
		case 2: /* 2 colon, non-unit stride */
			k = atoi(p + k + 1);
			s = atoi(p + j + 1);
			j = atoi(p);
			count += ((k - j) / s) + 1;
			break;
		}
		p = strtok(NULL, ",");
	}

	lasttxt = txt;
	lastcount = count;

	return(count);
}

int* GetIntArray(char* txt) {
	int  i, j, k, s, nc, l;
	int* list = NULL;
	char* p = NULL;
	char* cpy = NULL;
	int count;

	count = GetNumValues(txt);
	list = (int*)malloc(count * sizeof(int));

	/* go to the equals sign */
	i = 0;
	while (txt[i] != '=') {
		i++;
	}
	i++;

	/* go thru each token in the text */
	count = 0;
	cpy = strdup(txt + i);
	p = strtok(cpy, ",");
	while (p != NULL) {
		/* look for commas and colons */
		i = 0;
		nc = 0;
		while (p[i] != 0) {
			if (p[i] == ':') {
				if (nc == 0) {
					j = i;
				}
				else if (nc == 1) {
					k = i;
				}
				nc++;
			}
			i++;
		}
		switch (nc) {
		case 0: /* no colons, just a single number */
			list[count] = atoi(p);
			count++;
			break;
		case 1: /* 1 colon, assume unit stride */
			k = atoi(p + j + 1);
			j = atoi(p);
			for (l = j; l <= k; l++) {
				list[count] = l;
				count++;
			}
			break;
		case 2: /* 2 colons, non-unit stride */
			k = atoi(p + k + 1);
			s = atoi(p + j + 1);
			j = atoi(p);
			for (l = j; l <= k; l += s) {
				list[count] = l;
				count++;
			}
			break;
		}
		p = strtok(NULL, ",");
	}

	return(list);
}

real* GetRealArray(char* txt) {
	int  i, j, k, s, nc;
	char* p = NULL;
	char* cpy = NULL;
	int count;
	real  l;
	real* list = NULL;

	count = GetNumValues(txt);
	list = (real*)malloc(count * sizeof(real));

	/* go to the equals sign */
	i = 0;
	while (txt[i] != '=') {
		i++;
	}
	i++;

	/* go thru each token in the text */
	count = 0;
	cpy = strdup(txt + i);
	p = strtok(cpy, ",");
	while (p != NULL) {
		/* look for commas and colons */
		i = 0;
		nc = 0;
		while (p[i] != 0) {
			if (p[i] == ':') {
				if (nc == 0) {
					j = i;
				}
				else if (nc == 1) {
					k = i;
				}
				nc++;
			}
			i++;
		}
		switch (nc) {
		case 0: /* no colons, just a single number */
			list[count] = atof(p);
			count++;
			break;
		case 1: /* 1 colon, assume unit stride */
			k = atof(p + j + 1);
			j = atof(p);
			for (l = j; l <= k; l += 1.0) {
				list[count] = l;
				count++;
			}
			break;
		case 2: /* 2 colons, non-unit stride */
			k = atof(p + k + 1);
			s = atof(p + j + 1);
			j = atof(p);
			for (l = j; l <= k; l += s) {
				list[count] = l;
				count++;
			}
			break;
		}
		p = strtok(NULL, ",");
	}

	return(list);
}


bool GetTFValue(char* txt) {
	int i;

	/* move to the equals sign */
	i = 0;
	while (txt[i] != '=') {
		i++;
	}
	i++;

	if (txt[i] == 0) {
		/* this is really an error condition, but no real way */
		/* to indicate that with only 'true' or 'false'       */
		return(false);
	}

	/* look for the 't' in 'true' or the 'y' in 'yes' */
	if ((txt[i] == 't') || (txt[i] == 'T') || (txt[i] == '1')
		|| (txt[i] == 'y') || (txt[i] == 'Y')) {
		return(true);
	}
	/* look for the word 'on' */
	if ((txt[i] == 'o') || (txt[i] == 'O')) {
		if ((txt[i + 1] == 'n') || (txt[i + 1] == 'N')) {
			return(true);
		}
	}

	return(false);
}