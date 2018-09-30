#ifndef _CRNGLOBALVARIABLESCUH_
#define _CRNGLOBALVARIABLESCUH_

// Macro for 2D Finite Difference
#define BLOCK_DIM_1D	256

#define pi					 3.14159265359
#define cellLength			 100.0 			/* um */
#define cellDiameter		 16.0 			/* um */
#define R					 8.3143 			/* J/(Kmol) */
#define T					 310.0 			/* K */
#define F					 96.4867 		/* C/mmol */
#define Vcell				 20100.0 		/* um^3 */
#define Vi					 13668.0 		/* um^3 */
#define Vup					 1109.52 		/* um^3 */
#define Vrel				 96.48 			/* um^3 */
#define Ko					 5.4 			/* mM */
#define Nao					 140.0 			/* mM */
#define Cao					 1.8 			/* mM */
#define gna_0				 7.8e-4 			/* mS */
#define gk1_0				 0.09e-4 		/* mS */
#define gto_0				 0.1652e-4 		/* mS */
#define gkr_0                0.0294e-4 		/* mS */
#define gks_0				 0.129e-4 		/* mS */
#define gcal_0			     0.1238e-4 		/* mS */
#define gbca_0			     0.00113e-4 		/* mS */
#define gbna_0			     0.000674e-4 	/* mS */
#define Inakmax			     0.60e-4 		/* uA */
#define Inacamax		     1600.0e-4 		/* uA */
#define Ipcamax			     0.275e-4 		/* uA */
#define Iupmax			     0.005 			/* mM/msec */
#define Kq10			     3.0 			/* unitless */
#define lambda			     0.35 			/* unitless */
#define Kmnai			     10.0 			/* mM */
#define Kmko		    	 1.5 			/* mM */
#define Kmna			     87.5 			/* mM */
#define Kmca			     1.38 			/* mM */
#define ksat			     0.1 			/* unitless */
#define krel			     30.0 			/* msec^(-1) */
#define kup				     0.00092 		/* mM */
#define Caupmax			     15.0 			/* mM */
#define Cmdnmax			     0.05 			/* mM */
#define Trpnmax			     0.07 			/* mM */
#define Csqnmax			     10.0 			/* mM */
#define KmCmdn			     0.00238 		/* mM */
#define KmTrpn			     0.0005 			/* mM */
#define KmCsqn			     0.8 			/* mM */
#define taufca			     2.0 			/* msec */
#define tautr			     180.0 			/* msec */
#define tauu		         8.0 			/* msec */
#define CRN_RestVoltage_0	-81.2			/* mV */

#endif