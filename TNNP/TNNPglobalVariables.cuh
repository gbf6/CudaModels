
// Macro for 2D Finite Difference
#define BLOCK_DIM_1D (256)

#define pi (3.14159265359)

/*========================================================================
 * Model parameters defined as macros
 *
 * These parameters can be sent from the matlab side if the need arrises
 * with minimal effect on the performance of the code
 *========================================================================
 */


#define  Ko                   5.4
#define  Cao                  2.0
#define  Nao                  140.0
#define  Vc                   0.016404
#define  Vsr                  0.001094
#define  Bufc                 0.15
#define  Kbufc                0.001
#define  Bufsr                10.0
#define  Kbufsr               0.3
#define  taufca               2.0
#define  taug                 2.0
#define  Vmaxup               0.000425
#define  Kup                  0.00025
#define  R                    8314.472
#define  F                    96485.3415
#define  T                    310.0
#define  CAPACITANCE          0.185
#define  pKNa                 0.03
#define  KmK                  1.0
#define  KmNa                 40.0
#define  knak                 1.362
#define  knaca                1000
#define  KmNai                87.5
#define  KmCa                 1.38
#define  ksat                 0.1
#define  n                    0.35
#define  KpCa                 0.0005
#define  TNNP_GNa_0           14.838
#define  TNNP_GCaL_0          0.000175
#define  TNNP_Gto_0           0.294
#define  TNNP_Gkr_0           0.096
#define  TNNP_Gks_0           0.245
#define  TNNP_GK1_0           5.405
#define  TNNP_GbCa_0          0.000592
#define  TNNP_GpK_0           0.0146
#define  TNNP_GbNa_0          0.00029
#define  TNNP_GpCa_0          0.825
#define  TNNP_RestVoltage_0  -86.2

#define  RTONF             (R*T)/F
#define  inverseVcF2       1./(2.*Vc*F)
#define  inverseVcF        1./(Vc*F)
#define  Kupsquare         Kup*Kup
#define  BufcKbufc         Bufc*Kbufc
#define  Kbufcsquare       Kbufc*Kbufc
#define  Kbufc2            2.*Kbufc
#define  BufsrKbufsr       Bufsr*Kbufsr
#define  Kbufsrsquare      Kbufsr*Kbufsr
#define  Kbufsr2           2.*Kbufsr