
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

#define  McCan_RestVoltage  -49.44
#define  EK                 -87
#define  gkv                 0.25
#define  INaKbar             2.002
#define  KmK                 1.0
#define  KmNa                11.0
#define  B                  -200
#define  Vrev               -150
#define  Ko                  5.4
#define  Nai                 9 /* 11.6 */
#define  Nao                 140
#define  R                   8314.472
#define  F                   96485.3415
#define  T                   310.0
#define  GbNa                0.0095
#define  gK1                 0.4822
#define  shiftrs             0