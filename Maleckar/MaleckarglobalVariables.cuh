
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


#define  Maleckar_RestVoltage_0  -31.4
#define  Ki                       129.4349
#define  gkv_0                    0.22
#define  INaKbar                  1.355
#define  KmK                      1.0
#define  KmNa                     11.0
#define  B                       -200
#define  Vrev                    -150
#define  Ko                       5.3581
#define  Nai                      8.5547 /* 11.6 */
#define  Nao                      130.0110
#define  R                        8314.472
#define  F                        96485.3415
#define  T                        306.15
#define  GbNa_0                   0.0095
#define  gK1                      0.4822
#define  shiftrs_0                0