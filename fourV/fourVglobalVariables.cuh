
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

// Human-Epi parameters
/*
#define         tau_m_v1        60.0
#define         tau_m_v2        1150
#define         tau_p_v         1.4506
#define         tau_m_w1        60.0
#define         tau_m_w2        15
#define         tau_p_w1        200.0
#define         tau_p_w2        200.0
#define         tau_s1          2.7342
#define         tau_s2          16.0
#define         tau_fi          0.11
#define         tau_o1          400.
#define         tau_o2          6.0
#define         tau_so1         30.0181
#define         tau_so2         0.9957
#define         tau_si1         1.8875
#define         tau_si2         1.8875
#define         tau_winf        0.07
#define         theta_v         0.3
#define         theta_p_v       0.3
#define         theta_m_v       0.006
#define         theta_vinf      0.006
#define         theta_w         0.13
#define         theta_winf      0.13
#define         theta_so        0.13
#define         theta_si        0.13
#define         theta_p_si      0.13
#define         theta_si_c      0.13
#define         theta_s         0.13
#define         theta_o         0.006
#define         k_m_w           65.0
#define         k_p_w           1.0
#define         k_s             2.0994
#define         k_so            2.0458
#define         k_si            1.0
#define         k_si1           1.0
#define         k_si2           1.0
#define         k_si_c          1.0
#define         u_m_w           0.03
#define         u_s             0.9087
#define         u_o             0.0
#define         u_u             1.55
#define         u_so            0.65
#define         w_sinf          0.94
#define         w_p_c           1.0
#define         s_c             1.0
#define         alpha_w         1.0
#define         alpha_si        1.0
#define         beta_v          0.0
#define         gamma_si        1.0
#define         delta_w         1.0
#define         u_p_w           1.0
*/

// Pig-Ventricle
#define         tau_m_v1        40.0
#define         tau_m_v2        2000.0
#define         tau_p_v         10.0
#define         tau_m_w1        305.0
#define         tau_m_w2        305.0
#define         tau_p_w1        320.0
#define         tau_p_w2        320.0
#define         tau_s1          1.0
#define         tau_s2          1.0
#define         tau_fi          0.175
#define         tau_o1          4.5
#define         tau_o2          4.5
#define         tau_so1         35.0
#define         tau_so2         5.0
#define         tau_si1         1.0
#define         tau_si2         1.0
#define         tau_winf        1.0
#define         theta_v         0.25
#define         theta_p_v       0.1
#define         theta_m_v       0.0025
#define         theta_vinf      2.00
#define         theta_w         0.25
#define         theta_winf      -1.0
#define         theta_so        0.25
#define         theta_si        2.0
#define         theta_p_si      0.9
#define         theta_si_c      0.35
#define         theta_s         0.0
#define         theta_o         0.0
#define         k_m_w           0.0
#define         k_p_w           0.0
#define         k_s             0.0
#define         k_so            50.0
#define         k_si            0.0
#define         k_si1           4.5
#define         k_si2           10.0
#define         k_si_c          7.0
#define         u_m_w           0.0
#define         u_s             0.0
#define         u_o             0.0
#define         u_u             0.97
#define         u_so            0.85
#define         w_sinf          1.0
#define         w_p_c           0.0
#define         s_c             0.0
#define         alpha_w         4.0
#define         alpha_si        62.0
#define         beta_v          1.0
#define         gamma_si        0.0
#define         delta_w       1.0
#define         u_p_w         1.0

//#define        C_m              1.0
