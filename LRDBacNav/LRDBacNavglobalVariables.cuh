
// Macro for 2D Finite Difference
#define BLOCK_DIM_1D (256)

/*========================================================================
 * Model parameters defined as macros
 *
 * These parameters can be sent from the matlab side if the need arrises
 * with minimal effect on the performance of the code
 *========================================================================
 */


#define  LRDBacNav_RestVoltage_0  -90 

#define  LRDBacNav_R               8314 
#define  LRDBacNav_frdy            96485 
#define  LRDBacNav_temp            310 

#define  zk                  1 
#define  zna                 1 
#define  zca                 2 

#define  a                   0.0011              /*  cm  */
#define  l                   0.01                /*  cm  */
#define  pi                  3.141592   

#define  LRDBacNav_cm_0            1  

#define  nabm                140 
#define  kbm                 4.5 
#define  cabm                1.8 

#define  grelbarjsrol        4  

/* current activators  (0-not used  1-used) */
#define  LRDBacNav_ito_0           1 
#define  LRDBacNav_ikna_0          0  
#define  LRDBacNav_ikatp_0         0  
#define  LRDBacNav_insna_0         0 
#define  LRDBacNav_insk_0          0  
#define  LRDBacNav_cleft_0         0 


#define  LRDBacNav_Gna_0           16 
#define  BacNav_Gna_0			   16
#define  LRDBacNav_Gtca_0          0.05  
#define  LRDBacNav_Gkp_0           0.00552 
#define  LRDBacNav_Gkna            0.12848 
#define  LRDBacNav_Gitodv_0        0.5 
#define  LRDBacNav_Gcab_0          0.003016 
#define  LRDBacNav_Gnab_0          0.004 

#define  nicholsarea         0.00005 
#define  prnak               0.01833 
#define  pca                 0.00054 
#define  pna                 0.000000675 
#define  pk                  0.000000193 
#define  kmca                0.0006  
#define  c1                  0.00025 
#define  c2                  0.0001 
#define  gammas              0.15 
#define  Ibarnak             2.25 
#define  kmnai               10 
#define  kmko                1.5 
#define  Ibarpca             1.15 
#define  kmpca               0.0005 
#define  kdkna               66  
#define  atpi                3 
#define  katp                0.250 
#define  hatp                2 
#define  natp                0.24 
#define  gacai               1 
#define  gacao               0.341 
#define  ganai               0.75 
#define  ganao               0.75 
#define  gaki                0.75 
#define  gako                0.75 
#define  pnsca               0.000000175 
#define  kmnsca              0.0012 
#define  tautr               180 
#define  iupbar              0.00875 
#define  nsrbar              15 
#define  kmup                0.00092 
#define  taudiff             1000 
#define  gmaxrel             150 
#define  tauon               2 
#define  tauoff              2 
#define  csqnth              8.75 
#define  csqnbar             10 
#define  kmcsqn              0.8 
#define  trpnbar             0.070 
#define  kmtrpn              0.0005 
#define  cmdnbar             0.050 
#define  kmcmdn              0.00238 

#define  vcell			     1000.0*pi*a*a*l
#define  ageo			     2*pi*a*a+2*pi*a*l
#define  acap			     ageo*2
#define  vmyo                vcell*0.68
#define  vnsr                vcell*0.0552
#define  vjsr                vcell*0.0048
#define  vcleft              vcell*0.12/0.88