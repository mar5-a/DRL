lamb = 600, alpha = 1.2, 200k steps
======================================================================
COMPREHENSIVE EVALUATION
======================================================================
Episodes run                 : 20
Total steps                  : 4000
Mean return per episode      : 322.034 ± 23.704
Mean episode length          : 200.0
CRC threshold (β)            : 6.0 dB
Sub-bandwidth (B)            : 25.0 kHz (for util_bits only)

1) RELIABILITY
   Overall CRC success       : 73.90%
   Unjammed CRC success      : 83.20%
   Jammed CRC success        : 26.30%

2) JAMMING EXPOSURE
   Jam exposure fraction p_j : 16.35%
   Jam evasion rate          : 83.65%

3) SWITCHING
   Switching rate            : 93.57%
   Hop distance (mean/p50/p90): 7.03 / 6 / 14 bins

4) POWER SELECTION
   Mean power (W)            : 7.737e-04
   Median power (W)          : 8.282e-04
   % at P_min (1.58e-04)   : 0.1%
   % at P_max (1.00e-03)   : 3.2%
   Power index distribution (5 bins):
      0-10: ██ (149)
     10-20: █████ (293)
     20-30: █████████ (453)
     30-40: ████████████████████████ (1225)
     40-50: █████████████████████████████████████ (1880)

5) SINR (dB)
   Mean / Std                : 9.40 / 6.94
   Min / Median / Max        : -25.82 / 10.77 / 22.44
   Steps below β             : 1044/4000 (26.1%)

6) THROUGHPUT-LIKE (CRC-gated, bits/slot)
   Mean / p50 / p90          : 79925.00 / 92380.96 / 141682.77
======================================================================