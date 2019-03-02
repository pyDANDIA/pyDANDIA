# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 11:09:35 2018

@author: rstreet
"""

import numpy as np

def transform_2MASS_to_SDSS(JH, HK, MH=None):
    """Function to transform photometric in the 2MASS JHKs system to SDSS, 
    using the transformation functions presented in
    
    Bilir, S et al., 2008, MNRAS, 384, 1178
    """
    
    # Coefficients:
    if MH == None:
        a3 = 1.951
        siga3 = 0.032
        b3 = 1.199
        sigb3 = 0.050
        g3 = -0.230
        sigg3 = 0.015
        
        sig_gr = 0.135
        sig_ri = 0.107
        
        a4 = 0.991
        siga4 = 0.026
        b4 = 0.792
        sigb4 = 0.042
        g4 = -0.210
        sigg4 = 0.012
        
    elif MH > -0.4:
        a3 = 1.991
        siga3 = 0.040
        b3 = 1.348
        sigb3 = 0.066
        g3 = -0.247
        sigg3 = 0.019
        
        sig_gr = 0.136
        sig_ri = 0.120
        
        a4 = 1.000
        siga4 = 0.036
        b4 = 1.004
        sigb4 = 0.064
        g4 = -0.220
        sigg4 = 0.017
        
    elif MH > -1.2 and MH <= -0.4:
        a3 = 1.217
        siga3 = 0.078
        b3 = 0.491
        sigb3 = 0.091
        g3 = 0.050
        sigg3 = 0.030
        
        sig_gr = 0.083
        sig_ri = 0.037
        
        a4 = 0.600
        siga4 = 0.035
        b4 = 0.268
        sigb4 = 0.040
        g4 = -0.049
        sigg4 = 0.013
        
    elif MH > -3.0 and MH <= -1.2:
        a3 = 1.422
        siga3 = 0.065
        b3 = 0.600
        sigb3 = 0.076
        g3 = -0.003
        sigg3 = 0.029
        
        sig_gr = 0.099
        sig_ri = 0.045
        
        a4 = 0.609
        siga4 = 0.030
        b4 = 0.279
        sigb4 = 0.035
        g4 = -0.047
        sigg4 = 0.013
        
    else:
        a3 = 1.951
        siga3 = 0.032
        b3 = 1.199
        sigb3 = 0.050
        g3 = -0.230
        sigg3 = 0.015
        
        sig_gr = 0.135
        sig_ri = 0.107
        
        a4 = 0.991
        siga4 = 0.026
        b4 = 0.792
        sigb4 = 0.042
        g4 = -0.210
        sigg4 = 0.012
        
    gr = a3*JH + b3*HK + g3
    ri = a4*JH +b4*HK + g4
    
    return gr, sig_gr, ri, sig_ri
    