# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 16:50:57 2018

@author: rstreet
"""

from sys import argv
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
import photometry_classes


def get_essential_parameters(RC=None):
    """Function to provide the published values for the essential parameters
    of Red Clump giant stars.
    Sources:
    Bensby et al. (2017), 2017, A&A, 605A, 89 for V, I bands and
    Ruiz-Dern et al. (2017), A&A, 609A, 116 for Gaia, 2MASS, SDSS and Wise bands.
    """
    
    if RC == None:
        RC = photometry_classes.Star()
        
    RC.M_I_0 = -0.12
    RC.sig_MI_0 = 0.0
    RC.VI_0 = 1.09
    RC.sig_VI_0 = 0.0
    RC.M_V_0 = 0.97
    RC.sig_MV_0 = 0.0
    
    RC.M_G_0 = 0.495
    RC.sig_MG_0 = 0.009
    RC.M_J_0 = -0.945
    RC.sig_MJ_0 = 0.01
    RC.M_H_0 = -1.450
    RC.sig_MH_0 = 0.017
    RC.M_Ks_0 = -1.606
    RC.sig_MKs_0 = 0.009
    RC.M_g_0 = 1.331
    RC.sig_Mg_0 = 0.056
    RC.M_r_0 = 0.552
    RC.sig_Mr_0 = 0.026
    RC.M_i_0 = 0.262
    RC.sig_Mi_0 = 0.032
    RC.M_W1_0 = -1.711
    RC.sig_MW1_0 = 0.017
    RC.M_W2_0 = -1.585
    RC.sig_MW2_0 = 0.016
    RC.M_W3_0 = -1.638
    RC.sig_MW3_0 = 0.011
    RC.M_W4_0 = -1.704
    RC.sig_MW4_0 = 0.012

    RC.JH_0 = RC.M_J_0 - RC.M_H_0
    RC.sig_JH_0 = np.sqrt( (RC.sig_MJ_0*RC.sig_MJ_0) + \
                                (RC.sig_MH_0*RC.sig_MH_0) )
    RC.HK_0 = RC.M_H_0 - RC.M_Ks_0
    RC.sig_HK_0 = np.sqrt( (RC.sig_MH_0*RC.sig_MH_0) + \
                                (RC.sig_MKs_0*RC.sig_MKs_0) )
    
    RC.gr_0 = RC.M_g_0 - RC.M_r_0
    RC.sig_gr_0 = np.sqrt( (RC.sig_Mg_0*RC.sig_Mg_0) + \
                                (RC.sig_Mr_0*RC.sig_Mr_0) )
    RC.gi_0 = RC.M_g_0 - RC.M_i_0
    RC.sig_gi_0 = np.sqrt( (RC.sig_Mg_0*RC.sig_Mg_0) + \
                                (RC.sig_Mi_0*RC.sig_Mi_0) )
    RC.ri_0 = RC.M_r_0 - RC.M_i_0
    RC.sig_ri_0 = np.sqrt( (RC.sig_Mr_0*RC.sig_Mr_0) + \
                                (RC.sig_Mi_0*RC.sig_Mi_0) )
    
    return RC
    
def calc_red_clump_distance(ra,dec,log=None):
    """Function to estimate the distance of the Red Clump stars in the
    Galactic Bulge from the observer on Earth, taking into account the 
    bar structure, using the relations from 
    Nataf, D. M., Gould, A., Fouqué, P., et al. 2013, ApJ, 769, 88
    """
    
    c = SkyCoord(ra, dec, unit=(u.hourangle,u.degree), frame='icrs')
    
    output = 'Galactic coordinates for '+ra+', '+dec+' (l,b) [deg]: '+\
            str(c.galactic.l.deg)+', '+str(c.galactic.b.deg)+'\n'
    
    R_0 = 8.16 # Kpc
    phi = 40.0 * (np.pi/180.0)
    
    D_RC = R_0 / (np.cos(c.galactic.l.radian) + np.sin(c.galactic.l.radian)*(1.0/np.tan(phi)))
    
    output += 'Red Clump distance for ('+str(c.galactic.l.deg)+', '+\
                                    str(c.galactic.b.deg)+') = '+\
                                    str(D_RC)+' Kpc'
    if log != None:
        log.info(output)
    else:
        print(output)
        
    return D_RC

def calc_I_apparent(D_RC):
    """Function to return the apparent magnitude in I-band of the Red Clump
    stars given their distance from the observer, D_RC, and the measured 
    apparent I magnitude of Red Clump stars in the Galactic Center, 
    published by 
    Nataf, D. M., Gould, A., Fouqué, P., et al. 2013, ApJ, 769, 88
    """

    R_0 = 8.16 # Kpc
    
    delta_I = 5.0 * np.log10(R_0/D_RC)
    
    I_RC_app = 14.443 + delta_I
    
    print('\nOffset in delta_I magnitude of Red Clump apparent magnitude, due to separation from the Galactic centre: '+str(delta_I)+'mag')
    print('Apparent I_RC at this distnace, I_RC_app = '+str(I_RC_app)+'mag')
    
    return I_RC_app

def calc_apparent_magnitudes(RC):
    """Function to calculate the apparent magnitudes of Red Clump stars
    at a given distance from the observer, based on their absolute magnitudes
    """
    
    # Absolute magnitudes are for a distance of 10pc by definition:
    R_0 = 10.0 # pc
    
    delta_m = 5.0 * np.log10(RC.D*1000.0) - 5.0
    
    passbands = [ 'B', 'V', 'I', 'G', 'J', 'H', 'Ks', 'g', 'r', 'i', 'W1', 'W2', 'W3', 'W4' ]
    
    for f in passbands:
        
        abs_mag = 'M_'+f+'_0'
        abs_mag_err = 'sig_M'+f+'_0'
        app_mag = 'm_'+f+'_0'
        app_mag_err = 'sig_m'+f+'_0'
        
        if abs_mag in dir(RC):
            
            setattr(RC,app_mag, (getattr(RC,abs_mag) + delta_m))
            setattr(RC,app_mag_err, (getattr(RC,abs_mag_err)))
            
    return RC
    
def calc_distance(params):
    '''Function to calculate the distance to a star from its apparent and
    absolute magnitudes, using the standard distance modulus expression:
    D = 10^( m - M + 5 )/5
    '''
    
    def calc_D(m,M):
        log_D = (m - M + 5.0 ) / 5.0
        D = 10**( log_D )
        return log_D,D
	
    (log_D,D) = calc_D( params['app_mag'], params['abs_mag'] )
    (log_D_min,D_min) = calc_D( params['app_mag']-params['sig_app_mag'], \
                             params['abs_mag']-params['sig_abs_mag'] )
    (log_D_max,D_max) = calc_D( params['app_mag']+params['sig_app_mag'], \
                             params['abs_mag']+params['sig_abs_mag'] )
    sig_D = ( D_max - D_min ) / 2.0
    
    print('Distance = '+str(D)+'+/-'+str(sig_D)+'pc')


if __name__ == '__main__':
    
    if len(argv) == 1:
        ra = input('Please enter the RA (sexigesimal, J2000.0): ')
        dec = input('Please enter the Dec (sexigesimal, J2000.0): ')
    else:
        ra = argv[1]
        dec = argv[2]
    
    D_RC = calc_red_clump_distance(ra,dec)
    