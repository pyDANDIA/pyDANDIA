# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 11:21:52 2018

@author: rstreet
"""

import numpy as np

def transform_SDSS_to_JohnsonCousins(ug=None, sigug=None, gr=None, siggr=None,
                                     ri=None, sigri=None, g=None, sigg=None):
    """Function to convert SDSS-band magnitudes and colour data to the  
    Johnson/Cousins system using the transformations from
    
    Jester, S. et al. (2005), AJ, 130, 873.
    
    Valid for stars with Rc-Ic < 1.15.  The RMS uncertainties quoted in the 
    paper for these transformations are combined with the photometric 
    error on the measured quantities.
    
    Inputs are all optional but at least one colour must be provided:
    ug      float   SDSS (u-g)
    sigug   float   Uncertainty on SDSS (u-g)
    gr      float   SDSS (g-r)
    siggr   float   Uncertainty on SDSS (g-r)
    ri      float   SDSS (r-i)
    sigri   float   Uncertainty on SDSS (r-i)
    g       float   SDSS-g
    sigg    float   Uncertainty on SDSS-g
    
    Returns a results dictionary of the form:
    results = {'U-B': float or None,
               'sigUB': float or None,
               'B-V': float or None,
               'sigBV': float or None,
               'V-R': float or None,
               'sigVR': float or None,
               'Rc-Ic': float or None,
               'sigRI': float or None,
               'B': float or None,
               'sigB': float or None,
               'V': float or None,
               'sigV': float or None}
    ...where the transformed values returned depend on the optional 
    parameters provided.
    """
    
    results = {'U-B': None,
               'sigUB': None,
               'B-V': None,
               'sigBV': None,
               'V-R': None,
               'sigVR': None,
               'Rc-Ic': None,
               'sigRI': None,
               'B': None,
               'sigB': None,
               'V': None,
               'sigV': None}
    
    if ug != None:
        results['U-B'] = 0.78*ug - 0.88
        results['sigUB'] = np.sqrt( (sigug*sigug) + (0.05*0.05) )
        
    if gr != None:
        results['B-V'] = 0.98*gr + 0.22
        results['sigBV'] = np.sqrt( (siggr*siggr) + (0.04*0.04) )
    
    if ri != None:
        results['V-R'] = 1.09*ri + 0.22
        results['sigVR'] = np.sqrt( (sigri*sigri) + (0.03*0.03) )
        results['Rc-Ic'] = 1.00*ri + 0.21
        results['sigRI'] = np.sqrt( (sigri*sigri) + (0.01*0.01) )
    
    if g != None and gr != None:
        results['B'] = g + 0.39*gr + 0.21
        results['sigB'] = np.sqrt( (sigg*sigg) + (siggr*siggr) + (0.03*0.03) )
        results['V'] = g - 0.59*gr - 0.01
        results['sigV'] = np.sqrt( (sigg*sigg) + (siggr*siggr) + (0.01*0.01) )
    
    return results
    
def transform_JohnsonCousins_to_SDSS(UB=None, sigUB=None, BV=None, sigBV=None,
                                     RI=None, sigRI=None, V=None, sigV=None):
    """Function to convert Johnson/Cousins magnitudes and colour data to the  
    SDSS system using the transformations from
    
    Jester, S. et al. (2005), AJ, 130, 873.
    
    Valid for stars with Rc-Ic < 1.15.The RMS uncertainties quoted in the 
    paper for these transformations are combined with the photometric 
    error on the measured quantities.
    
    Inputs are all optional but at least one colour must be provided:
    UB      float   Johnson (U-B)
    sigUB   float   Uncertainty on Johnson (U-B)
    BV      float   Johnson (B-V)
    sigBV   float   Uncertainty on Johnson (B-V)
    RI      float   Cousins (Rc-Ic)
    sigRI   float   Uncertainty on Johnson (Rc-Ic)
    V       float   Johnson V
    sigV   float   Uncertainty on Johnson V
    
    Returns a results dictionary of the form:
    results = {'u-g': float or None,
               'sigug': float or None,
               'g-r': float or None,
               'siggr': float or None,
               'r-i': float or None,
               'sigri': float or None,
               'r-z': float or None,
               'sigrz': float or None,
               'g': float or None,
               'sigg': float or None,
               'r': float or None,
               'sigr': float or None}
    ...where the transformed values returned depend on the optional 
    parameters provided.
    """

    results = {'u-g': None,
               'sigug': None,
               'g-r': None,
               'siggr': None,
               'r-i': None,
               'sigri': None,
               'r-z': None,
               'sigrz': None,
               'g': None,
               'sigg': None,
               'r': None,
               'sigr': None}
               
    if UB != None:
        results['u-g'] = 1.28*UB + 1.13
        results['sigug'] = np.sqrt( ((results['u-g']/UB)**2 * sigUB*sigUB) + \
                                    (0.06*0.06) )
    
    if BV != None:
        results['g-r'] = 1.02*BV - 0.22
        results['siggr'] = np.sqrt( ((results['g-r']/BV)**2 * sigBV*sigBV) + \
                                    (0.04*0.04) )
    
    if RI != None:
        results['r-i'] = 0.91*RI - 0.20
        results['sigri'] = np.sqrt( ((results['r-i']/RI)**2 * sigRI*sigRI) + \
                                    (0.03*0.03) )
        results['r-z'] = 1.72*RI - 0.41
        results['sigrz'] = np.sqrt( ((results['r-z']/RI)**2 * sigRI*sigRI) + \
                                    (0.03*0.03) )
    
    if V != None and BV != None:
        results['g'] = V + 0.60*BV - 0.12
        results['sigg'] = np.sqrt( ((results['g']/V)**2 * sigV*sigV) + \
                                    ((results['g']/BV)**2 * sigBV*sigBV) + \
                                    (0.02*0.02) )
        results['r'] = V - 0.42*BV + 0.11
        results['sigr'] = np.sqrt( ((results['r']/V)**2 * sigV*sigV) + \
                                    ((results['r']/BV)**2 * sigBV*sigBV) + \
                                    (0.03*0.03) )

    return results

def calc_derived_colours_JohnsonCousins(results):
    """Function to calculate derived colours, given the results of 
    previous transformations"""
    
    results['R'] = results['V'] - results['V-R']
    results['sigR'] = np.sqrt( (results['sigV']*results['sigV']) + (results['sigVR']*results['sigVR']) ) 
    results['I'] = results['R'] - results['Rc-Ic']
    results['sigI'] = np.sqrt( (results['sigR']*results['sigR']) + (results['sigRI']*results['sigRI']) ) 
    results['V-I'] = results['V'] - results['I']
    results['sigVI'] = np.sqrt( (results['sigV']*results['sigV']) + (results['sigI']*results['sigI']) ) 

    return results
    