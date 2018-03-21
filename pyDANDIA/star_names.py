# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 17:15:15 2018

@author: rstreet
"""

import numpy as np

def get_star_ID(field_id,ra,dec):
    """Function to return a ROME/REA star identifier, given its 
    ROME field ID and RA, Dec in J2000.0 as inputs
    
    Naming convention is:
        RRFF_RA_DEC
    
    where RR = ROME/REA
          FF = Two-digit ROME field ID
          RA, DEC = star coordinates in decimal degrees
    """
    
    return 'RR'+field_id.split('_')[-1]+'_'+str(round(ra,5))+'_'+str(round(dec,5))

def get_star_names(field_id, ra_set, dec_set):
    """Function to return an array of star IDs, given the field_ID for which
    they have been measured, and arrays of RA, Dec in decimal degrees."""
    
    names = []
    for j in range(0,len(ra_set),1):
        
        names.append( get_star_ID(field_id,ra_set[j],dec_set[j]) )
    
    return np.array(names)