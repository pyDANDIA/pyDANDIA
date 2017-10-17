# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 19:59:22 2017

@author: rstreet
"""

from os import path
from sys import exit
import numpy as np

def psf_star_selection(setup,metadata,log,ref_star_catalog):
    """Function to select PSF stars from an existing list of stars detected
    in the reference image.
    
    Input parameters:
        setup       PipelineSetup object    Essential reduction parameters
        metadata    Table                   Reduction metadata
        log         logging object          Stage reduction log
        ref_star_catalog  Numpy array       Catalogue of detected stars
        
    Outputs:
        TBD
    """
    
    psf_stars_idx = np.ones(len(ref_star_catalog))
    
    # Exclude brightest and faintest 10% of star list
    psf_stars_idx = id_mid_range_stars(setup,metadata,log,
                                       ref_star_catalog,psf_stars_idx)
    
    # Calculate nearest neighbour separation
    
    # Identify stars without neighbours within proximity threshold
    # Does this need to be a function of neighbour brightness?
    
def id_mid_range_stars(setup,metadata,log,ref_star_catalog,psf_stars_idx):
    """Function to identify those stars which fall in the middle of the 
    brightness range of those detected in the reference image.  That is the 
    list of stars excluding the brightest and faintest ends of the range. 
    The percentage excluded is controlled by the end_range parameter in the 
    reduction metadata. 
    """
    
    psf_range_thresh = metadata.reduction_parameters[0]['psf_range_thresh']
    psf_comp_dist = metadata.reduction_parameters[0]['psf_comp_dist']
    psf_comp_flux = metadata.reduction_parameters[0]['psf_comp_flux']

    stars_bright_ordered = ref_star_catalog[:,5].argsort()
    
    brightest = ref_star_catalog[:,5].min()
    faintest = ref_star_catalog[:,5].max()
    
    nstar_cut = int(float(len(ref_star_catalog)) * psf_range_thresh)
    istart = nstar_cut
    iend = len(ref_star_catalog) - nstar_cut
    
    selected_idx = stars_bright_ordered[istart:iend]
    
    deselect = np.ones(len(ref_star_catalog))
    deselect[selected_idx] = 0
    
    psf_stars_idx[deselect] = 0
    
    return psf_stars_idx
    