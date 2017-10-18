# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 19:59:22 2017

@author: rstreet
"""

from os import path
from sys import exit
import numpy as np
import logs

def psf_star_selection(setup,reduction_metadata,log,ref_star_catalog):
    """Function to select PSF stars from an existing list of stars detected
    in the reference image.
    
    Input parameters:
        setup       PipelineSetup object    Essential reduction parameters
        metadata    Table                   Reduction metadata
        log         logging object          Stage reduction log
        ref_star_catalog    Numpy array     Catalogue of detected stars
        
    Outputs:
        psf_stars_idx       Numpy array     [0,1] indicating whether each 
                                            star in the ref_star_catalog is 
                                            selected.  
    """
    
    psf_stars_idx = np.ones(len(ref_star_catalog))
    
    # Exclude brightest and faintest 10% of star list
    psf_stars_idx = id_mid_range_stars(setup,reduction_metadata,log,
                                       ref_star_catalog,psf_stars_idx)
    
    # Exclude stars with neighbours within proximity threshold that
    # also exceed the flux ratio threshold:
    psf_stars_idx = id_crowded_stars(setup,reduction_metadata,log,
                                      ref_star_catalog,psf_stars_idx)
    
    # [Optionally] plot selection
    
    return psf_stars_idx
    
def id_mid_range_stars(setup,reduction_metadata,log,ref_star_catalog,psf_stars_idx):
    """Function to identify those stars which fall in the middle of the 
    brightness range of those detected in the reference image.  That is the 
    list of stars excluding the brightest and faintest ends of the range. 
    The percentage excluded is controlled by the end_range parameter in the 
    reduction metadata. 
    """
    
    psf_range_thresh = reduction_metadata.reduction_parameters[1]['PSF_RANGE_THRESH'][0]
    psf_comp_dist = reduction_metadata.reduction_parameters[1]['PSF_COMP_DIST'][0]
    psf_comp_flux = reduction_metadata.reduction_parameters[1]['PSF_COMP_FLUX'][0]

    log.info('Excluding top and bottom '+str(psf_range_thresh)+\
    '%  of a star catalog of '+str(len(ref_star_catalog))+' stars')
    
    stars_bright_ordered = ref_star_catalog[:,5].argsort()
    
    brightest = ref_star_catalog[:,5].min()
    faintest = ref_star_catalog[:,5].max()
    
    log.info('Brightness range of stars: '+str(brightest)+' - '+str(faintest))
    
    nstar_cut = int(float(len(ref_star_catalog)) * (psf_range_thresh/100.0))
    istart = nstar_cut
    iend = len(ref_star_catalog) - nstar_cut

    log.info('Deselecting '+str(nstar_cut)+' stars from the catalog')
    log.info('Selecting '+str(istart)+' to '+str(iend)+\
            ' ('+str(iend-istart+1)+') stars from the catalog')
    
    selected_idx = stars_bright_ordered[istart:iend]
    
    deselect = np.array([True]*len(ref_star_catalog))
    deselect[selected_idx] = False
    
    psf_stars_idx[deselect] = 0
    
    return psf_stars_idx
    
def id_crowded_stars(setup,reduction_metadata,log,
                                      ref_star_catalog,psf_stars_idx):
    """Function to identify those stars with bright neightbours in close 
    proximity and deselect them as potential PSF stars. 
    
    For this purpose, any star within a distance 0.5*psf_comp_dist*psf_size, 
    in units of FWHM, of another star is considered to be a companion of that 
    star, if its flux exceeds the threshold flux ratio psf_comp_flux.
    """
    
    zp = 25.0
    
    psf_range_thresh = reduction_metadata.reduction_parameters[1]['PSF_RANGE_THRESH'][0]
    psf_comp_dist = reduction_metadata.reduction_parameters[1]['PSF_COMP_DIST'][0]
    psf_comp_flux = reduction_metadata.reduction_parameters[1]['PSF_COMP_FLUX'][0]
    
    star_index = np.where(psf_stars_idx == 1)
        
    for j in star_index:
        
        xstar = ref_star_catalog[j,1]
        ystar = ref_star_catalog[j,2]
        fstar = mag_to_flux(zp,ref_star_catalog[j,5])
        
        x_near = np.where( (ref_star_catalog[:,1])-xstar < psf_comp_dist )
        
        # If there are neighbouring stars close in x, check separation
        if len(x_near) > 0:
            
            dx = ref_star_catalog[x_near,1]) - xstar
            dy = ref_star_catalog[x_near,1]) - ystar
            sep = np.sqrt( dx*dx + dy*dy )
            
            near = np.where(sep < psf_comp_dist)
            
            # If there are neighbouring stars, check flux ratios:
            if len(near) > 0:
                
                idx_near = ref_star_catalog[x_near[near],0]
                
                neighbour_fluxes = mag_to_flux(zp,ref_star_catalog[idx_near,5])
                
                bright = np.where( (neighbour_fluxes/fstar) >= psf_comp_flux )
                
                # If the neighbouring stars are too bright, deselect as 
                # a PSF star:
                if len(bright) > 0:
                    
                    psf_stars_idx[j] = 0
                    
    return psf_stars_idx
    
    
def mag_to_flux(zp,mag):   
    """Function to convert a magnitude to a flux value, using the magnitude
    zeropoint given."""
    
    flux = 10**((zp - mag)/-2.5)
    
    return flux
    