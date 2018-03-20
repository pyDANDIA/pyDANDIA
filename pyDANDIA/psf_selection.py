# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 19:59:22 2017

@author: rstreet
"""

from os import path
from sys import exit
import numpy as np
import logs
import matplotlib.pyplot as plt

def psf_star_selection(setup,reduction_metadata,log,ref_star_catalog,
                                                    diagnostics=False):
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
    
    ref_star_catalog[:,15] = psf_stars_idx
    idx = np.where(ref_star_catalog[:,15] == 1.0)
    
    psf_idx = ref_star_catalog[idx[0],0]
    
    if setup.verbosity >= 1:
        
        logs.ifverbose(log,setup,'Selected PSF stars: ')
        
        for i in range(0,len(psf_idx),1):
            
            j = int(psf_idx[i])
            
            logs.ifverbose(log,setup,str(j)+' at ('+
            str(ref_star_catalog[i,1])+', '+str(ref_star_catalog[i,2])+')')
    
    # [Optionally] plot selection
    if diagnostics == True:
        
        plot_ref_star_catalog_positions(setup,reduction_metadata,log,
                                    ref_star_catalog, psf_stars_idx)
    
    return ref_star_catalog
    
def id_mid_range_stars(setup,reduction_metadata,log,ref_star_catalog,psf_stars_idx):
    """Function to identify those stars which fall in the middle of the 
    brightness range of those detected in the reference image.  That is the 
    list of stars excluding the brightest and faintest ends of the range. 
    The percentage excluded is controlled by the end_range parameter in the 
    reduction metadata. 
    """
    
    log.info('Excluding crowded stars from PSF star candidates based on brightness')
    
    star_index = np.where(psf_stars_idx == 1)[0]
    
    log.info(str(len(star_index))+' stars selected as candidates at the start')
    
    psf_range_thresh = reduction_metadata.reduction_parameters[1]['PSF_RANGE_THRESH'][0]

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
    
    star_index = np.where(psf_stars_idx == 1)[0]
    
    log.info(str(len(star_index))+' stars selected as candidates following brightness tests')
    
    return psf_stars_idx
    
def id_crowded_stars(setup,reduction_metadata,log,
                                      ref_star_catalog,psf_stars_idx):
    """Function to identify those stars with bright neightbours in close 
    proximity and deselect them as potential PSF stars. 
    
    For this purpose, any star within a distance 0.5*psf_comp_dist*psf_size, 
    in units of FWHM, of another star is considered to be a companion of that 
    star, if its flux exceeds the threshold flux ratio psf_comp_flux.
    
    XXX Note at time of writing the FWHM measurements from stage 1 for all 
    images are not yet available, so PSF size is not yet implemented
    """
    
    log.info('Excluding crowded stars from PSF star candidates based on crowding')
    
    star_index = np.where(psf_stars_idx == 1)[0]
    
    log.info(str(len(star_index))+' stars selected as candidates at the start')
    
    zp = 25.0
    
    psf_comp_dist = reduction_metadata.reduction_parameters[1]['PSF_COMP_DIST'][0]
    psf_comp_flux = reduction_metadata.reduction_parameters[1]['PSF_COMP_FLUX'][0]
    
    # XXX Fetch psf_size for the reference image from the metadata.
    psf_size = 5.0
    
    sep_thresh = 0.5 * psf_comp_dist * psf_size
    
    log.info('Excluding stars with companions of flux ratio greater than '+\
            str(psf_comp_flux))
    log.info('that are within '+str(sep_thresh)+' pixels')
        
    for j in star_index:
        
        xstar = ref_star_catalog[j,1]
        ystar = ref_star_catalog[j,2]
        fstar = mag_to_flux(zp,ref_star_catalog[j,5])
        
        log.info('Star '+str(j)+', ('+str(xstar)+','+str(ystar)+\
                    '), mag='+str(ref_star_catalog[j,5])+', flux='+str(fstar))
        
        x_near = np.where( abs((ref_star_catalog[:,1])-xstar) < sep_thresh )
        
        log.info('Star '+str(j)+': found '+str(len(x_near[0])-1)+\
                    ' potential close neighbours')
        
        x_near = x_near[0].tolist()
        jj = x_near.pop(x_near.index(j))
        
        for i in x_near:
            log.info(' -> Potential neighbour '+str(i)+\
                    ' at ('+str(ref_star_catalog[i,1])+\
                    ', '+str(ref_star_catalog[i,2])+')')
        
        # If there are neighbouring stars close in x, check separation
        if len(x_near) > 0:
            
            dx = ref_star_catalog[x_near,1] - xstar
            dy = ref_star_catalog[x_near,2] - ystar

            sep = np.sqrt( dx*dx + dy*dy )

            near = np.where(sep < sep_thresh)[0]
            
            log.info(' -> Confirmed '+str(len(near))+' close neighbours:')
            
            for k,i in enumerate(near):
                log.info(' -> '+str(x_near[i])+\
                    ' at ('+str(ref_star_catalog[x_near[i],1])+\
                    ', '+str(ref_star_catalog[x_near[i],2])+\
                    '), separation='+str(sep[k]))
                    
            # If there are neighbouring stars, check flux ratios:
            if len(near) > 0:
                
                idx_near = np.array(x_near)[near]
                
                neighbour_fluxes = mag_to_flux(zp,ref_star_catalog[idx_near,5])
                
                neighbour_fratios = fstar/neighbour_fluxes
                
                bright = np.where( neighbour_fratios >= psf_comp_flux )
                                
                log.info(' -> Found '+str(len(bright[0]))+' close, bright neighbours')
                
                for i in range(0,len(bright[0]),1):
                    k = bright[0][i]
                    log.info(' -> '+str(k)+' has mag '+\
                        str(ref_star_catalog[k,5])+', flux '+\
                        str(neighbour_fluxes[i])+', fratio '+\
                        str(neighbour_fratios[i]))
                        
                # If the neighbouring stars are too bright, deselect as 
                # a PSF star:
                if len(bright[0]) > 0:
                    
                    psf_stars_idx[j] = 0
                    
                    log.info(' -> Excluded star '+str(j)+' for excessive crowding')
    
    star_index = np.where(psf_stars_idx == 1)[0]
    
    log.info(str(len(star_index))+' stars selected as candidates following crowding tests')
    
    return psf_stars_idx


def plot_ref_star_catalog_positions(setup,reduction_metadata,log,
                                    ref_star_catalog, psf_stars_idx):
    """Function to generate a diagnostic plot of the pixel positions of stars
    in the reference star catalog."""
    
    fig = plt.figure(1)
    
    mag_data = ref_star_catalog[:,5]
    
    if mag_data.mean() < 0:
        
        mag_data = mag_data + 20.0
    
    colours = mag_data
    
    area = mag_data
    
    plt.scatter(ref_star_catalog[:,1],ref_star_catalog[:,2],marker='o',\
                                s=area, c=colours, alpha=0.5)
    
    idx = np.where(psf_stars_idx == 1)

    x = ref_star_catalog[idx,1]
    y = ref_star_catalog[idx,2]
    
    plt.plot(x,y,'+',markersize=2,markeredgecolor='r',markerfacecolor='r')

    plt.xlabel('X pixel')

    plt.ylabel('Y pixel')
    
    if mag_data.min() == mag_data.max():
        
        log.info('''Warning: Insufficient range of star magnitudes to plot a 
        colourbar for the PSF star selection''')

    else:
        plt.colorbar()

    plt.savefig(path.join(setup.red_dir,'psf_stars_selected.png'))
    
    plt.close(1)

    
def mag_to_flux(zp,mag):   
    """Function to convert a magnitude to a flux value, using the magnitude
    zeropoint given."""
    
    flux = 10**((zp - mag)/-2.5)
    
    return flux
    