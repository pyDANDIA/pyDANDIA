# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 19:59:22 2017

@author: rstreet
"""

from os import path
from sys import exit
import numpy as np
from pyDANDIA import  logs
import matplotlib as mpl
mpl.use('Agg')
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

    # Exclude stars with neighbours within proPrestonManorximity threshold that
    # also exceed the flux ratio threshold:
    psf_stars_idx = id_crowded_stars(setup,reduction_metadata,log,
                                      ref_star_catalog,psf_stars_idx)

    # Ensure the pipeline configuration's limit on the maximum number of
    # PSF stars is respected:
    psf_stars_idx = apply_psf_star_limit(reduction_metadata,ref_star_catalog,
                                         psf_stars_idx,log)

    ref_star_catalog[:,11] = psf_stars_idx
    psf_idx = np.where(ref_star_catalog[:,11] == 1.0)[0]

    if setup.verbosity >= 1:

        if diagnostics:
            output_psf_star_index(psf_stars_idx, ref_star_catalog, log, final=True)
            log.info('Selected '+str(len(psf_idx))+' PSF stars')

    # [Optionally] plot selection
    if diagnostics == True:

        plot_ref_star_catalog_positions(setup,reduction_metadata,log,
                                    ref_star_catalog, psf_stars_idx)

    return ref_star_catalog

def id_mid_range_stars(setup,reduction_metadata,log,ref_star_catalog,
        psf_stars_idx,diagnostics=False):
    """Function to identify those stars which fall in the middle of the
    brightness range of those detected in the reference image.  That is the
    list of stars excluding the brightest and faintest ends of the range.
    The percentage excluded is controlled by the end_range parameter in the
    reduction metadata.
    """

    log.info('Excluding crowded stars from PSF star candidates based on brightness')

    star_index = np.where(psf_stars_idx == 1)[0]

    log.info(str(len(star_index))+' stars selected as candidates at the start')

    psf_range_thresh_lower = reduction_metadata.reduction_parameters[1]['PSF_RANGE_THRESH_LOWER'][0]
    psf_range_thresh_upper = reduction_metadata.reduction_parameters[1]['PSF_RANGE_THRESH_UPPER'][0]

    log.info('Excluding top '+str(psf_range_thresh_upper)+\
            ' and bottom '+str(psf_range_thresh_lower)+\
            '%  of a star catalog of '+str(len(ref_star_catalog))+' stars')

    stars_bright_ordered = ref_star_catalog[:,5].argsort()
    mask = (ref_star_catalog[:,5]>10.0) & (psf_stars_idx == 1)
    #brightest = ref_star_catalog[mask,5].min()
    #faintest = ref_star_catalog[mask,5].max()
    brightest = np.percentile(ref_star_catalog[:, 5][mask], 100-psf_range_thresh_upper)
    faintest = np.percentile(ref_star_catalog[:, 5][mask], psf_range_thresh_lower)

    log.info('Allowed flux range of stars: '+str(faintest)+' - '+str(brightest))


    mask_brightness = (ref_star_catalog[:, 5] < brightest) & (
                ref_star_catalog[:, 5] > faintest) & (psf_stars_idx == 1)


    psf_stars_idx[mask_brightness] = 1
    psf_stars_idx[~mask_brightness] = 0
    log.info(str(len(psf_stars_idx[mask_brightness]))+' stars selected as candidates following brightness tests')

    if diagnostics:
        output_psf_star_index(psf_stars_idx, ref_star_catalog, log)

    return psf_stars_idx

def id_crowded_stars(setup,reduction_metadata,log,
                                      ref_star_catalog,psf_stars_idx,
                                      diagnostics=False):
    """Function to identify those stars with bright neightbours in close
    proximity and deselect them as potential PSF stars.

    For this purpose, any star within a distance 0.5*psf_comp_dist*psf_diameter,
    in units of FWHM, of another star is considered to be a companion of that
    star, if its flux exceeds the threshold flux ratio psf_comp_flux.
    """

    log.info('Excluding crowded stars from PSF star candidates based on crowding')

    star_index = np.where(psf_stars_idx == 1)[0]

    log.info(str(len(star_index))+' stars selected as candidates at the start')

    zp = 25.0

    psf_comp_dist = reduction_metadata.reduction_parameters[1]['PSF_COMP_DIST'][0]
    psf_comp_flux = reduction_metadata.reduction_parameters[1]['PSF_COMP_FLUX'][0]
    psf_diameter = reduction_metadata.psf_dimensions[1]['psf_radius'][0]*2.0

    sep_thresh = 0.5 * psf_comp_dist * psf_diameter

    log.info('Excluding stars with companions of flux ratio greater than '+\
            str(psf_comp_flux))
    log.info('that are within '+str(sep_thresh)+' pixels')

    for j in star_index:

        xstar = ref_star_catalog[j,1]
        ystar = ref_star_catalog[j,2]
        fstar = mag_to_flux(zp,ref_star_catalog[j,5])

        x_near = np.where( abs((ref_star_catalog[:,1])-xstar) < sep_thresh )

        if diagnostics:
            log.info('Star '+str(j)+', ('+str(xstar)+','+str(ystar)+\
                    '), mag='+str(ref_star_catalog[j,5])+', flux='+str(fstar))

            log.info('Star '+str(j)+': found '+str(len(x_near[0])-1)+\
                    ' potential close neighbours')

        x_near = x_near[0].tolist()
        jj = x_near.pop(x_near.index(j))


        if diagnostics:
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

            if diagnostics:
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

                if diagnostics:
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

                    if diagnostics:
                        log.info(' -> Excluded star '+str(j)+' for excessive crowding')

    star_index = np.where(psf_stars_idx == 1)[0]

    log.info(str(len(star_index))+' stars selected as candidates following crowding tests')


    if diagnostics:
        output_psf_star_index(psf_stars_idx, ref_star_catalog, log)

    return psf_stars_idx

def apply_psf_star_limit(reduction_metadata,ref_star_catalog,psf_stars_idx,log,
    diagnostics=False):
    """Function to ensure that the configured limit to the number of PSF
    stars selected is respected"""

    max_psf_stars = reduction_metadata.reduction_parameters[1]['MAX_PSF_STARS'][0]

    psf_idx = np.where(psf_stars_idx == 1)[0]

    if len(psf_idx) > max_psf_stars:
        mask = []
        while len(mask) < max_psf_stars:
            i = np.random.randint(0,len(psf_idx))
            if i not in mask:
                mask.append(i)

        psf_idx = psf_idx[mask]

        psf_stars_idx = np.zeros(len(ref_star_catalog))
        psf_stars_idx[psf_idx] = 1

        log.info('Limited the number of PSF stars via random selection to configured maximum of '+\
                    str(max_psf_stars)+', length psf star index '+str(len(psf_stars_idx)))


    if diagnostics:
        output_psf_star_index(psf_stars_idx, ref_star_catalog, log)

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

def output_psf_star_index(psf_stars_idx, ref_star_catalog, log, final=False):

    if not final:
        log.info('Current PSF star selection:')
    else:
        log.info('Final selection of PSF stars:')

    for i in range(0,len(psf_stars_idx),1):
        if psf_stars_idx[i] == 1.0:
            j = int(i)
            log.info(str(j+1)+' at ('+
            str(ref_star_catalog[j,1])+', '+str(ref_star_catalog[j,2])+
            '), ref_flux='+str(ref_star_catalog[j,5]))

    npsf = len(np.where(psf_stars_idx == 1.0)[0])
    log.info(str(npsf)+' PSF stars selected')
    log.info('Length of psf_star_idx: '+str(len(psf_stars_idx)))
