# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 15:42:26 2017

@author: rstreet
"""

import os
import sys
import numpy as np
import logs
import metadata
import matplotlib.pyplot as plt
from astropy.io import fits
import starfind
import psf

def run_psf_photometry(setup,reduction_metadata,log,ref_star_catalog,
                       image_path,psf_model,sky_model,centroiding=True):
    """Function to perform PSF fitting photometry on all stars for a single
    image.
    
    :param SetUp object setup: Essential reduction parameters
    :param MetaData reduction_metadata: pipeline metadata for this dataset
    :param logging log: Open reduction log object
    :param array ref_star_catalog: catalog of objects detected in the image
    :param str image_path: Path to image to be photometered
    :param PSFModel object psf_model: PSF to be fitted to each star
    :param BackgroundModel object sky_model: Model for the image sky background
    :param boolean centroiding: Switch to (dis)-allow re-fitting of each star's
                                x, y centroid.  Default=allowed (True)
    
    Returns:
    
    :param array ref_star_catalog: catalog of objects detected in the image
    """
    
    log.info('Starting photometry of '+os.path.basename(image_path))
    
    data = fits.getdata(image_path)
    residuals = np.copy(data)
    
    psf_size = reduction_metadata.reduction_parameters[1]['PSF_SIZE'][0]
    half_psf = int(float(psf_size)/2.0)
    
    logs.ifverbose(log,setup,'Applying '+psf_model.psf_type()+\
                    ' PSF of diameter='+str(psf_size))
    
    Y_data, X_data = np.indices((int(psf_size),int(psf_size)))
    
    for j in range(0,len(ref_star_catalog),1):
        
        xstar = ref_star_catalog[j,1]
        ystar = ref_star_catalog[j,2]
        
        X_grid = X_data + (int(xstar) - half_psf)
        Y_grid = Y_data + (int(ystar) - half_psf)
        
        logs.ifverbose(log,setup,' -> Star '+str(j)+' at position ('+\
        str(xstar)+', '+str(ystar)+')')
        
        fitted_model = psf.fit_star_existing_model(setup, data, 
                                               xstar, ystar, half_psf, 
                                               psf_model, sky_model, 
                                               centroiding=centroiding,
                                               diagnostics=True)
        
        logs.ifverbose(log, setup,' -> Star '+str(j)+
        ' fitted model parameters = '+repr(fitted_model.get_parameters()))
        
        sub_psf_model = psf.get_psf_object('Moffat2D')
        
        pars = fitted_model.get_parameters()
        pars[1] = psf_size + (ystar-int(ystar))
        pars[2] = psf_size + (xstar-int(xstar))
        
        sub_psf_model.update_psf_parameters(pars)
        
        residuals = psf.subtract_psf_from_image(residuals,sub_psf_model,
                                                xstar,ystar,
                                                psf_size, psf_size)
                                             
        
        logs.ifverbose(log, setup,' -> Star '+str(j)+
        ' subtracted PSF from the residuals')
                
        (flux, flux_err) = fitted_model.calc_flux(Y_grid, X_grid)
        
        (mag, mag_err) = convert_flux_to_mag(flux, flux_err)
        
        ref_star_catalog[j,5] = mag
        ref_star_catalog[j,6] = mag_err
        
        logs.ifverbose(log,setup,' -> Star '+str(j)+
        ' flux='+str(flux)+' +/- '+str(flux_err)+' ADU, '
        'mag='+str(mag)+' +/- '+str(mag_err)+' mag')
    
    res_image_path = image_path.replace('.fits','_res.fits')
    
    hdu = fits.PrimaryHDU(residuals)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(res_image_path, overwrite=True)
    
    logs.ifverbose(log, setup, 'Output residuals image '+res_image_path)

    log.info('Completed photometry')
    
    return ref_star_catalog
    
    
def convert_flux_to_mag(flux, flux_err):
    """Function to convert the flux of a star from its fitted PSF model 
    and its uncertainty onto the magnitude scale.
    
    :param float flux: Total star flux
    :param float flux_err: Uncertainty in star flux
    
    Returns:
    
    :param float mag: Measured star magnitude
    :param float flux_mag: Uncertainty in measured magnitude
    """
    def flux2mag(ZP, flux):
        
        return ZP - 2.5 * np.log10(flux)
    
    if flux < 0.0 or flux_err < 0.0:
        
        mag = 0.0
        mag_err = 0.0

    else:        

        ZP = 25.0
        
        mag = flux2mag(ZP, flux)
        
        m1 = flux2mag(ZP, (flux + flux_err))
        m2 = flux2mag(ZP, (flux - flux_err))
        
        mag_err = (m2 - m1)/2.0
    
    return mag, mag_err