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
from photutils.detection import IRAFStarFinder
from photutils.psf import IntegratedGaussianPRF, DAOGroup
from photutils.background import SigmaClip, MMMBackground, MADStdBackgroundRMS
from photutils import DAOStarFinder, IterativelySubtractedPSFPhotometry
from astropy.modeling.fitting import LevMarLSQFitter
import starfind

def run_iterative_PSF_photometry(setup,reduction_metadata,image_path,log,
                                 diagnostics=False):
    """Function to perform PSF-fitting photometry to all objects detected
    in an image, using DAOphot-standard routines from photutils.
    """
    
    log.info('Performing PSF-fitting photometry on '+os.path.basename(image_path))
    
    image_data = fits.getdata(image_path)
    
    psf_size = reduction_metadata.reduction_parameters[1]['PSF_SIZE'][0]
    
    image_idx = reduction_metadata.images_stats[1]['IM_NAME'].tolist().index(os.path.basename(image_path))
    
    fwhm = reduction_metadata.images_stats[1]['FWHM_X'][image_idx]
    
    log.info('Applying psf size = '+str(psf_size))
    log.info('         fwhm = '+str(fwhm))

    psf_radius = psf_size * fwhm
    
    log.info('         psf size = '+str(psf_radius))
    
    star_finder = starfind.build_star_finder(reduction_metadata, image_path, log)
    
    daogroup = DAOGroup(2.0*fwhm)
    
    log.info(' -> Build star grouping object')
    
    sigma_clip = SigmaClip(sigma=3.0)

    mmm_bkg = MMMBackground(sigma_clip=sigma_clip)
    
    log.info(' -> Build sky background model')
    
    fitter = LevMarLSQFitter()
    
    psf_model = IntegratedGaussianPRF(sigma=fwhm)
    
    log.info(' -> Build PSF model')
    
    psf_x = calc_psf_dimensions(psf_size,fwhm,log)
    
    photometer = IterativelySubtractedPSFPhotometry(finder=star_finder,
                                                    group_maker=daogroup,
                                                    bkg_estimator=mmm_bkg,
                                                    psf_model=psf_model,
                                                    fitter=fitter,
                                                    niters=3, 
                                                    fitshape=(psf_x,psf_x))

    photometry = photometer(image=image_data)
    print photometry.colnames
    
    log.info('Photometry warnings, if any: '+repr(fitter.fit_info['message']))
    
    log.info('Executed photometry of '+str(len(photometry))+' stars')
    
    if diagnostics:
        store_residual_image(setup,photometer,image_path,log)
    
    return photometry

def calc_psf_dimensions(psf_size,fwhm,log):
    """Function to calculate the image stamp dimensions to be used for each 
    PSF.  This has to be an odd number to work with photutils. """
    
    psf_x = psf_size * fwhm
    if np.mod(int(psf_x),2.0) == 0.0:
        psf_x = int(psf_x) + 1
    else:
        psf_x = int(psf_x)
    
    log.info(' -> Calculated PSF stamp pixel dimensions = '+\
            str(psf_x)+'x'+str(psf_x))
    
    return psf_x

def store_residual_image(setup,photometer,image_path,log):
    """Function to store the residual image produced following PSF-fitting
    photometry to the filesystem."""
    
    try:
        residual_image = photometer.get_residual_image()
    
        hdu = fits.PrimaryHDU(residual_image)
        
        hdulist = fits.HDUList([hdu])
        
        res_image_name = os.path.basename(image_path).replace('.fits','_res.fits')
        res_image_path = os.path.join(os.path.dirname(image_path),res_image_name)
        
        hdulist.writeto(res_image_path,overwrite=True)

        log.info('Output PSF-photometry residual image to '+res_image_path)
        
    except:
        
        log.info('Error: fault occurred during output of PSF-photometry residual image')
        