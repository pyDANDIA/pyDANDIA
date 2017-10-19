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
from photutils import DAOStarFinder
from astropy.modeling.fitting import LevMarLSQFitter
import starfind

def run_iterative_PSF_photometry(reduction_metadata,image_path,log):
    """Function to perform PSF-fitting photometry to all objects detected
    in an image, using DAOphot-standard routines from photutils.
    """
    
    image_data = fits.getdata(reduction_metadata.reference_image_path)
    
    psf_size = reduction_metadata.reduction_parameters[1]['PSF_SIZE'][0]
    
    fwhm = reduction_metadata.reduction_parameters[1]['FWHM'][0]
    
    psf_radius = psf_size * fwhm
    
    star_finder = starfind.build_star_finder(reduction_metadata, image_data, log)
    
    daogroup = DAOGroup(2.0*fwhm)
    
    sigma_clip = SigmaClip(sigma=3.0)

    mmm_bkg = MMMBackground(sigma_clip=sigma_clip)
    
    fitter = LevMarLSQFitter()
    
    psf_model = IntegratedGaussianPRF(sigma=fwhm)
    
    psf_x = calc_psf_dimensions(psf_size,fwhm)
    
    photometry = IterativelySubtractedPSFPhotometry(finder=star_finder,
                                                    group_maker=daogroup,
                                                    bkg_estimator=mmm_bkg,
                                                    psf_model=psf_model,
                                                    fitter=LevMarLSQFitter(),
                                                    niters=1, 
                                                    fitshape=(psf_x,psf_x))

def calc_psf_dimensions(psf_size,fwhm):
    """Function to calculate the image stamp dimensions to be used for each 
    PSF.  This has to be an odd number to work with photutils. """
    
    psf_x = psf_size * fwhm
    
    if np.mod(psf_x,2.0) == 0.0:
        psf_x += 1.0
    
    psf_x = int(psf_x)
    
    return psf_x