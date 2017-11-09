# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 12:35:17 2017

@author: rstreet
"""
import os
import sys
from astropy.io import fits
import numpy as np
import logs
import metadata
import starfind
import pipeline_setup
import sky_background
import wcs
import psf
import psf_selection
import photometry

VERSION = 'pyDANDIA_stage3_v0.2'

def run_stage3(setup):
    """Driver function for pyDANDIA Stage 3: 
    Detailed star find and PSF modeling
    """
        
    log = logs.start_stage_log( setup.red_dir, 'stage3', version=VERSION )
    
    reduction_metadata = metadata.MetaData()
    reduction_metadata.load_a_layer_from_file( setup.red_dir, 
                                              'pyDANDIA_metadata.fits', 
                                              'reduction_parameters' )
    reduction_metadata.load_a_layer_from_file( setup.red_dir, 
                                              'pyDANDIA_metadata.fits', 
                                              'images_stats' )
                                            
    reduction_metadata.reference_image_path = os.path.join(setup.red_dir,'data',
                        reduction_metadata.images_stats[1]['IM_NAME'].data[0])
    reduction_metadata.background_type = 'constant'
    
    meta_pars = extract_parameters_stage3(reduction_metadata)
    
    sane = sanity_checks(reduction_metadata,log,meta_pars)
    
    if sane == True:
        
        scidata = fits.getdata(meta_pars['ref_image_path'])
        
        detected_sources = starfind.detect_sources(reduction_metadata,
                                        meta_pars['ref_image_path'],
                                        (scidata-meta_pars['ref_sky_bkgd']),
                                        log)
                                        
        ref_star_catalog = wcs.reference_astrometry(log,
                                        reduction_metadata.reference_image_path,
                                        detected_sources,
                                        diagnostics=True)        
                                                    
        sky_model = sky_background.model_sky_background(setup,
                                        reduction_metadata,log,ref_star_catalog)
                
        ref_star_catalog = psf_selection.psf_star_selection(setup,
                                        reduction_metadata,
                                        log,ref_star_catalog,
                                        diagnostics=True)
                                                     
        reduction_metadata.create_star_catalog_layer(ref_star_catalog,log=log)
        
                                                    
        psf_model = psf.build_psf(setup, reduction_metadata, log, scidata, 
                              ref_star_catalog, sky_model)
                              
        ref_star_catalog = photometry.run_psf_photometry(setup, 
                                                         reduction_metadata, 
                                                         log, 
                                                         ref_star_catalog,
                                                         meta_pars['ref_image_path'],
                                                         psf_model,
                                                         sky_model)
                                                         
        reduction_metadata.create_star_catalog_layer(ref_star_catalog,log=log)
        
        reduction_metadata.save_a_layer_to_file(setup.red_dir, 
                                                'pyDANDIA_metadata.fits',
                                                'star_catalog', log=log)
        
    logs.close_log(log)
    
    
def sanity_checks(reduction_metadata,log,meta_pars):
    """Function to check that stage 3 has all the information that it needs 
    from the reduction metadata and reduction products from earlier stages 
    before continuing.
    
    :param MetaData reduction_metadata: pipeline metadata for this dataset
    :param logging log: Open reduction log object
    :param dict meta_pars: Essential parameters from the metadata
    
    Returns:
    
    :param boolean status: Status parameter indicating if conditions are OK 
                            to continue the stage.
    """
    
    if os.path.isfile(reduction_metadata.reference_image_path) == False:
        # Message reduction_control?  Return error code?
        log.info('ERROR: Stage 3 cannot access reference image at '+\
                        reduction_metadata.reference_image_path)
        return False

    for key, value in meta_pars.items():
        
        if value == None:
            
            log.info('ERROR: Stage 3 cannot access essential metadata parameter '+key)
            
            return False
    
    log.info('Passed stage 3 sanity checks')
    
    return True

def extract_parameters_stage3(reduction_metadata):
    """Function to extract the metadata parameters necessary for this stage.
    
    :param MetaData reduction_metadata: pipeline metadata for this dataset

    Returns:

    :param dict meta_params: Dictionary of parameters and their values    
    """
    
    meta_pars = {}
    
    try:
        
        meta_pars['ref_image_path'] = reduction_metadata.reference_image_path
    
    except AttributeError:
        
        meta_pars['ref_image_path'] = None
    
    idx = np.where(reduction_metadata.images_stats[1]['IM_NAME'].data == os.path.basename(reduction_metadata.reference_image_path))
    
    if len(idx[0]) > 0:
    
        meta_pars['ref_sky_bkgd'] = reduction_metadata.images_stats[1]['SKY'].data[idx[0][0]]

    else:
        
        meta_pars['ref_sky_bkgd'] = None
        
    try:
        
        meta_pars['sky_model_type'] = reduction_metadata.background_type
    
    except AttributeError:
        
        meta_pars['sky_model_type'] = 'constant'
        
    return meta_pars
    