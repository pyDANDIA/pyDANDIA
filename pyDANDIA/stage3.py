# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 12:35:17 2017

@author: rstreet
"""
import os
import sys
from astropy.io import fits
import numpy as np
from pyDANDIA import  logs
from pyDANDIA import  metadata
from pyDANDIA import  starfind
from pyDANDIA import  pipeline_setup
from pyDANDIA import  sky_background
from pyDANDIA import  wcs
from pyDANDIA import  psf
from pyDANDIA import  psf_selection
from pyDANDIA import  photometry

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
    reduction_metadata.load_a_layer_from_file( setup.red_dir, 
                                              'pyDANDIA_metadata.fits', 
                                              'data_architecture' )
    
    sane = sanity_checks(reduction_metadata,log,meta_pars)
        
    if sane:
        
        meta_pars = extract_parameters_stage3(reduction_metadata)
    
        scidata = fits.getdata(meta_pars['ref_image_path'])
        
        detected_sources = starfind.detect_sources(setup, reduction_metadata,
                                        meta_pars['ref_image_path'],
                                        (scidata-meta_pars['ref_sky_bkgd']),
                                        log,
                                        diagnostics=False)

        ref_star_catalog = wcs.reference_astrometry(setup,log,
                                        meta_pars['ref_image_path'],
                                        detected_sources,
                                        diagnostics=False)        
                                                    
        sky_model = sky_background.model_sky_background(setup,
                                        reduction_metadata,log,ref_star_catalog)
                
        ref_star_catalog = psf_selection.psf_star_selection(setup,
                                        reduction_metadata,
                                        log,ref_star_catalog,
                                        diagnostics=False)
                                                     
        reduction_metadata.create_star_catalog_layer(ref_star_catalog,log=log)
        
                                                    
        (psf_model,psf_status) = psf.build_psf(setup, reduction_metadata, 
                                            log, scidata, 
                                            ref_star_catalog, sky_model)
        
        ref_star_catalog = photometry.run_psf_photometry(setup, 
                                             reduction_metadata, 
                                             log, 
                                             ref_star_catalog,
                                             meta_pars['ref_image_path'],
                                             psf_model,
                                             sky_model,
                                             centroiding=True)
                                                         
        reduction_metadata.create_star_catalog_layer(ref_star_catalog,log=log)
        
        reduction_metadata.save_a_layer_to_file(setup.red_dir, 
                                                'pyDANDIA_metadata.fits',
                                                'star_catalog', log=log)
        
    status = 'OK'
    report = 'Completed successfully'
    
    log.info('Stage 3: '+report)
    logs.close_log(log)
    
    return status, report
    
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

    if 'REF_PATH' not in reduction_metadata.data_architecture[1].keys():
        
        log.info('ERROR: Stage 3 cannot find path to reference image in metadata')
        
        return False

    ref_path =  str(reduction_metadata.data_architecture[1]['REF_PATH'][0]) +'/'+ str(reduction_metadata.data_architecture[1]['REF_IMAGE'][0])
    if not os.path.isfile(ref_path):
        # Message reduction_control?  Return error code?
        log.info('ERROR: Stage 3 cannot access reference image at '+\
                        ref_path)
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
        
        meta_pars['ref_image_path'] = str(reduction_metadata.data_architecture[1]['REF_PATH'][0]) +'/'+ str(reduction_metadata.data_architecture[1]['REF_IMAGE'][0])
    
    except AttributeError:
        
        meta_pars['ref_image_path'] = None
    
    idx = np.where(reduction_metadata.images_stats[1]['IM_NAME'].data ==  reduction_metadata.data_architecture[1]['REF_IMAGE'][0])


    if len(idx[0]) > 0:
    
        meta_pars['ref_sky_bkgd'] = reduction_metadata.images_stats[1]['SKY'].data[idx[0][0]]

    else:
        
        meta_pars['ref_sky_bkgd'] = None

	
        
    try:
        
        meta_pars['sky_model_type'] = reduction_metadata.reduction_parameters[1]['BACK_VAR'][0]

    except AttributeError:
        
        meta_pars['sky_model_type'] = 'constant'
        
    return meta_pars
    
