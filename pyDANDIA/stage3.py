# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 12:35:17 2017

@author: rstreet
"""
from os import path, getcwd
from sys import exit
import logs
import metadata
import starfind
import pipeline_setup
from astropy.io import fits

VERSION = 'pyDANDIA_stage3_v0.2'
CODE_DIR = '/Users/rstreet/software/pyDANDIA/'

def run_stage3(setup):
    """Driver function for pyDANDIA Stage 3: 
    Detailed star find and PSF modeling
    """
        
    log = logs.start_stage_log( cwd, 'stage3', version=VERSION )
    
    reduction_metadata = metadata.MetaData()
    reduction_metadata.load_a_layer_from_file( setup.red_dir, 
                                              'pyDANDIA_metadata.fits', 
                                              'reduction_parameters' )
    reduction_metadata.load_a_layer_from_file( setup.red_dir, 
                                              'pyDANDIA_metadata.fits', 
                                              'images_stats' )


    status = sanity_checks(reduction_metadata,log)
    
    if status == True:
        
        scidata = fits.getdata(meta.reference_image_path)
        
        # SKY BACKGROUND MODEL
        
        # GIVE BOTH UNCHANGED SCIDATA + SKY MODEL
        sources = starfind.detect_sources(meta,(scidata-meta.sky_bkgd),log)
        
        ref_source_catalog = wcs.reference_astrometry(log,
                                                    reduction_metadata.reference_image_path,
                                                    detected_sources,
                                                    diagnostics=True)
    
        reduction_metadata.create_star_catalog_layer(ref_source_catalog,log=log)
        
        psf_stars_idx = psf_selection.psf_star_selection(setup,reduction_metadata,
                                                     log,ref_star_catalog,
                                                     diagnostics=True)
    
        psf_model = psf.build_psf(setup, reduction_metadata, log, scidata, 
                              ref_star_catalog, psf_stars_idx, sky_model)
                              
        
    # In subregions: measure PSF flux for all stars
    
    # Output data products and message reduction_control
    
    
def sanity_checks(meta,log):
    """Function to check that stage 3 has everything that it needs before
    continuing."""
    
    if path.isfile(meta.reference_image_path) == False:
        # Message reduction_control?  Return error code?
        log.info('ERROR: Stage 3 cannot access reference image at '+\
                        meta.reference_image_path)
        return False
