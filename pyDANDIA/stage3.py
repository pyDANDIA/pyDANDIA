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
from astropy.io import fits

VERSION = 'pyDANDIA_stage3_v0.1'
CODE_DIR = '/Users/rstreet/software/pyDANDIA/'

def stage3(red_dir):
    """Driver function for pyDANDIA Stage 3: 
    Detailed star find and PSF modeling
    """
    
    # How do we find the metadata to read what we need for a given reduction?
    # From reduction_control?
    # Switched off until metadata code completed
    meta = get_meta_data(red_dir)

    log = logs.start_stage_log(meta.log_dir, 'stage3', version=VERSION)
    
    status = sanity_checks(meta,log)
    if status == True:
        
        scidata = fits.getdata(meta.reference_image_path)
        scidata = scidata - meta.sky_bkgd
    
        sources = starfind.detect_sources(meta,scidata,log)
        
        
    # WCS fit
    
    # Select PSF stars
    
    # In subregions: generate PSF models
    
    # In subregions: measure PSF flux for all stars
    
    # Output data products and message reduction_control
    
class TmpMeta(object):
    """Temporary metadata object for use in code development until code
    development of the pipeline metadata is complete"""
    
    def __init__(self, red_dir):
        self.log_dir = red_dir
        self.reference_image_path = path.join(CODE_DIR,'pyDANDIA','tests','data',
                                'lsc1m005-fl15-20170701-0144-e91_cropped.fits')
        self.sky_bkgd = 1200.0
        self.sky_bkgd_sig = 100.0
        self.avg_fwhm = 6.0

def get_meta_data(red_dir):
    """Function to access those aspects of the pipeline metadata which Stage 3
    needs to operate
    """
    use_meta = False
    if use_meta == True:
        meta = metadata.MetaData()
        meta_file = path.basename(red_dir)+'_meta.fits'
        meta.load_a_layer_from_file(red_dir,meta_file,'data_architecture')
        meta.load_a_layer_from_file(red_dir,meta_file,'reduction_parameters')
    else:
        meta = TmpMeta(red_dir)
    
    return meta
    
def sanity_checks(meta,log):
    """Function to check that stage 3 has everything that it needs before
    continuing."""
    
    if path.isfile(meta.reference_image_path) == False:
        # Message reduction_control?  Return error code?
        log.info('ERROR: Stage 3 cannot access reference image at '+\
                        meta.reference_image_path)
        return False
