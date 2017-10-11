######################################################################
#                                                                   
# stage2.py - Third stage of the pipeline. 
#             Reads metadata and selects a reference image
#
# dependencies:
#      numpy 1.8+
#      astropy 1.0+ 
######################################################################

import numpy as np
import os
from astropy.io import fits
import sys

import config_utils
from astropy.table import Table
from astropy.nddata import Cutout2D
from operator import itemgetter

import metadata
import pixelmasks
import logs
import stage0

def run_stage2(setup):
    """Main driver function to run stage 2: reference selection.

    This stage is processing the metadata file, looks for the output of 
    stages0 and stage1 and checks if a reference file already
    exists.

    It creates a reference frame based on the selection criteria
    defined in the configuration. If no such configuration exists, it 
    falls back to a standard configuration.

    If stage1 has failed to produce output it selects a reference
    based on header information.
    """
    
    stage2_version = 'stage2 v0.1'
    
    log = logs.start_stage_log(setup.red_dir, 'stage2', version=stage2_version)
    log.info('Setup:\n'+setup.summary()+'\n')

    # Create or load the metadata file
    reduction_metadata = stage0.create_or_load_the_reduction_metadata(
                                    setup.red_dir, 
                                    metadata_name='pyDANDIA_metadata.fits', 
                                    verbose=True,log=log)
                                      
    # Check data inventory on metadata

    try:
        n_images = len(red.data_inventory)
    except AttributeError:
        log.info('stage2: data inventory missing.')      
        status = 'FAILED'
        report = 'Data inventory (stage1) missing.'
        return status, report, reduction_metadata
    
    # The configuration file should specify basic parameters for the selection
    config_file_path = os.path.join(setup.pipeline_config_dir, 'config.json')
    conf_dict = config_utils.read_config(config_file_path)

    #shouldn't that come from the instrument config file?
    #or rather the header_summary?
    gain =  conf_dict['gain']['value']
    read_noise = conf_dict['ron']['value']

    #conf_dict -> instrument_dict?
    if 'MOONFRAC' in conf_dict and 'MOONDIST' in conf_dict:
        moon = moon_brightness_header(conf_dict)
    else:
        moon = 'dark'

    reference_ranking = [] 

    #to be completed

    #add ranking for data_inventory images 
#    for img in reduction_metadata.data_inventory:
#        reference_ranking.append(add_header_rank(img, header,))
#        reduction_metadata.add_row_to_layer(key_layer='data_inventory', 
#                                            new_row=entry)

#    best_image = sorted(reference_ranking, key=itemgetter(1))
        
    log.info('Updating metadata with info on new images...')

#    reduction_metadata.save_updated_metadata(
#            reduction_metadata.data_architecture[1]['REF_DIRECTORY'][0],
#            log=log)
    
    logs.close_log(log)
    
    status = 'OK'
    report = 'Completed successfully'
    
    return status, report, reduction_metadata
 


def moon_brightness_header(conf_dict):
    '''
    Based on header information, determine if image was
    taken with bright/gray or dark moon
    roughly following the ESO definitions
    https://www.eso.org/sci/observing/phase2/ObsConditions.html
    '''
    if float(header['MOONFRAC']) < 0.4:
        return 'dark'
    else:
        if (float(header['MOONFRAC']) > 0.4 and float(header['MOONFRAC']) < 0.7 and float(header['MOONDIST']) > 90.0):
            return 'gray'
        else:
            return 'bright'

def add_header_rank(image_name, header, data_inventory):
    '''Check for images that cannot be used as reference image regardless
    of their average FWHM.
    '''
    if check_header_thresholds(header)==True:
        signal_electrons = 10.0**(-0.4 * electrons_per_second_sinistro(target_magnitude)) * header['EXPTIME'] * header['GAIN']
        sky = header['L1MEDIAN']
        sky_electrons = sky * header['GAIN']
        # Simplified number of pixels for optimal aperture photometry
        npix = np.pi * header['PIXSCALE'] * (0.67 * header['L1FWHM'])**2
        readout_noise = header['RDNOISE']
        signal_to_noise_metric.append([image_name, (signal_electrons) / (signal_electrons + npix * (sky_electrons + readout_noise**2 ))**0.5])
            
        
def electrons_per_second_sinistro(mag):
    '''
    Temporary definition of electrons per second for a given
    camera. Ideally, the coefficients used here should be
     provided in each instrument file.
    '''
    return 1.47371235e+09 * 10.**(-0.4 * mag)
        

def check_header_thresholds(moon_status,stage1_entry):
    '''
    Check for images that cannot be used as reference image
    however small the FWHM was. 
    '''
    criteria = [moon_brightness_header != 'bright'
                ,int(header['NPFWHM'])>25,float(header['ELLIP'])>0.9]

        
