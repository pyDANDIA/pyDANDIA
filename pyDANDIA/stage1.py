######################################################################
#
# stage1.py - For a set of images, provide in a metadata fits file: 
#                 FWHM (in pix) in x and y directions
#		  sky background
#                 correlation coefficient
#                 
# dependencies:
#       starfind.py
#
# Developed by Yiannis Tsapras
# as part of the ROME/REA LCO Key Project.
#
# version 0.1a (development)
#
# Last update: 11 Oct 2017
###############################################################################

import config_utils
import starfind
import os
import sys
import glob
import metadata
import stage0
import logs

def run_stage1(setup, verbosity=0):
    """Main driver function to run stage 1 of pyDANDIA: measurement of image 
    properties. This stage populates the metadata with the FWHM and sky 
    background for each image.
    Input: setup - is an instance of the ReductionSetup class. See 
           reduction_control.py
	   verbocity - controls the level of verbosity (default = 0): 
	                                     0 = write only in logs
					     1 = print basic messages
					     2 = print detailed messages 
    Output: updates the 'data_inventory' layer in the metadata file
    """
    
    stage1_version = 'stage1 v0.1'
    
    log = logs.start_stage_log(setup.red_dir, 'stage1', version=stage1_version)
    log.info('Setup:\n'+setup.summary()+'\n')
    
    if (verbosity >= 1):
        print ('Setup:\n'+setup.summary()+'\n')
    
    # Create or load the metadata file
    reduction_metadata = stage0.create_or_load_the_reduction_metadata(
                                    setup.red_dir, 
                                    metadata_name='pyDANDIA_metadata.fits', 
                                    verbose=True,log=log)
                                    
    # Collect the image files
    images = glob.glob(os.path.join(setup.red_dir,'data','*fits'))
    
    log.info('Analyzing '+str(len(images))+' images')
    
    if (verbosity >= 1):
        print ('Analyzing '+str(len(images))+' images')
    
    # The configuration file specifies the header information for 
    # the input images
    config_file_path = os.path.join(setup.pipeline_config_dir, 'config.json')
    conf_dict = config_utils.read_config(config_file_path)
    gain =  conf_dict['gain']['value']
    read_noise = conf_dict['ron']['value']

    # Create new layer called 'data_inventory' in the metadata file 
    # (if it doesn't already exist)
    table_structure = [
                       ['IM_NAME','FWHM_X','FWHM_Y','SKY','CORR_XY'],
                       ['S100','float','float','float','float'],
                       [None, 'arcsec', 'arcsec', 'ADU_counts', None]
                      ]
    
    reduction_metadata.create_a_new_layer(layer_name = 'data_inventory', 
                                          data_structure = table_structure,
                                          data_columns = None)
    
    log.info('Created data inventory table in metadata')
    log.info('Running starfind on all images')
    if (verbosity >= 1):
        print ('Created data inventory table in metadata')
        print ('Running starfind on all images')
	
    # For the set of given images, set the metadata information
    for im in images:
        (status, report, params) = starfind.starfind(setup, im, plot_it=False, 
                                                    log=log)
	
	# The name of the image
        imname = im.split('/')[-1]
        
	if (verbosity >= 2):
	    print ('Processing image %s' % imname)
	
        # Add a new row to the data_inventory layer 
	# (if it doesn't already exist)
        entry = [ 
	         imname, 
                 params['fwhm_x'], 
                 params['fwhm_y'], 
                 params['sky'], 
                 params['corr_xy'] 
		]
	
        reduction_metadata.add_row_to_layer(key_layer='data_inventory', 
                                            new_row=entry)
    
    # Save the updated layer to the metadata file
    reduction_metadata.save_a_layer_to_file(metadata_directory = setup.red_dir,
                                            metadata_name = 'pyDANDIA_metadata.fits',
                                            key_layer = 'data_inventory')

    log.info('Updated the data inventory table in metadata')
    if (verbosity >= 1):
        print ('Updated the data inventory table in metadata')
        print ('Completed successfully')
            
    status = 'OK'
    report = 'Completed successfully'
    return status, report
    
