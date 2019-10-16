# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 14:49:11 2018

@author: rstreet
"""

from sys import argv, exit
from os import path, mkdir
from pyDANDIA import  metadata
from shutil import copyfile
from astropy.table import Column

def assign_ref_image():
    """Function to manually set the reference image to be used for a 
    reduction, independently of the automatic selection"""
    
    params = get_args_check_sanity()
    
    update_metadata(params)
    
    copy_image_to_ref(params)
    
def get_args_check_sanity():
    """Function to acquire the arguments necessary to assign a new
    reference image for a reduction"""
    
    params = {}
    
    if len(argv) == 1:
        
        params['red_dir'] = input('Please enter the path to the reduction directory: ')
        params['ref_image'] = input('Please enter the name of the image to use as reference: ')
        
    else:
        
        params['red_dir'] = argv[1]
        params['ref_image'] = argv[2]

    
    mfile = path.join(params['red_dir'],'pyDANDIA_metadata.fits')
    
    if not path.isfile(mfile):
        
        print('ERROR: Cannot find an existing metadata file for this reduction')
        print('       Please run stages 0-2 first')
        
        exit()
    
    if not path.isfile(path.join(params['red_dir'],'data',params['ref_image'])):
        
        print('ERROR: Cannot find the selected image ('+\
                            params['ref_image']+\
                            ') to use as reference')
        exit()
    
    return params

def update_metadata(params):
    """Function to edit an existing pipeline metadata file for the 
    given reduction with the name of the new reference image"""
    
    mfile = path.join(params['red_dir'],'pyDANDIA_metadata.fits')
    
    reduction_metadata = metadata.MetaData()
    
    reduction_metadata.load_a_layer_from_file( params['red_dir'], 
                                              'pyDANDIA_metadata.fits', 
                                              'data_architecture' )
    reduction_metadata.load_a_layer_from_file( params['red_dir'], 
                                              'pyDANDIA_metadata.fits', 
                                              'images_stats' )
    reduction_metadata.load_a_layer_from_file( params['red_dir'], 
                                              'pyDANDIA_metadata.fits', 
                                              'psf_dimensions' )
    
    try:
        reduction_metadata.data_architecture[1]['REF_IMAGE'][0] = params['ref_image']
    
    except KeyError:
        ref_path = path.join(params['red_dir'], 'ref')
        col1 = Column([ref_path], name='REF_PATH')
        reduction_metadata.data_architecture[1].add_column(col1)
        
        col2 = Column([params['ref_image']], name='REF_IMAGE')
        reduction_metadata.data_architecture[1].add_column(col2)

    reduction_metadata.calc_psf_radii()    
    
    reduction_metadata.save_a_layer_to_file(params['red_dir'], 
                                                'pyDANDIA_metadata.fits',
                                                'psf_dimensions')
                                                
    reduction_metadata.save_a_layer_to_file(params['red_dir'], 
                                                'pyDANDIA_metadata.fits',
                                                'data_architecture')

def copy_image_to_ref(params):
    """Function to copy the selected image into the ref subdirectory"""
    
    src = path.join(params['red_dir'],'data',params['ref_image'])
    dest = path.join(params['red_dir'],'ref',params['ref_image'])
    
    if path.isdir(path.join(params['red_dir'],'ref')) == False:
        mkdir(path.join(params['red_dir'],'ref'))
        
    copyfile(src, dest)
    

if __name__ == '__main__':
    
    assign_ref_image()
    