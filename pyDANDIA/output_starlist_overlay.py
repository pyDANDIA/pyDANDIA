# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 09:42:25 2018

@author: rstreet
"""

from sys import argv
from os import path
from pyDANDIA import  metadata

def output_starlist_regions():
    """Function to output a list of all stars in the star catalog in DS9s
    regions format so that it can be easily overlaid"""
    
    params = get_args()
    
    check_sanity(params)
    
    reduction_metadata = metadata.MetaData()
    reduction_metadata.load_a_layer_from_file( params['red_dir'], 
                                              'pyDANDIA_metadata.fits', 
                                              'star_catalog' )
                                              
    output_star_catalog_regions(params, reduction_metadata)
    
def get_args():
    """Function to gather the required user inputs"""
    
    params = {}
    
    if len(argv) == 1:
        params['red_dir'] = raw_input('Please enter the path to the reduction directory: ')
        
    else:
        params['red_dir'] = argv[1]
    
    params['metadata'] = path.join(params['red_dir'],'pyDANDIA_metadata.fits')
    params['output'] = path.join(params['red_dir'],'ref','star_catalog.reg')
    
    return params

def check_sanity(params):
    """Function to ensure all necessary input is available"""
    
    if path.isdir(params['red_dir']) == False:
        raise IOError('Cannot find reduction directory '+params['red_dir'])
        exit()
        
    if path.isfile(params['metadata']) == False:
        raise IOError('Cannot find the reduction metadata file '+params['metadata'])
        exit()

def output_star_catalog_regions(params, reduction_metadata):
    """Function to output all stars detected in the star_catalog 
    table in a file in the DS9 regions format.
    
    Note that this function adds 1 pixel to the star positions in x, y, to 
    accomodate the display offset caused by the difference between python 
    numbering from zero and FITS numbering from 1.    
    """
    
    f = open(params['output'],'w')
    
    for i in range(0,len(reduction_metadata.star_catalog[1]),1):
        
        f.write('point '+str(reduction_metadata.star_catalog[1]['x_pixel'][i]+1.0)+\
                    ' '+str(reduction_metadata.star_catalog[1]['y_pixel'][i]+1.0)+'\n')
    
    f.close()
    
if __name__ == '__main__':
    output_starlist_regions()