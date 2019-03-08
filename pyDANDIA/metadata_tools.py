# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 15:36:34 2018

@author: rstreet
"""

from os import path
from sys import argv
from pyDANDIA import metadata
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
import numpy as np

def find_closest_star():
    """Function to identify the star in a pyDANDIA Star Catalog closest in
    x,y pixel position to the position given"""
    
    params = get_args()
        
    (reduction_metadata, star_catalog) = fetch_metadata(params)
    
    dx = star_catalog['x']-params['x']
    dy = star_catalog['y']-params['y']
    sep = np.sqrt(dx*dx + dy*dy)
    
    idx = np.where( sep == sep.min() )
    
    j = star_catalog['star_index'][idx][0]
    xstar = star_catalog['x'][idx][0]
    ystar = star_catalog['y'][idx][0]
    rastar = star_catalog['RA_J2000'][idx][0]
    decstar = star_catalog['DEC_J2000'][idx][0]
    
    c = SkyCoord(rastar*u.degree, decstar*u.degree, frame='icrs')
    
    print('Closest star to ('+str(params['x'])+','+str(params['y'])+') is '+\
            str(j)+' at ('+str(xstar)+','+str(ystar)+') with coordinates ('+\
            str(rastar)+', '+str(decstar)+') -> '+c.to_string('hmsdms'))
    
def get_args():
    """Function to gather the necessary commandline arguments"""
    
    params = { 'metadata_file': '',
               'x': '',
               'y': '',
            }

    if len(argv) > 1:
        params['metadata_file'] = argv[1]
        params['x'] = float(argv[2])
        params['y'] = float(argv[3])
    else:
        params['metadata_file'] = input('Please enter the path to the metadata file: ')
        params['x'] = float(input('Please enter the target x position [pixels]: '))
        params['y'] = float(input('Please enter the target y position [pixels]: '))
    
    (params['red_dir'],params['metadata']) = path.split(params['metadata_file'])
    
    return params

def fetch_metadata(params):
    """Function to extract the information necessary for the photometric
    calibration from a metadata file, adding information to the params 
    dictionary"""

    (red_dir,meta_file) = path.split(params['metadata_file'])
    
    reduction_metadata = metadata.MetaData()
    reduction_metadata.load_a_layer_from_file( red_dir, 
                                              meta_file,
                                              'star_catalog' )
    
    star_catalog = Table()
    star_catalog['star_index'] = reduction_metadata.star_catalog[1]['star_index']
    star_catalog['x'] = reduction_metadata.star_catalog[1]['x_pixel']
    star_catalog['y'] = reduction_metadata.star_catalog[1]['y_pixel']
    star_catalog['RA_J2000'] = reduction_metadata.star_catalog[1]['RA_J2000']
    star_catalog['DEC_J2000'] = reduction_metadata.star_catalog[1]['DEC_J2000']
    
    return reduction_metadata, star_catalog

if __name__ == '__main__':
    
    find_closest_star()
    