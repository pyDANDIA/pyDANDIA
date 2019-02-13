# -*- coding: utf-8 -*-
"""
Created on Wed May 23 17:45:56 2018

@author: rstreet
"""

import sys
from os import path
from pyDANDIA import metadata
from astropy.table import Table
from astropy.coordinates import SkyCoord

def offset_star_catalog_coordinates():
    """Function to apply an offset in RA, Dec to correct the star catalog
    for an erroneous WCS fit"""
    
    params = get_args()
    
    (star_catalog, reduction_metadata) = read_star_catalog(params)
    
    star_catalog = apply_offset(star_catalog,params)

    update_metadata(reduction_metadata,star_catalog,params)

def update_metadata(reduction_metadata,star_catalog,params):
    """Function to output the updated columns of the star catalog"""
    
    reduction_metadata.star_catalog[1]['RA_J2000'] = star_catalog['RA']
    reduction_metadata.star_catalog[1]['DEC_J2000'] = star_catalog['DEC']

    reduction_metadata.save_a_layer_to_file(params['red_dir'], 
                                            path.basename(params['metadata']),
                                             'star_catalog')
                             
def apply_offset(star_catalog, params):
    """Function to apply the RA, Dec offset to all stars in the star_catalog"""
    
    stars = SkyCoord(star_catalog['RA'], star_catalog['DEC'], unit="deg")
    
    new_stars = SkyCoord(stars[:].ra.value + params['delta_ra'], 
                         stars[:].dec.value + params['delta_dec'], unit="deg")
    
    star_catalog['RA'] = new_stars.ra.value
    star_catalog['DEC'] = new_stars.dec.value
    
    return star_catalog
    
def read_star_catalog(params):
    """Function to read in the star catalog layer of an existing metadata file"""
    
    meta_file = params['metadata']
    
    reduction_metadata = metadata.MetaData()
    reduction_metadata.load_a_layer_from_file( path.dirname(meta_file), path.basename(meta_file), 'star_catalog' )

    star_catalog = Table()
    star_catalog['RA'] = reduction_metadata.star_catalog[1]['RA_J2000']
    star_catalog['DEC'] = reduction_metadata.star_catalog[1]['DEC_J2000']

    return star_catalog, reduction_metadata
    
def get_args():
    """Function to gather the required arguments"""
    
    params = {}
    
    if len(sys.argv) == 1:
        
        params['metadata'] = raw_input('Please enter the path to the metadata file: ')
        params['delta_ra'] = float(raw_input('Please the RA offset in decimal arcsec: '))
        params['delta_dec'] = float(raw_input('Please enter the Dec offset in decimal arcsec: '))
        
    else:

        params['metadata'] = sys.argv[1]
        params['delta_ra'] = float(sys.argv[2])
        params['delta_dec'] = float(sys.argv[3])
        
    params['delta_ra'] = params['delta_ra']/3600.0
    params['delta_dec'] = params['delta_dec']/3600.0
    params['red_dir'] = path.dirname(params['metadata'])
    
    return params

if __name__ == '__main__':
    
    offset_star_catalog_coordinates()