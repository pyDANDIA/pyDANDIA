# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 20:26:36 2018

@author: rstreet
"""
import os
import sys
import logs
import glob
import metadata
import catalog_utils
from astropy.coordinates import matching
from astropy.coordinates import SkyCoord
import astropy.units as u


VERSION = 'pyDANDIA_survey_catalog_v0.1'

def build_survey_catalog():
    """Function to build a single catalogue including all stars detected in 
    the ROME/REA survey.
    """
    
    params = get_args()
    
    log = logs.start_stage_log( params['log_dir'], 'survey_catalog', version=VERSION )
    
    params = list_reduced_datasets(params,log)

    star_catalog = None
    
    for red_dir in params['datasets']:
    
        catalog = read_star_catalog(red_dir,log)

        star_catalog = merge_catalogs(catalog, star_catalog)
    
    catalog_file = os.path.join(params['log_dir'],'survey_star_catalog.fits')
    
    catalog_utils.output_ref_catalog_file(catalog_file,star_catalog)
        
    log.info('Survey catalogue construction complete.')
    logs.close_log(log)

def merge_catalogs(catalog1, catalog2, log):
    """Function to cross match the stars in one catalog against those in 
    another catalog, and add to the latter any stars not previously known."""

    match_table = xmatch_catalogs(catalog1, catalog2)
    
    print (642 in match_table[0])
    
    idx = range(0,len(catalog1),1)
    
    # table match produces more matches than table entries
    print len(match_table[0])
    print len(idx)
    print len(catalog2)
    missing = list(set(match_table[0]) - set(idx))
    
    print missing
    
    for j in missing:
        
        data = catalog1[j,:]
        
        catalog2.add_row(data)
    
    return catalog2

def xmatch_catalogs(catalog1, catalog2):
    """Function to cross-match objects between two catalogs in astropy.Table
    format.
    Based on code by Y. Tsapras.    
    """
    
    stars1 = SkyCoord(catalog1['RA_J2000'],catalog1['DEC_J2000'],unit=u.deg)
    
    stars2 = SkyCoord(catalog2['RA_J2000'],catalog2['DEC_J2000'],unit=u.deg)

    match_table = matching.search_around_sky(stars1, stars2, seplimit=1.0*u.arcsec)

    return match_table
    
def read_star_catalog(red_dir,log):
    """Function to extract the star catalog from a given reduction of a single
    dataset, using the information in that reduction's metadata file.

    Inputs:
        :param str red_dir: Path to the reduction directory
        :param logger log: Script's own logging object
    
    Returns:
        :param Table catalog: Catalog of objects from a single reduction
    """
    
    meta_file = os.path.join(red_dir, 'pyDANDIA_metadata.fits')
    
    catalog = None
    
    if os.path.isfile(meta_file):
        
        m = metadata.MetaData()
        
        m.load_a_layer_from_file( red_dir, 
                                 'pyDANDIA_metadata.fits', 
                                 'star_catalog' )
        
        catalog = m.star_catalog[1]
        
        log.info('Read star catalog from metadata for '+\
                    os.path.basename(red_dir))
        
    return catalog
   
def list_reduced_datasets(params,log):
    """Function to identify reduced datasets to be combined into the final
    survey catalogue.
    
    Inputs:
        :param dict params: Dictionary of script parameters
        :param logger log: Script's own logging object
        
    Returns:
    
        :param dict params: Dictionary of script parameters with 'datasets' 
                            list added
    """
    
    dir_list = glob.glob(os.path.join(params['data_dir'],'ROME-FIELD*-dom?-1m0-??-fl*'))

    params['datasets'] = []
    
    for dir_path in dir_list:
        
        meta = os.path.join(dir_path,'pyDANDIA_metadata.fits')
        
        if os.path.isfile(meta):
            
            params['datasets'].append(dir_path)
    
    log.info('Found '+str(len(params['datasets']))+' datasets to process')
    
    return params

def get_args():
    """Function to gather necessary commandline arguments and perform
    sanity checks that the arguments provided are sensible.
    
    Inputs:
        None
        
    Returns:
        :param dict params: Dictionary of script parameters
    """

    params = {}
    
    if len(sys.argv) > 1:
        
        params['data_dir'] = sys.argv[1]
        params['log_dir'] = sys.argv[2]
    
    else:
        
        params['data_dir'] = raw_input('Please enter the path to the top-level reduced data directory: ')
        params['log_dir'] = raw_input('Please enter the path to the logging directory: ')

    for p in params.values():
   
       if not os.path.isdir(p):
           
           print 'ERROR: Cannot find '+p
           
           sys.exit()
           

if __name__ == '__main__':
    
    build_survey_catalog()