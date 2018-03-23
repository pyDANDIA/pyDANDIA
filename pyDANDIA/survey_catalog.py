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
from astropy.table import Table, Column
from astropy.io import fits
import astropy.units as u
import numpy as np
import utilities

VERSION = 'pyDANDIA_survey_catalog_v0.1'

def build_survey_catalog():
    """Function to build a single catalogue including all stars detected in 
    the ROME/REA survey.
    """
    
    params = get_args()
    
    log = logs.start_stage_log( params['log_dir'], 'survey_catalog', version=VERSION )
        
    params = list_reduced_datasets(params,log)
    
    star_catalog = read_existing_survey_catalog(params,log)
        
    for red_dir in params['datasets']:
    
        catalog = read_star_catalog(red_dir,log)

        star_catalog = merge_catalogs(catalog, star_catalog, log)
    
    catalog_file = os.path.join(params['log_dir'],'survey_star_catalog.fits')
    
    catalog_utils.output_survey_catalog(catalog_file,star_catalog,log)
        
    log.info('Survey catalogue construction complete.')
    logs.close_log(log)

def create_survey_catalog(log,nrows,data=None):
    """Function to create an empty table in the format of a star catalog
    with sufficient space to hold the combined dataset.
    
    Columns are:
    ID RA  DEC  Blend
    
    where blend = {1: True, 0: False}
    """
    
    if data == None:
        
        star_catalog = Table([np.zeros(nrows), np.zeros(nrows), np.zeros(nrows), np.zeros(nrows)], 
              names=('ID', 'RA_J2000', 'DEC_J2000', 'Blend'), 
              meta={'name': 'survey_catalogue'},
              dtype=('i8', 'f8', 'f8', 'i1'))
              
    else:
        
        star_catalog = Table([data[0], data[1], data[2], data[3]], 
              names=('ID', 'RA_J2000', 'DEC_J2000', 'Blend'), 
              meta={'name': 'survey_catalogue'},
              dtype=('i8', 'f8', 'f8', 'i1'))
              
    log.info('Created holding table for survey star catalogue.')
    
    return star_catalog
    
def read_existing_survey_catalog(params,log):
    """Function to read an existing star catalog file, adding the information
    to the holding array.  The existing file may contain only a fraction of the
    complete dataset."""
    
    star_catalog = None
    
    if os.path.isfile( params['old_star_catalog'] ):
        
        hdu1 = fits.open(params['old_star_catalog'])
        
        old_catalog = Table( hdu1[1].data )
        
        hdu1.close()
        
        data = (old_catalog['ID'], old_catalog['RA_J2000'],old_catalog['DEC_J2000'], old_catalog['Blend'])
        
        star_catalog = create_survey_catalog(log,len(old_catalog['ID']), data)
        
        log.info('Read in pre-existing survey catalogue '+params['old_star_catalog'])
        
    else:
        
        log.info('No pre-existing survey catalogue to read in.')
        
    return star_catalog
    
    
def merge_catalogs(new_catalog, master_catalog, log):
    """Function to cross match the stars in one catalog against those in 
    another catalog, and add to the latter any stars not previously known."""

    if master_catalog == None:
        
        new_stars = new_catalog['star_index'].tolist()
        
        master_catalog = add_new_stars_to_catalog(new_stars,new_catalog,master_catalog,log)
        
    else:
     
        (match_table,blends) = xmatch_catalogs(new_catalog, master_catalog, log)
    
        log.info('Cross-matched against master star catalog, found '+\
                str(len(blends))+' blends')
    
        match_table = identify_blended_stars(match_table,blends,log)
    
        new_stars = find_new_stars(new_catalog,match_table)
    
        master_catalog = add_new_stars_to_catalog(new_stars,new_catalog,master_catalog,log)
    
    return master_catalog

def add_new_stars_to_catalog(new_stars,new_catalog,master_catalog,log):
    """Function to add stars to the master catalog
    
    Inputs:
        :param set new_stars: Set of row indices in new_catalog for new stars
        :param Table new_catalog: New reference image catalogue
        :param Table master_catalog: Master star catalogue
        :param logger log: Script log
    
    Returns:
        :param Table master_catalog: Master catalogue with new stars added
    """

    
    if master_catalog == None:
        
        master_catalog = create_survey_catalog(log,len(new_catalog))
        
        master_catalog['ID'] = Column(new_catalog['star_index'], unit=None, description='Star Identifier')
        master_catalog['RA_J2000'] = Column(new_catalog['RA_J2000'], unit=u.deg, description='Right Ascension')
        master_catalog['DEC_J2000'] = Column(new_catalog['DEC_J2000'], unit=u.deg, description='Declination')
        master_catalog['Blend'] = Column(np.zeros(len(new_catalog['star_index'])), unit=None, description='Blend flag')
    
        log.info('Transferred whole catalogue of '+str(len(new_catalog))+\
                    ' to empty master catalogue')
        
    else:
    
        for j in new_stars:
            
            row = (new_catalog['star_index'][j], 
                   new_catalog['RA_J2000'][j],
                    new_catalog['DEC_J2000'][j],
                    0)
            
            master_catalog.add_row(row)
        
            log.info('-> Adding star '+str(j)+': '+repr(row))
    
    log.info('Added '+str(len(new_stars))+' to the master catalog')
    
    return master_catalog
    
def find_new_stars(new_catalog,match_table):
    """Function to extract a list of the star indices in the new catalogue
    which have no corresponding entry in the master catalogue.
    
    Inputs:
        :param Table new_catalog: New reference image catalogue
        :param list of lists match_table: List of matched star indices
    
    Returns:
        :param list new_stars: List of indices of newly-detected stars
    """
    
    all_stars = range(0,len(new_catalog),1)
    
    new_stars = set(all_stars) - set(match_table[0])
    
    return new_stars

    
def identify_blended_stars(match_table,blends,log):
    """Function to cross-identify stars in clusters between the two catalogues.
    
    This function reviews the dictionary of blends, where one star in the new
    catalog has multiple possible matches in the master catalog.
    
    It finds the closest of all the potential matches in the master catalog, 
    and replaces the multiple entries for that object in the match_table with
    a single entry.  
    
    Inputs:
        :param list of lists match_table: Table of cross-matched star indices
        :param dict blends: Dictionary of the stars from the new catalog that
                            have been matched against multiple stars in the
                            master catalogue.
        :param logger log: Script log
    
    Returns:
        :param list of lists match_table: Amended matched table
    """
    
    if len(blends) > 0:
        log.info('Matching stars in blended clusters...')
    
    for b, entries in blends.items():
        
        (bra, bdec) = entries[0]
        
        smin = 1.0e5
        
        match = None
        
        for star in entries[1:]:
            
            s = utilities.separation_two_points(entries[0],(star[1],star[2]))
    
            if s < smin:
                
                match = star
        
        log.info(str(b)+' at ('+str(bra)+', '+str(bdec)+') matched to '+\
                str(match[0])+' at ('+str(match[1])+', '+str(match[2])+')')
        
        match_table = remove_star_from_match_tables(b,match_table)
            
        np.append(match_table[0],b)
        np.append(match_table[1],match[0])
        
    return match_table
    
def remove_star_from_match_tables(star_idx,match_table):
    """Function to remove all rows in the match table which refer to the 
    star identifier given."""
    
    for i in range(0,len(match_table[0]),1):
        
        if match_table[0][i] == star_idx:
            
            np.delete(match_table[0],i)
            
            np.delete(match_table[1],i)
    
    return match_table

def xmatch_catalogs(catalog1, catalog2, log):
    """Function to cross-match objects between two catalogs in astropy.Table
    format.
    Based on code by Y. Tsapras.    
    """
    
    stars1 = SkyCoord(catalog1['RA_J2000'],catalog1['DEC_J2000'],unit=u.deg)
    
    stars2 = SkyCoord(catalog2['RA_J2000'],catalog2['DEC_J2000'],unit=u.deg)

    match_table = matching.search_around_sky(stars1, stars2, seplimit=0.5*u.arcsec)
    
    matches1 = match_table[0].tolist()
    
    blends = {}
    
    blend_idx = set([x for x in matches1 if matches1.count(x) > 1])
    
    for b in blend_idx:
        
        idx = np.where(match_table[0] == b)[0]
        
        for i in idx:
            
            c = match_table[1][i]
            
            if b not in blends.keys():
                
                blends[b] = [ (catalog1['RA_J2000'][b],catalog1['DEC_J2000'][b]),
                              (c, catalog2['RA_J2000'][c],catalog2['DEC_J2000'][c]) ]
                
            else:
                
                blends[b].append( (c, catalog2['RA_J2000'][c],catalog2['DEC_J2000'][c]) )
            
            log.info('Match '+str(b)+' '+\
            str(catalog1['RA_J2000'][b])+' '+str(catalog1['DEC_J2000'][b])+' -> '+\
            str(c)+' '+str(catalog2['RA_J2000'][b])+' '+str(catalog2['DEC_J2000'][b]))
            
    return match_table, blends
    
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
        params['old_star_catalog'] = sys.argv[3]
    
    else:
        
        params['data_dir'] = raw_input('Please enter the path to the top-level reduced data directory: ')
        params['log_dir'] = raw_input('Please enter the path to the logging directory: ')
        params['old_star_catalog'] = raw_input('Please enter the path to an existing survey catalog if one exists, or NONE: ')

    for p in ['data_dir', 'log_dir']:
   
       if not os.path.isdir(params[p]):
           
           print 'ERROR: Cannot find '+params[p]
           
           sys.exit()
    
    return params
    
if __name__ == '__main__':
    
    build_survey_catalog()