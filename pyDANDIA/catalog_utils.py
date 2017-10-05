# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 15:14:35 2017

@author: rstreet
"""

from os import path
from astropy.io import fits

def extract_star_catalog(star_cat_file, ra_min=None, dec_min=None, 
                                      ra_max=None, dec_max=None):
    """Function to read a catalogue of stars in standard FITS binary table 
    format.  
    
    If an RA, Dec and radius are given, the function will return a catalogue 
    of all stars within the search box of that location found within the main 
    catalogue. 
    
    Otherwise, the entire catalogue will be returned. 
    
    If the star_cat_file cannot be found, a unit-length array is returned. 
    
    Input:
        star_cat_file   str               Path to input star catalog
        ra_min          float, optional   RA of target star [decimal deg]
        dec_min         float, optional   Dec of target star [decimal deg]
        ra_max          float, optional   RA of target star [decimal deg]
        dec_max         float, optional   Dec of target star [decimal deg]
    Output:
        star_cat        np.array          Array of star catalogue data
    """
    
    if path.isfile(star_cat_file) == False:
        return np.zeros(1)
    
    catalog = fits.open(star_cat_file,'r')
    
    table_idx = get_tables_to_search(catalog.header,ra_min,dec_min,
                                         ra_max,dec_max)
    data = []
    for t in table_idx:
        table = catalog  # READ APPROPRIATE TABLE EXTENSION
        for j in range(0,len(table),1):
            if table[j,0] >= ra_min and table[j,0] <= ra_max and \
                table[j,1] >= dec_min and table[j,1] <= dec_max:
                data.append( [ table[j,0], table[j,1], table[j,2] ] )
    star_cat = np.array(data)
    
    return star_cat

def get_tables_to_search(cat_header,ra_min,dec_min,ra_max,dec_max):
    """Function to identify which of the catalog's binary table extensions 
    will contain data of interest.  
    In the case that no star specified, this will be the entire table list.
    If a search box has been given, this function will return a list of tables 
    containing entries within that search box.
    """

    tables = {}
    for record in cat_header:
        if 'RA_MIN' in record[0] or 'RA_MAX' in record[0] \
            or 'DEC_MIN' in record[0] or 'DEC_MAX' in record[0]:
            idx = record[0].split('_')[-1]
            if idx in tables.keys():
                table_entry = tables[idx]
            else:
                table_entry = []
            par = record[0][:-2]
            table_entry[par] = float(record[1])
            tables[idx] = table_entry
    
    if ra_min == None:
        table_idx = tables.keys()
    else:
        table_idx = []
        for idx, entry in tables.items():
            if ra_max > entry['RA_MIN'] and dec_max > entry['DEC_MAX'] and \
                ra_min < entry['RA_MAX'] and dec_min < entry['DEC_MAX']:
                table_idx.append(idx)
    
    table_idx.sort()
    
    return table_idx
    