# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 15:14:35 2017

@author: rstreet
"""

from os import path
from astropy.io import fits
import numpy as np

def read_source_catalog(catalog_file):
    """Function to read the file of detected sources
    Expected format: ASCII file with columns:
    id xcentroid ycentroid sharpness roundness1 roundness2 npix sky peak flux mag
    """
    
    if path.isfile(catalog_file) == False:
        
        return np.zeros(1)
    
    file_lines = open(catalog_file,'r').readlines()

    data = []    
    
    for line in file_lines[1:]:

        entry = []

        for item in line.replace('\n','').split():
            entry.append(float(item))

        data.append(entry)
    
    return np.array(data)

def output_ref_catalog(catalog_file,ref_catalog):
    """Function to output a catalog of the information on sources detected
    within the reference image

    Format of output is a FITS binary table with the following columns:
    idx x  y  ra  dec  inst_mag inst_mag_err J  Jerr  H Herr   K   Kerr
    """
    
    header = fits.Header()
    header['NSTARS'] = len(ref_catalog)
    prihdu = fits.PrimaryHDU(header=header)
    
    tbhdu = fits.BinTableHDU.from_columns(\
            [fits.Column(name='Index', format='I', array=ref_catalog[:,0]),\
            fits.Column(name='X_pixel', format='E', array=ref_catalog[:,1]),\
            fits.Column(name='Y_pixel', format='E', array=ref_catalog[:,2]),\
            fits.Column(name='RA_J2000_deg]', format='D', array=ref_catalog[:,3]),\
            fits.Column(name='Dec_J2000_deg]', format='D', array=ref_catalog[:,4]),\
            fits.Column(name='Instr_mag', format='E', array=ref_catalog[:,5]),\
            fits.Column(name='Instr_mag_err', format='E', array=ref_catalog[:,6]),\
            fits.Column(name='J_mag', format='E', array=ref_catalog[:,7]),\
            fits.Column(name='J_mag_err', format='E', array=ref_catalog[:,8]),\
            fits.Column(name='H_mag', format='E', array=ref_catalog[:,9]),\
            fits.Column(name='H_mag_err', format='E', array=ref_catalog[:,10]),\
            fits.Column(name='Ks_mag', format='E', array=ref_catalog[:,11]),\
            fits.Column(name='Ks_mag_err', format='E', array=ref_catalog[:,12]),\
            ])
    
    thdulist = fits.HDUList([prihdu, tbhdu])
    
    thdulist.writeto(catalog_file,overwrite=True)
    
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
    