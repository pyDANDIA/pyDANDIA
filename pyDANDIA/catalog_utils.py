# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 15:14:35 2017

@author: rstreet
"""

from os import path
from astropy.io import fits
import numpy as np
from astropy import table

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

def output_ref_catalog_file(catalog_file,ref_catalog):
    """Function to output a catalog of the information on sources detected
    within the reference image

    Format of output is a FITS binary table with the following columns:
    idx x  y  ra  dec  ref_flux ref_flux_err ref_mag  ref_mag_err J  Jerr  H Herr   K   Kerr psf_star
    """
    
    header = fits.Header()
    header['NSTARS'] = len(ref_catalog)
    prihdu = fits.PrimaryHDU(header=header)
    
    tbhdu = fits.BinTableHDU.from_columns(\
            [fits.Column(name='Index', format='I', array=ref_catalog[:,0]),\
            fits.Column(name='X_pixel', format='E', array=ref_catalog[:,1]),\
            fits.Column(name='Y_pixel', format='E', array=ref_catalog[:,2]),\
            fits.Column(name='RA_J2000_deg', format='D', array=ref_catalog[:,3]),\
            fits.Column(name='Dec_J2000_deg', format='D', array=ref_catalog[:,4]),\
            fits.Column(name='ref_flux', format='E', array=ref_catalog[:,5]),\
            fits.Column(name='ref_flux_err', format='E', array=ref_catalog[:,6]),\
            fits.Column(name='ref_mag', format='E', array=ref_catalog[:,7]),\
            fits.Column(name='ref_mag_err', format='E', array=ref_catalog[:,8]),\
            fits.Column(name='J_mag', format='E', array=ref_catalog[:,9]),\
            fits.Column(name='J_mag_err', format='E', array=ref_catalog[:,10]),\
            fits.Column(name='H_mag', format='E', array=ref_catalog[:,11]),\
            fits.Column(name='H_mag_err', format='E', array=ref_catalog[:,12]),\
            fits.Column(name='Ks_mag', format='E', array=ref_catalog[:,13]),\
            fits.Column(name='Ks_mag_err', format='E', array=ref_catalog[:,14]),\
            fits.Column(name='psf_star', format='I', array=ref_catalog[:,15]),\
            ])
    
    thdulist = fits.HDUList([prihdu, tbhdu])
    
    thdulist.writeto(catalog_file,overwrite=True)

def read_ref_star_catalog_file(catalog_file):
    """Function to read an external star_catalog file in FITS binary table
    format
    
    If the star_cat_file cannot be found, a unit-length array is returned. 
    """
    
    if path.isfile(catalog_file) == False:
        
        return np.zeros(1)
    
    hdulist = fits.open(catalog_file)
    
    data = hdulist[1].data
    
    ref_star_catalog = []

    for i in range(0,len(data),1):
        
        ref_star_catalog.append( list( data[i] ) )
    
    ref_star_catalog = np.array( ref_star_catalog )
    
    return ref_star_catalog

def output_survey_catalog(catalog_file,star_catalog,log):
    """Function to output a survey catalog file
    
    Format of output is a FITS binary table with the following columns:
    idx x  y  ra  dec  Blend
    
    where Blend is a {0,1}={False,True} flag
    """
    
    header = fits.Header()
    header['NSTARS'] = len(star_catalog)
    prihdu = fits.PrimaryHDU(header=header)
    
    tbhdu = fits.BinTableHDU.from_columns(\
            [fits.Column(name='ID', format='I', array=star_catalog['ID']),\
            fits.Column(name='RA_J2000', format='E', array=star_catalog['RA_J2000']),\
            fits.Column(name='DEC_J2000', format='E', array=star_catalog['DEC_J2000']),\
            fits.Column(name='Blend', format='I', array=star_catalog['Blend']),])
    
    thdulist = fits.HDUList([prihdu, tbhdu])
    
    thdulist.writeto(catalog_file,overwrite=True)
    
    log.info('Output star catalogue to '+catalog_file)
    

def output_vphas_catalog_file(catalog_file,vphas_catalog,match_index=None):
    """Function to output a catalog of the information on sources detected
    within the reference image

    Format of output is a FITS binary table with the following columns:
    idx ra  dec  gmag  e_gmag   rmag   e_rmag   imag   e_imag   clean
    """

    header = fits.Header()
    header['NSTARS'] = len(vphas_catalog)
    prihdu = fits.PrimaryHDU(header=header)
    
    col_list = [fits.Column(name='_RAJ2000', format='D', array=vphas_catalog['_RAJ2000']),\
            fits.Column(name='_DEJ2000', format='D', array=vphas_catalog['_DEJ2000']),\
            fits.Column(name='gmag', format='E', array=vphas_catalog['gmag']),\
            fits.Column(name='e_gmag', format='E', array=vphas_catalog['e_gmag']),\
            fits.Column(name='rmag', format='E', array=vphas_catalog['rmag']),\
            fits.Column(name='e_rmag', format='E', array=vphas_catalog['e_rmag']),\
            fits.Column(name='imag', format='E', array=vphas_catalog['imag']),\
            fits.Column(name='e_imag', format='E', array=vphas_catalog['e_imag']),\
            fits.Column(name='clean', format='I', array=vphas_catalog['clean'])]
                
    if type(match_index) == type(np.zeros(1)):
        
        idx = np.zeros(len(vphas_catalog))
        
        idx[match_index[:,1]] = match_index[:,0]
        
        col_list.append(fits.Column(name='match_star_index', format='I', array=idx))
        
    tbhdu = fits.BinTableHDU.from_columns(col_list)
    
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

def output_vizier_catalog(path_to_output_file, catalog):
    
    col_names = ['_RAJ2000', '_DEJ2000', 'Jmag', 'e_Jmag', \
                 'Hmag', 'e_Hmag', 'Kmag', 'e_Kmag']
    
    formats = [ 'D', 'D', 'E', 'E', 'E', 'E', 'E', 'E' ]
    
    units = [ 'deg', 'deg', 'mag', 'mag', 'mag', 'mag', 'mag', 'mag' ]
    
    col_list = []
    for i in range(0,len(col_names),1):
        
        col_list.append(fits.Column(name=col_names[i], 
                                    format=formats[i], 
                                    unit=units[i],
                                    array=catalog[col_names[i]].data))
    
    phdu = fits.PrimaryHDU()
    tbhdu = fits.BinTableHDU.from_columns(col_list)
    thdulist = fits.HDUList([phdu, tbhdu])
    thdulist.writeto(path_to_output_file, overwrite=True)

def read_vizier_catalog(path_to_cat_file):
    
    if path.isfile(path_to_cat_file):
        
        data = fits.getdata(path_to_cat_file)
        
        table_data = [ table.Column(name='_RAJ2000', data=data.field(0)),
                  table.Column(name='_DEJ2000', data=data.field(1)),
                  table.Column(name='Jmag', data=data.field(2)),
                  table.Column(name='e_Jmag', data=data.field(3)),
                  table.Column(name='Hmag', data=data.field(4)),
                  table.Column(name='e_Hmag', data=data.field(5)),
                  table.Column(name='Kmag', data=data.field(4)),
                  table.Column(name='e_Kmag', data=data.field(5)) ]
            
        new_table = table.Table(data=table_data)
    
    else:
        
        new_table = None
        
    return new_table
    