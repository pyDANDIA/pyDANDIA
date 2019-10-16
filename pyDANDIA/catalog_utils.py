# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 15:14:35 2017

@author: rstreet
"""

from os import path
from astropy.io import fits
import numpy as np
from astropy import table

def read_source_catalog(catalog_file,table_format=False):
    """Function to read the file of detected sources
    Expected format: ASCII file with columns:
    id xcentroid ycentroid sharpness roundness1 roundness2 npix sky peak flux mag
    """
    
    if path.isfile(catalog_file) == False:
        
        return np.zeros(1)
    
    file_lines = open(catalog_file,'r').readlines()

    header = str(file_lines[0]).replace('\n','').split()
    
    data = []    
    
    for line in file_lines[1:]:

        entry = []

        for item in line.replace('\n','').split():
            entry.append(float(item))

        data.append(entry)
    
    data = np.array(data)
    
    if table_format:
        
        tdata = []
        
        for i,col in enumerate(header):
            
            if i==0:
                c = table.Column(name='index', data=data[:,i])
            else:
                c = table.Column(name=col, data=data[:,i])
            
            tdata.append(c)
    
        data = table.Table(data=tdata)
    
    return data

def output_ref_catalog_file(catalog_file,ref_catalog):
    """Function to output a catalog of the information on sources detected
    within the reference image
    """
    
    header = fits.Header()
    header['NSTARS'] = len(ref_catalog)
    prihdu = fits.PrimaryHDU(header=header)
    
    tbhdu = fits.BinTableHDU.from_columns(\
            [fits.Column(name='Index', format='I', array=ref_catalog['index'].data),\
            fits.Column(name='X_pixel', format='E', array=ref_catalog['x'].data),\
            fits.Column(name='Y_pixel', format='E', array=ref_catalog['y'].data),\
            fits.Column(name='ra', format='D', array=ref_catalog['ra'].data),\
            fits.Column(name='dec', format='D', array=ref_catalog['dec'].data),\
            fits.Column(name='ref_flux', format='E', array=ref_catalog['ref_flux'].data),\
            fits.Column(name='ref_flux_error', format='E', array=ref_catalog['ref_flux_error'].data),\
            fits.Column(name='ref_mag', format='E', array=ref_catalog['ref_mag'].data),\
            fits.Column(name='ref_mag_error', format='E', array=ref_catalog['ref_mag_error'].data),\
            fits.Column(name='cal_ref_mag', format='E', array=ref_catalog['cal_ref_mag'].data),\
            fits.Column(name='cal_ref_mag_error', format='E', array=ref_catalog['cal_ref_mag_error'].data),\
            fits.Column(name='Gaia_source_id', format='A30', array=ref_catalog['gaia_source_id'].data),\
            fits.Column(name='Gaia_ra', format='D', array=ref_catalog['gaia_ra'].data),\
            fits.Column(name='Gaia_ra_error', format='E', array=ref_catalog['gaia_ra_error'].data),\
            fits.Column(name='Gaia_dec', format='D', array=ref_catalog['gaia_dec'].data),\
            fits.Column(name='Gaia_dec_error', format='E', array=ref_catalog['gaia_dec_error'].data),\
            fits.Column(name='phot_g_mean_flux', format='E', array=ref_catalog['phot_g_mean_flux'].data),\
            fits.Column(name='phot_g_mean_flux_error', format='E', array=ref_catalog['phot_g_mean_flux_error'].data),\
            fits.Column(name='phot_bp_mean_flux', format='E', array=ref_catalog['phot_bp_mean_flux'].data),\
            fits.Column(name='phot_bp_mean_flux_error', format='E', array=ref_catalog['phot_bp_mean_flux_error'].data),\
            fits.Column(name='phot_rp_mean_flux', format='E', array=ref_catalog['phot_rp_mean_flux'].data),\
            fits.Column(name='phot_rp_mean_flux_error', format='E', array=ref_catalog['phot_rp_mean_flux_error'].data),\
            fits.Column(name='VPHAS_source_id', format='A30', array=ref_catalog['vphas_source_id'].data),\
            fits.Column(name='VPHAS_ra', format='D', array=ref_catalog['vphas_ra'].data),\
            fits.Column(name='VPHAS_dec', format='E', array=ref_catalog['vphas_dec'].data),\
            fits.Column(name='gmag', format='E', array=ref_catalog['gmag'].data),\
            fits.Column(name='gmag_error', format='E', array=ref_catalog['gmag_error'].data),\
            fits.Column(name='rmag', format='E', array=ref_catalog['rmag'].data),\
            fits.Column(name='rmag_error', format='E', array=ref_catalog['rmag_error'].data),\
            fits.Column(name='imag', format='E', array=ref_catalog['imag'].data),\
            fits.Column(name='imag_error', format='E', array=ref_catalog['imag_error'].data),\
            fits.Column(name='clean', format='E', array=ref_catalog['clean'].data),\
            fits.Column(name='psf_star', format='I', array=ref_catalog['psf_star'].data),\
            ])
    
    thdulist = fits.HDUList([prihdu, tbhdu])
    
    thdulist.writeto(catalog_file,overwrite=True)

def load_ref_star_catalog_from_metadata(reduction_metadata):
    """Function to extract a reference star catalog as a np.array in the 
    format expected by the photometry.
    
    idx x  y  ra  dec  ref_flux  ref_flux_err ref_mag ref_mag_err   cal_ref_mag   cal_ref_mag_error
    """
    
    ref_star_catalog = np.zeros((len(reduction_metadata.star_catalog[1]),14))
    
    ref_star_catalog[:,0] = reduction_metadata.star_catalog[1]['index'].data
    ref_star_catalog[:,1] = reduction_metadata.star_catalog[1]['x'].data
    ref_star_catalog[:,2] = reduction_metadata.star_catalog[1]['y'].data
    ref_star_catalog[:,3] = reduction_metadata.star_catalog[1]['ra'].data
    ref_star_catalog[:,4] = reduction_metadata.star_catalog[1]['dec'].data
    ref_star_catalog[:,5] = reduction_metadata.star_catalog[1]['ref_flux'].data
    ref_star_catalog[:,6] = reduction_metadata.star_catalog[1]['ref_flux_error'].data
    ref_star_catalog[:,7] = reduction_metadata.star_catalog[1]['ref_mag'].data
    ref_star_catalog[:,8] = reduction_metadata.star_catalog[1]['ref_mag_error'].data
    ref_star_catalog[:,9] = reduction_metadata.star_catalog[1]['cal_ref_mag'].data
    ref_star_catalog[:,10] = reduction_metadata.star_catalog[1]['cal_ref_mag_error'].data
    ref_star_catalog[:,11] = reduction_metadata.star_catalog[1]['psf_star'].data
    ref_star_catalog[:,12] = reduction_metadata.star_catalog[1]['sky_background'].data
    ref_star_catalog[:,13] = reduction_metadata.star_catalog[1]['sky_background_error'].data
    
    return ref_star_catalog

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

def output_vizier_catalog(path_to_output_file, catalog, catalog_source):
    
    if catalog_source == '2MASS':
    
        col_names = ['_RAJ2000', '_DEJ2000', 'Jmag', 'e_Jmag', \
                     'Hmag', 'e_Hmag', 'Kmag', 'e_Kmag']
    
        formats = [ 'D', 'D', 'E', 'E', 'E', 'E', 'E', 'E' ]
        
        units = [ 'deg', 'deg', 'mag', 'mag', 'mag', 'mag', 'mag', 'mag' ]
        
    elif catalog_source == 'Gaia':
        col_names = ['ra','dec','source_id','ra_error','dec_error',
                     'phot_g_mean_flux','phot_g_mean_flux_error',
                     'phot_rp_mean_flux','phot_rp_mean_flux_error',
                     'phot_bp_mean_flux','phot_bp_mean_flux_error']
                     
        formats = [ 'D', 'D', 'A30', 'E', 'E', 'E', 'E', 'E', 'E','E', 'E' ]
    
        units = [ 'deg', 'deg', '', 'mas', 'mas', 
                 "'electron'.s**-1", "'electron'.s**-1",
                 "'electron'.s**-1", "'electron'.s**-1",
                 "'electron'.s**-1", "'electron'.s**-1"]
    
    else:
        raise IOError('Unrecognized catalog source '+catalog_source)
        exit()
        
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

def read_vizier_catalog(path_to_cat_file, catalog_source):
    
    if path.isfile(path_to_cat_file):
        
        data = fits.getdata(path_to_cat_file)
        
        if catalog_source == '2MASS':
            
            col_names = ['_RAJ2000', '_DEJ2000', 'Jmag', 'e_Jmag', \
                     'Hmag', 'e_Hmag', 'Kmag', 'e_Kmag']
                     
        elif catalog_source == 'Gaia':
            
            col_names = ['ra','dec','source_id','ra_error','dec_error',
                     'phot_g_mean_flux','phot_g_mean_flux_error',
                     'phot_rp_mean_flux','phot_rp_mean_flux_error',
                     'phot_bp_mean_flux','phot_bp_mean_flux_error']
                     
        else:
            raise IOError('Unrecognized catalog source '+catalog_source)
            exit()
            
        table_data = [ ]
        
        for i in range(0,len(col_names),1):
            
            table_data.append( table.Column(name=col_names[i], data=data.field(i)) )
            
                     
        new_table = table.Table(data=table_data)
    
    else:
        
        new_table = None
        
    return new_table
    
def read_Gaia_vizier_catalog(path_to_cat_file):
    
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

def output_ds9_overlay_from_table(catalog,file_path,radius=None,colour='green',
                                  transformed_coords=False):
    """Function to output from an astropy Table that includes x,y 
    image coordinates as Columns named 'x' and 'y' as an region overlay
    file for use in DS9"""
    
    if transformed_coords:
        xcol = 'x1'
        ycol = 'y1'
    else:
        xcol = 'x'
        ycol = 'y'
        
    f = open(file_path,'w')
    for j in range(0,len(catalog),1):
        
        if radius != None:
            
            f.write('circle '+str(catalog[xcol][j])+' '+str(catalog[ycol][j])+
                ' '+str(radius)+' # color='+colour+'\n')
            
        else:
            
            f.write('point '+str(catalog[xcol][j])+' '+str(catalog[ycol][j])+
                ' # color='+colour+'\n')
                
    f.close()
    