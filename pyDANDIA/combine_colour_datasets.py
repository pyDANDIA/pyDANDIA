# -*- coding: utf-8 -*-
"""
Created on Tue May 15 20:48:12 2018

@author: rstreet
"""

import sys
from os import path
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.coordinates import matching
import astropy.units as u
from pyDANDIA import metadata
from pyDANDIA import logs
import numpy as np
import select_image_trios

VERSION = 'combine_colour_datasets_0.0.1'

def combine_colour_datasets():
    """Function to plot colour magnitude and colour-colour plots"""
    
    datasets = {'ip': None, 'rp': None, 'gp': None,
                'ip_images': None, 'rp_images': None, 'gp_images': None}
    
    params = get_args()
    
    log = logs.start_stage_log( params['red_dir'], 'combine_colour_datasets', version=VERSION )
    
    for f in ['ip', 'rp', 'gp']:
        
        if params[f] != None:
            
            (datasets[f],datasets[f+'_images']) = extract_star_catalog(params,f,log)
        
    if datasets.values().count(None) >= 2:
        
        log.info('ERROR: Data available for only 1 passband, cannot produce figures')
        logs.close_log(log)

        exit()
        
    (combined_catalog,col_names,formats,units,f1,f2,f3) = combine_star_catalogs(datasets,log)
    
    image_trios = identify_image_trios(params,datasets,log)
    
    output_combined_catalog(combined_catalog,col_names,formats,units,f1,f2,f3,
                            image_trios,params,log)
    
    logs.close_log(log)
    
def get_args():
    """Function to gather the necessary commandline arguments"""

    params = {}
    
    if len(sys.argv) == 1:
        
        params['ip'] = raw_input('Please enter the path to the metadata file for SDSS-i [or none]: ')
        params['rp'] = raw_input('Please enter the path to the metadata file for SDSS-r [or none]: ')
        params['gp'] = raw_input('Please enter the path to the metadata file for SDSS-g [or none]: ')
        params['red_dir'] = raw_input('Please enter the path to the output directory: ')
        
    else:

        params['ip'] = sys.argv[1]
        params['rp'] = sys.argv[2]
        params['gp'] = sys.argv[3]
        params['red_dir'] = sys.argv[4]
    
    for key, value in params.items():
        
        if 'none' in str(value).lower():
            
            value = None
        
            params[key] = value
            
    return params
    
def extract_star_catalog(params,filter_id,log):
    """Function to read the photometric and star catalog data from a metadata file"""
    
    meta_file = params[filter_id]
    
    reduction_metadata = metadata.MetaData()
    reduction_metadata.load_a_layer_from_file( path.dirname(meta_file), path.basename(meta_file), 'star_catalog' )
    reduction_metadata.load_a_layer_from_file( path.dirname(meta_file), path.basename(meta_file), 'phot_calib' )
    reduction_metadata.load_a_layer_from_file( path.dirname(meta_file), path.basename(meta_file), 'images_stats' )
    
    star_catalog = Table()
    star_catalog['star_index'] = reduction_metadata.star_catalog[1]['star_index']
    star_catalog['RA'] = reduction_metadata.star_catalog[1]['RA_J2000']
    star_catalog['DEC'] = reduction_metadata.star_catalog[1]['DEC_J2000']
    star_catalog['mag'] = reduction_metadata.star_catalog[1]['ref_mag']
    star_catalog['mag_err'] = reduction_metadata.star_catalog[1]['ref_mag_err']
    star_catalog['cal_ref_mag'] = reduction_metadata.phot_calib[1]['cal_ref_mag']
    star_catalog['cal_ref_mag_err'] = reduction_metadata.phot_calib[1]['cal_ref_mag_err']
    star_catalog['_RAJ2000'] = reduction_metadata.phot_calib[1]['_RAJ2000']
    star_catalog['_DEJ2000'] = reduction_metadata.phot_calib[1]['_DEJ2000']
    star_catalog['imag'] = reduction_metadata.phot_calib[1]['imag']
    star_catalog['e_imag'] = reduction_metadata.phot_calib[1]['e_imag']
    star_catalog['rmag'] = reduction_metadata.phot_calib[1]['rmag']
    star_catalog['e_rmag'] = reduction_metadata.phot_calib[1]['e_rmag']
    star_catalog['gmag'] = reduction_metadata.phot_calib[1]['gmag']
    star_catalog['e_gmag'] = reduction_metadata.phot_calib[1]['e_gmag']


    log.info('Extracted data for '+str(len(star_catalog))+\
            ' stars for dataset '+filter_id)
            
    image_table = Table()
    image_table['im_name'] = reduction_metadata.images_stats[1]['IM_NAME']
    image_table['fwhm_x'] = reduction_metadata.images_stats[1]['FWHM_X']
    image_table['fwhm_y'] = reduction_metadata.images_stats[1]['FWHM_Y']
    image_table['sky'] = reduction_metadata.images_stats[1]['SKY']
    
    log.info('Extracted data for '+str(len(image_table))+\
            ' images in the dataset')
            
    return star_catalog, image_table

def combine_star_catalogs(datasets,log):
    """Function to cross-match stars between datasets"""
    
    (f1,dataset1,f2,dataset2) = select_first_catalog_match(datasets,log)
    
    (combined_catalog, col_names, formats, units) = init_combined_catalog(datasets,f1,log)
    
    match_index = cross_match_stars(f1,dataset1,f2,dataset2,log)

    combined_catalog = populate_combined_catalog(combined_catalog,f1,dataset1,
                                                 match_index,log,
                                                 f2=f2,dataset2=dataset2)
                                                 
    (f3,dataset3) = select_second_catalog_match(datasets,f1,log)
    
    if f3 != None:
        
        match_index = cross_match_stars(f1,dataset1,f3,dataset3,log)
        
        combined_catalog = populate_combined_catalog(combined_catalog,f1,dataset1,
                                                 match_index,log,
                                                 f3=f3,dataset3=dataset3)
                                                 
    return combined_catalog, col_names, formats, units,f1,f2,f3
    
def init_combined_catalog(datasets,f1,log):
    """Function to build the combined catalog data structure"""
    
    max_stars = len(datasets[f1])
    
    col_names = [ 'star_index', 'RA', 'DEC' ]
    formats = [ 'J', 'D', 'D' ]
    units = [ None, 'deg', 'deg' ]
    
    for f in ['ip', 'rp', 'gp']:
        if datasets[f] != None:
            col_names += [ 'ref_mag_'+f, 'ref_mag_err_'+f, 
                           'cal_ref_mag_'+f, 'cal_ref_mag_err_'+f ]
            formats += [ 'E', 'E', 'E', 'E' ]
            units += [ 'mag', 'mag', 'mag', 'mag' ]
    
    col_names += [ 'imag', 'e_imag', 'rmag', 'e_rmag', 'gmag', 'e_gmag' ]
    formats += [ 'E', 'E', 'E', 'E', 'E', 'E' ]
    units += [ 'mag', 'mag', 'mag', 'mag', 'mag', 'mag' ]
    
    combined_catalog = np.zeros([max_stars,len(col_names)])

    log.info('Initialized combined catalog for '+str(max_stars)+' stars')
                
    return combined_catalog, col_names, formats, units
    
def select_first_catalog_match(datasets,log):
    """Function to select the first set of catalogs to cross-match"""
    
    if datasets['ip'] != None:
        f1 = 'ip'
    elif datasets['rp'] != None:
        f1 = 'rp'
    else:
        log.info('ERROR: Dataset contains an unrecognized combination of filters:')
        log.info(repr(datasets))
        exit()

    dataset1 = datasets[f1]
    
    if 'rp' not in f1 and datasets['rp'] != None:
        f2 = 'rp'
        dataset2 = datasets[f2]
    elif 'rp' in f1 and datasets['gp'] != None:
        f2 = 'gp'
        dataset2 = datasets[f2]
    else:
        log.info('ERROR: Dataset contains an unrecognized combination of filters:')
        log.info(repr(datasets))
        exit()
    
    log.info('Primary dataset selected: '+f1+' with '+str(len(dataset1))+' stars')
    log.info('Second dataset selected: '+f2+' with '+str(len(dataset2))+' stars')
    
    return f1,dataset1,f2,dataset2
        
def select_second_catalog_match(datasets,f1,log):
    """Function to identify the second pair of datasets to cross-match"""
    
    if f1 == 'ip' and datasets['gp'] != None:
        f3 = 'gp'
        dataset3 = datasets['gp']
    else:
        f3 = None
        dataset3 = None
    
    if f3 != None:
        log.info('Second dataset selected: '+f3+' with '+str(len(dataset3))+' stars')
    else:
        log.info('No 3rd colour dataset available')
        
    return f3, dataset3
    
def cross_match_stars(f1,dataset1,f2,dataset2,log):
    """Function to cross-match stars by position.
    
    Returns:
        :param dict match_index: { Index in vphas_cat: index in star_cat }
    """
    
    stars1 = SkyCoord(dataset1['RA'], dataset1['DEC'], unit="deg")
    stars2 = SkyCoord(dataset2['RA'], dataset2['DEC'], unit="deg")
    
    tolerance = 1.0 * u.arcsec
    
    match_data = matching.search_around_sky(stars1, stars2, 
                                             seplimit=tolerance)   
                                             
    idx = np.argsort(match_data[2].value)
    
    match_index = np.array(zip(match_data[0][idx],match_data[1][idx]))
    
    log.info('Matched '+str(len(match_index))+' stars between '+f1+' and '+f2)
        
    return match_index

def populate_combined_catalog(combined_catalog,f1,dataset1,match_index,log,
                              f2=None,dataset2=None,f3=None,dataset3=None):
    """Function to populate the data table for the combined catalog with two
    cross-matched datasets.  If the catalog is currently empty, the function 
    uses the first dataset as the main index for the combined catalog."""
    
    if len(np.where(combined_catalog != 0.0)[0]) == 0:
        
        combined_catalog[:,0] = dataset1['star_index']
        combined_catalog[:,1] = dataset1['RA']
        combined_catalog[:,2] = dataset1['DEC']
        combined_catalog[:,3] = dataset1['mag']
        combined_catalog[:,4] = dataset1['mag_err']
        combined_catalog[:,5] = dataset1['cal_ref_mag']
        combined_catalog[:,6] = dataset1['cal_ref_mag_err']
        combined_catalog[:,10] = 0.0            # cal_ref_err2
        combined_catalog[:,14] = 0.0             # cal_ref_err3
        combined_catalog[:,15] = dataset1['imag']
        combined_catalog[:,16] = dataset1['e_imag']
        combined_catalog[:,17] = dataset1['rmag']
        combined_catalog[:,18] = dataset1['e_rmag']
        combined_catalog[:,19] = dataset1['gmag']
        combined_catalog[:,20] = dataset1['e_gmag']
        
        log.info('Populated the combined catalog with primary dataset, '+f1)
    
    if f2 != None:
        
        combined_catalog[match_index[:,0],7] = dataset2['mag'][match_index[:,1]]
        combined_catalog[match_index[:,0],8] = dataset2['mag_err'][match_index[:,1]]
        combined_catalog[match_index[:,0],9] = dataset2['cal_ref_mag'][match_index[:,1]]
        combined_catalog[match_index[:,0],10] = dataset2['cal_ref_mag_err'][match_index[:,1]]
        
        combined_catalog[match_index[:,0],15] = dataset2['imag'][match_index[:,1]]
        combined_catalog[match_index[:,0],16] = dataset2['e_imag'][match_index[:,1]]
        combined_catalog[match_index[:,0],17] = dataset2['rmag'][match_index[:,1]]
        combined_catalog[match_index[:,0],18] = dataset2['e_rmag'][match_index[:,1]]
        combined_catalog[match_index[:,0],19] = dataset2['gmag'][match_index[:,1]]
        combined_catalog[match_index[:,0],20] = dataset2['e_gmag'][match_index[:,1]]

        log.info('Populated the combined catalog with '+str(len(match_index))+\
            ' stars from dataset '+f2)
            
    elif f3 != None:

        combined_catalog[match_index[:,0],11] = dataset3['mag'][match_index[:,1]]
        combined_catalog[match_index[:,0],12] = dataset3['mag_err'][match_index[:,1]]
        combined_catalog[match_index[:,0],13] = dataset3['cal_ref_mag'][match_index[:,1]]
        combined_catalog[match_index[:,0],14] = dataset3['cal_ref_mag_err'][match_index[:,1]]
    
        combined_catalog[match_index[:,0],15] = dataset3['imag'][match_index[:,1]]
        combined_catalog[match_index[:,0],16] = dataset3['e_imag'][match_index[:,1]]
        combined_catalog[match_index[:,0],17] = dataset3['rmag'][match_index[:,1]]
        combined_catalog[match_index[:,0],18] = dataset3['e_rmag'][match_index[:,1]]
        combined_catalog[match_index[:,0],19] = dataset3['gmag'][match_index[:,1]]
        combined_catalog[match_index[:,0],20] = dataset3['e_gmag'][match_index[:,1]]
        
        log.info('Populated the combined catalog with '+str(len(match_index))+\
            ' stars from dataset '+f3)
            
    else:
        log.info('ERROR: To populate the combined catalog, one of dataset 2 or 3 must contain data')
        exit()
        
    return combined_catalog

def identify_image_trios(params,datasets,log):
    """Function to identify image trios between the datasets"""
    
    triodata = select_image_trios.TriColourDataset()
    
    triodata.gdir_stats = datasets['gp_images']
    triodata.rdir_stats = datasets['rp_images']
    triodata.idir_stats = datasets['ip_images']
    
    triodata.outdir = params['red_dir']
    
    triodata.make_image_table()
    
    triodata.quality_control()
    
    triodata.identify_image_trios()
    
    return triodata.image_trios_table
    
def output_combined_catalog(combined_catalog,col_names,formats,units,f1,f2,f3,
                            image_trios, params,log):
    """Function to write the combined catalog to a FITS table"""

    header = fits.Header()
    header['NSTARS'] = len(combined_catalog)
    header['FILTER1'] = f1
    header['FILTER2'] = f2
    if f3 != None:
        header['FILTER3'] = f3
    else:
        header['FILTER3'] = 'None'
        
    prihdu = fits.PrimaryHDU(header=header)
    
    col_list = []
    for i in range(0,len(col_names),1):
        
        if formats[i] == 'E' or formats[i] == 'D':
            
            col_list.append(fits.Column(name=col_names[i], 
                                    format=formats[i], 
                                    unit=units[i],
                                    array=combined_catalog[:,i]))
                                    
        elif formats[i] == 'J':
            
            col_list.append(fits.Column(name=col_names[i], 
                                    format=formats[i], 
                                    unit=units[i],
                                    array=combined_catalog[:,i].astype('int')))
                                    
    tbhdu = fits.BinTableHDU.from_columns(col_list)
    
    col_list2 = [ fits.Column(name='g_images',format='60A', 
                              unit=None, array=image_trios['g_images'].data),
                  fits.Column(name='r_images',format='60A', 
                              unit=None, array=image_trios['r_images'].data),
                  fits.Column(name='i_images',format='60A', 
                              unit=None, array=image_trios['i_images'].data) ]

    tbhdu2 = fits.BinTableHDU.from_columns(col_list2)
    
    thdulist = fits.HDUList([prihdu, tbhdu, tbhdu2])
    
    catalog_file = path.join(params['red_dir'],'combined_star_catalog.fits')
    
    thdulist.writeto(catalog_file, overwrite=True)

    log.info('Output combined catalog to '+catalog_file)
    
if __name__ == '__main__':
    
    combine_colour_datasets()
    