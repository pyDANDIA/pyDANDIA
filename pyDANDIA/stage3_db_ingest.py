# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 09:26:13 2019

@author: rstreet
"""

import os
import sys
from astropy.io import fits
from astropy import table
from astropy import time
import numpy as np
from pyDANDIA import  logs
from pyDANDIA import  metadata
from pyDANDIA import  phot_db
from pyDANDIA import  pipeline_setup
from pyDANDIA import  time_utils

VERSION = 'stage3_ingest_v1.0'

def run_stage3_db_ingest():
    """Function to commit the information on, and measurements from, the 
    reference image(s) used to reduce the data for a given field.
    """
    
    params = configure_setup()
    
    facility_keys = ['facility_code', 'site', 'enclosure', 
                     'telescope', 'instrument']
    software_keys = ['code_name', 'stage', 'version']
    image_keys =    ['filename', 'field_id',
                     'date_obs_utc','date_obs_jd','exposure_time',
                     'fwhm','fwhm_err',
                     'ellipticity','ellipticity_err',
                     'slope','slope_err','intercept','intercept_err',
                     'wcsfrcat','wcsimcat','wcsmatch','wcsnref','wcstol','wcsra',
                     'wcsdec','wequinox','wepoch','radecsys',
                     'ctype1','ctype2','cdelt1','cdelt2','crota1','crota2',
                     'secpix1','secpix2',
                     'wcssep','equinox',
                     'cd1_1','cd1_2','cd2_1','cd2_2','epoch',
                     'airmass','moon_phase','moon_separation',
                     'delta_x','delta_y']
    
    log = logs.start_stage_log( params['log_dir'], 'stage3_db_ingest', 
                               version=VERSION )
                               
    conn = phot_db.get_connection(dsn=params['database_file_path'])
    
    for f in ['gp', 'rp', 'ip']:
        
        log.info('\nIngesting data for '+f+' data')
        
        setup = params['setup_'+f]
        
        reduction_metadata = metadata.MetaData()
        reduction_metadata.load_a_layer_from_file( setup.red_dir, 
                                              'pyDANDIA_metadata.fits', 
                                              'data_architecture' )
        reduction_metadata.load_a_layer_from_file( setup.red_dir, 
                                              'pyDANDIA_metadata.fits', 
                                              'reduction_parameters' )
        reduction_metadata.load_a_layer_from_file( setup.red_dir, 
                                              'pyDANDIA_metadata.fits', 
                                              'headers_summary' )
        reduction_metadata.load_a_layer_from_file( setup.red_dir, 
                                              'pyDANDIA_metadata.fits', 
                                              'images_stats' )
        reduction_metadata.load_a_layer_from_file( setup.red_dir, 
                                              'pyDANDIA_metadata.fits', 
                                              'software' )
        
        dataset_params = harvest_dataset_parameters(setup,reduction_metadata)
        
        phot_db.check_before_commit(conn, dataset_params, 'facilities', facility_keys, 'facility_code')
        phot_db.check_before_commit(conn, dataset_params, 'software', software_keys, 'version')
        phot_db.check_before_commit(conn, dataset_params, 'images', image_keys, 'filename')
        
        commit_reference_image(conn, dataset_params, log)
        commit_reference_component(conn, dataset_params, log)
        
        star_catalog = read_combined_star_catalog(params)
        
        star_ids = commit_stars(conn, dataset_params, star_catalog, log)
        
        commit_photometry(conn, dataset_params, star_catalog, star_ids, log)
        
    conn.close()
    
    status = 'OK'
    report = 'Completed successfully'
    
    log.info('Stage 3 DB ingest: '+report)
    logs.close_log(log)
        
    return status, report
    
def configure_setup(log=None):

    params = {'datasets': [('red_dir_gp', 1), ('red_dir_rp', 2), ('red_dir_ip',3)],
              'stage': 'stage3_db_ingest',
              'red_dir_gp': None,
              'red_dir_rp': None,
              'red_dir_ip': None}
    
    if len(sys.argv) == 1:
        
        params['red_dir_gp'] = input('Please enter the path to the SDSS-g reduction directory (or None): ')
        params['red_dir_rp'] = input('Please enter the path to the SDSS-r reduction directory (or None): ')
        params['red_dir_ip'] = input('Please enter the path to the SDSS-i reduction directory (or None): ')
        params['combined_starcat'] = input('Please enter the path to the combined star catalogue file: ')
        params['database_file_path'] = input('Please enter the path to the database file: ')
        params['log_dir'] = input('Please enter the path to the logging directory: ')
        
        for f, i in params['datasets']:
            
            if 'none' in params[f].lower():

                params[f] = None
                
    else:
        
        params['red_dir_gp'] = sys.argv[1]
        params['red_dir_rp'] = sys.argv[2]
        params['red_dir_ip'] = sys.argv[3]
        params['combined_starcat'] = sys.argv[4]
        params['database_file_path'] = sys.argv[5]
        params['log_dir'] = sys.argv[6]

    if log!=None:
        
        log.info('Input dataset configurations:')
        
    for f, i in params['datasets']:
        
        if f != None:
            
            pars = {'stage': params['stage'], 'red_dir': params[f]}
            
            setup = pipeline_setup.pipeline_setup(pars)
    
            params[f.replace('red_dir','setup')] = setup
            
            if log!=None:
                log.info(setup.summary())
                
    return params

def harvest_dataset_parameters(setup,reduction_metadata):
    """Function to harvest the parameters required for ingest of a single 
    dataset into the photometric database."""
    
    dataset_params = {}
    
    ref_path = reduction_metadata.data_architecture[1]['REF_PATH'][0]
    ref_filename = reduction_metadata.data_architecture[1]['REF_IMAGE'][0]
    
    ref_hdr_image = fits.getheader(os.path.join(ref_path, ref_filename))
    
    # Facility
    dataset_params['site'] = ref_hdr_image['SITEID']
    dataset_params['enclosure'] = ref_hdr_image['ENCID']
    dataset_params['telescope'] = ref_hdr_image['TELID']
    dataset_params['instrument'] = ref_hdr_image['INSTRUME']
    dataset_params['facility_code'] = dataset_params['site']+'-'+\
                                      dataset_params['enclosure']+'-'+\
                                      dataset_params['telescope']+'-'+\
                                      dataset_params['instrument']
    
    # Software
    dataset_params['version'] = reduction_metadata.software[1]['stage3_version'][0]
    dataset_params['stage'] = 'stage3'
    dataset_params['code_name'] = 'stage3.py'
    
    # Image parameters for single-frame reference image
    # NOTE: Stacked reference images not yet supported
    dataset_params['filename'] = ref_filename

    idx = np.where(reduction_metadata.headers_summary[1]['IMAGES'] == dataset_params['filename'])
    ref_hdr_meta = reduction_metadata.headers_summary[1][idx]
    idx = np.where(reduction_metadata.images_stats[1]['IM_NAME'] == dataset_params['filename'])
    ref_stats = reduction_metadata.images_stats[1][idx]
    
    dataset_params['field_id'] = ref_hdr_meta['OBJKEY'][0]
    dataset_params['date_obs_utc'] = ref_hdr_meta['DATEKEY'][0]
    t = time.Time(dataset_params['date_obs_utc'],format='isot', scale='utc')
    dataset_params['date_obs_jd'] = t.jd
    dataset_params['exposure_time'] = float(ref_hdr_meta['EXPKEY'][0])
    dataset_params['RA'] = ref_hdr_image['RA']
    dataset_params['Dec'] = ref_hdr_image['DEC']
    dataset_params['filter_name'] = ref_hdr_image['FILTER']
    dataset_params['fwhm'] = np.sqrt(ref_stats['FWHM_X'][0]*ref_stats['FWHM_X'][0] + 
                                        ref_stats['FWHM_Y'][0]*ref_stats['FWHM_Y'][0])
    dataset_params['fwhm_err'] = None
    dataset_params['ellipticity'] = None
    dataset_params['ellipticity_err'] = None
    dataset_params['slope'] = None
    dataset_params['slope_err'] = None
    dataset_params['intercept'] = None
    dataset_params['intercept_err'] = None
    dataset_params['wcsfrcat'] = None
    dataset_params['wcsimcat'] = None
    dataset_params['wcsmatch'] = None
    dataset_params['wcsnref'] = None
    dataset_params['wcstol'] = None
    dataset_params['wcsra'] = None
    dataset_params['wcsdec'] = None
    dataset_params['wequinox'] = None
    dataset_params['wepoch'] = None
    dataset_params['radecsys'] = None
    dataset_params['ctype1'] = set_if_present(ref_hdr_image,'CTYPE1')
    dataset_params['ctype2'] = set_if_present(ref_hdr_image,'CTYPE2')
    dataset_params['cdelt1'] = set_if_present(ref_hdr_image,'CDELT1')
    dataset_params['cdelt2'] = set_if_present(ref_hdr_image,'CDELT2')
    dataset_params['crota1'] = set_if_present(ref_hdr_image,'CROTA1')
    dataset_params['crota2'] = set_if_present(ref_hdr_image,'CROTA2')
    dataset_params['secpix1'] = set_if_present(ref_hdr_image,'PIXSCALE')
    dataset_params['secpix2'] = set_if_present(ref_hdr_image,'PIXSCALE')
    dataset_params['wcssep'] = None
    dataset_params['equinox'] = None
    dataset_params['cd1_1'] = set_if_present(ref_hdr_image,'CD1_1')
    dataset_params['cd1_2'] = set_if_present(ref_hdr_image,'CD1_2')
    dataset_params['cd2_1'] = set_if_present(ref_hdr_image,'CD2_1')
    dataset_params['cd2_2'] = set_if_present(ref_hdr_image,'CD2_2')
    dataset_params['epoch'] = None
    dataset_params['airmass'] = set_if_present(ref_hdr_image,'AIRMASS')
    dataset_params['moon_phase'] = set_if_present(ref_hdr_image,'MOONFRAC')
    dataset_params['moon_separation'] = set_if_present(ref_hdr_image,'MOONDIST')
    dataset_params['delta_x'] = None
    dataset_params['delta_y'] = None
    
    dataset_params['hjd_ref'] = time_utils.calc_hjd(dataset_params['date_obs_utc'],
                                  dataset_params['RA'],dataset_params['Dec'],
                                  dataset_params['exposure_time'])
    
    print(dataset_params)
    return dataset_params

def set_if_present(header, key):
    
    if key in header.keys():
        return header[key]
    else:
        return None

def commit_reference_image(conn, params, log):

    query = 'SELECT refimg_id,filename FROM reference_images WHERE filename ="'+params['filename']+'"'
    ref_image = phot_db.query_to_astropy_table(conn, query, args=())    

    if len(ref_image) != 0:
        
        log.info('Reference image '+params['filename']+\
                    ' is already in the phot_db as entry '+\
                        str(ref_image['refimg_id']))
                        
    else:
        
        query = 'SELECT facility_id, facility_code FROM facilities WHERE facility_code="'+params['facility_code']+'"'
        facility = phot_db.query_to_astropy_table(conn, query, args=())
        error_wrong_number_entries(facility,params['facility_code'])
        
        query = 'SELECT filter_id, filter_name FROM filters WHERE filter_name="'+params['filter_name']+'"'
        f = phot_db.query_to_astropy_table(conn, query, args=())
        error_wrong_number_entries(f,params['filter_name'])
        
        query = 'SELECT code_id, version FROM software WHERE version="'+params['version']+'"'
        code = phot_db.query_to_astropy_table(conn, query, args=())
        error_wrong_number_entries(code,params['version'])
        
        command = 'INSERT OR REPLACE INTO reference_images (facility,filter,software,filename) VALUES ('+\
            str(facility['facility_id'][0])+','+str(f['filter_id'][0])+','+str(code['code_id'][0])+',"'+str(params['filename'])+'")'
        
        cursor = conn.cursor()
        
        cursor.execute(command)
        
        conn.commit()
        
        log.info('Submitted reference_image '+params['filename']+' to phot_db')

def error_wrong_number_entries(results_table,param_value):
    
    if len(results_table) == 0:
        raise IOError(param_value+' is unknown to the database')
    elif len(results_table) > 1:
        raise IOError('Mulitple entries named '+param_value+' are known to the database')
    
def commit_reference_component(conn, params, log):
    """Function to create the reference component entry for a SINGLE-IMAGE 
    reference"""
    
    table_keys = [ 'image', 'reference_image' ]
    
    query = 'SELECT img_id, filename FROM images WHERE filename ="'+params['filename']+'"'
    image = phot_db.query_to_astropy_table(conn, query, args=())    
    error_wrong_number_entries(image,params['filename'])
    
    query = 'SELECT refimg_id, filename FROM reference_images WHERE filename ="'+params['filename']+'"'
    refimage = phot_db.query_to_astropy_table(conn, query, args=())    
    error_wrong_number_entries(refimage,params['filename'])
    
    table_keys = [ 'image', 'reference_image' ]
    
    command = 'INSERT OR REPLACE INTO reference_components (image,reference_image) VALUES ('+\
                str(image['img_id'][0])+','+str(refimage['refimg_id'][0])+')'
        
    cursor = conn.cursor()
        
    cursor.execute(command)
        
    conn.commit()
        
    log.info('Submitted reference_component '+params['filename']+' to phot_db')

def read_combined_star_catalog(params):
    
    if os.path.isfile(params['combined_starcat']) == False:
        
        raise IOError('Cannot find combined star catalogue file '+\
                        params['combined_starcat'])
    
    star_catalog = fits.getdata(params['combined_starcat'])
    
    return star_catalog

def commit_stars(conn, params, star_catalog, log):
    
    query = 'SELECT refimg_id, filename FROM reference_images WHERE filename ="'+params['filename']+'"'
    refimage = phot_db.query_to_astropy_table(conn, query, args=())    
    error_wrong_number_entries(refimage,params['filename'])
    
    tol = 1.0/3600.0
    
    star_ids = np.zeros(len(star_catalog))
    
    for j in range(0,len(star_catalog),1):
        
        ra = star_catalog[j][1]
        dec = star_catalog[j][2]

        results = phot_db.box_search_on_position(conn, ra, dec, tol, tol)
        
        if len(results) > 0:
            
            log.info('Catalog star at RA, Dec '+str(ra)+','+str(dec)+\
                     ' matches a star already in the phot_db with RA, Dec '+\
                     str(results['ra'][0])+','+str(results['dec'][0]))
            
            star_ids[j] = int(results['star_id'][0])
            
        else:
            
            command = 'INSERT OR REPLACE INTO stars (ra,dec,reference_image) VALUES ('+\
                str(ra)+','+str(dec)+','+str(refimage['refimg_id'][0])+')'

            cursor = conn.cursor()
                
            cursor.execute(command)
                
            conn.commit()
            
            results = phot_db.box_search_on_position(conn, ra, dec, tol, tol)
            
            log.info('Commited catalog star at RA, Dec '+str(ra)+','+str(dec)+\
                     ' to the phot_db as star_id='+str(results['star_id'][0]))
            
            star_ids[j] = int(results['star_id'][0])
    
    return star_ids

def commit_photometry(conn, params, star_catalog, star_ids, log):
    
    query = 'SELECT facility_id, facility_code FROM facilities WHERE facility_code="'+params['facility_code']+'"'
    facility = phot_db.query_to_astropy_table(conn, query, args=())
    error_wrong_number_entries(facility,params['facility_code'])
    
    query = 'SELECT filter_id, filter_name FROM filters WHERE filter_name="'+params['filter_name']+'"'
    f = phot_db.query_to_astropy_table(conn, query, args=())
    error_wrong_number_entries(f,params['filter_name'])
        
    query = 'SELECT code_id, version FROM software WHERE version="'+params['version']+'"'
    code = phot_db.query_to_astropy_table(conn, query, args=())
    error_wrong_number_entries(code,params['version'])
    
    query = 'SELECT refimg_id, filename FROM reference_images WHERE filename ="'+params['filename']+'"'
    refimage = phot_db.query_to_astropy_table(conn, query, args=())    
    error_wrong_number_entries(refimage,params['filename'])
    
    query = 'SELECT img_id, filename FROM images WHERE filename ="'+params['filename']+'"'
    image = phot_db.query_to_astropy_table(conn, query, args=())    
    error_wrong_number_entries(image,params['filename'])
    
    key_list = ['star_id', 'reference_image', 'image', 
                'facility', 'filter', 'software', 
                'x', 'y', 'hjd', 'magnitude', 'magnitude_err', 
                'calibrated_mag', 'calibrated_mag_err',
                'flux', 'flux_err', 
                'phot_scale_factor', 'phot_scale_factor_err',
                'local_background', 'local_background_err',
                'phot_type']
    
    wildcards = ','.join(['?']*len(key_list))
    
    # Data column indices in the combined star catalog, must match
    # combine_colour_data.py.  Ideally this should be in a shared 
    # config file when I get around to it.
    if params['filter_name'] == 'gp':
        xcol = 25
        ycol = 26
        mag_col = 11
        mag_err_col = 12
        cal_mag_col = 13
        cal_mag_err_col = 14
        flux_col = 31
        flux_err_col = 32
    elif params['filter_name'] == 'rp':
        xcol = 23
        ycol = 24
        mag_col = 7
        mag_err_col = 8
        cal_mag_col = 9
        cal_mag_err_col = 10
        flux_col = 29
        flux_err_col = 30
    elif params['filter_name'] == 'ip':
        xcol = 21
        ycol = 22
        mag_col = 3
        mag_err_col = 4
        cal_mag_col = 5
        cal_mag_err_col = 6
        flux_col = 27
        flux_err_col = 28
    
    values = []
    for j in range(0,len(star_catalog),1):
        
        entry = (str(int(star_ids[j])), str(refimage['refimg_id'][0]), str(image['img_id'][0]),
                   str(facility['facility_id'][0]), str(f['filter_id'][0]), str(code['code_id'][0]),
                    str(star_catalog[j][xcol]), str(star_catalog[j][ycol]),
                    str(params['hjd_ref']),
                    str(star_catalog[j][mag_col]), str(star_catalog[j][mag_err_col]),
                    str(star_catalog[j][cal_mag_col]), str(star_catalog[j][cal_mag_err_col]),
                    str(star_catalog[j][flux_col]), str(star_catalog[j][flux_err_col]),
                    '0.0', '0.0',   # No phot scale factor for PSF fitting photometry
                    '0.0', '0.0',   # No background measurements propageted
                    'PSF_FITTING' )
                    
        values.append(entry)
    
    command = 'INSERT OR REPLACE INTO phot('+','.join(key_list)+\
                ') VALUES ('+wildcards+')'
    
    cursor = conn.cursor()
        
    cursor.executemany(command,values)
    
    conn.commit()
        
    log.info('Completed ingest of photometry for '+str(len(star_ids))+' stars')
 