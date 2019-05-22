# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 09:26:13 2019

@author: rstreet
"""

from os import path
from sys import argv
from astropy.io import fits
from astropy import table
from astropy import time
from astropy import units
from astropy.coordinates import SkyCoord
import numpy as np
import glob
from shutil import move
from pyDANDIA import  logs
from pyDANDIA import  metadata
from pyDANDIA import  phot_db
from pyDANDIA import  pipeline_setup
from pyDANDIA import  time_utils
from pyDANDIA import  match_utils
from pyDANDIA import  calc_coord_offsets

VERSION = 'stage3_ingest_v1.0'

def run_stage3_db_ingest(setup, primary_ref=False):
    """Function to commit the information on, and measurements from, the 
    reference image(s) used to reduce the data for a given field.
    """
    
    (facility_keys, software_keys, image_keys) = define_table_keys()
    
    log = logs.start_stage_log( setup.red_dir, 'stage3_db_ingest', 
                               version=VERSION )
    if primary_ref:
        log.info('Running in PRIMARY-REF mode.')
        
    archive_existing_db(setup,primary_ref,log)
    
    conn = phot_db.get_connection(dsn=setup.phot_db_path)
    
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
    reduction_metadata.load_a_layer_from_file( setup.red_dir, 
                                          'pyDANDIA_metadata.fits', 
                                          'star_catalog' )
    
    dataset_params = harvest_dataset_parameters(setup,reduction_metadata)
    
    phot_db.check_before_commit(conn, dataset_params, 'facilities', facility_keys, 'facility_code')
    phot_db.check_before_commit(conn, dataset_params, 'software', software_keys, 'version')
    phot_db.check_before_commit(conn, dataset_params, 'images', image_keys, 'filename')
    
    ref_id_list = phot_db.find_reference_image_for_dataset(conn,dataset_params)
    
    if ref_id_list != None and len(ref_id_list) > 0:
        phot_db.cascade_delete_reference_images(conn, ref_id_list, log)
        
    commit_reference_image(conn, dataset_params, log)
    commit_reference_component(conn, dataset_params, log)
    
    if primary_ref:
        
        star_ids = commit_stars(conn, dataset_params, reduction_metadata, log)
        
        commit_photometry(conn, dataset_params, reduction_metadata, star_ids, log)
        
    else:
        
        starlist = fetch_field_starlist(conn,dataset_params,log)
        
        primary_refimg_id = phot_db.find_primary_reference_image_for_field(conn)
        
        matched_stars = match_catalog_entries_with_starlist(conn,dataset_params,
                                                            starlist,
                                                            reduction_metadata,
                                                            primary_refimg_id,log)
        
        transform = calc_transform_to_primary_ref(setup,matched_stars,log)
        
        matched_stars = match_all_entries_with_starlist(setup,conn,dataset_params,
                                                        starlist,reduction_metadata,
                                                        primary_refimg_id,transform,log,
                                                        verbose=True)
                                        
        commit_photometry_matching(conn, dataset_params, reduction_metadata, matched_stars, log)

    conn.close()
    
    status = 'OK'
    report = 'Completed successfully'
    
    log.info('Stage 3 DB ingest: '+report)
    logs.close_log(log)
    
    return status, report

def define_table_keys():
    
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

    return facility_keys, software_keys, image_keys
    
def configure_setup(log=None):

    params = {'datasets': [('red_dir_gp', 1), ('red_dir_rp', 2), ('red_dir_ip',3)],
              'stage': 'stage3_db_ingest',
              'red_dir_gp': None,
              'red_dir_rp': None,
              'red_dir_ip': None}
    
    if len(argv) == 1:
        
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
        
        params['red_dir_gp'] = argv[1]
        params['red_dir_rp'] = argv[2]
        params['red_dir_ip'] = argv[3]
        params['combined_starcat'] = argv[4]
        params['database_file_path'] = argv[5]
        params['log_dir'] = argv[6]

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

def archive_existing_db(setup,primary_ref,log):
    """Function to archive an existing photometric DB, if one exists, in the
    event that a new primary reference is adopted"""
    
    def identify_archive_index(db_list):
        
        if len(db_list) > 0:
            
            idx = 0
            for db_file in db_list:
                
                suffix = db_file.split('.')[-1]
                
                if 'db' not in suffix:
                    try:
                        i = int(suffix)
                    except TypeError:
                        pass
                    if i > idx:
                        idx = i
            
        else:
            
            idx = 0
        
        idx += 1
        
        return idx
        
    if primary_ref:
        
        db_list = glob.glob(setup.phot_db_path+'*')
        
        idx = identify_archive_index(db_list)
        
        if path.isfile(setup.phot_db_path):
            
            dest = setup.phot_db_path + '.' + str(idx)
            
            move(setup.phot_db_path, dest)
            
            log.info('-> Archived old PHOT_DB to '+dest)
            
def harvest_dataset_parameters(setup,reduction_metadata):
    """Function to harvest the parameters required for ingest of a single 
    dataset into the photometric database."""
    
    dataset_params = {}
    
    ref_path = reduction_metadata.data_architecture[1]['REF_PATH'][0]
    ref_filename = reduction_metadata.data_architecture[1]['REF_IMAGE'][0]
    
    ref_hdr_image = fits.getheader(path.join(ref_path, ref_filename))
    
    # Facility
    dataset_params['site'] = ref_hdr_image['SITEID']
    dataset_params['enclosure'] = ref_hdr_image['ENCID']
    dataset_params['telescope'] = ref_hdr_image['TELID']
    dataset_params['instrument'] = ref_hdr_image['INSTRUME']
    dataset_params['facility_code'] = phot_db.get_facility_code(dataset_params)
    
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
    
    if path.isfile(params['combined_starcat']) == False:
        
        raise IOError('Cannot find combined star catalogue file '+\
                        params['combined_starcat'])
    
    star_catalog = fits.getdata(params['combined_starcat'])
    
    return star_catalog

def commit_stars(conn, params, reduction_metadata, log, 
                 search_for_match=False):
    
    query = 'SELECT refimg_id, filename FROM reference_images WHERE filename ="'+params['filename']+'"'
    refimage = phot_db.query_to_astropy_table(conn, query, args=())    
    error_wrong_number_entries(refimage,params['filename'])
    
    tol = 1.0/3600.0
    
    n_stars = len(reduction_metadata.star_catalog[1])
    
    star_keys = ['star_index', 'ra', 'dec', 'reference_image', 
                 'gaia_source_id', 'gaia_ra', 'gaia_ra_error', 'gaia_dec', 'gaia_dec_error', 
                 'gaia_phot_g_mean_flux', 'gaia_phot_g_mean_flux_error', 
                 'gaia_phot_bp_mean_flux', 'gaia_phot_bp_mean_flux_error', 
                 'gaia_phot_rp_mean_flux', 'gaia_phot_rp_mean_flux_error',
                 'vphas_source_id', 'vphas_ra', 'vphas_dec', 
                 'vphas_gmag', 'vphas_gmag_error', 
                 'vphas_rmag', 'vphas_rmag_error', 
                 'vphas_imag', 'vphas_imag_error', 'vphas_clean']
                 
    star_ids = np.zeros(n_stars)
    
    for j in range(0,n_stars,1):
        
        idx = reduction_metadata.star_catalog[1]['index'][j]
        ra = reduction_metadata.star_catalog[1]['ra'][j]
        dec = reduction_metadata.star_catalog[1]['dec'][j]
        
        star_pars = [ str(idx), str(ra), str(dec), str(refimage['refimg_id'][0]),
                    str(reduction_metadata.star_catalog[1]['gaia_source_id'][j]),
                    str(reduction_metadata.star_catalog[1]['gaia_ra'][j]),
                    str(reduction_metadata.star_catalog[1]['gaia_ra_error'][j]),
                    str(reduction_metadata.star_catalog[1]['gaia_dec'][j]),
                    str(reduction_metadata.star_catalog[1]['gaia_dec_error'][j]),
                    str(reduction_metadata.star_catalog[1]['phot_g_mean_flux'][j]),
                    str(reduction_metadata.star_catalog[1]['phot_g_mean_flux_error'][j]),
                    str(reduction_metadata.star_catalog[1]['phot_bp_mean_flux'][j]),
                    str(reduction_metadata.star_catalog[1]['phot_bp_mean_flux_error'][j]),
                    str(reduction_metadata.star_catalog[1]['phot_rp_mean_flux'][j]),
                    str(reduction_metadata.star_catalog[1]['phot_rp_mean_flux_error'][j]),
                    str(reduction_metadata.star_catalog[1]['vphas_source_id'][j]),
                    str(reduction_metadata.star_catalog[1]['vphas_ra'][j]),
                    str(reduction_metadata.star_catalog[1]['vphas_dec'][j]),
                    str(reduction_metadata.star_catalog[1]['gmag'][j]),
                    str(reduction_metadata.star_catalog[1]['gmag_error'][j]),
                    str(reduction_metadata.star_catalog[1]['rmag'][j]),
                    str(reduction_metadata.star_catalog[1]['rmag_error'][j]),
                    str(reduction_metadata.star_catalog[1]['imag'][j]),
                    str(reduction_metadata.star_catalog[1]['imag_error'][j]),
                    str((int(reduction_metadata.star_catalog[1]['clean'][j]))) ]

        if search_for_match:
            
            results = phot_db.box_search_on_position(conn, ra, dec, tol, tol)
        
            if len(results) > 0:
            
                log.info('Catalog star at RA, Dec '+str(ra)+','+str(dec)+\
                     ' matches a star already in the phot_db with RA, Dec '+\
                     str(results['ra'][0])+','+str(results['dec'][0]))
            
                star_ids[j] = int(results['star_id'][0])
                
                submit = False
                
            else:
                
                submit = True
            
        else:
            
            submit = True
            
        if submit:
            
            command = 'INSERT OR REPLACE INTO stars ('+','.join(star_keys)+') VALUES ("'
            command += '","'.join(star_pars) + '")'
            
            cursor = conn.cursor()
                
            cursor.execute(command)
                
            conn.commit()
            
            results = phot_db.box_search_on_position(conn, ra, dec, tol, tol)
            
            if len(results) > 1:
                idx = np.where(results['separation'] == results['separation'].min())[0][0]
            else:
                idx = 0
                
            log.info('Commited catalog star at RA, Dec '+str(ra)+','+str(dec)+\
                     ' to the phot_db as star_id='+str(results['star_id'][idx]))
            
            star_ids[j] = int(results['star_id'][idx])
    
    return star_ids

def commit_photometry(conn, params, reduction_metadata, star_ids, log):
    
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
    
    n_stars = len(reduction_metadata.star_catalog[1])
    
    values = []
    for j in range(0,n_stars,1):
        
        k = reduction_metadata.star_catalog[1]['index'][j]
        x = str(reduction_metadata.star_catalog[1]['x'][j])
        y = str(reduction_metadata.star_catalog[1]['y'][j])
        mag = str(reduction_metadata.star_catalog[1]['ref_mag'][j])
        mag_err = str(reduction_metadata.star_catalog[1]['ref_mag_error'][j])
        cal_mag = str(reduction_metadata.star_catalog[1]['cal_ref_mag'][j])
        cal_mag_err = str(reduction_metadata.star_catalog[1]['cal_ref_mag_error'][j])
        flux = str(reduction_metadata.star_catalog[1]['ref_flux'][j])
        flux_err = str(reduction_metadata.star_catalog[1]['ref_flux_error'][j])
        
        entry = (str(int(star_ids[j])), str(refimage['refimg_id'][0]), str(image['img_id'][0]),
                   str(facility['facility_id'][0]), str(f['filter_id'][0]), str(code['code_id'][0]),
                    x, y, str(params['hjd_ref']), 
                    mag, mag_err, cal_mag, cal_mag_err, flux, flux_err,
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

def commit_photometry_matching(conn, params, reduction_metadata, matched_stars, log):
    
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
    
    n_stars = len(reduction_metadata.star_catalog[1])
    
    values = []
    for i in range(0,matched_stars.n_match,1):
        
        j_cat = matched_stars.cat1_index[i]     # Starlist index in DB
        j_new = matched_stars.cat2_index[i]     # Star detected in image
        
        x = str(reduction_metadata.star_catalog[1]['x'][j_new])
        y = str(reduction_metadata.star_catalog[1]['y'][j_new])
        mag = str(reduction_metadata.star_catalog[1]['ref_mag'][j_new])
        mag_err = str(reduction_metadata.star_catalog[1]['ref_mag_error'][j_new])
        cal_mag = str(reduction_metadata.star_catalog[1]['cal_ref_mag'][j_new])
        cal_mag_err = str(reduction_metadata.star_catalog[1]['cal_ref_mag_error'][j_new])
        flux = str(reduction_metadata.star_catalog[1]['ref_flux'][j_new])
        flux_err = str(reduction_metadata.star_catalog[1]['ref_flux_error'][j_new])
        
        entry = (str(int(j_cat)), str(refimage['refimg_id'][0]), str(image['img_id'][0]),
                   str(facility['facility_id'][0]), str(f['filter_id'][0]), str(code['code_id'][0]),
                    x, y, str(params['hjd_ref']), 
                    mag, mag_err, cal_mag, cal_mag_err, flux, flux_err,
                    '0.0', '0.0',   # No phot scale factor for PSF fitting photometry
                    '0.0', '0.0',   # No background measurements propageted
                    'PSF_FITTING' )
                
        values.append(entry)
        
    command = 'INSERT OR REPLACE INTO phot('+','.join(key_list)+\
                ') VALUES ('+wildcards+')'
    
    cursor = conn.cursor()
        
    cursor.executemany(command,values)
    
    conn.commit()
    
    log.info('Completed ingest of photometry for '+str(len(matched_stars.cat1_index))+' stars')

def fetch_field_starlist(conn,params,log):
    
    query = 'SELECT * FROM stars'
    starlist = phot_db.query_to_astropy_table(conn, query, args=())
    
    log.info('Selected '+str(len(starlist))+' stars known in this field')
    
    return starlist

def match_catalog_entries_with_starlist(conn,params,starlist,reduction_metadata,
                                        refimg_id,log,verbose=False):
    
    log.info('Matching all stars from the reference frame with catalog identifications with the DB')
    
    matched_stars = match_utils.StarMatchIndex()
    
    idx = np.where(starlist['gaia_source_id'] != 'None')[0]
    jdx = np.where(reduction_metadata.star_catalog[1]['gaia_source_id'] != 'None')[0]
    
    log.info(str(len(idx))+' stars with Gaia identifications selected from the field starlist')
    log.info(str(len(jdx))+' stars with Gaia identifications selected from new reference image catalog')
    
    for star in starlist[idx]:
                
        kdx = np.where(reduction_metadata.star_catalog[1]['gaia_source_id'][jdx] == star['gaia_source_id'])

        if len(kdx[0]) == 1:
            
            query = 'SELECT star_id,x,y FROM phot WHERE reference_image="'+str(refimg_id)+'" AND star_id="'+str(star['star_id'])+'"'
            phot_data = phot_db.query_to_astropy_table(conn, query, args=())
            
            dx = phot_data['x'] - reduction_metadata.star_catalog[1]['x'][jdx[kdx[0]]]
            dy = phot_data['y'] - reduction_metadata.star_catalog[1]['y'][jdx[kdx[0]]]
            separation = np.sqrt( dx*dx + dy*dy )
            
            p = {'cat1_index': star['star_id'],
                 'cat1_ra': star['ra'],
                 'cat1_dec': star['dec'],
                 'cat1_x': phot_data['x'][0],
                 'cat1_y': phot_data['y'][0],
                 'cat2_index': jdx[kdx[0]], 
                 'cat2_ra': reduction_metadata.star_catalog[1]['ra'][jdx[kdx[0]]][0],
                 'cat2_dec': reduction_metadata.star_catalog[1]['dec'][jdx[kdx[0]]][0],
                 'cat2_x': reduction_metadata.star_catalog[1]['x'][jdx[kdx[0]]][0],
                 'cat2_y': reduction_metadata.star_catalog[1]['y'][jdx[kdx[0]]][0],
                 'separation': separation[0]}
            
            matched_stars.add_match(p)
            
            if verbose:
                log.info(matched_stars.summarize_last())

    return matched_stars

def calc_transform_to_primary_ref(setup,matched_stars,log):
    
    primary_cat = table.Table( [ table.Column(name='x', data=matched_stars.cat1_x),
                                 table.Column(name='y', data=matched_stars.cat1_y) ] )
    
    refframe_cat = table.Table( [ table.Column(name='x', data=matched_stars.cat2_x),
                                  table.Column(name='y', data=matched_stars.cat2_y) ] )
                                 
    transform = calc_coord_offsets.calc_pixel_transform(setup, 
                                        refframe_cat, primary_cat,
                                        log)
                                        
    return transform

def match_all_entries_with_starlist(setup,conn,params,starlist,reduction_metadata,
                                    refimg_id,transform,log, verbose=False):
    
    tol = 2.0  # pixels
    
    log.info('Matching all stars from starlist with the transformed coordinates of stars detected in the new reference image')
    log.info('Match tolerance: '+str(tol)+' pixels')
    
    matched_stars = match_utils.StarMatchIndex()
    
    query = 'SELECT star_id,x,y FROM phot WHERE reference_image="'+str(refimg_id)+'" AND star_id IN '+str(tuple(starlist['star_id'].data))
    phot_data = phot_db.query_to_astropy_table(conn, query, args=())
    
    refframe_coords = table.Table( [ table.Column(name='x', data=reduction_metadata.star_catalog[1]['x']),
                                     table.Column(name='y', data=reduction_metadata.star_catalog[1]['y']) ] )
                                     
    refframe_coords = calc_coord_offsets.transform_coordinates(setup, refframe_coords, transform, coords='pixel')
    
    log.info('Transformed star coordinates from the reference image')
    log.info('Matching all stars against field starlist:')
    
    for j in range(0,len(phot_data),1):

        dx = phot_data['x'][j] - refframe_coords['x1']
        dy = phot_data['y'][j] - refframe_coords['y1']
        separation = np.sqrt( dx*dx + dy*dy )
        
        jdx = np.where(separation == separation.min())[0]
        
        p = {'cat1_index': phot_data['star_id'][j],
             'cat1_ra': starlist['ra'][j],
             'cat1_dec': starlist['dec'][j],
             'cat1_x': phot_data['x'][j],
             'cat1_y': phot_data['y'][j],
             'cat2_index': jdx[0], 
             'cat2_ra': reduction_metadata.star_catalog[1]['ra'][jdx[0]],
             'cat2_dec': reduction_metadata.star_catalog[1]['dec'][jdx[0]],
             'cat2_x': reduction_metadata.star_catalog[1]['x'][jdx[0]],
             'cat2_y': reduction_metadata.star_catalog[1]['y'][jdx[0]],
             'separation': separation[jdx[0]]}
        
        if separation[jdx[0]] <= tol:
            matched_stars.add_match(p)
        
            if verbose:
                log.info(matched_stars.summarize_last())

    return matched_stars
    