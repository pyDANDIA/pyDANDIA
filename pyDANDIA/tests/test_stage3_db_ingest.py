# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 09:42:07 2019

@author: rstreet
"""
import os
import sys
cwd = os.getcwd()
sys.path.append(os.path.join(cwd,'../'))
from astropy.io import fits
from astropy import table
import numpy as np
import sqlite3
from pyDANDIA import  logs
from pyDANDIA import  metadata
from pyDANDIA import  phot_db
from pyDANDIA import  stage3_db_ingest
from pyDANDIA import  pipeline_setup


TEST_DIR = os.path.join(cwd,'data','proc',
                        'ROME-FIELD-0002_lsc-doma-1m0-05-fl15_ip')
DB_FILE = 'test.db'
db_file_path = os.path.join(TEST_DIR, '..', DB_FILE)

def test_configure_setup():
    
    params = stage3_db_ingest.configure_setup()
    
    test_setup = pipeline_setup.PipelineSetup()
    
    for key in ['setup_g', 'setup_r', 'setup_i']:
        assert type(params[key]) == type(test_setup)

def test_harvest_dataset_parameters():
    
    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DIR})
    
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
    
    dataset_params = stage3_db_ingest.harvest_dataset_parameters(setup,reduction_metadata)
    
    assert type(dataset_params) == type({})
    assert len(dataset_params) > 1

def test_commit_reference_image():
    
    if os.path.isfile(db_file_path):
        os.remove(db_file_path)
    
    log = logs.start_stage_log( TEST_DIR, 'stage3_db_ingest_test' )

    (facility_keys, software_keys, image_keys, params) = fetch_test_db_contents()

    conn = phot_db.get_connection(dsn=db_file_path)
    
    phot_db.check_before_commit(conn, params, 'facilities', facility_keys, 'facility_code')
    phot_db.check_before_commit(conn, params, 'software', software_keys, 'version')
    phot_db.check_before_commit(conn, params, 'images', image_keys, 'filename')
    
    stage3_db_ingest.commit_reference_image(conn, params, log)
    
    query = 'SELECT refimg_id,filename FROM reference_images WHERE filename="'+\
            params['filename']+'"'
    t = phot_db.query_to_astropy_table(conn, query, args=())

    assert len(t) == 1
    
    conn.close()
    
    logs.close_log(log)

def test_commit_reference_component():

    if os.path.isfile(db_file_path):
        os.remove(db_file_path)
    
    log = logs.start_stage_log( TEST_DIR, 'stage3_db_ingest_test' )

    (facility_keys, software_keys, image_keys, params) = fetch_test_db_contents()
    
    conn = phot_db.get_connection(dsn=db_file_path)
    
    phot_db.check_before_commit(conn, params, 'facilities', facility_keys, 'facility_code')
    phot_db.check_before_commit(conn, params, 'software', software_keys, 'version')
    phot_db.check_before_commit(conn, params, 'images', image_keys, 'filename')
    
    
    query = 'SELECT img_id FROM images WHERE filename="'+\
            params['filename']+'"'
    image = phot_db.query_to_astropy_table(conn, query, args=())
    
    stage3_db_ingest.commit_reference_image(conn, params, log)
    
    stage3_db_ingest.commit_reference_component(conn, params, log)
    
    query = 'SELECT component_id,image,reference_image FROM reference_components WHERE image='+\
            str(image['img_id'][0])
    t = phot_db.query_to_astropy_table(conn, query, args=())
    
    assert len(t) == 1
    
    conn.close()
    
    logs.close_log(log)
    
def test_run_stage3_db_ingest():
    
    stage3_db_ingest.run_stage3_db_ingest()

def fetch_test_db_contents():
    
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
    
    params = {'facility_code': 'lsc-doma-1m0a-fl15', 
              'site': 'lsc', 
              'enclosure': 'doma', 
              'telescope': '1m0a', 
              'instrument': 'fl15',
              'code_name': 'stage3_test_version', 
              'stage': 'stage3_test', 
              'version': 'stage3_test_v0.1',
              'filename': 'lsc1m005-fl15-20170418-0131-e91_cropped.fits', 
              'field_id': 'ROME-FIELD-16',
              'filter_name': 'gp',
              'date_obs_utc': '2016-05-18T10:57:30',
              'date_obs_jd': 2458000.0,
              'exposure_time': 300.0,
              'fwhm': None, 'fwhm_err':None,
              'ellipticity':None,'ellipticity_err':None,
              'slope':None,'slope_err':None,
              'intercept':None,'intercept_err':None,
              'wcsfrcat':None,'wcsimcat':None,'wcsmatch':None,
              'wcsnref':None,'wcstol':None,'wcsra':None,
              'wcsdec':None,'wequinox':None,'wepoch':None,'radecsys':None,
              'ctype1':None,'ctype2':None,
              'cdelt1':None,'cdelt2':None,
              'crota1':None,'crota2':None,
              'secpix1':None,'secpix2':None,
              'wcssep':None,'equinox':None,
              'cd1_1':None,'cd1_2':None,'cd2_1':None,'cd2_2':None,'epoch':None,
              'airmass':None,'moon_phase':None,'moon_separation':None,
              'delta_x':None,'delta_y':None}

    return facility_keys, software_keys, image_keys, params

def test_read_combined_star_catalog():
    
    file_path = os.path.join(TEST_DIR,'..','combined_star_catalog.fits')
    
    star_catalog = stage3_db_ingest.read_combined_star_catalog({'combined_starcat': file_path})
    
    assert len(star_catalog) > 0

def test_commit_stars():
    
    if os.path.isfile(db_file_path):
        os.remove(db_file_path)
    
    log = logs.start_stage_log( TEST_DIR, 'stage3_db_ingest_test' )
    
    (facility_keys, software_keys, image_keys, params) = fetch_test_db_contents()
        
    conn = phot_db.get_connection(dsn=db_file_path)
    phot_db.check_before_commit(conn, params, 'facilities', facility_keys, 'facility_code')
    phot_db.check_before_commit(conn, params, 'software', software_keys, 'version')
    phot_db.check_before_commit(conn, params, 'images', image_keys, 'filename')
    stage3_db_ingest.commit_reference_image(conn, params, log)
    
    star_catalog = [ (1, 270.13802459, -28.32124566),
                     (2, 269.93099598, -28.32181808),
                     (3, 270.16307615, -28.32229145) ]
                     
    star_ids = stage3_db_ingest.commit_stars(conn, params, star_catalog, log)
    
    assert len(star_ids) == len(star_catalog)
    assert type(star_ids) == type(np.zeros(1))
    
    query = 'SELECT star_id,ra,dec,reference_image FROM stars'
    t = phot_db.query_to_astropy_table(conn, query, args=())

    assert len(t) == 3
    for j in range(0,len(t),1):
        assert t['star_id'][j] == star_catalog[j][0]
        assert t['ra'][j] == star_catalog[j][1]
        assert t['dec'][j] == star_catalog[j][2]
    
    conn.close()
    
    logs.close_log(log)
    
if __name__ == '__main__':
    
    #test_configure_setup()
    #test_harvest_dataset_parameters()
    #test_commit_reference_image()
    #test_commit_reference_component()
    #test_read_combined_star_catalog()
    test_commit_stars()
    #test_run_stage3_db_ingest()
