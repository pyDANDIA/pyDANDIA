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
from skimage.transform import AffineTransform
from pyDANDIA import  logs
from pyDANDIA import  metadata
from pyDANDIA import  phot_db
from pyDANDIA import  stage3_db_ingest
from pyDANDIA import  pipeline_setup
from pyDANDIA import  match_utils


TEST_DIR = os.path.join(cwd,'data','proc',
                        'ROME-FIELD-0002_lsc-doma-1m0-05-fl15_ip')
DB_FILE = 'test.db'
db_file_path = os.path.join(TEST_DIR, '..', DB_FILE)

def test_configure_setup():

    params = stage3_db_ingest.configure_setup()

    test_setup = pipeline_setup.PipelineSetup()

    for key in ['setup_g', 'setup_r', 'setup_i']:
        assert type(params[key]) == type(test_setup)

def test_harvest_stage3_parameters():

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

    dataset_params = stage3_db_ingest.harvest_stage3_parameters(setup,reduction_metadata)

    assert type(dataset_params) == type({})
    assert len(dataset_params) > 1

    #print(dataset_params)

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

    (facility_keys, software_keys, image_keys, params, star_catalog, cols) = fetch_test_db_contents()

    conn = phot_db.get_connection(dsn=db_file_path)

    phot_db.check_before_commit(conn, params, 'facilities', facility_keys, 'facility_code')
    phot_db.check_before_commit(conn, params, 'software', software_keys, 'version')

    query = 'SELECT facility_id FROM facilities WHERE facility_code ="'+params['facility_code']+'"'
    params['facility'] = phot_db.query_to_astropy_table(conn, query, args=())['facility_id'][0]

    query = 'SELECT filter_id,filter_name FROM filters WHERE filter_name ="'+params['filter_name']+'"'
    result = phot_db.query_to_astropy_table(conn, query, args=())
    params['filter'] = result['filter_id'][0]

    phot_db.check_before_commit(conn, params, 'images', image_keys, 'filename')

    query = 'SELECT img_id FROM images WHERE filename="'+\
            params['filename']+'"'
    image = phot_db.query_to_astropy_table(conn, query, args=())
    print(image)

    stage3_db_ingest.commit_reference_image(conn, params, log)

    stage3_db_ingest.commit_reference_component(conn, params, log)

    query = 'SELECT component_id,image,reference_image FROM reference_components WHERE image='+\
            str(image['img_id'][0])
    t = phot_db.query_to_astropy_table(conn, query, args=())
    assert len(t) == 1

    conn.close()

    logs.close_log(log)

def test_run_stage3_db_ingest_primary_ref():

    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DIR})

    stage3_db_ingest.run_stage3_db_ingest(setup, primary_ref=True)


def test_run_stage3_db_ingest():

    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DIR})

    stage3_db_ingest.run_stage3_db_ingest(setup, primary_ref=True)

    stage3_db_ingest.run_stage3_db_ingest(setup, primary_ref=False)


def fetch_test_db_contents():

    facility_keys = ['facility_code', 'site', 'enclosure',
                     'telescope', 'instrument']
    software_keys = ['code_name', 'stage', 'version']
    image_keys =    ['facility', 'filter', 'field_id', 'filename',
                     'date_obs_utc','date_obs_jd','exposure_time',
                     'fwhm','fwhm_err',
                     'ellipticity','ellipticity_err',
                     'slope','slope_err','intercept','intercept_err',
                     'wcsfrcat','wcsimcat','wcsmatch','wcsnref','wcstol','wcsra',
                     'wcsdec','wequinox','wepoch','radecsys',
                     'ctype1','ctype2','crpix1', 'crpix2', 'crval1', 'crval2',
                     'cdelt1','cdelt2','crota1','crota2',
                     'cunit1', 'cunit2',
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
              'ref_filename': 'lsc1m005-fl15-20170418-0131-e91_cropped.fits',
              'field_id': 'ROME-FIELD-16',
              'filter_name': 'gp',
              'date_obs_utc': '2016-05-18T10:57:30',
              'date_obs_jd': 2458000.0,
              'exposure_time': 300.0,
              'hjd_ref': 2458000.0,
              'RA': '17:59:27.05',
              'Dec': '-28:36:37.0',
              'fwhm': None, 'fwhm_err':None,
              'ellipticity':None,'ellipticity_err':None,
              'slope':None,'slope_err':None,
              'intercept':None,'intercept_err':None,
              'wcsfrcat':None,'wcsimcat':None,'wcsmatch':None,
              'wcsnref':None,'wcstol':None,'wcsra':None,
              'wcsdec':None,'wequinox':None,'wepoch':None,'radecsys':None,
              'ctype1':None,'ctype2':None,
              'crpix1':None,'crpix2':None,
              'crval1':None,'crval2':None,
              'cdelt1':None,'cdelt2':None,
              'crota1':None,'crota2':None,
              'cunit1':None,'cunit2':None,
              'secpix1':None,'secpix2':None,
              'wcssep':None,'equinox':None,
              'cd1_1':None,'cd1_2':None,'cd2_1':None,'cd2_2':None,'epoch':None,
              'airmass':None,'moon_phase':None,'moon_separation':None,
              'delta_x':None,'delta_y':None}

    star_catalog = [ (1, 270.13802459251167, -28.321245655324272,
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      2607.8596, 2.6583354, 0.0, 0.0, 0.0, 0.0,
                      -8455.866, -99.999, 0.0, 0.0, 0.0, 0.0),
                      (2, 269.93099597613417, -28.32181808234723,
                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                       921.8587, 8.439114, 0.0, 0.0, 0.0, 0.0,
                       -7544.8057, -99.999, 0.0, 0.0, 0.0, 0.0),
                      (3, 270.1630761544735, -28.322291445004893,
                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                       2811.868, 12.470031, 0.0, 0.0, 0.0, 0.0,
                       -9883.625, -99.999, 0.0, 0.0, 0.0, 0.0) ]

    cols = {'gp': {'xcol': 25,
                   'ycol': 26,
                   'mag_col': 11,
                   'mag_err_col': 12,
                   'cal_mag_col': 13,
                   'cal_mag_err_col': 14,
                   'flux_col': 31,
                   'flux_err_col': 32},
            'rp': {'xcol': 23,
                   'ycol': 24,
                   'mag_col': 7,
                   'mag_err_col': 8,
                   'cal_mag_col': 9,
                   'cal_mag_err_col': 10,
                   'flux_col': 29,
                   'flux_err_col': 30},
            'ip':{'xcol': 21,
                   'ycol': 22,
                   'mag_col': 3,
                   'mag_err_col': 4,
                   'cal_mag_col': 5,
                   'cal_mag_err_col': 6,
                   'flux_col': 27,
                   'flux_err_col': 28}}

    return facility_keys, software_keys, image_keys, params, star_catalog, cols

def test_read_combined_star_catalog():

    file_path = os.path.join(TEST_DIR,'..','combined_star_catalog.fits')

    star_catalog = stage3_db_ingest.read_combined_star_catalog({'combined_starcat': file_path})

    assert len(star_catalog) > 0

def test_commit_stars():

    if os.path.isfile(db_file_path):
        os.remove(db_file_path)

    log = logs.start_stage_log( TEST_DIR, 'stage3_db_ingest_test' )

    (facility_keys, software_keys, image_keys, params, star_catalog, cols) = fetch_test_db_contents()

    conn = phot_db.get_connection(dsn=db_file_path)
    phot_db.check_before_commit(conn, params, 'facilities', facility_keys, 'facility_code')
    phot_db.check_before_commit(conn, params, 'software', software_keys, 'version')
    phot_db.check_before_commit(conn, params, 'images', image_keys, 'filename')
    stage3_db_ingest.commit_reference_image(conn, params, log)

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

def test_commit_photometry():

    if os.path.isfile(db_file_path):
        os.remove(db_file_path)

    log = logs.start_stage_log( TEST_DIR, 'stage3_db_ingest_test' )

    (facility_keys, software_keys, image_keys, params, star_catalog, cols) = fetch_test_db_contents()

    conn = phot_db.get_connection(dsn=db_file_path)
    phot_db.check_before_commit(conn, params, 'facilities', facility_keys, 'facility_code')
    phot_db.check_before_commit(conn, params, 'software', software_keys, 'version')
    phot_db.check_before_commit(conn, params, 'images', image_keys, 'filename')
    stage3_db_ingest.commit_reference_image(conn, params, log)
    star_ids = stage3_db_ingest.commit_stars(conn, params, star_catalog, log)

    for i,f in enumerate([ 'gp', 'rp', 'ip']):
        params['filter_name'] = f

        stage3_db_ingest.commit_photometry(conn, params, [star_catalog[i]], star_ids, log)

        query = 'SELECT phot_id,x,y,magnitude, magnitude_err, calibrated_mag, calibrated_mag_err, flux, flux_err,filter,software,image,reference_image,facility FROM phot WHERE star_id='+str(star_ids[i])
        t = phot_db.query_to_astropy_table(conn, query, args=())

        if len(t['x']) > 0:
            assert t['x'][0] == star_catalog[i][cols[f]['xcol']]
        if len(t['y']) > 0:
            assert t['y'][0] == star_catalog[i][cols[f]['ycol']]
        if len(t['magnitude']) > 0:
            assert t['magnitude'][0] == star_catalog[i][cols[f]['mag_col']]
        if len(t['magnitude_err']) > 0:
            assert t['magnitude_err'][0] == star_catalog[i][cols[f]['mag_err_col']]
        if len(t['calibrated_mag']) > 0:
            assert t['calibrated_mag'][0] == star_catalog[i][cols[f]['cal_mag_col']]
        if len(t['calibrated_mag_err']) > 0:
            assert t['calibrated_mag_err'][0] == star_catalog[i][cols[f]['cal_mag_err_col']]
        if len(t['flux']) > 0:
            assert t['flux'][0] == star_catalog[i][cols[f]['flux_col']]
        if len(t['flux_err']) > 0:
            assert t['flux_err'][0] == star_catalog[i][cols[f]['flux_err_col']]

    conn.close()

    logs.close_log(log)

def test_fetch_field_starlist():

    if os.path.isfile(db_file_path):
        os.remove(db_file_path)

    log = logs.start_stage_log( TEST_DIR, 'stage3_db_ingest_test' )

    (facility_keys, software_keys, image_keys, params, star_catalog, cols) = fetch_test_db_contents()

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
                                          'star_catalog' )

    conn = phot_db.get_connection(dsn=db_file_path)
    phot_db.check_before_commit(conn, params, 'facilities', facility_keys, 'facility_code')
    phot_db.check_before_commit(conn, params, 'software', software_keys, 'version')
    phot_db.check_before_commit(conn, params, 'images', image_keys, 'filename')
    stage3_db_ingest.commit_reference_image(conn, params, log)
    star_ids = stage3_db_ingest.commit_stars(conn, params, reduction_metadata, log)

    starlist = stage3_db_ingest.fetch_field_starlist(conn,params,log)

    test_table = table.Table()

    assert type(starlist) == type(test_table)
    assert len(starlist) > 0

    conn.close()

    logs.close_log(log)

def test_match_catalog_entries_with_starlist():

    if os.path.isfile(db_file_path):
        os.remove(db_file_path)

    log = logs.start_stage_log( TEST_DIR, 'stage3_db_ingest_test' )

    (facility_keys, software_keys, image_keys, params, star_catalog, cols) = fetch_test_db_contents()

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
                                          'star_catalog' )

    conn = phot_db.get_connection(dsn=db_file_path)
    phot_db.check_before_commit(conn, params, 'facilities', facility_keys, 'facility_code')
    phot_db.check_before_commit(conn, params, 'software', software_keys, 'version')
    phot_db.check_before_commit(conn, params, 'images', image_keys, 'filename')
    stage3_db_ingest.commit_reference_image(conn, params, log)
    star_ids = stage3_db_ingest.commit_stars(conn, params, reduction_metadata, log)
    stage3_db_ingest.commit_photometry(conn, params, reduction_metadata, star_ids, log)

    starlist = stage3_db_ingest.fetch_field_starlist(conn,params,log)

    ref_id_list = phot_db.find_reference_image_for_dataset(conn,params)

    matched_stars = stage3_db_ingest.match_catalog_entries_with_starlist(conn,params,starlist,
                                                                  reduction_metadata,
                                                                  ref_id_list[0],
                                                                  log)

    assert type(matched_stars) == type(match_utils.StarMatchIndex())
    assert len(matched_stars.cat1_index) > 0

    conn.close()

    logs.close_log(log)

def test_commit_photometry_matching():

    if os.path.isfile(db_file_path):
        os.remove(db_file_path)

    log = logs.start_stage_log( TEST_DIR, 'stage3_db_ingest_test' )

    (facility_keys, software_keys, image_keys, params, star_catalog, cols) = fetch_test_db_contents()

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
                                          'star_catalog' )

    conn = phot_db.get_connection(dsn=db_file_path)
    phot_db.check_before_commit(conn, params, 'facilities', facility_keys, 'facility_code')
    phot_db.check_before_commit(conn, params, 'software', software_keys, 'version')
    phot_db.check_before_commit(conn, params, 'images', image_keys, 'filename')
    stage3_db_ingest.commit_reference_image(conn, params, log)
    star_ids = stage3_db_ingest.commit_stars(conn, params, reduction_metadata, log)

    stage3_db_ingest.commit_photometry(conn, params, reduction_metadata, star_ids, log)

    starlist = stage3_db_ingest.fetch_field_starlist(conn,params,log)

    ref_id_list = phot_db.find_reference_image_for_dataset(conn,params)

    matched_stars = stage3_db_ingest.match_new_data_with_starlist(conn,params,starlist,
                                                                  reduction_metadata,
                                                                  ref_id_list[0],
                                                                  log)

    stage3_db_ingest.commit_photometry_matching(conn, params, reduction_metadata, matched_stars, log)

    query = 'SELECT phot_id,x,y,magnitude, magnitude_err, calibrated_mag, calibrated_mag_err, flux, flux_err,filter,software,image,reference_image,facility FROM phot WHERE star_id="'+str(matched_stars.cat1_index[0])+'"'
    t = phot_db.query_to_astropy_table(conn, query, args=())

    for j in range(0,10,1):
        assert len(t) == 2
        for col in ['x', 'y', 'magnitude', 'magnitude_err']:
            assert t[col][0] == t[col][1]

    conn.close()

    logs.close_log(log)

def test_calc_transform_to_primary_ref():

    if os.path.isfile(db_file_path):
        os.remove(db_file_path)

    log = logs.start_stage_log( TEST_DIR, 'stage3_db_ingest_test' )

    (facility_keys, software_keys, image_keys, params, star_catalog, cols) = fetch_test_db_contents()

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
                                          'star_catalog' )

    conn = phot_db.get_connection(dsn=db_file_path)
    phot_db.check_before_commit(conn, params, 'facilities', facility_keys, 'facility_code')
    phot_db.check_before_commit(conn, params, 'software', software_keys, 'version')
    phot_db.check_before_commit(conn, params, 'images', image_keys, 'filename')
    stage3_db_ingest.commit_reference_image(conn, params, log)
    star_ids = stage3_db_ingest.commit_stars(conn, params, reduction_metadata, log)

    stage3_db_ingest.commit_photometry(conn, params, reduction_metadata, star_ids, log)

    starlist = stage3_db_ingest.fetch_field_starlist(conn,params,log)

    ref_id_list = phot_db.find_reference_image_for_dataset(conn,params)

    matched_stars = stage3_db_ingest.match_new_data_with_starlist(conn,params,starlist,
                                                                  reduction_metadata,
                                                                  ref_id_list[0],
                                                                  log)

    transform = stage3_db_ingest.calc_transform_to_primary_ref(setup,matched_stars,log)

    assert type(transform) == type(AffineTransform())
    assert round(transform.params[0,2],3) == 0.0
    assert round(transform.params[1,2],3) == 0.0

    conn.close()

    logs.close_log(log)

def test_match_all_entries_with_starlist():

    if os.path.isfile(db_file_path):
        os.remove(db_file_path)

    log = logs.start_stage_log( TEST_DIR, 'stage3_db_ingest_test' )

    (facility_keys, software_keys, image_keys, params, star_catalog, cols) = fetch_test_db_contents()

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
                                          'star_catalog' )

    conn = phot_db.get_connection(dsn=db_file_path)
    phot_db.check_before_commit(conn, params, 'facilities', facility_keys, 'facility_code')
    phot_db.check_before_commit(conn, params, 'software', software_keys, 'version')
    phot_db.check_before_commit(conn, params, 'images', image_keys, 'filename')
    stage3_db_ingest.commit_reference_image(conn, params, log)
    star_ids = stage3_db_ingest.commit_stars(conn, params, reduction_metadata, log)

    stage3_db_ingest.commit_photometry(conn, params, reduction_metadata, star_ids, log)

    starlist = stage3_db_ingest.fetch_field_starlist(conn,params,log)

    transform = AffineTransform()

    matched_stars = stage3_db_ingest.match_all_entries_with_starlist(setup,conn,params,starlist,
                                                     reduction_metadata,
                                                     1,transform,log)

    assert type(matched_stars) == type(match_utils.StarMatchIndex())
    assert len(matched_stars.cat1_index) == len(starlist)
    assert matched_stars.cat1_x == matched_stars.cat2_x
    assert matched_stars.cat1_y == matched_stars.cat2_y

    conn.close()

    logs.close_log(log)

def test_generate_primary_ref_match_table():

    log = logs.start_stage_log( TEST_DIR, 'stage3_db_ingest_test' )
    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DIR})

    test_matched = match_utils.StarMatchIndex()
    test_transform = AffineTransform()

    reduction_metadata = metadata.MetaData()
    reduction_metadata.load_a_layer_from_file( setup.red_dir,
                                      'pyDANDIA_metadata.fits',
                                      'star_catalog' )

    (matched_stars, transform) = stage3_db_ingest.generate_primary_ref_match_table(reduction_metadata,log)

    assert type(matched_stars) == type(test_matched)
    assert type(transform) == type(test_transform)
    
    logs.close_log(log)

def test_harvest_image_params():

    kwargs = {'build_phot_db': False}

    image_params = harvest_image_params(
        reduction_metadata,
        image_path,
        ref_image_path,
        **kwargs
    )


if __name__ == '__main__':

    #test_configure_setup()
    #test_harvest_stage3_parameters()
    #test_commit_reference_image()
    #test_commit_reference_component()
    #test_read_combined_star_catalog()
    #test_commit_stars()
    #test_commit_photometry()
    #test_run_stage3_db_ingest_primary_ref()
    #test_fetch_field_starlist()
    #test_match_new_data_with_starlist()
    #test_commit_photometry_matching()
    #test_calc_transform_to_primary_ref()
    #test_match_all_entries_with_starlist()
    #test_run_stage3_db_ingest()
    test_generate_primary_ref_match_table()
