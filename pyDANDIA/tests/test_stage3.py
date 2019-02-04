# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 11:58:40 2017

@author: rstreet
"""

import os
import sys
cwd = os.getcwd()
sys.path.append(os.path.join(cwd,'../'))
import logs
import pipeline_setup
import metadata
import stage3
import starfind
import phot_db
from astropy.io import fits
import numpy as np

TEST_DIR = os.path.join(cwd,'data','proc',
                        'ROME-FIELD-0002_lsc-doma-1m0-05-fl15_ip')

DB_FILE = 'test.db'
db_file_path = os.path.join(TEST_DIR, '..', DB_FILE)

def test_run_stage3():
    """Function to test the execution of Stage 3 of the pipeline, end-to-end"""
    
    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DIR})
    
    (status, report) = stage3.run_stage3(setup)

def test_find_reference_flux():
    
    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DIR})
    
    meta = metadata.MetaData()
    meta.load_a_layer_from_file( setup.red_dir, 
                                'pyDANDIA_metadata.fits', 
                                'data_architecture' )
    meta.load_a_layer_from_file( setup.red_dir, 
                                'pyDANDIA_metadata.fits', 
                                'images_stats' )
    
    reference_image_path = os.path.join(str(meta.data_architecture[1]['REF_PATH'][0]),
                                     str(meta.data_architecture[1]['REF_IMAGE'][0]))

    log = logs.start_stage_log( cwd, 'test_stage3' )
    
    scidata = fits.getdata(reference_image_path)
    
    detected_sources = starfind.detect_sources(setup,meta,reference_image_path,
                                               scidata,log)

    ref_flux = stage3.find_reference_flux(detected_sources,log)
    
    assert(type(ref_flux) == type(np.sqrt(9.0)))
    assert(ref_flux > 0.0)

    logs.close_log(log)
    
def test_add_reference_image_to_db():
    
    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DIR})
    setup.phot_db_path = db_file_path
    
    log = logs.start_stage_log( cwd, 'test_stage3' )
    
    meta = metadata.MetaData()
    meta.load_a_layer_from_file( setup.red_dir, 
                                'pyDANDIA_metadata.fits', 
                                'headers_summary' )
    meta.load_a_layer_from_file( setup.red_dir, 
                                'pyDANDIA_metadata.fits', 
                                'data_architecture' )
    
    ref_image_name = meta.data_architecture[1]['REF_IMAGE'].data[0]
    
    ref_db_id = stage3.add_reference_image_to_db(setup, meta, log=log)
    
    conn = phot_db.get_connection(dsn=setup.phot_db_path)
    
    query = 'SELECT refimg_name FROM reference_images'
    t = phot_db.query_to_astropy_table(conn, query, args=())
    
    assert (ref_image_name in t[0]['refimg_name'])
    assert type(ref_db_id) == type(np.int64(1))
    
    logs.close_log(log)
    conn.close()

def test_ingest_stars_to_db():
    
    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DIR})
    setup.phot_db_path = db_file_path
    
    log = logs.start_stage_log( cwd, 'test_stage3' )
    
    if os.path.isfile(setup.phot_db_path):
        os.remove(setup.phot_db_path)
    
    meta = metadata.MetaData()
    meta.load_a_layer_from_file( setup.red_dir, 
                                'pyDANDIA_metadata.fits', 
                                'star_catalog' )
    
    ref_star_catalog = np.zeros([len(meta.star_catalog[1]),5])
    ref_star_catalog[:,0] = meta.star_catalog[1]['star_index'].data
    ref_star_catalog[:,1] = meta.star_catalog[1]['x_pixel'].data
    ref_star_catalog[:,2] = meta.star_catalog[1]['y_pixel'].data
    ref_star_catalog[:,3] = meta.star_catalog[1]['RA_J2000'].data
    ref_star_catalog[:,4] = meta.star_catalog[1]['DEC_J2000'].data
    
    star_ids = stage3.ingest_stars_to_db(setup, ref_star_catalog, log=log)
    
    
    conn = phot_db.get_connection(dsn=db_file_path)
    
    query = 'SELECT star_id, ra, dec FROM stars'
    t = phot_db.query_to_astropy_table(conn, query, args=())

    assert (len(star_ids) == len(ref_star_catalog))
    assert (len(t) == len(ref_star_catalog))
    
    logs.close_log(log)
    conn.close()

def test_ingest_star_catalog_to_db():
    
    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DIR})
    setup.phot_db_path = db_file_path
    
    log = logs.start_stage_log( cwd, 'test_stage3' )
    
    if os.path.isfile(setup.phot_db_path):
        os.remove(setup.phot_db_path)
    
    conn = phot_db.get_connection(dsn=db_file_path)
    
    meta = metadata.MetaData()
    meta.load_a_layer_from_file( setup.red_dir, 
                                'pyDANDIA_metadata.fits', 
                                'headers_summary' )
    meta.load_a_layer_from_file( setup.red_dir, 
                                'pyDANDIA_metadata.fits', 
                                'data_architecture' )
    meta.load_a_layer_from_file( setup.red_dir, 
                                'pyDANDIA_metadata.fits', 
                                'star_catalog' )
    
    ref_star_catalog = np.zeros([len(meta.star_catalog[1]),9])
    ref_star_catalog[:,0] = meta.star_catalog[1]['star_index'].data
    ref_star_catalog[:,1] = meta.star_catalog[1]['x_pixel'].data
    ref_star_catalog[:,2] = meta.star_catalog[1]['y_pixel'].data
    ref_star_catalog[:,3] = meta.star_catalog[1]['RA_J2000'].data
    ref_star_catalog[:,4] = meta.star_catalog[1]['DEC_J2000'].data
    ref_star_catalog[:,5] = meta.star_catalog[1]['ref_flux'].data
    ref_star_catalog[:,6] = meta.star_catalog[1]['ref_flux_err'].data
    ref_star_catalog[:,7] = meta.star_catalog[1]['ref_mag'].data
    ref_star_catalog[:,8] = meta.star_catalog[1]['ref_mag_err'].data
    
    ref_db_id = stage3.add_reference_image_to_db(setup, meta, log=log)
    
    star_ids = stage3.ingest_stars_to_db(setup, ref_star_catalog, log=log)
    
    bandpass = 'i'
    
    stage3.ingest_star_catalog_to_db(setup, ref_star_catalog, 
                                     ref_db_id, star_ids,
                                     bandpass, log=log)
    
    conn = phot_db.get_connection(dsn=db_file_path)
    
    query = 'SELECT reference_images, star_id, reference_mag_i, reference_mag_err_i FROM ref_phot'
    t = phot_db.query_to_astropy_table(conn, query, args=())
    
    assert (len(star_ids) == len(ref_star_catalog))
    assert (len(t) == len(ref_star_catalog))
    
    logs.close_log(log)
    conn.close()

if __name__ == '__main__':
    
    #test_find_reference_flux()
    test_run_stage3()
    #test_add_reference_image_to_db()
    #test_ingest_stars_to_db()
    #test_ingest_star_catalog_to_db()
    