# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 10:10:35 2019

@author: rstreet
"""

import os
import sys
cwd = os.getcwd()
sys.path.append(os.path.join(cwd,'../'))
import phot_db
import pipeline_setup
import metadata
import stage3_db_ingest
import logs
import sqlite3
import numpy as np
from astropy import table
from astropy.io import fits

TEST_DIR = os.path.join(cwd,'data','proc',
                        'ROME-FIELD-0002_lsc-doma-1m0-05-fl15_ip')
DB_FILE = 'test.db'
db_file_path = os.path.join(TEST_DIR, '..', DB_FILE)

def generate_test_catalog():
    """Function to provide a basic star catalog for testing purposes"""

    names = [ 'star_id', 'ra', 'dec' ]

    tuples = [ (1, 269.5155763609925, -27.996414691517113),
               (2, 269.526266123634,  -27.996503792470577),
               (3, 269.54404280561437,-27.99650536269626),
               (4, 269.51712783133013,-27.996565934355385),
               (5, 269.5252284716887, -27.996590871295435) ]

    return names, tuples

def test_get_connection():
    """Function to test the initialization of a connection to an SQLITE3
    database"""

    conn = phot_db.get_connection(dsn=db_file_path)

    test_conn = sqlite3.connect(db_file_path)

    assert type(conn) == type(test_conn)

    conn.close()

def test_populate_db_defaults():
    """Function to test the population of default values in the filters table
    of the phot_db"""

    conn = phot_db.get_connection(dsn=db_file_path)

    query = 'SELECT filter_name FROM filters'
    t = phot_db.query_to_astropy_table(conn, query, args=())

    assert len(t) == 3
    for f in ['gp', 'rp', 'ip']:
        assert f in t['filter_name']

    conn.close()

def test_check_before_commit():
    """Function to test the submission of facilities to the database

    NOTE: Separators MUST be overscores NOT dashes, since dash-separated
    quantities do not parse properly on DB commit.
    """

    params = {'facility_code': 'lsc-doma-1m0a-fl15',
              'site': 'lsc',
              'enclosure': 'doma',
              'telescope': '1m0a',
              'instrument': 'fl15'}

    conn = phot_db.get_connection(dsn=db_file_path)

    table_keys = ['facility_code', 'site', 'enclosure', 'telescope', 'instrument']

    phot_db.check_before_commit(conn, params, 'facilities', table_keys, 'facility_code')

    query = 'SELECT facility_code,site,enclosure,telescope,instrument FROM facilities WHERE facility_code = "'+params['facility_code']+'"'
    t = phot_db.query_to_astropy_table(conn, query, args=())

    assert params['facility_code'] == t['facility_code']
    assert params['site'] == t['site']
    assert params['enclosure'] == t['enclosure']
    assert params['telescope'] == t['telescope']
    assert params['instrument'] == t['instrument']

    phot_db.check_before_commit(conn, params, 'facilities', table_keys, 'facility_code')

    t = phot_db.query_to_astropy_table(conn, query, args=())

    idx = np.where(t['facility_code'] == params['facility_code'])[0]

    assert len(idx) == 1

    conn.close()

def test_feed_to_table_many():

    if os.path.isfile(db_file_path):
        os.remove(db_file_path)

    conn = phot_db.get_connection(dsn=db_file_path)

    (names, tuples) = generate_test_catalog()

    phot_db.feed_to_table_many(conn, 'Stars', names, tuples)

    query = 'SELECT star_id,ra,dec FROM stars'
    t = phot_db.query_to_astropy_table(conn, query, args=())

    assert len(t) == len(tuples)
    conn.close()

def test_ingest_astropy_table():

    if os.path.isfile(db_file_path):
        os.remove(db_file_path)

    conn = phot_db.get_connection(dsn=db_file_path)

    n_entries = 2000

    data = [ table.Column(name='ra', data=np.linspace(269.0,272.0,n_entries)),
              table.Column(name='dec', data=np.linspace(-27.5, -28.9, n_entries)),
              table.Column(name='type', data=['dwarf']*n_entries) ]

    stars = table.Table(data=data)

    phot_db.ingest_astropy_table(conn, 'stars', stars)

    query = 'SELECT star_id, ra, dec,type FROM stars'
    t = phot_db.query_to_astropy_table(conn, query, args=())

    assert len(t) == n_entries
    conn.close()

def test_ingest_reference_in_db():
    """Function to test the ingestion of a RefereinceImage into the Phot DB"""

    if os.path.isfile(db_file_path):
        os.remove(db_file_path)

    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DIR})
    setup.phot_db_path = db_file_path

    conn = phot_db.get_connection(dsn=setup.phot_db_path)

    meta = metadata.MetaData()
    meta.load_a_layer_from_file( setup.red_dir,
                                'pyDANDIA_metadata.fits',
                                'headers_summary' )
    meta.load_a_layer_from_file( setup.red_dir,
                                'pyDANDIA_metadata.fits',
                                'data_architecture' )

    ref_image_dir = meta.data_architecture[1]['REF_PATH'].data[0]

    ref_image_name = meta.data_architecture[1]['REF_IMAGE'].data[0]

    i = np.where(ref_image_name == meta.headers_summary[1]['IMAGES'].data)[0][0]

    ref_header = meta.headers_summary[1][i]

    field_id = 'ROME-TEST-01'
    version = 'Code_test'

    phot_db.ingest_reference_in_db(conn, setup, ref_header,
                                                ref_image_dir,
                                                ref_image_name,
                                                field_id, version)

    query = 'SELECT refimg_name,field_id,current_best FROM reference_images WHERE refimg_name="'+ref_image_name+'"'
    t = phot_db.query_to_astropy_table(conn, query, args=())

    assert t[0]['refimg_name'] == ref_image_name
    assert t[0]['current_best'] == 1

    conn.close()

def test_box_search_on_position():

    if os.path.isfile(db_file_path):
        os.remove(db_file_path)

    conn = phot_db.get_connection(dsn=db_file_path)

    (names, tuples) = generate_test_catalog()

    phot_db.feed_to_table_many(conn, 'Stars', names, tuples)

    ra_centre = 269.5
    dec_centre = -28.0
    dra = 0.05
    ddec = 0.1

    results = phot_db.box_search_on_position(conn, ra_centre, dec_centre, dra, ddec)

    assert len(results) == len(tuples)

    conn.close()

def test_update_table_entry():

    if os.path.isfile(db_file_path):
        os.remove(db_file_path)

    conn = phot_db.get_connection(dsn=db_file_path)

    (names, tuples) = generate_test_catalog()

    phot_db.feed_to_table_many(conn, 'Stars', names, tuples)

    query = 'SELECT star_id,ra,dec FROM stars'
    t = phot_db.query_to_astropy_table(conn, query, args=())
    print(t)

    table_name = 'Stars'
    key_name = 'ra'
    search_key = 'star_id'
    entry_id = 1
    value = 18.000

    phot_db.update_table_entry(conn,table_name,key_name,search_key,entry_id,value)

    query = 'SELECT star_id,ra,dec FROM stars'
    t = phot_db.query_to_astropy_table(conn, query, args=())
    print(t)

    idx = np.where(t['star_id'] == entry_id)

    assert t['ra'][idx].data == value

    conn.close()

def test_update_stars_ref_image_id():

    if os.path.isfile(db_file_path):
        os.remove(db_file_path)

    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DIR})
    setup.phot_db_path = db_file_path

    conn = phot_db.get_connection(dsn=db_file_path)

    meta = metadata.MetaData()
    meta.load_a_layer_from_file( setup.red_dir,
                                'pyDANDIA_metadata.fits',
                                'headers_summary' )
    meta.load_a_layer_from_file( setup.red_dir,
                                'pyDANDIA_metadata.fits',
                                'data_architecture' )

    ref_image_dir = meta.data_architecture[1]['REF_PATH'].data[0]

    ref_image_name = meta.data_architecture[1]['REF_IMAGE'].data[0]

    i = np.where(ref_image_name == meta.headers_summary[1]['IMAGES'].data)[0][0]

    ref_header = meta.headers_summary[1][i]

    field_id = 'ROME-TEST-01'
    version = 'Code_test'

    phot_db.ingest_reference_in_db(conn, setup, ref_header,
                                                ref_image_dir,
                                                ref_image_name,
                                                field_id, version)

    query = 'SELECT refimg_id FROM reference_images WHERE refimg_name="'+ref_image_name+'"'
    t = phot_db.query_to_astropy_table(conn, query, args=())
    refimg_id = t['refimg_id'].data

    (names, tuples) = generate_test_catalog()

    phot_db.feed_to_table_many(conn, 'Stars', names, tuples)

    query = 'SELECT star_id FROM stars'
    t = phot_db.query_to_astropy_table(conn, query, args=())
    star_ids = t['star_id'].data[:-3]
    print(star_ids)

    phot_db.update_stars_ref_image_id(conn, ref_image_name, star_ids)

    query = 'SELECT star_id,ra,dec,reference_images FROM stars'
    t = phot_db.query_to_astropy_table(conn, query, args=())

    assert t['reference_images'].all() == refimg_id

    conn.close()

def test_find_previous_reference_image_for_dataset():

    if os.path.isfile(db_file_path):
        os.remove(db_file_path)

    log = logs.start_stage_log( TEST_DIR, 'test_phot_db' )

    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DIR})
    setup.phot_db_path = db_file_path

    conn = phot_db.get_connection(dsn=db_file_path)

    meta = metadata.MetaData()
    meta.load_a_layer_from_file( setup.red_dir,
                                'pyDANDIA_metadata.fits',
                                'headers_summary' )
    meta.load_a_layer_from_file( setup.red_dir,
                                'pyDANDIA_metadata.fits',
                                'data_architecture' )

    ref_image_dir = meta.data_architecture[1]['REF_PATH'].data[0]

    ref_image_name = meta.data_architecture[1]['REF_IMAGE'].data[0]

    i = np.where(ref_image_name == meta.headers_summary[1]['IMAGES'].data)[0][0]

    ref_header = fits.getheader(os.path.join(ref_image_dir, ref_image_name))

    field_id = 'ROME-TEST-01'
    version = 'Code_test'

    (facility_keys, software_keys, image_keys) = stage3_db_ingest.define_table_keys()

    params = {'filename': ref_image_name}
    params['version'] = 'test'
    params['code_name'] = 'test_phot_db'
    params['stage'] = 'stage3_db_ingest'
    params['site'] = ref_header['SITEID']
    params['enclosure'] = ref_header['ENCID']
    params['telescope'] = ref_header['TELID']
    params['instrument'] = ref_header['INSTRUME']
    params['filter_name'] = ref_header['FILTER']
    params['facility_code'] = phot_db.get_facility_code(params)

    phot_db.check_before_commit(conn, params, 'facilities', facility_keys, 'facility_code')
    phot_db.check_before_commit(conn, params, 'software', software_keys, 'version')
    stage3_db_ingest.commit_reference_image(conn, params, log)

    query = 'SELECT refimg_id FROM reference_images WHERE filename="'+ref_image_name+'"'
    t = phot_db.query_to_astropy_table(conn, query, args=())
    test_refimg_id = t['refimg_id'].data

    ref_id = phot_db.find_previous_reference_image_for_dataset(conn,params)

    assert test_refimg_id == ref_id

    logs.close_log(log)
    conn.close()

def setup_test_phot_db(log):

    if os.path.isfile(db_file_path):
        os.remove(db_file_path)

    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DIR})
    setup.phot_db_path = db_file_path

    conn = phot_db.get_connection(dsn=db_file_path)

    meta = metadata.MetaData()
    meta.load_all_metadata(setup.red_dir, 'pyDANDIA_metadata.fits')

    ref_image_dir = meta.data_architecture[1]['REF_PATH'].data[0]

    ref_image_name = meta.data_architecture[1]['REF_IMAGE'].data[0]

    ref_image_path = os.path.join(ref_image_dir, ref_image_name)

    i = np.where(ref_image_name == meta.headers_summary[1]['IMAGES'].data)[0][0]

    ref_header = fits.getheader(os.path.join(ref_image_dir, ref_image_name))

    field_id = 'ROME-TEST-01'
    version = 'Code_test'

    (facility_keys, software_keys, image_keys) = stage3_db_ingest.define_table_keys()

    image_path = os.path.join(setup.red_dir, 'data', 'lsc1m005-fl15-20170610-0096-e91_cropped.fits')

    params = stage3_db_ingest.harvest_image_params(meta,image_path,ref_image_path)

    params['filename'] = ref_image_name
    params['ref_filename'] = ref_image_name
    params['version'] = 'test'
    params['code_name'] = 'test_phot_db'
    params['stage'] = 'stage3_db_ingest'
    params['site'] = ref_header['SITEID']
    params['enclosure'] = ref_header['ENCID']
    params['telescope'] = ref_header['TELID']
    params['instrument'] = ref_header['INSTRUME']
    params['filter_name'] = ref_header['FILTER']
    query = 'SELECT filter_id FROM filters WHERE filter_name ="'+params['filter_name']+'"'
    params['filter'] = phot_db.query_to_astropy_table(conn, query, args=())['filter_id'][0]
    params['field_id'] = field_id
    params['facility_code'] = phot_db.get_facility_code(params)
    params['stamp'] = '0'

    phot_db.check_before_commit(conn, params, 'facilities', facility_keys, 'facility_code')
    phot_db.check_before_commit(conn, params, 'software', software_keys, 'version')

    query = 'SELECT facility_id FROM facilities WHERE facility_code ="'+params['facility_code']+'"'
    params['facility'] = phot_db.query_to_astropy_table(conn, query, args=())['facility_id'][0]
    phot_db.check_before_commit(conn, params, 'images', image_keys, 'filename')

    stage3_db_ingest.commit_reference_image(conn, params, log)
    stage3_db_ingest.commit_reference_component(conn, params, log)

    star_ids = stage3_db_ingest.commit_stars(conn, params, meta, log)


    params['filename'] = os.path.basename(image_path)
    params['ref_filename'] = ref_image_name
    phot_db.check_before_commit(conn, params, 'images', image_keys, 'filename')

    stage3_db_ingest.commit_stamps_to_db(conn, meta)

    return conn, params, ref_image_name

def test_cascade_delete_reference_image():

    log = logs.start_stage_log( TEST_DIR, 'test_phot_db' )

    (conn,params,ref_image_name) = setup_test_phot_db(log)

    query = 'SELECT refimg_id FROM reference_images WHERE filename="'+ref_image_name+'"'
    t = phot_db.query_to_astropy_table(conn, query, args=())
    refimg_id = t['refimg_id'].data[0]

    query = 'SELECT component_id, image FROM reference_components WHERE reference_image="'+str(refimg_id)+'"'
    t = phot_db.query_to_astropy_table(conn, query, args=())

    print('Pre-deletion: ',t)
    print('Reference image entry = '+str(refimg_id))

    phot_db.cascade_delete_reference_images(conn, [refimg_id], log)

    t = phot_db.query_to_astropy_table(conn, query, args=())
    print('Post deletion: ',t)

    assert len(t) == 0

    logs.close_log(log)
    conn.close()

def test_find_reference_image_for_dataset():

    log = logs.start_stage_log( TEST_DIR, 'test_phot_db' )

    (conn,params,ref_image_name) = setup_test_phot_db(log)

    id_list = phot_db.find_reference_image_for_dataset(conn,params)

    assert len(id_list) == 1
    assert id_list[0] == 1

    logs.close_log(log)
    conn.close()

def test_find_primary_reference_image_for_field():

    log = logs.start_stage_log( TEST_DIR, 'test_phot_db' )

    (conn,params,ref_image_name) = setup_test_phot_db(log)

    primary_refimg_id = phot_db.find_primary_reference_image_for_field(conn)

    assert type(primary_refimg_id) == type(np.int64())
    assert primary_refimg_id == 1

    logs.close_log(log)

    conn.close()

def test_add_table_to_existing_db():

    if os.path.isfile(db_file_path):
        os.remove(db_file_path)

    conn = phot_db.get_connection(dsn=db_file_path)

    (names, tuples) = generate_test_catalog()

    phot_db.feed_to_table_many(conn, 'Stars', names, tuples)

    query = 'SELECT star_id,ra,dec FROM stars'
    t = phot_db.query_to_astropy_table(conn, query, args=())

    assert len(t) == len(tuples)
    conn.close()

    conn = phot_db.get_connection(dsn=db_file_path)

    phot_db.ensure_extra_table(conn,'DetrendingParameters')

    conn.close()

if __name__ == '__main__':

    #test_get_connection()
    #test_populate_db_defaults()
    #test_feed_to_table_many()
    #test_ingest_astropy_table()
    #test_ingest_reference_in_db()
    #test_box_search_on_position()
    #test_update_table_entry()
    #test_update_stars_ref_image_id()
    #test_check_before_commit()
    #test_find_previous_reference_image_for_dataset()
    #test_cascade_delete_reference_image()
    #test_find_reference_image_for_dataset()
    #test_find_primary_reference_image_for_field()
    test_add_table_to_existing_db()
