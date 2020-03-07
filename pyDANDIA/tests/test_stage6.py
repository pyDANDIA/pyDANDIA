import numpy as np
import os
import sys
cwd = os.getcwd()
sys.path.append(os.path.join(cwd,'../'))
import stage6
import pipeline_setup
import logs
import hd5_utils
import test_phot_db

TEST_DIR = os.path.join(cwd,'data','proc',
                        'ROME-FIELD-0002_lsc-doma-1m0-05-fl15_ip')
DB_FILE = 'test.db'
db_file_path = os.path.join(TEST_DIR, '..', DB_FILE)

ncolumns_matched = 23
ncolumns_unmatched = 22

def test_build_photometry_array():

    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DIR})

    log = logs.start_stage_log( cwd, 'test_photometry' )

    test_output_file = os.path.join(setup.red_dir,'photometry.hdf5')
    nstars = 1000
    nimages1 = 150
    nimages2 = 300

    if os.path.isfile(test_output_file):
        os.remove(test_output_file)

    print('Testing array building with no pre-existing photometry')
    (matched_phot, unmatched_phot) = stage6.build_photometry_array(setup,nimages1,nstars,log)

    assert matched_phot.shape == (nstars,nimages1,ncolumns_matched)
    assert unmatched_phot.shape == (nstars,nimages1,ncolumns_unmatched)

    print('Testing array building with pre-existing photometry')
    matched_dataset_phot_data = np.ones((nstars,nimages1,ncolumns_matched))
    unmatched_dataset_phot_data = np.ones((nstars,nimages1,ncolumns_unmatched))
    hd5_utils.write_phot_hd5(setup,matched_dataset_phot_data,
                                    unmatched_dataset_phot_data,log=log)

    (matched_phot, unmatched_phot) = stage6.build_photometry_array(setup,nimages2,nstars,log)

    assert matched_phot.shape == (nstars,nimages2,ncolumns_matched)
    assert unmatched_phot.shape == (nstars,nimages2,ncolumns_unmatched)

    assert matched_phot[:,0,:].all() == 1.0
    assert matched_phot[:,-1,:].all() == 0.0
    assert unmatched_phot[:,0,:].all() == 1.0
    assert unmatched_phot[:,-1,:].all() == 0.0

    logs.close_log(log)

def test_get_entry_db_indices():

    log = logs.start_stage_log( cwd, 'test_photometry' )

    (conn,params,ref_image_name) = test_phot_db.setup_test_phot_db(log)

    db_pk = stage6.get_entry_db_indices(conn, params, 'lsc1m005-fl15-20170610-0096-e91_cropped.fits', log)

    assert type(db_pk) == type({})
    for key in ['facility', 'filter', 'code', 'refimage', 'image', 'stamp']:
        assert key in db_pk.keys()
        assert type(db_pk[key]) == np.int64

    logs.close_log(log)


def test_store_stamp_photometry_to_array():

    log = logs.start_stage_log( cwd, 'test_photometry' )

    (conn,params,ref_image_name) = test_phot_db.setup_test_phot_db(log)

    meta = metadata.MetaData()
    meta.load_all_metadata(setup.red_dir, 'pyDANDIA_metadata.fits')

    nstars = 100
    nimages = 50
    matched_dataset_phot_data = np.zeros((nstars,nimages,ncolumns_matched))
    unmatched_dataset_phot_data = np.zeros((nstars,nimages,ncolumns_unmatched))

    phot_table = np.ones((nstars,ncolumns_matched))

    matched_stars = match_utils.StarMatchIndex()
    for j in range(0,nstars,1):
        star = {'cat1_index': j,
                'cat1_ra': 260.0, 'cat1_dec': -19.0, 'cat1_x': j, 'cat1_y': j,
                'cat2_index': j,
                'cat2_ra': 260.0, 'cat2_dec': -19.0, 'cat2_x': j, 'cat2_y': j,
                'separation': 0.0}
        matched_stars.add_match(star)

    new_image = 'lsc1m005-fl15-20170610-0096-e91_cropped.fits'

    stage6.store_stamp_photometry_to_array(conn, params, meta,
                                        matched_photometry_data, unmatched_photometry_data,
                                        phot_table, matched_stars,
                                        new_image, log)

    logs.close_log(log)

if __name__ == '__main__':

    #test_build_photometry_array()
    test_get_entry_db_indices()
