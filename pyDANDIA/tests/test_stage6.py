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
import metadata
import match_utils
from astropy.table import Table, Column

TEST_DIR = os.path.join(cwd,'data','proc',
                        'ROME-FIELD-0002_lsc-doma-1m0-05-fl15_ip')
DB_FILE = 'test.db'
db_file_path = os.path.join(TEST_DIR, '..', DB_FILE)

ncolumns = 23

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
    matched_phot = stage6.build_photometry_array(setup,nimages1,nstars,log)

    assert matched_phot.shape == (nstars,nimages1,ncolumns)

    print('Testing array building with pre-existing photometry')
    matched_dataset_phot_data = np.ones((nstars,nimages1,ncolumns))

    hd5_utils.write_phot_hd5(setup,matched_dataset_phot_data, log=log)

    matched_phot = stage6.build_photometry_array(setup,nimages2,nstars,log)

    assert matched_phot.shape == (nstars,nimages2,ncolumns)

    assert matched_phot[:,0,:].all() == 1.0
    assert matched_phot[:,-1,:].all() == 0.0

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

    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DIR})

    log = logs.start_stage_log( cwd, 'test_photometry' )

    (conn,params,ref_image_name) = test_phot_db.setup_test_phot_db(log)

    meta = metadata.MetaData()
    meta.load_all_metadata(setup.red_dir, 'pyDANDIA_metadata.fits')

    nstars = 200000
    nimages = 300
    matched_phot_data = np.zeros((nstars,nimages,ncolumns))
    unmatched_phot_data = np.zeros((nstars,nimages,ncolumns-1))

    table_data = [Column(name='star_id', data=np.arange(1.0,nstars+1.0,1.0)),
                  Column(name='diff_flux', data=np.ones((nstars))),
                  Column(name='diff_flux_err', data=np.ones((nstars))),
                  Column(name='magnitude', data=np.ones((nstars))),
                  Column(name='magnitude_err', data=np.ones((nstars))),
                  Column(name='cal_magnitude', data=np.ones((nstars))),
                  Column(name='cal_magnitude_err', data=np.ones((nstars))),
                  Column(name='flux', data=np.ones((nstars))),
                  Column(name='flux_err', data=np.ones((nstars))),
                  Column(name='cal_flux', data=np.ones((nstars))),
                  Column(name='cal_flux_err', data=np.ones((nstars))),
                  Column(name='phot_scale_factor', data=np.ones((nstars))),
                  Column(name='phot_scale_factor_err', data=np.ones((nstars))),
                  Column(name='local_background', data=np.ones((nstars))),
                  Column(name='local_background_err', data=np.ones((nstars))),
                  Column(name='residual_x', data=np.ones((nstars))),
                  Column(name='residual_y', data=np.ones((nstars))),
                  Column(name='radius', data=np.ones((nstars)))]

    phot_table = Table(data=table_data)

    matched_stars = match_utils.StarMatchIndex()
    for j in range(0,nstars,1):
        star = {'cat1_index': j,
                'cat1_ra': 260.0, 'cat1_dec': -19.0, 'cat1_x': j, 'cat1_y': j,
                'cat2_index': j,
                'cat2_ra': 260.0, 'cat2_dec': -19.0, 'cat2_x': j, 'cat2_y': j,
                'separation': 0.0}
        matched_stars.add_match(star)

    new_image = 'lsc1m005-fl15-20170610-0096-e91_cropped.fits'
    image_dataset_id = np.where(new_image == meta.headers_summary[1]['IMAGES'].data)[0][0]

    log.info('Starting test of original array storage function')
    (matched_phot_data, unmatched_phot_data) = stage6.store_stamp_photometry_to_array_starloop(conn, params, meta,
                                        matched_phot_data, unmatched_phot_data,
                                        phot_table, matched_stars,
                                        new_image, log, verbose=False)
    log.info('Completed test of original function')

    for col in range(10,22,1):
        assert matched_phot_data[0:-1,image_dataset_id-1,col].all() == 1.0
    assert unmatched_phot_data.all() == 0.0

    log.info('Starting test of new array storage function')
    phot_data = stage6.store_stamp_photometry_to_array(conn, params, meta,
                                        matched_phot_data,
                                        phot_table, matched_stars,
                                        new_image, log, verbose=True)
    log.info('Completed test of new function')

    logs.close_log(log)

if __name__ == '__main__':

    #test_build_photometry_array()
    #test_get_entry_db_indices()
    test_store_stamp_photometry_to_array()
