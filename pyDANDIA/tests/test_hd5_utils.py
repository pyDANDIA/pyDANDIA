import numpy as np
import os
import sys
cwd = os.getcwd()
sys.path.append(os.path.join(cwd,'../'))
import hd5_utils
import pipeline_setup
import logs
from astropy.table import Table, Column

TEST_DIR = os.path.join(cwd,'data','proc',
                        'ROME-FIELD-0002_lsc-doma-1m0-05-fl15_ip')

def test_write_phot_hd5():

    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DIR})

    test_output_file = os.path.join(setup.red_dir,'photometry.hdf5')

    if os.path.isfile(test_output_file):
        os.remove(test_output_file)

    log = logs.start_stage_log( cwd, 'test_photometry' )

    matched_dataset_phot_data = np.ones((200000,300,17))
    #unmatched_dataset_phot_data = np.ones((200000,300,17))

    hd5_utils.write_phot_hd5(setup,matched_dataset_phot_data,log=log)

    assert os.path.isfile(test_output_file)

    logs.close_log(log)

def test_read_phot_hd5():

    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DIR})

    log = logs.start_stage_log( cwd, 'test_photometry' )

    test_dataset_shape = (200000,300,17)
    dataset_phot_data = np.ones(test_dataset_shape)

    hd5_utils.write_phot_hd5(setup,dataset_phot_data,log=log)

    dataset = hd5_utils.read_phot_hd5(setup)

    assert(dataset.shape == test_dataset_shape)

    logs.close_log(log)

def test_read_quadrants():

    red_dir = input('Please enter the path to the test data directory: ')
    file_rootname = input('Please enter the prefix of the photometry HDF files: ')
    phot_data = hd5_utils.load_four_quadrant_photometry(red_dir, file_rootname,
                                                        verbose=True)

    assert(phot_data.shape[0] > 100000)
    print('Read in photometry data with array shape: '+repr(phot_data.shape))

def test_mask_phot_array():

    # Create a test photometry array
    phot_data = np.ones((200,30,17))
    mag_col = 7
    mag_err_col = 8
    qc_col = 16
    phot_data[:,:,qc_col].fill(0.0)

    # Mask out all images for a particular star
    quad_idx = 100
    phot_data[quad_idx,:,mag_col] = -9999.9
    phot_data[quad_idx,:,mag_err_col] = -9999.9
    phot_data[quad_idx,:,qc_col].fill(2.0)

    # Call the masking function
    phot_data = hd5_utils.mask_phot_array(phot_data, mag_col, mag_err_col,
                                          qc_col=qc_col)

    assert(type(phot_data) == type(np.ma.masked_array([], mask=[])))
    mask = np.ma.getmask(phot_data)
    assert((mask[quad_idx,:,mag_col] == True).all())
    assert((mask[quad_idx+1,:,mag_col] == False).all())

def test_write_norm_hd5():

    test_output_file = os.path.join(TEST_DIR,'star_dataset_normalizations.hdf5')

    if os.path.isfile(test_output_file):
        os.remove(test_output_file)

    # Simulate a dataset to be output.  For each primary reference dataset,
    # the table consists of field_id, delta_mag_dset1, delta_mag_error_dset1, etc
    pri_dset_list = ['lsc-doma', 'cpt-doma', 'coj-doma']
    dset_list = ['lsc-doma_ip', 'lsc-doma_rp', 'lsc-doma_gp',
                 'cpt-domb_ip', 'cpt-domb_rp', 'cpt-domb_gp']
    nstars = 100
    ndatasets = 4
    normalizations = {}
    for pdset in pri_dset_list:
        column_list = [Column(name='field_id', data=np.arange(1,nstars+1,1))]
        for dset in dset_list:
            column_list.append(Column(name='delta_mag+'+dset, data=np.ones(nstars)))
            column_list.append(Column(name='delta_mag_error+'+dset, data=np.ones(nstars)))
        normalizations[pdset] = Table(column_list)

    hd5_utils.write_normalizations_hd5(TEST_DIR, normalizations)

    assert os.path.isfile(test_output_file)


if __name__ == '__main__':
    #test_write_phot_hd5()
    #test_read_phot_hd5()
    #test_read_quadrants()
    #test_mask_phot_array()
    test_write_norm_hd5()
