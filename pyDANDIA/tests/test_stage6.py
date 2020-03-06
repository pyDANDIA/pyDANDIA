import numpy as np
import os
import sys
cwd = os.getcwd()
sys.path.append(os.path.join(cwd,'../'))
import stage6
import pipeline_setup
import logs
import hd5_utils

TEST_DIR = os.path.join(cwd,'data','proc',
                        'ROME-FIELD-0002_lsc-doma-1m0-05-fl15_ip')

def test_build_photometry_array():

    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DIR})

    log = logs.start_stage_log( cwd, 'test_photometry' )

    test_output_file = os.path.join(setup.red_dir,'photometry.hdf5')
    nstars = 1000
    nimages1 = 150
    nimages2 = 300
    ncolumns_matched = 23
    ncolumns_unmatched = 22

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


if __name__ == '__main__':

    test_build_photometry_array()
