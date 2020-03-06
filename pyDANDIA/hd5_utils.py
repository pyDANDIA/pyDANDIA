import os
import h5py
import numpy as np

def write_phot_hd5(setup, matched_dataset_phot_data,
                            unmatched_dataset_phot_data, log=None):
    """Function to output a dataset photometry table to an HD5 file"""

    output_path = os.path.join(setup.red_dir,'photometry.hdf5')

    with h5py.File(output_path, "w") as f:
        dset = f.create_dataset('matched_dataset_photometry',
                                    matched_dataset_phot_data.shape,
                                    dtype='float64',
                                    data=matched_dataset_phot_data)
        dset2 = f.create_dataset('unmatched_dataset_photometry',
                                    unmatched_dataset_phot_data.shape,
                                    dtype='float64',
                                    data=unmatched_dataset_phot_data)
    f.close()

    if log:
        log.info('Output photometry dataset for '+str(setup.red_dir)+\
                ' with '+repr(matched_dataset_phot_data.shape)+\
                ' datapoints matched to field reference catalogue and '+\
                repr(unmatched_dataset_phot_data.shape)+' unmatched datapoints')

def read_phot_hd5(setup,log=None):
    """Function to read an existing dataset photometry table in HD5 format
    Function returns two zero-length arrays if none is available"""

    input_path = os.path.join(setup.red_dir,'photometry.hdf5')

    if os.path.isfile(input_path):
        f = h5py.File(input_path, "r")
        dset1 = f['matched_dataset_photometry']
        dset2 = f['unmatched_dataset_photometry']

        if log:
            log.info('Loaded photometry data with '+repr(dset1.shape)+\
            ' datapoints matched to field reference catalogue and '+\
            repr(dset2.shape)+' unmatched datapoints')

        return dset1, dset2

    else:
        if log:
            log.info('No existing photometry available to load')

        return np.array([]), np.array([])
