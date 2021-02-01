import os
import h5py
import numpy as np

def write_phot_hd5(setup, dataset_phot_data, log=None,
                    filename=None):
    """Function to output a dataset photometry table to an HD5 file"""

    if not filename:
        output_path = os.path.join(setup.red_dir,'photometry.hdf5')
    else:
        output_path = os.path.join(setup.red_dir,filename)

    with h5py.File(output_path, "w") as f:
        dset = f.create_dataset('dataset_photometry',
                                    dataset_phot_data.shape,
                                    dtype='float64',
                                    data=dataset_phot_data)
    f.close()

    if log:
        log.info('Output photometry dataset for '+str(setup.red_dir)+\
                ' with '+repr(dataset_phot_data.shape)+\
                ' datapoints')

def read_phot_hd5(setup,log=None, filename=None):
    """Function to read an existing dataset photometry table in HD5 format
    Function returns two zero-length arrays if none is available"""

    if not filename:
        input_path = os.path.join(setup.red_dir,'photometry.hdf5')
    else:
        input_path = os.path.join(setup.red_dir,filename)
        
    if os.path.isfile(input_path):
        f = h5py.File(input_path, "r")
        dset = f['dataset_photometry']

        if log:
            log.info('Loaded photometry data with '+repr(dset.shape)+\
            ' datapoints')

        return dset

    else:
        if log:
            log.info('No existing photometry available to load')

        return np.array([])
