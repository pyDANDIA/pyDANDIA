import os
import h5py

def write_phot_hd5(setup,dataset_phot_data,log=None):
    """Function to output a dataset photometry table to an HD5 file"""

    output_path = os.path.join(setup.red_dir,'photometry.hdf5')

    with h5py.File(output_path, "w") as f:
        dset = f.create_dataset('dataset_phot_data', dataset_phot_data.shape,
                                                    dtype='float64')
    f.close()

    if log:
        log.info('Output photometry dataset for '+str(setup.red_dir)+\
                ' with '+repr(dataset_phot_data.shape)+' datapoints')

def read_phot_hd5(setup,log=None):
    """Function to read an existing dataset photometry table in HD5 format
    Function returns a zero-length array if none is available"""

    input_path = os.path.join(setup.red_dir,'photometry.hdf5')

    if os.path.isfile(input_path):
        f = h5py.File(input_path, "r")
        dset = f['dataset_phot_data']

        if log:
            log.info('Loaded photometry data with '+\
                        repr(dataset_phot_data.shape)+' datapoints')

        return dset

    else:
        if log:
            log.info('No existing photometry available to load')
            
        return np.array([])
