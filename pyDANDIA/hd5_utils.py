import os
import h5py
import numpy as np

def write_phot_hd5(setup, dataset_phot_data, log=None):
    """Function to output a dataset photometry table to an HD5 file"""

    output_path = os.path.join(setup.red_dir,'photometry.hdf5')

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

def read_phot_hd5(setup,log=None):
    """Function to read an existing dataset photometry table in HD5 format
    Function returns two zero-length arrays if none is available"""

    input_path = os.path.join(setup.red_dir,'photometry.hdf5')

    if log!=None:
        log.info('Load timeseries photometry from '+input_path)

    if os.path.isfile(input_path):
        f = h5py.File(input_path, "r")
        dset = f['dataset_photometry']

        if log:
            log.info('Loaded photometry data with '+repr(dset.shape)+\
            ' datapoints')

        return dset

    else:
        if log:
            log.info('No existing photometry available to load found at '+input_path)

        return np.array([])

def load_dataset_timeseries_photometry(setup,log,ncolumns):
    """Function to load data from an existing photometry HDF5 file into an array
    with number of columns = ncolumns.  If ncolumns is greater than the current
    number of columns of data, the existing data will be transfered into columns
    0:n_existing_columns of the array, leaving subsequent columns set to zero.
    This is a convenience function when a code needs to add columns of output
    to the photometry data."""

    existing_phot = read_phot_hd5(setup,log=log)

    if len(existing_phot) == 0:
        raise IOError('No existing photometry found for '+setup.red_dir)

    nstars = existing_phot.shape[0]
    nimages = existing_phot.shape[1]
    ncolumns_old = existing_phot.shape[2]

    # Add two columns for the cross-calibrated photometry:
    photometry_data = np.zeros((nstars,nimages,ncolumns))

    # If available, transfer the existing photometry into the data arrays
    if len(existing_phot) > 0:
        photometry_data[0:nstars, 0:nimages, 0:ncolumns_old] = existing_phot

    log.info('Completed build of the photometry array')

    return photometry_data
