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

def read_phot_hd5(setup,log=None, filename=None, return_type='hdf5'):
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

        if return_type == 'hdf5':
            return dset
        else:
            return np.array(dset[:])
    else:
        if log:
            log.info('No existing photometry available to load')

        return np.array([])


def read_phot_from_hd5_file(file_path, return_type='hdf5'):
    """Function to read an existing dataset photometry table in HD5 format
    Function returns two zero-length arrays if none is available"""

    if not os.path.isfile(file_path):
        raise IOError('Cannot find input photometry file '+file_path)

    f = h5py.File(file_path, "r")
    dset = f['dataset_photometry']

    if return_type == 'hdf5':
        return dset
    else:
        return np.array(dset[:])

def read_star_from_hd5_file(file_path, quad_idx):
    """Function to read an existing dataset photometry table in HD5 format
    Function returns two zero-length arrays if none is available"""

    dset = read_phot_from_hd5_file(file_path, return_type='hdf5')

    return np.array(dset[quad_idx,:,:])

def load_four_quadrant_photometry(red_dir, file_rootname, verbose=False):
    """Function to read the timeseries photometry from all four quadrants"""

    for q in range(1,5,1):
        file_path = os.path.join(red_dir, file_rootname+'_quad'+str(q)+'_photometry.hdf5')
        quad_data = read_phot_from_hd5_file(file_path, return_type='array')
        if q == 1:
            phot_data = quad_data
        else:
            phot_data = np.concatenate((phot_data, quad_data))
        if verbose: print('Read in photometry for quadrant '+str(q))

    if verbose: print('Completed read of timeseries photometry: '+repr(phot_data.shape))

    return phot_data

def mask_phot_array(phot_data, col, err_col, qc_col=None):
    """Function to create a Numpy masked array based on the results of selecting
    valid photometric entries from a standard-format photometry array."""

    # Select valid data.  Invalid photometry measurements are usually set to
    # -99.0
    selection = np.logical_and(phot_data[:,:,col] > 0.0,
                                phot_data[:,:,err_col] > 0.0)
    if qc_col != None:
        selection = np.logical_and(phot_data[:,:,qc_col] == 0.0, selection)

    mask = np.invert(selection)

    expand_mask = np.empty((mask.shape[0], mask.shape[1], phot_data.shape[2]))
    for col in range(0,expand_mask.shape[2],1):
        expand_mask[:,:,col] = mask

    phot_data = np.ma.masked_array(phot_data, mask=expand_mask)

    return phot_data

def unmask_phot_array(phot_data):
    """Function to unmask a masked photometry array.  Convienence wrapper
    for np.ma function to match the syntax used for the masking function"""
    return np.ma.getdata(phot_data)

def write_normalizations_hd5(red_dir, file_prefix, normalizations):
    """Function to output a per-star, per-dataset normalization coefficients
     tables to an HD5 file.

     The structure of the tables output have the columns:
     field_id, delta_mag_<dset1>, delta_mag_error_<dset1>, delta_mag_<dset2>, ...
     where the datasets are listed in the same order as the datasets table
     in the CrossMatchTable.
     """

    output_path = os.path.join(red_dir,
                                file_prefix+'_star_dataset_normalizations.hdf5')

    column_names = []
    with h5py.File(output_path, "w") as f:
        for dset_code, table in normalizations.items():
            data = np.zeros((len(table),len(table.colnames)))
            for c,cname in enumerate(table.colnames):
                data[:,c] = table[cname]
            if len(column_names) == 0:
                column_names = table.colnames
            dset = f.create_dataset(dset_code,
                                    data.shape,
                                    dtype='float64',
                                    data=data)
    f.close()
