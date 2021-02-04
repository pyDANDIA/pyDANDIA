from os import path
from sys import argv
import numpy as np
from pyDANDIA import logs
from pyDANDIA import metadata
from pyDANDIA import crossmatch
from pyDANDIA import crossmatch_datasets
from pyDANDIA import hd5_utils
from pyDANDIA import pipeline_setup
from astropy.table import Table, Column

def combine_photometry_from_all_datasets():

    params = get_args()

    log = logs.start_stage_log( params['log_dir'], 'field_photometry' )

    params = crossmatch_datasets.parse_dataset_list(params,log)

    xmatch = crossmatch.CrossMatchTable()
    xmatch.load(params['crossmatch_file'],log=log)


    for dataset in xmatch.datasets:
        log.info('Populating image and star tables with data from '+dataset['dataset_code'])
        setup = pipeline_setup.PipelineSetup()
        setup.red_dir = dataset['dataset_red_dir']
        dataset_metadata = metadata.MetaData()
        dataset_metadata.load_all_metadata(setup.red_dir, 'pyDANDIA_metadata.fits')
        phot_data = hd5_utils.read_phot_hd5(setup, log=log)

        (xmatch,dataset_image_index) = populate_images_table(dataset,dataset_metadata, xmatch, log)
        (xmatch, field_star_index, dataset_stars_index) = populate_stars_table(dataset,xmatch,dataset_metadata,log)

    photometry = init_field_data_table(xmatch,log)
    for dataset in xmatch.datasets:
        log.info('Populating timeseries photometry with data from '+dataset['dataset_code'])
        (xmatch,photometry) = populate_photometry_array(field_star_index, dataset_stars_index,
                                        dataset_image_index, photometry, xmatch, log)

    # Output tables to field HDF5 file in quadrants
    output_field_photometry(params, xmatch, photometry, log)

    log.info('Field photometry: complete')

    logs.close_log(log)

def output_field_photometry(params, xmatch, photometry, log):

    for q in range(1,5,1):
        setup = pipeline_setup.PipelineSetup()
        setup.red_dir = path.join(path.dirname(params['crossmatch_file']))
        filename = params['field_name']+'_quad'+str(q)+'_photometry.hdf5'

        idx = np.where(xmatch.field_index['quadrant'] == q)
        quad_phot_data = photometry[idx,:,:]

        hd5_utils.write_phot_hd5(setup, quad_phot_data, log=log,
                                    filename=filename)

def parse_sloan_filter_ids(filter_name):
    """Function to parse LCO-centric filter names where necessary"""

    new_filter_name = filter_name
    if filter_name == 'gp':
        new_filter_name = 'g'
    elif filter_name == 'rp':
        new_filter_name = 'r'
    elif filter_name == 'ip':
        new_filter_name = 'i'

    return new_filter_name

def build_array_index(array_indices, verbose=False):
    """Function to construct the multi-dimensional array location index from
    arrays indicating the selection to be made in each of the dimensions.

    array_indices  list of np.arrays or list of lists
    """

    # Total number of entries in all of the output selection index elements
    nentries = len(array_indices[0])
    for array in array_indices[1:]:
        nentries *= len(array)

    if verbose:
        print('Nentries = ',nentries,' based on input array indices of length:')
        [print(len(array)) for array in array_indices]

    index = []

    for axis in range(0,len(array_indices),1):
        index.append(np.zeros(nentries, dtype='int'))

        if axis+1 < len(array_indices):
            nrepeat = len(array_indices[axis+1])
            for i in range(axis+2,len(array_indices),1):
                nrepeat *= len(array_indices[i])
                if verbose: print('nrepeat*= ',nrepeat, i, len(array_indices[i]))
            if verbose: print('Axis: ',axis,' nrepeat: ',nrepeat)

        else:
            nrepeat = 1

        istart = 0
        iend = istart + nrepeat
        while iend <= nentries:
            for entry in array_indices[axis]:
                index[axis][istart:iend].fill(entry)
                if verbose: print('Loop: ',istart, iend, entry,' axis: ',axis)
                istart = iend
                iend = istart + nrepeat

    if verbose: print(index)
    return tuple(index)

def populate_photometry_array(field_star_index, dataset_star_index,
                                dataset_image_index, photometry, dataset_photometry,
                                xmatch, log):

    ndata = len(dataset_photometry[0,:,0])
    # hjd, instrumental_mag, instrumental_mag_err, calibrated_mag, calibrated_mag_err, corrected_mag, corrected_mag_err
    phot_index = build_array_index([field_star_index, dataset_image_index,[0]])
    data_index = build_array_index([dataset_star_index, np.arange(0,ndata,1), [9]])
    photometry[phot_index] = dataset_photometry[data_index]

    phot_index = build_array_index([field_star_index, dataset_image_index,[1]])
    data_index = build_array_index([dataset_star_index, np.arange(0,ndata,1), [11]])
    photometry[phot_index] = dataset_photometry[data_index]

    phot_index = build_array_index([field_star_index, dataset_image_index,[2]])
    data_index = build_array_index([dataset_star_index, np.arange(0,ndata,1), [12]])
    photometry[phot_index] = dataset_photometry[data_index]

    phot_index = build_array_index([field_star_index, dataset_image_index,[3]])
    data_index = build_array_index([dataset_star_index, np.arange(0,ndata,1), [13]])
    photometry[phot_index] = dataset_photometry[data_index]

    phot_index = build_array_index([field_star_index, dataset_image_index,[4]])
    data_index = build_array_index([dataset_star_index, np.arange(0,ndata,1), [14]])
    photometry[phot_index] = dataset_photometry[data_index]

    # Also update the images table with the timestamp data:
    xmatch.images['hjd'][dataset_image_index] = dataset_photometry[0,:,9]

    log.info('-> Populated photometry array with dataset timeseries photometry')
    return xmatch, photometry

def populate_stars_table(dataset,xmatch,dataset_metadata,log):

    dataset_id = '_'.join(dataset['dataset_code'].split('_')[1].split('-')[0:2])
    filter_name = parse_sloan_filter_ids(dataset['dataset_filter'])

    if dataset_id in ['lsc_doma', 'cpt_doma', 'coj_doma']:
        mag_column = 'cal_'+filter_name+'_mag_'+dataset_id
        mag_error_column = 'cal_'+filter_name+'_magerr_'+dataset_id

        idx = np.where(xmatch.field_index[dataset['dataset_code']+'_index'] > 0)
        dataset_array_idx = xmatch.field_index[dataset['dataset_code']+'_index'][idx] - 1
        field_array_idx = xmatch.field_index['field_id'][idx] - 1

        xmatch.stars[mag_column][field_array_idx] = dataset_metadata.star_catalog[1]['cal_ref_mag'][dataset_array_idx]
        xmatch.stars[mag_error_column][field_array_idx] = dataset_metadata.star_catalog[1]['cal_ref_mag_error'][dataset_array_idx]

        log.info('-> Populated stars table with dataset reference image photometry')
    else:
        log.info('-> Dataset not used as a reference dataset')

    return xmatch, field_array_idx, dataset_array_idx

def populate_images_table(dataset,dataset_metadata, xmatch, log):

    iimage = len(xmatch.images)
    image_index = []
    for i,image in enumerate(dataset_metadata.headers_summary[1]):
        xmatch.images.add_row([iimage+i, image['IMAGES'], dataset['dataset_code'], image['FILTKEY'], 0.0])
        image_index.append(iimage+i)

    log.info('-> Populated image table')
    return xmatch, image_index

def init_field_data_table(xmatch,log):
    # Photometry data array is initialized as a list because this is a
    # faster way to add rows.  Structure is:
    # [Nimages, Nstars, Ncolumns]
    # Columns: hjd, instrumental_mag, instrumental_mag_err, calibrated_mag, calibrated_mag_err, corrected_mag, corrected_mag_err
    # Note: Last two columns included to allow for likely future expansion;
    # not yet populated
    photometry = np.zeros( (len(xmatch.stars), len(xmatch.images), 7) )
    log.info('Initialized timeseries photometry array')

    return photometry

def get_args():

    params = {}
    if len(argv) == 1:
        params['datasets_file'] = input('Please enter the path to the dataset list: ')
        params['crossmatch_file'] = input('Please enter the path to the field crossmatch table: ')
        params['log_dir'] = input('Please enter the path to the log directory: ')
        params['field_name'] = input('Please enter the field identifier: ')
    else:
        params['datasets_file'] = argv[1]
        params['crossmatch_file'] = argv[2]
        params['log_dir'] = argv[3]
        params['field_name'] = argv[4]

    return params

if __name__ == '__main__':
    combine_photometry_from_all_datasets()
