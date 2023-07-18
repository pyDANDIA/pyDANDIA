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
import pdb

def combine_photometry_from_all_datasets():

    params = get_args()

    log = logs.start_stage_log( params['log_dir'], 'field_photometry' )

    params = crossmatch_datasets.parse_dataset_list(params,log)

    xmatch = crossmatch.CrossMatchTable()
    xmatch.load(params['crossmatch_file'],log=log)
    xmatch.create_images_table()

    for dataset in xmatch.datasets:
        log.info('Populating image and star tables with data from '+dataset['dataset_code'])
        setup = pipeline_setup.PipelineSetup()
        setup.red_dir = dataset['dataset_red_dir']
        dataset_metadata = metadata.MetaData()
        dataset_metadata.load_all_metadata(setup.red_dir, 'pyDANDIA_metadata.fits')

        (xmatch,dataset_image_index) = populate_images_table(dataset,dataset_metadata, xmatch, log)
        (xmatch, field_star_index, dataset_stars_index) = populate_stars_table(dataset,xmatch,dataset_metadata,log)
        xmatch = populate_stamps_table(xmatch, dataset['dataset_code'], dataset_metadata, log)

    # These loops are deliberately separated because it makes it easier to
    # initialize the photometry array for the whole field to the correct size
    for q in range(1,5,1):
        log.info('Populating timeseries photometry for quadrant '+str(q))
        quad_photometry = init_quad_field_data_table(xmatch,q,log)
        for dataset in xmatch.datasets:
            log.info('-> Including data timeseries photometry with data from '+dataset['dataset_code'])
            setup = pipeline_setup.PipelineSetup()
            setup.red_dir = dataset['dataset_red_dir']
            dataset_metadata = metadata.MetaData()
            dataset_metadata.load_all_metadata(setup.red_dir, 'pyDANDIA_metadata.fits')
            phot_data = hd5_utils.read_phot_hd5(setup, log=log, return_type='array')
            if len(phot_data) > 0:
                (quad_star_index, dataset_stars_index) = get_dataset_quad_star_indices(dataset, xmatch, q)
                dataset_image_index = get_dataset_image_index(dataset, xmatch)
                (xmatch,quad_photometry) = populate_quad_photometry_array(quad_star_index, dataset_stars_index,
                                                dataset_image_index, quad_photometry, phot_data, xmatch, log,
                                                dataset_metadata)

        # Output tables to quadrant HDF5 file
        output_quad_photometry(params, xmatch, photometry, q, log)

    # Update the xmatch table
    xmatch.save(params['crossmatch_file'])

    log.info('Field photometry: complete')

    logs.close_log(log)

def output_quad_photometry(params, xmatch, quad_photometry, q, log):

    log.info('Outputting quadrant '+str(q)+' photometry, array shape: '
                +repr(photometry.shape))

    setup = pipeline_setup.PipelineSetup()
    setup.red_dir = path.join(path.dirname(params['crossmatch_file']))
    filename = params['field_name']+'_quad'+str(q)+'_photometry.hdf5'
    hd5_utils.write_phot_hd5(setup, quad_phototometry, log=log,
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

def update_array_col_index(index3d, new_col):
    """Function to substitute a new array column number into the 3rd index in a
    3D tuple"""

    new_index = np.zeros(len(index3d[2]), dtype='int')
    new_index.fill(new_col)
    new_tuple = (index3d[0], index3d[1], new_index)

    return new_tuple

def get_field_photometry_columns(phot_columns='instrumental'):
    """Function to return the column indices of the magnitude and magnitude uncertainties
    in the photometry array for a single dataset.  Options are:
    instrumental
    calibrated
    corrected
    """

    if phot_columns == 'instrumental':
        mag_col = 1
        merr_col = 2
    elif phot_columns == 'calibrated':
        mag_col = 3
        merr_col = 4
    elif phot_columns == 'corrected':
        mag_col = 5
        merr_col = 6
    elif phot_columns == 'normalized':
        mag_col = 7
        merr_col = 8

    return mag_col, merr_col

def populate_quad_photometry_array(quad_star_index, dataset_star_index,
                                dataset_image_index, quad_photometry, dataset_photometry,
                                xmatch, log, dataset_metadata):

    # Columns: hjd, instrumental_mag, instrumental_mag_err,
    # calibrated_mag, calibrated_mag_err, corrected_mag, corrected_mag_err,
    # normalized_mag, normalized_mag_err,
    # phot_scale_factor, phot_scale_factor_err, stamp_index,
    # sub_image_sky_bkgd, sub_image_sky_bkgd_err,
    # residual_x, residual_y
    # qc_flag
    # Build corresponding 3D array index locators for the whole field photometry
    # array and the dataset photometry array, and use them to transfer the
    # timestamp data
    log.info('Dimensions of field quadrant photometry array: '+repr(quad_photometry.shape))

    ndata = len(dataset_photometry[0,:,0])
    log.info('N datapoints in image data: '+str(ndata)+', len dataset_image_index: '+str(len(dataset_image_index)))
    log.info('Len quad_star_index: '+str(len(quad_star_index))+' len dataset_star_index: '+str(len(dataset_star_index)))

    # Transfering the HJD information for all images in this dataset to the
    # combined photometry array. Note that the phot_index and data_index
    # contain the full frame field index,
    phot_index = build_array_index([quad_star_index, dataset_image_index,[0]])
    data_index = build_array_index([dataset_star_index, np.arange(0,ndata,1), [9]])
    quad_photometry[phot_index] = dataset_photometry[data_index]

    # Update the array indices to refer to the photometry columns, and
    # transfer those data as well.  Tuples listed below give the array indices
    # of the following columns in the (main, dataset) photometry arrays.
    # inst_mag, inst_mag_err, cal_mag, cal_mag_err, corr_mag, corr_mag_err,
    # norm_mag, norm_mag_err, ps, ps_err, bkgd, bkgd_err, res_x, res_y, qc_flag
    column_index = [(1,11),(2,12),(3,13),(4,14),(5,23),(6,24),(7,26),(8,27),\
                    (9,19),(10,20),(12,21),(13,22),(14,7),(15,8),(16,25)]
    for column in column_index:
        phot_index = update_array_col_index(phot_index, column[0])
        data_index = update_array_col_index(data_index, column[1])
        quad_photometry[phot_index] = dataset_photometry[data_index]

    # Populate the stamp index:
    list_of_stamps = dataset_metadata.stamps[1]['PIXEL_INDEX'].tolist()
    for stamp in list_of_stamps:
        stamp_row = np.where(dataset_metadata.stamps[1]['PIXEL_INDEX'] == stamp)[0][0]
        xmin = int(dataset_metadata.stamps[1][stamp_row]['X_MIN'])
        xmax = int(dataset_metadata.stamps[1][stamp_row]['X_MAX'])
        ymin = int(dataset_metadata.stamps[1][stamp_row]['Y_MIN'])
        ymax = int(dataset_metadata.stamps[1][stamp_row]['Y_MAX'])

        stamp_stars = (dataset_metadata.star_catalog[1]['x'] < xmax) & (dataset_metadata.star_catalog[1]['x'] > xmin) & \
                     (dataset_metadata.star_catalog[1]['y'] < ymax) & (dataset_metadata.star_catalog[1]['y'] > ymin)

        stamp_star_idx = dataset_metadata.star_catalog[1]['index'][stamp_stars] - 1

        quad_stamp_star_idx = []
        for dataset_idx in stamp_star_idx:
            idx = np.where(dataset_star_index == dataset_idx)[0]
            if len(idx) > 0:
                quad_stamp_star_idx.append(quad_star_index[idx[0]])

        phot_index = build_array_index([quad_stamp_star_idx, dataset_image_index,[0]])
        phot_index = update_array_col_index(phot_index, 11)

        quad_photometry[phot_index] = stamp

    # Also update the images table with the timestamp data:
    for i in range(0,ndata,1):
        jdx = np.where(dataset_photometry[:,i,9] > 0)[0]
        if len(jdx) > 0:
            xmatch.images['hjd'][dataset_image_index[i]] = dataset_photometry[jdx[0],i,9]

    log.info('-> Populated photometry array with dataset timeseries photometry')
    return xmatch, quad_photometry

def get_dataset_star_indices(dataset, xmatch):

    field_array_idx = np.where(xmatch.field_index[dataset['dataset_code']+'_index'] > 0)[0]
    dataset_array_idx = xmatch.field_index[dataset['dataset_code']+'_index'][field_array_idx] - 1

    return field_array_idx, dataset_array_idx

def get_dataset_quad_star_indices(dataset, xmatch, qid):

    # Identify the field indices for all stars with measurements from this dataset
    field_array_idx = np.where(xmatch.field_index[dataset['dataset_code']+'_index'] > 0)[0]

    # Identify the field indices for all stars in this quadrant
    quad_stars = np.where(xmatch.field_index['quadrant'] == qid)[0]

    # The intersection of these indices gives the field indices for all
    # quadrant stars with valid measurements for this dataset
    idx = list(set(field_array_idx).intersection(set(quad_stars)))

    # We want the quadrant array indices for all the selected stars, so we
    # convert the field index to the quadrant index
    quad_array_idx = xmatch.field_index['quadrant_id'][idx] - 1

    # Now extract the corresponding entries for the same stars in the
    # dataset's array
    dataset_array_idx = xmatch.field_index[dataset['dataset_code']+'_index'][idx] - 1

    return quad_array_idx, dataset_array_idx

def get_dataset_image_index(dataset, xmatch):

    dataset_image_idx = np.where(xmatch.images['dataset_code'] == dataset['dataset_code'])[0]

    return dataset_image_idx

def check_for_reference_dataset(dataset_code):
    dataset_id = '_'.join(dataset_code.split('_')[1].split('-')[0:2])
    if dataset_id in ['lsc_doma', 'cpt_doma', 'coj_doma']:
        return True
    else:
        return False

def populate_stars_table(dataset,xmatch,dataset_metadata,log):

    dataset_id = '_'.join(dataset['dataset_code'].split('_')[1].split('-')[0:2])
    filter_name = parse_sloan_filter_ids(dataset['dataset_filter'])

    mag_column = 'cal_'+filter_name+'_mag_'+dataset_id
    mag_error_column = 'cal_'+filter_name+'_magerr_'+dataset_id

    (field_array_idx,dataset_array_idx) = get_dataset_star_indices(dataset,xmatch)

    xmatch.stars[mag_column][field_array_idx] = dataset_metadata.star_catalog[1]['cal_ref_mag'][dataset_array_idx]
    xmatch.stars[mag_error_column][field_array_idx] = dataset_metadata.star_catalog[1]['cal_ref_mag_error'][dataset_array_idx]

    log.info('-> Populated stars table with '+dataset_id+' reference image photometry for filter '+filter_name)

    return xmatch, field_array_idx, dataset_array_idx

def populate_images_table(dataset, dataset_metadata, xmatch, log):
    """Function to populate the table of image properties from the corresponding
    data in the datasets metadata tables.  Table columns are:

    Index filename dataset_code filter hjd datetime exposure RA Dec moon_ang_separation moon_fraction airmass sigma_x sigma_y \
    sky median_sky fwhm corr_xy nstars frac_sat_pix symmetry use_phot use_ref shift_x shift_y pscale pscale_err \
    var_per_pix_diff n_unmasked skew_diff kurtosis_diff"""

    iimage = len(xmatch.images)
    image_index = []
    for i,image in enumerate(dataset_metadata.headers_summary[1]):
        row = [iimage+i, image['IMAGES'], dataset['dataset_code'], image['FILTKEY'], 0.0, \
                                image['DATEKEY'], image['EXPKEY'], image['RAKEY'], image['DECKEY'], \
                                image['MOONDKEY'], image['MOONFKEY'], image['AIRMASS'] ] + \
                                [0.0]*6 + [0] +[0.0]*2 + [0,0] + [0.0]*24 + [0]
        xmatch.images.add_row(row)
        image_index.append(iimage+i)
    log.info('-> Populated images table with data from FITS image headers')

    for image in dataset_metadata.reduction_status[1]:
        log.info('-> Populating image QC flag from reduction status '+image['IMAGES'])
        i = np.where(xmatch.images['filename'] == image['IMAGES'])
        log.info('--> Entry '+str(i)+' in images table')
        qc_flag = 0
        for k in range(0,7,1):
            log.info('--> stage '+str(k)+' '+str(image['STAGE_'+str(k)])+' '+repr(type(image['STAGE_'+str(k)])))
            if int(image['STAGE_'+str(k)]) == -1:
                qc_flag = -1
        xmatch.images['qc_flag'][i] = qc_flag
        log.info('--> xmatch.images qc flag entry: '+str(xmatch.images['qc_flag'][i]))
    log.info('-> Populated images table with reduction status QC data')

    images_stats_keys = ['sigma_x', 'sigma_y', 'sky', 'median_sky', 'fwhm', \
                         'corr_xy', 'nstars', 'frac_sat_pix', 'symmetry',  \
                         'use_phot', 'use_ref', 'shift_x', 'shift_y', \
                         'pscale', 'pscale_err', 'var_per_pix_diff', 'n_unmasked',\
                         'skew_diff', 'kurtosis_diff']
    red_dir = dataset_metadata.data_architecture[1]['OUTPUT_DIRECTORY'][0]
    for image in dataset_metadata.images_stats[1]:
        log.info('-> Populating image statistics and warp matrix for '+image['IM_NAME'])
        i = np.where(xmatch.images['filename'] == image['IM_NAME'])
        for key in images_stats_keys:
            xmatch.images[key][i] = image[key.upper()]

        matrix_file = path.join(red_dir, 'resampled', image['IM_NAME'], 'warp_matrice_image.npy')
        if path.isfile(matrix_file):
            matrix = np.load(matrix_file)
            log.info('--> Loaded matrix: '+repr(matrix))
            transformation = matrix.ravel()
        else:
            transformation = np.zeros(9)
        for j in range(0,len(transformation),1):
            xmatch.images['warp_matrix_'+str(j)][i] = transformation[j]
            log.info('--> Storing matrix element '+str(j)+' for image '+repr(i)+' '+str(xmatch.images[i]['warp_matrix_'+str(j)]))

    log.info('-> Populated image table')

    return xmatch, image_index

def populate_stamps_table(xmatch, dataset_code, dataset_metadata, log):

    red_dir = dataset_metadata.data_architecture[1]['OUTPUT_DIRECTORY'][0]
    dataset_stamps_idx = np.where(xmatch.stamps['dataset_code'] == dataset_code)[0]

    for i in dataset_stamps_idx:
        image_file = xmatch.stamps['filename'][i]
        sid = xmatch.stamps['stamp_id'][i]
        matrix_file = path.join(red_dir, 'resampled', image_file, 'warp_matrice_stamp_'+str(sid)+'.npy')
        if path.isfile(matrix_file):
            matrix = np.load(matrix_file)
            transformation = matrix.ravel()
        else:
            transformation = np.zeros(9)
        for j in range(0,len(transformation),1):
            xmatch.stamps[i]['warp_matrix_'+str(j)] = transformation[j]

    log.info('-> Populated the stamps table with transform data for '+\
            str(len(dataset_stamps_idx))+' stamps for dataset '+dataset_code)

    return xmatch

def init_quad_field_data_table(xmatch,q,log):
    # Photometry data array is initialized as a list because this is a
    # faster way to add rows.  Structure is:
    # [Nstars, Nimages, Ncolumns]
    # Columns: hjd, instrumental_mag, instrumental_mag_err,
    # calibrated_mag, calibrated_mag_err, corrected_mag, corrected_mag_err,
    # normalized_mag, normalized_mag_err,
    # phot_scale_factor, phot_scale_factor_err, stamp_index,
    # sub_image_sky_bkgd, sub_image_sky_bkgd_err,
    # residual_x, residual_y
    # qc_flag
    quad_stars = np.where(xmatch.field_index['quadrant'] == q)[0]
    photometry = np.zeros( (len(quad_stars), len(xmatch.images), 17) )
    log.info('Initialized timeseries photometry array for quadrant '+str(q))

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
