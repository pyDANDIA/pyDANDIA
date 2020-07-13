######################################################################
#
# stage5.py - Fifth stage of the pipeline. Calculate kernel solution
# and optionally subtract

#
# dependencies:
#      numpy 1.8+
#      astropy 1.0+
#      scipy 1.0+
######################################################################
import os, sys
import copy
import numpy as np
from astropy.io import fits
from scipy.signal import convolve2d
from scipy.stats import skew, kurtosis
from scipy.ndimage.filters import gaussian_filter
from pyDANDIA.read_images_stage5 import open_reference, open_images, open_data_image,mask_the_image
from pyDANDIA.stage0 import open_an_image
from pyDANDIA.subtract_subimages import subtract_images, subtract_subimage
from multiprocessing import Pool
import multiprocessing as mp
import scipy.optimize as so

from astropy.stats import sigma_clipped_stats
from photutils import datasets

from pyDANDIA import config_utils
from pyDANDIA import metadata
from pyDANDIA import logs
from pyDANDIA import psf
from pyDANDIA import stage4
from pyDANDIA import image_handling

import matplotlib.pyplot as plt


def run_stage5(setup, **kwargs):
    """Main driver function to run stage 5: kernel_solution
    This stage finds the kernel solution and (optionally) subtracts the model
    image
    :param object setup : an instance of the ReductionSetup class. See reduction_control.py

    :return: [status, report, reduction_metadata], stage5 status, report,
     metadata file
    :rtype: array_like
    """

    stage5_version = 'stage5 v0.1'

    log = logs.start_stage_log(setup.red_dir, 'stage5', version=stage5_version)
    log.info('Setup:\n' + setup.summary() + '\n')
    try:
        from umatrix_routine import umatrix_construction, umatrix_bvector_construction, bvector_construction
        from umatrix_routine import umatrix_construction_nobkg, bvector_construction_nobkg
        #from umatrix_routine import umatrix_construction_clean, bvector_construction_clean

    except ImportError:
        log.info('Uncompiled cython code, please run setup.py: e.g.\n python setup.py build_ext --inplace')
        status = 'KO'
        report = 'Uncompiled cython code, please run setup.py: e.g.\n python setup.py build_ext --inplace'
        return status, report

    # find the metadata
    reduction_metadata = metadata.MetaData()
    reduction_metadata.load_all_metadata(setup.red_dir, 'pyDANDIA_metadata.fits')

    log.info('Determining the kernel size for all images based on their FWHM')
    fwhm_max = 0.
    shift_max = 0
    shifts = []
    fwhms = []
    pixscale = reduction_metadata.reduction_parameters[1]['PIX_SCALE'][0]

    for stats_entry in reduction_metadata.images_stats[1]:

        if np.isfinite(float(stats_entry['SHIFT_X'])):
            shifts.append(float(stats_entry['SHIFT_X']))

        if np.isfinite(float(stats_entry['SHIFT_Y'])):
            shifts.append(float(stats_entry['SHIFT_Y']))

        # Note this is using the sigma of a bi-variate normal distribution, NOT
        # the true FWHM:
        # if float(stats_entry['FWHM'])> fwhm_max:
        #    fwhm_max = stats_entry['FWHM']
        if float(stats_entry['SIGMA_X']) > fwhm_max:
            fwhm_max = stats_entry['SIGMA_X']
        if float(stats_entry['SIGMA_Y']) > fwhm_max:
            fwhm_max = stats_entry['SIGMA_Y']

        if abs(float(stats_entry['SHIFT_X'])) > shift_max:
            shift_max = abs(float(stats_entry['SHIFT_X']))

        if abs(float(stats_entry['SHIFT_Y'])) > shift_max:
            shift_max = abs(float(stats_entry['SHIFT_Y']))

       # fwhms.append(((float(stats_entry['SIGMA_Y'])) ** 2 + (float(stats_entry['SIGMA_Y'])) ** 2) ** 0.5)  # arcsec
        fwhms.append(float(stats_entry['FWHM']))  # arcsec
    fwhms = np.array(fwhms)
    mask = np.isnan(fwhms)
    fwhms[mask] = 99

    # image smaller or equal 500x500
    large_format_image = False
    sigma_max = fwhm_max / (2. * (2. * np.log(2.)) ** 0.5)

    # Factor 4 corresponds to the radius of 2*FWHM the old pipeline
    log.info('Finding kernel_sizes for multiple pre-calculated umatrices')
    kernel_percentile = [20., 40.]  # assumes ker_rad = 2 * FWHM, check config!
    kernel_size_array = []

    for percentile in kernel_percentile:

        kernel_size_tmp = int(
            1.0*float(reduction_metadata.reduction_parameters[1]['KER_RAD'][0]) * np.percentile(fwhms, percentile))

        try:
            if kernel_size_tmp == kernel_size_array[-1]:
                kernel_size_tmp += 1
        except:
            pass
        if kernel_size_tmp % 2 == 0:
            kernel_size_tmp += 1

        if kernel_size_tmp < 1:
            kernel_size_tmp = 1

        kernel_size_array.append(kernel_size_tmp)

    shifts = np.array(shifts)
    # requires images to be sufficiently aligned and adds a safety margin of 10 -> mv to config.json
    maxshift = int(np.max(shifts)) + 10

    log.info('Identifying the images that need to be processed')
    all_images = reduction_metadata.find_all_images(setup, reduction_metadata,
                                                    os.path.join(setup.red_dir, 'data'), log=log)

    new_images = reduction_metadata.find_images_need_to_be_process(setup, all_images,
                                                                   stage_number=5, log=log)
    image_red_status = reduction_metadata.fetch_image_status(5)

    kernel_directory_path = os.path.join(setup.red_dir, 'kernel')
    diffim_directory_path = os.path.join(setup.red_dir, 'diffim')

    if not os.path.exists(kernel_directory_path):
        os.mkdir(kernel_directory_path)

    if not os.path.exists(diffim_directory_path):
        os.mkdir(diffim_directory_path)

    reduction_metadata.update_column_to_layer('data_architecture', 'KERNEL_PATH', kernel_directory_path)

    # difference images are written for verbosity level > 0
    reduction_metadata.update_column_to_layer('data_architecture', 'DIFFIM_PATH', diffim_directory_path)
    data_image_directory = os.path.join(setup.red_dir, 'resampled')
    ref_directory_path = '.'

    # For a quick image subtraction, pre-calculate a sufficiently large u_matrix
    # based on the largest FWHM and store it to disk -> needs config switch
    log.info('Pre-calculating a u-matrix based on the largest FWHM')

    try:
        reference_image_name = str(reduction_metadata.data_architecture[1]['REF_IMAGE'][0])
        reference_image_directory = str(reduction_metadata.data_architecture[1]['REF_PATH'][0])
        max_adu = float(reduction_metadata.reduction_parameters[1]['MAXVAL'][0])
        ref_row_index = np.where(reduction_metadata.images_stats[1]['IM_NAME'] == str(
            reduction_metadata.data_architecture[1]['REF_IMAGE'][0]))[0][0]

        ref_sigma_x = reduction_metadata.images_stats[1][ref_row_index]['SIGMA_X']
        ref_sigma_y = reduction_metadata.images_stats[1][ref_row_index]['SIGMA_Y']
        ref_fwhm_x = reduction_metadata.images_stats[1][ref_row_index]['FWHM']
        ref_fwhm_y = reduction_metadata.images_stats[1][ref_row_index]['FWHM']
        ref_stats = [ref_fwhm_x, ref_fwhm_y, ref_sigma_x, ref_sigma_y]

        logs.ifverbose(log, setup, 'Using reference image:' + reference_image_name)
        logs.ifverbose(log, setup, 'with statistics ref_fwhm, ref_fwhm, ref_sigma_x, ref_sigma_y = ' + repr(ref_stats))

    except Exception as e:
        log.ifverbose(log, setup, 'Reference/Images ! Abort stage5' + str(e))
        status = 'KO'
        report = 'No reference image found!'
        return status, report, reduction_metadata

    if not ('SHIFT_X' in reduction_metadata.images_stats[1].keys()) and (
            'SHIFT_Y' in reduction_metadata.images_stats[1].keys()):
        log.ifverbose(log, setup, 'No xshift! run stage4 ! Abort stage5')
        status = 'KO'
        report = 'No alignment data found!'
        return status, report, reduction_metadata

    #quality_metrics = subtract_with_constant_kernel(new_images, reference_image_name, reference_image_directory,
    #                                                reduction_metadata, setup, data_image_directory, kernel_size_array,
    #                                                max_adu, ref_stats, maxshift, kernel_directory_path,
    #                                                diffim_directory_path, log)

    quality_metrics = subtract_with_constant_kernel_on_stamps(new_images, reference_image_name, reference_image_directory,
                                                    reduction_metadata, setup, data_image_directory, kernel_size_array,
                                                    max_adu, ref_stats, maxshift, kernel_directory_path,
                                                    diffim_directory_path, log)
    data = np.copy(quality_metrics)
    if ('PSCALE' in reduction_metadata.images_stats[1].keys()):

        for idx in range(len(quality_metrics)):

            target_image = data[idx][0]
            pscale = data[idx][1]
            pscale_err = data[idx][2]
            median_sky = data[idx][3]
            variance_per_pixel = data[idx][4]
            ngood = float(data[idx][5])
            kurtosis_quality = data[idx][6]
            skew_quality = data[idx][7]
            row_index = np.where(reduction_metadata.images_stats[1]['IM_NAME'].data == target_image)[0][0]

            try:
                reduction_metadata.update_a_cell_to_layer('images_stats', row_index, 'PSCALE', pscale)
                reduction_metadata.update_a_cell_to_layer('images_stats', row_index, 'PSCALE_ERR', pscale_err)
                reduction_metadata.update_a_cell_to_layer('images_stats', row_index, 'MEDIAN_SKY', median_sky)
                reduction_metadata.update_a_cell_to_layer('images_stats', row_index, 'VAR_PER_PIX_DIFF',
                                                          variance_per_pixel)
                reduction_metadata.update_a_cell_to_layer('images_stats', row_index, 'N_UNMASKED', ngood)
                reduction_metadata.update_a_cell_to_layer('images_stats', row_index, 'SKEW_DIFF', skew_quality)
                reduction_metadata.update_a_cell_to_layer('images_stats', row_index, 'KURTOSIS_DIFF', pscale_err)

            except:
                logs.ifverbose(log, setup, 'Updating image stats -> some expected metrics missing')

    else:
        log.info('Constructing quality metrics columns in metadata')
        sorted_data = sort_quality_metrics(quality_metrics, reduction_metadata)

        column_format = 'float'
        column_unit = ''
        reduction_metadata.add_column_to_layer('images_stats', 'PSCALE', sorted_data[:, 1],
                                               new_column_format=column_format,
                                               new_column_unit=column_unit)
        reduction_metadata.add_column_to_layer('images_stats', 'PSCALE_ERR', sorted_data[:, 2],
                                               new_column_format=column_format,
                                               new_column_unit=column_unit)
        reduction_metadata.add_column_to_layer('images_stats', 'MEDIAN_SKY', sorted_data[:, 3],
                                               new_column_format=column_format,
                                               new_column_unit=column_unit)
        reduction_metadata.add_column_to_layer('images_stats', 'VAR_PER_PIX_DIFF', sorted_data[:, 4],
                                               new_column_format=column_format,
                                               new_column_unit=column_unit)
        reduction_metadata.add_column_to_layer('images_stats', 'N_UNMASKED', sorted_data[:, 5],
                                               new_column_format=column_format,
                                               new_column_unit=column_unit)
        reduction_metadata.add_column_to_layer('images_stats', 'SKEW_DIFF', sorted_data[:, 6],
                                               new_column_format=column_format,
                                               new_column_unit=column_unit)
        reduction_metadata.add_column_to_layer('images_stats', 'KURTOSIS_DIFF', sorted_data[:, 7],
                                               new_column_format=column_format,
                                               new_column_unit=column_unit)

    log.info('Updating metadata')
    image_red_status = metadata.set_image_red_status(image_red_status,'1',image_list=new_images)
    reduction_metadata.update_reduction_metadata_reduction_status_dict(image_red_status,
                                                stage_number=5, log=log)
    reduction_metadata.save_updated_metadata(
        reduction_metadata.data_architecture[1]['OUTPUT_DIRECTORY'][0],
        reduction_metadata.data_architecture[1]['METADATA_NAME'][0],
        log=log)

    logs.close_log(log)
    status = 'OK'
    report = 'Completed successfully'
    return status, report


def fit_kernel(params, data_image, ref_image, mask):
    from scipy.signal import convolve2d, fftconvolve
    ker = params.reshape(int(len(params) ** 0.5), int(len(params) ** 0.5))

    model = fftconvolve(ref_image, ker, mode='same')

    res = model - data_image

    weights = np.ones(data_image.shape)
    weights[mask] = 0

    chi = np.sum(res ** 2 * weights ** 2)
    print(chi)
    return chi


def round_unc(val, err):
    '''
    Round to uncertainty digits

    :param float val: measured value
    :param float err: uncertainty of value

    :return: formatted uncertainty
    '''
    if val == 0.0 and err == 0.0:
        return "0.0 +/- 0.0"
    else:
        try:
            digs = abs(int(np.log10(err / abs(val))))
        except ValueError:
            print('ERROR in round_unc, inputs: ',err, val)
            exit()
        val_round = round(val, digs)
        unc_round = round(err, digs)
        return "{0} +/- {1}".format(val_round, unc_round)

def sort_quality_metrics(quality_metrics, reduction_metadata):
    """Function to sort the quality metrics array into the same order as
    the metadata's image_stat table.

    Inputs:
        :param array quality_metrics: QC indices per NEW image
        :param metadata reduction_metadata: Metadata for the current dataset

    Output:
        :param array sorted_data: Sorted quality_metrics list
    """

    image_list = reduction_metadata.images_stats[1]['IM_NAME']

    sorted_data = []
    for image in image_list:
        sorted_data.append([image, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0])
    sorted_data = np.array(sorted_data)

    new_images = list(np.array(quality_metrics)[:,0])
    for i,target_image in enumerate(image_list):
        try:
            new_index = new_images.index( str(target_image) )
            sorted_data[i] = quality_metrics[new_index]
        except ValueError:
            sorted_data[i] = [target_image, -1.0, -1.0, -1.0, -1.0, 0, -1.0, -1.0]

    return sorted_data

def smoothing_2sharp_images(reduction_metadata, ref_fwhm_x, ref_fwhm_y, ref_sigma_x, ref_sigma_y, row_index):
    smoothing = 0.
    smoothing_y = 0.

    if reduction_metadata.images_stats[1][row_index]['SIGMA_X'] < ref_fwhm_x:
        sigma_x = reduction_metadata.images_stats[1][row_index]['SIGMA_X'] / (2. * (2. * np.log(2.)) ** 0.5)
        smoothing = (ref_sigma_x ** 2 - sigma_x ** 2) ** 0.5

    if reduction_metadata.images_stats[1][row_index]['SIGMA_Y'] < ref_fwhm_y:
        sigma_y = reduction_metadata.images_stats[1][row_index]['SIGMA_Y'] / (2. * (2. * np.log(2.)) ** 0.5)
        smoothing_y = (ref_sigma_y ** 2 - sigma_y ** 2) ** 0.5

    if smoothing_y > smoothing:
        smoothing = smoothing_y

    if smoothing >0 :
        smoothing = 3* (ref_sigma_x ** 2 + ref_sigma_y ** 2) ** 0.5
    smoothing = 0
    return smoothing


def resampled_median_stack(setup, reduction_metadata, new_images, log):
    log.info('Resampling a stacked median image')

    image_stack = []
    data_image_directory = os.path.join(setup.red_dir, 'resampled')
    image_sum = []

    for new_image in new_images:

        row_index = np.where(reduction_metadata.images_stats[1]['IM_NAME'] == new_image)[0][0]

        if image_sum == []:
            image_sum = open_an_image(setup, data_image_directory, new_image, log).data

        else:
            image_sum = image_sum + open_an_image(setup, data_image_directory, new_image, log).data

    stack_median_image = image_sum / float(len(new_images))

    return stack_median_image


def subtract_with_constant_kernel(new_images, reference_image_name, reference_image_directory, reduction_metadata,
                                  setup, data_image_directory, kernel_size_array, max_adu, ref_stats, maxshift,
                                  kernel_directory_path, diffim_directory_path, log):
    """subtracting image with a single kernel
    This routine calculates the umatrix of the least squares problem defining the kernel
    and subtracts the model
    :param object new images : list of unprocessed images

    :return: None
    :rtype: None
    """

    log.info('Starting image subtraction with a constant kernel')

    grow_kernel = 4. * float(reduction_metadata.reduction_parameters[1]['KER_RAD'][0])
    pixscale = reduction_metadata.reduction_parameters[1]['PIX_SCALE'][0]
    log.info('Grow_kernel factor = ' + str(grow_kernel))

    if len(new_images) > 0:
        try:

            master_mask = fits.open(
                os.path.join(reduction_metadata.data_architecture[1]['REF_PATH'][0], 'master_mask.fits'))
            # master_mask = np.where(master_mask[0].data > 0.85 * np.max(master_mask[0].data))
            master_mask = master_mask[0].data > 0



        except:
            master_mask = []

        try:
            resampled_median_image = resampled_median_stack(setup, reduction_metadata, new_images, log)
        except:
            resampled_median_image = np.zeros(np.shape(master_mask[0].data))

        kernel_size_max = max(kernel_size_array)
        reference_images = []
        log.info('Opening and masking ' + str(len(kernel_size_array)) + ' images for the reference image array')

        for idx in range(len(kernel_size_array)):
            log.info(' -> Starting image ' + str(idx) + ', kernel size ' + str(kernel_size_array[idx]))
            reference_images.append(
                open_reference(setup, reference_image_directory, reference_image_name, kernel_size_array[idx], max_adu,
                               ref_extension=0, log=log, central_crop=maxshift, master_mask=master_mask,
                               external_weight=resampled_median_image))
            log.info(' -> Completed image masking')

        if os.path.exists(os.path.join(kernel_directory_path, 'unweighted_u_matrix.npy')):

            log.info('Loading pre-existing u-matrix')

            umatrices, kernel_sizes, max_adu_restored = np.load(
                os.path.join(kernel_directory_path, 'unweighted_u_matrix.npy'), allow_pickle=True)

            if (kernel_sizes != kernel_size_array) or (max_adu_restored != max_adu):
                # calculate and store unweighted umatrices
                umatrices = []
                for idx in range(len(kernel_size_array)):
                    umatrices.append(
                        umatrix_constant(reference_images[idx][0], kernel_size_array[idx], reference_images[idx][3]))
                np.save(os.path.join(kernel_directory_path, 'unweighted_u_matrix.npy'),
                        [umatrices, kernel_size_array, max_adu])

        else:

            log.info('Calculating and storing unweighted umatrices')
            umatrices = []

            for idx in range(len(kernel_size_array)):
                log.info(' -> Starting image ' + str(idx) + ', kernel size ' + str(kernel_size_array[idx]))
                umatrices.append(
                    umatrix_constant(reference_images[idx][0], kernel_size_array[idx], reference_images[idx][3]))
                log.info(' -> Completed u-matrix calculation')

            np.save(os.path.join(kernel_directory_path, 'unweighted_u_matrix.npy'),
                    [umatrices, kernel_size_array, max_adu])
            log.info('Stored u-matrix')

    log.info('Performing image subtraction')
    quality_metrics = []
    for new_image in new_images:
        log.info(new_image + ' quality metrics:')

        row_index = np.where(reduction_metadata.images_stats[1]['IM_NAME'] == new_image)[0][0]
        ref_fwhm_x, ref_fwhm_y, ref_sigma_x, ref_sigma_y = ref_stats
        x_shift, y_shift = -reduction_metadata.images_stats[1][row_index]['SHIFT_X'], - \
        reduction_metadata.images_stats[1][row_index]['SHIFT_Y']
        x_fwhm, y_fwhm = -reduction_metadata.images_stats[1][row_index]['SIGMA_X'], - \
        reduction_metadata.images_stats[1][row_index]['SIGMA_Y']
        x_fwhm, y_fwhm = -reduction_metadata.images_stats[1][row_index]['SIGMA_X'], - \
        reduction_metadata.images_stats[1][row_index]['SIGMA_Y']
        try:
            fwhm_val = int(grow_kernel * (float(x_fwhm) ** 2 + float(y_fwhm) ** 2) ** 0.5)
        except:
            fwhm_val = 999

        umatrix_index = int(np.digitize(fwhm_val, np.array(kernel_size_array)))
        umatrix_index = min(umatrix_index, len(kernel_size_array) - 1)
        umatrix = umatrices[umatrix_index]
        kernel_size = kernel_size_array[umatrix_index]
        reference_image, bright_reference_mask, reference_image_unmasked, noise_image = reference_images[umatrix_index]
        x_shift, y_shift = 0, 0

        log.info('Smoothing the data if the reference is not as sharp as a data image')
        smoothing = smoothing_2sharp_images(reduction_metadata, ref_fwhm_x, ref_fwhm_y, ref_sigma_x, ref_sigma_y,
                                            row_index)

        try:
            image_structure = image_handling.determine_image_struture(os.path.join(data_image_directory, new_image),log=log)
            data_image, data_image_unmasked = open_data_image(setup, data_image_directory, new_image,
                                                              bright_reference_mask, kernel_size, max_adu,
                                                              xshift=x_shift, yshift=y_shift, sigma_smooth=smoothing,
                                                              central_crop=maxshift, data_extension=image_structure['sci'])

            missing_data_mask = (data_image == 0.)
            b_vector = bvector_constant(reference_image, data_image, kernel_size, noise_image)
            kernel_matrix, bkg_kernel, kernel_uncertainty = kernel_solution(umatrix, b_vector, kernel_size,
                                                                            circular=False)
            # res = so.minimize(fit_kernel,kernel_matrix.ravel(),args=(data_image,reference_image,bright_reference_mask))

            pscale = np.sum(kernel_matrix)
            pscale_err = np.sum(kernel_uncertainty ** 2) ** 0.5
            np.save(os.path.join(kernel_directory_path, 'kernel_' + new_image + '.npy'), [kernel_matrix, bkg_kernel])
            kernel_header = fits.Header()
            kernel_header['SCALEFAC'] = str(pscale)
            kernel_header['KERBKG'] = bkg_kernel
            hdu_kernel = fits.PrimaryHDU(kernel_matrix, header=kernel_header)
            hdu_kernel.writeto(os.path.join(kernel_directory_path, 'kernel_' + new_image), overwrite=True)
            hdu_kernel_err = fits.PrimaryHDU(kernel_uncertainty)
            hdu_kernel_err.writeto(os.path.join(kernel_directory_path, 'kernel_err_' + new_image), overwrite=True)
            # Particle data group formatting
            pscale_formatted = round_unc(pscale, pscale_err)
            difference_image = subtract_images(data_image_unmasked, reference_image_unmasked, kernel_matrix,
                                               kernel_size, bkg_kernel)
            # unmasked subtraction (for quality stats)
            mean_sky, median_sky, std_sky = sigma_clipped_stats(reference_image_unmasked, sigma=5.0)
            difference_image_um = subtract_images(data_image, reference_image, kernel_matrix, kernel_size, bkg_kernel)
            mask = reference_image != 0
            ngood = len(difference_image_um[mask])
            kurtosis_quality = kurtosis(difference_image_um[mask])
            skew_quality = skew(difference_image_um[mask])
            variance_per_pixel = np.var(difference_image_um[mask]) / float(ngood)
            if log is not None:
                logs.ifverbose(log, setup,
                               'b_vector calculated for:' + new_image + ' and scale factor ' + pscale_formatted + ' variace per pixel ' + str(
                                   round(variance_per_pixel, 4)) + ' in kernel bin ' + str(umatrix_index))

                # EXPERIMENTAL -> DISCARD outliers
            # mean_diff, median_diff, std_diff = sigma_clipped_stats(difference_image, sigma=4.0)
            # out3sig = np.where(np.abs(difference_image) > 4.*+std_diff)
            # outliers_list = []
            # for rm_idx in range(len(out3sig[0])):
            #    outliers_list.append((out3sig[0][rm_idx],out3sig[1][rm_idx]))
            # update umatrix
            # b_vector_2 = bvector_constant_clean(reference_image, data_image, kernel_size, b_vector, outliers_list)
            # umatrix_2 = umatrix_constant_clean(reference_image, kernel_size, umatrix, outliers_list)
            # kernel_matrix_2, bkg_kernel_2, kernel_uncertainty_2 = kernel_solution(umatrix_2, b_vector_2, kernel_size, circular = False)
            # difference_image = subtract_images(data_image_unmasked, reference_image_unmasked, kernel_matrix_2, kernel_size, bkg_kernel_2)
            new_header = fits.Header()
            new_header['SCALEFAC'] = pscale
            new_header['SCALEERR'] = pscale_err
            new_header['VARPP'] = variance_per_pixel
            new_header['NGOOD'] = ngood
            new_header['SKY'] = median_sky
            new_header['KURTOSIS'] = kurtosis_quality
            new_header['SKEW'] = skew_quality
            quality_metrics.append(
                [new_image, pscale, pscale_err, median_sky, variance_per_pixel, ngood, kurtosis_quality, skew_quality])
            difference_image_hdu = fits.PrimaryHDU(difference_image, header=new_header)
            difference_image_hdu.writeto(os.path.join(diffim_directory_path, 'diff_' + new_image), overwrite=True)

        except Exception as e:

            quality_metrics.append([new_image, -1.0, -1.0, -1.0, -1.0, 0, -1.0, -1.0])

            if log is not None:
                logs.ifverbose(log, setup,
                               'kernel matrix computation or shift failed:' + new_image + '. skipping! ' + str(e))
            else:
                print(str(e))

        log.info(' -> ' + repr(quality_metrics[-1]))

    return quality_metrics


def subtract_with_constant_kernel_on_stamps(new_images, reference_image_name, reference_image_directory,
                                            reduction_metadata,
                                            setup, data_image_directory, kernel_size_array, max_adu, ref_stats,
                                            maxshift,
                                            kernel_directory_path, diffim_directory_path, log):
    """subtracting image with a single kernel individual stamps
    This routine calculates the umatrix of the least squares problem defining the kernel
    and subtracts the model
    :param object new images : list of unprocessed images

    :return: None
    :rtype: None
    """

    log.info('Starting image subtraction with a constant kernel')

    grow_kernel = 1.0* float(reduction_metadata.reduction_parameters[1]['KER_RAD'][0])
    pixscale = reduction_metadata.reduction_parameters[1]['PIX_SCALE'][0]

    list_of_stamps = reduction_metadata.stamps[1]['PIXEL_INDEX'].tolist()

    log.info('Grow_kernel factor = ' + str(grow_kernel))

    if len(new_images) > 0:
        try:

            master_mask = fits.open(
                os.path.join(reduction_metadata.data_architecture[1]['REF_PATH'][0], 'master_mask.fits'))
            master_mask = master_mask[0].data > 0

        except:
            master_mask = []

        kernel_size_max = max(kernel_size_array)
        reference_images = []
        log.info('Opening and masking ' + str(len(kernel_size_array)) + ' images for the reference image array')

        for idx in range(len(kernel_size_array)):
            log.info(' -> Starting image ' + str(idx) + ', kernel size ' + str(kernel_size_array[idx]))
            ref_structure = image_handling.determine_image_struture(os.path.join(reference_image_directory, reference_image_name),log=log)
            reference_images.append(
                open_reference(setup, reference_image_directory, reference_image_name, kernel_size_array[idx], max_adu,
                               ref_extension=ref_structure['sci'], log=log, central_crop=maxshift, master_mask=master_mask,
                               external_weight=None))
            log.info(' -> Completed image masking')

        try:

            umatrices_grid, kernel_sizes, max_adu_restored = np.load(os.path.join(kernel_directory_path, 'unweighted_u_matrix.npy'), allow_pickle=True)
            log.info('Loading pre-existing u-matrices')

            if (kernel_sizes != kernel_size_array) or (max_adu_restored != max_adu):
                # calculate and store unweighted umatrices
                umatrices_grid = []
                for idx in range(len(kernel_size_array)):
                    umatrices = []

                    for stamp in list_of_stamps:
                        stamp_row = np.where(reduction_metadata.stamps[1]['PIXEL_INDEX'] == stamp)[0][0]
                        xmin = reduction_metadata.stamps[1]['PIXEL_INDEX'][stamp_row]['X_MIN'][0]
                        xmax = reduction_metadata.stamps[1]['PIXEL_INDEX'][stamp_row]['X_MAX'][0]
                        ymin = reduction_metadata.stamps[1]['PIXEL_INDEX'][stamp_row]['Y_MIN'][0]
                        ymax = reduction_metadata.stamps[1]['PIXEL_INDEX'][stamp_row]['Y_MAX'][0]
                        umatrices.append(
                            umatrix_constant(reference_images[idx][0][ymin:ymax, xmin:xmax], kernel_size_array[idx],
                                             reference_images[idx][3][ymin:ymax, xmin:xmax]))
                    umatrices_grid.append(umatrices)

                np.save(os.path.join(kernel_directory_path, 'unweighted_u_matrix.npy'),
                        [umatrices_grid, kernel_size_array, max_adu])

        except:

            log.info('Calculating and storing unweighted umatrices')

            umatrices_grid = []
            for idx in range(len(kernel_size_array)):
                log.info(' -> Starting image ' + str(idx) + ', kernel size ' + str(kernel_size_array[idx]))
                umatrices = []

                for stamp in list_of_stamps:
                    stamp_row = np.where(reduction_metadata.stamps[1]['PIXEL_INDEX'] == stamp)[0][0]
                    xmin = int( reduction_metadata.stamps[1][stamp_row]['X_MIN'])
                    xmax = int( reduction_metadata.stamps[1][stamp_row]['X_MAX'])
                    ymin = int( reduction_metadata.stamps[1][stamp_row]['Y_MIN'])
                    ymax = int( reduction_metadata.stamps[1][stamp_row]['Y_MAX'])
                    umatrices.append(
                        umatrix_constant(reference_images[idx][0][ymin:ymax, xmin:xmax], kernel_size_array[idx],
                                         reference_images[idx][3][ymin:ymax, xmin:xmax]))
                    log.info(' -> Completed u-matrix calculation on stamp '+str(stamp))
                umatrices_grid.append(umatrices)

            np.save(os.path.join(kernel_directory_path, 'unweighted_u_matrix.npy'),
                    [umatrices_grid, kernel_size_array, max_adu])
            log.info('Stored u-matrix')

    log.info('Performing image subtraction')
    quality_metrics = []
    for new_image in new_images:

        kernel_directory = os.path.join(kernel_directory_path, new_image)
        diffim_directory = os.path.join(diffim_directory_path, new_image)

        try:
            os.mkdir(kernel_directory)

            os.mkdir(diffim_directory)

        except:
            pass
        log.info(new_image + ' quality metrics:')

        row_index = np.where(reduction_metadata.images_stats[1]['IM_NAME'] == new_image)[0][0]
        fwhm_val = reduction_metadata.images_stats[1][row_index]['FWHM'] * grow_kernel
        ref_fwhm_x, ref_fwhm_y, ref_sigma_x, ref_sigma_y = ref_stats


        umatrix_index = int(np.digitize(fwhm_val, np.array(kernel_size_array)))
        umatrix_index = min(umatrix_index, len(kernel_size_array) - 1)
       # umatrix_index = -1
        umatrix_grid = umatrices_grid[umatrix_index]
        kernel_size = kernel_size_array[umatrix_index]
        reference_image, bright_reference_mask, reference_image_unmasked, noise_image = reference_images[umatrix_index]


        x_shift, y_shift = 0, 0

        log.info('Smoothing the data if the reference is not as sharp as a data image')
        smoothing = smoothing_2sharp_images(reduction_metadata, ref_fwhm_x, ref_fwhm_y, ref_sigma_x, ref_sigma_y,
                                            row_index)


        image_directory = reduction_metadata.data_architecture[1]['IMAGES_PATH'][0]
        data_image = fits.open(os.path.join(image_directory, new_image))[0].data

        stamps_directory = os.path.join(data_image_directory, new_image)

        warp_matrix =  np.load(os.path.join(stamps_directory, 'warp_matrice_image.npy'))
        resample_image = stage4.warp_image(data_image,warp_matrix)

        try:



            pscales = []
            pscales_err = []
            medians_sky = []
            variances_per_pixel = []
            ngoods = []
            kurtosises_quality = []
            skewes_quality = []



            for stamp in list_of_stamps:

                stamp_row = np.where(reduction_metadata.stamps[1]['PIXEL_INDEX'] == stamp)[0][0]
                xmin = int(reduction_metadata.stamps[1][stamp_row]['X_MIN'])
                xmax = int(reduction_metadata.stamps[1][stamp_row]['X_MAX'])
                ymin = int(reduction_metadata.stamps[1][stamp_row]['Y_MIN'])
                ymax = int(reduction_metadata.stamps[1][stamp_row]['Y_MAX'])


                ref = reference_image[kernel_size:-kernel_size, kernel_size:-kernel_size][ymin:ymax, xmin:xmax]

                ref_unmasked = reference_image_unmasked[ymin:ymax, xmin:xmax]


                img = resample_image[ymin:ymax, xmin:xmax]


                ref_extended = np.zeros((ref.shape[0]+2*kernel_size,ref.shape[1]+2*kernel_size))
                ref_extended[kernel_size:-kernel_size, kernel_size:-kernel_size] = ref

                ref_unmasked_extended = np.zeros((ref.shape[0] + 2 * kernel_size, ref.shape[1] + 2 * kernel_size))
                ref_unmasked_extended [kernel_size:-kernel_size, kernel_size:-kernel_size] = ref_unmasked

                bal_mask_extended = np.ones((np.shape(ref)[0] + 2 * kernel_size, np.shape(ref)[1] + 2 * kernel_size)).astype(bool)


                bal_mask_extended[kernel_size:-kernel_size, kernel_size:-kernel_size] = \
                    master_mask[ymin:ymax, xmin:xmax]
                #img = data_image[ymin:ymax, xmin:xmax]
                #img = fits.open(os.path.join(stamps_directory, 'resample__stamp_' + str(stamp) + '.fits'))[0].data
                #img_unmasked = np.copy(img)

                #img, img_unmasked = open_data_image(setup,stamps_directory, 'resample_stamp_'+str(stamp)+'.fits',
                #                                              bal_mask_extended, kernel_size, max_adu,
                #                                              xshift=x_shift, yshift=y_shift, sigma_smooth=smoothing,
                #                                              central_crop=maxshift)

                warp_matrix =  np.load(os.path.join(stamps_directory, 'warp_matrice_stamp_'+str(stamp)+'.npy'))
                img = stage4.warp_image(img,warp_matrix)

                data_image, data_image_unmasked = mask_the_image(img,max_adu,bal_mask_extended,kernel_size)



                noisy = noise_image[kernel_size:-kernel_size, kernel_size:-kernel_size][ymin:ymax, xmin:xmax]
                noise = np.zeros(ref_extended.shape)
                noise[kernel_size:-kernel_size, kernel_size:-kernel_size] = noisy


                umatrix = umatrix_grid[stamp_row]
                b_vector = bvector_constant(ref_extended,data_image, kernel_size, noise)

                kernel_matrix, bkg_kernel, kernel_uncertainty = kernel_solution(umatrix, b_vector, kernel_size,
                                                                            circular=False)

                pscale = np.sum(kernel_matrix)
                pscale_err = np.sum(kernel_uncertainty ** 2) ** 0.5
                np.save(os.path.join(kernel_directory, 'kernel_stamp_' + str(stamp) + '.npy'), [kernel_matrix, bkg_kernel])
                kernel_header = fits.Header()
                kernel_header['SCALEFAC'] = str(pscale)
                kernel_header['KERBKG'] = bkg_kernel
                hdu_kernel = fits.PrimaryHDU(kernel_matrix, header=kernel_header)
                hdu_kernel.writeto(os.path.join(kernel_directory, 'kernel_stamp_' + str(stamp) + '.fits'), overwrite=True)
                hdu_kernel_err = fits.PrimaryHDU(kernel_uncertainty)
                hdu_kernel_err.writeto(os.path.join(kernel_directory, 'kernel_err_stamp_' + str(stamp) + '.fits'), overwrite=True)
                # Particle data group formatting
                pscale_formatted = round_unc(pscale, pscale_err)
                difference_image = subtract_images(data_image_unmasked, ref_unmasked, kernel_matrix,
                                               kernel_size, bkg_kernel)

                # unmasked subtraction (for quality stats)
                mean_sky, median_sky, std_sky = sigma_clipped_stats(ref_unmasked, sigma=5.0)
                difference_image_um = subtract_images(data_image, ref_extended, kernel_matrix, kernel_size, bkg_kernel)
                mask = ref_extended != 0
                ngood = len(difference_image_um[mask])
                kurtosis_quality = kurtosis(difference_image_um[mask])
                skew_quality = skew(difference_image_um[mask])
                variance_per_pixel = np.var(difference_image_um[mask]) / float(ngood)



                new_header = fits.Header()
                new_header['SCALEFAC'] = pscale
                new_header['SCALEERR'] = pscale_err
                new_header['VARPP'] = variance_per_pixel
                new_header['NGOOD'] = ngood
                new_header['SKY'] = median_sky
                new_header['KURTOSIS'] = kurtosis_quality
                new_header['SKEW'] = skew_quality
                difference_image_hdu = fits.PrimaryHDU(difference_image, header=new_header)
                try:
                    os.mkdir(diffim_directory)
                except:
                    pass
                difference_image_hdu.writeto(os.path.join(diffim_directory, 'diff_stamp_' + str(stamp) + '.fits'), overwrite=True)

                pscales.append(pscale)
                pscales_err.append(pscale_err)
                medians_sky.append(median_sky)
                variances_per_pixel.append(variance_per_pixel)
                ngoods.append(ngood)
                kurtosises_quality.append(kurtosis_quality)
                skewes_quality.append(skew_quality)

            if log is not None:
                logs.ifverbose(log, setup,
                               'b_vector calculated for:' + new_image + ' and scale factor ' + str(np.median(pscales))
                               + ' +/- '+str(np.median(pscales_err)) +' variance per pixel ' + str(
                                   np.round(np.median(variances_per_pixel), 4)) + ' in kernel bin ' + str(umatrix_index))
            quality_metrics.append(
                [new_image, np.median(pscales), np.median(pscales_err), np.median(medians_sky),
                 np.median(variances_per_pixel), np.median(ngoods), np.median(kurtosises_quality),
                 np.median(skewes_quality)])


        except Exception as e:

            quality_metrics.append([new_image, -1.0, -1.0, -1.0, -1.0, 0, -1.0, -1.0])

            if log is not None:
                logs.ifverbose(log, setup,
                               'kernel matrix computation or shift failed:' + new_image + '. skipping! ' + str(e))
            else:
                print(str(e))

        log.info(' -> ' + repr(quality_metrics[-1]))

    return quality_metrics


def open_reference_stamps(setup, reduction_metadata, reference_image_directory, reference_image_name, kernel_size,
                          max_adu, log, maxshift, min_adu=None):
    reference_pool_stamps = []
    ref_structure = image_handling.determine_image_struture(os.path.join(reference_image_directory, reference_image_name))
    ref_image1 = fits.open(os.path.join(reference_image_directory, reference_image_name), mmap=True)
    # load all reference subimages
    for substamp_idx in range(len(reduction_metadata.stamps[1])):
        print(substamp_idx, 'of', len(reduction_metadata.stamps[1]))
        # prepare subset slice based on metadata

        subset_slice = [int(reduction_metadata.stamps[1][substamp_idx]['Y_MIN']),
                        int(reduction_metadata.stamps[1][substamp_idx]['Y_MAX']),
                        int(reduction_metadata.stamps[1][substamp_idx]['X_MIN']),
                        int(reduction_metadata.stamps[1][substamp_idx]['X_MAX'])]
        reference_image, bright_reference_mask, reference_image_unmasked = open_reference(setup,
                                                                                          reference_image_directory,
                                                                                          reference_image_name,
                                                                                          kernel_size, max_adu,
                                                                                          ref_extension=ref_structure['sci'], log=log,
                                                                                          central_crop=maxshift,
                                                                                          subset=subset_slice,
                                                                                          ref_image1=ref_image1,
                                                                                          min_adu=min_adu)
        reference_pool_stamps.append([reference_image, kernel_size, bright_reference_mask, reference_image_unmasked])
    return reference_pool_stamps


def subtract_large_format_image(new_images, reference_image_name, reference_image_directory, reduction_metadata, setup,
                                data_image_directory, kernel_size, max_adu, ref_stats, maxshift, kernel_directory_path,
                                diffim_directory_path, log=None):
    if len(new_images) > 0:
        reference_pool_stamps = open_reference_stamps(setup, reduction_metadata, reference_image_directory,
                                                      reference_image_name, kernel_size, max_adu, log, maxshift,
                                                      min_adu=200.)
        umatrix_stamps = []
        # generate or load u matrix grid
        if (not os.path.exists(os.path.join(kernel_directory_path, 'unweighted_u_matrix_subimages.npy'))):
            pool = Pool(processes=mp.cpu_count())
            umatrix_stamps = (pool.map(umatrix_pool, reference_pool_stamps))
            pool.terminate()
            np.save(os.path.join(kernel_directory_path, 'unweighted_u_matrix_subimages.npy'),
                    [umatrix_stamps, kernel_size, max_adu, maxshift])
        else:
            umatrix_stamps, kernel_size, max_adu, maxshift = np.load(
                os.path.join(kernel_directory_path, 'unweighted_u_matrix_subimages.npy'), allow_pickle=True)

    # iterate over all images and subimages
    for new_image in new_images:
        image_structure = image_handling.determine_image_struture(os.path.join(data_image_directory, new_image),log=log)
        kernel_stamps = []
        pool_stamps = []
        data_image1 = fits.open(os.path.join(data_image_directory, new_image), mmap=True)
        row_index = np.where(reduction_metadata.images_stats[1]['IM_NAME'] == new_image)[0][0]
        x_shift, y_shift = -reduction_metadata.images_stats[1][row_index]['SHIFT_X'], - \
        reduction_metadata.images_stats[1][row_index]['SHIFT_Y']
        for substamp_idx in range(len(reduction_metadata.stamps[1])):
            subset_slice = [int(reduction_metadata.stamps[1][substamp_idx]['Y_MIN']),
                            int(reduction_metadata.stamps[1][substamp_idx]['Y_MAX']),
                            int(reduction_metadata.stamps[1][substamp_idx]['X_MIN']),
                            int(reduction_metadata.stamps[1][substamp_idx]['X_MAX'])]
            data_image, data_image_unmasked = open_data_image(setup, data_image_directory, new_image,
                                                              reference_pool_stamps[substamp_idx][2], kernel_size,
                                                              max_adu, xshift=x_shift, yshift=y_shift, sigma_smooth=0,
                                                              central_crop=maxshift, subset=subset_slice,
                                                              data_image1=data_image1, min_adu=200.,
                                                              data_extension=image_structure['sci'])
            missing_data_mask = (data_image == 0.)
            pool_stamps.append(
                [umatrix_stamps[substamp_idx], reference_pool_stamps[substamp_idx][0], data_image, kernel_size])
            if log is not None:
                logs.ifverbose(log, setup, 'b_vector calculated for img and slice:' + new_image + ' ' + str(
                    reduction_metadata.stamps[1][substamp_idx]))
        try:
            # broadcast subimages to different threads
            pool = Pool(processes=mp.cpu_count())
            kernel_stamps = (pool.map(kernel_solution_pool, pool_stamps))
            pool.terminate()
            # save kernel grid
            np.save(os.path.join(kernel_directory_path, 'kernel_multi_' + new_image + '.npy'), kernel_stamps)
        except Exception as e:
            if log is not None:
                logs.ifverbose(log, setup,
                               'kernel matrix computation or shift failed:' + new_image + '. skipping!' + str(e))
            else:
                print(str(e))
        subtract_subimage(setup, kernel_directory_path, new_image, reduction_metadata)


def noise_model(model_image, gain=1., readout_noise=0., flat=None, initialize=None):
    noise_image = np.copy(model_image)
    mask = noise_image == 0
    noise_image[noise_image == 0] = 1.

    weights = 1. / noise_image

    #weights[noise_image == 1] = 0.
    weights = np.ones(noise_image.shape)
    weights[mask] = 0
    return weights


def noise_model_constant(model_image, smooth=None):
    noise_image = np.copy(model_image)
    noise_image[noise_image == 0] = 1.
    weights = np.ones(np.shape(model_image))
    weights[noise_image == 1] = 0.

    return weights


def noise_model_blurred_ref(reference_image, bright_mask, sigma_max):
    noise_image = np.copy(reference_image)
    good_region = np.copy(np.where(noise_image != 0))
    bad_region = np.copy(np.where(noise_image == 0))
    readout_noise = 12.
    noise_image = gaussian_filter(noise_image, sigma=sigma_max)
    noise_image = noise_image ** 2
    noise_image = noise_image + readout_noise * readout_noise
    weights = 1. / noise_image
    weights[bright_mask] = 0.
    return weights


def mask_kernel(kernel_size_plus):
    mask_kernel = np.ones(kernel_size_plus * kernel_size_plus, dtype=float)
    mask_kernel = mask_kernel.reshape((kernel_size_plus, kernel_size_plus))
    xyc = int(kernel_size_plus / 2)
    radius_square = (xyc) ** 2
    for idx in range(kernel_size_plus):
        for jdx in range(kernel_size_plus):
            if (idx - xyc) ** 2 + (jdx - xyc) ** 2 >= radius_square:
                mask_kernel[idx, jdx] = 0.

    return mask_kernel


def umatrix_constant(reference_image, ker_size, noise_image, model_image=None, sigma_max=None, bright_mask=None,
                     nobkg=None):
    '''
    The kernel solution is supposed to implement the approach outlined in
    the Bramich 2008 paper. It generates the u matrix which is required
    for finding the best kernel and assumes it can be calculated
    sufficiently if the noise model either is neglected or similar on all
    model images. In order to run, it needs the largest possible kernel size
    and carefully masked regions which are expected to be affected on all
    data images.

    :param object image: reference image
    :param integer kernel size: edge length of the kernel in px

    :return: u matrix
    '''
    try:
        from umatrix_routine import umatrix_construction, umatrix_bvector_construction, bvector_construction
        from umatrix_routine import umatrix_construction_nobkg, bvector_construction_nobkg

    except ImportError:
        print('cannot import cython module umatrix_routine')
        return []

    if ker_size:
        if ker_size % 2 == 0:
            kernel_size = ker_size + 1
        else:
            kernel_size = ker_size

    weights = noise_model(copy.deepcopy(noise_image))
    # Prepare/initialize indices, vectors and matrices
    pandq = []
    n_kernel = kernel_size * kernel_size
    ncount = 0
    half_kernel_size = int(int(kernel_size) / 2)
    for lidx in range(kernel_size):
        for midx in range(kernel_size):
            pandq.append((lidx - half_kernel_size, midx - half_kernel_size))
    if nobkg == True:
        u_matrix = umatrix_construction_nobkg(reference_image, weights, pandq, n_kernel, kernel_size)
    else:
        u_matrix = umatrix_construction(reference_image, weights, pandq, n_kernel, kernel_size)
    return u_matrix


def umatrix_constant_clean(reference_image, ker_size, first_umatrix, outliers, model_image=None, sigma_max=None,
                           bright_mask=None, nobkg=None):
    '''
    The kernel solution is supposed to implement the approach outlined in
    the Bramich 2008 paper. It generates the u matrix which is required
    for finding the best kernel and assumes it can be calculated
    sufficiently if the noise model either is neglected or similar on all
    model images. In order to run, it needs the largest possible kernel size
    and carefully masked regions which are expected to be affected on all
    data images.

    :param object image: reference image
    :param integer kernel size: edge length of the kernel in px

    :return: u matrix
    '''
    try:
        from umatrix_routine import umatrix_construction, umatrix_bvector_construction, bvector_construction
        from umatrix_routine import umatrix_construction_nobkg, bvector_construction_nobkg
        from umatrix_routine import umatrix_construction_clean, bvector_construction_clean
    except ImportError:
        print('cannot import cython module umatrix_routine')
        return []

    if ker_size:
        if ker_size % 2 == 0:
            kernel_size = ker_size + 1
        else:
            kernel_size = ker_size
    if model_image == None or sigma_max == None:
        weights = noise_model(reference_image)
    else:
        weights = noise_model_blurred_ref(reference_image, bright_mask, sigma_max)
    # Prepare/initialize indices, vectors and matrices
    pandq = []
    n_kernel = kernel_size * kernel_size
    ncount = 0
    half_kernel_size = int(int(kernel_size) / 2)
    for lidx in range(kernel_size):
        for midx in range(kernel_size):
            pandq.append((lidx - half_kernel_size, midx - half_kernel_size))

    u_matrix_2 = umatrix_construction_clean(reference_image, weights, first_umatrix, pandq, n_kernel, kernel_size,
                                            outliers, len(outliers))
    return u_matrix_2


def umatrix_constant_threading(reference_image, ker_size, model_image=None, sigma_max=None, bright_mask=None,
                               nobkg=None):
    '''
    The kernel solution is supposed to implement the approach outlined in
    the Bramich 2008 paper. It generates the u matrix which is required
    for finding the best kernel and assumes it can be calculated
    sufficiently if the noise model either is neglected or similar on all
    model images. In order to run, it needs the largest possible kernel size
    and carefully masked regions which are expected to be affected on all
    data images.

    :param object image: reference image
    :param integer kernel size: edge length of the kernel in px

    :return: u matrix
    '''
    try:
        from umatrix_routine import umatrix_construction, umatrix_bvector_construction, bvector_construction
        from umatrix_routine import umatrix_construction_nobkg, bvector_construction_nobkg

    except ImportError:
        print('cannot import cython module umatrix_routine')
        return []

    if ker_size:
        if ker_size % 2 == 0:
            kernel_size = ker_size + 1
        else:
            kernel_size = ker_size
    if model_image == None or sigma_max == None:
        weights = noise_model_constant(reference_image)
    else:
        weights = noise_model_blurred_ref(reference_image, bright_mask, sigma_max)
    # Prepare/initialize indices, vectors and matrices
    pandq = []
    n_kernel = kernel_size * kernel_size
    ncount = 0
    half_kernel_size = int(int(kernel_size) / 2)
    for lidx in range(kernel_size):
        for midx in range(kernel_size):
            pandq.append((lidx - half_kernel_size, midx - half_kernel_size))

    # and now try to broadcast that for each pandq and see if that can be broadcast
    # at the end: combine als umatrix components
    if nobkg == True:
        u_matrix = umatrix_construction_nobkg(reference_image, weights, pandq, n_kernel, kernel_size)
    else:
        # cpu_count = min(mp.cpu_count(),8)
        # pandq_subsets = [pandq[i:i + cpu_count] for i in range(0, len(pandq), cpu_count)]
        # pool = Pool(processes = cpu_count)
        # umatrix_stamps = (pool.map(umatrix_single_pool,pandq))
        # pool.terminate()
        u_matrix = umatrix_construction(reference_image, weights, pandq, n_kernel, kernel_size)
    return u_matrix


def umatrix_single_pool(input_arg):
    '''
     Multithreading support for single umatrix umatrix_constant
    :param object image: list of reference image and kernel size
    :return: u matrix
    '''
    reference_image = input_arg[0]
    ker_size = input_arg[1]
    pq_subset = input_arg[2]
    print("umatrix threaded single start")
    return umatrix_constant(reference_image, ker_size, model_image=None, sigma_max=None, bright_mask=None)


def umatrix_pool(input_arg):
    '''
     Multithreading support for umatrix_constant
    :param object image: list of reference image and kernel size
    :return: u matrix
    '''
    reference_image = input_arg[0]
    ker_size = input_arg[1]
    print("umatrix start")
    return umatrix_constant(reference_image, ker_size, model_image=None, sigma_max=None, bright_mask=None)


def bvector_constant(reference_image, data_image, ker_size, noise_image, model_image=None, sigma_max=None,
                     bright_mask=None, nobkg=None):
    '''
    The kernel solution is supposed to implement the approach outlined in
    the Bramich 2008 paper. It generates the b_vector which is required
    for finding the best kernel and assumes it can be calculated
    sufficiently if the noise model either is neglected or similar on all
    model images. In order to run, it needs the largest possible kernel size
    and carefully masked regions on both - data and reference image

    :param object image: reference image
    :param integer kernel size: edge length of the kernel in px

    :return: b_vector
    '''
    try:
        from umatrix_routine import umatrix_construction, umatrix_bvector_construction, bvector_construction
        from umatrix_routine import umatrix_construction_nobkg, bvector_construction_nobkg

    except ImportError:
        print('cannot import cython module umatrix_routine')
        return []

    if ker_size:
        if ker_size % 2 == 0:
            kernel_size = ker_size + 1
        else:
            kernel_size = ker_size
    weights = noise_model(copy.deepcopy(noise_image))

    # Prepare/initialize indices, vectors and matrices
    pandq = []
    n_kernel = kernel_size * kernel_size
    ncount = 0
    half_kernel_size = int(int(kernel_size) / 2)
    for lidx in range(kernel_size):
        for midx in range(kernel_size):
            pandq.append((lidx - half_kernel_size, midx - half_kernel_size))
    if nobkg == True:
        b_vector = bvector_construction_nobkg(reference_image, data_image, weights, pandq, n_kernel, kernel_size)
    else:
        b_vector = bvector_construction(reference_image, data_image, weights, pandq, n_kernel, kernel_size)

    return b_vector


def bvector_constant_clean(reference_image, data_image, ker_size, first_b_vector, outliers, model_image=None,
                           sigma_max=None, bright_mask=None, nobkg=None):
    '''
    The kernel solution is supposed to implement the approach outlined in
    the Bramich 2008 paper. It generates the b_vector which is required
    for finding the best kernel and assumes it can be calculated
    sufficiently if the noise model either is neglected or similar on all
    model images. In order to run, it needs the largest possible kernel size
    and carefully masked regions on both - data and reference image.
    After an initial run, the umatrix is corrected for potential outliers
    and b vector are both updated.

    :param object image: reference image
    :param integer kernel size: edge length of the kernel in px

    :return: b_vector
    '''
    try:
        from umatrix_routine import umatrix_construction, umatrix_bvector_construction, bvector_construction
        from umatrix_routine import umatrix_construction_nobkg, bvector_construction_nobkg
        from umatrix_routine import umatrix_construction_clean, bvector_construction_clean
    except ImportError:
        print('cannot import cython module umatrix_routine')
        return []

    if ker_size:
        if ker_size % 2 == 0:
            kernel_size = ker_size + 1
        else:
            kernel_size = ker_size
    if model_image == None or sigma_max == None:
        weights = noise_model_constant(reference_image)
    else:
        weights = noise_model_blurred_ref(reference_image, bright_mask, sigma_max)
    # Prepare/initialize indices, vectors and matrices
    pandq = []
    n_kernel = kernel_size * kernel_size
    ncount = 0
    half_kernel_size = int(int(kernel_size) / 2)
    for lidx in range(kernel_size):
        for midx in range(kernel_size):
            pandq.append((lidx - half_kernel_size, midx - half_kernel_size))
    b_vector_2 = bvector_construction_clean(reference_image, data_image, weights, first_b_vector, pandq, n_kernel,
                                            kernel_size, outliers, len(outliers))

    return b_vector_2


def bvector_pool(input_arg, reference_image, data_image, ker_size, model_image=None, sigma_max=None, bright_mask=None):
    '''
     Multithreading support for bvector_constant
    :param object image: list of reference images, data images and kernel size
    :return: b vector
    '''
    reference_image = input_arg[0]
    data_image = input_arg[1]
    ker_size = input_arg[2]
    return bvector_constant(reference_image, data_image, ker_size, model_image=None, sigma_max=None, bright_mask=None)


def kernel_preparation_matrix(data_image, reference_image, ker_size, model_image=None):
    '''
    The kernel solution is supposed to implement the approach outlined in
    the Bramich 2008 paper. The data image is indexed using i,j
    the reference image should have the same shape as the data image
    the kernel is indexed via l, m. kernel_size requires as input the edge
    length of the kernel. The kernel matrix k_lm and a background b0
    k_lm, where l and m correspond to the kernel pixel indices defines
    The resulting vector b is obtained from the matrix U_l,m,l

    :param object image: data image
    :param object image: reference image
    :param integer kernel size: edge length of the kernel in px
    :param string model: the image index of the astropy fits object

    :return: u matrix and b vector
    '''
    try:
        from umatrix_routine import umatrix_construction, umatrix_bvector_construction, bvector_construction
        from umatrix_routine import umatrix_construction_nobkg, bvector_construction_nobkg

    except ImportError:
        log.info('Uncompiled cython code, please run setup.py: e.g.\n python setup.py build_ext --inplace')
        status = 'KO'
        report = 'Uncompiled cython code, please run setup.py: e.g.\n python setup.py build_ext --inplace'
        return status, report, reduction_metadata

    if np.shape(data_image) != np.shape(reference_image):
        return None

    if ker_size:
        if ker_size % 2 == 0:
            kernel_size = ker_size + 1
        else:
            kernel_size = ker_size

    if model_image == None:
        # assume that the model image is the reference...
        weights = noise_model(data_image, 1., 12., flat=None, initialize=True)
    else:
        weights = noise_model(model_image, 1., 12., flat=None, initialize=True)

    # Prepare/initialize indices, vectors and matrices
    pandq = []
    n_kernel = kernel_size * kernel_size
    ncount = 0
    half_kernel_size = int(int(kernel_size) / 2)
    for lidx in range(kernel_size):
        for midx in range(kernel_size):
            pandq.append((lidx - half_kernel_size, midx - half_kernel_size))

    u_matrix, b_vector = umatrix_bvector_construction(reference_image, data_image,
                                                      weights, pandq, n_kernel, kernel_size)

    return u_matrix, b_vector


def kernel_solution(u_matrix, b_vector, kernel_size, circular=True):
    '''
    reshape kernel solution for convolution and obtain uncertainty
    from lstsq solution

    :param object array: u_matrix
     :param object array: b_vector
    :return: kernel matrix
    '''
    # For better stability: solve the least square problem via np.linalg.lstsq
    # inv_umatrix = np.linalg.inv(u_matrix)
    # a_vector = np.dot(inv_umatrix, b_vector)
    # recalculate residuals to apply standard lstsq uncertainties
    lstsq_result = np.linalg.lstsq(np.array(u_matrix), np.array(b_vector), rcond=None)
    a_vector = lstsq_result[0]
    lstsq_fit = np.dot(np.array(u_matrix), a_vector)
    resid = np.array(b_vector) - lstsq_fit
    reduced_chisqr = np.sum(resid ** 2) / (float(kernel_size * kernel_size))
    lstsq_cov = np.dot(np.array(u_matrix).T, np.array(u_matrix)) * reduced_chisqr
    resivar = np.var(resid, ddof=0) * float(len(a_vector))
    # use pinv in order to stabilize calculation
    a_var = np.diag(np.linalg.pinv(lstsq_cov) * resivar)

    a_vector_err = np.sqrt(a_var)
    output_kernel = np.zeros(kernel_size * kernel_size, dtype=float)
    if len(a_vector) > kernel_size * kernel_size:
        output_kernel = a_vector[:-1]
    else:
        output_kernel = a_vector
    output_kernel = output_kernel.reshape((kernel_size, kernel_size))

    err_kernel = np.zeros(kernel_size * kernel_size, dtype=float)
    if len(a_vector) > kernel_size * kernel_size:
        err_kernel = a_vector_err[:-1]
        err_kernel = err_kernel.reshape((kernel_size, kernel_size))
    else:
        err_kernel = a_vector_err
        err_kernel = err_kernel.reshape((kernel_size, kernel_size))
    #
    if circular:
        xyc = int(kernel_size / 2)
        radius_square = (xyc) ** 2
        for idx in range(kernel_size):
            for jdx in range(kernel_size):
                if (idx - xyc) ** 2 + (jdx - xyc) ** 2 >= radius_square:
                    output_kernel[idx, jdx] = 0.
                    # err_kernel[idx, jdx] = 0.

    output_kernel_2 = np.flip(np.flip(output_kernel, 0), 1)
    err_kernel_2 = np.flip(np.flip(err_kernel, 0), 1)

    return output_kernel_2, a_vector[-1], err_kernel_2


def kernel_solution_pool(input_pars):
    umatrix_stamp = input_pars[0]
    reference_stamp = input_pars[1]
    data_image = input_pars[2]
    kernel_size = input_pars[3]
    return kernel_solution(umatrix_stamp, bvector_constant(reference_stamp, data_image, kernel_size), kernel_size,
                           circular=False)


def difference_image_single_iteration(ref_imagename, data_imagename, kernel_size,
                                      mask=None, max_adu=np.inf):
    '''
    Finding the difference image for a given reference and data image
    for a defined kernel size without iteration or image subdivision

    :param object string: data image name
    :param object string: reference image name
    :param object integer kernel size: edge length of the kernel in px
    :param object image: bad pixel mask
    :param object integer: maximum allowed adu on images

    :return: u matrix and b vector
    '''

    ref_image, data_image, bright_mask = read_images(
        ref_imagename, data_imagename, kernel_size, max_adu)
    # construct U matrix
    n_kernel = kernel_size * kernel_size

    u_matrix, b_vector = naive_u_matrix(
        data_image, ref_image, kernel_size, model_image=None)

    output_kernel_2, bkg_kernel, err_kernel = kernel_solution(u_matrix, b_vector, kernel_size, circular=True)
    model_image = convolve2d(ref_image, output_kernel_2, mode='same')

    difference_image = model_image - data_image + bkg_kernel
    difference_image[bright_mask] = 0.

    difference_image[kernel_size:-kernel_size, kernel_size:-kernel_size] = np.array(difference_image, float)
    return difference_image, output_kernel, bkg_kernel


def difference_image_subimages(ref_imagename, data_imagename,
                               kernel_size, subimage_shape, overlap=0., mask=None, max_adu=np.inf):
    '''
    The overlap parameter follows the original DANDIA definition, i.e.
    it is applied to each dimension, i.e. 0.1 increases the subimage in
    x an y direction by (2*overlap + 1) * int(x_image_size/n_divisions)
    the subimage_element contains 4 entries: [x_divisions, y_divisions,
    x_selection, y_selection]
    '''
    ref_image, data_image, bright_mask = read_images_for_substamps(ref_imagename, data_imagename, kernel_size, max_adu)
    # allocate target image
    difference_image = np.zeros(np.shape(ref_image))
    # call via multiprocessing
    subimages_args = []
    for idx in range(subimage_shape[0]):
        for jdx in range(subimage_shape[1]):
            print('Solving for subimage ', [idx + 1, jdx + 1], ' of ', subimage_shape)
            x_shape, y_shape = np.shape(ref_image)
            x_subsize, y_subsize = x_shape / subimage_shape[0], y_shape / subimage_shape[1]
            subimage_element = subimage_shape + [idx, jdx]

            ref_image_substamp = ref_image[subimage_element[2] * x_subsize: (subimage_element[2] + 1) * x_subsize,
                                 subimage_element[3] * y_subsize: (subimage_element[3] + 1) * y_subsize]
            data_image_substamp = data_image[subimage_element[2] * x_subsize: (subimage_element[2] + 1) * x_subsize,
                                  subimage_element[3] * y_subsize: (subimage_element[3] + 1) * y_subsize]
            bright_mask_substamp = bright_mask[subimage_element[2] * x_subsize: (subimage_element[2] + 1) * x_subsize,
                                   subimage_element[3] * y_subsize: (subimage_element[3] + 1) * y_subsize]

            # extend image size for convolution and kernel solution
            data_substamp_extended = np.zeros((np.shape(data_image_substamp)[0] + 2 * kernel_size,
                                               np.shape(data_image_substamp)[1] + 2 * kernel_size))
            ref_substamp_extended = np.zeros((np.shape(ref_image_substamp)[0] + 2 * kernel_size,
                                              np.shape(ref_image_substamp)[1] + 2 * kernel_size))
            mask_substamp_extended = np.zeros((np.shape(ref_image_substamp)[0] + 2 * kernel_size,
                                               np.shape(ref_image_substamp)[1] + 2 * kernel_size))
            mask_substamp_extended = mask_substamp_extended > 0

            data_substamp_extended[kernel_size:-kernel_size,
            kernel_size:-kernel_size] = np.array(data_image_substamp, float)
            ref_substamp_extended[kernel_size:-kernel_size,
            kernel_size:-kernel_size] = np.array(ref_image_substamp, float)
            mask_substamp_extended[kernel_size:-kernel_size,
            kernel_size:-kernel_size] = np.array(bright_mask_substamp, float)
            # difference_subimage = substamp_difference_image(ref_substamp_extended, data_substamp_extended,
            #                                                mask_substamp_extended, kernel_size, subimage_element, np.shape(ref_image))
            difference_image[subimage_element[2] * x_subsize: (subimage_element[2] + 1) * x_subsize,
            subimage_element[3] * y_subsize: (subimage_element[3] + 1) * y_subsize] = difference_subimage
