######################################################################
#
# stage4.py - Fourth stage of the pipeline. Align images to the reference
# image using autocorellation

import os
import sys

#
# dependencies:
#      numpy 1.8+
#      astropy 1.0+
######################################################################
import matplotlib as mpl
import numpy as np
import scipy.optimize as so
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.table import Table, Column
from photutils import DAOStarFinder
from skimage import transform as tf
from skimage.transform import rotate
from skimage.measure import ransac

#mpl.use('Agg')

from pyDANDIA import metadata
from pyDANDIA import logs
from pyDANDIA import convolution
from pyDANDIA import psf
from pyDANDIA import quality_control
from pyDANDIA import image_handling
from pyDANDIA import read_images_stage5
from skimage.registration import phase_cross_correlation
import scipy.spatial as ssp

def run_stage4(setup, **kwargs):
    """Main driver function to run stage 4: image alignement.
    This stage align the images to the reference frame!
    :param object setup : an instance of the ReductionSetup class. See
    reduction_control.py

    :return: [status, report, reduction_metadata], the stage4 status, the report,
    the metadata file
    :rtype: array_like

    """

    stage4_version = 'stage4 v0.2.1'

    log = logs.start_stage_log(setup.red_dir, 'stage4', version=stage4_version)
    log.info('Setup:\n' + setup.summary() + '\n')

    # find the metadata
    reduction_metadata = metadata.MetaData()
    reduction_metadata.load_all_metadata(setup.red_dir, 'pyDANDIA_metadata.fits')

    # find the images needed to treat

    all_images = reduction_metadata.find_all_images(setup, reduction_metadata,
                                                    os.path.join(setup.red_dir, 'data'),
                                                    log=log)

    new_images = reduction_metadata.find_images_need_to_be_process(setup, all_images,
                                                                   stage_number=4,
                                                                   rerun_all=False,
                                                                   log=log)
    image_red_status = reduction_metadata.fetch_image_status(4)

    if len(all_images) > 0:
        try:
            reference_image_name = \
                reduction_metadata.data_architecture[1]['REF_IMAGE'].data[0]
            reference_image_directory = \
                reduction_metadata.data_architecture[1]['REF_PATH'].data[0]
            data_image_directory = \
                reduction_metadata.data_architecture[1]['IMAGES_PATH'][0]
            ref_row_index = \
                np.where(reduction_metadata.images_stats[1]['IM_NAME'] == str(
                    reduction_metadata.data_architecture[1]['REF_IMAGE'][0]))[0][0]
            resampled_directory_path = os.path.join(setup.red_dir, 'resampled')
            if not os.path.exists(resampled_directory_path):
                os.mkdir(resampled_directory_path)
        except KeyError:
            logs.ifverbose(log, setup,
                           'I cannot find any reference image! Abort stage4')

            status = 'KO'
            report = 'No reference image found!'

            return status, report

    if len(new_images) > 0:

        # find the reference image
        try:
            ref_structure = image_handling.determine_image_struture(
                os.path.join(reference_image_directory, reference_image_name), log=log)
            reference_image = open_an_image(setup, reference_image_directory,
                                            reference_image_name, log,
                                            image_index=ref_structure['sci'])
            logs.ifverbose(log, setup,
                           'I found the reference frame:' + reference_image_name)
        except KeyError:
            logs.ifverbose(log, setup,
                           'I can not find any reference image! Abort stage4')

            status = 'KO'
            report = 'No reference frame found!'

            return status, report

        data = []
        images_directory = reduction_metadata.data_architecture[1]['IMAGES_PATH'].data[
            0]

        for new_image in new_images:
            image_structure = image_handling.determine_image_struture(
                os.path.join(images_directory, new_image), log=log)
            target_image = open_an_image(setup, images_directory, new_image, log,
                                         image_index=image_structure['sci'])

            try:

                #x_new_center, y_new_center, x_shift, y_shift = \
                #    find_x_y_shifts_from_the_reference_image(
                #        setup,
                #        reference_image.astype(float),
                #        target_image.astype(float),
                #        edgefraction=0.5,
                #        log=None)
                x_new_center, y_new_center, x_shift, y_shift = 0,0,0,0
                data.append([new_image, x_shift, y_shift])
                logs.ifverbose(log, setup,
                               'I found the image translation (' + str(
                                   x_shift) + ',' + str(
                                   y_shift) + ') to the reference for frame:' +
                               new_image)

            except:

                logs.ifverbose(log, setup,
                               'WARNING: I can not find the image translation to the '
                               'reference for frame:' + new_image)

                data.append([new_image, None, None])

        if ('SHIFT_X' in reduction_metadata.images_stats[1].keys()) and (
                'SHIFT_Y' in reduction_metadata.images_stats[1].keys()):

            for index in range(len(data)):
                target_image = data[index][0]
                x_shift = data[index][1]
                y_shift = data[index][2]
                row_index = np.where(
                    reduction_metadata.images_stats[1]['IM_NAME'].data == target_image)[
                    0][0]
                reduction_metadata.update_a_cell_to_layer('images_stats', row_index,
                                                          'SHIFT_X', x_shift)
                reduction_metadata.update_a_cell_to_layer('images_stats', row_index,
                                                          'SHIFT_Y', y_shift)
                logs.ifverbose(log, setup,
                               'Updated metadata for image: ' + target_image)
        else:
            logs.ifverbose(log, setup,
                           'I have to construct SHIFT_X and SHIFT_Y columns')

            # sorted_data = np.copy(data)

            sorted_data = [['None', 0, 0]] * len(image_red_status)

            for index in range(len(data)):
                target_image = data[index][0]
                try:
                    row_index = np.where(reduction_metadata.images_stats[1][
                                             'IM_NAME'].data == target_image)[0][0]
                    sorted_data[row_index] = data[index]
                except IndexError:
                    log.info(
                        'ERROR: Cannot find an entry for ' + target_image + ' in the '
                                                                            'IMAGES '
                                                                            'STATS '
                                                                            'table.  '
                                                                            'Re-run '
                                                                            'stages 0 '
                                                                            '& 1?')
                    raise IndexError(
                        'Cannot find an entry for ' + target_image + ' in the IMAGES '
                                                                     'STATS table.  '
                                                                     'Re-run stages 0 '
                                                                     '& 1?')

            sorted_data = np.array(sorted_data)
            column_format = 'float'
            column_unit = 'pix'
            reduction_metadata.add_column_to_layer('images_stats', 'SHIFT_X',
                                                   sorted_data[:, 1],
                                                   new_column_format=column_format,
                                                   new_column_unit=column_unit)

            reduction_metadata.add_column_to_layer('images_stats', 'SHIFT_Y',
                                                   sorted_data[:, 2],
                                                   new_column_format=column_format,
                                                   new_column_unit=column_unit)

        max_threshold = reduction_metadata.reduction_parameters[1]['MAX_SHIFTS'][0]
        (new_images, image_red_status) = quality_control.verify_image_shifts(new_images,
                                                                             data,
                                                                             image_red_status,
                                                                             threshold=max_threshold,
                                                                             log=log)

        if len(new_images) == 0:
            log.info('No new images remain to be processed')

        else:
            px_scale = float(reduction_metadata.reduction_parameters[1]['PIX_SCALE'])

            image_red_status = resample_image_stamps(new_images, reference_image_name,
                                                     reference_image_directory,
                                                     reduction_metadata, setup,
                                                     data_image_directory,
                                                     resampled_directory_path,
                                                     ref_row_index, px_scale,
                                                     image_red_status, log=log,
                                                     mask_extension_in=-1)

        # image_red_status = metadata.set_image_red_status(image_red_status,'1',
        # image_list=new_images)
        reduction_metadata.update_reduction_metadata_reduction_status_dict(
            image_red_status,
            stage_number=4, log=log)

    reduction_metadata.save_updated_metadata(
        reduction_metadata.data_architecture[1]['OUTPUT_DIRECTORY'][0],
        reduction_metadata.data_architecture[1]['METADATA_NAME'][0],
        log=log)

    logs.close_log(log)

    status = 'OK'
    report = 'Completed successfully'

    return status, report


def open_an_image(setup, image_directory, image_name, log, image_index=0):
    '''
    Simply open an image using astropy.io.fits

    :param object reduction_metadata: the metadata object
    :param string image_directory: the image name
    :param string image_name: the image name
    :param string image_index: the image index of the astropy fits object

    :param boolean verbose: switch to True to have more informations

    :return: the opened image
    :rtype: astropy.image object
    '''
    image_directory_path = image_directory

    logs.ifverbose(log, setup,
                   'Attempting to open image ' + os.path.join(image_directory_path,
                                                              image_name))

    try:

        image_data = fits.open(os.path.join(image_directory_path, image_name),
                               mmap=True)
        image_data = image_data[image_index]

        logs.ifverbose(log, setup, image_name + ' open : OK')

        return image_data.data

    except:
        logs.ifverbose(log, setup, image_name + ' open : not OK!')

        return None


def find_x_y_shifts_from_the_reference_image(setup, reference_image, target_image,
                                             edgefraction=0.5, log=None):
    """
    Found the pixel offset of the target image with the reference image

    :param object setup: the setup object
    :param object reference_image: the reference image data (i.e image.data)
    :param object target_image: the image data of interest (i.e image.data)
    :param float edgefraction: the percentage of images use for the shift computation
    (smaller = faster, [0,1])
    :param object log: the log object


    :return: [x_new_center, y_new_center, x_shift, y_shift], the new center and the
    correspondind shift of this image
    :rtype: array_like
    """

    reference_shape = reference_image.shape
    if reference_shape != target_image.shape:
        logs.ifverbose(log, setup,
                       'The reference image and the target image dimensions does not '
                       'match! Abort stage4')
        sys.exit(1)

    x_center = int(reference_shape[0] / 2)
    y_center = int(reference_shape[1] / 2)

    half_x = int(edgefraction * float(reference_shape[0]) / 2)
    half_y = int(edgefraction * float(reference_shape[1]) / 2)

    # reduce_template = reference_image[
    #                 x_center - half_x:x_center + half_x, y_center - half_y:y_center
    #                 + half_y]

    # reduce_image = target_image[
    #               x_center - half_x:x_center + half_x, y_center - half_y:y_center +
    #               half_y]
    # x_shift, y_shift = correlation_shift(reduce_template, reduce_image)
    reduce_image = target_image
    reduce_template = reference_image

    # shifts, errors, phasediff = phase_cross_correlation(reduce_template.astype(
    # float),reduce_image,upsample_factor=10)
    # x_shift = shifts[1]
    # y_shift = shifts[0]

    sol = rot_scale_translate(reduce_template.astype(float), reduce_image.astype(float))

    x_shift = sol[0]
    y_shift = sol[1]

    x_new_center = -x_shift + x_center
    y_new_center = -y_shift + y_center

    return x_new_center, y_new_center, x_shift, y_shift


def extract_catalog(reduction_metadata, data_image, row_index, log):

    central_region_x, central_region_y = np.shape(data_image)
    center_x, center_y = int(central_region_x / 2), int(central_region_y / 2)

    mean_data, median_data, std_data = sigma_clipped_stats(
        data_image, sigma=3.0, maxiters=5)
    data_fwhm = reduction_metadata.images_stats[1][row_index]['FWHM']
    data_sigma_x = reduction_metadata.images_stats[1][row_index]['SIGMA_X']
    data_sigma_y = reduction_metadata.images_stats[1][row_index]['SIGMA_Y']
    log.info(' -> Catalog extraction measured median_data=' + str(
        median_data) + ' std_data=' + str(std_data) + ' FWHM=' + str(data_fwhm))

    # daofind2 = DAOStarFinder(fwhm=data_fwhm,
    #                         ratio=min(data_sigma_x, data_sigma_y) / max(
    #                         data_sigma_x, data_sigma_y),
    #                         threshold=3. * std_data, exclude_border=True)

    daofind2 = DAOStarFinder(fwhm=data_fwhm, threshold=3. * std_data,
                             exclude_border=True)

    # Error handling for cases where DAOfind crashes.  This seems to happen for
    # images with strong gradients in the sky background, possibly producing
    # an excessive number of false detections.

    try:
        data_sources = daofind2.find_stars(data_image - median_data)
        log.info(' -> DAOfind identifed ' + str(len(data_sources)) + ' sources')
    except MemoryError:
        if log != None:
            log.info(
                ' -> ERROR: DAOfind produced a MemoryError when attempting to extract '
                'this image catalog; returning empty data_sources table')
        data_sources = Table([
            Column(name='id', data=np.array([])),
            Column(name='xcentroid', data=np.array([])),
            Column(name='ycentroid', data=np.array([])),
            Column(name='sharpness', data=np.array([])),
            Column(name='roundness1', data=np.array([])),
            Column(name='roundness2', data=np.array([])),
            Column(name='npix', data=np.array([])),
            Column(name='sky', data=np.array([])),
            Column(name='peak', data=np.array([])),
            Column(name='flux', data=np.array([])),
            Column(name='mag', data=np.array([]))
        ])

    data_sources = data_sources[data_sources['flux'].argsort()[::-1]]
    return data_sources, data_fwhm



def crossmatch_catalogs(ref_sources, data_sources, transform=None):


    if transform:

        data_points = np.c_[
            data_sources[:,0], data_sources[:,1], [1] * len(
                data_sources)]

        data_points = np.dot(transform.params, data_points.T)

        data = np.c_[data_points[0], data_points[1], data_sources[:,2]]

    else:

        data =  data_sources

    pts1, pts2, matching = reformat_catalog(ref_sources, data,
                                            distance_threshold=1)
    pts1 = data_sources[:,:2][matching[:,1]]

    pts1 = np.c_[pts1, [1] * len(pts1)]
    pts2 = np.c_[pts2, [1] * len(pts2)]

    e_pos_data = 0.6 / data_sources[
        matching[:, 1],2]  # [order][:].data ** 0.5 # Kozlowski 2006
    e_pos_ref = 0.6 / ref_sources[
        matching[:, 0],2]  # [order][:].data ** 0.5 # Kozlowski 2006

    return pts1, pts2, e_pos_data, e_pos_ref, matching

def output_shifted_mask(mask_data, image_path):
    red_dir = os.path.join(os.path.dirname(image_path), '..')
    mask_dir = os.path.join(red_dir, 'data_masks')
    if not os.path.isdir(mask_dir):
        os.mkdir(mask_dir)

    mask_path = os.path.join(mask_dir,
                             os.path.basename(image_path).replace('.fits',
                                                                  '_mask.fits'))

    mask_hdu = fits.PrimaryHDU(mask_data)
    mask_hdu.writeto(mask_path, overwrite=True)


def warp_image(image_to_warp, warp_matrix):
    # warp_image = tf.warp(image_to_warp/image_to_warp.max(),
    # inverse_map=warp_matrix, output_shape=image_to_warp.shape, order=1,
    #                             mode='constant', cval=0, clip=True,
    #                             preserve_range=True)*image_to_warp.max()

    warp_image = tf.warp(image_to_warp, warp_matrix, order=1,preserve_range=True)

    return warp_image

def reformat_catalog(ref_catalog, data_catalog, distance_threshold=1.5):
    pts1 = []
    pts2 = []
    matching = []
    matching_ref = []

    indexes = np.arange(0, len(ref_catalog))
    for idx, values in enumerate(data_catalog):

        distances = (ref_catalog[:, 0] - values[0]) ** 2 + (
                ref_catalog[:, 1] - values[1]) ** 2

        mask = distances < distance_threshold ** 2

        if len(distances[mask]):

            thematch = np.argmax(ref_catalog[mask, 2])

            ind = indexes[mask][thematch]

            if (ind not in matching_ref):
                pts1.append(values)
                pts2.append(ref_catalog[ind])
                matching.append([ind, idx])
                matching_ref.append(ind)

    pts1 = np.array(pts1)
    pts2 = np.array(pts2)
    matching = np.array(matching)

    return pts1, pts2, matching




def rot_scale_translate(ref_image, data_image):
    solutions = []
    for i in range(4):
        sol = phase_cross_correlation(ref_image[100:-100],
                                      rotate(data_image[100:-100], 90 * i),
                                      upsample_factor=10)
        # import pdb; pdb.set_trace()
        solutions.append([sol[0][1], sol[0][0], i * np.pi / 2])

    solutions = np.array(solutions)
    print(solutions)
    good_combination = (solutions[:, 0] ** 2 + solutions[:, 1] ** 2).argmin()

    sol = solutions[good_combination]

    sol[:2] = np.dot(
        [[np.cos(sol[2]), -np.sin(sol[2])], [np.sin(sol[2]), np.cos(sol[2])]], sol[:2])
    return sol


def resample_image_stamps(new_images, reference_image_name, reference_image_directory,
                          reduction_metadata, setup,
                          data_image_directory, resampled_directory_path, ref_row_index,
                          px_scale,
                          image_red_status, log=None, mask_extension_in=-1):

    # import sep
    list_of_stamps = reduction_metadata.stamps[1]['PIXEL_INDEX'].tolist()

    if len(new_images) > 0:
        ref_image_path = os.path.join(reference_image_directory, reference_image_name)
        ref_structure = image_handling.determine_image_struture(ref_image_path,
                                                                log=None)
        reference_image_hdu = fits.open(ref_image_path, memmap=True)
        reference_image = np.copy(
            reference_image_hdu[ref_structure['sci']].data.astype(float))

        # I think mask_extension_in == BPM.
        # mask_reference = reference_image_hdu[mask_extension_in].data.astype(bool)
        mask_reference = reference_image_hdu[
            ref_structure['pyDANDIA_pixel_mask']].data.astype(bool)
    else:
        log.info('No images available to resample, halting.')

        raise ValueError('No images available to resample, halting.')

    masked_ref = np.copy(reference_image)
    masked_ref[mask_reference.astype(bool)] = 0
    bkg_ref = read_images_stage5.background_mesh_perc(masked_ref, perc=30,
                                                      box_guess=300,
                                                      master_mask=mask_reference.astype(
                                                          bool))

    ref_sources, ref_fwhm = extract_catalog(reduction_metadata, masked_ref - bkg_ref,
                                            ref_row_index, log)

    sources_in_ref = np.c_[ref_sources['xcentroid'], ref_sources[
        'ycentroid'], ref_sources['flux']]

    log.info('Starting image resampling')
    # Is there a mask already?
    try:
        master_mask = \
            fits.open(os.path.join(reference_image_directory, 'master_mask.fits'))[
                0].data
    except:
        master_mask = 0

    for new_image in new_images:
        log.info('Resampling image ' + new_image)

        row_index = \
            np.where(reduction_metadata.images_stats[1]['IM_NAME'] == new_image)[0][0]
        x_shift, y_shift = -reduction_metadata.images_stats[1][row_index]['SHIFT_X'], \
            - \
                reduction_metadata.images_stats[1][row_index]['SHIFT_Y']

        image_path = os.path.join(data_image_directory, new_image)
        image_structure = image_handling.determine_image_struture(image_path, log=None)

        data_image_hdu = fits.open(image_path, memmap=True)
        data_image = data_image_hdu[image_structure['sci']].data.astype(float)

        if image_structure['pyDANDIA_pixel_mask'] != None:
            mask_image = np.array(
                data_image_hdu[image_structure['pyDANDIA_pixel_mask']].data,
                dtype=float)
        else:
            mask_image = np.zeros(data_image.shape)

        mask_status = quality_control.verify_mask_statistics(reduction_metadata,
                                                             new_image, mask_image, log)

        if not mask_status:
            log.info(
                'WARNING: Mask statistics indicate a problem with this image, skipping')
            image_red_status[new_image] = -1

        else:
            bkg_img = read_images_stage5.background_mesh_perc(data_image, perc=30,
                                                              box_guess=300,
                                                              master_mask=mask_image.astype(
                                                                  bool))
            model_final = align_two_images(masked_ref, data_image, bkg_img,
                                           sources_in_ref,
                                           reduction_metadata,
                                        row_index, x_shift,
                                        y_shift, log)
            try:

                shifted = warp_image(data_image, model_final)
                shifted_mask = tf.warp(mask_image, inverse_map=model_final,
                                       output_shape=data_image.shape, order=1,
                                       mode='constant', cval=1, clip=True,
                                       preserve_range=True)

                # corr = np.corrcoef(reference_image[~shifted_mask.astype(bool)],
                # shifted[~shifted_mask.astype(bool)])[0, 1]

                nrmse = (1 - np.sum(reference_image * shifted) ** 2 / np.sum(
                    reference_image ** 2) / np.sum(shifted ** 2)) ** 0.5
                log.info('Ultimate correlation :' + str(nrmse))
            except:
                shifted_mask = np.zeros(np.shape(data_image))
                log.info(
                    ' -> Similarity Transform has failed to produce parameters')


            mask = np.abs(shifted_mask) < 10 ** -5
            shifted_mask[mask] = 0
            master_mask += shifted_mask

            # Debugging:
            output_shifted_mask(shifted_mask, image_path)

            # resample the stamps
            resample_directory = os.path.join(resampled_directory_path, new_image)
            try:
                os.mkdir(resample_directory)
            except:
                pass

            log.info(' -> Resampling image stamps')

            for stamp in list_of_stamps:
                stamp_row = \
                    np.where(reduction_metadata.stamps[1]['PIXEL_INDEX'] == stamp)[
                        0][0]
                xmin = reduction_metadata.stamps[1][stamp_row]['X_MIN'].astype(int)
                xmax = reduction_metadata.stamps[1][stamp_row]['X_MAX'].astype(int)
                ymin = reduction_metadata.stamps[1][stamp_row]['Y_MIN'].astype(int)
                ymax = reduction_metadata.stamps[1][stamp_row]['Y_MAX'].astype(int)

                img = shifted[ymin:ymax, xmin:xmax]
                log.info('-> Taking section from ' + str(stamp) + ': ' + str(
                    xmin) + ':' + str(xmax) + ', ' + str(ymin) + ':' + str(ymax))

                sub_ref = reference_image[ymin:ymax, xmin:xmax].astype(float)

                #shifts, errors, phasedifff = phase_cross_correlation(sub_ref, img,
                #
                #                                                     upsample_factor=10)

                shifts = 0,0
                #guess = np.r_[[[shifts[0], shifts[1], 0]], [[1, 1, 0]]]
                # if np.max(np.abs(shifts))>1:
                #    import pdb; pdb.set_trace()

                log.info('-> Calculated shifts: ' + repr(shifts))

                stamp_mask = (ref_sources['xcentroid'] < xmax) & (
                        ref_sources['xcentroid'] > xmin) & \
                             (ref_sources['ycentroid'] < ymax) & (
                                     ref_sources['ycentroid'] > ymin)
                log.info('-> Derived stamp mask')

                ref_stamps = ref_sources[stamp_mask]
                ref_stamps['xcentroid'] -= xmin
                ref_stamps['ycentroid'] -= ymin
                bkg = read_images_stage5.background_mesh_perc(img,
                                                              master_mask=shifted_mask[
                                                                          ymin:ymax,
                                                                          xmin:xmax].astype(
                                                                  bool))

                sources_in_ref_stamps = np.c_[ref_stamps['xcentroid'], ref_stamps[
                    'ycentroid'], ref_stamps['flux']]

                model_stamp = align_two_images(sub_ref, img, bkg, sources_in_ref_stamps,
                                               reduction_metadata,
                                               row_index, -shifts[1],
                                               -shifts[0], log)


                np.save(os.path.join(resample_directory,
                                     'warp_matrice_stamp_' + str(stamp) + '.npy'),
                        model_stamp.params)

            # save the warp matrices instead of images
            np.save(os.path.join(resample_directory, 'warp_matrice_image.npy'),
                    model_final.params)
            data_image_hdu.close()

            image_red_status[new_image] = 1

    if type(master_mask) == int:
        raise ValueError('No valid mask data found in dataset')

    mask = np.abs(master_mask) < 1.0
    master_mask[mask] = 0
    master_mask_hdu = fits.PrimaryHDU(master_mask)
    master_mask_hdu.writeto(os.path.join(reference_image_directory, 'master_mask.fits'),
                            overwrite=True)

    return image_red_status


def align_two_images(ref, img, bkg_img, ref_catalog, reduction_metadata, row_index,
                     x_shift,
                     y_shift, log):

    try:


        data_sources, data_fwhm = extract_catalog(reduction_metadata,
                                                  img - bkg_img,
                                                  row_index, log)

        sources_in_im = np.c_[data_sources['xcentroid'], data_sources[
            'ycentroid'], data_sources['flux']]
        sources_in_im = sources_in_im[sources_in_im[:, 2].argsort()[::-1],]

        init_transform, corr = find_init_transform(ref, img,
                                                   ref_catalog,
                                                   sources_in_im)

        pts_data, pts_reference, e_pos_data, e_pos_ref, \
            matching = crossmatch_catalogs(
            ref_catalog, sources_in_im, init_transform)

        pts_reference2 = np.copy(pts_reference)

        model_robust, inliers = ransac(
            (pts_reference2[:5000, :2], pts_data[:5000, :2]),
            tf.AffineTransform,
            min_samples=min(50, int(0.1 * len(pts_data[:5000]))),
            residual_threshold=1, max_trials=1000)

        if len(pts_data[:5000][inliers]) < 10:
            raise ValueError(
                "Not enough matching stars! Switching to translation")

        model_final = model_robust
        log.info(' -> Using Affine Transformation:')
        log.info(repr(model_final))

    except:

        model_final = tf.SimilarityTransform(
            translation=(-x_shift, -y_shift))

        log.info(' -> Using XY shifts:')
        log.info(repr(model_final))

    return model_final


def fit_the_points(params, ref, im, imshape):
    # mat = np.array([[np.cos(params[0]),-np.sin(params[0]),params[1]],[np.sin(params[0]),np.cos(params[0]),params[2]],[0,0,1]])

    mat_tot = build_mat(params, imshape)
    new_points = np.dot(mat_tot, np.c_[im, [1] * len(im)].T)

    dist = ssp.distance.cdist(ref, new_points.T[:, :2])

    return np.sum(np.min(dist, axis=0))


def build_mat(params, imshape):
    # rotate around image centers
    center = imshape

    mat_shift = np.eye(3)
    mat_shift[0, 2] = params[1]
    mat_shift[1, 2] = params[2]

    mat_center = np.eye(3)

    mat_center[0, 2] = -center[0] / 2 + 0.5
    mat_center[1, 2] = -center[1] / 2 + 0.5

    mat_rota = np.array([[np.cos(params[0]), -np.sin(params[0]), 0],
                         [np.sin(params[0]), np.cos(params[0]), 0], [0, 0, 1]])

    mat_anticenter = np.eye(3)
    mat_anticenter[0, 2] = +center[0] / 2 - 0.5
    mat_anticenter[1, 2] = +center[1] / 2 - 0.5

    mat_tot = np.dot(mat_anticenter, np.dot(mat_shift, np.dot(mat_rota, mat_center)))


    return mat_tot


def find_init_transform(ref, im, source_ref, source_im):

    bounds = [[0, 2 * np.pi], [-1000, 1000], [-1000, 1000]]
    guess = so.differential_evolution(fit_the_points, bounds,
                                      args=(source_ref[:1000,:2], source_im[:1000,:2], ref.shape),
                                      strategy='rand1bin',maxiter=5000)

    guess['x'][0] = guess['x'][0]%(2*np.pi)
    reso = so.minimize(fit_the_points, guess['x'],
                       args=(source_ref[:1000,:2], source_im[:1000,:2], ref.shape), method='Powell')

    recenter = tf.AffineTransform(matrix=build_mat(reso['x'], ref.shape))
    recentred = tf.warp(im.astype(float), inverse_map=recenter.inverse,
                        output_shape=im.shape, order=3, mode='constant', cval=0,
                        clip=True, preserve_range=True)

    corr_old = np.corrcoef(ref.ravel(), im.ravel())[0, 1]
    corr = np.corrcoef(ref.ravel(), recentred.ravel())[0, 1]
    print(reso['x'], corr_old, corr)

    recenter = tf.AffineTransform(
        matrix=build_mat(reso['x'], ref.shape))

    if corr_old>corr:
        recenter = tf.AffineTransform(
            matrix=build_mat([0,0,0], ref.shape))
        corr = corr_old
    return recenter, corr


def polyfit2d(x, y, z, order=3, errors=None):
    #Might be of use later#

    import math

    ncols = int(math.factorial(order + 2) / (math.factorial(order) * 2))
    G = np.zeros((x.size, ncols))

    k = 0
    for j in range(order + 1):
        for i in range(j + 1):
            G[:, k] = x ** (j - i) * y ** i
            k += 1

    if errors is not None:
        G *= 1 / errors[:, np.newaxis]
        Z = z * 1 / errors
    else:
        Z = z
    m, _, _, _ = np.linalg.lstsq(G, Z, rcond=None)
    return m
