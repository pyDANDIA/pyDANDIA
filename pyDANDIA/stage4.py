######################################################################
#
# stage4.py - Fourth stage of the pipeline. Align images to the reference
# image using autocorellation

#
# dependencies:
#      numpy 1.8+
#      astropy 1.0+
######################################################################

import numpy as np
import os
from astropy.io import fits
import sys
from skimage import transform as tf
from skimage import data
from skimage import img_as_float64
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import match_coordinates_sky
from astropy.stats import sigma_clipped_stats
from ccdproc import cosmicray_lacosmic
from astropy.table import Table
from photutils import datasets
from photutils import DAOStarFinder
from skimage import transform as tf
from skimage import data
from skimage import img_as_float64
from skimage.transform import resize
from pyDANDIA import config_utils

from pyDANDIA import metadata
from pyDANDIA import logs
from pyDANDIA import convolution



def run_stage4(setup):
    """Main driver function to run stage 4: image alignement.
    This stage align the images to the reference frame!
    :param object setup : an instance of the ReductionSetup class. See reduction_control.py

    :return: [status, report, reduction_metadata], the stage4 status, the report, the metadata file
    :rtype: array_like

    """

    stage4_version = 'stage4 v0.1'

    log = logs.start_stage_log(setup.red_dir, 'stage4', version=stage4_version)
    log.info('Setup:\n' + setup.summary() + '\n')

    # find the metadata
    reduction_metadata = metadata.MetaData()
    reduction_metadata.load_all_metadata(setup.red_dir, 'pyDANDIA_metadata.fits')

    # find the images needed to treat
    all_images = reduction_metadata.find_all_images(setup, reduction_metadata,
                                                    os.path.join(setup.red_dir, 'data'), log=log)

    new_images = reduction_metadata.find_images_need_to_be_process(setup, all_images,
                                                                   stage_number=4, rerun_all=None, log=log)

    if len(all_images) > 0:
        try:
            reference_image_name = reduction_metadata.data_architecture[1]['REF_IMAGE'].data[0]
            reference_image_directory = reduction_metadata.data_architecture[1]['REF_PATH'].data[0]
            data_image_directory = reduction_metadata.data_architecture[1]['IMAGES_PATH'][0]
            ref_row_index = np.where(reduction_metadata.images_stats[1]['IM_NAME'] == str(reduction_metadata.data_architecture[1]['REF_IMAGE'][0]))[0][0]
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
            reference_image = open_an_image(setup, reference_image_directory, reference_image_name, image_index=0,
                                            log=None)
            logs.ifverbose(log, setup,
                           'I found the reference frame:' + reference_image_name)
        except KeyError:
            logs.ifverbose(log, setup,
                           'I can not find any reference image! Abort stage4')

            status = 'KO'
            report = 'No reference frame found!'

            return status, report

        data = []
        images_directory = reduction_metadata.data_architecture[1]['IMAGES_PATH'].data[0]
        for new_image in new_images:
            target_image = open_an_image(setup, images_directory, new_image, image_index=0, log=None)

            try:
                x_new_center, y_new_center, x_shift, y_shift = find_x_y_shifts_from_the_reference_image(setup,
                                                                                                        reference_image,
                                                                                                        target_image,
                                                                                                        edgefraction=0.5,
                                                                                                        log=None)

                data.append([new_image, x_shift, y_shift])
                logs.ifverbose(log, setup,
                               'I found the image translation to the reference for frame:' + new_image)

            except:

                logs.ifverbose(log, setup,
                               'I can not find the image translation to the reference for frame:' + new_image + '. Abort stage4!')

                status = 'KO'
                report = 'No shift  found for image:' + new_image + ' !'

                return status, report
     

        if ('SHIFT_X' in reduction_metadata.images_stats[1].keys()) and (
                    'SHIFT_Y' in reduction_metadata.images_stats[1].keys()):

            for index in range(len(data)):
                target_image = data[index][0]
                x_shift = data[index][1]
                y_shift = data[index][2]
                row_index = np.where(reduction_metadata.images_stats[1]['IM_NAME'].data == new_image)[0][0]
                reduction_metadata.update_a_cell_to_layer('images_stats', row_index, 'SHIFT_X', x_shift)
                reduction_metadata.update_a_cell_to_layer('images_stats', row_index, 'SHIFT_Y', y_shift)
                logs.ifverbose(log, setup,
                               'Updated metadata for image: ' + target_image)
        else:
            logs.ifverbose(log, setup,
                           'I have to construct SHIFT_X and SHIFT_Y columns')

            sorted_data = np.copy(data)

            for index in range(len(data)):
                target_image = data[index][0]

                row_index = np.where(reduction_metadata.images_stats[1]['IM_NAME'].data == new_image)[0][0]

                sorted_data[row_index] = data[index]

            column_format = 'int'
            column_unit = 'pix'
            reduction_metadata.add_column_to_layer('images_stats', 'SHIFT_X', sorted_data[:, 1],
                                                   new_column_format=column_format,
                                                   new_column_unit=column_unit)

            reduction_metadata.add_column_to_layer('images_stats', 'SHIFT_Y', sorted_data[:, 2],
                                                   new_column_format=column_format,
                                                   new_column_unit=column_unit)

    reduction_metadata.update_reduction_metadata_reduction_status(new_images, stage_number=4, status=1, log=log)

    px_scale = float(reduction_metadata.reduction_parameters[1]['PIX_SCALE']) 
    resample_image(all_images, reference_image_name, reference_image_directory, reduction_metadata, setup, data_image_directory, resampled_directory_path, ref_row_index, px_scale, log = log, mask_extension_in = 3)

    reduction_metadata.save_updated_metadata(
        reduction_metadata.data_architecture[1]['OUTPUT_DIRECTORY'][0],
        reduction_metadata.data_architecture[1]['METADATA_NAME'][0],
        log=log)

    logs.close_log(log)

    status = 'OK'
    report = 'Completed successfully'

    return status, report

def open_an_image(setup, image_directory, image_name,
                  image_index=0, log=None):
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
                   'Attempting to open image ' + os.path.join(image_directory_path, image_name))

    try:

        image_data = fits.open(os.path.join(image_directory_path, image_name),
                               mmap=True)
        image_data = image_data[image_index]

        logs.ifverbose(log, setup, image_name + ' open : OK')

        return image_data.data

    except:
        logs.ifverbose(log, setup, image_name + ' open : not OK!')

        return None


def find_x_y_shifts_from_the_reference_image(setup, reference_image, target_image, edgefraction=0.5, log=None):
    """
    Found the pixel offset of the target image with the reference image

    :param object setup: the setup object
    :param object reference_image: the reference image data (i.e image.data)
    :param object target_image: the image data of interest (i.e image.data)
    :param float edgefraction: the percentage of images use for the shift computation (smaller = faster, [0,1])
    :param object log: the log object


    :return: [x_new_center, y_new_center, x_shift, y_shift], the new center and the correspondind shift of this image
    :rtype: array_like
    """

    reference_shape = reference_image.shape
    if reference_shape != target_image.shape:
        logs.ifverbose(log, setup, 'The reference image and the target image dimensions does not match! Abort stage4')
        sys.exit(1)

    x_center = int(reference_shape[0] / 2)
    y_center = int(reference_shape[1] / 2)

    half_x = int(edgefraction * float(reference_shape[0]) / 2)
    half_y = int(edgefraction * float(reference_shape[1]) / 2)

    reduce_template = reference_image[
                      x_center - half_x:x_center + half_x, y_center - half_y:y_center + half_y]

    reduce_image = target_image[
                   x_center - half_x:x_center + half_x, y_center - half_y:y_center + half_y]
    x_shift, y_shift = correlation_shift(reduce_template, reduce_image)

    x_new_center = -x_shift + x_center
    y_new_center = -y_shift + y_center

    return x_new_center, y_new_center, x_shift, y_shift


def correlation_shift(reference_image, target_image):
    """
    Found the pixel offset of the target image with the reference image


    :param object reference_image: the reference image data (i.e image.data)
    :param object target_image: the image data of interest (i.e image.data)



    :return: data
    :rtype: array_like
    """

    reference_shape = reference_image.shape

    x_center = int(reference_shape[0] / 2)
    y_center = int(reference_shape[1] / 2)

    correlation = convolution.convolve_image_with_a_psf(np.matrix(reference_image),
                                            np.matrix(target_image), correlate=1)

    x_shift, y_shift = np.unravel_index(np.argmax(correlation), correlation.shape)

    good_shift_y = y_shift - y_center
    good_shift_x = x_shift - x_center
    return good_shift_y, good_shift_x



def resample_image(new_images, reference_image_name, reference_image_directory, reduction_metadata, setup, data_image_directory, resampled_directory_path, ref_row_index, px_scale, log = None, mask_extension_in = -1):  

    if len(new_images) > 0:
        reference_image_hdu = fits.open(os.path.join(reference_image_directory, reference_image_name),memmap=True)
        reference_image = reference_image_hdu[0].data

        reference_image = cosmicray_lacosmic(reference_image, sigclip=7., objlim = 7, satlevel =  float(reduction_metadata.reduction_parameters[1]['MAXVAL'][0]))[0]
#       reference_image, bright_reference_mask, reference_image_unmasked = open_reference(setup, reference_image_directory, reference_image_name, ref_extension = 0, log = log, central_crop = maxshift)
        #generate reference catalog
        central_region_x, central_region_y = np.shape(reference_image)
        center_x, center_y = int(central_region_x/2), int(central_region_y/2)
        mean_ref, median_ref, std_ref = sigma_clipped_stats(reference_image[center_x-int(central_region_x/4):center_x+int(central_region_x/4),center_y-int(central_region_y/4):center_y+int(central_region_y/4)], sigma = 3.0, iters = 5)    
        ref_fwhm_x = reduction_metadata.images_stats[1][ref_row_index]['FWHM_X'] 
        ref_fwhm_y = reduction_metadata.images_stats[1][ref_row_index]['FWHM_Y'] 
        ref_fwhm = (ref_fwhm_x**2 + ref_fwhm_y**2)**0.5
        daofind = DAOStarFinder(fwhm = max(ref_fwhm_x, ref_fwhm_y), ratio = min(ref_fwhm_x,ref_fwhm_y)/max(ref_fwhm_x,ref_fwhm_y), threshold = 6. * std_ref, exclude_border = True)    
        ref_sources = daofind(reference_image - median_ref)
        ref_sources_x, ref_sources_y = np.copy(ref_sources['xcentroid']), np.copy(ref_sources['ycentroid'])

    #ref_sources_x, ref_sources_y =    ref_sources_x[flxsort], ref_sources_y[flxsort]
    ref_catalog = SkyCoord(ref_sources_x/float(central_region_x)*u.rad, ref_sources_y/float(central_region_x)*u.rad) 

    for new_image in new_images:

        row_index = np.where(reduction_metadata.images_stats[1]['IM_NAME'] == new_image)[0][0]
        x_shift, y_shift = -reduction_metadata.images_stats[1][row_index]['SHIFT_X'], -reduction_metadata.images_stats[1][row_index]['SHIFT_Y'] 
        data_image_hdu = fits.open(os.path.join(data_image_directory, new_image),memmap=True)
        data_image = data_image_hdu[0].data
        data_image = cosmicray_lacosmic(data_image, sigclip=7., objlim = 7, satlevel =  float(reduction_metadata.reduction_parameters[1]['MAXVAL'][0]))[0]
        if mask_extension_in > len(data_image_hdu)-1 or mask_extension_in == -1:
            mask_extension = -1
        else:
            mask_extension = mask_extension_in
            mask_image = np.array(data_image_hdu[mask_extension].data,dtype=float)
        central_region_x, central_region_y = np.shape(data_image)
        center_x, center_y = int(central_region_x/2), int(central_region_y/2)
        mean_data, median_data, std_data = sigma_clipped_stats(data_image[center_x-int(central_region_x/4):center_x+int(central_region_x/4),center_y-int(central_region_y/4):center_y+int(central_region_y/4)], sigma = 3.0, iters = 5)    
        data_fwhm_x = reduction_metadata.images_stats[1][row_index]['FWHM_X'] 
        data_fwhm_y = reduction_metadata.images_stats[1][row_index]['FWHM_Y'] 
        data_fwhm = (ref_fwhm_x**2 + ref_fwhm_y**2)**0.5
        daofind = DAOStarFinder(fwhm = max(data_fwhm_x,data_fwhm_y),ratio = min(data_fwhm_x,data_fwhm_y)/max(data_fwhm_x,data_fwhm_y) , threshold = 6. * std_data, exclude_border = True)    
        data_sources = daofind(data_image - median_data)
        data_sources_x, data_sources_y = np.copy(data_sources['xcentroid'].data), np.copy(data_sources['ycentroid'].data)
        #data_sources_x, data_sources_y = data_sources_x[flxsort], data_sources_y[flxsort]
        #correct for shift to facilitate cross-match
        data_sources_x -= x_shift
        data_sources_y -= y_shift
        data_catalog = SkyCoord(data_sources_x/float(central_region_x)*u.rad, data_sources_y/float(central_region_x)*u.rad) 

        idx_match, dist2d, dist3d = match_coordinates_sky( data_catalog,ref_catalog, storekdtree='kdtree_sky')  
        #reformat points for scikit
        pts1, pts2 = reformat_catalog(idx_match, dist2d, ref_sources, data_sources,float(central_region_x), distance_threshold = 3., max_points = 1000)
        #resample image if there is a sufficient number of stars
        if len(pts1)>10:
            #permit translation again to be part of the least-squares problem
            tform = tf.estimate_transform('affine',pts1,pts2)
            np.allclose(tform.inverse(tform(pts1)), pts1)    
            maxv = np.max(data_image)
            shifted = tf.warp(img_as_float64(data_image/maxv),inverse_map = tform.inverse)
            if mask_extension>-1:
                maxvmask = np.max(mask_image)
                shifted_mask = tf.warp(img_as_float64(mask_image/maxvmask),inverse_map = tform.inverse)
                shifted_mask = shifted_mask * maxvmask
            shifted = maxv * np.array(shifted)
            for repeat in range(2):
                daofind = DAOStarFinder(fwhm = max(data_fwhm_x,data_fwhm_y),ratio = min(data_fwhm_x,data_fwhm_y)/max(data_fwhm_x,data_fwhm_y) , threshold = 6. * std_data, exclude_border = True)    
                data_sources = daofind(shifted - median_data)
                data_sources_x, data_sources_y = np.copy(data_sources['xcentroid'].data), np.copy(data_sources['ycentroid'].data)
                data_catalog = SkyCoord(data_sources_x/float(central_region_x)*u.rad, data_sources_y/float(central_region_x)*u.rad) 
                idx_match, dist2d, dist3d = match_coordinates_sky( data_catalog,ref_catalog, storekdtree='kdtree_sky')  
                #reformat points for scikit
                pts1, pts2 = reformat_catalog(idx_match, dist2d, ref_sources, data_sources, float(central_region_x),distance_threshold = 1.4, max_points = 4000)
            #resample image if there is a sufficient number of stars
                if len(pts1)>10:
                    #permit translation again to be part of the least-squares problem
                 
                    tform = tf.estimate_transform('affine',pts1,pts2)
                    np.allclose(tform.inverse(tform(pts1)), pts1)    
                    maxv = np.max(shifted)
                    shifted = tf.warp(img_as_float64(shifted/maxv),inverse_map = tform.inverse)
                    if mask_extension>-1:
                        maxvmask = np.max(shifted_mask)
                        shifted_mask = tf.warp(img_as_float64(shifted_mask/maxvmask),inverse_map = tform.inverse)
                        shifted_mask = maxvmask * shifted_mask
                    shifted = maxv * np.array(shifted)
        #cosmic ray rejection
        try:
         resampled_image_hdu = fits.PrimaryHDU(shifted)
         resampled_image_hdu.writeto(os.path.join(resampled_directory_path,new_image),overwrite = True)
         if mask_extension > -1:
             if os.path.exists(os.path.join(reduction_metadata.data_architecture[1]['REF_PATH'][0],'master_mask.fits')):
                 mask = fits.open(os.path.join(reduction_metadata.data_architecture[1]['REF_PATH'][0],'master_mask.fits'))
                 bpm_mask = shifted_mask > 0.
                 mask[0].data[bpm_mask] = mask[0].data[bpm_mask] + 1
                 mask[0].data  = np.array(mask[0].data,dtype=np.int)
                 mask.writeto(os.path.join(reduction_metadata.data_architecture[1]['REF_PATH'][0],'master_mask.fits'), overwrite = True)
             else:
                 bpm_mask = shifted_mask > 0.
                 mask_out = np.zeros(np.shape(shifted_mask))
                 mask_out[bpm_mask] = 1.
                 resampled_mask_hdu = fits.PrimaryHDU(np.array(mask_out,dtype=np.int))
                 resampled_mask_hdu.writeto(os.path.join(reduction_metadata.data_architecture[1]['REF_PATH'][0],'master_mask.fits'), overwrite = True)
                
        except Exception as e:
         import pdb; pdb.set_trace()
         resampled_image_hdu = fits.PrimaryHDU(data_image)
         resampled_image_hdu.writeto(os.path.join(resampled_directory_path,new_image),overwrite = True)
         if log is not None:
             logs.ifverbose(log, setup,'resampling failed:' + new_image + '. skipping! '+str(e))
         else:
             print(str(e))

def reformat_catalog(idx_match, dist2d, ref_sources, data_sources, central_region_x, distance_threshold = 1.5, max_points = 2000):
    pts1=[]
    pts2=[]
    for idxref in range(len(idx_match)):
        idx = idx_match[idxref]
        if dist2d[idxref].rad*float(central_region_x)< distance_threshold:
            if float(ref_sources['sharpness'].data[idx]) > np.median(ref_sources['sharpness'].data):
                pts2.append(ref_sources['xcentroid'].data[idx])
                pts2.append(ref_sources['ycentroid'].data[idx])
                pts1.append(data_sources['xcentroid'].data[idxref])
                pts1.append(data_sources['ycentroid'].data[idxref])
                if len(pts1)>max_points:
                    break
    
    pts1 = np.array(pts1).reshape(int(len(pts1)/2),2)
    pts2 = np.array(pts2).reshape(int(len(pts2)/2),2)               
    return pts1,pts2



