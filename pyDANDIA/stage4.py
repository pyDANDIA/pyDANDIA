######################################################################
#
# stage4.py - Fourth stage of the pipeline. Align images to the reference
# image using autocorellation

#
# dependencies:
#      numpy 1.8+
#      astropy 1.0+
######################################################################
import matplotlib

matplotlib.use('tkagg')
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
# from ccdproc import cosmicray_lacosmic
from astropy.table import Table
from photutils import datasets
from photutils import DAOStarFinder
from skimage import transform as tf
from skimage import data
from skimage import img_as_float64
from skimage.transform import resize
from pyDANDIA import config_utils
import scipy.optimize as so
import scipy.ndimage as sn
from skimage.measure import ransac
import skimage.feature as sf
from photutils import centroid_com
import matplotlib.pyplot as plt

from pyDANDIA import metadata
from pyDANDIA import logs
from pyDANDIA import convolution
from pyDANDIA import psf
from pyDANDIA import quality_control
from skimage.feature import register_translation

from skimage.feature import (ORB, match_descriptors,
                             plot_matches)
import itertools
import matplotlib.pyplot as plt

class PolyTF_4(tf.PolynomialTransform):
    def estimate(*data):
        return tf.PolynomialTransform.estimate(*data, order=2)

def polyfit2d(x, y, z, order=3,errors=None):
    import math

    ncols = int(math.factorial(order+2)/(math.factorial(order)*2))
    G = np.zeros((x.size, ncols))

    k = 0
    for j in range(order+1):
        for i in range(j+1):
            G[:,k] = x**(j-i) * y**i
            k+=1

    if errors is not None:
        G *= 1/errors[:,np.newaxis]
        Z = z*1/errors
    else:
        Z=z
    m, _, _, _ = np.linalg.lstsq(G, Z)
    return m

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
                                                                   stage_number=4, rerun_all=False, log=log)
    image_red_status = reduction_metadata.fetch_image_status(4)

    if len(all_images) > 0:
        try:
            reference_image_name = reduction_metadata.data_architecture[1]['REF_IMAGE'].data[0]
            reference_image_directory = reduction_metadata.data_architecture[1]['REF_PATH'].data[0]
            data_image_directory = reduction_metadata.data_architecture[1]['IMAGES_PATH'][0]
            ref_row_index = np.where(reduction_metadata.images_stats[1]['IM_NAME'] == str(
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
            reference_image = open_an_image(setup, reference_image_directory, reference_image_name, log, image_index=0)
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
            target_image = open_an_image(setup, images_directory, new_image, log, image_index=0)

#            try:

            x_new_center, y_new_center, x_shift, y_shift = find_x_y_shifts_from_the_reference_image(setup,
                                                                                                        reference_image,
                                                                                                        target_image,
                                                                                                        edgefraction=0.5,
                                                                                                        log=None)

            data.append([new_image, x_shift, y_shift])
            logs.ifverbose(log, setup,
                               'I found the image translation ('+str(x_shift)+','+str(y_shift)+') to the reference for frame:' + new_image)

#            except:

#                logs.ifverbose(log, setup,
#                               'WARNING: I can not find the image translation to the reference for frame:' + new_image)

#                data.append([new_image, None, None])

        if ('SHIFT_X' in reduction_metadata.images_stats[1].keys()) and (
                'SHIFT_Y' in reduction_metadata.images_stats[1].keys()):

            for index in range(len(data)):
                target_image = data[index][0]
                x_shift = data[index][1]
                y_shift = data[index][2]
                row_index = np.where(reduction_metadata.images_stats[1]['IM_NAME'].data == target_image)[0][0]
                reduction_metadata.update_a_cell_to_layer('images_stats', row_index, 'SHIFT_X', x_shift)
                reduction_metadata.update_a_cell_to_layer('images_stats', row_index, 'SHIFT_Y', y_shift)
                logs.ifverbose(log, setup,
                               'Updated metadata for image: ' + target_image)
        else:
            logs.ifverbose(log, setup,
                           'I have to construct SHIFT_X and SHIFT_Y columns')

            #sorted_data = np.copy(data)

            sorted_data = [['None',0,0]]*len(image_red_status)

            for index in range(len(data)):
                target_image = data[index][0]
                row_index = np.where(reduction_metadata.images_stats[1]['IM_NAME'].data == target_image)[0][0]
                sorted_data[row_index] = data[index]

            sorted_data = np.array(sorted_data)
            column_format = 'float'
            column_unit = 'pix'
            reduction_metadata.add_column_to_layer('images_stats', 'SHIFT_X', sorted_data[:, 1],
                                                   new_column_format=column_format,
                                                   new_column_unit=column_unit)

            reduction_metadata.add_column_to_layer('images_stats', 'SHIFT_Y', sorted_data[:, 2],
                                                   new_column_format=column_format,
                                                   new_column_unit=column_unit)

        (new_images, image_red_status) = quality_control.verify_image_shifts(new_images,
                                                    data, image_red_status)

        px_scale = float(reduction_metadata.reduction_parameters[1]['PIX_SCALE'])
        #resample_image(new_images, reference_image_name, reference_image_directory, reduction_metadata, setup,
        #               data_image_directory, resampled_directory_path, ref_row_index, px_scale, log=log,
        #               mask_extension_in=3)
        image_red_status = resample_image_stamps(new_images, reference_image_name, reference_image_directory, reduction_metadata, setup,
                       data_image_directory, resampled_directory_path, ref_row_index, px_scale,
                       image_red_status, log=log, mask_extension_in=-1)

        #image_red_status = metadata.set_image_red_status(image_red_status,'1',image_list=new_images)
        reduction_metadata.update_reduction_metadata_reduction_status_dict(image_red_status,
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

    #reduce_template = reference_image[
    #                 x_center - half_x:x_center + half_x, y_center - half_y:y_center + half_y]

    #reduce_image = target_image[
    #               x_center - half_x:x_center + half_x, y_center - half_y:y_center + half_y]
    # x_shift, y_shift = correlation_shift(reduce_template, reduce_image)
    reduce_image = target_image
    reduce_template = reference_image
    from skimage.feature import register_translation
    shifts, errors, phasediff = register_translation(reduce_template, reduce_image, 10)

    x_shift = shifts[1]
    y_shift = shifts[0]

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


def mutual_information(hgram):
    """ Mutual information for joint histogram
    """
    # Convert bins counts to probability values
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1)  # marginal for x over y
    py = np.sum(pxy, axis=0)  # marginal for y over x
    px_py = px[:, None] * py[None, :]  # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0  # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))


def correlation_coefficient(patch1, patch2):
    product = np.mean((patch1 - patch1.mean()) * (patch2 - patch2.mean()))
    stds = patch1.std() * patch2.std()
    if stds == 0:
        return 0
    else:
        product /= stds
        return product

def corr2_coeff(A,B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:,None]
    B_mB = B - B.mean(1)[:,None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1);
    ssB = (B_mB**2).sum(1);

    # Finally get corr coeff
    return np.dot(A_mA,B_mB.T)/np.sqrt(np.dot(ssA[:,None],ssB[None]))

def quick_pos_fit2(params, reference, data, mask, tform):

    tform.params = params.reshape(3, 3)
    model =tf.warp(data, tform, output_shape=data.shape, order=1, mode='constant',
                          cval=0, clip=False, preserve_range=True)
    model_mask = tf.warp(mask, tform, output_shape=data.shape, order=1, mode='constant',
                          cval=1, clip=False, preserve_range=True).astype(bool)

    #corr = np.corrcoef(reference.ravel(), model.ravel())[0, 1]
    #print(corr)

    #corr2 = corr2_coeff(reference, model)
    #print(corr2)

    #sh_row, sh_col = reference.shape
    #import pdb;
    #pdb.set_trace()
    #mse = np.sum((reference - model) ** 2) / (reference.shape[0] * model.shape[1])
   # print(mse)
    norm_cross_corr = np.corrcoef(reference[~model_mask.astype(bool)],model[~model_mask.astype(bool)])[0,1]
    print(norm_cross_corr)
    if np.isnan(norm_cross_corr):
        return np.inf
    return -norm_cross_corr

def quick_pos_fit(params, pts1, pts2, e_pts1):
    model = point_transformation(params, pts1)

    distance = (pts2[:, 0] - model[0]) ** 2 + (pts2[:, 1] - model[1]) ** 2

    return distance / e_pts1 ** 2


def point_transformation(params, pts1):
    # import pdb;
    # pdb.set_trace()
    rota = tf.ProjectiveTransform
    rota.params = params.reshape(3, 3)
    model = np.dot(rota.params, pts1.T)

    return model



def extract_catalog(reduction_metadata, data_image, row_index, log):


    central_region_x, central_region_y = np.shape(data_image)
    center_x, center_y = int(central_region_x / 2), int(central_region_y / 2)

    mean_data, median_data, std_data = sigma_clipped_stats(
        data_image, sigma=3.0, maxiters=5)
    data_fwhm = reduction_metadata.images_stats[1][row_index]['FWHM']
    data_sigma_x = reduction_metadata.images_stats[1][row_index]['SIGMA_X']
    data_sigma_y = reduction_metadata.images_stats[1][row_index]['SIGMA_Y']
    daofind2 = DAOStarFinder(fwhm=data_fwhm,
                             ratio=min(data_sigma_x, data_sigma_y) / max(data_sigma_x, data_sigma_y),
                             threshold=3. * std_data, exclude_border=True)

    # Error handling for cases where DAOfind crashes.  This seems to happen for
    # images with strong gradients in the sky background, possibly producing
    # an excessive number of false detections.
    try:
        data_sources = daofind2.find_stars(data_image - median_data)
    except MemoryError:
        if log!=None:
            log.info(' -> ERROR: DAOfind produced a MemoryError when attempting to extract this image catalog')
        data_sources = None

    return data_sources, data_fwhm


def crossmatch_catalogs(ref_sources, data_sources, x_shift = 0 ,y_shift = 0):


    data_sources_x, data_sources_y = np.copy(data_sources['xcentroid'].data), np.copy(
        data_sources['ycentroid'].data)

    # correct for shift to facilitate cross-match
    data_sources_x -= x_shift
    data_sources_y -= y_shift

    pts1, pts2, matching = reformat_catalog2(np.c_[ref_sources['xcentroid'], ref_sources['ycentroid']], np.c_[
        data_sources['xcentroid'] - x_shift, data_sources['ycentroid'] - y_shift],
                                             distance_threshold=8)
    pts1 = np.c_[data_sources_x + x_shift, data_sources_y + y_shift][matching[:, 1]]
    pts2[:, 0] -= 0
    pts2[:, 1] -= 0
    order = ref_sources['flux'][matching[:, 0]].argsort()[::-1]

    pts1 = pts1[order][:]
    pts2 = pts2[order][:]

    # model_robust, inliers = ransac(( pts1,pts2), tf. AffineTransform,min_samples=10, residual_threshold=0.05, max_trials=300)
    pts1 = np.c_[pts1, [1] * len(pts1)]
    pts2 = np.c_[pts2, [1] * len(pts2)]

    e_pos = 0.6  / data_sources['flux'][matching[:, 1]][order][:].data ** 0.5 # Kozlowski 2006

    return pts1,pts2,e_pos

def refine_positions(image,positions):

    Y,X = np.ogrid[:image.shape[0], :image.shape[1]]

    fake_image = np.zeros(image.shape)

    refine_positions = np.copy(positions)
    for idx,pos in enumerate(positions):

        dist_from_center = ((X - pos[1]) ** 2 + (Y - pos[0]) ** 2)

        mask = dist_from_center<pos[2]**2

        fake_image[mask] = image[mask]

        com = centroid_com(fake_image)

        fake_image[mask] = 0

        refine_positions[idx,:2] = com[::-1]

    return refine_positions


def refine_positions2(image,positions):

    Y,X = np.mgrid[:image.shape[0], :image.shape[1]]


    refine_positions = []
    for idx,pos in enumerate(positions):


        stamp = image[int(np.round(pos[1]))-1:int(np.round(pos[1]))+2,int(np.round(pos[0]))-1:int(np.round(pos[0]))+2]

        weight_x = np.sum(stamp*X[int(np.round(pos[1]))-1:int(np.round(pos[1]))+2,int(np.round(pos[0]))-1:int(np.round(pos[0]))+2])/np.sum(stamp)
        weight_y = np.sum(stamp*Y[int(np.round(pos[1]))-1:int(np.round(pos[1]))+2,int(np.round(pos[0]))-1:int(np.round(pos[0]))+2])/np.sum(stamp)
        refine_positions.append([weight_x,weight_y,1])

    return refine_positions

def resample_image(new_images, reference_image_name, reference_image_directory, reduction_metadata, setup,
                   data_image_directory, resampled_directory_path, ref_row_index, px_scale, log=None,
                   mask_extension_in=-1):
    from skimage.feature import ORB, match_descriptors, plot_matches
    from skimage.measure import ransac
    import sep

    if len(new_images) > 0:

        reference_image_hdu = fits.open(os.path.join(reference_image_directory, reference_image_name), memmap=True)
        reference_image = reference_image_hdu[0].data

    ref_sources,ref_fwhm = extract_catalog(reduction_metadata, reference_image, ref_row_index, log)

    #pts_reference = sf.blob_log(reference_image, min_sigma=2, max_sigma=5, threshold=1)
    #pts_reference = refine_positions(reference_image, pts_reference)
    master_mask = 0

    for new_image in new_images:
        print(new_image)

        row_index = np.where(reduction_metadata.images_stats[1]['IM_NAME'] == new_image)[0][0]
        x_shift, y_shift = -reduction_metadata.images_stats[1][row_index]['SHIFT_X'], - \
            reduction_metadata.images_stats[1][row_index]['SHIFT_Y']

        data_image_hdu = fits.open(os.path.join(data_image_directory, new_image), memmap=True)
        data_image = data_image_hdu[0].data

        if mask_extension_in > len(data_image_hdu) - 1 or mask_extension_in == -1:
            mask_extension = -1
            mask_image = np.zeros(data_image.shape)
        else:
            mask_extension = mask_extension_in
            mask_image = np.array(data_image_hdu[mask_extension].data, dtype=float)

        shifted_mask = np.copy(mask_image)
        shifted = np.copy(data_image)

        iteration = 0
        corr_ini = np.corrcoef(reference_image.ravel(), shifted.ravel())[0, 1]

        while iteration < 1:




            data_sources,data_fwhm = extract_catalog(reduction_metadata, shifted, row_index)
            #pts_data = sf.blob_doh(shifted, min_sigma=2, max_sigma=5, threshold=1)
            #pts_data = refine_positions(shifted,pts_data)

           # import pdb;
           # pdb.set_trace()
            try:
                if iteration > 0 :

                    x_shift = 0
                    y_shift = 0
                    center =  int(len(data_image)/2)
                    original_matrix = model_final.params

                else:
                    center = int(len(data_image)/2)
                    original_matrix = np.identity(3)

                pts_data,pts_reference,e_pos = crossmatch_catalogs(ref_sources, data_sources, x_shift,y_shift )

                pts_reference2 = np.copy(pts_reference)

                #model_robust, inliers = ransac((pts_data[:5000, :2], pts_reference2[:5000, :2]), tf.AffineTransform, min_samples=min(50,int(0.1*len(pts_data[:5000]))), residual_threshold=0.1,max_trials=1000)
                model_robust, inliers = ransac((pts_reference2[:5000, :2]-center, pts_data[:5000, :2]-center),  tf.AffineTransform, min_samples=min(50,int(0.1*len(pts_data[:5000]))), residual_threshold=0.1,max_trials=1000)


                print('Using Affine Transformation')

            except:

                model_final = tf.SimilarityTransform(translation=(-x_shift, -y_shift))
                print('Using XY shifts')
            try:


                model_params = np.dot(original_matrix,model_robust.params)
                model_final = model_robust
                model_final.params = model_params
                model_final.params[0,2] += center*(1-model_final.params[0,0]-model_final.params[0,1])
                model_final.params[1,2] += center*(1-model_final.params[1,0]-model_final.params[1,1])

                #shifted = tf.warp(data_image, inverse_map=model_final.inverse, output_shape=data_image.shape, order=3,
                #                 mode='constant', cval=np.median(data_image), clip=False, preserve_range=False)
                #shifted_mask = tf.warp(mask_image,inverse_map=model_final.inverse, output_shape=data_image.shape, order=3,
                #                 mode='constant', cval=1, clip=False, preserve_range=False)
                #import astroalign as aa - REMOVED
                #model_final, (s_list, t_list) = aa.find_transform(reference_image,data_image)

                #res = so.minimize(quick_pos_fit2,model_final.params.ravel(), args = ( reference_image, data_image, mask_image, model_final),method='Powell')
                #model_final.params = res['x'].reshape(3,3)
                #model_final.params[0,2] = model_final.params[0,2] + (model_final.params[0,0]+model_final.params[0,1])*center
                #model_final.params[1,2] = model_final.params[1,2] + (model_final.params[1,0]+model_final.params[1,1])*center
                shifted = tf.warp(data_image, inverse_map=model_final, output_shape=data_image.shape, order=1,
                                 mode='constant', cval=0, clip=True, preserve_range=True)
                shifted_mask = tf.warp(mask_image,inverse_map=model_final, output_shape=data_image.shape, order=1,
                                 mode='constant', cval=1, clip=False, preserve_range=False)

                #shifted = manual_transformation(model_final.params,[center,center], data_image)
                #shifted_mask = manual_transformation(model_final.params,[center,center], mask_image)
                corr = np.corrcoef(reference_image[~shifted_mask.astype(bool)],shifted[~shifted_mask.astype(bool)])[0,1]




            except:
                shifted_mask = np.zeros(np.shape(data_image))
                print('Similarity Transform has failed to produce parameters')
            #print(iteration,len(pts_data[inliers]),corr_ini,corr)

            iteration += 1
        #import astroalign as aa - REMOVED
        #aligned_image, footprint = aa.register(data_image, reference_image)
        #transf, (s_list, t_list) = aa.find_transform(data_image, reference_image)
        #corr2 = np.corrcoef(reference_image[~shifted_mask.astype(bool)],aligned_image[~shifted_mask.astype(bool)])[0,1]
        #import imreg_dft as ird
        #shifted = ird.similarity(reference_image,data_image, numiter=3)['timg']
        #shifted[shifted_mask.astype(bool)]=0
       #res = so.minimize(quick_pos_fit2,model_final.params.ravel(), args = ( reference_image, data_image, mask_image, model_final),method='Powell')

        mask = np.abs(shifted_mask)<10**-5
        shifted_mask[mask] = 0
        master_mask += shifted_mask

        if mask_extension > -1 and model_final != None:
            shifted_mask = tf.warp(shifted_mask, inverse_map=model_final.inverse, preserve_range=True)

        resampled_image_hdu = fits.PrimaryHDU(shifted)
        resampled_image_hdu.writeto(os.path.join(resampled_directory_path, new_image), overwrite=True)
        data_image_hdu.close()
    master_mask_hdu = fits.PrimaryHDU(master_mask)
    master_mask_hdu.writeto(os.path.join(reference_image_directory, 'master_mask.fits'), overwrite=True)


def resample_image_stamps(new_images, reference_image_name, reference_image_directory, reduction_metadata, setup,
                   data_image_directory, resampled_directory_path, ref_row_index, px_scale,
                   image_red_status, log=None, mask_extension_in=-1):
    from skimage.feature import ORB, match_descriptors, plot_matches
    from skimage.measure import ransac
    import sep
    list_of_stamps = reduction_metadata.stamps[1]['PIXEL_INDEX'].tolist()

    if len(new_images) > 0:

        reference_image_hdu = fits.open(os.path.join(reference_image_directory, reference_image_name), memmap=True)
        reference_image = np.copy(reference_image_hdu[0].data)

        mask_reference = reference_image_hdu[mask_extension_in].data.astype(bool)
    else:
        log.info('No images available to resample, halting.')

        raise ValueError('No images available to resample, halting.')

    ref_sources, ref_fwhm = extract_catalog(reduction_metadata, reference_image, ref_row_index, log)

    log.info('Starting image resampling')

    master_mask = 0

    for new_image in new_images:
        log.info('Resampling image '+new_image)

        row_index = np.where(reduction_metadata.images_stats[1]['IM_NAME'] == new_image)[0][0]
        x_shift, y_shift = -reduction_metadata.images_stats[1][row_index]['SHIFT_X'], - \
            reduction_metadata.images_stats[1][row_index]['SHIFT_Y']

        data_image_hdu = fits.open(os.path.join(data_image_directory, new_image), memmap=True)
        data_image = np.copy(data_image_hdu[0].data)

        mask_image = np.array(data_image_hdu[mask_extension_in].data, dtype=float)

        mask_status = quality_control.verify_mask_statistics(new_image,mask_image, log)

        if not mask_status:
            log.info('WARNING: Mask statistics indicate a problem with this image, skipping')
            image_red_status[new_image] = -1

        else:
            shifted_mask = np.copy(mask_image)
            shifted = np.copy(data_image)
            shifted_catalog = np.copy(data_image)
            shifted_catalog[mask_image.astype(bool)] = 0
            iteration = 0
            corr_ini = np.corrcoef(reference_image.ravel(), shifted.ravel())[0, 1]

            while iteration < 1:

                data_sources, data_fwhm = extract_catalog(reduction_metadata, shifted_catalog, row_index, log)

                try:

                    if iteration > 0:

                        x_shift = 0
                        y_shift = 0
                        original_matrix = model_final.params

                    else:
                        original_matrix = np.identity(3)

                    pts_data, pts_reference, e_pos = crossmatch_catalogs(ref_sources, data_sources, x_shift, y_shift)

                    pts_reference2 = np.copy(pts_reference)

                    model_robust, inliers = ransac((pts_reference2[:5000, :2] , pts_data[:5000, :2] ), tf.AffineTransform,
                                                   min_samples=min(50, int(0.1 * len(pts_data[:5000]))),
                                                   residual_threshold=0.05, max_trials=1000)

                    if len(pts_data[:5000][inliers])<10:
                        raise ValueError("Not enough matching stars! Switching to translation")
                    model_final = np.dot(original_matrix, model_robust.params)
                    log.info(' -> Using Affine Transformation')

                except:

                    model_final = tf.SimilarityTransform(translation=(x_shift, y_shift)).params
                    log.info(' -> Using XY shifts')
                try:




                    shifted = tf.warp(data_image, inverse_map=model_final, output_shape=data_image.shape, order=5,mode='constant', cval=0, clip=True, preserve_range=True)
                    shifted_mask = tf.warp(mask_image, inverse_map=model_final, output_shape=data_image.shape, order=1, mode='constant', cval=1, clip=True, preserve_range=True)

                    corr = np.corrcoef(reference_image[~shifted_mask.astype(bool)], shifted[~shifted_mask.astype(bool)])[0, 1]


                except:
                    shifted_mask = np.zeros(np.shape(data_image))
                    log.info(' -> Similarity Transform has failed to produce parameters')

                iteration += 1

            mask = np.abs(shifted_mask) < 10 ** -5
            shifted_mask[mask] = 0
            master_mask += shifted_mask

            #resample the stamps
            resample_directory = os.path.join(resampled_directory_path, new_image)
            try:
                os.mkdir(resample_directory)
            except:
                pass

            log.info(' -> Resampling image stamps')
            for stamp in list_of_stamps:
                try:
                    stamp_row = np.where(reduction_metadata.stamps[1]['PIXEL_INDEX'] == stamp)[0][0]
                    xmin = reduction_metadata.stamps[1][stamp_row]['X_MIN'].astype(int)
                    xmax = reduction_metadata.stamps[1][stamp_row]['X_MAX'].astype(int)
                    ymin = reduction_metadata.stamps[1][stamp_row]['Y_MIN'].astype(int)
                    ymax = reduction_metadata.stamps[1][stamp_row]['Y_MAX'].astype(int)

                    img = shifted[ymin:ymax, xmin:xmax]

                    stamp_mask = (ref_sources['xcentroid']<xmax) & (ref_sources['xcentroid']>xmin ) &\
                                 (ref_sources['ycentroid']<ymax) & (ref_sources['ycentroid']>ymin )
                    ref_stamps = ref_sources[stamp_mask]

                    data_stamps, stamps_fwhm = extract_catalog(reduction_metadata, img, row_index, log)

                    data_stamps['xcentroid'] += xmin
                    data_stamps['ycentroid'] += ymin

                    pts_data, pts_reference, e_pos = crossmatch_catalogs(ref_stamps, data_stamps,0,0)

                    model_stamp, inliers = ransac((pts_reference[:5000, :2] , pts_data[:5000, :2] ),
                                               tf.AffineTransform, min_samples=min(50, int(0.1 * len(pts_data[:5000]))),
                                               residual_threshold=0.05, max_trials=1000)

                    #save the warp matrices instead of images
                    np.save(os.path.join(resample_directory, 'warp_matrice_stamp_' + str(stamp) + '.npy'), model_stamp.params)

                except:

                   model_stamp = tf.SimilarityTransform(translation=(0,0))
                   np.save(os.path.join(resample_directory, 'warp_matrice_stamp_' + str(stamp) + '.npy'), model_stamp.params)




            #save the warp matrices instead of images
            np.save(os.path.join(resample_directory, 'warp_matrice_image.npy'), model_final)
            data_image_hdu.close()

            image_red_status[new_image] = 1

    mask = np.abs(master_mask) < 1.0
    master_mask[mask] = 0
    master_mask_hdu = fits.PrimaryHDU(master_mask)
    master_mask_hdu.writeto(os.path.join(reference_image_directory, 'master_mask.fits'), overwrite=True)

    return image_red_status


def warp_image(image_to_warp,warp_matrix):

    warp_image = tf.warp(image_to_warp, inverse_map=warp_matrix, output_shape=image_to_warp.shape, order=5,
                                  mode='constant', cval=0, clip=True, preserve_range=True)

    return warp_image

def reformat_catalog(idx_match, dist2d, ref_sources, data_sources, central_region_x, distance_threshold=1.5,
                     max_points=2000):
    pts1 = []
    pts2 = []
    matching = []
    for idxref in range(len(idx_match)):
        idx = idx_match[idxref]
        if dist2d[idxref].rad * float(central_region_x) < distance_threshold:
            ##the following criterion seems to be paradox - the least sharp targets ensure no artefact is picked up
            if float(ref_sources['sharpness'].data[idx]) < np.median(ref_sources['sharpness'].data):
                pts2.append(ref_sources['xcentroid'].data[idx])
                pts2.append(ref_sources['ycentroid'].data[idx])
                pts1.append(data_sources['xcentroid'].data[idxref])
                pts1.append(data_sources['ycentroid'].data[idxref])
                matching.append([idx, idxref])
                if len(pts1) > max_points:
                    break

    pts1 = np.array(pts1).reshape(int(len(pts1) / 2), 2)
    pts2 = np.array(pts2).reshape(int(len(pts2) / 2), 2)
    matching = np.array(matching)

    return pts1, pts2, matching


def reformat_catalog2(ref_catalog, data_catalog, distance_threshold=1.5):
    pts1 = []
    pts2 = []
    matching = []

    for idx, values in enumerate(data_catalog):

        distances = (ref_catalog[:, 0] - values[0]) ** 2 + (ref_catalog[:, 1] - values[1]) ** 2

        minimum = np.min(distances)
        if minimum < distance_threshold ** 2:
            pts1.append(values)
            ind = np.argmin(distances)
            pts2.append(ref_catalog[ind])
            matching.append([ind, idx])

    pts1 = np.array(pts1)
    pts2 = np.array(pts2)
    matching = np.array(matching)

    return pts1, pts2, matching


def fit_transformation(params, ref_image, data_image):
    im = manual_transformation(params, data_image)
    print(params)
    res = np.ravel(ref_image - im) ** 2
    # plt.imshow(ref_image-im)
    # plt.show()
    if np.max(im) == 0:
        return np.inf
    return np.sum(res)


def manual_transformation(matrix, center, data_image):
    scale_x = (matrix[0, 0] ** 2 + matrix[1, 0] ** 2) ** 0.5
    scale_y = (matrix[0, 1] ** 2 + matrix[1, 1] ** 2) ** 0.5

    rot = np.arctan2(matrix[1, 0], matrix[0, 0])

    shear = np.arctan2(-matrix[0, 1], matrix[1, 1]) - rot

    translation = matrix[0:2, 2]
    #print(rot, shear, translation)
    #translation[0] += center[0]
    #translation[1] += center[1]

    good_matrix = np.array([
     [scale_x * np.cos(rot), -scale_y * np.sin(rot + shear), -center[0]],
     [scale_x * np.sin(rot), scale_y * np.cos(rot + shear), -center[1]],
     [0, 0, 1]])


    good_matrix[0:2, 2] = translation
    ##matrix_center = np.array([
    ## [1,0,center[0]],
     ##[0,1,center[1]],
    ## [0, 0, 1]])
    good_matrix = np.linalg.inv(good_matrix)

    model = tf._warps_cy._warp_fast(data_image, good_matrix, output_shape=None, order=3, mode='constant', cval=0)
    # i#mport matplotlib.pyplot as plt
    # plt.imshow(rr)
    # plt.show()

    return model
