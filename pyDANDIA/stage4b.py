######################################################################
#
# stage4b.py - additional fourth stage of the pipeline. Find
# image distortion and warp images accordingly

#
# dependencies:
#      numpy 1.8+
#      astropy 1.0+
#      scipy 1.0+
######################################################################
import os, sys
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import match_coordinates_sky
from random import shuffle
from ccdproc import cosmicray_lacosmic

from astropy.table import Table
from photutils import datasets
from photutils import DAOStarFinder
import matplotlib.pyplot as plt
from skimage import transform as tf
from skimage import data
from skimage import img_as_float64
from skimage.transform import resize

from pyDANDIA import config_utils
from pyDANDIA import metadata
from pyDANDIA import logs

def run_stage4b(setup):
    """Main driver function to run stage 4b: image resampling
    This stage finds the kernel solution and (optionally) subtracts the model
    image
    :param object setup : an instance of the ReductionSetup class. See reduction_control.py

    :return: [status, report, reduction_metadata], stage4b status, report, 
     metadata file
    :rtype: array_like
    """

    stage4b_version = 'stage4b v0.1'

    log = logs.start_stage_log(setup.red_dir, 'stage4b', version=stage4b_version)
    log.info('Setup:\n' + setup.summary() + '\n')
    reduction_metadata = metadata.MetaData()
    reduction_metadata.load_all_metadata(setup.red_dir, 'pyDANDIA_metadata.fits')
    #determine kernel size based on maximum FWHM
    fwhm_max = 0.
    shift_max = 0
    for stats_entry in reduction_metadata.images_stats[1]:
        if float(stats_entry['FWHM_X'])> fwhm_max:
            fwhm_max = stats_entry['FWHM_X']
        if float(stats_entry['FWHM_Y'])> fwhm_max:
            fwhm_max = stats_entry['FWHM_Y']
        if abs(float(stats_entry['SHIFT_X']))> shift_max:
            shift_max = abs(float(stats_entry['SHIFT_X']))
        if abs(float(stats_entry['SHIFT_Y']))> shift_max:
            shift_max = abs(float(stats_entry['SHIFT_Y']))
    
    # find the images that need to be processed
    all_images = reduction_metadata.find_all_images(setup, reduction_metadata,
                                                    os.path.join(setup.red_dir, 'data'), log=log)

    new_images = reduction_metadata.find_images_need_to_be_process(setup, all_images,
                                                                   stage_number=5, rerun_all=None, log=log)
 
    resampled_directory_path = os.path.join(setup.red_dir, 'resampled')
    if not os.path.exists(resampled_directory_path):
        os.mkdir(resampled_directory_path)
    reduction_metadata.update_column_to_layer('data_architecture', 'RESAMPLED_PATH', resampled_directory_path)

    # difference images are written for verbosity level > 0 
    data_image_directory = reduction_metadata.data_architecture[1]['IMAGES_PATH'][0]
    ref_directory_path = '.'
    #For a quick image subtraction, pre-calculate a sufficiently large u_matrix
    #based on the largest FWHM and store it to disk -> needs config switch

    try:
        reference_image_name = str(reduction_metadata.data_architecture[1]['REF_IMAGE'][0])
        reference_image_directory = str(reduction_metadata.data_architecture[1]['REF_PATH'][0])
        ref_row_index = np.where(reduction_metadata.images_stats[1]['IM_NAME'] == str(reduction_metadata.data_architecture[1]['REF_IMAGE'][0]))[0][0]
        logs.ifverbose(log, setup,'Using reference image:' + reference_image_name)
    except Exception as e:
        log.ifverbose(log, setup,'Reference/Images ! Abort stage4b'+str(e))
        status = 'KO'
        report = 'No reference image found!'
        return status, report, reduction_metadata

    if not ('SHIFT_X' in reduction_metadata.images_stats[1].keys()) and (
            'SHIFT_Y' in reduction_metadata.images_stats[1].keys()):
        log.ifverbose(log, setup,'No xshift! run stage4 ! Abort stage4b')
        status = 'KO'
        report = 'No alignment data found!'
        return status, report, reduction_metadata
 
    px_scale = float(reduction_metadata.reduction_parameters[1]['PIX_SCALE']) 
    resample_image(new_images, reference_image_name, reference_image_directory, reduction_metadata, setup, data_image_directory, resampled_directory_path, ref_row_index, px_scale, log = log, mask_extension_in = 3)

    #append some metric for the kernel, perhaps its scale factor...
    reduction_metadata.update_reduction_metadata_reduction_status(new_images, stage_number=5, status=1, log = log)
    logs.close_log(log)
    status = 'OK'
    report = 'Completed successfully'
    return status, report


def resample_image(new_images, reference_image_name, reference_image_directory, reduction_metadata, setup, data_image_directory, resampled_directory_path, ref_row_index, px_scale, log = None, mask_extension_in = -1):  
    if len(new_images) > 0:
        reference_image_hdu = fits.open(os.path.join(reference_image_directory, reference_image_name),memmap=True)
        reference_image = reference_image_hdu[0].data

        reference_image = cosmicray_lacosmic(reference_image, sigclip=7., objlim = 7, satlevel =  float(reduction_metadata.reduction_parameters[1]['MAXVAL'][0]))[0]
#       reference_image, bright_reference_mask, reference_image_unmasked = open_reference(setup, reference_image_directory, reference_image_name, ref_extension = 0, log = log, central_crop = maxshift)
        #generate reference catalog
        central_region_x, central_region_y = np.shape(reference_image)
        center_x, center_y = central_region_x/2, central_region_y/2
        mean_ref, median_ref, std_ref = sigma_clipped_stats(reference_image[center_x-central_region_x/4:center_x+central_region_x/4,center_y-central_region_y/4:center_y+central_region_y/4], sigma = 3.0, iters = 5)    
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
        center_x, center_y = central_region_x/2, central_region_y/2
        mean_data, median_data, std_data = sigma_clipped_stats(data_image[center_x-central_region_x/4:center_x+central_region_x/4,center_y-central_region_y/4:center_y+central_region_y/4], sigma = 3.0, iters = 5)    
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
            for repeat in range(5):
                daofind = DAOStarFinder(fwhm = max(data_fwhm_x,data_fwhm_y),ratio = min(data_fwhm_x,data_fwhm_y)/max(data_fwhm_x,data_fwhm_y) , threshold = 6. * std_data, exclude_border = True)    
                data_sources = daofind(shifted - median_data)
                data_sources_x, data_sources_y = np.copy(data_sources['xcentroid'].data), np.copy(data_sources['ycentroid'].data)
                data_catalog = SkyCoord(data_sources_x/float(central_region_x)*u.rad, data_sources_y/float(central_region_x)*u.rad) 
                idx_match, dist2d, dist3d = match_coordinates_sky( data_catalog,ref_catalog, storekdtree='kdtree_sky')  
                #reformat points for scikit
                pts1, pts2 = reformat_catalog(idx_match, dist2d, ref_sources, data_sources, float(central_region_x),distance_threshold = 1.5, max_points = 2000)
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
            if float(ref_sources['sharpness'].data[idx]) > np.median(ref_sources['sharpness'].data) + np.std(ref_sources['sharpness'].data):
                pts2.append(ref_sources['xcentroid'].data[idx])
                pts2.append(ref_sources['ycentroid'].data[idx])
                pts1.append(data_sources['xcentroid'].data[idxref])
                pts1.append(data_sources['ycentroid'].data[idxref])
                if len(pts1)>max_points:
                    break
    pts1 = np.array(pts1).reshape((len(pts1)/2,2))
    pts2 = np.array(pts2).reshape((len(pts2)/2,2))               
    return pts1,pts2


