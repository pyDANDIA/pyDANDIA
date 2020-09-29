# -*- coding: utf-8 -*-
"""

"""
######################################################################
#
# stage2.py - Second stage of the pipeline.
#             Reads metadata and selects a reference image
#
# dependencies:
#      numpy 1.8+
#      astropy 1.0+
######################################################################

from operator import itemgetter
import os
import sys
import numpy as np

from scipy.ndimage.interpolation import shift

from astropy.io import fits
from astropy.table import Table
from astropy.stats import sigma_clipped_stats
from shutil import copyfile

from pyDANDIA import  config_utils
from pyDANDIA import  convolution
from pyDANDIA import stage0
from pyDANDIA import stage1

from pyDANDIA import  metadata
from pyDANDIA import  pixelmasks
from pyDANDIA import  logs
from pyDANDIA import  empirical_psf_simple
from pyDANDIA import image_handling


def run_stage2(setup, **kwargs):
    """Main driver function to run stage 2: reference selection.

    This stage is processing the metadata file, looks for the output of
    stages0 and stage1 and checks if a reference file already
    exists.

    It creates a reference frame based on the selection criteria
    defined in the configuration. If no such configuration exists, it
    falls back to a standard configuration.

    If stage1 has failed to produce output it selects a reference
    based on header information.

    It always re-runs when called, since it is a lightweight function
    """

    stage2_version = 'stage2 v0.1'

    log = logs.start_stage_log(setup.red_dir, 'stage2', version=stage2_version)
    log.info('Setup:\n' + setup.summary() + '\n')

    kwargs = get_default_config(kwargs,log)

    reduction_metadata = metadata.MetaData()

    # Load all metadata
    try:
        reduction_metadata.load_all_metadata(metadata_directory=setup.red_dir,
                                             metadata_name='pyDANDIA_metadata.fits')

    # Check data inventory on metadata
        log.info('stage2 has loaded the reduction metadata')
    except Exception as estr:
        log.info('Could not load metadata!' + repr(estr))
        status = 'FAILED'
        report = 'Loading metadata failed:' + repr(estr)
        return status, report

    try:
        n_images = len(reduction_metadata.images_stats)
    except AttributeError:
        log.info('stage2: data inventory missing.')
        status = 'FAILED'
        report = 'Data inventory (stage1) missing.'
        logs.close_log(log)
        return status, report

    # All parameters are part of metadata

    table_structure = [
        ['IMAGE_NAME', 'MOON_STATUS', 'RANKING_KEY'],
        ['S100', 'S100', 'float'],
        ['degree', None, None]
    ]

    all_images = reduction_metadata.find_all_images(setup, reduction_metadata, os.path.join(setup.red_dir, 'data'), log=log)
    image_red_status = reduction_metadata.fetch_image_status(2)

    reduction_metadata.create_a_new_layer(layer_name='reference_inventory',
                                          data_structure=table_structure,
                                          data_columns=None)

    log.info('Create reference frame inventory table in metadata')

    # Iterate over images that are in the stage inventory

    reference_ranking = []

    fwhm_max = 0.
    for stats_entry in reduction_metadata.images_stats[1]:
        if stats_entry[8] == 1:
            if float(stats_entry['FWHM'])> fwhm_max:
                fwhm_max = stats_entry['FWHM']
            if float(stats_entry['FWHM'])> fwhm_max:
                fwhm_max = stats_entry['FWHM']

    # taking filenames from headers_summary (stage1 change pending)
    filename_images = reduction_metadata.images_stats[1]['IM_NAME']
    data_image_directory = reduction_metadata.data_architecture[1]['IMAGES_PATH'][0]
    max_adu = float(reduction_metadata.reduction_parameters[1]['MAXVAL'][0])
    psf_size = int(4.*float(reduction_metadata.reduction_parameters[1]['KER_RAD'][0]) * fwhm_max)
    empirical_psf_flag = False

    if empirical_psf_flag == True:

        for stats_entry in reduction_metadata.images_stats[1]:
            if stats_entry[10] == 1:
                image_filename = stats_entry[0]
                row_idx = np.where(reduction_metadata.images_stats[1]['IM_NAME'] == image_filename)[0][0]
                moon_status = 'dark'
                # to be reactivated as soon as it is part of metadata
                if 'MOONFKEY' in reduction_metadata.headers_summary[1].keys() and 'MOONDKEY' in reduction_metadata.headers_summary[1].keys():
                    moon_status = moon_brightness_header(reduction_metadata.headers_summary[1],row_idx)

                fwhm_arcsec = float(stats_entry['FWHM']) * float(reduction_metadata.reduction_parameters[1]['PIX_SCALE'])

                # extract data inventory row for image and calculate sorting key
                # if a sufficient number of stars has been detected at s1 (40)
                if int(stats_entry['NSTARS'])>34 and fwhm_arcsec<3. and (not 'bright' in moon_status):
                    image_structure = image_handling.determine_image_struture(os.path.join(data_image_directory, image_filename), log=log)
                    hdulist = fits.open(os.path.join(data_image_directory, image_filename), memmap = True)
                    image = hdulist[image_structure['sci']].data
                    ranking_key = empirical_psf_simple.empirical_snr_subframe(image, psf_size, max_adu)
                    hdulist.close()
                    reference_ranking.append([image_filename, ranking_key])
                    entry = [image_filename, moon_status, ranking_key]
                    reduction_metadata.add_row_to_layer(key_layer='reference_inventory', new_row=entry)

    else:
        for stats_entry in reduction_metadata.images_stats[1]:
            if stats_entry[10] == 1 and kwargs['empirical_ranking'] == False:
                image_filename = stats_entry[0]
                row_idx = np.where(reduction_metadata.images_stats[1]['IM_NAME'] == image_filename)[0][0]
                moon_status = 'dark'
                # to be reactivated as soon as it is part of metadata
                if 'MOONFKEY' in reduction_metadata.headers_summary[1].keys() and 'MOONDKEY' in reduction_metadata.headers_summary[1].keys():
                    moon_status = moon_brightness_header(reduction_metadata.headers_summary[1],row_idx)

                fwhm_arcsec = float(stats_entry['FWHM']) * float(reduction_metadata.reduction_parameters[1]['PIX_SCALE'])

                # extract data inventory row for image and calculate sorting key
                # if a sufficient number of stars has been detected at s1 (40)
                if int(stats_entry['NSTARS'])>34 and fwhm_arcsec<3. and (not 'bright' in moon_status):
                    ranking_key = add_stage1_rank(reduction_metadata, stats_entry)
                    reference_ranking.append([image_filename, ranking_key])
                    entry = [image_filename, moon_status, ranking_key]
                    reduction_metadata.add_row_to_layer(key_layer='reference_inventory',
                                                        new_row=entry)


    #relax criteria...
    if reference_ranking == [] or kwargs['empirical_ranking']:
        log.info('No meaningful automatic selection can be made. Assigning empirical reference.')
        for stats_entry in reduction_metadata.images_stats[1]:
            if (stats_entry[10] == 1):
                image_filename = stats_entry[0]
                row_idx = np.where(reduction_metadata.images_stats[1]['IM_NAME'] == image_filename)[0][0]
                moon_status = 'dark'

                # extract data inventory row for image and calculate sorting key
                fwhm_value = float(stats_entry['FWHM'])
                data_directory_path = os.path.join(setup.red_dir, 'data')
                image_structure = image_handling.determine_image_struture(os.path.join(data_directory_path,image_filename), log=log)
                hl_data = fits.open(os.path.join(data_directory_path,image_filename))
                data = hl_data[image_structure['sci']].data
                mean, median, std = sigma_clipped_stats(data, sigma=3.0)
                fraction_3sig = float(len(np.where(data>3.*std+median)[1]))/data.size
                hl_data.close()
                ranking_key = 1/(1/(fraction_3sig**2) +fwhm_value**2 )
                reference_ranking.append([image_filename, ranking_key])
                entry = [image_filename, moon_status, ranking_key]
                reduction_metadata.add_row_to_layer(key_layer='reference_inventory',
                                                    new_row=entry)

    #import pdb; pdb.set_trace()
    #Etienne empirical ranking....
    if reference_ranking == [] or kwargs['empirical_ranking']:
        log.info('Etienne empirical ranking....')

        best_image = (reduction_metadata.images_stats[1]['NSTARS']/reduction_metadata.images_stats[1]['SKY']).argmax()
        stats_entry =  reduction_metadata.images_stats[1][best_image]

        image_filename = stats_entry[0]
        row_idx = np.where(reduction_metadata.images_stats[1]['IM_NAME'] == image_filename)[0][0]
        moon_status = 'dark'

        # extract data inventory row for image and calculate sorting key
        fwhm_value = float(stats_entry['FWHM'])
        data_directory_path = os.path.join(setup.red_dir, 'data')
        image_structure = image_handling.determine_image_struture(os.path.join(data_directory_path,image_filename), log=log)
        hl_data = fits.open(os.path.join(data_directory_path,image_filename))
        data = hl_data[image_structure['sci']].data
        mean, median, std = sigma_clipped_stats(data, sigma=3.0)
        fraction_3sig = float(len(np.where(data>3.*std+median)[1]))/data.size
        hl_data.close()
        ranking_key = 1/(1/(fraction_3sig**2) +fwhm_value**2 )
        reference_ranking.append([image_filename, ranking_key])
        entry = [image_filename, moon_status, ranking_key]
        reduction_metadata.add_row_to_layer(key_layer='reference_inventory',
                                            new_row=entry)

    # Save the updated layer to the metadata file
    reduction_metadata.save_a_layer_to_file(metadata_directory=setup.red_dir,
                                            metadata_name='pyDANDIA_metadata.fits',
                                            key_layer='reference_inventory')

    if reference_ranking != [] and kwargs['n_stack'] == 1:
        best_image = sorted(reference_ranking, key=itemgetter(1))[-1]
        ref_directory_path = os.path.join(setup.red_dir, 'ref')
        if not os.path.exists(ref_directory_path):
            os.mkdir(ref_directory_path)

        ref_img_path = os.path.join(
            str(reduction_metadata.data_architecture[1]['IMAGES_PATH'][0]), best_image[0])

        log.info('New reference '+best_image[0]+' in '+ref_img_path)


        try:
            copyfile(reduction_metadata.data_architecture[1]['IMAGES_PATH'][0]+'/'+best_image[0],ref_directory_path+'/'+best_image[0])
        except:
            log.info('WARNING: Copy ref failed: ',best_image[0])

        if not 'REF_PATH' in reduction_metadata.data_architecture[1].keys():
            reduction_metadata.add_column_to_layer('data_architecture',
                                                   'REF_PATH', [ref_directory_path],
                                                    new_column_format=None,
                                                    new_column_unit=None)
        else:
            reduction_metadata.update_a_cell_to_layer('data_architecture', 0,'REF_PATH', ref_directory_path)
        if  not 'REF_IMAGE' in reduction_metadata.data_architecture[1].keys():
            reduction_metadata.add_column_to_layer('data_architecture',
                                                   'REF_IMAGE', [os.path.basename(ref_img_path)],
                                                    new_column_format=None,
                                                   new_column_unit=None)
        else:
            reduction_metadata.update_a_cell_to_layer('data_architecture', 0,'REF_IMAGE', os.path.basename(ref_img_path))
        # Update the REDUCTION_STATUS table in metadata for stage 2

        reduction_metadata.calc_psf_radii()

        image_red_status = metadata.set_image_red_status(image_red_status,'1')

        reduction_metadata.update_reduction_metadata_reduction_status_dict(image_red_status,
                                                          stage_number=2, log=log)
        reduction_metadata.save_updated_metadata(metadata_directory=setup.red_dir,
                                                 metadata_name='pyDANDIA_metadata.fits')

        status = 'OK'
        report = 'Completed successfully'
        log.info('Updating metadata with info on new images...')
        logs.close_log(log)

        return status, report


    if reference_ranking != [] and kwargs['n_stack'] > 1 :
        best_image = sorted(reference_ranking, key=itemgetter(1))[-1]
        n_min_stack  = min(len(reference_ranking), kwargs['n_stack']+1)
        best_images = sorted(reference_ranking, key=itemgetter(1))[-n_min_stack:]
        ref_structure = image_handling.determine_image_struture(os.path.join(reduction_metadata.data_architecture[1]['IMAGES_PATH'][0],best_image[0]), log=log)
        ref_hdu = fits.open(os.path.join(reduction_metadata.data_architecture[1]['IMAGES_PATH'][0],best_image[0]))
        coadd = np.copy(ref_hdu[ref_structure['sci']].data)
        shift_mask = np.ones(np.shape(coadd))
        ref_directory_path = os.path.join(setup.red_dir, 'ref')
        if not os.path.exists(ref_directory_path):
            os.mkdir(ref_directory_path)
        accepted = 0
        for image in best_images:
            image_structure = image_handling.determine_image_struture(os.path.join(reduction_metadata.data_architecture[1]['IMAGES_PATH'][0],image[0]), log=log)
            data_hdu = fits.open(os.path.join(reduction_metadata.data_architecture[1]['IMAGES_PATH'][0],image[0]))
            xs,ys =  find_shift(ref_hdu[image_structure['sci']].data, data_hdu[image_structure['sci']].data)
            shifted = shift(data_hdu[0].data, (ys,xs), cval=0.)
            #limit shift to 30
            if xs**2+ys**2<900.:
                coadd = coadd + shifted
                shift_mask[shifted==0] = 0.
                accepted += 1
            data_hdu.close()

        coadd[shift_mask==0] = 0.0
        ref_hdu[ref_structure['sci']].data = coadd/float(accepted)
        if ref_structure['bpm'] != None:
            ref_hdu[ref_structure['bpm']].data[shift_mask==0] = 1
        if ref_structure['pyDANDIA_PIXEL_MASK'] != None:
            ref_hdu[ref_structure['pyDANDIA_PIXEL_MASK']].data[shift_mask==0] = 1
        if ref_structure['bpm'] == None and ref_structure['pyDANDIA_PIXEL_MASK'] == None:
            log.info('no mask in extension 1')
            
# DEPRECIATED?
#        try:
#            ref_hdu[1].data[shift_mask==0] = 1
#        except:
#            log.info('no mask in extension 1')
#        try:
#            ref_hdu[2].data[shift_mask==0] = 1
#        except:
#            log.info('no mask in extension 2')

        ref_hdu.writeto(os.path.join(reduction_metadata.data_architecture[1]['IMAGES_PATH'][0],'ref.fits'),overwrite = True)
        ref_hdu.writeto(os.path.join(ref_directory_path,'ref.fits'),overwrite = True)
        ref_hdu.close()


        if not 'REF_PATH' in reduction_metadata.data_architecture[1].keys():
            reduction_metadata.add_column_to_layer('data_architecture',
                                               'REF_PATH', [ref_directory_path],
                                                new_column_format=None,
                                                new_column_unit=None)
        else:
            reduction_metadata.update_a_cell_to_layer('data_architecture', 0,'REF_PATH', ref_directory_path)

        if  not 'REF_IMAGE' in reduction_metadata.data_architecture[1].keys():
            reduction_metadata.add_column_to_layer('data_architecture',
                                               'REF_IMAGE', ['ref.fits'],
                                                new_column_format=None,
                                               new_column_unit=None)
        else:
            reduction_metadata.update_a_cell_to_layer('data_architecture', 0,'REF_IMAGE', 'ref.fits')
        # Update the REDUCTION_STATUS table in metadata for stage 2


        reduction_metadata.calc_psf_radii()

        image_red_status = metadata.set_image_red_status(image_red_status,'1')

        reduction_metadata.update_reduction_metadata_reduction_status_dict(image_red_status,
                                                          stage_number=2, log=log)
        reduction_metadata.save_updated_metadata(metadata_directory=setup.red_dir,
                                                 metadata_name='pyDANDIA_metadata.fits')

        #tbd: update status of the coadded image


        status = 'OK'
        report = 'Completed successfully'
        log.info('Updating metadata with info on new images...')
        logs.close_log(log)


        (status_s0, report_s0, reduction_metadata_s0) = stage0.run_stage0(setup)
        #print(status_s0, report_s0)
        (status_s1, report_s1) = stage1.run_stage1(setup)
        #print(status_s1, report_s1)

        return status, report

    if reference_ranking == []:
        status = 'FAILED'
        report = 'No suitable image found.'

        log.info('No reference image found...')
        logs.close_log(log)

        return status, report

def get_default_config(kwargs,log):

    default_config = { 'empirical_ranking': False,
                        'n_stack': 1 }

    kwargs = config_utils.set_default_config(default_config,kwargs,log)

    return kwargs

def find_shift(reference_image, target_image):
    """
    Using a reference image and a target image
    determine the x and y offset
    :param list header: images
    """
    reference_shape = reference_image.shape
    x_center = int(reference_shape[0] / 2)
    y_center = int(reference_shape[1] / 2)
    correlation = convolution.convolve_image_with_a_psf(np.matrix(reference_image),np.matrix(target_image), correlate=1)
    x_shift, y_shift = np.unravel_index(np.argmax(correlation), correlation.shape)
    good_shift_y = y_shift - y_center
    good_shift_x = x_shift - x_center
    return good_shift_y, good_shift_x

def moon_brightness_header(header,row_idx):
    """
    Based on header information, determine if image was
    taken with bright/gray or dark moon
    roughly following the ESO definitions
    https://www.eso.org/sci/observing/phase2/ObsConditions.html
    but a lunar distance of 60 degrees is considered as gray time
    :param list header: header or metadata dictionary
    """

    if float(header['MOONFKEY'][row_idx]) < 0.4:
        return 'dark'
    else:
        if float(header['MOONFKEY'][row_idx]) > 0.4 and float(header['MOONFKEY'][row_idx]) < 0.7 and float(header['MOONDKEY'][row_idx]) > 60.0:
            return 'gray'
        else:
            return 'bright'


def add_stage1_rank(reduction_metadata, image_stats_entry):
    """
    Image ranking based on the data_inventory (stage1 metadata)

    :param object reduction_metadata: the metadata object

    """
    target_magnitude = 19.  # to be defined in metadata, optimally suited for
                            # fainter mag 19 stars (sky more relevant)
    magzero_electrons = 1.47371235e+09

    # finding the index of the stats_entry image in the headers
    header_idx = list(reduction_metadata.headers_summary[1]['IMAGES']).index(
        image_stats_entry['IM_NAME'])
    # needs to be updated so that the corresponding values
    # can be configured
    signal_electrons = 10.0**(-0.4 * electrons_per_second_sinistro(
        magzero_electrons, target_magnitude)) * float(reduction_metadata.reduction_parameters[1]['GAIN']) * float(reduction_metadata.headers_summary[1][header_idx]['EXPKEY'])
    sky = float(image_stats_entry['SKY'])
    sky_electrons = sky * \
        float(reduction_metadata.reduction_parameters[1]['GAIN'])
    # Simplified number of pixels for optimal aperture photometry
    fwhm_avg = float(image_stats_entry['FWHM'])
    npix = np.pi * \
        float(reduction_metadata.reduction_parameters[1]['PIX_SCALE']) * (
            0.67 * fwhm_avg)**2

    readout_noise = float(reduction_metadata.reduction_parameters[1]['RON'])
    return (signal_electrons) / (signal_electrons + npix * (sky_electrons + readout_noise**2))**0.5

def add_stage1_rank_sharpest(reduction_metadata, image_stats_entry):
    """
    Image ranking based on the data_inventory (stage1 metadata)

    :param object reduction_metadata: the metadata object

    """
    # finding the index of the stats_entry image in the headers
    header_idx = list(reduction_metadata.headers_summary[1]['IMAGES']).index(
        image_stats_entry['IM_NAME'])
    fwhm_avg = float(image_stats_entry['FWHM'])

    return 1./fwhm_avg

def electrons_per_second_sinistro(mag, magzero_electrons):
    """
    Temporary definition of electrons per second for a given
    camera. Ideally, the coefficients used here should be
    provided in each instrument file.
    :param object mag: the metadata object
    """
    return magzero_electrons * 10.**(-0.4 * mag)


def check_header_thresholds(moon_status, stage1_entry):
    """
    Check for images that cannot be used as reference image
    however small the FWHM was.
    """
    criteria = [moon_brightness_header != 'bright']
