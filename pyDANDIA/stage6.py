######################################################################
#
# stage6.py - Sixth stage of the pipeline. Subtract and make photometry
# on residuals

#
# dependencies:
#      numpy 1.8+
#      astropy 1.0+
######################################################################

import numpy as np
import os
import sys
from astropy.io import fits
from astropy.table import Table
from astropy.table import Column

import config_utils

import metadata
import logs
import convolution


def run_stage6(setup):
    """Main driver function to run stage 6: image substraction and photometry.
    This stage align the images to the reference frame!
    :param object setup : an instance of the ReductionSetup class. See reduction_control.py

    :return: [status, report, reduction_metadata], the stage4 status, the report, the metadata file
    :rtype: array_like

    """

    stage6_version = 'stage6 v0.1'

    log = logs.start_stage_log(setup.red_dir, 'stage6', version=stage6_version)
    log.info('Setup:\n' + setup.summary() + '\n')

    # find the metadata
    reduction_metadata = metadata.MetaData()
    reduction_metadata.load_all_metadata(setup.red_dir, 'pyDANDIA_metadata.fits')

    # find the images needed to treat
    all_images = reduction_metadata.find_all_images(setup, reduction_metadata,
                                                    os.path.join(setup.red_dir, 'data'), log=log)

    new_images = reduction_metadata.find_images_need_to_be_process(setup, all_images,
                                                                   stage_number=6, rerun_all=None, log=log)

    if len(new_images) > 0:

        # find the reference image
        try:
            reference_image_name = reduction_metadata.data_architecture[1]['REFERENCE_NAME']
            reference_image_directory = reduction_metadata.data_architecture[1]['IMAGES_PATH']
            reference_image = open_an_image(setup, reference_image_directory, reference_image_name, image_index=0,
                                            log=None)
            logs.ifverbose(log, setup,
                           'I found the reference frame:' + reference_image_name)
        except KeyError:
            logs.ifverbose(log, setup,
                           'I can not find any reference image! Aboard stage6')

            status = 'KO'
            report = 'No reference frame found!'

            return status, report

        # find the kernels directory
        try:

            kernels_directory = reduction_metadata.data_architecture[1]['KERNELS_PATH']

            logs.ifverbose(log, setup,
                           'I found the kernels directory:' + kernels_directory)
        except KeyError:
            logs.ifverbose(log, setup,
                           'I can not find the kernels directory! Aboard stage6')

            status = 'KO'
            report = 'No kernels directory found!'

            return status, report

        # find the reference image psf model/parameters
        try:
            ####?????
            psf_model = reduction_metadata.data_architecture[1]['KERNELS_PATH']

            logs.ifverbose(log, setup,
                           'I found the kernels directory:' + kernels_directory)
        except KeyError:
            logs.ifverbose(log, setup,
                           'I can not find the kernels directory! Aboard stage6')

            status = 'KO'
            report = 'No kernels directory found!'

            return status, report
        data = []
        for new_image in new_images:

            target_image = open_an_image(setup, reference_image_directory, new_image, image_index=0, log=None)
            kernel_image = find_the_associated_kernel(setup, kernels_directory, new_image)

            difference_image = image_substraction(setup, reference_image, kernel_image, target_image)


	    photometric_table = photometry_on_the_difference_image(setup, difference_image, list_of_stars, psf_model, psf_parameters, kernel)
	     
            ingest_photometric_table_in_db(setup, photometric_table)

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

        return image_data

    except:
        logs.ifverbose(log, setup, image_name + ' open : not OK!')

        return None


def image_substraction(setup, reference_image_data, kernel_data, image_data, log=None):
    '''
    Subtract the image from model, i.e residuals = image-convolution(reference_image,kernel)

    :param object reduction_metadata: the metadata object
    :param array_like reference_image_data: the reference image data
    :param array_like kernel_data: the kernel image data
    :param array_like image_data: the image data

    :param boolean verbose: switch to True to have more informations

    :return: the difference image
    :rtype: array_like
    '''

    model = convolution.convolve_image_with_a_psf(reference_image_data, kernel_data)

    difference_image = image_data - model

    return difference_image


def find_the_associated_kernel(setup, kernels_directory, image_name):
    '''
    Find the appropriate kernel associated to an image
    :param object reduction_metadata: the metadata object
    :param string kernels_directory: the path to the kernels
    :param string image_name: the image name

    :return: the associated kernel to the image
    :rtype: array_like
    '''

    kernel_name = image_name.replace('.fits', '.ker')

    kernel = open_an_image(setup, kernels_directory, kernel_name,
                           image_index=0, log=None)

    return kernel


def photometry_on_the_difference_image(setup, difference_image, list_of_stars, psf_model, psf_parameters, kernel):
    '''
    Find the appropriate kernel associated to an image
    :param object reduction_metadata: the metadata object
    :param string kernels_directory: the path to the kernels
    :param string image_name: the image name

    :return: the associated kernel to the image
    :rtype: array_like
    '''


    differential_photometry = photometry.run_psf_photometry_on_difference_image(setup,reduction_metadata,log,ref_star_catalog,
                       								difference_image,psf_model,sky_model, kernel, centroiding=True)
    
    column_names = ('Exposure_id','Star_id','Ref_mag','Ref_mag_err','Ref_flux','Ref_flux_err','Delta_flux','Delta_flux_err','Mag','Mag_err',
                    'Phot_scale_factor','Phot_scale_factor_err','Back','Back_err','delta_x','delta_y')
   
    table = Table(differential_photometry, names = column_names)


    return table	


def ingest_photometric_table_in_db(setup, photometric_table):

	pass
