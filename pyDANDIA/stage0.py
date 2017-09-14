######################################################################
#                                                                   
# stage0.py - First stage of the pipeline. Handle data, create bad pixels mask...
# More details in individual fonctions.

#
# dependencies:
#      numpy 1.8+
#      astropy 1.0+ 
######################################################################

import numpy as np
import os
from astropy.io import fits
import sys

import config
from astropy.table import Table
from astropy.nddata import Cutout2D

from pyDANDIA import metadata
from pyDANDIA import pixelmasks


def open_the_variables_catalog(variables_catalog_directory, variables_catalog_name):
    '''
    NOT IMPLEMENTED YET
    '''

    variable_catalog = None
    pass


def read_the_config_file(config_directory, config_file_name='config.json'):
    '''
    This read the reauired informations from the config file.

    :param string config_directory: the directory to the config file
    :param string config_file_name: the name of the config file
   
    :return: the config file
    :rtype: dictionnary
    '''

    pipeline_configuration = config.read_config(config_directory + config_file_name)

    return pipeline_configuration


def create_or_load_the_reduction_metadata(output_metadata_directory, metadata_name='pyDANDIA_metadata.fits',
                                          verbose=False):
    '''
    This creates (new reduction) or load (ongoing reduction) the metadata file linked to this reduction.

    :param string output_metadata_directory: the directory where to place the metadata
    :param string metadata_name: the name of the metadata file
    :param boolean verbose: switch to True to have more informations

    :return: the metadata object
    :rtype: metadata object
    '''
    try:

        meta_data_exist = [i for i in os.listdir(output_metadata_directory) if (i == metadata_name)]

        if meta_data_exist == []:

            reduction_metadata = metadata.MetaData()

            reduction_metadata.create_metadata_file(output_metadata_directory, metadata_name)

            if verbose == True:
                print('Successfully create the reduction metadata')

        else:

            reduction_metadata = metadata.MetaData()
            reduction_metadata.load_all_metadata(output_metadata_directory, metadata_name)
            if verbose == True:
                print('Successfully found the reduction metadata')
    except:

        if verbose == True:
            print('No metadata created or loaded : check this!')

        sys.exit(1)

    return reduction_metadata


def find_all_images(reduction_metadata, images_directory_path=None, verbose=False):
    '''
    This found all the images.

    :param object reduction_metadata: the metadata object
    :param string images_directory_path: the directory of the images
    :param boolean verbose: switch to True to have more informations

    :return: the list of images (strings)
    :rtype: list
    '''
    try:

        path = reduction_metadata.data_architecture[1]['IMAGES_PATH']

    except:

        if images_directory_path:
            path = images_directory_path

            reduction_metadata.add_column_to_layer('data_architecture', 'images_path', [path])

    try:

        list_of_images = [i for i in os.listdir(path)]

        if list_of_images == []:

            if verbose == True:
                print('No images to process. I take a rest :)')

            return None


        else:

            if verbose == True:
                print('Find ' + str(len(list_of_images)) + ' images to treat.')

            return list_of_images

    except:

        if verbose == True:
            print('Something went wrong on images search!')

        return None


def find_images_need_to_be_process(reduction_metadata, list_of_images, verbose=False):
    '''
    This founds the images that need to be processed by the pipeline, i.e not already done.

    :param object reduction_metadata: the metadata object
    :param  list list_of_images: the directory of the images
    :param boolean verbose: switch to True to have more informations

    :return: the new images that need to be processed.
    :rtype: list
    '''
    try:
        layer = reduction_metadata.reduction_status

        if layer == [None, None]:

            new_images = list_of_images

        else:

            new_images = []

            old_images = layer[1]['IMAGES']

            for name in list_of_images:

                if name not in old_images:
                    if verbose == True:
                        print(name + ' is a new image to treat!')
                    new_images.append(name)

    except:
        if verbose == True:
            print('Something went wrong on images/metadata matching !')

    return new_images


def open_an_image(reduction_metadata, image_name, verbose=False):
    '''
    Simply open an image using astropy.io.fits

    :param object reduction_metadata: the metadata object
    :param string image_name: the image name
    :param boolean verbose: switch to True to have more informations

    :return: the opened image
    :rtype: astropy.image object
    '''
    image_directory_path = reduction_metadata.data_architecture[1]['IMAGES_PATH'][0]
    try:

        image_data = fits.open(image_directory_path + image_name, mmap=True)
        if verbose == True:
            print(image_name + ' open : OK')

        return image_data
    except:
        if verbose == True:
            print(image_name + ' open : not OK!')

        return None


def construct_the_bad_pixel_mask(image_bad_pixel_mask, integers_to_flag):
    '''
    Construct the bad pixel mask : 0 = good pixel
                                   1 = bad pixel

    :param array_like image_bad_pixel_mask: an array of integer
    :param list  integers_to_flag: the integers value that need to be flagged as BAD pixels


    :return: the bad pixel mask
    :rtype: array_like
    '''

    bad_pixel_mask = np.zeros(image_bad_pixel_mask.shape, int)
    try:

        bad_pixel_mask = np.zeros(image_bad_pixel_mask.shape, int)

        for flag in integers_to_flag:
            # BANZAI definition of bp == 1 or bp == 3
            index_saturation = np.where((bad_pixel_mask == flag))
            bad_pixel_mask[index_saturation] = 1



    except:

        pass

    return bad_pixel_mask


def construct_the_variables_star_mask(open_image, variable_star_pixels=10):
    '''
    Construct the variable stars pixel mask : 0 = good pixel
                                              1 = bad pixel

    NEED WORK :)

    :param astropy.image open_image: the opened image
    :param int  variable_star_pixels: the pixel radius you want to flagged around the variable object

    :return: the variable star pixel mask
    :rtype: array_like
    '''

    try:

        pass

    except:

        pass

    pass


def construct_the_saturated_pixel_mask(open_image, saturation_level=65535):
    '''
    Construct the saturated pixel mask : 0 = good pixel
                                         1 = bad pixel

    :param astropy.image open_image: the opened image
    :param int saturation_level: the level above considered as saturated


    :return: the saturated pixel mask
    :rtype: array_like
    '''

    try:

        if saturation_level:
            pass

        else:

            saturation_level = open_image[0].header['SATURATE']

        mask = open_image[0].data >= saturation_level
        saturated_pixel_mask = mask.astype(int)

    except:

        saturated_pixel_mask = np.zeros(open_image[0].data.shape, int)

    return saturated_pixel_mask


def construct_the_low_level_pixel_mask(open_image, low_level=0):
    '''
    Construct the low level pixel mask : 0 = good pixel
                                         1 = bad pixel

    :param astropy.image open_image: the opened image
    :param int low_level: the level below is considered as bad value


    :return: the low level pixel mask
    :rtype: array_like
    '''
    try:
        data = open_image[0].data

        mask = data <= low_level

        low_level_pixel_mask = mask.astype(int)

    except:

        low_level_pixel_mask = np.zeros(open_image[0].data.shape, int)

    return low_level_pixel_mask


def construct_the_pixel_mask(open_image, bad_pixel_mask, integers_to_flag,
                             saturation_level=65535, low_level=0):
    '''
    Construct the global pixel mask  using a bitmask approach.

    :param astropy.image open_image: the opened image
    :param list integers_to_flag: the list of integers corresponding to a bad pixel
    :param float saturation_level: the value above which a pixel is consider saturated
    :param float low_level: the value below which a pixel is consider as bad value


    :return: the low level pixel mask
    :rtype: array_like
    '''
    try:
        bad_pixel_mask = construct_the_bad_pixel_mask(bad_pixel_mask, integers_to_flag)

        # variables_pixel_mask = construct_the_variables_star_mask(open_image, variable_star_pixels=10)

        saturated_pixel_mask = construct_the_saturated_pixel_mask(open_image, saturation_level)

        low_level_pixel_mask = construct_the_low_level_pixel_mask(open_image, low_level)

        original_master_mask = np.zeros(open_image[0].data.shape, int)
        list_of_masks = [bad_pixel_mask, saturated_pixel_mask, low_level_pixel_mask]

        master_mask = pixelmasks.construct_a_master_mask(original_master_mask, list_of_masks)

        return master_mask

    except:

        master_mask = np.zeros(open_image[0].data.shape, int)

    return master_mask


def save_the_pixel_mask_in_image(reduction_metadata, image_name, open_image, master_mask):
    '''
    Construct the global pixel mask  using a bitmask approach.

    :param object reduction_metadata: the metadata object
    :param string image_name: the name of the image
    :param astropy.image open_image: the opened image
    :param array_like master_mask: the master mask which needs to be kept

    '''
    master_pixels_mask = fits.ImageHDU(master_mask)
    master_pixels_mask.name = 'MASTER_PIXEL_MASK'

    try:
        open_image['MASTER_PIXEL_MASK'] = master_pixels_mask
    except:

        open_image.append(master_pixels_mask)

    image_directory = reduction_metadata.data_architecture[1]['IMAGES_PATH'][0]

    open_image.writeto(image_directory + image_name, overwrite=True)


def update_reduction_metadata_with_config_file(reduction_metadata, config_dictionnary):
    '''
    Update the metadata with the config files

    :param object reduction_metadata: the metadata object
    :param dictionnary config_dictionnary: a python dictionnary containing the pyDANDIA parameters

    '''


    keys = config_dictionnary.keys()

    data = []
    for key in keys:

        try:
            data.append([key, config_dictionnary[key]['value'],  config_dictionnary[key]['format'],
                         config_dictionnary[key]['unit']])

        except:

            print('Something went wrong with the config file on the key'+key)
            sys.exit(1)

    data = np.array(data)
    names = [i.upper() for i in data[:, 0]]
    units = data[:, 2]
    formats = data[:, 3]

    reduction_metadata.create_reduction_parameters_layer(names, units, formats, data[:,1])





def parse_the_image_header(reduction_metadata, open_image):
    '''
    Update the metadata with the header keywords

    :param object reduction_metadata: the metadata object
    :param astropy.image open_image: the opened image

    '''

    layer = reduction_metadata.header_summary[1]





def construct_the_stamps(open_image, stamp_size=None, arcseconds_stamp_size=(60, 60), pixel_scale=None,
                         number_of_overlaping_pixels=25, verbose=False):
    image = open_image[0].data

    full_image_y_size, full_image_x_size = image.shape

    if stamp_size:

        y_stamp_size = stamp_size[0]
        x_stamp_size = stamp_size[1]

    else:
        try:

            y_stamp_size = int(arcseconds_stamp_size[0] / pixel_scale)
            x_stamp_size = int(arcseconds_stamp_size[1] / pixel_scale)

        except:
            print('No pixel scale found!')
            sys.exit(1)

    x_stamps_center = np.arange(x_stamp_size / 2, full_image_x_size, x_stamp_size)
    y_stamps_center = np.arange(y_stamp_size / 2, full_image_y_size, y_stamp_size)

    stamps_center_x, stamps_center_y = np.meshgrid(y_stamps_center, x_stamps_center)

    stamps_y_min = stamps_center_y - y_stamp_size / 2 - number_of_overlaping_pixels
    mask = stamps_y_min < 0
    stamps_y_min[mask] = 0

    stamps_y_max = stamps_center_y + y_stamp_size / 2 + number_of_overlaping_pixels
    mask = stamps_y_max > full_image_y_size
    stamps_y_min[mask] = full_image_y_size

    stamps_x_min = stamps_center_x - x_stamp_size / 2 - number_of_overlaping_pixels
    mask = stamps_x_min < 0
    stamps_x_min[mask] = 0

    stamps_x_max = stamps_center_x + x_stamp_size / 2 + number_of_overlaping_pixels
    mask = stamps_x_max > full_image_x_size
    stamps_x_min[mask] = full_image_x_size

    stamps = [[j * (i + 1), stamps_y_min[i, j], stamps_y_max[i, j], stamps_x_min[i, j], stamps_x_max[i, j]]
              for i in range(stamps_x_min.shape[0]) for j in range(stamps_x_min.shape[1])]

    return np.array(stamps)
