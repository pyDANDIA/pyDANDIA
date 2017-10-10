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

import config_utils
from astropy.table import Table
from astropy.nddata import Cutout2D

#from pyDANDIA import metadata
#from pyDANDIA import pixelmasks
import metadata
import pixelmasks
import logs

def run_stage0(setup):
    """Main driver function to run stage 0: data preparation.
    
    The tasks of this stage are to ensure that all images are prepared for 
    reduction, and to make sure the reduction meta data is up to date.
    """
    
    stage0_version = 'stage0 v0.1'
    
    log = logs.start_stage_log(setup.red_dir, 'stage0', version=stage0_version)
    log.info('Setup:\n'+setup.summary())
    
    pipeline_config = read_the_config_file(setup.pipeline_config_dir,log=log)    
    
    reduction_metadata = create_or_load_the_reduction_metadata(
                                    setup.red_dir, 
                                    metadata_name='pyDANDIA_metadata.fits', 
                                    verbose=True,log=log)
    
    update_reduction_metadata_with_config_file(reduction_metadata, 
                                    pipeline_config,log=log)

    data = find_all_images(reduction_metadata, 
                                os.path.join(setup.red_dir,'data'),
                                verbose=True,log=log)
                                
    new_images=find_images_need_to_be_process(reduction_metadata, data, 
                                verbose=False, log=log)
    
    
    if len(reduction_metadata.data_inventory[1])==0:
        
        create_reduction_metadata_data_inventory(reduction_metadata, 
                                new_images, status=0, log=log)
    
    if len(new_images) > 0:
        update_reduction_metadata_headers_summary_with_new_images(
                            reduction_metadata, new_images, log=log)
    
        open_image = open_an_image(
                    reduction_metadata.data_architecture[1]['IMAGES_PATH'][0], 
                    new_images[0],image_index=0, verbose=True, log=log)

        update_reduction_metadata_stamps(reduction_metadata, open_image,
                     stamp_size=None, 
                     arcseconds_stamp_size=(60, 60),
                     pixel_scale=None, 
                     number_of_overlaping_pixels=25,
                     verbose=False, log=log)
    
        set_bad_pixel_mask_directory(reduction_metadata, 
                     bpm_directory_path=os.path.join(setup.red_dir,'data'), 
                     verbose=False, log=log)

        log.info('Updating metadata with info on new images...')
        for new_image in new_images:
            open_image = open_an_image( 
                reduction_metadata.data_architecture[1]['IMAGES_PATH'][0],
                new_image, image_index=0, verbose=True, log=log)
            
            bad_pixel_mask = open_an_image( 
                reduction_metadata.data_architecture[1]['BPM_PATH'][0],
                new_image, image_index=2, verbose=True, log=log)
            
            construct_the_pixel_mask(open_image, bad_pixel_mask, [1,3],
                             saturation_level=65535, low_level=0, log=log)
            
            log.info(' -> '+new_image)
        
        construct_the_pixel_mask(open_image, bad_pixel_mask, [1,3],
                         saturation_level=65535, low_level=0, log=log)

    
        update_reduction_metadata_data_inventory(reduction_metadata, 
                        new_images, status=1, log=log)
    
    reduction_metadata.save_updated_metadata(
            reduction_metadata.data_architecture[1]['OUTPUT_DIRECTORY'][0],
            reduction_metadata.data_architecture[1]['METADATA_NAME'][0],
            log=log)
    
    logs.close_log(log)
    
    status = 'OK'
    report = 'Completed'
    
    return status, report, reduction_metadata
    

def open_the_variables_catalog(variables_catalog_directory, variables_catalog_name):
    '''
    NOT IMPLEMENTED YET
    '''

    variable_catalog = None
    pass


def read_the_config_file(config_directory, config_file_name='config.json',
                         log=None):
    '''
    This read the required informations from the config file.

    :param string config_directory: the directory to the config file
    :param string config_file_name: the name of the config file
   
    :return: the config file
    :rtype: dictionnary
    '''

    config_file_path = os.path.join(config_directory, config_file_name)

    pipeline_configuration = config_utils.read_config(config_file_path)

    if log != None:
        log.info('Read pipeline configuration from '+config_file_path)

    return pipeline_configuration


def create_or_load_the_reduction_metadata(output_metadata_directory, metadata_name='pyDANDIA_metadata.fits',
                                          verbose=False,log=None):
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

            if verbose == True and log != None:
                log.info('Successfully created the reduction metadata file')

        else:

            reduction_metadata = metadata.MetaData()
            reduction_metadata.load_all_metadata(output_metadata_directory, metadata_name)
            if verbose == True and log != None:
                log.info('Successfully found the reduction metadata')
    except:

        if verbose == True and log != None:
            log.info('No metadata created or loaded : check this!')

        sys.exit(1)

    return reduction_metadata


def set_bad_pixel_mask_directory(reduction_metadata, bpm_directory_path=None, 
                                 verbose=False, log=None):
    '''
    This found all the images.

    :param object reduction_metadata: the metadata object
    :param string images_directory_path: the directory of the images
    :param boolean verbose: switch to True to have more informations

    :return: the list of images (strings)
    :rtype: list
    '''

    reduction_metadata.add_column_to_layer('data_architecture', 
                                           'BPM_PATH', [bpm_directory_path],
                                           new_column_format=None,
                                           new_column_unit=None)
    
    if verbose == True and log != None:
        log.info('Set bad pixel mask directory')

def find_all_images(reduction_metadata, images_directory_path=None, 
                    verbose=False, log=None):
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

        list_of_images = [i for i in os.listdir(path) if ('.fits' in i) and ('.gz' not in i) and ('.bz2' not in i)]

        if list_of_images == []:

            if verbose == True and log != None:
                log.info('No images to process. I take a rest :)')

            return None


        else:

            if verbose == True and log != None:
                log.info('Found ' + str(len(list_of_images)) + \
                        ' images in this dataset')

            return list_of_images

    except:

        if verbose == True and log != None:
            log.info('Something went wrong on images search!')

        return None


def find_images_need_to_be_process(reduction_metadata, list_of_images, 
                                   verbose=False, log=None):
    '''
    This founds the images that need to be processed by the pipeline, i.e not already done.

    :param object reduction_metadata: the metadata object
    :param  list list_of_images: the directory of the images
    :param boolean verbose: switch to True to have more informations

    :return: the new images that need to be processed.
    :rtype: list
    '''

    try:
        layer = reduction_metadata.data_inventory

        if len(layer[1]) == 0:

            new_images = list_of_images

        else:

            new_images = []

            old_images = layer[1]['IMAGES']

            for name in list_of_images:

                if name not in old_images:
                    if verbose == True and log != None:
                        log.info(name + ' is a new image to treat!')
                    new_images.append(name)

    except:
        if log != None:
            log.info('Error in scanning for new images to reduce')

    if log != None:
        log.info('Total of '+str(len(new_images))+' images need reduction')
    
    return new_images


def open_an_image(image_directory, image_name, image_index=0, 
                  verbose=False, log=None):
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
    
    if verbose == True and log != None:
        log.info('Attempting to open image '+os.path.join(image_directory_path,image_name))
    
    try:

        image_data = fits.open(os.path.join(image_directory_path,image_name), 
                               mmap=True)
        image_data = image_data[image_index]
    
        if verbose == True and log != None:
            log.info(image_name + ' open : OK')
    
        return image_data
    except:
        if verbose == True and log != None:
            log.info(image_name + ' open : not OK!')

        return None


def open_an_bad_pixel_mask(reduction_metadata, bpm_name, bpm_index=1, verbose=False):
    '''
    Simply open an image using astropy.io.fits

    :param object reduction_metadata: the metadata object
    :param string bpm_name: the bad pixel mask name
    :param string bpm_index: the bad pixel mask index of the astropy fits object

    :param boolean verbose: switch to True to have more informations

    :return: the opened bad pixel mask
    :rtype: astropy.image object
    '''
    bpm_directory_path = reduction_metadata.data_architecture[1]['BPM_PATH'][0]

    try:

        image_data = fits.open(bpm_directory_path + bpm_name, mmap=True)
        image_data = image_data[bpm_index]

        if verbose == True:
            print(bpm_name + ' open : OK')

        return image_data
    except:
        if verbose == True:
            print(bpm_name + ' open : not OK!')

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

        mask = open_image.data >= saturation_level
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
        data = open_image.data

        mask = data <= low_level

        low_level_pixel_mask = mask.astype(int)

    except:

        low_level_pixel_mask = np.zeros(open_image[0].data.shape, int)

    return low_level_pixel_mask


def construct_the_pixel_mask(open_image, bad_pixel_mask, integers_to_flag,
                             saturation_level=65535, low_level=0, log=None):
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

        original_master_mask = np.zeros(open_image.data.shape, int)
        list_of_masks = [bad_pixel_mask, saturated_pixel_mask, low_level_pixel_mask]

        master_mask = pixelmasks.construct_a_master_mask(original_master_mask, list_of_masks)

        if log != None:
            log.info('Successfully built a BPM')

        return master_mask

    except:

        master_mask = np.zeros(open_image.data.shape, int)

        if log != None:
            log.info('Error building the BPM; using zeroed array')
            
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


def update_reduction_metadata_with_config_file(reduction_metadata, 
                                               config_dictionnary,log=None):
    '''
    Update the metadata with the config files

    :param object reduction_metadata: the metadata object
    :param dictionnary config_dictionnary: a python dictionnary containing the pyDANDIA parameters

    '''

    keys = config_dictionnary.keys()

    data = []
    for key in keys:

        try:
            data.append([key, config_dictionnary[key]['value'], config_dictionnary[key]['format'],
                         config_dictionnary[key]['unit']])

        except:
            if log != None:
                log.info('Error in config file on key' + key)
            sys.exit(1)

    data = np.array(data)
    names = [i.upper() for i in data[:, 0]]
    formats = data[:, 2]
    units = data[:, 3]

    reduction_metadata.create_reduction_parameters_layer(names, formats, units, data[:, 1])

    if log != None:
        log.info('Updated metadata with pipeline configuration parameters')

def parse_the_image_header(reduction_metadata, open_image):
    '''
    Update the metadata with the header keywords

    :param object reduction_metadata: the metadata object
    :param astropy.image open_image: the opened image

    :return an array containing the needed header info
    :rtype array_like
    '''

    layer_reduction_parameters = reduction_metadata.reduction_parameters[1]
    image_header = open_image.header

    header_infos = []

    for key in layer_reduction_parameters.keys():

        try:
            info = [key, image_header[layer_reduction_parameters[key][0]], 
                    layer_reduction_parameters[key].dtype]
            header_infos.append(info)

        except:
            pass

    return np.array(header_infos)


def update_reduction_metadata_headers_summary_with_new_images(reduction_metadata, 
                                                              new_images, log=None):
    '''
    Update the metadata with the header keywords

    :param object reduction_metadata: the metadata object
    :param list new_images: list of strings

    :return an array containing the needed header info
    :rtype array_like
    '''
    
    for image_name in new_images:
        open_image = open_an_image(reduction_metadata.data_architecture[1]['IMAGES_PATH'][0], 
                                   image_name, verbose=False,log=log)
        
        header_infos = parse_the_image_header(reduction_metadata, open_image)

        names = np.append('IMAGES', header_infos[:, 0])
        values = np.append(image_name, header_infos[:, 1])
        formats = np.append('S200', header_infos[:, 2])

        if reduction_metadata.headers_summary[1]:
            reduction_metadata.add_row_to_layer('headers_summary', values)

        else:

            reduction_metadata.create_headers_summary_layer(names, formats, 
                                                            units=None, 
                                                            data=values)
            
    if log != None:
        log.info('Added data on new images to the metadata')

def construct_the_stamps(open_image, stamp_size=None, arcseconds_stamp_size=(60, 60), pixel_scale=None,
                         number_of_overlaping_pixels=25, verbose=False):
    '''
    Define the stamps for an image variable kernel definition

    :param object reduction_metadata: the metadata object
    :param list stamp_sizes: list of integer give the X,Y stamp size , i.e [150,52] give 150 pix in X, 52 in Y
    :param tuple arcseconds_stamp_size: list of integer give the X,Y stamp size in arcseconds units
    :param float pixel_scale: pixel scale of the CCD, in arcsec/pix
    :param int number_of_overlaping_pixels : half of  number of pixels in both direction you want overlaping
    :param boolean verbose: switch to True to have more informations



    :return an array containing the pixel index, Y_min, Y_max, X_min, X_max (i.e matrix index definition)
    :rtype array_like
    '''

    image = open_image.data

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


def update_reduction_metadata_stamps(reduction_metadata, open_image,
                                     stamp_size=None, arcseconds_stamp_size=(60, 60),
                                     pixel_scale=None, number_of_overlaping_pixels=25,
                                     verbose=False, log=None):
    '''
    Create the stamps definition in the reduction_metadata

    :param object reduction_metadata: the metadata object
    :param astropy.image open_image: the opened image
    :param list stamp_sizes: list of integer give the X,Y stamp size , i.e [150,52] give 150 pix in X, 52 in Y
    :param tuple arcseconds_stamp_size: list of integer give the X,Y stamp size in arcseconds units
    :param float pixel_scale: pixel scale of the CCD, in arcsec/pix
    :param int number_of_overlaping_pixels : half of  number of pixels in both direction you want overlaping
    :param boolean verbose: switch to True to have more informations

    '''

    if pixel_scale:
        pass
    else:
        pixel_scale = float(reduction_metadata.headers_summary[1]['PIXEL_SCALE'][0])

    stamps = construct_the_stamps(open_image, stamp_size, arcseconds_stamp_size, pixel_scale,
                                  number_of_overlaping_pixels, verbose)

    names = ['PIXEL_INDEX', 'Y_MIN', 'Y_MAX', 'X_MIN', 'X_MAX']
    formats = ['int', 'S200', 'S200', 'S200', 'S200']
    units = ['', 'pixel', 'pixel', 'pixel', 'pixel']

    reduction_metadata.create_stamps_layer(names, formats, units, stamps)

    if log != None:
        log.info('Updated reduction metadata stamps')

def create_reduction_metadata_data_inventory(reduction_metadata, new_images, 
                                             status=0, log=None):
    '''
        Create the data_inventory layer with all status set to status

        :param object reduction_metadata: the metadata object
        :param list new_images: list of string with the new images names
        :param int status: status of stage0 reducitonfor a frame. 0 : not done
                                                                  1 : done
    '''
    for new_image in new_images:
        reduction_metadata.add_row_to_layer('data_inventory', [new_image] + [status] *
                                            (len(reduction_metadata.data_inventory[1].keys()) - 1))

    if log != None:
        log.info('Completed inventory of the data')
        
def update_reduction_metadata_data_inventory(reduction_metadata, new_images, 
                                             status=1, log=None):
    '''
        Update the data_inventory layer with all status set to status

        :param object reduction_metadata: the metadata object
        :param list new_images: list of string with the new images names
        :param int status: status of stage0 reducitonfor a frame. 0 : not done
                                                                  1 : done
    '''
    for new_image in new_images:
        index_image = np.where(reduction_metadata.data_inventory[1]['IMAGES'] == new_image)[0][0]
        reduction_metadata.data_inventory[1][index_image][1] = status

    if log != None:
        log.info('Updated the reduction data inventory')
        