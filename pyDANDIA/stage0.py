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

from pyDANDIA import config_utils
from astropy.nddata import Cutout2D

from pyDANDIA import metadata
from pyDANDIA import pixelmasks
#import metadata
#import pixelmasks
from pyDANDIA import logs
import quality_control

def run_stage0(setup):
    """Main driver function to run stage 0: data preparation.    
    The tasks of this stage are to ensure that all images are prepared for 
    reduction, and to make sure the reduction metadata is up to date.
    Input: setup - is an instance of the ReductionSetup class. See 
           reduction_control.py
    Output: prepares the metadata file
    """

    stage0_version = 'stage0 v0.1'

    log = logs.start_stage_log(setup.red_dir, 'stage0', version=stage0_version)
    log.info('Setup:\n' + setup.summary() + '\n')

    # find and update the pipeline config
    pipeline_config = read_the_config_file(setup.pipeline_config_dir, log=log)

    reduction_metadata = create_or_load_the_reduction_metadata(setup,
                                                               setup.red_dir,
                                                               metadata_name='pyDANDIA_metadata.fits',
                                                               log=log)

    update_reduction_metadata_with_config_file(reduction_metadata,
                                               pipeline_config, log=log)

    # find all images

    all_images = reduction_metadata.find_all_images(setup, reduction_metadata,
                                                    os.path.join(setup.red_dir, 'data'), log=log)

    # find and update the inst pipeline config

    image_name = all_images[0]

    inst_config_file_name = find_the_inst_config_file_name(setup, reduction_metadata, image_name,
                                                           setup.pipeline_config_dir,
                                                           image_index=0, log=None)
    
    if inst_config_file_name == None:
        
            status = 'ERROR'
            report = 'Cannot find a pipeline configuration file for this dataset'
            
            return status, report, None
            
    inst_config = read_the_inst_config_file(setup.pipeline_config_dir, inst_config_file_name, log=log)
    update_reduction_metadata_with_inst_config_file(reduction_metadata,
                                                    inst_config, log=log)



    # find images need to be run, based on the metadata file, if any. If rerun_all = True, force a rereduction

    new_images = reduction_metadata.find_images_need_to_be_process(setup, all_images,
                                                                   stage_number=0, rerun_all=None, log=log)
    # create new rows on reduction status for new images

    reduction_metadata.update_reduction_metadata_reduction_status(new_images, stage_number=0, status=0, log=log)

    # construct the stamps if needed
    if reduction_metadata.stamps[1]:
        pass
    else:

        open_image = open_an_image(setup, reduction_metadata.data_architecture[1]['IMAGES_PATH'][0],
                                   new_images[0], image_index=0, log=log)

        update_reduction_metadata_stamps(setup, reduction_metadata, open_image,
                                         stamp_size=None,
                                         arcseconds_stamp_size=(110, 110),
                                         pixel_scale=None,
                                         number_of_overlaping_pixels=25, log=log)

    if len(new_images) > 0:

        update_reduction_metadata_headers_summary_with_new_images(setup,
                                                                  reduction_metadata, new_images, log=log)

        set_bad_pixel_mask_directory(setup, reduction_metadata,
                                     bpm_directory_path=os.path.join(setup.red_dir, 'data'),
                                     log=log)

        logs.ifverbose(log, setup, 'Updating metadata with info on new images...')

        for new_image in new_images:
            open_image = open_an_image(setup, reduction_metadata.data_architecture[1]['IMAGES_PATH'][0],
                                       new_image, image_index=0, log=log)

            bad_pixel_mask = open_an_image(setup, reduction_metadata.data_architecture[1]['BPM_PATH'][0],
                                           new_image, image_index=2, log=log)
            
            # Occasionally, the LCO BANZAI pipeline fails to produce an image
            # catalogue for an image.  If this happens, there will only be 2 
            # extensions to the FITS image HDU, the PrimaryHDU (main image data)
            # and the ImageHDU (BPM).  
            if bad_pixel_mask == None:
                
                bad_pixel_mask = open_an_image(setup, reduction_metadata.data_architecture[1]['BPM_PATH'][0],
                                           new_image, image_index=1, log=log)
                                           
            master_mask = construct_the_pixel_mask(open_image, bad_pixel_mask, [1, 3],
                                                   saturation_level=65535, low_level=0, log=log)

            save_the_pixel_mask_in_image(reduction_metadata, new_image, master_mask)
            logs.ifverbose(log, setup, ' -> ' + new_image)

    reduction_metadata.update_reduction_metadata_reduction_status(new_images, stage_number=0, status=1, log=log)

    reduction_metadata.save_updated_metadata(
        reduction_metadata.data_architecture[1]['OUTPUT_DIRECTORY'][0],
        reduction_metadata.data_architecture[1]['METADATA_NAME'][0],
        log=log)

    (status,report) = quality_control.verify_stage0_output(setup,log)
    
    logs.close_log(log)

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
        log.info('Read pipeline configuration from ' + config_file_path)

    return pipeline_configuration


def find_the_inst_config_file_name(setup, reduction_metadata, image_name, inst_config_directory, image_index=0,
                                   log=None):
    '''
    This found the name of the inst_config_file needs for the reduction.

    :param object reduction_metadata: the metadata object
    :param string image_name: the image name of the astropy fits object
    :param string inst_config_directory: the directory of the inst config file

    :param int image_index: the image index of the astropy fits object
    :param object log: the log object to add info in

    :return: the name of inst_config_file
    :rtype: string
    '''

    inst_config_files = [i for i in os.listdir(inst_config_directory) if ('inst_config' in i)]

    potential_cameras_names = [i.split('_')[-1][:-5] for i in inst_config_files]

    open_image = open_an_image(setup, reduction_metadata.data_architecture[1]['IMAGES_PATH'][0],
                               image_name, image_index=image_index, log=log)

    potential_inst_names = open_image.header.values()

    for name in potential_inst_names:

        if name in potential_cameras_names:
            good_camera_name = name
            inst_config_file_name = 'inst_config_' + good_camera_name + '.json'
            return inst_config_file_name

    return None


def read_the_inst_config_file(inst_config_directory, inst_config_file_name='inst_config.json', log=None):
    '''
    This read the required informations from the config file, i.e the pipeline configuration.

    :param string inst_config_directory: the directory to the instrument config file
    :param string inst_config_file_name: the name of the instrument config file

    :return: the config file
    :rtype: dictionnary
    '''

    config_file_path = os.path.join(inst_config_directory, inst_config_file_name)

    instrument_configuration = config_utils.read_config(config_file_path)

    if log != None:
        log.info('Read instrument configuration from ' + config_file_path)

    return instrument_configuration


def update_reduction_metadata_with_inst_config_file(reduction_metadata,
                                                    inst_config_dictionnary, log=None):
    '''
    Update the metadata with the config files

    :param object reduction_metadata: the metadata object
    :param dictionnary inst_config_dictionnary: a python dictionnary containing the instrument parameters

    '''

    keys = inst_config_dictionnary.keys()
    existing_keys_in_metadata = reduction_metadata.reduction_parameters[1].keys()
    for key in keys:

        try:

            value = inst_config_dictionnary[key]['value']
            format = inst_config_dictionnary[key]['format']
            unit = inst_config_dictionnary[key]['unit']

            if key.upper() in existing_keys_in_metadata:
                reduction_metadata.update_a_cell_to_layer('reduction_parameters', 0, key.upper(), value)

            else:

                reduction_metadata.add_column_to_layer('reduction_parameters', key, [value], format, unit)

        except:

            if log != None:
                log.info('Error in inst config file on key' + key)
            sys.exit(1)

    if log != None:
        log.info('Updated metadata with instrument configuration parameters')


def create_or_load_the_reduction_metadata(setup, output_metadata_directory,
                                          metadata_name='pyDANDIA_metadata.fits',
                                          log=None):
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

            logs.ifverbose(log, setup,
                           'Successfully created the reduction metadata file')

        else:

            reduction_metadata = metadata.MetaData()
            reduction_metadata.load_all_metadata(output_metadata_directory, metadata_name)
            logs.ifverbose(log, setup, 'Successfully found the reduction metadata')
    except:

        logs.ifverbose(log, setup, 'No metadata created or loaded : check this!')

        sys.exit(1)

    return reduction_metadata


def set_bad_pixel_mask_directory(setup, reduction_metadata,
                                 bpm_directory_path=None,
                                 verbose=False, log=None):
    '''
    This found all the images.

    :param object reduction_metadata: the metadata object
    :param string images_directory_path: the directory of the images
    :param boolean verbose: switch to True to have more informations

    :return: the list of images (strings)
    :rtype: list
    '''

    if 'BPM_PATH' in reduction_metadata.data_architecture[1].keys():
        reduction_metadata.update_a_cell_to_layer('data_architecture', 0, 'BPM_PATH', bpm_directory_path)

    else:

        reduction_metadata.add_column_to_layer('data_architecture',
                                               'BPM_PATH', [bpm_directory_path],
                                               new_column_format=None,
                                               new_column_unit=None)

    logs.ifverbose(log, setup, 'Set bad pixel mask directory')


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
                   'Attempting to open image ' + \
                   os.path.join(image_directory_path, image_name))

    try:

        image_data = fits.open(os.path.join(image_directory_path, image_name),
                               mmap=True)
        image_data = image_data[image_index]

        logs.ifverbose(log, setup, image_name + ' open : OK')

        return image_data

    except IndexError:
        
        logs.ifverbose(log, setup, image_name + \
                ' open : not OK!  Cannot open FITS extension '+str(image_index))

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
            index_saturation = np.where((image_bad_pixel_mask == flag))
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

        if bad_pixel_mask.shape != open_image.data.shape:
            if log != None:
                log.warning('The bad_pixel_mask dimensions does not match image dimensions! '
                            'I created an empty bad pixel mask then ....')
                bad_pixel_mask = np.zeros(open_image.data.shape, int)

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


def save_the_pixel_mask_in_image(reduction_metadata, image_name, master_mask):
    '''
    Construct the global pixel mask  using a bitmask approach.

    :param object reduction_metadata: the metadata object
    :param string image_name: the name of the image
    :param array_like master_mask: the master mask which needs to be kept

    '''
    master_pixels_mask = fits.ImageHDU(master_mask)
    master_pixels_mask.name = 'pyDANDIA_PIXEL_MASK'

    open_image = fits.open(os.path.join(reduction_metadata.data_architecture[1]['IMAGES_PATH'][0], image_name))

    try:
        open_image['pyDANDIA_PIXEL_MASK'] = master_pixels_mask
    except:

        open_image.append(master_pixels_mask)

    open_image.writeto(os.path.join(reduction_metadata.data_architecture[1]['IMAGES_PATH'][0], image_name),
                       overwrite=True)


def update_reduction_metadata_with_config_file(reduction_metadata,
                                               config_dictionnary, log=None):
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

    if reduction_metadata.reduction_parameters[1]:

        for index, key in enumerate(names):
            reduction_metadata.update_a_cell_to_layer('reduction_parameters', 0, key, data[index, 1])


    else:
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


def update_reduction_metadata_headers_summary_with_new_images(setup,
                                                              reduction_metadata,
                                                              new_images, log=None):
    '''
    Update the metadata with the header keywords

    :param object reduction_metadata: the metadata object
    :param list new_images: list of strings

    :return an array containing the needed header info
    :rtype array_like
    '''

    for image_name in new_images:
        layer = reduction_metadata.headers_summary[1]

        open_image = open_an_image(setup, reduction_metadata.data_architecture[1]['IMAGES_PATH'][0],
                                   image_name, log=log)

        header_infos = parse_the_image_header(reduction_metadata, open_image)

        names = np.append('IMAGES', header_infos[:, 0])
        values = np.append(image_name, header_infos[:, 1])
        formats = np.append('S200', header_infos[:, 2])

        if layer:

            reduction_metadata.add_row_to_layer('headers_summary',  values)


        else:

            reduction_metadata.create_headers_summary_layer(names, formats,
                                                            units=None,
                                                            data=values)

    if log != None:
        log.info('Added data on new images to the metadata')


def construct_the_stamps(open_image, stamp_size=None, arcseconds_stamp_size=(110, 110),
                         pixel_scale=None,
                         fraction_of_overlaping_pixels=0.1,number_of_overlaping_pixels=None, log=None):
    '''
    Define the stamps for an image variable kernel definition

    :param object reduction_metadata: the metadata object
    :param list stamp_sizes: list of integer give the X,Y stamp size , i.e [150,52] give 150 pix in X, 52 in Y
    :param tuple arcseconds_stamp_size: list of integer give the X,Y stamp size in arcseconds units
    :param float pixel_scale: pixel scale of the CCD, in arcsec/pix
    :param float fraction_of_overlaping_pixels : half of  number of pixels as 1D substamp fraction 
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
            #we want to distribute the stamp size as evenly as possible
            #that requires to use the corresponding fraction of the envisaged
            #stamp size fits into the frame (ceiling with overlap...)
            subimage_shape = [int(np.ceil(float(full_image_x_size)/x_stamp_size)), int(np.ceil(float(full_image_y_size)/y_stamp_size))]
            x_subsize = int(full_image_x_size/subimage_shape[0])
            y_subsize = int(full_image_y_size/subimage_shape[1])
            #overlapping fraction in pixels
            if number_of_overlaping_pixels:
                overlap_x = number_of_overlaping_pixels
                overlap_y =  number_of_overlaping_pixels                        
            else:
                overlap_x = int(fraction_of_overlaping_pixels * x_subsize)
                overlap_y = int(fraction_of_overlaping_pixels * y_subsize)
           
            subimage_slices = []
            for idx in range(subimage_shape[0]):
                for jdx in range(subimage_shape[1]):            
                    subimage_element = subimage_shape+[idx,jdx]
                    x_subsize, y_subsize = full_image_x_size/subimage_element[0], full_image_y_size/subimage_element[1]

                    xslice = [subimage_element[2] * x_subsize , (subimage_element[2] + 1) * x_subsize]
                    yslice = [subimage_element[3] * y_subsize , (subimage_element[3] + 1) * y_subsize]
                    #this is the slice without overlapping region, but for
                    #obtaining a higher accurracy and to defeat edge effects
                    #we check if the slice starts or ends at the edge and add
                    #the corresponding overlap
        except Exception as e:
            status = 'ERROR'
            report = 'No pixel scale found!'+str(e)
            log.info(status + ': ' + report)
            return status, report, np.zeros(1)


    x_stamps_center = np.arange(int(x_stamp_size / 2), full_image_x_size, x_stamp_size)
    y_stamps_center = np.arange(int(y_stamp_size / 2), full_image_y_size, y_stamp_size)
    if x_stamps_center.size == 0:
        x_stamps_center = np.array([int(x_stamp_size / 2)])
    if y_stamps_center.size == 0:
        y_stamps_center = np.array([int(y_stamp_size / 2)])
    stamps_center_x, stamps_center_y = np.meshgrid(y_stamps_center, x_stamps_center)

    stamps_y_min = stamps_center_y - int(y_stamp_size / 2) - overlap_y
    mask = stamps_y_min < 0
    stamps_y_min[mask] = 0

    stamps_y_max = stamps_center_y + int(y_stamp_size / 2) + overlap_y
    mask = stamps_y_max > full_image_y_size
    stamps_y_max[mask] = full_image_y_size

    stamps_x_min = stamps_center_x - int(x_stamp_size / 2) - overlap_x
    mask = stamps_x_min < 0
    stamps_x_min[mask] = 0
    stamps_x_max = stamps_center_x + int(x_stamp_size / 2) + overlap_x
    mask = stamps_x_max > full_image_x_size
    stamps_x_max[mask] = full_image_x_size

    stamps = [[stamps_x_min.shape[1] * i + j, stamps_y_min[i, j], stamps_y_max[i, j], stamps_x_min[i, j],
               stamps_x_max[i, j]]
              for i in range(stamps_x_min.shape[0]) for j in range(stamps_x_min.shape[1])]
    print stamps
    status = 'OK'
    report = 'Completed successfully'
    return status, report, np.array(stamps)


def update_reduction_metadata_stamps(setup, reduction_metadata, open_image,
                                     stamp_size=None, arcseconds_stamp_size=(110, 110),
                                     pixel_scale=None, number_of_overlaping_pixels=25,
                                     log=None):
    '''
    Create the stamps definition in the reduction_metadata

    :param object reduction_metadata: the metadata object
    :param astropy.image open_image: the opened image
    :param list stamp_sizes: list of integer give the X,Y stamp size , i.e [150,52] give 150 pix in X, 52 in Y
    :param tuple arcseconds_stamp_size: list of integer give the X,Y stamp size in arcseconds units
    :param float pixel_scale: pixel scale of the CCD, in arcsec/pix
    :param int number_of_overlaping_pixels : half of  number of pixels in both direction you want overlaping

    '''

    if pixel_scale:
        pass
    else:
        pixel_scale = float(reduction_metadata.reduction_parameters[1]['PIX_SCALE'][0])

    (status, report, stamps) = construct_the_stamps(open_image, stamp_size, arcseconds_stamp_size,
                                                    pixel_scale, number_of_overlaping_pixels=number_of_overlaping_pixels, log=log)

    names = ['PIXEL_INDEX', 'Y_MIN', 'Y_MAX', 'X_MIN', 'X_MAX']
    formats = ['int', 'S200', 'S200', 'S200', 'S200']
    units = ['', 'pixel', 'pixel', 'pixel', 'pixel']

    reduction_metadata.create_stamps_layer(names, formats, units, stamps)

    logs.ifverbose(log, setup, 'Updated reduction metadata stamps')
