# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 14:08:54 2017

@author: rstreet
"""
from os import path
from astropy.io import fits
from astropy.table import Table
from astropy.table import Column
from astropy.utils.exceptions import AstropyWarning

import numpy as np
import collections
import warnings
import logs
import os


def update_a_dictionary(dictionary, new_key, new_value):
    '''
    Update a namedtuple dictionary with a new key and new value

    :param namedtuple_object dictionary: the dictionary need to be updated
    :param string new_key: the new key desirec in the new dictionnary
    :param new_value:  the new value associated ot the new key.

    :return new_dictionary:  the updated namedtuple dictionary
    :rtype namedtuple dictionary
    '''

    new_keys = dictionary._fields + (new_key,)
    new_dictionary = collections.namedtuple(dictionary.__name__, new_keys)

    for index, key in enumerate(dictionary._fields):
        value = getattr(dictionary, key)

        setattr(new_dictionary, key, value)

    setattr(new_dictionary, new_key, new_value)

    return new_dictionary


class MetaData:
    """Class defining the data structure produced by the pyDANDIA pipeline
    to hold metadata regarding the reduction of a single dataset, including
    reduction configuration parameters, the data inventory and key measured
    parameters from each stage. 
    """

    def __init__(self):

        # attributes = [astropy.header,astropy.Table]

        self.data_architecture = [None, None]
        self.reduction_parameters = [None, None]
        self.headers_summary = [None, None]
        self.reduction_status = [None, None]
        self.images_stats = [None, None]

        self.stamps = [None, None]

    def create_metadata_file(self, metadata_directory, metadata_name):
        '''
        Create a metadata fits file from scratch

        :param string metadata_directory: the metadata directory where this file gonna be saved
        :param string metadata_name: the name of the metadata file

        '''
        metadata = fits.HDUList()

        self.create_data_architecture_layer(metadata_directory, metadata_name)
        self.create_reduction_status_layer()

        tbhdu1 = fits.BinTableHDU(self.data_architecture[1], header=self.data_architecture[0])
        tbhdu2 = fits.BinTableHDU(self.reduction_status[1], header=self.reduction_status[0])

        tbhdu1.name = tbhdu1.header['name']
        tbhdu2.name = tbhdu2.header['name']

        metadata.append(tbhdu1)
        metadata.append(tbhdu2)

        metadata.writeto(path.join(metadata_directory, metadata_name), overwrite=True)

    def create_a_new_layer(self, layer_name, data_structure, data_columns=None):
        '''
        Add a new layer to the metadata object

        :param string layer_name: the name associated to the layer
        :param list data_structure: a list containing the [[columns names],[columns format],[columns units]]
        :param array_like data_columns: the content of the astropy.table


        '''

        layer_header = fits.Header()
        layer_header.update({'NAME': layer_name})

        names = data_structure[0]
        try:

            data_format = data_structure[1]
        except:

            data_format = None

        try:

            data_units = data_structure[2]
        except:

            data_units = None

        try:

            data = data_columns

        except:

            data = None

        layer_table = Table(data, names=names, dtype=data_format)

        try:

            for index, key_column in enumerate(layer_table.keys()):
                layer_table[key_column].unit = data_units[index]

        except:

            pass

        layer = [layer_header, layer_table]

        setattr(self, layer_name, layer)

    def create_data_architecture_layer(self, metadata_directory, metadata_name):
        '''
        Create the data architecture layer, which contains the different directories paths, names etc...

        :param string metadata_directory: the metadata directory where this file gonna be saved
        :param string metadata_name: the nbame of the metadata file

        '''
        layer_name = 'data_architecture'
        data_structure = [['METADATA_NAME', 'OUTPUT_DIRECTORY'],
                          ]
        data = [[metadata_name], [metadata_directory]]
        self.create_a_new_layer(layer_name, data_structure, data)

    def create_reduction_parameters_layer(self, names, formats, units, data=None):
        '''
        Create the reduction parameters layer, which contains the different informations contains in the config.json
        file

        :param list names: the list of names (string) of the columns
        :param list formats: the list of format (dtype) of the columns
        :param list units: the list of units (string) of the columns
        :param array_like: the data need to fill the astropy.table

        '''
        name = 'reduction_parameters'

        data_structure = [names,
                          formats,
                          units]

        self.create_a_new_layer(name, data_structure, data)

    def create_headers_summary_layer(self, names, formats, units=None, data=None):
        '''
        Create the headers_summary layer, which contains the different informations contains in each image header needed
        by pyDANDIA

        :param list names: the list of names (string) of the columns
        :param list formats: the list of format (dtype) of the columns
        :param list units: the list of units (string) of the columns
        :param array_like: the data need to fill the astropy.table

        '''
        layer_name = 'headers_summary'
        data_structure = [names,
                          formats,
                          units]
        self.create_a_new_layer(layer_name, data_structure, data)

    def create_reduction_status_layer(self):
        '''
        Create the reduction_status layer, which summarizes the status of the reduction for all images vs all stages

        '''
        layer_name = 'reduction_status'
        data_structure = [
            ['IMAGES', 'STAGE_0', 'STAGE_1', 'STAGE_2', 'STAGE_3', 'STAGE_4', 'STAGE_5', 'STAGE_6', 'STAGE_7'],
            ['S200', 'S10', 'S10', 'S10', 'S10', 'S10', 'S10', 'S10', 'S10'],
        ]

        self.create_a_new_layer(layer_name, data_structure)

    def create_images_stats_layer(self):
        '''
        Create the images_stats layer, which contains the FHWM, sky level and correlation parameters for each image.
        This is generated by stage 1 of the pipeline.

        '''
        layer_name = 'images_stats'
        data_structure = [
            ['IM_NAME', 'FWHM_X', 'FWHM_Y', 'SKY', 'CORR_XY'],
            ['S100', 'float', 'float', 'float', 'float'],
            [None, 'arcsec', 'arcsec', 'ADU_counts', None]
        ]

        self.create_a_new_layer(layer_name, data_structure)

    def create_stamps_layer(self, names, formats, units=None, data=None):
        '''
        Create the stamps layer, which contains the stamps index, and coordinates of each frames subdivision

        :param list names: the list of names (string) of the columns
        :param list formats: the list of format (dtype) of the columns
        :param list units: the list of units (string) of the columns
        :param array_like: the data need to fill the astropy.table

        '''
        layer_name = 'stamps'
        data_structure = [names,
                          formats,
                          units]
        self.create_a_new_layer(layer_name, data_structure, data)

    def create_star_catalog_layer(self,data=None,log=None):
        """Function to create the layer in the reduction metadata file
        containing the star catalogue of objects detected within the reference
        image.
        
        :param array_like: the data need to fill the astropy.table
        """
        
        layer_name = 'star_catalog'
        
        names = [ 'star_index', 
                'x_pixel', 'y_pixel', 
                'RA_J2000', 'DEC_J2000',
                'Instr_mag', 'Instr_mag_err',
                'J_mag', 'J_mag_err',
                'H_mag', 'H_mag_err', 
                'Ks_mag', 'Ks_mag_err',
                'psf_star']
        
        formats = [ 'int',
                   'float', 'float',
                   'float', 'float',
                   'float', 'float',
                   'float', 'float',
                   'float', 'float',
                   'float', 'float',
                   'int'
                   ]
                   
        units = [ None, 
                 'pixel', 'pixel',
                 'deg', 'deg',
                 'mag', 'mag',
                 'mag', 'mag',
                 'mag', 'mag',
                 'mag', 'mag',
                 None
                 ]
                 
        data_structure = [ names, 
                         formats, 
                         units]
        
        self.create_a_new_layer(layer_name, data_structure, data)

        if log != None:
            
            log.info('Output reference source catalogue to reduction metadata')

    def load_a_layer_from_file(self, metadata_directory, metadata_name, key_layer):
        '''
        Load into the metadata object the layer from the metadata file.

        :param string metadata_directory: the metadata directory where this file gonna be saved
        :param string metadata_name: the name of the metadata file
        :param string key_layer: the layer which gonna be load from the file


        '''
            
        metadata = fits.open(path.join(metadata_directory,metadata_name), mmap=True)

        layer = metadata[key_layer]

        header = layer.header
        table = Table(layer.data)

        setattr(self, key_layer, [header, table])

    def load_all_metadata(self, metadata_directory, metadata_name):
        '''
        Load into the metadata object all layers contains in the metadata file.

        :param string metadata_directory: the metadata directory where this file gonna be saved
        :param string metadata_name: the name of the metadata file


        '''

        metadata = fits.open(path.join(metadata_directory, metadata_name), mmap=True)

        all_layers = [i.header['NAME'] for i in metadata[1:]]

        for key_layer in all_layers:

            try:
                self.load_a_layer_from_file(metadata_directory, metadata_name, key_layer)
            except:

                print('No Layer with key name :' + key_layer)

    def save_updated_metadata(self, metadata_directory, metadata_name, log=None):
        '''
        Save in the metadata file the updated metadata object (i.e all layers).

        :param string metadata_directory: the metadata directory where this file gonna be saved
        :param string metadata_name: the name of the metadata file


        '''
        all_layers = self.__dict__.keys()

        for key_layer in all_layers:
            layer = getattr(self, key_layer)
            if layer != [None, None]:
                if log != None:
                    log.info('Writing meta data layer ' + key_layer)
                self.save_a_layer_to_file(metadata_directory, metadata_name, key_layer, log=log)

        if log != None:
            log.info('Stored updated metadata')

    def save_a_layer_to_file(self, metadata_directory, metadata_name,
                             key_layer, log=None):
        '''
        Save in the metadata file the updated layer.

        :param string metadata_directory: the metadata directory where this file gonna be saved
        :param string metadata_name: the name of the metadata file
        :param string key layer: the name of the layer need to be saved

        '''

        layer = getattr(self, key_layer)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', AstropyWarning)
            update_layer = fits.BinTableHDU(layer[1], header=layer[0])
            update_layer.name = update_layer.header['name']

        try:
            metadata = fits.open(path.join(metadata_directory, metadata_name),
                                 mmap=True)
            try:

                metadata[key_layer] = update_layer

            except:

                metadata.append(update_layer)

            metadata.writeto(path.join(metadata_directory, metadata_name),
                             overwrite=True)

        except IOError:
            if log != None:
                log.info()
                log.info('ERROR: Cannot output metadata to file ' + \
                         path.join(metadata_directory, metadata_name))

    def transform_2D_table_to_dictionary(self, key_layer):
        '''
        Transform a 2D astropy.table to a collection.namedtuple dictionary

        :param string key_layer: the name of the layer transform to a dictionary

        :return dictionary : a namedutple.dicitonary containing the astropy.table
        :rtype collections.namedtuple
        '''
        layer = getattr(self, key_layer)

        keys = layer[1].keys()

        dictionary = collections.namedtuple(key_layer + '_dictionary', keys)

        for index, key in enumerate(dictionary._fields):
            setattr(dictionary, key, layer[1][key][0])

        return dictionary

    def update_2D_table_with_dictionary(self, key_layer, dictionary):
        '''
        Update a layer with a dictionary

        :param string key_layer: the name of the layer need to be saved
        :param collecitions.namedtuple dictionary: the dictionary that will be translate to an astropy.table

        '''
        layer = getattr(self, key_layer)
        column_names = layer[1].keys()

        for index, key in enumerate(dictionary._fields):

            value = getattr(dictionary, key)

            if key in column_names:

                layer[1][key][0] = value

            else:

                layer[1].add_column(Column([value], name=key, dtype=type(value)))

    def add_row_to_layer(self, key_layer, new_row):
        '''
        Add a row to a specific layer

        :param string key_layer: the name of the layer need to be saved
        :param list new_row: the list of value which gonna be append to the layer

        '''
        layer = getattr(self, key_layer)
        layer[1].add_row(new_row)

    def add_column_to_layer(self, key_layer, new_column_name, new_column_data, new_column_format=None,
                            new_column_unit=None):
        '''
        Add a entire column to a specific layer

        :param string key_layer: the name of the layer need to be saved
        :param dtype new_column_name: the name of the new_column
        :param list new_column_data: the data representing the column added
        :param string new_column_unit: the dunit of the new column



        '''

        layer = getattr(self, key_layer)
        new_column = Column(new_column_data, name=new_column_name.upper(),
                            dtype=new_column_format, unit=new_column_unit)
        layer[1].add_column(new_column)

    def update_row_to_layer(self, key_layer, row_index, new_row):
        '''
        Modify an entire row of the layer

        :param string key_layer: the name of the layer need to be saved
        :param int row_index: the index of the line
        :param list new_row: the new line content

        '''
        layer = getattr(self, key_layer)
        layer[1][row_index] = new_row

    def update_column_to_layer(self, key_layer, key_column, new_column):
        '''
        Modify an entire column of the layer

        :param string key_layer: the name of the layer need to be saved
        :param string key_column: the name  of the column
        :param list new_column: the new line content

        '''
        layer = getattr(self, key_layer)
        layer[1][key_column] = new_column

    def find_all_images(self, setup, reduction_metadata, images_directory_path=None, log=None):
        '''
        This found all the images.

        :param object reduction_metadata: the metadata object
        :param string images_directory_path: the directory of the images
        :param boolean verbose: switch to True to have more informations

        :return: the list of images (strings)
        :rtype: list
        '''

        try:

            path = reduction_metadata.data_architecture[1]['IMAGES_PATH'][0]

        except:

            if images_directory_path:
                path = images_directory_path

                reduction_metadata.add_column_to_layer('data_architecture', 'images_path', [path])

        try:

            list_of_images = [i for i in os.listdir(path) if ('.fits' in i) and ('.gz' not in i) and ('.bz2' not in i)]

            if list_of_images == []:

                logs.ifverbose(log, setup, 'No images to process. I take a rest :)')

                return None


            else:

                logs.ifverbose(log, setup, 'Found ' + str(len(list_of_images)) + \
                               ' images in this dataset')

                return list_of_images

        except:

            logs.ifverbose(log, setup, 'Something went wrong on images search!')

            return None

    def find_images_need_to_be_process(self, setup, list_of_images, stage_number=None,
                                       rerun_all=None,
                                       log=None):
        '''
        This founds the images that need to be processed by the pipeline, i.e not already done.

        :param object reduction_metadata: the metadata object
        :param  list list_of_images: the directory of the images
        :param boolean verbose: switch to True to have more informations

        :return: the new images that need to be processed.
        :rtype: list
        '''

        column_name = 'STAGE_'+str(stage_number)
        if rerun_all:
            for name in list_of_images:
                self.update_a_cell_to_layer('reduction_status', 0,column_name,0)

        layer = self.reduction_status

        try:

            if len(layer[1]) == 0:

                new_images = list_of_images

            else:

                new_images = []

                for name in list_of_images:

                    image_row = np.where(layer[1]['IMAGES'] == name)[0][0]

                    if layer[1][image_row][column_name] != '1':
                        logs.ifverbose(log, setup,
                                       name + ' is a new image to process by stage number: ' + str(stage_number))
                        new_images.append(name)

        except:
            if log != None:
                log.info('Error in scanning for new images to reduce')

        if log != None:
            log.info('Total of ' + str(len(new_images)) + ' images need reduction')

        return new_images

    def update_a_cell_to_layer(self, key_layer, row_index, column_name, new_value):
        '''
        Modify an entire row of the layer

        :param string key_layer: the name of the layer need to be saved
        :param int row_index: the index of the line
        :param string the column name: the column name
        :param new value: the new value of the cell

        '''

        layer = getattr(self, key_layer)
        layer[1][column_name][row_index] = new_value

    def update_reduction_metadata_reduction_status(self, new_images, stage_number=0,
        status = 0, log = None):
        '''
        Update the reduction_status layer with all image of the stage set to status

        :param object reduction_metadata: the metadata object
        :param list new_images: list of string with the new images names
        :param int status: status of stage0 reducitonfor a frame. 0 : not done
                                                                  1 : done
        '''

        layer = self.reduction_status
        number_of_columns = len(layer[1].keys())-1
        if len(layer[1])==0:
            for image in new_images:

                    self.add_row_to_layer('reduction_status',[image]+number_of_columns*[0])

        else:
            column_name = 'STAGE_'+str(stage_number)
            for image in new_images:

                index_image = np.where(layer[1]['IMAGES'] == image)[0][0]
                self.update_a_cell_to_layer('reduction_status', index_image, column_name, status)

        if log != None:
            log.info('Updated the reduction status layer')

    def set_all_reduction_status_to_0(self, log=None):
        '''
            Update the reduction_status layer with all image of the stage set to status

            :param object reduction_metadata: the metadata object
            :param list new_images: list of string with the new images names
            :param int status: status of stage0 reducitonfor a frame. 0 : not done
                                                                      1 : done
        '''


        if self.reduction_status:
            layer = self.reduction_status
            length_table = len(layer[1])

            if length_table !=0:

                for key in layer[1].keys()[1:]:

                    self.update_column_to_layer('reduction_status', key,length_table*['0'])
###
def set_pars(self, par_dict):
    for key, value in par_dict.items():
        setattr(self, key, value)


def set_reduction_paths(self, red_dir):
    """Method to establish the reduction directory path.  The directory
    basename will also be taken to be the reduction code
    e.g. ROME-FIELD-01_lsc_doma-1m0-05-fl15_ip
    and this will be used to set the path to the metadata file,
    e.g. ROME-FIELD-01_lsc_doma-1m0-05-fl15_ip_meta.fits
    """

    self.red_dir = red_dir
    self.red_code = path.basename(self.red_dir)
    self.metadata_file = path.join(self.red_dir, self.red_code + '_meta.fits')


def write(self):
    """Method to output the reduction metadata in the pyDANDIA
    pipeline-standard multi-extension FITS binary table format.
    """

    hdulist = fits.HDUList()

    level0 = self.get_level0()
    hdulist.append(level0)

    level1 = self.get_level1()
    hdulist.append(level1)

    level2 = self.get_level2()
    hdulist.append(level2)

    level3 = self.get_level3()
    hdulist.append(level3)

    level4 = self.get_level4()
    hdulist.append(level4)

    hdulist.writeto(self.metadata_file, clobber=True)
    print('Output metadata to ' + self.metadata_file)


def build_hdu(self, data):
    """Method to construct a Primary Header Data Unit from a list of
    entries of the format:
    list [ self.attribute, keyword, format, comment_string]
    """

    hdu = fits.PrimaryHDU()
    for attr, key, keytype, comment in data:
        value = getattr(self, attr)
        if keytype == 'string':
            value = str(value)
        elif keytype == 'int':
            value = int(value)
        elif keytype == 'float':
            value = float(value)
        hdu.header[key] = (value, comment)

    return hdu


def get_level0(self):
    """Method that defines the FITS header keywords and comments for
    Level 0 of the pyDANDIA metadata file:
    Dataset description parameters
    """

    data = [['field', 'FIELD', 'string', 'Name of target field'],
            ['site', 'SITE', '5A', 'Site code'],
            ['enclosure', 'DOME', '10A', 'Dome code'],
            ['telescope', 'TEL', '20A', 'Telescope'],
            ['instrument', 'CAMERA', '20A', 'Instrument'],
            ['filter', 'FILTER', '20A', 'Filter'],
            ['binx', 'BINX', 'I5', 'Instrument binning factor in x-axis [pix]'],
            ['biny', 'BINY', 'I5', 'Instrument binning factor in y-axis [pix]'],
            ['pixel_scale', 'PIXSCALE', 'E', 'Pixel scale of instrument [arcsec/pix]'],
            ]

    hdu = self.build_hdu(data)

    return hdu


def get_level1(self):
    """Method that defines the FITS header keywords and comments for
    Level 1 of the pyDANDIA metadata file:
    Reduction configuration parameters
    """

    data = [['year', 'YEAR', 'int', 'Year of observations'],
            ['back_var', 'BACKVAR', 'int', 'Switch for a spatially variable differential background'],
            ['coeff2', 'COEFF2', 'float', ''],
            ['coeff3', 'COEFF3', 'float', ''],
            ['datekey', 'DATE-KEY', 'string', 'Name of date keyword in image headers'],
            ['deckey', 'DEC-KEY', 'string', 'Name of Declination keyword in image headers'],
            ['det_thresh', 'DETTHRS', 'float', 'Detection threshold [image sky sigma]'],
            ['diffpro', 'DIFFPRO', 'int', 'Switch for the method of difference image creation'],
            ['expfrac', 'EXPFRAC', 'float', 'Fraction of the exposure time to be added to the UTC'],
            ['expkey', 'EXP-KEY', 'string', 'Name of exposure time keyword in image header'],
            ['filtkey', 'FILT-KEY', 'string', 'Name of filter keyword in image header'],
            ['growsatx', 'GROWSATX', 'float', 'Half saturated pixel box size in the x direction [pix]'],
            ['growsaty', 'GROWSATY', 'float', 'Half saturated pixel box size in the y direction [pix]'],
            ['imagedx', 'IMAGE-DX', 'float', 'Width of image subframe [pix]'],
            ['imagedy', 'IMAGE-DY', 'float', 'Height of image subframe [pix]'],
            ['imagex1', 'IMAGEX1', 'int', 'Subframe starting pixel in x-axis [pix]'],
            ['imagex2', 'IMAGEX2', 'int', 'Subframe end pixel in x-axis [pix]'],
            ['imagexmax', 'IMGXMAX', 'int', 'Last pixel of image area in x-axis [pix]'],
            ['imagexmin', 'IMGXMIN', 'int', 'First pixel of image area in x-axis [pix]'],
            ['imagey1', 'IMAGEY1', 'int', 'Subframe starting pixel in y-axis [pix]'],
            ['imagey2', 'IMAGEY2', 'int', 'Subframe end pixel in y-axis [pix]'],
            ['imageymax', 'IMGYMAX', 'int', 'Last pixel of image area in y-axis [pix]'],
            ['imageymin', 'IMGYMIN', 'int', 'First pixel of image area in y-axis [pix]'],
            ['ker_rad', 'KERRAD', 'float', 'Radius of the kernel pixel array [FWHM]'],
            ['max_nim', 'MAX-NIM', 'int', 'Maximum number of images to combine for the reference image'],
            ['max_sky', 'MAX-SKY', 'float', 'Maximum allowed sky background [counts] for reference image'],
            ['min_ell', 'MIN-ELL', 'float', 'Minimum allowed ellipticity for reference image'],
            ['obskey', 'OBSTKEY', 'string', 'Name of data type keywork in image header'],
            ['obskeyb', 'OBSTBIAS', 'string', 'Obstype entry if image is a bias'],
            ['obskeyd', 'OBSTDARK', 'string', 'Obstype entry if image is a dark'],
            ['obskeyf', 'OBSTFLAT', 'string', 'Obstype entry if image is a flat'],
            ['obskeys', 'OBSTSCI', 'string', 'Obstype entry if image is a science image'],
            ['oscanx1', 'OSCANX1', 'int', 'Starting pixel of overscan region in x [pix]'],
            ['oscanx2', 'OSCANX2', 'int', 'End pixel of overscan region in x [pix]'],
            ['oscany1', 'OSCANY1', 'int', 'Starting pixel of overscan region in y [pix]'],
            ['oscany2', 'OSCANY2', 'int', 'End pixel of overscan region in y[pix]'],
            ['psf_comp_dist', 'PSFDIST', 'float', 'Minimum separation of PSF neighbour stars [PSF FWHM]'],
            ['psf_comp_flux', 'PSFCFLUX', 'float', 'Maximum flux ratio of PSF neighbour stars'],
            ['rakey', 'RA-KEY', 'string', 'Name of RA keyword in image header'],
            ['subframes_x', 'SUBREGX', 'int', 'Number of image subregions in x-axis'],
            ['subframes_y', 'SUBREGY', 'int', 'Number of image subregions in y-axis'],
            ['timekey', 'TIME-KEY', 'string', 'Name of exposure timestamp keyword in image header'],
            ]

    hdu = self.build_hdu(data)

    return hdu


def get_level2(self):
    """Method that defines the FITS header keywords and comments for
    Level 2 of the pyDANDIA metadata file
    Data inventory
    """

    level2 = [[0, 'IMAGE', '100A', ''],
              [1, 'FIELD', '100A', ''],
              [2, 'DATE', '10A', 'UTC'],
              [3, 'TIME', '12A', 'UTC'],
              [4, 'PROCSTAT', '1A', ''],
              ]

    data = np.array(self.inventory)
    table = []
    for col, key, fstr, unit in level2:
        table.append(fits.Column(name=key, format=fstr,
                                 array=data[:, col],
                                 unit=unit))

    tbhdu = fits.BinTableHDU.from_columns(table)

    return tbhdu


def get_level3(self):
    """Method that defines the FITS header keywords and comments for
    Level 2 of the pyDANDIA metadata file
    Image data parameters (~old trendlog.imred)
    """

    level3 = [['image', 'IMAGE', '100A', ''],
              [0, 'HJD', 'E', ''],
              [1, 'EXPTIME', 'E', 's'],
              [2, 'SKYBKGD', 'E', 'counts'],
              [3, 'SKYSIG', 'E', 'counts'],
              [4, 'FWHM', 'E', 'pix'],
              [5, 'NSTARS', 'I', ''],
              [None, 'AIRMASS', 'E', ''],
              [None, 'MOONSEP', 'E', 'degrees'],
              [None, 'MOONFRAC', 'E', '%'],
              ]
    image_list = list(self.imred.keys())
    image_list.sort
    data = []
    for image in image_list:
        data.append(self.imred[image])
    data = np.array(data)
    table = []
    for col, key, fstr, unit in level3:
        if col == 'image':
            table.append(fits.Column(name=key, format=fstr,
                                     array=np.array(image_list),
                                     unit=unit))
        elif col != None and col > 0:
            table.append(fits.Column(name=key, format=fstr,
                                     array=data[:, col],
                                     unit=unit))
        else:
            table.append(fits.Column(name=key, format=fstr,
                                     array=np.zeros(len(data[:, 0])),
                                     unit=unit))
    tbhdu = fits.BinTableHDU.from_columns(table)

    return tbhdu


def get_level4(self):
    """Method that defines the FITS header keywords and comments for
    Level 2 of the pyDANDIA metadata file
    Geometric alignment parameters (~trendlog.gimred)
    """

    level1 = [['image', 'IMAGE', '100A', ''],
              [0, 'A0', 'E', ''],
              [1, 'A1', 'E', 's'],
              [2, 'A2', 'E', 'counts'],
              [3, 'A3', 'E', 'counts'],
              [4, 'A4', 'E', 'pix'],
              [5, 'A5', 'E', ''],
              [6, 'A6', 'E', ''],
              [7, 'NSMATCH', 'I', 'degrees'],
              [8, 'RMSX', 'E', '%'],
              [9, 'RMSY', 'E', '%'],
              ]

    image_list = list(self.gimred.keys())
    image_list.sort
    data = []
    for image in image_list:
        data.append(self.gimred[image])
    data = np.array(data)
    table = []
    for col, key, fstr, unit in level1:
        if col == 'image':
            table.append(fits.Column(name=key, format=fstr,
                                     array=np.array(image_list),
                                     unit=unit))
        elif col != None and col > 0:
            table.append(fits.Column(name=key, format=fstr,
                                     array=data[:, col],
                                     unit=unit))
        else:
            table.append(fits.Column(name=key, format=fstr,
                                     array=np.zeros(len(data[:, 0])),
                                     unit=unit))
    tbhdu = fits.BinTableHDU.from_columns(table)

    return tbhdu
