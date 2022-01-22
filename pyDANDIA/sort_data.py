# -*- coding: utf-8 -*-
"""
Created on Thu May 10 09:41:02 2018

@author: rstreet
"""

from sys import argv, exit
from os import path, mkdir, makedirs
from os import stat as filestatus
from astropy.io import fits
from sys import argv
import glob
from shutil import move
from pyDANDIA import automatic_pipeline

def sort_data(data_dir,option,log=None):
    """Function to sort a directory of FITS frames into per-target, per-filter
    sub-directories"""

    image_list = make_image_list(data_dir)

    for image in image_list:

        ds = get_image_dataset(image,option,log=log)

        if ds:
            sort_image_to_dataset(image,ds,data_dir,log=log)

class Dataset():
    """Class describing a unique dataset from the LCO Network"""

    def __init__(self):
        self.network = None
        self.site = None
        self.enclosure = None
        self.tel = None
        self.instrument = None
        self.filter = None
        self.target = None
        self.id_code = None
        self.facilities = {'Faulkes Telescope South': ['LCOGT','COJ','CLMA','FTS'],
                           'Faulkes Telescope North': ['LCOGT','OGG','CLMA', 'FTN'],
                    	   'Liverpool Telescope': ['LJMU','LAP','LT','LT'] }

    def parse_telescope(self):

        if self.tel in self.facilities.keys():
            (self.network,self.site,self.enclosure,self.tel) = self.facilities[self.tel]

        if self.network == None:
            self.network = 'EXTN'

    def get_dataset_code(self, separate_instruments=False):

        if separate_instruments:
            self.id_code = str(self.target)+'_' + \
                    str(self.site).lower()+'-'+\
                    str(self.enclosure).lower()+'-'+\
                    str(self.tel).lower()+'-'+\
                    str(self.instrument).lower()+'_'+\
                    self.filter
        else:
            if 'fa' in str(self.instrument) or 'fl' in str(self.instrument):
                self.id_code = str(self.target)+ '_' + self.filter
            else:
                self.id_code = str(self.target)+ '_' + str(self.instrument).lower() + '_' + self.filter

def get_image_parameters(hdr):

    ds = Dataset()
    ds.target = hdr['OBJECT'].replace('/','').replace(' ','-')
    ds.site = hdr['SITEID'].replace('/','')
    ds.enclosure = hdr['ENCID'].replace('/','')
    ds.tel = hdr['TELESCOP'].replace('/','')
    ds.instrument = hdr['INSTRUME'].replace('/','').replace('fl','fa')
    try:
        ds.filter = hdr['FILTER']
    except KeyError:
        ds.filter = get_dominant_filter(hdr)

    return ds

def get_dominant_filter(hdr):
    filter_list = []
    for key in ['FILTER1', 'FILTER2', 'FILTER3']:
        if 'air' not in hdr[key] and 'None' not in hdr[key]:
            filter_list.append(hdr[key])

    return filter_list[0]

def get_image_dataset(image, option, log=None):
    """Function to identify what dataset an image belongs to, based on the
    target, instrument and filter information from its FITS header.

    Inputs:
        :param str image: Full path to image FITS file
        :param bool option: Sort data into separate instruments or not
    """

    telescopes = {'Faulkes Telescope South': ['LCOGT','COJ','CLMA','FTS'],
                  'Faulkes Telescope North': ['LCOGT','OGG','CLMA', 'FTN'],
        		'Liverpool Telescope': ['LJMU','LAP','LT','LT'] }

    try:
        hdu = fits.open(image)

        try:
            hdr = hdu[0].header
            ds = get_image_parameters(hdu[0].header)
        except KeyError:
            ds = get_image_parameters(hdu[1].header)

        ds.parse_telescope()

        ds.get_dataset_code(separate_instruments=option)

    except OSError:
        if log!=None:
            log.info('ERROR opening image '+image)
        ds = None

    return ds


def make_image_list(data_dir):
    """Function to check to see if the data directory contains FITS files"""

    if path.isdir(data_dir):

        frame_list = glob.glob(path.join(data_dir,'*.fits'))

        if len(frame_list) == 0:

            print('Cannot find any FITS images in the input directory')
            exit()

    else:

        print('Cannot find input directory: '+data_dir)
        exit()

    return frame_list

def sort_image_to_dataset(image,ds,data_dir,log=None):
    """Function to move the given image to a sub-directory determined by
    its dataset ID code"""

    red_dir = path.join(data_dir,ds.id_code)
    dest_dir = path.join(red_dir,'data')
    unlocked = automatic_pipeline.check_dataset_dir_unlocked(red_dir,log)

    if not path.isdir(dest_dir):
        makedirs(dest_dir)

    if unlocked:
        move(image,path.join(dest_dir,path.basename(image)))
        message = path.basename(image)+' --> '+dest_dir
        log_message(message,log)

    else:
        message = 'Reduction directory '+red_dir+' is locked.  Data will remain in incoming directory'
        log_message(message,log)

def log_message(message,log=None):
    if log == None:
        print(message)
    else:
        log.info(message)

if __name__ == '__main__':

    if len(argv) == 1:
        data_dir = input('Please enter data directory path: ')
        option = input('Separate data for a given target from multiple instruments?  T or F: ')
    else:
        data_dir = argv[1]
        option = argv[2]

    if 't' in str(option).lower():
        option = True
    else:
        option = False

    sort_data(data_dir, option)

    print('Completed data sorting')
