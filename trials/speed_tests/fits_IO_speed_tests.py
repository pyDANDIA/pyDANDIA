# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 16:14:48 2017

@author: rstreet
"""

from astropy.io import fits
from astropy.time import Time
from os import remove

def fits_read(file_name,mmap=False):
    """Function to read a FITS file, supporting optional memory-mapping, 
    and provide timing information
    """
    t1 = Time.now()
    hdu = fits.open(file_name,mmap=mmap)
    t2 = Time.now()
    dt = t2 - t1
    print('Read: ',file_name, ' in ',dt.value*1e6,'microsec')
    return hdu

def fits_write(hdu,new_file):
    """Function to output the contents of an HDU to a new FITS file, 
    and provide timing information
    """

    t1 = Time.now()
    hdu_new = fits.PrimaryHDU(hdu[0].data, header=hdu[0].header)
    hdu_new.writeto(new_file)
    t2 = Time.now()
    dt = t2 - t1
    print('Write: ',new_file, ' in ',dt.value*1e6,'microsec')

def fits_IO_speed_test():
    """Function to benchmark the speed at which cropped and full-frame
    Sinistro images can be read by astropy.fits
    """
    
    file_list = ['lsc1m005-fl15-20170614-0130-e91_cropped.fits',
                 'lsc1m005-fl15-20170614-0130-e91.fits']
    
    print('Testing native astropy FITS IO:')
    for image in file_list:
        new_file = image.replace('.fits','_new.fits')
        hdu = fits_read(image)
        fits_write(hdu,new_file)
        remove(new_file)
        hdu.close()
        
    print('\nTesting memory-mapped astropy FITS IO:')
    for image in file_list:
        hdu = fits_read(image,mmap=True)
        print('Write: no memory-mapping option available')

def read_compressed_data():
    """Function to benchmark reading compressed FITS images"""
    
    file_list = ['lsc1m005-fl15-20170614-0130-e91_cropped.fits.gz',
                 'lsc1m005-fl15-20170614-0130-e91.fits.gz',
                 'lsc1m005-fl15-20170614-0130-e91_cropped.fits.bz2',
                 'lsc1m005-fl15-20170614-0130-e91.fits.bz2']
    
    print('\nTesting native astropy FITS IO, compressed, without mmap:')
    for image in file_list:
        extn = image.split('.')[-1]
        new_file = image.replace('.fits.'+extn,'_new.fits.'+extn)
        hdu = fits_read(image)
        hdu.close()

    print('\nTesting native astropy FITS IO, compressed, with mmap:')
    for image in file_list:
        extn = image.split('.')[-1]
        new_file = image.replace('.fits.'+extn,'_new.fits.'+extn)
        hdu = fits_read(image,mmap=True)
        hdu.close()

if __name__ == '__main__':
    fits_IO_speed_test()
    read_compressed_data()