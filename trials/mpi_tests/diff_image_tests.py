# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 16:51:30 2017

@author: rstreet
"""
from astropy.io import fits
from astropy.time import Time

def diff_image_test():
    """Function to benchmark how long it takes to calculate a difference of 
    two images.
    """
    
    f1 = 'lsc1m005-fl15-20170614-0130-e91.fits'
    f2 = 'lsc1m005-fl15-20170614-0188-e91.fits'
    
    hdu1 = fits.open(f1,mmap=True)
    image1 = hdu1[0].data
    hdu2 = fits.open(f2,mmap=True)
    image2 = hdu2[0].data

    t1 = Time.now()
    dimage = image1 - image2
    t2 = Time.now()
    dt = t2 - t1
    print('Calculated the difference of two images in ',dt.value*1e6,'microsec')

if __name__ == '__main__':
    diff_image_test()
