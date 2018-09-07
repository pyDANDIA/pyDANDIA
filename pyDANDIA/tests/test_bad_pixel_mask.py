# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 13:10:43 2018

@author: rstreet, ebachelet
"""
import os
import sys
cwd = os.getcwd()
sys.path.append(os.path.join(cwd,'../'))
import logs
import pipeline_setup
import bad_pixel_mask
import numpy as np
import glob
from astropy.io import fits

params = {'red_dir': os.path.join(cwd, 'data', 'proc', 
                                   'ROME-FIELD-0002_lsc-doma-1m0-05-fl15_ip'), 
          'log_dir': os.path.join(cwd, 'data', 'proc', 
                                   'logs'),
          'pipeline_config_dir': os.path.join(cwd, 'data', 'proc', 
                                   'config'),
          'software_dir': os.path.join(cwd, '..'),
          'verbosity': 2
         }
            
def test_read_mask():
    """Function to check that a standard FITS format bad pixel mask can be 
    read in correctly"""
    
    file_path = raw_input('Please enter the path to a bad pixel mask file: ')
    
    bpm = bad_pixel_mask.BadPixelMask()
    
    bpm.read_mask(file_path)
    
    assert type(bpm.data) == type(np.zeros([1]))
    assert bpm.data.shape[0] > 4000
    assert bpm.data.shape[1] > 4000

def test_load_mask():
    """Function to verify that the pipeline can identify and load the most 
    recent example of a bad pixel file from the given configuration directory
    """
    
    setup = pipeline_setup.pipeline_setup(params)
    
    setup.pipeline_config_dir = raw_input('Please enter the path to the config directory: ')
    camera = raw_input('Please enter the camera to search for: ')
    
    bpm = bad_pixel_mask.BadPixelMask()
    
    bpm.load_latest_mask(camera,setup)

    bpm_list = glob.glob(os.path.join(setup.pipeline_config_dir,'bpm*'+camera+'*fits'))
    
    date_list = []
    
    for f in bpm_list:
        
        date_list.append(str(os.path.basename(f)).replace('.fits','').split('_')[2])
    
    idx = (np.array(date_list)).argsort()
    
    latest_date = str(os.path.basename(bpm_list[idx[-1]])).replace('.fits','').split('_')[2]
    
    assert bpm.camera == camera
    assert bpm.dateobs == latest_date
    assert type(bpm.data) == type(np.zeros([1]))
    assert bpm.data.shape[0] > 4000
    assert bpm.data.shape[1] > 4000

def test_construct_the_saturated_pixel_mask():
    """Function to test the identification of saturated pixels from an image"""
    
    image_bad_pixel_mask = np.zeros((3, 3))
    
    image_bad_pixel_mask[1, 1] = 42

    image = fits.PrimaryHDU(image_bad_pixel_mask)

    bpm = bad_pixel_mask.construct_the_saturated_pixel_mask(image, [42])

    assert np.allclose(bpm, image_bad_pixel_mask / 42)
    
def test_id_neighbouring_pixels():
    """Function to test the method of identifying neighouring or boundary
    pixels"""
    
    test_list = [(49, 49), (49, 50), (49, 51), (50, 49), (50, 51), (51, 49), (51, 50), (51, 51)]
    
    c = bad_pixel_mask.PixelCluster(index=1,xc=50,yc=50)
    
    c.id_neighbouring_pixels((100,100))
    
    assert c.neighbours == test_list
    
    test_list = [(49, 49), (50, 49), (51, 49)]
    
    c = bad_pixel_mask.PixelCluster(index=1,xc=50,yc=50)
    
    c.id_neighbouring_pixels((50,100))
    
    assert c.neighbours == test_list
    
def test_find_clusters_saturated_pixels():
    """Function to test the clustering algorithm for saturated pixels"""
    
    image_shape = (50,50)
    
    saturated_pixel_mask = np.zeros(image_shape)
    
    x = range(10,15,1)
    y = range(10,15,1)
    xx,yy = np.meshgrid(x,y)
    saturated_pixel_mask[yy,xx] = 1
    
    x = range(15,20,1)
    y = range(12,17,1)
    xx,yy = np.meshgrid(x,y)
    saturated_pixel_mask[yy,xx] = 1
    
    hdu = fits.PrimaryHDU(saturated_pixel_mask)
    hdu.writeto('saturated_pixel_mask.fits', overwrite=True)
    
    clusters = bad_pixel_mask.find_clusters_saturated_pixels(saturated_pixel_mask,
                                                             image_shape)
    
    print len(clusters)
    
if __name__ == '__main__':
    
    #test_read_mask()
    #test_load_mask()
    #test_construct_the_saturated_pixel_mask()
    #test_id_neighbouring_pixels()
    test_find_clusters_saturated_pixels()