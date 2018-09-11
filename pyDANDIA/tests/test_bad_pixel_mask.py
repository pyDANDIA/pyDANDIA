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
import mock
import matplotlib.pyplot as plt

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
    
    assert type(bpm.instrument_mask) == type(np.zeros([1]))
    assert bpm.instrument_mask.dtype == int
    assert bpm.instrument_mask.shape[0] > 4000
    assert bpm.instrument_mask.shape[1] > 4000

def test_load_mask():
    """Function to verify that the pipeline can identify and load the most 
    recent example of a bad pixel file from the given configuration directory
    """
    
    setup = pipeline_setup.pipeline_setup(params)
    
    setup.pipeline_config_dir = raw_input('Please enter the path to the config directory: ')
    camera = raw_input('Please enter the camera to search for: ')
    
    bpm = bad_pixel_mask.BadPixelMask()
    
    bpm.load_latest_instrument_mask(camera,setup)

    bpm_list = glob.glob(os.path.join(setup.pipeline_config_dir,'bpm*'+camera+'*fits'))
    
    date_list = []
    
    for f in bpm_list:
        
        date_list.append(str(os.path.basename(f)).replace('.fits','').split('_')[2])
    
    idx = (np.array(date_list)).argsort()
    
    latest_date = str(os.path.basename(bpm_list[idx[-1]])).replace('.fits','').split('_')[2]
    
    assert bpm.camera == camera
    assert bpm.dateobs == latest_date
    assert type(bpm.instrument_mask) == type(np.zeros([1]))
    assert bpm.instrument_mask.shape[0] > 4000
    assert bpm.instrument_mask.shape[1] > 4000

def test_add_image_saturated_pixels():
    """Function to test the identification of saturated pixels from an image"""
    
    bpm = bad_pixel_mask.BadPixelMask()
    
    bpm.create_empty_masks([3,3])
    
    image_bad_pixel_mask = np.zeros((3, 3))
    
    image_bad_pixel_mask[1, 1] = 42

    image = fits.PrimaryHDU(image_bad_pixel_mask)

    bpm.mask_image_saturated_pixels(image, [42])
    
    assert np.allclose(bpm.saturated_pixels, image_bad_pixel_mask / 42)


def test_add_image_low_level_pixels():
    
    bpm = bad_pixel_mask.BadPixelMask()
    
    bpm.create_empty_masks([3,3])
    
    image_bad_pixel_mask = np.zeros((3, 3))
    
    image_bad_pixel_mask[1, 1] = -42

    image = fits.PrimaryHDU(image_bad_pixel_mask)

    bpm.mask_image_low_level_pixels(image, [-42])

    assert np.allclose(bpm.low_pixels, image_bad_pixel_mask / -42)


def test_construct_the_variables_star_mask():
    # not yet implemented
    assert 1 == 1

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
    
    setup = pipeline_setup.pipeline_setup(params)
    
    log = logs.start_pipeline_log(setup.log_dir, 'test_bpm_clusters')
    
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
    
    x = range(30,35,1)
    y = range(25,45,1)
    xx,yy = np.meshgrid(x,y)
    saturated_pixel_mask[yy,xx] = 1
    
    saturated_pixel_mask[41,14] = 1
    saturated_pixel_mask[42,23] = 1
    saturated_pixel_mask[47,37] = 1
    saturated_pixel_mask[10,29] = 1
    saturated_pixel_mask[7,41] = 1
        
    hdu = fits.PrimaryHDU(saturated_pixel_mask)
    hdu.writeto('saturated_pixel_input.fits', overwrite=True)
    
    clusters = bad_pixel_mask.find_clusters_saturated_pixels(setup,
                                                             saturated_pixel_mask,
                                                             image_shape,log)
    
    assert len(clusters) == 7
    for c in clusters:
        assert len(c.pixels) in [ 1, 50, 100 ]
    
    mask = np.zeros(image_shape)
    for ic,c in enumerate(clusters):
        for p in c.pixels:
            mask[p[1],p[0]] = ic+1
    
    hdu = fits.PrimaryHDU(mask)
    hdu.writeto('saturated_pixel_mask.fits', overwrite=True)
    
    logs.close_log(log)

def test_mask_ccd_blooming():
    """Function to test the blooming detection method of BPM"""
    
    setup = pipeline_setup.pipeline_setup(params)
    setup.verbosity = 2
    
    reduction_metadata = mock.MagicMock()
    reduction_metadata.reduction_parameters = [0, {'PSF_SIZE': [8], 
                                                   'MAXVAL': [1.4e5]}]

    log = logs.start_pipeline_log(setup.log_dir, 'test_mask_blooming')
    
    file_path = raw_input('Please enter the path to a bad pixel mask file: ')
    
    hdul = fits.open(file_path)
    
    image = hdul[0].data
    
    image_shape = image.shape
    
    saturated_pixel_mask = np.zeros(image_shape)
    
    idx = np.where(hdul[3].data == 2)
    
    saturated_pixel_mask[idx] = 1
        
    generate = False
    if generate:
        image_shape = (4096,4096)
        
        image = np.random.normal(2000.0,200.0,image_shape)
        #image = np.zeros(image_shape)
        
        x = range(1000,1020,1)
        y = range(800,1200,1)
        xx,yy = np.meshgrid(x,y)
        image[yy,xx] = 1.4e5
        
        x = range(900,1110,1)
        y = range(900,1110,1)
        xx,yy = np.meshgrid(x,y)
        model = 100000.0 * np.exp(-(((xx - 1010) / 16.0)**2 +((yy - 1000) / 16.0)**2) / 2)
        image[yy,xx] += model
        
        saturated_pixel_mask = np.zeros(image_shape)
        
        x = range(1000,1020,1)
        y = range(800,1200,1)
        xx,yy = np.meshgrid(x,y)
        saturated_pixel_mask[yy,xx] = 1
        
        x = range(15,20,1)
        y = range(12,17,1)
        xx,yy = np.meshgrid(x,y)
        saturated_pixel_mask[yy,xx] = 1
        
        saturated_pixel_mask[41,14] = 1
        saturated_pixel_mask[42,23] = 1
        saturated_pixel_mask[47,37] = 1
        saturated_pixel_mask[10,29] = 1
        saturated_pixel_mask[7,41] = 1
        

    hdu = fits.PrimaryHDU(image)
    hdu.writeto('test_image_with_saturated_star.fits', overwrite=True)
    
    hdu = fits.PrimaryHDU(saturated_pixel_mask)
    hdu.writeto('test_image_saturated_mask.fits', overwrite=True)
        
    bpm = bad_pixel_mask.BadPixelMask()
    
    bpm.create_empty_masks(image_shape)
    
    bpm.saturated_pixels = saturated_pixel_mask
    
    bpm.mask_ccd_blooming(setup,reduction_metadata,hdul[0],log,
                          diagnostic_plots=True)
    
    logs.close_log(log)
    
def test_load_banzai_mask():
    
    bpm = bad_pixel_mask.BadPixelMask()
    
    bpm.create_empty_masks((3,3))
    
    image_bad_pixel_mask = np.zeros((3, 3))

    image_bad_pixel_mask[1, 1] = 42

    bpm.load_banzai_mask(image_bad_pixel_mask, [42])

    assert np.allclose(bpm.banzai_mask, image_bad_pixel_mask / 42)

def test_construct_the_pixel_mask():
    
    setup = pipeline_setup.pipeline_setup(params)
    
    log = logs.start_pipeline_log(setup.log_dir, 'test_construct_bpm')
    
    image_dims = [4096, 4096]
    reduction_metadata = mock.MagicMock()
    reduction_metadata.data_architecture = [0, {
        'IMAGES_PATH': ['../tests/data/proc/ROME-FIELD-0002_lsc-doma-1m0-05-fl15_ip/data']}]
    reduction_metadata.reduction_parameters = [0, {'IMAGEY2': [image_dims[0]], 
                                                   'IMAGEX2': [image_dims[1]],
                                                   'MAXVAL': [1.4e5],
                                                   'INSTRID': ['fl15'],
                                                   'PSF_SIZE': [8.0]}]
    
    image_bad_pixel_mask = np.zeros(image_dims)

    image_bad_pixel_mask += 89

    image = fits.PrimaryHDU(image_bad_pixel_mask)

    bpm = bad_pixel_mask.construct_the_pixel_mask(setup, reduction_metadata, 
                                                  image, 
                                                  image_bad_pixel_mask, [8],
                                                  log, low_level=0)
    
    logs.close_log(log)

    assert np.allclose(bpm.master_mask, image_bad_pixel_mask - 89)
    
def test_create_empty_masks():
    
    bpm = bad_pixel_mask.BadPixelMask()
    
    image_dims = [ 100, 200 ]
    
    bpm.create_empty_masks(image_dims)
    
    for mask in [ 'instrument_mask', 'saturated_pixels', 'low_pixels', 'banzai_mask' ]:
        
        data = getattr(bpm,mask)
        
        assert data.shape[0] == image_dims[0]
        assert data.shape[1] == image_dims[1]

def test_find_columns_with_blooming():

    image_dims = [4096, 4096]
    
    file_path = raw_input('Please enter the path to a bad pixel mask file: ')
    
    with fits.open(file_path) as hdul:
        
        image = hdul[0].data
        
        image_shape = image.shape
        
        saturated_pixel_mask = np.zeros(image_shape)
        
        idx = np.where(hdul[3].data == 2)
        
        saturated_pixel_mask[idx] = 1
    
    bad_pixel_mask.find_columns_with_blooming(saturated_pixel_mask)

def test_select_pixels_in_theta_range():
    
    x = np.arange(-np.pi, np.pi, (np.pi/180.0))
    y = np.arange(-np.pi, np.pi, (np.pi/180.0))
    theta = np.meshgrid(x,y)[0]

    theta_min = 0.0
    theta_max = (10.0*np.pi)/180.0
    
    idx = bad_pixel_mask.select_pixels_in_theta_range(theta, theta_min, theta_max)
    
    theta_selected = np.zeros(theta.shape)
    theta_selected[idx] = theta[idx]
    
    fig = plt.figure(3)
    plt.subplot(211)
    plt.imshow(theta, cmap='hot')
    plt.subplot(212)
    plt.imshow(theta_selected,cmap='hot')
    plt.colorbar()  
    
    plt.savefig('test_pixel_selection_by_theta.png')
    plt.close(3)
    
    assert len(idx[0]) == 3600
    assert len(idx[1]) == 3600

def test_measure_mask_radius():
    
    x = np.arange(0.0,200.0,1.0)
    y = 20.0 * np.e**(-0.05*x)
    threshold = 10.0
    
    mask_radius = bad_pixel_mask.measure_mask_radius(x,y, 10000.0, threshold)
    
    fig = plt.figure(3)
    plt.plot(x,y,'k-')
    y2 = np.arange(0.0,y.max(),1.0)
    plt.plot([mask_radius]*len(y2), y2, 'r-.')
    plt.plot(x, [threshold]*len(x), 'b--')
    plt.savefig('test_measure_mask_radius.png')
    plt.close(3)
                              
    idx = np.where(y < threshold)
    
    assert mask_radius == x[idx[0][0]]
    
if __name__ == '__main__':
    
    #test_read_mask()
    #test_load_mask()
    #test_load_banzai_mask()
    #test_add_image_saturated_pixels()
    #test_add_image_low_level_pixels()
    #test_id_neighbouring_pixels()
    #test_find_clusters_saturated_pixels()
    #test_create_empty_masks()
    test_construct_the_pixel_mask()
    #test_find_columns_with_blooming()
    #test_select_pixels_in_theta_range()
    #test_measure_mask_radius()
    #test_mask_ccd_blooming()