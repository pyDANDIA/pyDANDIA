# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 14:07:33 2017

@author: rstreet
"""

import os
import sys
cwd = os.getcwd()
sys.path.append(os.path.join(cwd,'../'))
import numpy as np
import logs
import pipeline_setup
import metadata
import photometry
import catalog_utils
import psf
from astropy.table import Table
from astropy.io import fits

TEST_DATA = os.path.join(cwd,'data')
TEST_DIR = os.path.join(cwd,'data','proc',
                        'ROME-FIELD-0002_lsc-doma-1m0-05-fl15_ip')

def test_run_psf_photometry():
    """Function to test the PSF-fitting photometry module for a single image"""

    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DIR})

    log = logs.start_stage_log( cwd, 'test_photometry' )

    reduction_metadata = metadata.MetaData()
    reduction_metadata.load_a_layer_from_file( setup.red_dir,
                                              'pyDANDIA_metadata.fits',
                                              'reduction_parameters' )
    reduction_metadata.load_a_layer_from_file( setup.red_dir,
                                              'pyDANDIA_metadata.fits',
                                              'images_stats' )
    reduction_metadata.load_a_layer_from_file( setup.red_dir,
                                              'pyDANDIA_metadata.fits',
                                              'data_architecture' )
    reduction_metadata.load_a_layer_from_file( setup.red_dir,
                                              'pyDANDIA_metadata.fits',
                                              'star_catalog' )
    reduction_metadata.load_a_layer_from_file( setup.red_dir,
                                              'pyDANDIA_metadata.fits',
                                              'headers_summary' )
    log.info('Read metadata')

    image_path = os.path.join(reduction_metadata.data_architecture[1]['REF_PATH'][0],
                           reduction_metadata.data_architecture[1]['REF_IMAGE'][0])

    ref_star_catalog = np.zeros((len(reduction_metadata.star_catalog[1]),9))
    ref_star_catalog[:,0] = reduction_metadata.star_catalog[1]['star_index']
    ref_star_catalog[:,1] = reduction_metadata.star_catalog[1]['x_pixel']
    ref_star_catalog[:,2] = reduction_metadata.star_catalog[1]['y_pixel']

    psf_model = psf.get_psf_object('Moffat2D')

    xstar = 194.654006958
    ystar = 180.184967041
    psf_size = 8.0
    x_cen = psf_size + (xstar-int(xstar))
    y_cen = psf_size + (ystar-int(ystar))
    psf_params = [ 5807.59961215, x_cen, y_cen, 7.02930822229, 11.4997891585 ]

    psf_model.update_psf_parameters(psf_params)

    sky_model = psf.ConstantBackground()
    sky_model.background_parameters.constant = 5000.0

    ref_flux = 12.0

    log.info('Performing PSF fitting photometry on '+os.path.basename(image_path))

    ref_star_catalog = photometry.run_psf_photometry(setup,reduction_metadata,
                                                     log,ref_star_catalog,
                                                     image_path,
                                                     psf_model,sky_model,
                                                     centroiding=True,
                                                     diagnostics=True)

    assert ref_star_catalog[:,5].max() > 0.0
    assert ref_star_catalog[:,6].max() > 0.0
    assert ref_star_catalog[:,7].max() > 0.0
    assert ref_star_catalog[:,8].max() > 0.0

    logs.close_log(log)


def test_run_psf_photometry_naylor():
    """Function to test the PSF-fitting photometry module for a single image"""

    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DIR})

    log = logs.start_stage_log( cwd, 'test_photometry' )

    reduction_metadata = metadata.MetaData()
    reduction_metadata.load_a_layer_from_file( setup.red_dir,
                                              'pyDANDIA_metadata.fits',
                                              'reduction_parameters' )
    reduction_metadata.load_a_layer_from_file( setup.red_dir,
                                              'pyDANDIA_metadata.fits',
                                              'images_stats' )
    reduction_metadata.load_a_layer_from_file( setup.red_dir,
                                              'pyDANDIA_metadata.fits',
                                              'data_architecture' )
    reduction_metadata.load_a_layer_from_file( setup.red_dir,
                                              'pyDANDIA_metadata.fits',
                                              'star_catalog' )
    reduction_metadata.load_a_layer_from_file( setup.red_dir,
                                              'pyDANDIA_metadata.fits',
                                              'headers_summary' )
    log.info('Read metadata')

    image_path = os.path.join(reduction_metadata.data_architecture[1]['REF_PATH'][0],
                           reduction_metadata.data_architecture[1]['REF_IMAGE'][0])

    ref_star_catalog = np.zeros((len(reduction_metadata.star_catalog[1]),9))
    ref_star_catalog[:,0] = reduction_metadata.star_catalog[1]['star_index']
    ref_star_catalog[:,1] = reduction_metadata.star_catalog[1]['x_pixel']
    ref_star_catalog[:,2] = reduction_metadata.star_catalog[1]['y_pixel']

    psf_model = psf.get_psf_object('Moffat2D')

    xstar = 194.654006958
    ystar = 180.184967041
    psf_size = 8.0
    x_cen = psf_size + (xstar-int(xstar))
    y_cen = psf_size + (ystar-int(ystar))
    psf_params = [ 5807.59961215, x_cen, y_cen, 7.02930822229, 11.4997891585 ]

    psf_model.update_psf_parameters(psf_params)

    sky_model = psf.ConstantBackground()
    sky_model.background_parameters.constant = 5000.0

    ref_flux = 12.0

    log.info('Performing PSF fitting photometry on '+os.path.basename(image_path))

    ref_star_catalog = photometry.run_psf_photometry(setup,reduction_metadata,
                                                     log,ref_star_catalog,
                                                     image_path,
                                                     psf_model,sky_model,
                                                     ref_flux,
                                                     centroiding=True,
                                                     diagnostics=True)

    assert ref_star_catalog[:,5].max() > 0.0
    assert ref_star_catalog[:,6].max() > 0.0
    assert ref_star_catalog[:,5].max() <= 25.0
    assert ref_star_catalog[:,6].max() <= 10.0

    logs.close_log(log)

def test_sim_run_psf_photometry():
    """Function to test the PSF-fitting photometry module for a single image"""

    # Simulation parameters:
    image_path = os.path.join(TEST_DATA,'sim_star_phot.fits')
    sky_value = 300.0

    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DIR})

    log = logs.start_stage_log( cwd, 'test_photometry' )

    reduction_metadata = metadata.MetaData()
    reduction_metadata.load_a_layer_from_file( setup.red_dir,
                                              'pyDANDIA_metadata.fits',
                                              'reduction_parameters' )
    reduction_metadata.load_a_layer_from_file( setup.red_dir,
                                              'pyDANDIA_metadata.fits',
                                              'headers_summary' )

    row = reduction_metadata.headers_summary[1][-1]
    row['IMAGES'] = os.path.basename(image_path)
    exp_time = float(row['EXPKEY'])

    # Generate test image:
    maxx = 100
    maxy = 100
    image = np.zeros((maxy,maxx))
    image = np.random.rand(image.shape[0],image.shape[1])
    image += sky_value

    # Simulated star parameters
    x_cen = 50.0
    y_cen = 50.0
    psf_radius = 8.0
    psf_params = [ 103301.241291, y_cen, x_cen, 226.750731765,
                  13004.8930993, 103323.763627 ]
    Y_data, X_data = np.indices((maxy,maxx))

    # Adding star to the sim image:
    sim_star = psf.Moffat2D()
    sim_star.update_psf_parameters(psf_params)

    sim_star_image = psf.generate_psf_image('Moffat2D',psf_params,image.shape,psf_radius)
    sim_star_image = sim_star.psf_model(Y_data, X_data, psf_params)
    (star_flux,ferr) = sim_star.calc_flux(Y_data, X_data)
    star_flux = star_flux / exp_time

    corners = psf.calc_stamp_corners(x_cen, y_cen, psf_radius, psf_radius,
                                 image.shape[1], image.shape[0],
                                 over_edge=True)

    image += sim_star_image

    # Output sim image for the record:
    hdu = fits.PrimaryHDU(image)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(image_path, overwrite=True)

    hdu = fits.PrimaryHDU(sim_star_image)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(os.path.join(TEST_DATA,'sim_star_psf.fits'),
                    overwrite=True)

    # Construct a test ref_star_catalog:
    ref_star_catalog = np.zeros( (1,9) )
    ref_star_catalog[0,0] = 1
    ref_star_catalog[0,1] = x_cen
    ref_star_catalog[0,2] = y_cen

    # Initialise th PSF model to be fitted:
    psf_model = psf.get_psf_object('Moffat2D')
    psf_params = [ 1.0, (psf_radius/2.0), (psf_radius/2.0), 226.750731765,
                  13004.8930993, 103323.763627 ]
    psf_model.update_psf_parameters(psf_params)

    sky_model = psf.ConstantBackground()
    sky_model.background_parameters.constant = sky_value
    sky_model.varience = np.sqrt(sky_value)

    ref_star_catalog = photometry.run_psf_photometry(setup,reduction_metadata,log,ref_star_catalog,
                       image_path,psf_model,sky_model,
                       centroiding=False,diagnostics=True, psf_size=None)

    assert ref_star_catalog[0,5] > 0.0
    assert ref_star_catalog[0,5] <= star_flux

def test_sim_run_psf_photometry_naylor():
    """Function to test the PSF-fitting photometry module for a single image"""

    # Simulation parameters:
    image_path = os.path.join(TEST_DATA,'sim_star_phot.fits')
    sky_value = 300.0

    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DIR})

    log = logs.start_stage_log( cwd, 'test_photometry' )

    reduction_metadata = metadata.MetaData()
    reduction_metadata.load_a_layer_from_file( setup.red_dir,
                                              'pyDANDIA_metadata.fits',
                                              'reduction_parameters' )
    reduction_metadata.load_a_layer_from_file( setup.red_dir,
                                              'pyDANDIA_metadata.fits',
                                              'headers_summary' )

    row = reduction_metadata.headers_summary[1][-1]
    row['IMAGES'] = os.path.basename(image_path)
    exp_time = float(row['EXPKEY'])

    # Generate test image:
    maxx = 100
    maxy = 100
    image = np.zeros((maxy,maxx))
    image = np.random.rand(image.shape[0],image.shape[1])
    image += sky_value

    # Simulated star parameters
    x_cen = 50.0
    y_cen = 50.0
    psf_radius = 8.0
    psf_params = [ 103301.241291, psf_radius/2.0, psf_radius/2.0, 226.750731765,
                  13004.8930993, 103323.763627 ]
    Y_data, X_data = np.indices((int(psf_radius),int(psf_radius)))

    # Adding star to the sim image:
    sim_star = psf.Moffat2D()
    sim_star.update_psf_parameters(psf_params)

    sim_star_image = psf.generate_psf_image('Moffat2D',psf_params,image.shape,psf_radius)
    sim_star_image = sim_star.psf_model(Y_data, X_data, psf_params)
    (star_flux,ferr) = sim_star.calc_flux(Y_data, X_data)
    star_flux = star_flux / exp_time

    corners = psf.calc_stamp_corners(x_cen, y_cen, psf_radius, psf_radius,
                                 image.shape[1], image.shape[0],
                                 over_edge=True)

    image[corners[2]:corners[3],corners[0]:corners[1]] += sim_star_image

    # Output sim image for the record:
    hdu = fits.PrimaryHDU(image)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(image_path, overwrite=True)
    hdu = fits.PrimaryHDU(sim_star_image)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(os.path.join(TEST_DATA,'sim_star_psf.fits'),
                    overwrite=True)

    # Construct a test ref_star_catalog:
    ref_star_catalog = np.zeros( (1,9) )
    ref_star_catalog[0,0] = 1
    ref_star_catalog[0,1] = x_cen
    ref_star_catalog[0,2] = y_cen

    # Initialise th PSF model to be fitted:
    psf_model = psf.get_psf_object('Moffat2D')
    psf_params = [ 1.0, (psf_radius/2.0), (psf_radius/2.0), 226.750731765,
                  13004.8930993, 103323.763627 ]
    psf_model.update_psf_parameters(psf_params)

    sky_model = psf.ConstantBackground()
    sky_model.background_parameters.constant = sky_value
    sky_model.varience = np.sqrt(sky_value)

    ref_flux = 12.0

    ref_star_catalog = photometry.run_psf_photometry(setup,reduction_metadata,log,ref_star_catalog,
                       image_path,psf_model,sky_model,ref_flux,
                       centroiding=True,diagnostics=True, psf_size=None)

    assert ref_star_catalog[0,5] > 0.0
    assert ref_star_catalog[0,5] <= star_flux

def test_plot_ref_mag_errors():
    """Function to test the plotting function"""

    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DIR})

    reduction_metadata = metadata.MetaData()
    reduction_metadata.load_a_layer_from_file( setup.red_dir,
                                              'pyDANDIA_metadata.fits',
                                              'star_catalog' )

    idx = reduction_metadata.star_catalog[1]['star_index'].data
    x = reduction_metadata.star_catalog[1]['x_pixel'].data
    y = reduction_metadata.star_catalog[1]['y_pixel'].data
    ra = reduction_metadata.star_catalog[1]['RA_J2000'].data
    dec = reduction_metadata.star_catalog[1]['DEC_J2000'].data
    mag = reduction_metadata.star_catalog[1]['Instr_mag'].data
    merr = reduction_metadata.star_catalog[1]['Instr_mag_err'].data


    ref_star_catalog = []

    for i in range(0,len(idx),1):

        ref_star_catalog.append( [idx[i], x[i], y[i], ra[i], dec[i], mag[i], merr[i]] )

    ref_star_catalog = np.array(ref_star_catalog)

    photometry.plot_ref_mag_errors(setup,ref_star_catalog)

    plot_file = os.path.join(setup.red_dir,'ref','ref_image_phot_errors.png')

    assert os.path.isfile(plot_file)

def test_extract_exptime():


    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DIR})

    reduction_metadata = metadata.MetaData()
    reduction_metadata.load_a_layer_from_file( setup.red_dir,
                                              'pyDANDIA_metadata.fits',
                                              'reduction_parameters' )
    reduction_metadata.load_a_layer_from_file( setup.red_dir,
                                              'pyDANDIA_metadata.fits',
                                              'images_stats' )
    reduction_metadata.load_a_layer_from_file( setup.red_dir,
                                              'pyDANDIA_metadata.fits',
                                              'data_architecture' )
    reduction_metadata.load_a_layer_from_file( setup.red_dir,
                                              'pyDANDIA_metadata.fits',
                                              'headers_summary' )

    image_name = reduction_metadata.data_architecture[1]['REF_IMAGE'][0]

    exp_time = photometry.extract_exptime(reduction_metadata,image_name)

    assert(exp_time == 300.0)


def test_convert_flux_to_mag():

    def nearly_equal(a,b,sig_fig=5):
        return ( a==b or
                 int(a*10**sig_fig) == int(b*10**sig_fig)
               )

    f = 1000.0
    ferr = np.sqrt(f)
    expt = 30.0

    (m,merr,f_scaled,ferr_scaled) = photometry.convert_flux_to_mag(f,ferr,exp_time=expt)

    assert nearly_equal(m, 21.1, sig_fig=1)
    assert nearly_equal(merr, 0.034, sig_fig=3)

def test_calc_calib_mags():

    fit_params = np.array([1.047361046162702, -3.695617826430103])
    covar_fit = np.array([ [0.00030368, -0.00560597], [-0.00560597, 0.10369162] ])
    mag = 19.016
    mag_err = 0.00592

    (cal_mag, cal_mag_err) = photometry.calc_calib_mags(fit_params, covar_fit, mag, mag_err)

    print(cal_mag, cal_mag_err)

if __name__ == '__main__':

    #test_run_psf_photometry()
    #test_plot_ref_mag_errors()
    #test_extract_exptime()
    #test_convert_flux_to_mag()
    #test_run_psf_photometry()
    #test_plot_ref_mag_errors()
    #test_sim_run_psf_photometry()
    #test_sim_run_psf_photometry_naylor()
    test_calc_calib_mags()
