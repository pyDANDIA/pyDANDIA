# -*- coding: utf-8 -*-
"""
Created on Tue May 15 16:46:57 2018

@author: rstreet
"""
from os import getcwd, path
from sys import exit
from sys import path as systempath
cwd = getcwd()
systempath.append(path.join(cwd,'../'))
import calibrate_photometry
import numpy as np
import matplotlib.pyplot as plt
import logs
import catalog_utils
import metadata
import pipeline_setup
from astropy.table import Table, Column

cwd = getcwd()
TEST_DATA = path.join(cwd,'data')
TEST_DIR = path.join(cwd,'data','proc',
                        'ROME-FIELD-0002_lsc-doma-1m0-05-fl15_ip')

def test_calc_transform():
    """Function to test the photometric transform function

    Expected calibration function is of the form:
    mag_cal = p[0]*mag + p[1]
    """

    a = [ 0.15, 16.0 ]
    uncertainty = [0.01, 0.5]
    x = np.linspace(10.0,20.0,10)
    y = (a[0]*x) + a[1] + np.random.normal(0.0, scale=0.5)

    p = [ -1.0, -10.0 ]

    (fit,covar_fit) = calibrate_photometry.calc_transform(p, x, y)

    for i in range(0,1,1):
        np.testing.assert_almost_equal(fit[i],a[i],uncertainty[i])

    fig = plt.figure(1)

    xplot = np.linspace(x.min(),x.max(),10)
    yplot = fit[0]*xplot + fit[1]

    plt.plot(x,y,'m.')

    plt.plot(xplot, yplot,'k-')

    plt.xlabel('X')

    plt.ylabel('Y')

    plt.savefig('test_transform_function.png')

    plt.close(1)

def test_fetch_catalog_sources_within_image():

    log = logs.start_stage_log( cwd, 'test_wcs' )

    params = {}
    params['fov'] = 0.196
    params['ra'] = '18:00:17.99'
    params['dec'] = '-28:32:15.2'

    vphas_cat = calibrate_photometry.fetch_catalog_sources_within_image(params,log)

    #catalog_utils.output_vphas_catalog_file('test_vphas_catalog.fits',vphas_cat)

    assert len(vphas_cat) > 0

    logs.close_log(log)

def test_fetch_catalog_sources_from_metadata():

    log = logs.start_stage_log( cwd, 'test_wcs' )

    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DIR})

    meta = metadata.MetaData()
    meta.load_a_layer_from_file( setup.red_dir,
                                'pyDANDIA_metadata.fits',
                                'star_catalog' )

    vphas_cat = calibrate_photometry.fetch_catalog_sources_from_metadata(meta,log)

    print(vphas_cat)

    logs.close_log(log)

def test_parse_phot_calibration_file():

    test_file_path = path.join(TEST_DIR,'../config/phot_calib_gp.json')
    key_list = ['a0', 'a1', 'c0', 'c1', 'c2', 'c3']

    phot_calib = calibrate_photometry.parse_phot_calibration_file(test_file_path)

    assert(type(phot_calib) == type({}))
    for key in key_list:
        assert(key in phot_calib.keys())
        assert(type(phot_calib[key]) == type(1.0))

def test_calc_transform_uncertainty():

    A = [1.0, 0.0]
    x = np.linspace(0.0, 100.0, 100)
    y = A[0]*x + A[1]

    sigma_y2 = calibrate_photometry.calc_transform_uncertainty(A, x, y)

    assert(sigma_y2 == 0.0)

    test_sigma = 0.01
    randoms = np.random.normal(loc=0.0, scale=test_sigma, size=len(y))
    y2 = y + randoms

    sigma_y2 = calibrate_photometry.calc_transform_uncertainty(A, x, y2)

    np.testing.assert_almost_equal(sigma_y2, test_sigma, 2)

def test_calc_calibrated_mags():

    log = logs.start_stage_log( cwd, 'test_calibrate' )
    fit_params = np.array([1.047361046162702, -3.695617826430103])
    covar_fit = np.array([ [0.00030368, -0.00560597], [-0.00560597, 0.10369162] ])
    mag = 19.016
    mag_err = 0.00592
    test_cal_mag = 16.221
    test_cal_mag_err = 0.018

    star_catalog = Table([
                        Column(name='mag', data=np.array([mag])),
                        Column(name='mag_err', data=np.array([mag_err])),
                        Column(name='cal_ref_mag', data=np.array([0.0])),
                        Column(name='cal_ref_mag_err', data=np.array([0.0])),
                        ])

    star_catalog = calibrate_photometry.calc_calibrated_mags(fit_params, covar_fit, star_catalog, log)

    np.testing.assert_almost_equal(star_catalog['cal_ref_mag'], test_cal_mag, 3)
    np.testing.assert_almost_equal(star_catalog['cal_ref_mag_err'], test_cal_mag_err, 3)

    logs.close_log(log)


if __name__ == '__main__':

    test_calc_transform()
    #test_fetch_catalog_sources_within_image()
    #test_fetch_catalog_sources_from_metadata()
    #test_parse_phot_calibration_file()
    #test_calc_transform_uncertainty()
    #test_calc_calibrated_mags()
