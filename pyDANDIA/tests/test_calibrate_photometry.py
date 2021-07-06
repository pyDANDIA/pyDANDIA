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

cwd = getcwd()
TEST_DATA = path.join(cwd,'data')
TEST_DIR = path.join(cwd,'data','proc',
                        'ROME-FIELD-0002_lsc-doma-1m0-05-fl15_ip')

def test_calc_transform():
    """Function to test the photometric transform function"""

    a = [ 16.0, 0.15 ]
    x = np.linspace(1.0,100.0,100)
    y = a[0] + (x * a[1]) + np.random.normal(0.0, scale=0.5)

    p = [ -1.0, -10.0 ]

    (fit,covar_fit) = calibrate_photometry.calc_transform(p, x, y)

    assert fit.all() == np.array(a).all()

    fig = plt.figure(1)

    xplot = np.linspace(x.min(),x.max(),10)
    yplot = fit[0] + xplot * fit[1]

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

if __name__ == '__main__':

    #test_calc_transform()
    #test_fetch_catalog_sources_within_image()
    #test_fetch_catalog_sources_from_metadata()
    test_parse_phot_calibration_file()
