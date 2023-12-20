import numpy as np
import mock
import pytest
import os
from os import getcwd, path
from sys import path as systempath
import collections
from astropy.io import fits
from astropy.table import Table, Column
import copy
from skimage import transform as tf
from skimage.measure import ransac

cwd = getcwd()
systempath.append(path.join(cwd, '../'))

import stage4
import metadata


def test_open_an_image():
    setup = mock.MagicMock()

    image = stage4.open_an_image(setup, './', 'dummy.fits')

    assert image is None

    image = stage4.open_an_image(setup, '../tests/data/proc/ROME-FIELD-0002_lsc-doma-1m0-05-fl15_ip/data',
                                 'lsc1m005-fl15-20170418-0131-e91_cropped.fits')
    assert image is not None


def test_find_x_y_shifts_from_the_reference_image():
    setup = mock.MagicMock()

    reference_image = np.zeros((5, 5))
    reference_image[1, 1] = 1

    target_image = np.zeros((5, 5))
    target_image[1, 1] = 1
    x_new_center, y_new_center, x_shift, y_shift = stage4.find_x_y_shifts_from_the_reference_image(setup,
                                                                                                   reference_image,
                                                                                                   target_image,
                                                                                                   edgefraction=1.0,
                                                                                                   log=None)

    assert x_new_center == 2
    assert y_new_center == 2
    assert x_shift == 0
    assert y_shift == 0


def test_correlation_shift():
    reference_image = np.zeros((3, 3))
    reference_image[1, 1] = 1

    target_image = np.zeros((3, 3))
    target_image[2, 2] = 1
    x_shift, y_shift = stage4.correlation_shift(reference_image, target_image)
    
    assert x_shift == 0
    assert y_shift == 0


def test_convolve_image_with_a_psf():
    reference_image = np.ones((3, 3))


    psf = np.zeros((3, 3))
    psf[1, 1] = 1
    convolution = stage4.convolve_image_with_a_psf(reference_image, psf, fourrier_transform_psf=None,
                                                   fourrier_transform_image=None,
                                                   correlate=None, auto_correlation=None)

    assert np.allclose(convolution,reference_image)

def test_crossmatch_catalogs():
    naxis1 = 500
    ycenter = naxis1/2
    naxis2 = 500
    xcenter = naxis2/2
    nstars = 100
    median_bkgd = 100.0
    xshift = 10.0
    yshift = 5.0
    sat_value = 150000.0

    ref_sources = Table(
        [
            Column(name='xcentroid', data=np.random.uniform(1, naxis2-1, nstars)),
            Column(name='ycentroid', data=np.random.uniform(1, naxis2-1, nstars)),
            Column(name='flux', data=median_bkgd + np.random.uniform(10.0, sat_value, nstars))
         ]
    )
    data_sources = copy.copy(ref_sources)
    data_sources['xcentroid'] += xshift
    data_sources['ycentroid'] += yshift

    (pts_data,pts_ref,e_pos) = stage4.crossmatch_catalogs(ref_sources, data_sources)
    print('PTS_DATA: ', pts_data)
    print('PTS_REF: ', pts_ref)

    model_robust, inliers = ransac((pts_ref[:, :2]-xcenter, pts_data[:, :2]-ycenter), tf.AffineTransform,
                                   min_samples=min(50, int(0.1 * len(pts_data))),
                                   residual_threshold=0.1, max_trials=1000)
    print(model_robust.params)
    model_final = copy.copy(model_robust)
    model_final.params[0, 2] += xcenter * (1 - model_final.params[0, 0] - model_final.params[0, 1])
    model_final.params[1, 2] += ycenter * (1 - model_final.params[1, 0] - model_final.params[1, 1])
    print(model_final)

if __name__ == '__main__':
    test_crossmatch_catalogs()