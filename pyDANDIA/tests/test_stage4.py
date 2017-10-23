import numpy as np
import mock
import pytest
import os
from os import getcwd, path
from sys import path as systempath
import collections
from astropy.io import fits

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