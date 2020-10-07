# -*- coding: utf-8 -*-
"""
@author: rstreet
"""
from os import getcwd, path, remove
from pyDANDIA import automatic_pipeline
from pyDANDIA import logs
from pyDANDIA import plot_rms
import numpy as np
from pytest import approx

def generate_test_data():
    nimages = 10
    nstars = 5
    base_mag = 17.0
    sigma = 0.03
    test_error = 1.0/((1.0/(sigma*sigma))*nimages)

    mean_mags = np.zeros(nstars)
    rms = np.zeros(nstars)

    data = np.zeros( (nstars,nimages,15) )
    data[:,:,0] = np.linspace(2459000.0, 2459010.0, nimages)
    for j in range(0,nstars,1):
        data[j,:,11] = base_mag + np.random.randn(nimages)*sigma
        data[j,:,12] = np.random.randn(nimages)*sigma
        data[j,:,13] =base_mag + np.random.randn(nimages)*sigma
        data[j,:,14] = np.random.randn(nimages)*sigma

        mean_mags[j] = np.median(data[j,:,11])
        inv_err = 1.0 / (data[j,:,12]*data[j,:,12])
        rms[j] = np.sqrt( (((data[j,:,11]-mean_mags[j])**2)*inv_err).sum() / inv_err.sum() )

    return data, test_error, sigma, base_mag, mean_mags, rms

def test_calc_weighted_mean_2D():

    (data, test_error, sigma, base_mag, mean_mags, rms) = generate_test_data()

    (wmean, werror) = plot_rms.calc_weighted_mean_2D(data, 11, 12)

    assert np.all(wmean == approx(mean_mags, rel=0.02))
    assert np.all(werror == approx(test_error, abs=sigma))

def test_calc_weighted_rms():

    (data, test_error, sigma, base_mag, mean_mags, test_rms) = generate_test_data()

    (wmean, werror) = plot_rms.calc_weighted_mean_2D(data, 11, 12)

    rms = plot_rms.calc_weighted_rms(data, wmean, 11, 12)
    assert np.all(rms == approx(test_rms, rel=0.05))

if __name__ == '__main__':

    test_calc_weighted_mean_2D()
    test_calc_weighted_rms()
