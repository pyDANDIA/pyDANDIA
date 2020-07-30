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

def test_calc_weighted_mean_2D():

    nimages = 10
    nstars = 5
    mean_mag = 17.0
    sigma = 0.03
    test_error = 1.0/((1.0/(sigma*sigma))*nimages)

    data = np.zeros( (nstars,nimages,5) )
    data[:,:,0] = np.linspace(2459000.0, 2459010.0, nimages)
    for j in range(0,nstars,1):
        data[:,j,1] = mean_mag + np.random.randn(nstars)*sigma
        data[:,j,2] = np.random.randn(nstars)*sigma
        data[:,j,3] = mean_mag + np.random.randn(nstars)*sigma
        data[:,j,4] = np.random.randn(nstars)*sigma

    (wmean, werror) = plot_rms.calc_weighted_mean_2D(data, 1, 2)

    assert np.all(wmean == approx(mean_mag, rel=0.02))
    assert np.all(werror == approx(test_error, abs=sigma))


if __name__ == '__main__':

    test_calc_weighted_mean_2D()
