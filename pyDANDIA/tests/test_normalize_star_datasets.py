from pyDANDIA import normalize_photometry_stars
import numpy as np
import matplotlib.pyplot as plt

def test_bin_lc_in_time():

    ndays = 10
    interval = 0.25  # days
    hjd_min = 2455555.0
    hjds = np.arange(hjd_min, hjd_min+(float(ndays)), interval)
    lc = np.zeros((len(hjds),4))
    lc[:,0] = hjds
    lc[:,1] = np.random.randn(len(hjds)) + 17.0
    lc[:,2] = np.random.randn(len(hjds)) * 1e-3

    binned_lc = normalize_photometry_stars.bin_lc_in_time(lc,
                                                                bin_width=1.0)
    fig = plt.figure(1,(10,10))
    plt.rcParams.update({'font.size': 18})
    plt.plot(lc[:,0]-2450000.0, lc[:,1], 'k.')
    plt.plot(binned_lc[:,0]-2450000.0, binned_lc[:,1], 'gd')
    plt.xlabel('HJD-2450000.0')
    plt.ylabel('Mag')
    plt.savefig('test_lightcurve_binning.png')

if __name__ == '__main__':
    test_bin_lc_in_time()
