from os import path, remove
from pyDANDIA import crossmatch
from pyDANDIA import normalize_photometry
from pyDANDIA import logs
from astropy.table import Table, Column
from astropy import units as u
import numpy as np
from numpy.testing import assert_almost_equal
import matplotlib.pyplot as plt

def simulate_photometry(nstars, nimages):
    """Simulated dataset, with a sigma=1.0 applied to all stars"""
    sigma = 1.0
    data = np.zeros((nstars, nimages,2))
    mean_mag = np.linspace(14.0, 22.0, nstars)
    dmag = np.resize(mean_mag, (nimages, len(mean_mag)))
    dmag = np.swapaxes(dmag,0,1)
    data[:,:,0] = sigma * np.random.randn(nstars, nimages) + dmag
    data[:,:,1].fill(1.0)

    return data, mean_mag

def test_calc_weighted_rms():
    """This function expects to receive an array of photometry for
    all stars in a single image as a 2D array shaped (nstars, 2),
    with columns of magnitude and magnitude error.
    It also requires a single-column array with the mean magnitudes of
    all stars in the image.
    """

    # Simulated dataset, with a sigma=1.0 applied to all stars
    nstars = 1000
    nimages = 100
    (data, mean_mag) = simulate_photometry(nstars, nimages)

    rms = normalize_photometry.calc_weighted_rms(data, mean_mag)

    fig = plt.figure(1,(10,10))
    plt.hist(rms)
    plt.savefig('test_distro.png')
    plt.close(1)

    drms = abs(rms - 1.0)
    assert (drms < 1.0).all()

def test_calc_weighted_mean():

    # Simulated dataset, with a sigma=1.0 applied to all stars
    nstars = 1000
    nimages = 100
    (data, mean_mag) = simulate_photometry(nstars, nimages)

    (wmean, wmean_error) = normalize_photometry.calc_weighted_mean(data)

    dmean = abs(wmean - mean_mag)
    assert (dmean < 1.0).all()

def test_find_constant_stars():

    # Simulated dataset, with a sigma=1.0 applied to all stars
    nstars = 1000
    nimages = 100
    (sim_phot, mean_mag) = simulate_photometry(nstars, nimages)

    # Replace default high-scatter lightcurves with a sample of constant stars
    constant_idx = np.arange(0,100,10)
    dmag = np.resize(mean_mag[constant_idx], (nimages, len(constant_idx)))
    dmag = np.swapaxes(dmag,0,1)
    sim_phot[constant_idx,:,0] = 0.01 * np.random.randn(len(constant_idx), nimages) \
                                + dmag
    sim_phot[constant_idx,:,1].fill(0.01)

    # Replace a few lightcurves with zero-scatter samples to simulate stars
    # with very few measurements and artifically low scatter.
    poor_data = np.arange(1,nstars,10)
    sim_phot[poor_data,::2,0] = 0.0
    sim_phot[poor_data,::2,1].fill(0.0)

    # Transfer the photometry to the full size array
    data = np.zeros((nstars, nimages, 25))
    data[:,:,23] = sim_phot[:,:,0]
    data[:,:,24] = sim_phot[:,:,1]

    # Simulate a xmatch table
    xmatch = crossmatch.CrossMatchTable()
    xmatch.datasets = Table([
                            Column(name='dataset_code', data=['ROME-FIELD-01_lsc-doma-1m0-05-fa15_ip']),
                            Column(name='dataset_red_dir', data=['/no/path/used']),
                            Column(name='dataset_filter', data=['ip']),
                            Column(name='primary_ref', data=[1]),
                            Column(name='norm_a0', data=[1.0]),
                            Column(name='norm_a1', data=[0.0]),
                            Column(name='norm_covar_0', data=[0.0]),
                            Column(name='norm_covar_1', data=[0.0]),
                            Column(name='norm_covar_2', data=[0.0]),
                            Column(name='norm_covar_3', data=[0.0]),
                            ])
    xmatch.images = Table([
                    Column(name='dataset_code', data=np.array(['ROME-FIELD-01_lsc-doma-1m0-05-fa15_ip']*nimages)),
                    ])

    constant_stars = normalize_photometry.find_constant_stars(xmatch, data)

    assert (constant_stars == constant_idx).all()

def test_calc_phot_normalization():
    nstars = 1000
    factor = 0.005
    ref_phot = np.zeros((nstars,2))
    ref_phot[:,0] = np.linspace(14.0, 22.0, nstars)
    ref_phot[:,1].fill(0.005)
    dset_phot = np.zeros((nstars,2))
    dset_phot[:,0] = np.linspace(14.0, 22.0, nstars)
    dset_phot[:,0] += np.random.randn(nstars)*factor
    dset_phot[:,1].fill(0.01)
    constant_stars = np.arange(0,nstars,1)

    (fit, covar_fit) = normalize_photometry.calc_phot_normalization(ref_phot, dset_phot,
                                                                    constant_stars)
    assert (abs(fit[0]-1.0) < factor)
    assert (fit[1]<factor)
    assert (covar_fit < factor).all()

def test_apply_phot_normalization_single_frame():

    log = logs.start_stage_log( '.', 'postproc_phot_norm' )

    # Simulate photometry and parameters of the photometric calibration
    nstars = 100
    frame_phot_data = np.zeros((nstars,2))
    frame_phot_data[:,0] = np.linspace(14.0, 22.0, nstars)
    frame_phot_data[:,1] = np.linspace(0.001, 0.2, nstars)

    fit = np.array([1.0, 0.5])
    covar_fit = np.array([ [0.00016, -0.0028], [-0.0028, 0.05] ])

    cal_phot = normalize_photometry.apply_phot_normalization_single_frame(fit, covar_fit, frame_phot_data,
                                                0, 1, log)

    np.testing.assert_array_almost_equal(cal_phot[:,0], frame_phot_data[:,0]+fit[1])
    assert ((np.median(cal_phot[:,1]) > 0.0) & (np.median(cal_phot[:,1]) < 0.1))

    logs.close_log(log)

def test_normalize_timeseries_photometry():

    log = logs.start_stage_log( '.', 'postproc_phot_norm' )

    nstars = 100
    nimages = 10
    mag_col = 23
    mag_err_col = 24
    norm_mag_col = 26
    norm_mag_err_col = 27
    phot_data = np.zeros((nstars,nimages,28))


    dmag = np.resize(np.linspace(14.0,22.0,nstars), (nimages, nstars))
    dmag = np.swapaxes(dmag,0,1)
    phot_data[:,:,23] = dmag
    phot_data[:,:,24].fill(0.005)

    image_index = np.arange(0,nimages,1)

    fit = np.array([1.0, 0.5])
    covar_fit = np.array([ [0.00016, -0.0028], [-0.0028, 0.05] ])

    phot_data = normalize_photometry.normalize_timeseries_photometry(phot_data, image_index,
                                                fit, covar_fit,
                                                mag_col, mag_err_col,
                                                norm_mag_col, norm_mag_err_col,
                                                log)

    np.testing.assert_array_almost_equal(phot_data[:,:,26], phot_data[:,:,23]+fit[1])
    assert ((np.median(phot_data[:,:,27]) > 0.0) & (np.median(phot_data[:,:,27]) < 0.1))

    logs.close_log(log)

if __name__ == '__main__':
    #test_calc_weighted_rms()
    #test_calc_weighted_mean()
    #test_find_constant_stars()
    #test_calc_phot_normalization()
    #test_apply_phot_normalization_single_frame()
    test_normalize_timeseries_photometry()
