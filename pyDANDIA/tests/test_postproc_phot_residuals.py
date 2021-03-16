import numpy as np
from pyDANDIA import logs
from pyDANDIA import postproc_phot_residuals
from pyDANDIA import plot_rms

def test_params():
    params = {'primary_ref': 'ROME-FIELD-01_lsc-doma-1m0-05-fa15_ip',
              'datasets': { 'ROME-FIELD-01_lsc-doma-1m0-05-fa15_ip': ['primary_ref', '/Users/rstreet1/OMEGA/test_data/ROME-FIELD-01_lsc-doma-1m0-05-fa15_ip/', 'ip'],
                            'ROME-FIELD-01_lsc-doma-1m0-05-fa15_rp' : [ 'non_ref', '/Users/rstreet1/OMEGA/test_data/ROME-FIELD-01_lsc-doma-1m0-05-fa15_rp/', 'rp' ],
                            'ROME-FIELD-01_lsc-domb-1m0-09-fa15_gp' : [ 'non_ref', '/Users/rstreet1/OMEGA/test_data/ROME-FIELD-01_lsc-domb-1m0-09-fa15_gp/', 'gp' ]},
              'file_path': 'crossmatch_table.fits',
              'log_dir': '.'}

    return params

def test_photometry(log):

    nstars = 10
    nimages = 10
    ncols = 25
    photometry = np.zeros((nstars,nimages,ncols))

    phot_stats = np.zeros((nstars,3))
    phot_stats[:,0] = np.linspace(16.0, 21.0, nstars)
    phot_stats[:,1] = phot_scatter_model(phot_stats[:,0])

    for j in range(0,nstars,1):
        photometry[j,:,11] = np.random.normal(phot_stats[j,0], scale=phot_stats[j,1], size=nimages)
        photometry[j,:,12] = phot_scatter_model(photometry[j,:,11])
        photometry[j,:,13] = photometry[j,:,11] + 0.01
        photometry[j,:,14] = photometry[j,:,12] + 0.001

    phot_stats = plot_rms.calc_mean_rms_mag(photometry,log,use_calib_mag=True)

    return photometry, phot_stats

def phot_scatter_model(mags):
    return 0.01 + np.log10(mags)*0.1

def test_grow_photometry_array():

    params = test_params()
    log = logs.start_stage_log( params['log_dir'], 'test_postproc_phot' )

    nstars = 10
    nimages = 10
    ncols = 23
    test_photometry = np.ones((nstars,nimages,ncols))

    photometry = postproc_phot_residuals.grow_photometry_array(test_photometry,log)

    assert(photometry.shape[2] == ncols+2)
    assert((photometry[0,0,ncols] == 0.0).all())
    assert((photometry[0,0,ncols+1] == 0.0).all())

    logs.close_log(log)

def test_calc_phot_residuals():

    params = test_params()
    log = logs.start_stage_log( params['log_dir'], 'test_postproc_phot' )

    (photometry, phot_stats) = test_photometry(log)

    phot_residuals = postproc_phot_residuals.calc_phot_residuals(photometry, phot_stats, log)

    assert(type(phot_residuals) == type(photometry))
    assert(phot_residuals.shape == (photometry.shape[0], photometry.shape[1],2))

    for j in range(0,len(phot_stats),1):
        assert(phot_residuals[j,:,0] == photometry[j,:,13] - phot_stats[j,0]).all()

    logs.close_log(log)

def test_calc_image_residuals():

    params = test_params()
    log = logs.start_stage_log( params['log_dir'], 'test_postproc_phot' )

    nstars = 10
    nimages = 10
    test_value = -1.0
    test_uncert = 0.01
    inv_err = 1.0/(test_uncert*test_uncert)
    test_mean_uncert = 1.0/(inv_err*nstars)
    phot_residuals = np.zeros((nstars, nimages,2))
    phot_residuals[:,:,0].fill(test_value)
    phot_residuals[:,:,1].fill(test_uncert)

    image_residuals = postproc_phot_residuals.calc_image_residuals(phot_residuals,log)

    assert(image_residuals[:,0] == test_value).all()
    assert(image_residuals[:,1] == test_mean_uncert).all()
    logs.close_log(log)

if __name__ == '__main__':
    #test_grow_photometry_array()
    #test_calc_phot_residuals()
    test_calc_image_residuals()
