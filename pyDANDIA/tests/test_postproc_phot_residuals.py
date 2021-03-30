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

    nstars = 100
    nimages = 10
    ncols = 26
    photometry = np.zeros((nstars,nimages,ncols))

    phot_stats = np.zeros((nstars,3))
    phot_stats[:,0] = np.linspace(16.0, 21.0, nstars)
    phot_stats[:,1] = phot_scatter_model(phot_stats[:,0])
    for j in range(0,nstars,1):
        photometry[j,:,11] = np.random.normal(phot_stats[j,0], scale=phot_stats[j,1], size=nimages)
        photometry[j,:,12] = phot_scatter_model(photometry[j,:,11])
        photometry[j,:,13] = photometry[j,:,11] + 0.01
        photometry[j,:,14] = photometry[j,:,12] + 0.001

    photometry = np.ma.masked_array(photometry, mask=None)

    phot_stats = plot_rms.calc_mean_rms_mag(photometry,log,'calibrated')

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

    assert(photometry.shape[2] == ncols+3)
    assert((photometry[0,0,ncols] == 0.0).all())
    assert((photometry[0,0,ncols+1] == 0.0).all())
    assert((photometry[0,0,ncols+2] == 0.0).all())

    logs.close_log(log)

def test_mask_photometry_array():

    params = test_params()
    log = logs.start_stage_log( params['log_dir'], 'test_postproc_phot' )

    (photometry, phot_stats) = test_photometry(log)

    photometry[0,0,13] = 0.0

    photometry = postproc_phot_residuals.mask_photometry_array(photometry, log, use_calib_mag=True)

    print(photometry)
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
    (photometry, phot_stats) = test_photometry(log)

    nstars = 10
    nimages = 10
    test_value = -1.0
    test_uncert = 0.01
    inv_err = 1.0/(test_uncert*test_uncert)
    test_mean_uncert = 1.0/(inv_err*nstars)
    phot_residuals = np.zeros((nstars, nimages,2))
    phot_residuals[:,:,0].fill(test_value)
    phot_residuals[:,:,1].fill(test_uncert)

    exclude_stars = [5]
    exclude_images = [5]
    phot_residuals[exclude_stars,exclude_images,0] = 0.0
    mask = np.empty(phot_residuals.shape)
    mask.fill(False)
    mask[exclude_stars,exclude_images,0] = True
    include_images = []
    for i in range(0,nimages,1):
        if i not in exclude_images:
            include_images.append(i)

    phot_residuals = np.ma.masked_array(phot_residuals, mask=mask)

    image_residuals = postproc_phot_residuals.calc_image_residuals(photometry, phot_residuals,log)

    assert(image_residuals.shape == (nimages,3))
    assert(image_residuals[include_images,0] == test_value).all()
    assert(image_residuals[include_images,1] == test_mean_uncert).all()
    logs.close_log(log)

def test_apply_image_mag_correction():

    params = test_params()
    log = logs.start_stage_log( params['log_dir'], 'test_postproc_phot' )

    (photometry, phot_stats) = test_photometry(log)

    image_residuals = np.zeros((photometry.shape[1],3))
    image_residuals[:,0].fill(-0.02)
    image_residuals[:,1].fill(0.002)
    image_residuals[:,2] = np.linspace(2456655.0, 2456670.0, len(image_residuals))

    photometry = postproc_phot_residuals.apply_image_mag_correction(image_residuals, photometry, log,
                                                                        'calibrated')

    assert( (photometry[:,:,23]-image_residuals[:,0] == photometry[:,:,13]).all() )
    assert( (photometry[:,:,24] == np.sqrt(photometry[:,:,14]*photometry[:,:,14] + \
                                    image_residuals[:,1]*image_residuals[:,1])).all() )

    logs.close_log(log)

def test_calc_image_rms():

    params = test_params()
    log = logs.start_stage_log( params['log_dir'], 'test_postproc_phot' )

    (photometry, phot_stats) = test_photometry(log)

    phot_residuals = postproc_phot_residuals.calc_phot_residuals(photometry, phot_stats, log)

    rms = postproc_phot_residuals.calc_image_rms(phot_residuals, log)

    assert( type(rms) == type(np.zeros(1)) )
    assert( rms.shape == (photometry.shape[1],) )

    logs.close_log(log)

def test_apply_image_merr_correction():

    params = test_params()
    log = logs.start_stage_log( params['log_dir'], 'test_postproc_phot' )

    (photometry, phot_stats) = test_photometry(log)
    photometry[:,:,23] = photometry[:,:,13]
    photometry[:,:,24] = photometry[:,:,14]

    image_rms = np.array([0.01]*photometry.shape[1])

    photometry = postproc_phot_residuals.apply_image_merr_correction(photometry, image_rms, log, use_calib_mag=True)

    assert( (photometry[:,:,24] == np.sqrt(photometry[:,:,14]*photometry[:,:,14] + \
                                    image_rms*image_rms)).all() )

    logs.close_log(log)

def test_mask_phot_from_bad_images():

    params = test_params()
    log = logs.start_stage_log( params['log_dir'], 'test_postproc_phot' )

    (photometry, phot_stats) = test_photometry(log)

    image_residuals = np.zeros((photometry.shape[1],3))
    image_residuals[:,0].fill(-0.02)
    image_residuals[:,1].fill(0.002)
    image_residuals[:,2] = np.linspace(2456655.0, 2456670.0, len(image_residuals))

    mask_image = 1
    image_residuals[mask_image,0] = -0.4

    photometry = postproc_phot_residuals.mask_phot_from_bad_images(photometry, image_residuals, log)

    mask = np.ma.getmask(photometry)

    assert( (mask[:,mask_image,:] == True).all() )

    logs.close_log(log)

def test_set_photometry_qc_flags():

    params = test_params()
    log = logs.start_stage_log( params['log_dir'], 'test_postproc_phot' )

    (photometry, phot_stats) = test_photometry(log)

    bad_j = [0,3,4]
    bad_i = [0,1,6]
    photometry[bad_j,bad_i,13] = -99.999
    photometry[bad_j,bad_i,14] = -99.999
    mask = np.empty(photometry.shape)
    mask.fill(False)
    mask[bad_j,bad_i,13] = True
    mask[bad_j,bad_i,14] = True
    photometry = np.ma.masked_array(photometry[:,:,:], mask=mask)

    photometry = postproc_phot_residuals.set_photometry_qc_flags(photometry, log)

    assert( (photometry[bad_j, bad_i, 25] == -1).all() )

    logs.close_log(log)

def test_set_star_photometry_qc_flags():

    params = test_params()
    log = logs.start_stage_log( params['log_dir'], 'test_postproc_phot' )

    (photometry, phot_stats) = test_photometry(log)

    photometry = postproc_phot_residuals.set_star_photometry_qc_flags(photometry, phot_stats, log)

    logs.close_log(log)

if __name__ == '__main__':
    #test_grow_photometry_array()
    #test_calc_phot_residuals()
    #test_calc_image_residuals()
    #test_apply_image_mag_correction()
    #test_calc_image_rms()
    #test_apply_image_merr_correction()
    #test_mask_photometry_array()
    #test_mask_phot_from_bad_images()
    #test_set_photometry_qc_flags()
    test_set_star_photometry_qc_flags(
