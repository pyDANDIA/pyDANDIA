import numpy as np
from pyDANDIA import logs
from pyDANDIA import postproc_qc
from pyDANDIA import plot_rms
from pyDANDIA import metadata
from pyDANDIA import pipeline_setup
from astropy.table import Table, Column
from astropy.io import fits
import astropy.units as u

def test_params():
    params = {'primary_ref': 'ROME-FIELD-01_lsc-doma-1m0-05-fa15_ip',
              'datasets': { 'ROME-FIELD-01_lsc-doma-1m0-05-fa15_ip': ['primary_ref', '/Users/rstreet1/OMEGA/test_data/ROME-FIELD-01_lsc-doma-1m0-05-fa15_ip/', 'ip'],
                            'ROME-FIELD-01_lsc-doma-1m0-05-fa15_rp' : [ 'non_ref', '/Users/rstreet1/OMEGA/test_data/ROME-FIELD-01_lsc-doma-1m0-05-fa15_rp/', 'rp' ],
                            'ROME-FIELD-01_lsc-domb-1m0-09-fa15_gp' : [ 'non_ref', '/Users/rstreet1/OMEGA/test_data/ROME-FIELD-01_lsc-domb-1m0-09-fa15_gp/', 'gp' ]},
              'file_path': 'crossmatch_table.fits',
              'log_dir': '.',
              'residuals_threshold': 0.05,
              'psexpt_threshold': 0.7}

    return params

def test_photometry(log):

    nstars = 100
    nimages = 10
    ncols = 28
    photometry = np.zeros((nstars,nimages,ncols))

    phot_stats = np.zeros((nstars,3))
    phot_stats[:,0] = np.linspace(16.0, 21.0, nstars)
    phot_stats[:,1] = phot_scatter_model(phot_stats[:,0])
    for j in range(0,nstars,1):
        photometry[j,:,11] = np.random.normal(phot_stats[j,0], scale=phot_stats[j,1], size=nimages)
        photometry[j,:,12] = phot_scatter_model(photometry[j,:,11])
        photometry[j,:,13] = photometry[j,:,11] + 0.01
        photometry[j,:,14] = photometry[j,:,12] + 0.001
        photometry[j,:,19] = np.random.normal(1.0, scale=0.5, size=nimages)

    photometry = np.ma.masked_array(photometry, mask=None)

    phot_stats = plot_rms.calc_mean_rms_mag(photometry,log,'calibrated')

    return photometry, phot_stats

def test_headers_summary(meta):
    names = np.array(['IMAGES', 'EXPKEY', 'OBJKEY', 'OBSKEY', 'DATEKEY', \
                        'TIMEKEY', 'RAKEY', 'DECKEY', 'FILTKEY', \
                        'MOONDKEY', 'MOONFKEY', 'AIRMASS', 'HJD'])
    formats = np.array(['S200']*11 + ['float']*2)
    data = np.array([['lsc1m005-fa15-20190610-0205-e91.fits', '300.0', 'ROME-FIELD-01',
            'EXPOSE', '2019-06-13T02:36:47.449', '02:36:47.449', '17:51:22.5696',
            '-30:03:37.550', 'ip', '60.7240446', '0.799005', '0.91', '2452103.2'],
            ['lsc1m005-fa15-20190610-0183-e91.fits', '300.0', 'ROME-FIELD-01',
            'EXPOSE', '2019-06-13T02:37:00.449', '02:36:47.449', '17:51:22.5696',
            '-30:03:37.550', 'ip', '60.7240446', '0.799005', '0.92', '2452103.21'],
            ['lsc1m005-fa15-20190612-0218-e91.fits', '300.0', 'ROME-FIELD-01',
            'EXPOSE', '2019-06-13T02:38:00.449', '02:36:47.449', '17:51:22.5696',
            '-30:03:37.550', 'ip', '60.7240446', '0.799005', '0.93', '2452103.22'],
            ['lsc1m005-fa15-20190610-0183-e91.fits', '300.0', 'ROME-FIELD-01',
            'EXPOSE', '2019-06-13T02:37:00.449', '02:36:47.449', '17:51:22.5696',
            '-30:03:37.550', 'ip', '60.7240446', '0.799005', '0.92', '2452103.21'],
            ['lsc1m005-fa15-20190612-0218-e91.fits', '300.0', 'ROME-FIELD-01',
            'EXPOSE', '2019-06-13T02:38:00.449', '02:36:47.449', '17:51:22.5696',
            '-30:03:37.550', 'ip', '60.7240446', '0.799005', '0.93', '2452103.22'],
            ['lsc1m005-fl15-20180710-0088-e91.fits', '300.0', 'ROME-FIELD-01',
            'EXPOSE', '2019-06-13T02:37:00.449', '02:36:47.449', '17:51:22.5696',
            '-30:03:37.550', 'ip', '60.7240446', '0.799005', '0.92', '2452103.21'],
            ['lsc1m005-fa15-20190612-0218-e91.fits', '300.0', 'ROME-FIELD-01',
            'EXPOSE', '2019-06-13T02:38:00.449', '02:36:47.449', '17:51:22.5696',
            '-30:03:37.550', 'ip', '60.7240446', '0.799005', '0.93', '2452103.22'],
            ['lsc1m005-fa15-20190610-0183-e91.fits', '300.0', 'ROME-FIELD-01',
            'EXPOSE', '2019-06-13T02:37:00.449', '02:36:47.449', '17:51:22.5696',
            '-30:03:37.550', 'ip', '60.7240446', '0.799005', '0.92', '2452103.21'],
            ['lsc1m005-fa15-20190612-0218-e91.fits', '300.0', 'ROME-FIELD-01',
            'EXPOSE', '2019-06-13T02:38:00.449', '02:36:47.449', '17:51:22.5696',
            '-30:03:37.550', 'ip', '60.7240446', '0.799005', '0.93', '2452103.22'],
            ['lsc1m005-fa15-20190612-0218-e91.fits', '300.0', 'ROME-FIELD-01',
            'EXPOSE', '2019-06-13T02:38:00.449', '02:36:47.449', '17:51:22.5696',
            '-30:03:37.550', 'ip', '60.7240446', '0.799005', '0.93', '2452103.22']
            ])
    meta.create_headers_summary_layer(names, formats, units=None, data=data)

    return meta

def test_data_architecture(meta):

    table_data = [
                Column(name='METADATA_NAME', data=np.array(['pyDANDIA_metadata.fits']), unit=None, dtype='S100'),
                Column(name='OUTPUT_DIRECTORY', data=np.array(['/Users/rstreet1/ROMEREA/test_data/ROME-FIELD-01_lsc-doma-1m0-05-fa15_ip/']), unit=None, dtype='S100'),
                Column(name='IMAGES_PATH', data=np.array(['/Users/rstreet1/ROMEREA/test_data/ROME-FIELD-01_lsc-doma-1m0-05-fa15_ip/data']), unit=None, dtype='S100'),
                Column(name='BPM_PATH', data=np.array(['/Users/rstreet1/ROMEREA/test_data/ROME-FIELD-01_lsc-doma-1m0-05-fa15_ip/data']), unit=None, dtype='S100'),
                Column(name='REF_PATH', data=np.array(['/Users/rstreet1/ROMEREA/test_data/ROME-FIELD-01_lsc-doma-1m0-05-fa15_ip/ref']), unit=None, dtype='S100'),
                Column(name='REF_IMAGE', data=np.array(['lsc1m005-fl15-20180710-0088-e91.fits']), unit=None, dtype='S100'),
                Column(name='KERNEL_PATH', data=np.array(['/Users/rstreet1/ROMEREA/test_data/ROME-FIELD-01_lsc-doma-1m0-05-fa15_ip/kernel']), unit=None, dtype='S100'),
                Column(name='DIFFIM_PATH', data=np.array(['/Users/rstreet1/ROMEREA/test_data/ROME-FIELD-01_lsc-doma-1m0-05-fa15_ip/diff_images']), unit=None, dtype='S100'),
                ]

    layer_header = fits.Header()
    layer_header.update({'NAME': 'data_architecture'})
    layer_table = Table(table_data)
    layer = [layer_header, layer_table]

    setattr(meta, 'data_architecture', layer)

    return meta

def test_star_catalog(meta):
    nstars = 100
    xmax = 4000.0
    ymax = 4000.0

    table_data = [
                Column(name='index', data=np.arange(0,nstars,1), unit=None, dtype='int'),
                Column(name='x', data=np.random.uniform(low=1.0,high=xmax,size=nstars), unit=None, dtype='float'),
                Column(name='y', data=np.random.uniform(low=1.0,high=ymax,size=nstars), unit=None, dtype='float'),
                Column(name='ra', data=np.random.uniform(low=268.0,high=270.0,size=nstars), unit=None, dtype='float'),
                Column(name='dec', data=np.random.uniform(low=-30.0,high=-27.0,size=nstars), unit=None, dtype='float'),
                Column(name='ref_flux', data=np.zeros((nstars)), unit=None, dtype='float'),
                Column(name='ref_flux_error', data=np.zeros((nstars)), unit=None, dtype='float'),
                Column(name='ref_mag', data=np.zeros((nstars)), unit=None, dtype='float'),
                Column(name='ref_mag_error', data=np.zeros((nstars)), unit=None, dtype='float'),
                Column(name='cal_ref_mag', data=np.zeros((nstars)), unit=None, dtype='float'),
                Column(name='cal_ref_mag_error', data=np.zeros((nstars)), unit=None, dtype='float'),
                Column(name='cal_ref_flux', data=np.zeros((nstars)), unit=None, dtype='float'),
                Column(name='cal_ref_flux_error', data=np.zeros((nstars)), unit=None, dtype='float'),
                Column(name='sky_background', data=np.zeros((nstars)), unit=None, dtype='float'),
                Column(name='sky_background_error', data=np.zeros((nstars)), unit=None, dtype='float'),
                Column(name='gaia_source_id', data=np.zeros((nstars)), unit=None, dtype='int'),
                Column(name='gaia_ra', data=np.zeros((nstars)), unit=None, dtype='float'),
                Column(name='gaia_ra_error', data=np.zeros((nstars)), unit=None, dtype='float'),
                Column(name='gaia_dec', data=np.zeros((nstars)), unit=None, dtype='float'),
                Column(name='gaia_dec_error', data=np.zeros((nstars)), unit=None, dtype='float'),
                Column(name='phot_g_mean_flux', data=np.zeros((nstars)), unit=None, dtype='float'),
                Column(name='phot_g_mean_flux_error', data=np.zeros((nstars)), unit=None, dtype='float'),
                Column(name='phot_bp_mean_flux', data=np.zeros((nstars)), unit=None, dtype='float'),
                Column(name='phot_bp_mean_flux_error', data=np.zeros((nstars)), unit=None, dtype='float'),
                Column(name='phot_rp_mean_flux', data=np.zeros((nstars)), unit=None, dtype='float'),
                Column(name='phot_rp_mean_flux_error', data=np.zeros((nstars)), unit=None, dtype='float'),
                Column(name='vphas_source_id', data=np.zeros((nstars)), unit=None, dtype='int'),
                Column(name='vphas_ra', data=np.zeros((nstars)), unit=None, dtype='float'),
                Column(name='vphas_dec', data=np.zeros((nstars)), unit=None, dtype='float'),
                Column(name='gmag', data=np.zeros((nstars)), unit=None, dtype='float'),
                Column(name='gmag_error', data=np.zeros((nstars)), unit=None, dtype='float'),
                Column(name='rmag', data=np.zeros((nstars)), unit=None, dtype='float'),
                Column(name='rmag_error', data=np.zeros((nstars)), unit=None, dtype='float'),
                Column(name='imag', data=np.zeros((nstars)), unit=None, dtype='float'),
                Column(name='imag_error', data=np.zeros((nstars)), unit=None, dtype='float'),
                Column(name='clean', data=np.zeros((nstars)), unit=None, dtype='int'),
                Column(name='psf_star', data=np.zeros((nstars)), unit=None, dtype='float'),
                ]

    layer_header = fits.Header()
    layer_header.update({'NAME': 'star_catalog'})
    layer_table = Table(table_data)
    layer = [layer_header, layer_table]

    setattr(meta, 'star_catalog', layer)

    return meta

def test_stamps_table(meta):

    table_data = [
                Column(name='PIXEL_INDEX', data=np.arange(0,16,1), unit=None, dtype='int'),
                Column(name='YMIN', data=np.array([0,0,0,0,990,990,990,990,1990,1990,1990,1990,2990,2990,2990,2990]),
                                unit=None, dtype='float'),
                Column(name='YMAX', data=np.array([1010,1010,1010,1010,2010,2010,2010,2010,3010,3010,3010,3010,4000,4000,4000,4000]),
                                unit=None, dtype='float'),
                Column(name='XMIN', data=np.array([0,990,1990,2990,0,990,1990,2990,0,990,1990,2990,0,990,1990,2990]),
                                unit=None, dtype='float'),
                Column(name='XMAX', data=np.array([1010,2010,3010,4000,1010,2010,3010,4000,1010,2010,3010,4000,1010,2010,3010,4000]),
                                unit=None, dtype='float'),
                ]

    layer_header = fits.Header()
    layer_header.update({'NAME': 'stamps'})
    layer_table = Table(table_data)
    layer = [layer_header, layer_table]

    setattr(meta, 'stamps', layer)

    return meta

def phot_scatter_model(mags):
    return 0.01 + np.log10(mags)*0.1

def test_grow_photometry_array():

    params = test_params()
    log = logs.start_stage_log( params['log_dir'], 'test_postproc_phot' )

    nstars = 10
    nimages = 10
    ncols = 23
    test_photometry = np.ones((nstars,nimages,ncols))

    photometry = postproc_qc.grow_photometry_array(test_photometry,log)

    assert(photometry.shape[2] == ncols+3)
    assert((photometry[0,0,ncols] == 0.0).all())
    assert((photometry[0,0,ncols+1] == 0.0).all())
    assert((photometry[0,0,ncols+2] == 0.0).all())

    logs.close_log(log)

def test_mask_photometry_array():

    params = test_params()
    log = logs.start_stage_log( params['log_dir'], 'test_postproc_phot' )

    (photometry, phot_stats) = test_photometry(log)

    itest = 0
    jtest = 0
    photometry[jtest,itest,13] = 0.0

    photometry = postproc_qc.mask_photometry_array(photometry, 1, log)

    data = np.ma.getdata(photometry)
    mask = np.ma.getmask(photometry)
    assert(data[jtest,itest,25] == 1.0)
    assert( (mask[jtest,itest,:] == True).all() )

    logs.close_log(log)

def test_calc_phot_residuals():

    params = test_params()
    log = logs.start_stage_log( params['log_dir'], 'test_postproc_phot' )

    (photometry, phot_stats) = test_photometry(log)

    phot_residuals = postproc_qc.calc_phot_residuals(photometry, phot_stats, log)

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

    image_residuals = postproc_qc.calc_image_residuals(photometry, phot_residuals,log)

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

    photometry = postproc_qc.apply_image_mag_correction(image_residuals, photometry, log,
                                                                        'calibrated')

    assert( (photometry[:,:,23]-image_residuals[:,0] == photometry[:,:,13]).all() )
    assert( (photometry[:,:,24] == np.sqrt(photometry[:,:,14]*photometry[:,:,14] + \
                                    image_residuals[:,1]*image_residuals[:,1])).all() )

    logs.close_log(log)

def test_calc_image_rms():

    params = test_params()
    log = logs.start_stage_log( params['log_dir'], 'test_postproc_phot' )

    (photometry, phot_stats) = test_photometry(log)

    phot_residuals = postproc_qc.calc_phot_residuals(photometry, phot_stats, log)

    rms = postproc_qc.calc_image_rms(phot_residuals, log)

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

    photometry = postproc_qc.apply_image_merr_correction(photometry, image_rms, log, use_calib_mag=True)

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
    test_err_code = 2
    image_residuals[mask_image,0] = -0.4

    photometry = postproc_qc.mask_phot_from_bad_images(params, photometry, image_residuals, test_err_code, log)

    mask = np.ma.getmask(photometry)
    data = np.ma.getdata(photometry)

    # Test that affected data are masked, and that the error_code set for the
    # affected datapoints is the error code applied to this error case
    assert( (mask[:,mask_image,:] == True).all() )
    assert( (data[:,mask_image,25] == test_err_code).all() )

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

    photometry = postproc_qc.set_photometry_qc_flags(photometry, log)

    assert( (photometry[bad_j, bad_i, 25] == -1).all() )

    logs.close_log(log)

def test_set_star_photometry_qc_flags():

    params = test_params()
    log = logs.start_stage_log( params['log_dir'], 'test_postproc_phot' )

    (photometry, phot_stats) = test_photometry(log)

    test_star = 0
    photometry[test_star,:,24].fill(1.0)

    photometry = postproc_qc.set_star_photometry_qc_flags(photometry, phot_stats, log)

    assert((photometry[test_star,:,25] < 0).all())

    logs.close_log(log)

def test_calc_ps_exptime():

    params = test_params()
    log = logs.start_stage_log( params['log_dir'], 'test_postproc_phot' )

    (photometry, phot_stats) = test_photometry(log)

    meta = metadata.MetaData()
    meta = test_headers_summary(meta)
    meta = test_data_architecture(meta)

    ps_expt = postproc_qc.calc_ps_exptime(meta, photometry, log)

    assert(type(ps_expt) == type(np.ma.masked_array(np.zeros(1),mask=np.zeros(1))))
    assert(ps_expt.shape == (photometry.shape[0],photometry.shape[1]))
    assert(abs(ps_expt).max() < 5.0)

    logs.close_log(log)

def test_mask_phot_with_bad_psexpt():
    params = test_params()
    log = logs.start_stage_log( params['log_dir'], 'test_postproc_phot' )

    (photometry, phot_stats) = test_photometry(log)

    meta = metadata.MetaData()
    meta = test_headers_summary(meta)
    meta = test_data_architecture(meta)
    test_err_code = 4

    photometry = postproc_qc.mask_phot_with_bad_psexpt(params, meta, photometry, test_err_code, log)

    ps_expt = postproc_qc.calc_ps_exptime(meta, photometry, log)

    threshold = 0.7

    idx = np.where(ps_expt < threshold)
    mask = np.ma.getmask(photometry)
    data = np.ma.getdata(photometry)

    assert( (mask[idx[0],idx[1],:] == True).all() )
    assert( (data[idx[0],idx[1],25] == test_err_code).all() )

    logs.close_log(log)

def test_mask_datapoints_by_image_stamp():

    params = test_params()

    log = logs.start_stage_log( params['log_dir'], 'test_postproc_phot' )

    (photometry, phot_stats) = test_photometry(log)

    meta = metadata.MetaData()
    meta = test_headers_summary(meta)
    meta = test_star_catalog(meta)
    meta = test_stamps_table(meta)
    test_err_code = 16
    nimages = len(meta.star_catalog[1])
    nstamps = len(meta.stamps[1])
    bad_images = [0,1]
    bad_stamps = [2]

    bad_index0 = []
    bad_index1 = []
    for i in bad_images:
        for j in bad_stamps:
            bad_index0.append(i)
            bad_index1.append(j)
    bad_data_index = (np.array(bad_index0), np.array(bad_index1))
    print(bad_data_index)

    photometry = postproc_qc.mask_datapoints_by_image_stamp(photometry, meta, bad_data_index, test_err_code)
    phot_mask = np.ma.getmask(photometry)
    phot_data = np.ma.getdata(photometry)

    for k in range(0,len(bad_data_index[0]),1):
        i = bad_data_index[0][k]
        s = bad_data_index[1][k]
        stamp_dims = meta.stamps[1][s]
        affected_stars = np.where( (meta.star_catalog[1]['x'] >= stamp_dims['XMIN']) & \
                        (meta.star_catalog[1]['x'] < stamp_dims['XMAX']) & \
                        (meta.star_catalog[1]['y'] >= stamp_dims['YMIN']) & \
                        (meta.star_catalog[1]['y'] < stamp_dims['YMAX']) )[0]
        print(i,s, meta.star_catalog[1][affected_stars])

        print(phot_data[affected_stars,i,25])
        print(phot_mask[affected_stars,i,s])
        assert( (phot_data[affected_stars,i,25] >= test_err_code).all() )
        assert( (phot_mask[affected_stars,i,s] == True).all() )

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
    #test_set_star_photometry_qc_flags()
    #test_calc_ps_exptime()
    #test_mask_phot_with_bad_psexpt()
    test_mask_datapoints_by_image_stamp()
