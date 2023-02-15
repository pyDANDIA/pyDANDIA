import numpy as np
from pyDANDIA import field_photometry
from pyDANDIA import crossmatch
from pyDANDIA import logs
from pyDANDIA import metadata
from pyDANDIA import pipeline_setup
from pyDANDIA import hd5_utils
from astropy.table import Table, Column
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
import h5py
from os import path, getcwd
from sys import argv
from datetime import datetime, timedelta

def generate_test_data(params):
    """Function to generate a small-scale dataset in the field dataset
    format for code verification purposes"""

    # Configuration
    params['field_id'] = 'TEST1'
    params['nstars'] = 1000
    params['nimages'] = 20
    params['primary_ref'] = 'TEST1_lsc-doma-1m0-05-fa15_ip'
    params['match_dataset'] = 'TEST1_cpt-doma-1m0-10-fa16_ip'
    params['datasets'] = {'TEST1_lsc-doma-1m0-05-fa15_ip': ['primary_ref', getcwd(), 'ip'],
                          'TEST1_cpt-doma-1m0-10-fa16_ip': ['non-ref', getcwd(), 'ip']}
    params['image_prefix'] = {'TEST1_lsc-doma-1m0-05-fa15_ip': 'lsc1m005-fa15',
                              'TEST1_cpt-doma-1m0-10-fa16_ip': 'cpt1m010-fa16'}
    params['phot_col_prefix'] = [params['primary_ref'].split('_')[1][0:8].replace('-','_'),
                                 params['match_dataset'].split('_')[1][0:8].replace('-','_')]
    params['filters'] = ['ip']
    params['ra_center'] = 270.0  # deg
    params['dec_center'] = -29.8  # deg
    s = SkyCoord(params['ra_center'], params['dec_center'], frame='icrs', unit=(u.deg, u.deg))
    params['ra_str'] = s.ra.to_string(unit=u.hourangle, sep=':', pad=True)
    params['dec_str'] = s.dec.to_string(unit=u.deg, sep=':', pad=True)
    params['field_radius'] = 0.25  # deg
    params['obs_date'] = '2019-06-01'
    params['hjd'] = 2458646.0
    params['texp'] = 300.0
    params['moon_ang_separation'] = 80.0
    params['moon_fraction'] = 0.5
    params['sky'] = 2000.0
    params['norm_a0'] = 1.0
    params['norm_a1'] = 0.4

    # Produce the crossmatch table:
    xmatch = generate_xmatch(params)

    # Add reference frame photometry for both datasets
    xmatch = generate_refframe_photometry(params, xmatch)

    # Generate timeseries photometry
    phot_data = generate_timeseries_photometry(params, xmatch)

    # Store data products:
    xmatch.save(path.join(params['red_dir'],params['field_id']+'_field_crossmatch.fits'))
    output_field_photometry(params, xmatch, phot_data)

def generate_xmatch(params):

    # Initialize crossmatch table
    xmatch = crossmatch.CrossMatchTable()
    xmatch.create(params)

    # Simulate the dataset
    quadrants = np.random.randint(1,5,size=params['nstars'])
    quadrant_ids = np.zeros(params['nstars'])
    for qid in range(1,5,1):
        jdx = np.where(quadrants == qid)[0]
        quadrant_ids[jdx] = np.arange(1,len(jdx)+1,1)
    ras = np.random.uniform(low=params['ra_center']-params['field_radius'],
                            high=params['ra_center']+params['field_radius'],
                            size=params['nstars'])
    ra_errs = np.random.normal(loc=0.2, scale=0.05, size=params['nstars'])
    decs = np.random.uniform(low=params['dec_center']-params['field_radius'],
                            high=params['dec_center']+params['field_radius'],
                            size=params['nstars'])
    dec_errs = np.random.normal(loc=0.2, scale=0.05, size=params['nstars'])
    gaia_ids = np.random.uniform(low=1e8, high=1e9, size=params['nstars'])
    gaia_ids += 4056350970000000000
    gaia_ids = gaia_ids.astype('str')
    gaia_phot_g = np.random.normal(loc=1000, scale=200, size=params['nstars'])
    gaia_phot_gerr = np.random.normal(loc=1.3, scale=1.0, size=params['nstars'])
    gaia_phot_bp = np.random.normal(loc=1000, scale=200, size=params['nstars'])
    gaia_phot_bperr = np.random.normal(loc=1.3, scale=1.0, size=params['nstars'])
    gaia_phot_rp = np.random.normal(loc=1000, scale=200, size=params['nstars'])
    gaia_phot_rperr = np.random.normal(loc=1.3, scale=1.0, size=params['nstars'])
    gaia_pm = np.random.normal(loc=5.0, scale=5.0, size=params['nstars'])
    gaia_pm_ra = np.random.normal(loc=5.0, scale=5.0, size=params['nstars'])
    gaia_pm_ra_err = np.random.normal(loc=0.5, scale=0.5, size=params['nstars'])
    gaia_pm_dec = np.random.normal(loc=5.0, scale=5.0, size=params['nstars'])
    gaia_pm_dec_err = np.random.normal(loc=0.5, scale=0.5, size=params['nstars'])
    gaia_parallax = np.random.normal(loc=0.1, scale=0.1, size=params['nstars'])
    gaia_parallax_err = np.random.normal(loc=0.1, scale=0.1, size=params['nstars'])
    moon_ang_separation = np.random.normal(loc=params['moon_ang_separation'], scale=10.0, size=params['nimages']*2)
    moon_fraction = np.random.normal(loc=params['moon_fraction'], scale=0.1, size=params['nimages']*2)
    airmass = np.linspace(1.0, 1.8, params['nimages']*2)
    sigma_x = np.random.uniform(low=1.2, high=2.8, size=params['nimages']*2)
    sigma_y = np.random.uniform(low=1.2, high=2.8, size=params['nimages']*2)
    sky = np.random.normal(loc=params['sky'],scale=500.0, size=params['nimages']*2)
    median_sky = [51.008]*params['nimages']*2
    fwhm = np.random.normal(loc=4.0,scale=2.0, size=params['nimages']*2)
    corr_xy = np.random.normal(loc=0.0,scale=0.5, size=params['nimages']*2)
    nstars = np.random.normal(loc=params['nstars'],scale=20.0, size=params['nimages']*2)
    frac_sat_pix = np.zeros(params['nimages']*2)
    symmetry = np.random.normal(loc=0.1,scale=0.05, size=params['nimages']*2)
    use_phot = np.ones(params['nimages']*2)
    use_ref = np.zeros(params['nimages']*2)
    shift_x = np.random.normal(loc=0.0,scale=50.0, size=params['nimages']*2)
    shift_y = np.random.normal(loc=0.0,scale=50.0, size=params['nimages']*2)
    pscale = np.random.normal(loc=1.0,scale=0.5, size=params['nimages']*2)
    pscale_err = np.random.normal(loc=1e-08,scale=5e-9, size=params['nimages']*2)
    var_per_pix_diff = np.random.normal(loc=0.02,scale=0.02, size=params['nimages']*2)
    nunmasked = np.array([9.988e5]*params['nimages']*2)
    skew_diff = np.random.normal(loc=-12.0,scale=7.0, size=params['nimages']*2)
    kurtosis_diff = np.random.normal(loc=1e-8,scale=1e-9, size=params['nimages']*2)
    warp0 = np.random.normal(loc=0.0,scale=20.0, size=params['nimages']*2)
    warp1 = np.random.normal(loc=1.0,scale=1e-4, size=params['nimages']*2)
    warp2 = np.random.normal(loc=-0.0002,scale=1e-4, size=params['nimages']*2)
    warp3 = np.random.normal(loc=0.0,scale=1e-8, size=params['nimages']*2)
    warp4 = np.random.normal(loc=-1e-9,scale=1e-9, size=params['nimages']*2)
    warp5 = np.random.normal(loc=1e-9,scale=1e-10, size=params['nimages']*2)
    warp6 = np.random.normal(loc=-60.0,scale=20.0, size=params['nimages']*2)
    warp7 = np.random.normal(loc=0.0002,scale=1e-5, size=params['nimages']*2)
    warp8 = np.random.normal(loc=1.0,scale=1e-5, size=params['nimages']*2)
    warp9 = np.random.normal(loc=-2e-9,scale=1e-10, size=params['nimages']*2)
    warp10 = np.random.normal(loc=-1e-8,scale=1e-9, size=params['nimages']*2)
    warp11 = np.random.normal(loc=0.0,scale=1e-8, size=params['nimages']*2)
    warp12 = np.zeros(params['nimages']*2)
    warp13 = np.zeros(params['nimages']*2)
    warp14 = np.zeros(params['nimages']*2)
    warp15 = np.zeros(params['nimages']*2)
    qcflag = np.zeros(params['nimages']*2)

    # Generate the field index
    column_list = [ Column(name='field_id', data=np.arange(1,params['nstars']+1,1), dtype='int'),
                    Column(name='ra', data=ras, dtype='float'),
                    Column(name='dec', data=decs, dtype='float'),
                    Column(name='quadrant', data=quadrants, dtype='int'),
                    Column(name='quadrant_id', data=quadrant_ids, dtype='int'),
                    Column(name='gaia_source_id', data=gaia_ids, dtype='S19'),
                    Column(name=params['primary_ref']+'_index', data=np.arange(1,params['nstars']+1,1), dtype='int'),
                    Column(name=params['match_dataset']+'_index', data=np.arange(1,params['nstars']+1,1), dtype='int') ]
    xmatch.field_index = Table(column_list)

    # Generate the stars table
    for j in range(0,len(xmatch.field_index),1):
        row = [xmatch.field_index['field_id'][j], xmatch.field_index['ra'][j], xmatch.field_index['dec'][j]]
        row += [0.0]*4*3*9 # No photometry yet
        row += [xmatch.field_index['gaia_source_id'][j],
                xmatch.field_index['ra'][j], ra_errs[j],
                xmatch.field_index['dec'][j], dec_errs[j],
                gaia_phot_g[j], gaia_phot_gerr[j],
                gaia_phot_bp[j], gaia_phot_bperr[j],
                gaia_phot_rp[j], gaia_phot_rperr[j],
                gaia_pm[j], gaia_pm_ra[j], gaia_pm_ra_err[j],
                gaia_pm_dec[j], gaia_pm_dec_err[j],
                gaia_parallax[j], gaia_parallax_err[j]]
        xmatch.stars.add_row(row)

    # Generate the images table entries for each dataset:
    datestr = params['obs_date'].replace('-','')
    start_time = datetime.strptime(params['obs_date']+'T00:00:00.0','%Y-%m-%dT%H:%M:%S.%f')
    dt = timedelta(days = params['texp']/(60.0*60.0*24.0))
    iimage = 0
    for dset,prefix in params['image_prefix'].items():
        for i in range(0,params['nimages'],1):
            iimage += 1
            simage = image_counter(iimage)
            ts = params['hjd']+((iimage*params['texp'])/(60.0*60.0*24.0))
            dateobs = (start_time + (dt*iimage)).strftime('%Y-%m-%dT%H:%M:%S.%f')
            row = [i, prefix+'-'+datestr+'-'+simage+'-e91.fits',
                    dset, params['filters'][0], ts, dateobs, params['texp'],
                    params['ra_str'], params['dec_str'],
                    moon_ang_separation[i], moon_fraction[i], airmass[i],
                    sigma_x[i], sigma_y[i], sky[i], median_sky[i], fwhm[i],
                    corr_xy[i], nstars[i], frac_sat_pix[i], symmetry[i],
                    use_phot[i], use_ref[i], shift_x[i], shift_y[i],
                    pscale[i], pscale_err[i], var_per_pix_diff[i], nunmasked[i],
                    skew_diff[i], kurtosis_diff[i],
                    warp0[i], warp1[i], warp2[i], warp3[i], warp4[i], warp5[i],
                    warp6[i], warp7[i], warp8[i], warp9[i], warp10[i], warp11[i],
                    warp12[i], warp13[i], warp14[i], warp15[i], qcflag[i]]
            xmatch.images.add_row(row)

    # Generate stamps table
    stamps= {0: [0.0, 1010.0, 0.0, 1010.0],
             1: [990.0, 2010.0, 0.0, 1010.0],
             2: [1990.0, 3010.0, 0.0, 1010.0],
             3: [2990.0, 4096.0, 0.0, 1010.0],
             4: [0.0, 1010.0, 990.0, 2010.0],
             5: [990.0, 2010.0, 990.0, 2010.0],
             6: [1990.0, 3010.0, 990.0, 2010.0],
             7: [2990.0, 4096.0, 990.0, 2010.0],
             8: [0.0, 1010.0, 1990.0, 3010.0],
             9: [990.0, 2010.0, 1990.0, 3010.0],
             10: [1990.0, 3010.0, 1990.0, 3010.0],
             11: [2990.0, 4096.0, 1990.0, 3010.0],
             12: [0.0, 1010.0, 2990.0, 4096.0],
             13: [990.0, 2010.0, 2990.0, 4096.0],
             14: [1990.0, 3010.0, 2990.0, 4096.0],
             15: [2990.0, 4096.0, 2990.0, 4096.0]}
    for image_row in xmatch.images:
        for stamp_id,ranges in stamps.items():
            row = [image_row['dataset_code'], image_row['filename'],
                    stamp_id, ranges[0], ranges[1], ranges[2], ranges[3],
                    np.random.normal(loc=1.0,scale=1e-3, size=1),
                    np.random.normal(loc=1e-5,scale=1e-6, size=1),
                    np.random.normal(loc=-0.03,scale=1e-2, size=1),
                    np.random.normal(loc=1e-6,scale=1e-6, size=1),
                    np.random.normal(loc=1.0,scale=1e-5, size=1),
                    np.random.normal(loc=0.005,scale=1e-3, size=1),
                    0.0, 0.0, 1.0]
            xmatch.stamps.add_row(row)

    return xmatch

def generate_refframe_photometry(params, xmatch):

    # Generate the mean magnitudes and magnitude errors for the primary
    # reference dataset
    mean_mags = np.linspace(14.0, 22.0, len(xmatch.stars))
    ZP = 25.0
    flux = 10**( (mean_mags - ZP) / -2.5 )
    flux_err = 1.0 / np.sqrt(flux)
    mag_err = (2.5 / np.log(10.0)) * flux_err / flux
    idx = np.where(mag_err <= 0.001)
    mag_err[idx] = 0.001

    primary_mags_col = 'cal_'+params['filters'][0].replace('p','')+'_mag_'+params['phot_col_prefix'][0]
    primary_merr_col = 'cal_'+params['filters'][0].replace('p','')+'_magerr_'+params['phot_col_prefix'][0]
    xmatch.stars[primary_mags_col] = mean_mags
    xmatch.stars[primary_merr_col] = mag_err

    # Generate the mean mags and errors for the second dataset by applying
    # the configured photometric transform
    dset_mags_col = 'cal_'+params['filters'][0].replace('p','')+'_mag_'+params['phot_col_prefix'][1]
    dset_merr_col = 'cal_'+params['filters'][0].replace('p','')+'_magerr_'+params['phot_col_prefix'][1]
    xmatch.stars[dset_mags_col] = params['norm_a0']*mean_mags + params['norm_a1']
    xmatch.stars[dset_merr_col] = mag_err

    return xmatch

def generate_timeseries_photometry(params, xmatch):
    """Generates a data cube with columns:
    hjd, instrumental_mag, instrumental_mag_err,
    calibrated_mag, calibrated_mag_err, corrected_mag, corrected_mag_err,
    normalized_mag, normalized_mag_err,
    phot_scale_factor, phot_scale_factor_err, stamp_index,
    sub_image_sky_bkgd, sub_image_sky_bkgd_err,
    residual_x, residual_y
    qc_flag"""

    phot_data = np.zeros((params['nstars'],params['nimages']*2,17))

    for dset, prefix in params['image_prefix'].items():
        image_index = np.where(xmatch.images['dataset_code'] == dset)[0]
        dset_code = dset.split('_')[1][0:8].replace('-','_')

        for j in range(0,params['nstars'],1):
            # Add the timestamps relevant to this dataset's images:
            phot_data[j,image_index,0] = xmatch.images['hjd'][image_index]

            # Generate the instrumental magnitude measurements, and
            # reflect these in the calibrated and corrected columns without
            # further changes.  Normalized magnitude columns are left
            # zeroed since this is what we want to test.
            dset_mags_col = 'cal_'+params['filters'][0].replace('p','')+'_mag_'+dset_code
            dset_merr_col = 'cal_'+params['filters'][0].replace('p','')+'_magerr_'+dset_code
            mean_mag = xmatch.stars[dset_mags_col][j]
            mean_merr = xmatch.stars[dset_merr_col][j]
            phot_data[j,image_index,1] = np.random.normal(loc=mean_mag,
                                                            scale=mean_merr,
                                                            size=len(image_index))
            phot_data[j,image_index,2] = np.random.normal(loc=mean_merr,
                                                            scale=mean_merr/10.0,
                                                            size=len(image_index))
            phot_data[j,image_index,3] = phot_data[j,image_index,1]
            phot_data[j,image_index,4] = phot_data[j,image_index,2]
            phot_data[j,image_index,5] = phot_data[j,image_index,1]
            phot_data[j,image_index,6] = phot_data[j,image_index,2]

            # Columns 7,8 are the normalized photometry - left zeroed here

            # Add the photometric scale factor information
            phot_data[j,image_index,9] = xmatch.images['pscale'][image_index]
            phot_data[j,image_index,10] = xmatch.images['pscale_err'][image_index]

            # Add the indices of the stamp where this star is found
            phot_data[j,image_index,11] = np.random.randint(1,15)

            # Add sky background data
            phot_data[j,image_index,12] = xmatch.images['sky'][image_index]
            phot_data[j,image_index,13] = np.sqrt(xmatch.images['sky'][image_index])

            # Add residual x,y offsets
            phot_data[j,image_index,14] = np.random.normal(loc=0.0,scale=5.0,
                                                            size=len(image_index))
            phot_data[j,image_index,15] = np.random.normal(loc=0.0,scale=5.0,
                                                            size=len(image_index))

            # All QC flags left as 0 = OK

    return phot_data

def output_field_photometry(params, xmatch, phot_data):

    for q in range(1,5,1):
        setup = pipeline_setup.PipelineSetup()
        setup.red_dir = params['red_dir']
        filename = params['field_id']+'_quad'+str(q)+'_photometry.hdf5'

        idx = np.where(xmatch.field_index['quadrant'] == q)[0]
        if len(idx) > 0:
            quad_phot_data = phot_data[idx,:,:]
            hd5_utils.write_phot_hd5(setup, quad_phot_data, log=None,
                                    filename=filename)

def image_counter(iimage):
    simage = str(iimage)
    while len(simage) < 4:
        simage = '0'+simage
    return simage

def get_args():
    params = {}
    if len(argv) > 1:
        params['red_dir'] = argv[1]
    else:
        params['red_dir'] = input('Please enter the path to the data directory: ')

    return params

if __name__ == '__main__':
    params = get_args()
    generate_test_data(params)
