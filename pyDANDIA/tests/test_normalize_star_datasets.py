from pyDANDIA import normalize_photometry_stars
from pyDANDIA import logs
from pyDANDIA import crossmatch
import numpy as np
import matplotlib.pyplot as plt
import copy
from astropy.table import Table, Column
from astropy import units as u

def simulate_field_dataproducts(nstars, nimages_dataset):

    # Simulate a xmatch table
    datasets = [ 'ROME-FIELD-01_lsc-doma-1m0-05-fa15_ip',
                 'ROME-FIELD-01_coj-doma-1m0-11-fa12_ip' ]
    datacodes = np.array( [datasets[0]]*nimages_dataset \
                    + [datasets[1]]*nimages_dataset )
    nimages = len(datacodes)
    interval = 0.25  # days
    hjd_min = 2455555.0
    hjds1 = np.linspace(hjd_min, hjd_min+(float(nimages_dataset)*interval), nimages_dataset)
    hjds2 = np.linspace(hjd_min, hjd_min+(float(nimages_dataset)*interval), nimages_dataset)
    hjds = np.concatenate((hjds1,hjds2))

    xmatch = crossmatch.CrossMatchTable()
    xmatch.datasets = Table([
                        Column(name='dataset_code', data=datasets),
                        Column(name='dataset_red_dir', data=['/no/path/used']*len(datasets)),
                        Column(name='dataset_filter', data=['ip']*len(datasets)),
                        Column(name='primary_ref', data=[1]*len(datasets)),
                        Column(name='norm_a0', data=[1.0]*len(datasets)),
                        Column(name='norm_a1', data=[0.0]*len(datasets)),
                        Column(name='norm_covar_0', data=[0.0]*len(datasets)),
                        Column(name='norm_covar_1', data=[0.0]*len(datasets)),
                        Column(name='norm_covar_2', data=[0.0]*len(datasets)),
                        Column(name='norm_covar_3', data=[0.0]*len(datasets)),
                        ])
    xmatch.images = Table([
                        Column(name='dataset_code', data=datacodes),
                        Column(name='hjd', data=hjds),
                        ])
    xmatch.field_index = Table([
                        Column(name='field_id', data=np.arange(1,nstars+1,1).astype('int')),
                        Column(name='quadrant', data=np.ones(nstars).astype('int')),
                        Column(name='quadrant_id', data=np.arange(1,nstars+1,1).astype('int')),
                        Column(name=datasets[0]+'_index', data=np.arange(1,nstars+1,1)),
                        Column(name=datasets[1]+'_index', data=np.arange(1,nstars+1,1)),
                        ])

    # Simulate only normalized photometry data
    quad_phot = np.zeros((nstars, nimages, 17))
    baseline_mag = 17.0
    baseline_mag_error = 1e-3
    quad_phot[:,:,7].fill(baseline_mag)
    quad_phot[:,:,8].fill(baseline_mag_error)
    for j in range(0,nstars,1):
        quad_phot[j,:,0] = xmatch.images['hjd']

    return xmatch, quad_phot

def test_bin_lc_in_time():

    ndays = 2
    interval = 0.25  # days
    hjd_min = 2455555.0
    hjds = np.arange(hjd_min, hjd_min+(float(ndays)), interval)
    lc = np.zeros((len(hjds),4))
    lc[:,0] = hjds
    lc[:,1] = np.random.randn(len(hjds)) + 17.0
    lc[:,2] = np.random.randn(len(hjds)) * 1e-3

    survey_time_bins = np.arange(hjd_min, hjd_min+(float(ndays)), 1.0)

    binned_lc = normalize_photometry_stars.bin_lc_in_time(lc,survey_time_bins)

    for b in range(0,len(binned_lc),1):
        print(b, binned_lc[b,0])

    fig = plt.figure(1,(10,10))
    plt.rcParams.update({'font.size': 18})
    plt.plot(lc[:,0]-2450000.0, lc[:,1], 'k.')
    plt.plot(binned_lc[:,0]-2450000.0, binned_lc[:,1], 'gd')
    plt.xlabel('HJD-2450000.0')
    plt.ylabel('Mag')
    plt.savefig('test_lightcurve_binning.png')

def test_bin_photometry_datasets():

    nstars = 10
    nimages_dataset = 20
    baseline_mag = 17.0
    (xmatch, quad_phot) = simulate_field_dataproducts(nstars, nimages_dataset)

    hjd_min = quad_phot[:,:,0].min()
    hjd_max = quad_phot[:,:,0].max()
    ndays = int(hjd_max - hjd_min)
    survey_time_bins = np.arange(hjd_min, hjd_min+(float(ndays)), 1.0)

    (survey_time_index, binned_phot) = normalize_photometry_stars.bin_photometry_datasets(xmatch, quad_phot, survey_time_bins)

    for dset, binned_data in binned_phot.items():
        assert(binned_data.shape == (nstars,len(survey_time_bins),3))
        for i in range(0,len(survey_time_bins),1):
            assert((binned_data[:,i,0]==survey_time_bins[i]).all())
            if i in survey_time_index:
                assert((binned_data[:,i,1]==baseline_mag).all())

def test_measure_dataset_offset():

    log = logs.start_stage_log( '.', 'test_norm_star' )

    # Build test data arrays:
    ndays = 10
    interval = 0.25  # days
    hjd_min = 2455555.0
    hjds = np.arange(hjd_min, hjd_min+(float(ndays)), interval)

    lc1 = np.zeros((len(hjds),3))
    lc1[:,0] = hjds
    lc1[:,1] = np.random.randn(len(hjds)) + 17.0
    lc1[:,2] = np.random.randn(len(hjds)) * 1e-3

    test_offset = 0.5
    lc2 = np.zeros((len(hjds),3))
    lc2[:,0] = hjds
    lc2[:,1] = lc1[:,1] + test_offset
    lc2[:,2] = np.random.randn(len(hjds)) * 1e-3

    (offset,offset_error) = normalize_photometry_stars.measure_dataset_offset(lc1, lc2, log)

    fig = plt.figure(1,(10,10))
    plt.rcParams.update({'font.size': 18})
    plt.errorbar(lc1[:,0]-2450000.0, lc1[:,1], yerr=lc1[:,2],
            mfc='purple', mec='purple', fmt='o')
    plt.errorbar(lc2[:,0]-2450000.0, lc2[:,1], yerr=lc2[:,2],
            mfc='green', mec='green', fmt='d')
    plt.xlabel('HJD-2450000.0')
    plt.ylabel('Mag')
    plt.savefig('test_measure_offset.png')

    assert(abs(test_offset-(-1.0*offset))<1e-3)

    logs.close_log(log)

def test_apply_dataset_offset():

    log = logs.start_stage_log( '.', 'test_norm_star' )

    # Build test data arrays:
    ndays = 10
    interval = 0.25  # days
    hjd_min = 2455555.0
    hjds = np.arange(hjd_min, hjd_min+(float(ndays)), interval)

    lc1 = np.zeros((len(hjds),3))
    lc1[:,0] = hjds
    lc1[:,1] = np.random.randn(len(hjds)) + 17.0
    lc1[:,2] = np.random.randn(len(hjds)) * 1e-3

    offset = 0.5
    offset_error = 1e-4
    lc2 = np.zeros((len(hjds),3))
    lc2[:,0] = hjds
    lc2[:,1] = lc1[:,1] + offset
    lc2[:,2] = np.random.randn(len(hjds)) * 1e-3

    lc3 = copy.deepcopy(lc2)

    lc3 = normalize_photometry_stars.apply_dataset_offset(lc3, -offset, offset_error, log)

    fig = plt.figure(1,(10,10))
    plt.rcParams.update({'font.size': 18})
    plt.errorbar(lc1[:,0]-2450000.0, lc1[:,1], yerr=lc1[:,2],
            mfc='purple', mec='purple', fmt='o')
    plt.errorbar(lc2[:,0]-2450000.0, lc2[:,1], yerr=lc2[:,2],
            mfc='green', mec='green', fmt='d')
    plt.errorbar(lc3[:,0]-2450000.0, lc3[:,1], yerr=lc3[:,2],
            mfc='black', mec='black', fmt='x')
    plt.xlabel('HJD-2450000.0')
    plt.ylabel('Mag')
    plt.savefig('test_apply_offset.png')

    assert(lc1[:,1].mean() == lc3[:,1].mean())

    logs.close_log(log)

def test_update_norm_field_photometry_for_star_idx():

    # Simulated test star:
    field_idx = 2
    nstars = 10
    nimages_dataset = 5
    (xmatch, quad_phot) = simulate_field_dataproducts(nstars, nimages_dataset)

    # Empty the photometry array and enter data for a single star, for later
    # ease of testing
    quad_phot = np.zeros((nstars, nimages, 17))
    baseline_mag = 17.0
    baseline_mag_error = 1e-3
    quad_phot[field_idx,:,7].fill(baseline_mag)
    quad_phot[field_idx,:,8].fill(baseline_mag_error)

    # Simulate revised photometry for one dataset
    offset = 0.5
    lc = {}
    data = np.zeros((nimages_dataset,4))
    data[:,0].fill(2455555.5)
    data[:,1].fill(baseline_mag)
    data[:,2].fill(baseline_mag_error)
    lc[datasets[0]] = data
    data2 = np.zeros((nimages_dataset,4))
    data2[:,0].fill(2455555.5)
    data2[:,1].fill(baseline_mag+offset)
    data2[:,2].fill((np.sqrt( 2.0*(baseline_mag_error*baseline_mag_error) )))
    lc[datasets[1]] = data2

    quad_phot = normalize_photometry_stars.update_norm_field_photometry_for_star_idx(
                                                    field_idx, xmatch,
                                                    quad_phot, lc)

    quad_idx = xmatch.field_index['quadrant_id'][field_idx]

    # Check that the photometry for the first dataset is unchanged:
    idx = np.where(xmatch.images['dataset_code'] == datasets[0])[0]
    assert(quad_phot[field_idx, idx, 7] == baseline_mag).all()
    assert(quad_phot[field_idx, idx, 8] == baseline_mag_error).all()

    # Check that the photometry for the second dataset has changed:
    idx = np.where(xmatch.images['dataset_code'] == datasets[1])[0]
    assert(quad_phot[field_idx, idx, 7] == data2[:,1]).all()
    assert(quad_phot[field_idx, idx, 8] == data2[:,2]).all()

    # Check no other changes have been made:
    for col in range(0,7,1):
        assert((quad_phot[:,:,col]==0.0).all())
    for col in range(9,17,1):
        assert((quad_phot[:,:,col]==0.0).all())

def test_update_mag_offsets_table():

    # Simulate a mag_offsets table:
    params = {'primary_ref': 'ROME-FIELD-01_lsc-doma-1m0-05-fa15_ip',
              'datasets': { 'ROME-FIELD-01_lsc-doma-1m0-05-fa15_ip': ['primary_ref', '/Users/rstreet1/OMEGA/test_data/non_ref_dataset_p/', 'ip'],
                            'ROME-FIELD-01_coj-doma-1m0-11-fa12_ip' : [ 'non_ref', '/Users/rstreet1/OMEGA/test_data/non_ref_dataset0/', 'ip' ]},
              'file_path': 'crossmatch_table.fits',
              'log_dir': '.',
              'gaia_dr': 'Gaia_DR2',
              'separation_threshold': (2.0/3600.0)*u.deg}
    xmatch = crossmatch.CrossMatchTable()
    xmatch.create(params)
    xmatch.field_index.add_row([1,267.61861696019145, -29.829605383706895, 4, 1, '4056436121079692032', 1, 0])
    xmatch.field_index.add_row([2,267.70228408545813, -29.83032824102953, 4, 2, '4056436121079692033', 2, 0])
    xmatch.field_index.add_row([3,267.9873108673885, -29.829734325692858, 3, 1, '4056436121079692034', 3, 0])
    xmatch.field_index.add_row([4,267.9585073984874, -29.83002538112054, 3, 2, '4056436121079692035', 4, 0])

    xmatch.create_normalizations_table()

    offset = 0.5
    offset_error = 1e-3
    field_idx = 1
    dset = 'ROME-FIELD-01_lsc-doma-1m0-05-fa15_ip'
    xmatch = normalize_photometry_stars.update_mag_offsets_table(xmatch,
                                            field_idx, dset,
                                            offset, offset_error)
    cname1 = 'delta_mag_'+xmatch.get_dataset_shortcode(dset)
    cname2 = 'delta_mag_error_'+xmatch.get_dataset_shortcode(dset)

    assert(xmatch.normalizations[cname1][field_idx] == offset)
    assert(xmatch.normalizations[cname2][field_idx] == offset_error)

if __name__ == '__main__':
    #test_bin_lc_in_time()
    #test_measure_dataset_offset()
    #test_apply_dataset_offset()
    #test_update_norm_field_photometry_for_star_idx()
    #test_update_mag_offsets_table()
    test_bin_photometry_datasets()
