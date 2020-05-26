from sys import argv
import numpy as np
from scipy import optimize
import astropy.units as u
from astropy.table import Table, Column
from pyDANDIA import metadata
from pyDANDIA import gaia_phot_transforms

def load_test_star_catalog(red_dir):

    reduction_metadata = metadata.MetaData()
    reduction_metadata.load_a_layer_from_file( red_dir,
                                              'pyDANDIA_metadata.fits',
                                              'data_architecture' )
    reduction_metadata.load_a_layer_from_file( red_dir,
                                               'pyDANDIA_metadata.fits',
                                              'reduction_parameters' )
    reduction_metadata.load_a_layer_from_file( red_dir,
                                              'pyDANDIA_metadata.fits',
                                              'headers_summary' )
    reduction_metadata.load_a_layer_from_file( red_dir,
                                              'pyDANDIA_metadata.fits',
                                              'star_catalog' )

    star_catalog = Table()
    star_catalog['index'] = reduction_metadata.star_catalog[1]['index']
    star_catalog['RA'] = reduction_metadata.star_catalog[1]['ra']
    star_catalog['DEC'] = reduction_metadata.star_catalog[1]['dec']
    star_catalog['mag'] = reduction_metadata.star_catalog[1]['ref_mag']
    star_catalog['mag_err'] = reduction_metadata.star_catalog[1]['ref_mag_error']
    star_catalog['gaia_ra'] = reduction_metadata.star_catalog[1]['ra']
    star_catalog['gaia_dec'] = reduction_metadata.star_catalog[1]['dec']
    star_catalog['vphas_source_id'] = reduction_metadata.star_catalog[1]['vphas_source_id']
    star_catalog['vphas_ra'] = reduction_metadata.star_catalog[1]['vphas_ra']
    star_catalog['vphas_dec'] = reduction_metadata.star_catalog[1]['vphas_dec']
    star_catalog['gmag'] = reduction_metadata.star_catalog[1]['gmag']
    star_catalog['e_gmag'] = reduction_metadata.star_catalog[1]['gmag_error']
    star_catalog['rmag'] = reduction_metadata.star_catalog[1]['rmag']
    star_catalog['e_rmag'] = reduction_metadata.star_catalog[1]['rmag_error']
    star_catalog['imag'] = reduction_metadata.star_catalog[1]['imag']
    star_catalog['e_imag'] = reduction_metadata.star_catalog[1]['imag_error']
    star_catalog['clean'] = np.zeros(len(reduction_metadata.star_catalog[1]['cal_ref_mag']))
    star_catalog['cal_ref_mag'] = np.zeros(len(reduction_metadata.star_catalog[1]['cal_ref_mag']))
    star_catalog['cal_ref_mag_err'] = np.zeros(len(reduction_metadata.star_catalog[1]['cal_ref_mag_error']))
    star_catalog['cal_ref_flux'] = np.zeros(len(reduction_metadata.star_catalog[1]['cal_ref_flux']))
    star_catalog['cal_ref_flux_err'] = np.zeros(len(reduction_metadata.star_catalog[1]['cal_ref_flux_error']))

    (Gmag, Gmerr) = gaia_phot_transforms.gaia_flux_to_mag(reduction_metadata.star_catalog[1]['phot_g_mean_flux'],
                                                          reduction_metadata.star_catalog[1]['phot_g_mean_flux_error'],
                                                          passband="G")
    (BPmag, BPmerr) = gaia_phot_transforms.gaia_flux_to_mag(reduction_metadata.star_catalog[1]['phot_g_mean_flux'],
                                                        reduction_metadata.star_catalog[1]['phot_g_mean_flux_error'],
                                                        passband="G_BP")
    (RPmag, RPmerr) = gaia_phot_transforms.gaia_flux_to_mag(reduction_metadata.star_catalog[1]['phot_g_mean_flux'],
                                                        reduction_metadata.star_catalog[1]['phot_g_mean_flux_error'],
                                                        passband="G_RP")

    star_catalog['gaia_source_id'] = reduction_metadata.star_catalog[1]['gaia_source_id']
    star_catalog['gaia_Gmag'] = Gmag
    star_catalog['gaia_Gmag_err'] = Gmerr
    star_catalog['gaia_BPmag'] = BPmag
    star_catalog['gaia_BPmag_err'] = BPmerr
    star_catalog['gaia_RPmag'] = RPmag
    star_catalog['gaia_RPmag_err'] = RPmerr

    print('Extracted star catalog')

    return star_catalog

def test_calc_gaia_colours(red_dir):

    star_catalog = load_test_star_catalog(red_dir)

    (BP_RP, BPRPerr) = gaia_phot_transforms.calc_gaia_colours(star_catalog['gaia_BPmag'],star_catalog['gaia_BPmag_err'],
                                                   star_catalog['gaia_RPmag'],star_catalog['gaia_RPmag_err'])

    assert type(BP_RP) == type(Column())
    assert type(BP_RP) == type(Column())
    idx = np.where(BP_RP > 0.0)
    assert BP_RP[idx].mean() > 0.0 and BP_RP[idx].mean() < 1.0
    assert BPRPerr[idx].mean() > 0.0 and BPRPerr[idx].mean() < 0.075

def test_transform_gaia_phot_to_SDSS(red_dir):

    star_catalog = load_test_star_catalog(red_dir)

    (BP_RP, BPRPerr) = gaia_phot_transforms.calc_gaia_colours(star_catalog['gaia_BPmag'],star_catalog['gaia_BPmag_err'],
                                               star_catalog['gaia_RPmag'],star_catalog['gaia_RPmag_err'])

    phot = gaia_phot_transforms.transform_gaia_phot_to_SDSS(star_catalog['gaia_Gmag'], star_catalog['gaia_Gmag_err'],
                                        BP_RP, BPRPerr)

    assert type(phot) == type({})
    for key in ['g', 'g_err', 'r', 'r_err', 'i', 'i_err']:
        assert key in phot.keys()
        assert type(phot[key]) == type(np.zeros(1)) or type(phot[key]) == type(Column())

if __name__ == '__main__':

    if len(argv) > 1:
        red_dir = argv[1]
    else:
        red_dir = input('Please enter path to reduction directory for testing: ')

    test_calc_gaia_colours(red_dir)
    test_transform_gaia_phot_to_SDSS(red_dir)
