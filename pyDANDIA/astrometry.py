from os import path
from sys import argv
from pyDANDIA import metadata
from pyDANDIA import pipeline_setup
from pyDANDIA import logs
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from scipy import optimize
from scipy.stats import binned_statistic_2d
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

def refine_astrometry(setup, reduction_metadata):
    log = logs.start_stage_log( setup.red_dir, 'astrometry' )

    (rome_positions, catalog_positions) = extract_star_positions(reduction_metadata, log)

    initial_residuals = calc_astrometric_residuals_spherical(rome_positions, catalog_positions)
    plot_angular_separations(setup, initial_residuals, catalog_positions, 'initial_astrometric_residuals.png')

    init_params = guess_starting_parameters(initial_residuals, log)

    fit_params = model_astrometric_residuals(rome_positions, catalog_positions, init_params, log)

    refined_rome_positions = apply_coordinate_transform(rome_positions, fit_params)

    residuals = calc_astrometric_residuals_spherical(refined_rome_positions, catalog_positions)
    plot_angular_separations(setup, residuals, catalog_positions, 'final_astrometric_residuals.png')

    for j in range(0,10,1):
        print(rome_positions[j], refined_rome_positions[j], catalog_positions[j],
        initial_residuals[j], residuals[j])

    log.info('Astrometry refinement: complete')
    logs.close_log(log)

def plot_angular_separations(setup, separations, positions, filename):
    """Based on code from Markus Hundertmark"""

    nxbins = 50
    nybins = 50
    #print('Plotting stars between RA='+str(positions.ra.value.min())+' - '+str(positions.ra.value.max())+\
    #        ' and Dec='+str(positions.dec.value.min())+' - '+str(positions.dec.value.max()))

    binned_stat = binned_statistic_2d(positions.ra.value, positions.dec.value,
                                      separations,
                                      statistic='median',
                                      bins = [nxbins,nybins],
                                      range=[[positions.ra.value.min(), positions.ra.value.max()],
                                            [positions.dec.value.min(), positions.dec.value.max()]])
    (fig, ax1) = plt.subplots(1,1)
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.2, top=0.9)
    im = ax1.imshow(binned_stat.statistic.T,
                    cmap = 'gist_rainbow', origin='bottom',
                    extent=(0, nxbins, 0, nybins),
                    vmin = 0.0, vmax = separations.max())

    nticks = 5
    bin_incr = float(nxbins)/float(nticks)
    xincr = (positions.ra.value.max() - positions.ra.value.min())/nticks
    yincr = (positions.dec.value.max() - positions.dec.value.min())/nticks
    xticks = []
    xlabels = []
    yticks = []
    ylabels = []
    for i in range(0,nticks,1):
        xticks.append(i*bin_incr)
        xlabels.append(str(np.round((positions.ra.value.min()+i*xincr),3)))
        yticks.append(i*bin_incr)
        ylabels.append(str(np.round((positions.dec.value.min()+i*yincr),3)))

    plt.xticks(xticks, xlabels, rotation=45.0)
    plt.yticks(yticks, ylabels, rotation=45.0)

    cb = fig.colorbar(im, ax = ax1, label = 'Angular residual [arcsec]')
    plt.xlabel('RA [deg]')
    plt.ylabel('Dec [deg]')
    plt.savefig(path.join(setup.red_dir,filename))

def guess_starting_parameters(residuals, log):

    params = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,\
              0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]

    log.info('Starting guess parameters: '+repr(params))

    return params

def model_astrometric_residuals(rome_positions, catalog_positions, pguess, log):

    (popt, pcov) = optimize.leastsq(calc_astrometric_residuals, pguess,
                            args=(rome_positions, catalog_positions))
    log.info('Least-squares fit parameters: '+repr(popt))

    return popt

def apply_coordinate_transform(positions, params):

    # NOTE: this may not wrap around 360deg properly
    revised_ra = params[0] + params[1]*positions.ra.value + params[2]*positions.dec.value \
                            + params[3]*positions.ra.value*positions.ra.value + \
                            + params[4]*positions.dec.value*positions.dec.value + \
                            + params[5]*positions.ra.value**3 + \
                            + params[6]*positions.dec.value**3
    revised_dec = params[7] + params[8]*positions.ra.value + params[9]*positions.dec.value \
                            + params[10]*positions.ra.value*positions.ra.value + \
                            + params[11]*positions.dec.value*positions.dec.value + \
                            + params[12]*positions.ra.value**3 + \
                            + params[13]*positions.dec.value**3

    return SkyCoord(revised_ra, revised_dec, frame='icrs', unit=(u.deg,u.deg))

def calc_astrometric_residuals(params, rome_positions, catalog_positions):

    revised_positions = apply_coordinate_transform(rome_positions, params)

    return calc_astrometric_residuals_spherical(revised_positions, catalog_positions)

def extract_star_positions(reduction_metadata, log):
    mask = np.where(reduction_metadata.star_catalog[1]['gaia_ra'].data > 0.0)[0]

    catalog_positions = SkyCoord(reduction_metadata.star_catalog[1]['gaia_ra'][mask], reduction_metadata.star_catalog[1]['gaia_dec'][mask],
                                frame='icrs', unit=(u.deg,u.deg))
    rome_positions = SkyCoord(reduction_metadata.star_catalog[1]['ra'][mask], reduction_metadata.star_catalog[1]['dec'][mask],
                                frame='icrs', unit=(u.deg,u.deg))

    log.info('Extracted '+str(len(mask))+' stars with valid Gaia matches from the metadata')

    return rome_positions, catalog_positions

def calc_astrometric_residuals_spherical(positions1, positions2,
                                        small_angles=True):

    if not small_angles:
        a1 = (90.0 - positions1.dec.value) * (np.pi/180.0)
        a2 = (90.0 - positions2.dec.value) * (np.pi/180.0)
        a3 = (positions1.ra.value - positions2.ra.value) * (np.pi/180.0)
        cos_gamma = np.cos(a1)*np.cos(a2) + np.sin(a1)*np.sin(a2)*np.cos(a3)

        idx = np.where(cos_gamma-1.0 < 1e-15)[0]
        cos_gamma[idx] = 1.0

        separations = np.arccos(cos_gamma) * (180.0/np.pi)

    else:
        delta_ra = positions1.ra.value - positions2.ra.value
        delta_dec = positions1.dec.value - positions2.dec.value
        separations = np.sqrt( (delta_ra*np.cos(positions1.dec.value))**2 +
                                delta_dec*delta_dec )

    # Ensure units in arcsec
    separations = separations * 3600

    return separations

def calc_astrometric_residuals_cartesian(catalog_positions, rome_positions):

    separations = np.sqrt( (rome_positions.ra.value - catalog_positions.ra.value)**2 +
                        (rome_positions.dec.value - catalog_positions.dec.value)**2 )

    # Ensure units in arcsec
    separations = separations * 3600

    return separations

def get_args():
    if len(argv) == 1:
        red_dir = input('Please enter the path to the reduction directory: ')
    else:
        red_dir = argv[1]

    setup = pipeline_setup.PipelineSetup()
    setup.red_dir = red_dir

    reduction_metadata = metadata.MetaData()
    reduction_metadata.load_all_metadata(setup.red_dir, 'pyDANDIA_metadata.fits')

    return setup, reduction_metadata

if __name__ == '__main__':
    (setup, reduction_metadata) = get_args()
    refine_astrometry(setup, reduction_metadata)
