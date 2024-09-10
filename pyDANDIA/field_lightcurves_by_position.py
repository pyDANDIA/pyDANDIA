import argparse
from pyDANDIA import hd5_utils
from pyDANDIA import crossmatch
from pyDANDIA import field_lightcurves
from pyDANDIA import plotly_lightcurves
from pyDANDIA import logs
from astropy.coordinates import SkyCoord
from astropy import units as u
from os import path

def fetch_lc_by_position(args):
    log = logs.start_stage_log(args.output_dir, 'field_lightcurves')

    # Load field crossmatch file
    xmatch = crossmatch.CrossMatchTable()
    xmatch.load(args.crossmatch_file)

    # Parse input coordinates into decimal degrees
    s = SkyCoord(args.RA, args.Dec, frame='icrs', unit=(u.hourangle, u.deg))

    # Find the nearest star in the field index
    params = {'ra_centre': s.ra.deg, 'dec_centre': s.dec.deg, 'radius': float(args.radius)/3600.0}
    results = xmatch.cone_search(params)
    if len(results) == 0:
        log.info('No catalog entry found within search radius at that position')
        logs.close_log(log)
        return

    # Identify the closest object in the field index
    field_idx = results['field_id'][0] - 1
    quad_idx = results['quadrant_id'][0] - 1
    qid = results['quadrant'][0]
    log.info('Nearest matching star is ' + str(field_idx+1) + ' in quadrant ' + str(qid)
          + ' at ' + str(results['ra'][0]) + ', ' + str(results['dec'][0]))

    # Extract the corresponding lightcurve data:
    phot_hdf_file = path.join(args.data_dir, args.field_name + '_quad'+str(qid)+'_photometry.hdf5')
    star_phot = hd5_utils.read_star_from_hd5_file(phot_hdf_file, quad_idx)

    params = {'phot_type': args.phot_type, 'output_dir': args.output_dir, 'field_id': field_idx+1}
    lc = field_lightcurves.fetch_field_photometry_for_star_idx(params, field_idx, xmatch,
                                             star_phot, log)
    if 'True' in args.combine_data:
        lc = field_lightcurves.combine_datasets_by_filter(lc, log)

    # Output to plot and datafiles:
    filters = ['gp', 'rp', 'ip']
    title = 'Lightcurves of star field ID=' + str(field_idx+1)
    plot_file = path.join(args.output_dir,
                          'star_' + str(field_idx+1) + '_lightcurve_' + args.phot_type + '.html')
    plotly_lightcurves.plot_interactive_lightcurve(lc, filters, plot_file,
                                                   title=title)

    field_lightcurves.output_datasets_to_file(params, lc, log)

    logs.close_log(log)

    return

def get_args():
    # NOTE: call with python field_lightcurves_by_position.py -- <arguments> for negative declinations
    parser = argparse.ArgumentParser()
    parser.add_argument('crossmatch_file', help='Path to crossmatch file')
    parser.add_argument('data_dir', help='Path to data directory for the field')
    parser.add_argument('field_name', help='Name of field')
    parser.add_argument('RA', help='Central RA to search for [sexigesimal]')
    parser.add_argument('Dec', help='Central Dec to search for [sexigesimal]')
    parser.add_argument('radius', help='Search radius in arcsec')
    parser.add_argument('output_dir', help='Path to output directory')
    parser.add_argument('combine_data', help='Combine data from all cameras? {True, False}')
    parser.add_argument('phot_type',
        help='Columns of photometry to plot {instrumental,calibrated,corrected,normalized}')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    fetch_lc_by_position(args)