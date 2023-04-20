from sys import argv
from os import getcwd, path
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units
from astropy.table import Table, Column
from pyDANDIA import hd5_utils
from pyDANDIA import crossmatch
from pyDANDIA import logs
from pyDANDIA import pipeline_setup
from pyDANDIA import plotly_lightcurves
from pyDANDIA import field_photometry
from pyDANDIA import field_lightcurves
import csv
import argparse
import json

def plot_field_lightcurves_enmasse():

    # Get commandline arguments
    args = get_args()

    # Logging:
    log = logs.start_stage_log( args.output_dir, 'field_lc_bulk')

    # Load the JSON file containing the field IDs and data on the targets.
    # This file can be produced using the search_crossmatch_bulk.py code.
    targets = load_target_list(args.target_file)

    # Load the crossmatch table:
    xmatch = crossmatch.CrossMatchTable()
    xmatch.load(path.join(args.input_dir, args.xmatch_file))

    # Preview the list of targets and group the targets by quadrant.
    # This is done so that the timeseries photometry for that quadrant can be
    # read in once, rather than re-read those data for each star separately.
    # The search results can contain multiple hits for any given star.
    # In these cases, the object with the smallest separation from the expected
    # coordinates is plotted.
    quadrant_targets = {1:[], 2:[], 3:[], 4:[]}
    for targetname, target_data in targets.items():
        #selected_target = find_closest_match(target_data)
        for star in target_data['rome_stars']:
            quadrant_targets[star['quadrant']].append(star)
    log.info('Sorted '+str(len(targets))+' into data quadrants to extract lightcurves')


    # Extract the lightcurves for targets in each quadrant.
    # By default, extract the normalised photometry.
    params = {'phot_type': 'normalized'}
    filters = ['gp', 'rp', 'ip']
    for qid in range(1,5,1):
        phot_file = path.join(args.input_dir,
                              args.field_name+'_quad'+str(qid)+'_photometry.hdf5')
        if not path.isfile(phot_file):
            log.info('Cannot extract lightcurves for quadrant '+str(qid)+' as photometry file not found')
        else:
            log.info('Loading timeseries photometry for quadrant '+str(qid))
            phot_file = path.join(args.input_dir,
                                  args.field_name+'_quad'+str(qid)+'_photometry.hdf5')
            quad_phot = hd5_utils.read_phot_from_hd5_file(phot_file,
        												  return_type='array')

            log.info('Extracting star lightcurves for field IDs:')
            for star in quadrant_targets[qid]:
                log.info(' -> '+str(star['field_id']))
                field_idx = star['field_id'] - 1
                lc = field_lightcurves.fetch_field_photometry_for_star_idx(params,
                                                field_idx, xmatch, quad_phot, log)
                title = 'Lightcurves of star field ID='+str(star['field_id'])
                plot_file = path.join(args.output_dir,
            				'star_'+str(star['field_id'])+'_lightcurve_'+params['phot_type']+'.html')
                plotly_lightcurves.plot_interactive_lightcurve(lc, filters, plot_file,
            													title=title)

    logs.close_log(log)

def find_closest_match(target_data):

    min_sep = 1e5
    selected_target = None
    for star in target_data['rome_stars']:
        if star['separation_deg'] < min_sep:
            selected_target = star
            min_sep = star['separation_deg']

    return selected_target

def load_target_list(input_file):

    if not path.isfile(input_file):
        raise IOError('Cannot find input list of targets at '
                        + input_file)

    with open(input_file, "r") as read_file:
        data = json.load(read_file)

    return data

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("target_file", help="Path to JSON file containing the list of targets to plot", type=str)
    parser.add_argument("input_dir", help="Path to the input data directory containing the photometry", type=str)
    parser.add_argument("xmatch_file", help="Name of the crossmatch table", type=str)
    parser.add_argument("field_name", help="Field name prefix to photometry HDF5 files", type=str)
    parser.add_argument("output_dir", help="Path to the directory for output lightcurves", type=str)
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    plot_field_lightcurves_enmasse()
