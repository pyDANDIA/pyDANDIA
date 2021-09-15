from sys import argv
import sqlite3
from os import getcwd, path, remove, environ
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units, table
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from pyDANDIA import  phot_db
from pyDANDIA import  hd5_utils
from pyDANDIA import  pipeline_setup
from pyDANDIA import  metadata
from pyDANDIA import  logs
from pyDANDIA import lightcurves
import csv

def extract_star_lightcurve_star_id(params, log=None, format='dat',
									valid_data_only=True,phot_error_threshold=10.0,
									output_neighbours=False,psfactor_threshold=0.8):
	"""Function to extract a lightcurve for a single star based on its star_id
	in the star_catolog in the metadata for a single reduction."""

	log = logs.start_stage_log( params['red_dir'], 'lightcurves' )
	reduction_metadata = metadata.MetaData()
	reduction_metadata.load_all_metadata(params['red_dir'], 'pyDANDIA_metadata.fits')
	if 'filter_name' not in params.keys():
		params['filter_name'] = reduction_metadata.headers_summary[1]['FILTKEY'][0]

	lc_files = []

    if log != None:
        log.info('Searching for star ID='+str(params['star_id']))
        log.info('Configured threshold for valid photometric datapoints is phot uncertainty <= '+\
                    str(phot_error_threshold)+' mag')

    if len(reduction_metadata.star_catalog[1]) >= params['star_id']:

        photometry_data = lightcurves.fetch_photometry_for_isolated_dataset(params, params['star_id'], log)

        lc_files = lightcurves.output_lightcurve(params, reduction_metadata, photometry_data, params['star_id'], format,
                                        valid_data_only, phot_error_threshold, psfactor_threshold, log)

    message = 'OK'
    logs.close_log(log)

    return message, lc_files

if __name__ == '__main__':
    params = {}

    if len(argv) == 1:
        params['red_dir'] = input('Please enter the path to a dataset reduction directory: ')
        params['star_id'] = input('Please enter the star ID from the metadata.star_catalog: ')
        params['output_dir'] = input('Please enter the path to the output directory: ')

    else:
        params['red_dir'] = argv[1]
        params['star_id'] = argv[2]
        params['output_dir'] = argv[3]

    params['star_id'] = int(params['star_id'])

    (message, lc_files) = extract_star_lightcurve_star_id(params, log=None, format='datcsv',
												valid_data_only=False)

    print(message)
    for f in lc_files:
        print(f)
