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
import csv

def plot_field_star_lightcurves(params, log=None):
	"""Function to extract a lightcurve for a single star based on its RA, Dec
	using the star_catolog in the metadata for a single reduction."""

	log = logs.start_stage_log( params['output_dir'], 'field_lightcurves' )

	xmatch = crossmatch.CrossMatchTable()
	xmatch.load(params['crossmatch_file'], log=log)

	if log != None:
		log.info('Plotting lightcurve for star ID '+str(params['field_id']))

	# Offset field ID by one to get array index of data
	field_idx = params['field_id'] - 1
	plot_file = path.join(params['output_dir'],
				'star_'+str(params['field_id'])+'_lightcurve_'+params['phot_type']+'.html')

	lc = fetch_field_photometry_for_star_idx(params, field_idx, xmatch, log)
	filters = ['gp', 'rp', 'ip']
	title = 'Lightcurves of star field ID='+str(params['field_id'])

	plotly_lightcurves.plot_interactive_lightcurve(lc, filters, plot_file,
													title=title)

	message = 'OK'
	logs.close_log(log)

	return message

def fetch_field_photometry_for_star_idx(params, field_idx, xmatch, log):

	log.info('Extracting target timeseries photometry from '+params['phot_hdf_file'])

	quad_phot = hd5_utils.read_phot_from_hd5_file(params['phot_hdf_file'],
												  return_type='array')
	quad_idx = xmatch.field_index['quadrant_id'][field_idx] - 1
	log.info('Extracting the lightcurve of star with field index='
			+str(field_idx)+' and quadrant index='+str(quad_idx))

	lc = {}
	(mag_col, merr_col) = field_photometry.get_field_photometry_columns(params['phot_type'])
	qc_col = 16
	for dataset in xmatch.datasets:
		# Extract the photometry of this object for the images from this dataset,
		# if the field index indicates that the object was measured in this dataset
		if xmatch.field_index[dataset['dataset_code']+'_index'][field_idx] != 0:
			shortcode = xmatch.get_dataset_shortcode(dataset['dataset_code'])
			# Select those images from the HDF5 pertaining to this dataset,
			# then select valid measurements for this star
			idx1 = np.where(xmatch.images['dataset_code'] == dataset['dataset_code'])[0]
			idx2 = np.where(quad_phot[quad_idx,:,0] > 0.0)[0]
			idx3 = np.where(quad_phot[quad_idx,:,mag_col] > 0.0)[0]
			idx = set(idx1).intersection(set(idx2))
			idx = list(idx.intersection(set(idx3)))

			# Store the photometry
			if len(idx) > 0:
				photometry = np.zeros((len(idx),4))
				photometry[:,0] = quad_phot[quad_idx,idx,0]
				photometry[:,1] = quad_phot[quad_idx,idx,mag_col]
				photometry[:,2] = quad_phot[quad_idx,idx,merr_col]
				photometry[:,3] = quad_phot[quad_idx,idx,qc_col]
				lc[shortcode] = photometry

				log.info('-> Extracted '+str(len(idx))
					+' valid datapoints of timeseries photometry for star '
					+str(field_idx+1)+' from dataset '+dataset['dataset_code'])
			else:
				log.info('-> No valid datapoints in the lightcurve for star '
				+str(field_idx+1)+' from dataset '+dataset['dataset_code'])

		else:
			log.info('-> Star '+str(field_idx+1)+' was not measured in dataset '
					+dataset['dataset_code'])

	return lc

if __name__ == '__main__':
	params = {}

	if len(argv) == 1:
		params['crossmatch_file'] = input('Please enter the path to the field crossmatch file: ')
		params['phot_hdf_file'] = input('Please enter the path to the directory containing the field photometry HDF5 files: ')
		params['field_id'] = int(float(input('Please enter the field ID of the star in the field index: ')))
		params['phot_type'] = input('Please enter the columns of photometry to plot {instrumental,calibrated,corrected,normalized}: ')
		params['output_dir'] = input('Please enter the path to the output directory: ')

	else:
		params['crossmatch_file'] = argv[1]
		params['phot_hdf_file'] = argv[2]
		params['field_id'] = int(float(argv[3]))
		params['phot_type'] = argv[4]
		params['output_dir'] = argv[5]

	plot_field_star_lightcurves(params, log=None)
