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
import copy

def plot_field_star_lightcurves(params, log=None):
	"""Function to extract a lightcurve for a single star based on its RA, Dec
	using the star_catolog in the metadata for a single reduction."""

	log = logs.start_stage_log( params['output_dir'], 'field_lightcurves' )

	xmatch = crossmatch.CrossMatchTable()
	xmatch.load(params['crossmatch_file'], log=log)

	sanity_check(params,xmatch)

	if log != None:
		log.info('Plotting lightcurve for star ID '+str(params['field_id']))

	# Offset field ID by one to get array index of data
	field_idx = params['field_id'] - 1
	quad_idx = xmatch.field_index['quadrant_id'][field_idx] - 1
	if log:
		log.info('Extracting the lightcurve of star with field index='
			+str(field_idx)+' and quadrant index='+str(quad_idx))
	plot_file = path.join(params['output_dir'],
				'star_'+str(params['field_id'])+'_lightcurve_'+params['phot_type']+'.html')

	log.info('Extracting target timeseries photometry from '+params['phot_hdf_file'])

	#quad_phot = hd5_utils.read_phot_from_hd5_file(params['phot_hdf_file'],
	#											  return_type='array')
	star_phot = hd5_utils.read_star_from_hd5_file(params['phot_hdf_file'], quad_idx)

	lc = fetch_field_photometry_for_star_idx(params, field_idx, xmatch,
											 star_phot, log)
	if params['combine_data']:
		lc = combine_datasets_by_filter(lc, log)

	filters = ['gp', 'rp', 'ip']
	title = 'Lightcurves of star field ID='+str(params['field_id'])

	plotly_lightcurves.plot_interactive_lightcurve(lc, filters, plot_file,
													title=title)

	output_datasets_to_file(params, lc, log)

	message = 'OK'
	logs.close_log(log)

	return message

def sanity_check(params,xmatch):

	field_idx = params['field_id'] - 1
	star_quadrant = int(xmatch.field_index['quadrant'][field_idx])
	hdf_quadrant = int(path.basename(params['phot_hdf_file']).split('_')[1].replace('quad',''))

	if star_quadrant != hdf_quadrant:
		raise IOError('The star requested is in quadrant '+str(star_quadrant)
				+' but the photometry file provided is from quadrant '+str(hdf_quadrant))

def fetch_field_photometry_for_star_idx(params, field_idx, xmatch, star_phot, log):

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
			idx2 = np.where(star_phot[:,0] > 0.0)[0]
			idx3 = np.where(star_phot[:,mag_col] > 0.0)[0]
			idx = set(idx1).intersection(set(idx2))
			idx = list(idx.intersection(set(idx3)))

			# Store the photometry
			if len(idx) > 0:
				photometry = np.zeros((len(idx),4))
				photometry[:,0] = star_phot[idx,0]
				photometry[:,1] = star_phot[idx,mag_col]
				photometry[:,2] = star_phot[idx,merr_col]
				photometry[:,3] = star_phot[idx,qc_col]
				lc[shortcode] = photometry

				if log:
					log.info('-> Extracted '+str(len(idx))
					+' valid datapoints of timeseries photometry for star '
					+str(field_idx+1)+' from dataset '+dataset['dataset_code'])
			else:
				if log:
					log.info('-> No valid datapoints in the lightcurve for star '
				+str(field_idx+1)+' from dataset '+dataset['dataset_code'])

		else:
			if log:
				log.info('-> Star '+str(field_idx+1)+' was not measured in dataset '
					+dataset['dataset_code'])

	return lc

def combine_datasets_by_filter(lc, log):

	log.info('Combining datasets from multiple telescopes for each filter: ')
	# Group the datasets available for this star's lightcurve by filter
	filters = {}
	for dset in lc.keys():
		f = dset.split('_')[-1]
		if f not in filters.keys():
			filters[f] = [dset]
		else:
			filters[f].append(dset)
	log.info(repr(filters))

	# Combine the lightcurves per filter
	lc_per_filter = {}
	for f in filters.keys():
		for i,dset in enumerate(filters[f]):
			if i == 0:
				data = copy.deepcopy(lc[dset])
			else:
				data = np.concatenate((data, lc[dset]))

		# Sort into time order
		jdx = np.argsort(data[:,0])
		lc_per_filter['lco_'+f] = data[jdx]

	return lc_per_filter

def output_datasets_to_file(params,lc,log):

	for dset, data in lc.items():
		if len(data) > 0:
			file_path = path.join(params['output_dir'],
								'star_'+str(params['field_id'])+'_'+dset+'.dat')
			f = open(file_path, 'w')
			f.write('# Photometry type: '+params['phot_type']+'\n')
			f.write('# HJD    mag       mag_error     QC_code\n')
			for i in range(0,len(data),1):
				f.write(str(data[i,0])+' '+str(data[i,1])+' '
							+str(data[i,2])+' '+str(data[i,3])+'\n')
			f.close()
			log.info('Output data for '+dset+' to '+file_path)
		else:
			log.info('No valid data found for '+dset)

if __name__ == '__main__':
	params = {}

	if len(argv) == 1:
		params['crossmatch_file'] = input('Please enter the path to the field crossmatch file: ')
		params['phot_hdf_file'] = input('Please enter the path to the directory containing the field photometry HDF5 files: ')
		params['field_id'] = int(float(input('Please enter the field ID of the star in the field index: ')))
		params['phot_type'] = input('Please enter the columns of photometry to plot {instrumental,calibrated,corrected,normalized}: ')
		params['output_dir'] = input('Please enter the path to the output directory: ')
		params['combine_data'] = input('Combine dataset lightcurves by filter?  Y or N: ')

	else:
		params['crossmatch_file'] = argv[1]
		params['phot_hdf_file'] = argv[2]
		params['field_id'] = int(float(argv[3]))
		params['phot_type'] = argv[4]
		params['output_dir'] = argv[5]
		params['combine_data'] = argv[6]

	if 'Y' in str(params['combine_data']).upper():
		params['combine_data'] = True
	else:
		params['combine_data'] = False

	plot_field_star_lightcurves(params, log=None)
