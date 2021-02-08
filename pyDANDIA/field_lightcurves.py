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
import csv

def extract_field_star_lightcurves(params, log=None, format='dat'):
	"""Function to extract a lightcurve for a single star based on its RA, Dec
	using the star_catolog in the metadata for a single reduction."""

	log = logs.start_stage_log( params['output_dir'], 'field_lightcurves' )

	xmatch = crossmatch.CrossMatchTable()
	xmatch.load(params['crossmatch_file'], log=log)

	if log != None:
		log.info('Searching for star at RA,Dec='+str(params['ra'])+', '+str(params['dec']))

	c = SkyCoord(params['ra'], params['dec'], frame='icrs', unit=(units.hourangle, units.deg))

	if 'radius' in params.keys():
		radius = float(params['radius'])
	else:
		radius = 2.0

	results = xmatch.cone_search({'ra_centre': c.ra.deg,
								  'dec_centre': c.dec.deg,
								  'radius': radius}, log=log)

	if log != None and len(results) == 0:
		log.info('No matching objects found')

	if log != None and len(results) > 0:
		log.info('Extracting lightcurves for the following matching objects')

	for star in results:

		if log!=None:
			log.info('-> Star dataset ID: '+str(star['field_id'])+' separation: '+str(star['separation'])+' deg')

		photometry_data = fetch_field_photometry_for_star(params, star, xmatch, log)

		for dataset_code, phot_data in photometry_data.items():
			time_order = np.argsort(phot_data['hjd'])

			lc_file = path.join(params['output_dir'],'star_'+str(star['field_id'])+'_'+dataset_code+'.'+str(format))

			if format == 'dat':
				datafile = open(lc_file,'w')
				datafile.write('# HJD    Instrumental mag, mag_error   Calibrated mag, mag_error\n')

				for i in time_order:
					datafile.write(str(phot_data['hjd'][i])+'  '+\
							str(phot_data['instrumental_mag'][i])+'  '+str(phot_data['instrumental_mag_err'][i])+'  '+\
							str(phot_data['calibrated_mag'][i])+'  '+str(phot_data['calibrated_mag_err'][i])+'\n')

				datafile.close()

			elif format == 'csv':
				with open(lc_file, 'w', newline='') as csvfile:
					datafile = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
					datafile.writerow(['time', 'filter', 'magnitude', 'error'])
					for i in time_order:
						if photdata['instrumental_mag'][i] > 0.0:
							datafile.writerow([str(phot_data['hjd'][i]), params['filter_name'],
												str(phot_data['instrumental_mag'][i]),
												str(phot_data['instrumental_mag_err'][i])])

			else:
				log.info('Unrecognized lightcurve format requested ('+str(format)+') no output possible')

			if log!=None:
				log.info('-> Output photometry for dataset '+dataset_code+' to '+lc_file)

	message = 'OK'
	logs.close_log(log)

	return message

def fetch_field_photometry_for_star(params, star, xmatch, log):

	phot_file = params['field_name']+'_quad'+str(star['quadrant'])+'_photometry.hdf5'
	log.info('Extracting target timeseries photometry from '+phot_file)

	setup = pipeline_setup.PipelineSetup()
	setup.red_dir = params['phot_dir']

	quad_phot = hd5_utils.read_phot_hd5(setup,log=log,filename=phot_file)

	photometry = {}

	for dataset in xmatch.datasets:
		idx = np.where(xmatch.images['dataset_code'] == dataset['dataset_code'])[0]
		photometry[dataset['dataset_code']] = Table([ Column(name='hjd', data=quad_phot[star['quadrant_id']-1,idx,0], dtype='float'),
								Column(name='instrumental_mag', data=quad_phot[star['quadrant_id']-1,idx,1], dtype='float'),
								Column(name='instrumental_mag_err', data=quad_phot[star['quadrant_id']-1,idx,2], dtype='float'),
								Column(name='calibrated_mag', data=quad_phot[star['quadrant_id']-1,idx,3], dtype='float'),
								Column(name='calibrated_mag_err', data=quad_phot[star['quadrant_id']-1,idx,4], dtype='float'),
								Column(name='corrected_mag', data=quad_phot[star['quadrant_id']-1,idx,5], dtype='float'),
								Column(name='corrected_mag_err', data=quad_phot[star['quadrant_id']-1,idx,6], dtype='float'),
								])

		log.info('-> Extract timeseries photometry for star '+str(star['field_id'])+' from dataset '+dataset['dataset_code'])

	return photometry

if __name__ == '__main__':
	params = {}

	if len(argv) == 1:
		params['crossmatch_file'] = input('Please enter the path to the field crossmatch file: ')
		params['phot_dir'] = input('Please enter the path to the directory containing the field photometry HDF5 files: ')
		params['field_name'] = input('Please enter the field name used for the photometry files: ')
		params['ra'] = input('Please enter the RA [sexigesimal]: ')
		params['dec'] = input('Please enter the Dec [sexigesimal]: ')
		params['radius'] = input('Please enter the search radius in arcsec: ')
		params['output_dir'] = input('Please enter the path to the output directory: ')

	else:
		params['crossmatch_file'] = argv[1]
		params['phot_dir'] = argv[2]
		params['field_name'] = argv[3]
		params['ra'] = argv[4]
		params['dec'] = argv[5]
		params['radius'] = argv[6]
		params['output_dir'] = argv[7]

	# Ensure units are decimal degrees
	params['radius'] = float(params['radius'])/3600.0

	#message = extract_star_lightcurves_on_position(params)
    #print(message)

	extract_field_star_lightcurves(params, log=None, format='dat')
