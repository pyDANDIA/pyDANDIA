# -*- coding: utf-8 -*-
"""
Created on Thu May 30 14:30:19 2019

@author: rstreet
"""

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
import csv


def extract_star_lightcurves_on_cone_to_list(params):
	"""Function to extract a lightcurve from a phot_db"""

	log = logs.start_stage_log( params['red_dir'], 'lightcurves' )

	reduction_metadata = metadata.MetaData()
	reduction_metadata.load_a_layer_from_file( params['red_dir'],
		                                      'pyDANDIA_metadata.fits',
		                                      'matched_stars' )
	matched_stars = reduction_metadata.load_matched_stars()

	conn = phot_db.get_connection(dsn=params['db_file_path'])

	facilities = phot_db.fetch_facilities(conn)
	filters = phot_db.fetch_filters(conn)
	code_id = phot_db.get_stage_software_id(conn,'stage6')

	c = SkyCoord(params['ra'], params['dec'], frame='icrs', unit=(units.hourangle, units.deg))

	if 'radius' in params.keys():
		radius = float(params['radius'])
	else:
		radius = 2.0 / 3600.0

	results = phot_db.box_search_on_position(conn, c.ra.deg, c.dec.deg, radius, radius)
	lcs = []
	for star_field_id in results['star_id']:



		photometry_data = fetch_photometry_for_dataset(params, star_field_id, matched_stars, log)

		lcs.append(np.c_[photometry_data['hjd'],photometry_data['calibrated_mag'],photometry_data['calibrated_mag_err']])

	logs.close_log(log)

	return lcs




def extract_star_lightcurves_on_cone(params, log=None):
	"""Function to extract a lightcurve from a phot_db"""

	log = logs.start_stage_log( params['red_dir'], 'lightcurves' )

	reduction_metadata = metadata.MetaData()
	reduction_metadata.load_a_layer_from_file( params['red_dir'],
		                                      'pyDANDIA_metadata.fits',
		                                      'matched_stars' )
	matched_stars = reduction_metadata.load_matched_stars()

	conn = phot_db.get_connection(dsn=params['db_file_path'])

	facilities = phot_db.fetch_facilities(conn)
	filters = phot_db.fetch_filters(conn)
	code_id = phot_db.get_stage_software_id(conn,'stage6')

	if log != None:
		log.info('Searching for star at RA,Dec='+str(params['ra'])+', '+str(params['dec']))

	c = SkyCoord(params['ra'], params['dec'], frame='icrs', unit=(units.hourangle, units.deg))

	if 'radius' in params.keys():
		radius = float(params['radius'])
	else:
		radius = 2.0 / 3600.0

	results = phot_db.box_search_on_position(conn, c.ra.deg, c.dec.deg, radius, radius)

	if log != None and len(results['star_id']) > 0:
		log.info('Extracting lightcurves for the following matching objects')

	for star_field_id in results['star_id']:

		if log!=None:
			log.info('-> Star field ID: '+str(star_field_id))

		photometry_data = fetch_photometry_for_dataset(params, star_field_id, matched_stars, log)

		#setname = path.basename(params['red_dir']).split('_')[1]
		setname = path.basename("_".join((params['red_dir']).split('_')[1:]))

		datafile = open(path.join(params['output_dir'],'star_'+str(star_field_id)+'_'+setname+'.dat'),'w')

		for i in range(0,len(photometry_data),1):

		    datafile.write(str(photometry_data['hjd'][i])+'  '+\
				    str(photometry_data['instrumental_mag'][i])+'  '+str(photometry_data['instrumental_mag_err'][i])+'  '+\
				    str(photometry_data['calibrated_mag'][i])+'  '+str(photometry_data['calibrated_mag_err'][i])+'\n')

		datafile.close()
		if log!=None:
			log.info('-> Output dataset '+setname)

	message = 'OK'
	logs.close_log(log)

	return message

def extract_star_lightcurve_isolated_reduction(params, log=None, format='dat',
											valid_data_only=True,phot_error_threshold=10.0):
	"""Function to extract a lightcurve for a single star based on its RA, Dec
	using the star_catolog in the metadata for a single reduction."""

	log = logs.start_stage_log( params['red_dir'], 'lightcurves' )

	reduction_metadata = metadata.MetaData()
	reduction_metadata.load_all_metadata(params['red_dir'], 'pyDANDIA_metadata.fits')

	#filter_name = reduction_metadata.fetch_reduction_filter()

	if log != None:
		log.info('Searching for star at RA,Dec='+str(params['ra'])+', '+str(params['dec']))
		log.info('Configured threshold for valid photometric datapoints is phot uncertainty <= '+\
					str(phot_error_threshold)+' mag')

	c = SkyCoord(params['ra'], params['dec'], frame='icrs', unit=(units.hourangle, units.deg))

	if 'radius' in params.keys():
		radius = float(params['radius'])
	else:
		radius = 2.0

	results = reduction_metadata.cone_search_on_position({'ra_centre': c.ra.deg,
														 'dec_centre': c.dec.deg,
														 'radius': radius})
	if log != None and len(results['star_id']) == 0:
		log.info('No matching objects found')

	if log != None and len(results['star_id']) > 0:
		log.info('Extracting lightcurves for the following matching objects')

	for star_dataset_id in results['star_id']:

		if log!=None:
			log.info('-> Star dataset ID: '+str(star_dataset_id)+' separation: '+str(results['separation'])+' deg')

		photometry_data = fetch_photometry_for_isolated_dataset(params, star_dataset_id, log)

		time_order = np.argsort(photometry_data['hjd'])
		#setname = path.basename(params['red_dir']).split('_')[1]
		setname = path.basename("_".join((params['red_dir']).split('_')[1:]))

		lc_file = path.join(params['output_dir'],'star_'+str(star_dataset_id)+'_'+setname+'.'+str(format))

		if format == 'dat':
			datafile = open(lc_file,'w')
			datafile.write('# HJD    Instrumental mag, mag_error   Calibrated mag, mag_error\n')

			for i in time_order:
				if valid_data_only:
					if photometry_data['instrumental_mag'][i] > 0.0 and \
							photometry_data['instrumental_mag_err'][i] <= phot_error_threshold:
						datafile.write(str(photometry_data['hjd'][i])+'  '+\
						str(photometry_data['instrumental_mag'][i])+'  '+str(photometry_data['instrumental_mag_err'][i])+'  '+\
						str(photometry_data['calibrated_mag'][i])+'  '+str(photometry_data['calibrated_mag_err'][i])+'\n')
				else:
					datafile.write(str(photometry_data['hjd'][i])+'  '+\
					str(photometry_data['instrumental_mag'][i])+'  '+str(photometry_data['instrumental_mag_err'][i])+'  '+\
					str(photometry_data['calibrated_mag'][i])+'  '+str(photometry_data['calibrated_mag_err'][i])+'\n')

			datafile.close()

		elif format == 'csv':
			with open(lc_file, 'w', newline='') as csvfile:
				datafile = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
				datafile.writerow(['time', 'filter', 'magnitude', 'error'])
				for i in time_order:
					if photometry_data['instrumental_mag'][i] > 0.0:
						datafile.writerow([str(photometry_data['hjd'][i]), params['filter_name'],
											str(photometry_data['instrumental_mag'][i]),
											str(photometry_data['instrumental_mag_err'][i])])

		else:
			log.info('Unrecognized lightcurve format requested ('+str(format)+') no output possible')

		if log!=None:
			log.info('-> Output dataset '+setname)

	message = 'OK'
	logs.close_log(log)

	return message





def extract_star_lightcurves_on_position(params):
	"""Function to extract a lightcurve from a phot_db"""
	log = logs.start_stage_log( params['red_dir'], 'lightcurves' )

	reduction_metadata = metadata.MetaData()
	reduction_metadata.load_a_layer_from_file(params['red_dir'], 'pyDANDIA_metadata.fits', 'data_architecture')
	reduction_metadata.load_a_layer_from_file(params['red_dir'], 'pyDANDIA_metadata.fits', 'matched_stars')
	matched_stars = reduction_metadata.load_matched_stars()

	setname = path.basename(str(reduction_metadata.data_architecture[1]['OUTPUT_DIRECTORY'][0]))

	conn = phot_db.get_connection(dsn=params['db_file_path'])

	facilities = phot_db.fetch_facilities(conn)
	filters = phot_db.fetch_filters(conn)
	code_id = phot_db.get_stage_software_id(conn,'stage6')

	c = SkyCoord(params['ra'], params['dec'], frame='icrs', unit=(units.hourangle, units.deg))

	if 'radius' in params.keys():
		radius = float(params['radius'])
	else:
		radius = 2.0 / 3600.0

	results = phot_db.box_search_on_position(conn, c.ra.deg, c.dec.deg, radius, radius)

	if len(results) > 0:

		star_idx = np.where(results['separation'] == results['separation'].min())
		star_field_id = results['star_id'][star_idx][0]

		print('Identifed nearest star as '+str(results['star_id'][star_idx][0]))

        #query = 'SELECT filter, facility, hjd, calibrated_mag, calibrated_mag_err FROM phot WHERE star_id="'+str(results['star_id'][star_idx][0])+\
        #            '" AND software="'+str(code_id)+'"'

        #phot_table = phot_db.query_to_astropy_table(conn, query, args=())

        #datasets = identify_unique_datasets(phot_table,facilities,filters)

		photometry_data = fetch_photometry_for_dataset(params, star_field_id, matched_stars, log)

		datafile = open(path.join(params['output_dir'],'star_'+str(star_field_id)+'_'+setname+'.dat'),'w')

		for i in range(0,len(photometry_data),1):

			if photometry_data['hjd'][i] > 0.0:
				datafile.write(str(photometry_data['hjd'][i])+'  '+\
                            str(photometry_data['instrumental_mag'][i])+'  '+str(photometry_data['instrumental_mag_err'][i])+'  '+\
                            str(photometry_data['calibrated_mag'][i])+'  '+str(photometry_data['calibrated_mag_err'][i])+'\n')

		datafile.close()
		print('-> Output dataset '+setname)

		message = 'OK'

	else:
		message = 'No stars within search region'

	conn.close()

	logs.close_log(log)

	return message

def identify_unique_datasets(phot_table,facilities,filters):
    """Function to extract a list of the unique datasets from a table of
    photometry, i.e. the list of unique combinations of facility and filter
    """

    datasets = {}

    for j in range(0,len(phot_table),1):

        i = np.where(facilities['facility_id'] == phot_table['facility'][j])
        facility_code = facilities['facility_code'][i][0]

        i = np.where(filters['filter_id'] == phot_table['filter'][j])
        fcode = filters['filter_name'][i][0]

        dataset_code = str(facility_code)+'_'+str(fcode)

        if dataset_code in datasets.keys():

            setlist = datasets[dataset_code]

        else:

            setlist = [ phot_table['facility'][j],
                        phot_table['filter'][j],
                        [] ]

        setlist[2].append(j)

        datasets[dataset_code] =  setlist

    print('Found '+str(len(datasets))+' lightcurve datasets')

    return datasets

def fetch_photometry_for_dataset(params, star_field_id, matched_stars, log):

    setup = pipeline_setup.pipeline_setup({'red_dir': params['red_dir']})

    dataset_photometry = hd5_utils.read_phot_hd5(setup)

    (star_field_ids, star_dataset_ids) = matched_stars.find_starlist_match_ids('cat1_index', np.array([star_field_id]), log,
                                                                                verbose=True)
    star_dataset_id = star_dataset_ids[0]

    log.info('Star field ID = '+str(star_field_id))
    log.info('Star dataset ID = '+str(star_dataset_id))

    star_dataset_index = star_dataset_id - 1

    log.info('Star array index: '+str(star_dataset_index))

    photometry_data = dataset_photometry[star_dataset_index,:,:]

    photometry_data = table.Table( [ table.Column(name='hjd', data=dataset_photometry[star_dataset_index,:,9]),
                                     table.Column(name='instrumental_mag', data=dataset_photometry[star_dataset_index,:,11]),
                                     table.Column(name='instrumental_mag_err', data=dataset_photometry[star_dataset_index,:,12]),
                                      table.Column(name='calibrated_mag', data=dataset_photometry[star_dataset_index,:,13]),
                                      table.Column(name='calibrated_mag_err', data=dataset_photometry[star_dataset_index,:,14]),
                                      ] )

    return photometry_data

def fetch_photometry_for_isolated_dataset(params, star_dataset_id, log):

	setup = pipeline_setup.pipeline_setup({'red_dir': params['red_dir']})

	dataset_photometry = hd5_utils.read_phot_hd5(setup)

	log.info('Star dataset ID = '+str(star_dataset_id))

	star_dataset_index = star_dataset_id - 1

	log.info('Star array index: '+str(star_dataset_index))
	photometry_data = dataset_photometry[star_dataset_index,:,:]

	photometry_data = table.Table( [ table.Column(name='hjd', data=dataset_photometry[star_dataset_index,:,9]),
									 table.Column(name='instrumental_mag', data=dataset_photometry[star_dataset_index,:,11]),
									 table.Column(name='instrumental_mag_err', data=dataset_photometry[star_dataset_index,:,12]),
									  table.Column(name='calibrated_mag', data=dataset_photometry[star_dataset_index,:,13]),
									  table.Column(name='calibrated_mag_err', data=dataset_photometry[star_dataset_index,:,14]),
									  ] )

	return photometry_data

def read_pydandia_lightcurve(file_path, skip_zero_entries=True):
	"""Function to read the pyDANDIA lightcurve file format to an astropy Table"""

	if path.isfile(file_path) == False:
		raise IOError('Cannot find input lightcurve file '+file_path)

	data = np.loadtxt(file_path,skiprows=0)

	if skip_zero_entries:
		idx = np.where(data[:,0] != 0.0)[0]
	else:
		idx = np.arange(0,len(data),1)

	lc = table.Table( [ table.Column(name='hjd', data=data[idx,0]),
						table.Column(name='instrumental_mag', data=data[idx,1]),
						table.Column(name='instrumental_mag_err', data=data[idx,2]),
						 table.Column(name='calibrated_mag', data=data[idx,3]),
						 table.Column(name='calibrated_mag_err', data=data[idx,4]) ] )

	return lc

if __name__ == '__main__':
	params = {}

	if len(argv) == 1:
		#params['db_file_path'] = input('Please enter the path to the field photometric DB: ')
		params['red_dir'] = input('Please enter the path to a dataset reduction directory: ')
		params['ra'] = input('Please enter the RA [sexigesimal]: ')
		params['dec'] = input('Please enter the Dec [sexigesimal]: ')
		params['radius'] = input('Please enter the search radius in arcsec: ')
		params['output_dir'] = input('Please enter the path to the output directory: ')

	else:
		#params['db_file_path'] = argv[1]
		params['red_dir'] = argv[1]
		params['ra'] = argv[2]
		params['dec'] = argv[3]
		params['radius'] = argv[4]
		params['output_dir'] = argv[5]

	# Ensure units are decimal degrees
	params['radius'] = float(params['radius'])/3600.0

	#message = extract_star_lightcurves_on_position(params)
    #print(message)

	extract_star_lightcurve_isolated_reduction(params, log=None, format='dat',
												valid_data_only=False)
