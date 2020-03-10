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
import matplotlib.pyplot as plt
from pyDANDIA import  phot_db
from pyDANDIA import  hd5_utils
from pyDANDIA import  pipeline_setup

def extract_star_lightcurves_on_position(params):
    """Function to extract a lightcurve from a phot_db"""

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

        photometry_data = fetch_photometry_for_dataset(params, star_field_id)

        setname = path.basename(params['red_dir']).split('_')[1]

        datafile = open(path.join(params['output_dir'],'star_'+str(star_field_id)+'_'+setname+'.dat'),'w')

        for i in range(0,len(photometry_data),1):

            datafile.write(str(photometry_data['hjd'][i])+'  '+\
                            str(photometry_data['instrumental_mag'][i])+'  '+str(photometry_data['instrumental_mag_err'][i])+'  '+\
                            str(photometry_data['calibrated_mag'][i])+'  '+str(photometry_data['calibrated_mag_err'][i])+'\n')

        datafile.close()
        print('-> Output dataset '+setname)

        message = 'OK'

    else:
        message = 'No stars within search region'

    conn.close()

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

def fetch_photometry_for_dataset(params, star_field_id):

    setup = pipeline_setup.pipeline_setup({'red_dir': params['red_dir']})

    dataset_photometry = hd5_utils.read_phot_hd5(setup)

    jdx = np.where(dataset_photometry[:,:,0] == star_field_id)

    index_list = np.unique(jdx[0])
    star_dataset_index = index_list[0]

    print('Star array index: '+str(star_dataset_index))

    photometry_data = dataset_photometry[star_dataset_index,:,:]

    photometry_data = table.Table( [ table.Column(name='hjd', data=dataset_photometry[star_dataset_index,:,9]),
                                     table.Column(name='instrumental_mag', data=dataset_photometry[star_dataset_index,:,11]),
                                     table.Column(name='instrumental_mag_err', data=dataset_photometry[star_dataset_index,:,12]),
                                      table.Column(name='calibrated_mag', data=dataset_photometry[star_dataset_index,:,13]),
                                      table.Column(name='calibrated_mag_err', data=dataset_photometry[star_dataset_index,:,14]),
                                      ] )

    return photometry_data

if __name__ == '__main__':

    params = {}

    if len(argv) == 1:

        params['db_file_path'] = input('Please enter the path to the field photometric DB: ')
        params['red_dir'] = input('Please enter the top-level path to the reduced datasets: ')
        params['ra'] = input('Please enter the RA [sexigesimal]: ')
        params['dec'] = input('Please enter the Dec [sexigesimal]: ')
        params['output_dir'] = input('Please enter the path to the output directory: ')

    else:

        params['db_file_path'] = argv[1]
        params['red_dir'] = argv[2]
        params['ra'] = argv[3]
        params['dec'] = argv[4]
        params['output_dir'] = argv[5]

    message = extract_star_lightcurves_on_position(params)
    print(message)
