"""
Created on Sat Sept 7

@author: rstreet
"""
import numpy as np
import os
from sys import argv
import sqlite3
from os import getcwd, path
import numpy as np
from astropy import table
from astropy.coordinates import SkyCoord
from astropy import units
from pyDANDIA import  photometry_classes
from pyDANDIA import  phot_db
from pyDANDIA import  logs
from pyDANDIA import  plot_cmd

def compute_star_colours():
    """Function to extract the 3-band reference magnitudes for the primary
    reference dataset for a field and calculate the star colours, storing
    the results in the phot DB for the field."""

    params = get_args()

    log = logs.start_stage_log( params['log_dir'], 'compute_star_colours' )
    log.info('Starting parameters: '+repr(params))

    conn = phot_db.get_connection(dsn=params['db_file_path'])
    phot_db.ensure_extra_table(conn,'StarColours')

    primary_facility = phot_db.find_primary_reference_facility(conn,log)

    (photometry, stars) = plot_cmd.extract_reference_instrument_calibrated_photometry(conn,log)

    photometry = plot_cmd.calculate_colours(photometry,stars,log)

    output_colours_to_photdb(conn, primary_facility, stars, photometry)

    conn.close()

    logs.close_log(log)

def get_args():

    params = {}

    if len(argv) < 3:

        params['db_file_path'] = input('Please enter the path to the photometry database for the field: ')
        params['log_dir'] = input('Please enter the directory path for output: ')

    else:

        params['db_file_path'] = argv[1]
        params['log_dir'] = argv[2]

    return params

def output_colours_to_photdb(conn, primary_facility, stars, photometry):

    keys = ['star_id', 'facility',
            'cal_mag_corr_g', 'cal_mag_corr_g_err',
            'cal_mag_corr_r','cal_mag_corr_r_err',
            'cal_mag_corr_i', 'cal_mag_corr_i_err',
            'gi', 'gi_err', 'gr', 'gr_err', 'ri', 'ri_err']
    key_string = ', '.join(keys)

    value_string = ','.join( ['?']*len(keys) )

    command = 'INSERT OR REPLACE INTO star_colours ('+key_string+') VALUES ('+value_string+')'

    entries = []

    for j,star in enumerate(stars):

        entries.append( ( str(star['star_id']), str(primary_facility['facility_id'][0]), \
                        str(photometry['g'][j]), str(photometry['gerr'][j]), \
                        str(photometry['r'][j]), str(photometry['rerr'][j]), \
                        str(photometry['i'][j]), str(photometry['ierr'][j]), \
                        str(photometry['gi'][j]), str(photometry['gi_err'][j]), \
                        str(photometry['gr'][j]), str(photometry['gr_err'][j]), \
                        str(photometry['ri'][j]), str(photometry['ri_err'][j]), \
                        ) )

    cursor = conn.cursor()

    cursor.executemany(command, entries)

    conn.commit()

if __name__ == '__main__':

    compute_star_colours()
