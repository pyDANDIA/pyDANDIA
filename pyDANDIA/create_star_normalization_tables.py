from os import path, rename
from sys import argv
import numpy as np
import argparse
from pyDANDIA import crossmatch
from pyDANDIA import hd5_utils
from pyDANDIA import logs
from astropy.table import Table, Column

def init_normalization_tables(params):
    """Function to initialize the tables of the lightcurve normalization
    coefficients for each star in each dataset
    """

    log = logs.start_stage_log( params.red_dir, 'init_star_norm' )

    # First check to see if there is an existing normalization table file
    # already, to give the user a chance to save it
    output_file = path.join(params.red_dir,
                    params.field_name+'_star_dataset_normalizations.hdf5')
    if path.isfile(output_file):
        log.error('Error: Expected output file '+output_file+' already exists')
        raise IOError('Expected output file '+output_file+' already exists')

    # Read the field's crossmatch table.
    # This will be used to scale the size of the tables
    xmatch = crossmatch.CrossMatchTable()
    xmatch.load(params.crossmatch_file,log=log)

    # A set of datasets are declared to be the primary references for the
    # normalization process - meaning that all other datasets will be normalized
    # to these.
    ref_preference_order = ['lsc-doma', 'cpt-doma', 'coj-doma']

    # The data_table is designed to hold the magnitude offset and uncertainty
    # for each dataset from the primary reference in each case, so we
    # initalize tables for each of the primary references, and the number of
    # columns in each table depends on the number of datasets
    column_names = get_table_columns(xmatch)
    column_list = [ Column(name=column_names[0], data=xmatch.field_index['field_id'],
                            dtype='int') ]
    ns = len(xmatch.field_index)
    for col in column_names[1:]:
        column_list.append( Column(name=col, data=np.zeros(ns), dtype='float') )
    data_table = Table(column_list)

    tables = {}
    for ref in ref_preference_order:
        tables[ref] = data_table
    log.info('Initialized normalization coefficient tables for '+str(ns)+
                ' stars and '+str(len(xmatch.datasets))+' datasets')

    # Output the initialized table to file:
    hd5_utils.write_normalizations_hd5(params.red_dir, params.field_name,
                                        tables)
    log.info('Output initalized tables of star lightcurve normalization coefficients')
    logs.close_log(log)

def get_table_columns(xmatch):
    column_list = ['field_id']
    for dset in xmatch.datasets['dataset_code']:
        column_list.append('delta_mag_'+xmatch.get_dataset_shortcode(dset))
        column_list.append('delta_mag_error_'+xmatch.get_dataset_shortcode(dset))

    return column_list

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('red_dir', help='Path to the reduction directory')
    parser.add_argument('crossmatch_file', help='Path to the crosstable file')
    parser.add_argument('field_name', help='Field name, for use as file prefix')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    params = get_args()
    init_normalization_tables(params)
