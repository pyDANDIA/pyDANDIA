from os import path
from sys import argv
from pyDANDIA import crossmatch
from pyDANDIA import logs
from pyDANDIA import metadata
from pyDANDIA import match_utils
from pyDANDIA import vizier_tools
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table, Column
import numpy as np

def crossmatch_field_with_gaia_DR():

    params = get_args()

    log = logs.start_stage_log( params['log_dir'], 'crossmatch_gaia' )

    xmatch = crossmatch.CrossMatchTable()
    xmatch.load(params['crossmatch_file'],log=log)

    # Clear any previous Gaia identifications to avoid confusion over the source
    xmatch.field_index['gaia_source_id'].fill(None)

    if params['gaia_catalog_file'] == 'query':
        log.info('Searching Vizier for data from '+params['gaia_dr']+' within '+\
            repr(params['search_radius'])+' arcmin of '+params['ra']+', '+params['dec'])
        gaia_data = vizier_tools.search_vizier_for_sources(params['ra'], params['dec'],
                                    params['search_radius'], params['gaia_dr'],
                                    row_limit=-1,log=log,debug=True)
        log.info('Vizier reports '+str(len(gaia_data))+' entries in '+params['gaia_dr']+' catalogue')

    else:
        gaia_data = load_gaia_catalog(params, log)

    gaia_data = clean_gaia_catalog(gaia_data, log)

    xmatch.match_field_index_with_gaia_catalog(gaia_data, params, log)

    xmatch.save(params['crossmatch_file'])

    log.info('Crossmatch: complete')

    logs.close_log(log)

def load_gaia_catalog(params, log):
    """Function to load a preexisting Gaia catalog from file"""

    if not path.isfile(params['gaia_catalog_file']):
        log.into('Cannot find Gaia catalog file '+params['gaia_catalog_file'])
        raise IOError('Cannot find Gaia catalog file '+params['gaia_catalog_file'])

    data = fits.open(params['gaia_catalog_file'])

    table_data = []
    for i,col in enumerate(data[1].columns.names):
        table_data.append( Column(name=col, data=data[1].data[col]) )

    gaia_data = Table(table_data)

    log.info('Loaded data for '+str(len(gaia_data))+' stars from the catalog file '+\
                params['gaia_catalog_file'])
    
    return gaia_data

def clean_gaia_catalog(gaia_data, log):
    """Function to search for and remove Gaia catalog entries with NaNs in the
    RA and Dec columns, to avoid this causing issues during the crossmatch"""

    jdx1 = np.where(np.isnan(gaia_data['ra']))[0]
    gaia_data.remove_rows(jdx1)
    jdx2 = np.where(np.isnan(gaia_data['dec']))[0]
    gaia_data.remove_rows(jdx2)

    log.info('Scanned the Gaia data for NaN entries, found and removed '+str(len(jdx1)+len(jdx2)))

    return gaia_data

def get_args():

    params = {}
    if len(argv) < 2:
        params['crossmatch_file'] = input('Please enter the path to the crossmatch table for this field: ')
        params['gaia_dr'] = input('Please enter the ID of the Gaia data release to use [Gaia-DR2, Gaia-EDR3]: ')
        params['gaia_catalog_file'] = input('Please enter the path to the Gaia catalog file or enter "query": ')
        params['ra'] = input('Please enter search centroid RA in sexigesimal format: ')
        params['dec'] = input('Please enter search centroid Dec in sexigesimal format: ')
        params['separation_threshold'] = input('Please enter the maximum allowed separation in arcsec: ')
    else:
        params['crossmatch_file'] = argv[1]
        params['gaia_dr'] = argv[2]
        params['gaia_catalog_file'] = argv[3]
        params['ra'] = argv[4]
        params['dec'] = argv[5]
        params['separation_threshold'] = argv[6]

    params['log_dir'] = path.dirname(params['crossmatch_file'])
    # Default FOV for ROME (LCO/Sinistro cameras) in arcmin as expected by
    # Vizier query, matches ROME pipeline default
    params['search_radius'] = 18.8
    params['separation_threshold'] = float(params['separation_threshold'])/3600.0 * u.deg

    return params

if __name__ == '__main__':
    crossmatch_field_with_gaia_DR()
