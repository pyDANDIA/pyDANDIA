from os import path
from sys import argv
from pyDANDIA import crossmatch
from pyDANDIA import logs
from pyDANDIA import metadata
from pyDANDIA import match_utils
from pyDANDIA import vizier_tools

def crossmatch_field_with_gaia_DR():

    params = get_args()

    log = logs.start_stage_log( params['log_dir'], 'crossmatch_gaia' )

    xmatch = crossmatch.CrossMatchTable()
    xmatch.load(params['file_path'],log=log)

    log.info('Searching Vizier for data from '+params['gaia_dr']+' within '+\
            repr(params['search_radius'])+' of '+params['ra']+', '+params['dec'])
    gaia_data = search_vizier_for_sources(ra, dec, params['search_radius'], params['gaia_dr'])
    log.info('Vizier reports '+str(len(gaia_data))+' entries in '+params['gaia_dr']+' catalogue')

    xmatch.match_field_index_with_gaia_catalog(gaia_data, params, log)

    xmatch.save(params['file_path'])

    log.info('Crossmatch: complete')

    logs.close_log(log)
    
def get_args():

    params = {}
    if len(argv) < 2:
        params['crossmatch_file'] = input('Please enter the path to the crossmatch table for this field: ')
        params['gaia_dr'] = input('Please enter the ID of the Gaia data release to use [Gaia-DR2, Gaia-EDR3]: ')
        params['ra'] = input('Please enter search centroid RA in sexigesimal format: ')
        params['dec'] = input('Please enter search centroid Dec in sexigesimal format: ')
        params['separation_threshold'] = input('Please enter the maximum allowed separation in arcsec: ')
    else:
        params['crossmatch_file'] = argv[1]
        params['gaia_dr'] = argv[2]
        params['ra'] = argv[3]
        params['dec'] = argv[4]
        params['separation_threshold'] = argv[5]

    params['log_dir'] = path.dirname(params['crossmatch_file'])
    # Default FOV for ROME (LCO/Sinistro cameras)
    params['search_radius'] = 0.62 * u.debug

    return params

if __name__ == '__main__':
    crossmatch_field_with_gaia_DR()
