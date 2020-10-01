from os import path
from sys import argv
from pyDANDIA import metadata
from astropy.io import fits
import numpy as np

def add_star_to_star_catalog(params):
    """Tool to add a star to a pre-existing star catalog for a reduction"""

    reduction_metadata = metadata.MetaData()
    reduction_metadata.load_all_metadata(params['red_dir'], 'pyDANDIA_metadata.fits')

    star_catalog = reduction_metadata.star_catalog[1]

    row_data = [ len(star_catalog), params['x'], params['y'],
                                    params['ra'], params['dec'],
                                    params['ref_flux'], params['ref_flux_error'],
                                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                    None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                    None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    star_catalog.add_row( row_data )

    reduction_metadata.star_catalog[1] = star_catalog
    reduction_metadata.save_updated_metadata(params['red_dir'],'pyDANDIA_metadata.fits')

if __name__ == '__main__':

    params = {'red_dir': None, 'x': None, 'y': None, 'ra': None, 'dec': None,
              'ref_flux': None, 'ref_flux_error': None}

    if len(argv) > 1:
        for a in argv[1:]:
            (key, value) = a.split('=')
            if key in params.keys():
                if key == 'red_dir':
                    params[key] = value
                else:
                    params[key] = float(value)

    else:
        print('Please enter the following parameters for the new object:')
        for key in params.keys():
            params[key] = input(key+': ')

    for key, value in params.items():
        if not value:
            raise IOError('Missing parameter '+key)

    add_star_to_star_catalog(params)
