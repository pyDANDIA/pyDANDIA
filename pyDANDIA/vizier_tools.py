# -*- coding: utf-8 -*-
"""
Created on Wed May  9 11:35:31 2018

@author: rstreet
"""

from sys import argv
from astroquery.vizier import Vizier
from astroquery.gaia import Gaia
from astropy import wcs, coordinates, units, visualization

def search_vizier_for_sources(ra, dec, radius, catalog, row_limit=-1,
                              coords='sexigesimal'):
    """Function to perform online query of the 2MASS catalogue and return
    a catalogue of known objects within the field of view

    Inputs:
        :param str ra: RA J2000 in sexigesimal format [default, accepts degrees]
        :param str dec: Dec J2000 in sexigesimal format [default, accepts degrees]
        :param float radius: Search radius in arcmin
        :param str catalog: Catalog to search.  Options include:
                                    ['2MASS', 'VPHAS+']
    """

    supported_catalogs = { '2MASS': ['2MASS',
                                     ['_RAJ2000', '_DEJ2000', 'Jmag', 'e_Jmag', \
                                    'Hmag', 'e_Hmag','Kmag', 'e_Kmag'],
                                    {'Jmag':'<20'}],
                           'VPHAS+': ['II/341',
                                      ['sourceID', '_RAJ2000', '_DEJ2000', 'gmag', 'e_gmag', 'rmag', 'e_rmag', 'imag', 'e_imag', 'clean'],
                                    {}]
                           }

    (cat_id,cat_col_list,cat_filters) = supported_catalogs[catalog]

    v = Vizier(columns=cat_col_list,\
                column_filters=cat_filters)

    v.ROW_LIMIT = row_limit

    if 'sexigesimal' in coords:
        c = coordinates.SkyCoord(ra+' '+dec, frame='icrs', unit=(units.hourangle, units.deg))
    else:
        c = coordinates.SkyCoord(ra, dec, frame='icrs', unit=(units.deg, units.deg))

    r = radius * units.arcminute

    catalog_list = Vizier.find_catalogs(cat_id)

    result=v.query_region(c,radius=r,catalog=[cat_id])

    if len(result) == 1:
        result = result[0]

    return result

def search_vizier_for_gaia_sources(ra, dec, radius):
    """Function to perform online query of the 2MASS catalogue and return
    a catalogue of known objects within the field of view
    """

    c = coordinates.SkyCoord(ra+' '+dec, frame='icrs', unit=(units.deg, units.deg))
    r = units.Quantity(radius/60.0, units.deg)

    try:
        qs = Gaia.cone_search_async(c, r)
    except AttributeError:
        raise IOError('No search results received from Vizier service')
    result = qs.get_results()

    catalog = result['ra','dec','source_id','ra_error','dec_error',
                     'phot_g_mean_flux','phot_g_mean_flux_error',
                     'phot_rp_mean_flux','phot_rp_mean_flux_error',
                     'phot_bp_mean_flux','phot_bp_mean_flux_error']

    return catalog

if __name__ == '__main__':

    if len(argv) == 1:

        ra = input('Please enter search centroid RA in sexigesimal format: ')
        dec = input('Please enter search centroid Dec in sexigesimal format: ')
        radius = input('Please enter search radius in arcmin: ')
        catalog = input('Please enter the ID of the catalog to search [2MASS, VPHAS+]: ')

    else:

        ra = argv[1]
        dec = argv[2]
        radius = argv[3]
        catalog = argv[4]

    radius = float(radius)

    #qs = search_vizier_for_sources(ra, dec, radius, catalog)
    qs = search_vizier_for_gaia_sources(ra, dec, radius)

    print(repr(qs))
