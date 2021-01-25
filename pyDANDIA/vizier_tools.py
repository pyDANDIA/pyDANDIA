# -*- coding: utf-8 -*-
"""
Created on Wed May  9 11:35:31 2018

@author: rstreet
"""

from sys import argv
from astroquery.vizier import Vizier
from astroquery.gaia import Gaia
from astropy import wcs, coordinates, units, visualization, table
import requests

def search_vizier_for_sources(ra, dec, radius, catalog, row_limit=-1,
                              coords='sexigesimal', log=None, debug=False):
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
                                     {'_RAJ2000':'_RAJ2000', '_DEJ2000':'_DEJ2000', 'Jmag':'Jmag', 'e_Jmag':'e_Jmag', \
                                    'Hmag':'Hmag', 'e_Hmag':'e_Hmag','Kmag':'Kmag', 'e_Kmag':'e_Kmag'},
                                    {'Jmag':'<20'}],
                           'VPHAS+': ['II/341',
                                      {'sourceID':'sourceID', 'RAJ2000':'_RAJ2000', 'DEJ2000':'_DEJ2000',
                                      'gmag':'gmag', 'e_gmag':'e_gmag', 'rmag':'rmag', 'e_rmag':'e_rmag',
                                      'imag':'imag', 'e_imag':'e_imag', 'clean':'clean'},
                                    {}],
                            'Gaia-DR2': ['I/345/gaia2',
                                      {'RAJ2000':'ra', 'DEJ2000':'dec', 'Source':'source_id',
                                      'e_RAJ2000':'ra_error', 'e_DEJ2000':'dec_error',
                                      'FG':'phot_g_mean_flux', 'e_FG':'phot_g_mean_flux_error',
                                      'FBP':'phot_bp_mean_flux', 'e_FBP':'phot_bp_mean_flux_error',
                                      'FRP':'phot_rp_mean_flux', 'e_FRP':'phot_rp_mean_flux_error' },
                                    {}],
                            'Gaia-EDR3': ['I/350/gaiaedr3',
                                    {'RA_ICRS':'ra', 'DE_ICRS':'dec', 'Source':'source_id',
                                    'e_RA_ICRS':'ra_error', 'e_DE_ICRS':'dec_error',
                                    'FG':'phot_g_mean_flux', 'e_FG':'phot_g_mean_flux_error',
                                    'FBP':'phot_bp_mean_flux', 'e_FBP':'phot_bp_mean_flux_error',
                                    'FRP':'phot_rp_mean_flux', 'e_FRP':'phot_rp_mean_flux_error',
                                    'PM':'proper_motion', 'pmRA':'pm_ra', 'e_pmRA':'pm_ra_error',
                                    'pmDE':'pm_dec', 'e_pmDE':'pm_dec_error',
                                    'Plx':'parallax', 'e_Plx': 'parallax_error'},
                                    #'parallax':'parallax', 'parallax_error': 'parallax_error'},
                                    {}]
                           }

    (cat_id,cat_col_dict,cat_filters) = supported_catalogs[catalog]

    if catalog=='Gaia-EDR3':
        v = Vizier(column_filters=cat_filters)
    else:
        v = Vizier(columns=list(cat_col_dict.keys()),\
                column_filters=cat_filters)

    v = Vizier(column_filters=cat_filters)

    v.ROW_LIMIT = row_limit

    if 'sexigesimal' in coords:
        c = coordinates.SkyCoord(ra+' '+dec, frame='icrs', unit=(units.hourangle, units.deg))
    else:
        c = coordinates.SkyCoord(ra, dec, frame='icrs', unit=(units.deg, units.deg))

    r = radius * units.arcminute

    catalog_list = Vizier.find_catalogs(cat_id)

    (status, result) = query_vizier_servers(v, c, r, [cat_id], debug=debug)

    print(result[0])
    print(result[0].colnames)
    if len(result) == 1:
        col_list = []
        for col_id, col_name in cat_col_dict.items():
            print(col_id, col_name)
            col = table.Column(name=col_name, data=result[0][col_id].data)
            col_list.append(col)

        result = table.Table( col_list )

    return result

def query_vizier_servers(query_service, coord, search_radius, catalog_id, log=None,
                        debug=False):
    """Function to query different ViZier servers in order of preference, as
    a fail-safe against server outages.  Based on code from NEOExchange by
    T. Lister
    Input:
    query_service  Astroquery Vizier service object
    coord   SkyCoord Target coordinates
    search_radius   Angle    Search radius (deg)
    catalog_id  str      Name of catalog to be searched in ViZier's notation
    """

    vizier_servers_list = ['vizier.cfa.harvard.edu', 'vizier.hia.nrc.ca']

    query_service.VIZIER_SERVER = vizier_servers_list[0]

    query_service.TIMEOUT = 60

    continue_query = True
    iserver = 0
    status = True

    while continue_query:
        query_service.VIZIER_SERVER = vizier_servers_list[iserver]
        query_service.TIMEOUT = 60

        if debug:
            print('Searching catalog server '+repr(query_service.VIZIER_SERVER))

        if log != None:
            log.warning('Searching catalog server '+repr(query_service.VIZIER_SERVER))

        try:
            result = query_service.query_region(coord, radius=search_radius, catalog=catalog_id)

        # Handle long timeout requests:
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError):
            if debug:
                print('Catalog server '+repr(query_service.VIZIER_SERVER)+' timed out, trying longer timeout')
            if log!= None:
                log.warning('Catalog server '+repr(query_service.VIZIER_SERVER)+' timed out, trying longer timeout')

            query_service.TIMEOUT = 120
            result = query_service.query_region(coord, radius=search_radius, catalog=catalog_id)

        # Handle preferred-server timeout by trying the alternative server:
        except requests.exceptions.ConnectTimeout:
            if debug:
                print('Catalog server '+repr(query_service.VIZIER_SERVER)+' timed out again')
            if log != None:
                log.warning('Catalog server '+repr(query_service.VIZIER_SERVER)+' timed out again')

            iserver += 1
            if iserver >= len(vizier_servers_list):
                continue_query = False
                result = []
                status = False

                return status, result

        if len(result) > 0:
            continue_query = False
        elif len(result) == 0:
            iserver += 1
            if iserver >= len(vizier_servers_list):
                continue_query = False
                result = []
                status = False

    return status, result

def search_vizier_for_gaia_sources(ra, dec, radius, log=None):
    """Function to perform online query of the 2MASS catalogue and return
    a catalogue of known objects within the field of view
    """

    c = coordinates.SkyCoord(ra+' '+dec, frame='icrs', unit=(units.deg, units.deg))
    r = units.Quantity(radius/60.0, units.deg)

    query_service = Vizier(row_limit=1e6, column_filters={},
    columns=['RAJ2000', 'DEJ2000', 'e_RAJ2000', 'e_DEJ2000', 'Gmag', 'e_Gmag', 'Dup'])


    if log!=None:
        log.info('Searching for gaia sources within '+repr(r)+' of '+repr(c))

    try:
        qs = Gaia.cone_search_async(c, r)
    except AttributeError:
        if log!=None:
            log.info('No search results received from Vizier service')
        raise IOError('No search results received from Vizier service')
    except requests.exceptions.HTTPError:
        if log!=None:
            log.info('HTTP error while contacting ViZier')
        raise requests.exceptions.HTTPError()

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
        catalog = input('Please enter the ID of the catalog to search [2MASS, VPHAS+, Gaia-DR2, Gaia-EDR3]: ')

    else:

        ra = argv[1]
        dec = argv[2]
        radius = argv[3]
        catalog = argv[4]

    radius = float(radius)

    qs = search_vizier_for_sources(ra, dec, radius, catalog, debug=True)
    #qs = search_vizier_for_gaia_sources(ra, dec, radius)

    print(repr(qs))
