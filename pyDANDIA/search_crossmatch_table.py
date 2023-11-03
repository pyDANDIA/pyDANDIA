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
from pyDANDIA import plotly_lightcurves
from pyDANDIA import field_photometry
import csv
import argparse

def search_xmatch():

    # Get commandline arguments
    args = get_args()

    # Load the CrossMatchTable
    xmatch = crossmatch.CrossMatchTable()
    xmatch.load(args.xmatch_file)

    # Search for the target in the Stars table of the CrossMatchTable
    results = search_on_coordinates(xmatch, args.ra, args.dec, args.radius)

    print(results)

def search_on_coordinates(xmatch, ra, dec, radius, log=None):
    """Function to search the crossmatch table for a given ROME field to find
    all potential matches with a given set of coordinates and search radius.

    Parameters:
        xmatch  CrossMatchTable object
        ra      float         Single position in RA, ICRS, decimal degrees
        dec     float         Single position in Dec, ICRS, decimal degrees
        radius  float         Search radius in arcsec
    """

    if log:
        log.info('Searching the crossmatch table for stars within '
                    + str(radius)+' arcsec of '
                    + repr(coord))

    # Build a table of SkyCoord objects of all catalog stars for future reference
    if not hasattr(xmatch, 'star_coords'):
        xmatch.get_star_skycoords()

    # Search the table for matching objects.
    # The results table is filtered to exclude the indices from individual
    # datasets since this information is generally not required.
    search_params = {'ra_centre': ra, 'dec_centre': dec, 'radius': radius/3600.0}
    full_results = xmatch.cone_search(search_params, log=log)

    return full_results['field_id', 'ra', 'dec', 'separation', 'cal_g_mag_lsc_doma', 'cal_r_mag_lsc_doma', 'cal_i_mag_lsc_doma']


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("xmatch_file", help="Path to crossmatch table", type=str)
    parser.add_argument("ra", help="RA, in ICRS frame and decimal degrees", type=float)
    parser.add_argument("dec", help="Dec, in ICRS frame and decimal degrees", type=float)
    parser.add_argument("radius", help="Search radius, in decimal arcsec", type=float)
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    search_xmatch()
