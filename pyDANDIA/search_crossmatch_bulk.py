from sys import argv
from os import getcwd, path
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units
from astropy.table import Table, Column
import argparse
import json

def search_xmatch_enmasse():

    # Get commandline arguments
    args = get_args()

    # Parse the input list of objects.
    # The format expected is a JSON dictionary, in the following format:
    # {"object_name": [ra, dec, "type", "comment", field_id]}
    # where RA, Dec are in decimal degrees, and the field_id indicates the
    # ROME field expected to contain the star.  Note the field_id is optional;
    # the objects will be matched on coordinates regardless.
    data = load_object_list(args.input_file)

    

def load_object_list(input_file):

    if not path.isfile(input_file):
        raise IOError('Cannot find input list of target coordinates at '
                        + input_file)

    with open(input_file, "r") as read_file:
        data = json.load(read_file)

    return data

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("xmatch_file", help="Path to crossmatch table", type=str)
    parser.add_argument("input_file", help="Path to input file containing list of objects in JSON format", type=str)
    parser.add_argument("output_file", help="Path to output file of matching objects, in JSON", type=str)
    parser.add_argument("radius", help="Search radius, in decimal arcsec", type=float)
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    search_xmatch_enmasse()
