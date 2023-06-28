from sys import argv
from os import getcwd, path
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units
from astropy.table import Table, Column
import argparse
import json
from pyDANDIA import search_crossmatch_table
from pyDANDIA import crossmatch

def search_xmatch_enmasse():

    # Get commandline arguments
    args = get_args()

    # Parse the input list of objects.
    # The format expected is a JSON dictionary, in the following format:
    # {"object_name": [ra, dec, "type", "comment", field_id]}
    # where RA, Dec are in decimal degrees, and the field_id indicates the
    # ROME field expected to contain the star.  Note the field_id is optional;
    # the objects will be matched on coordinates regardless.
    targets = load_object_list(args.input_file)

    # Load the crossmatch table
    xmatch = crossmatch.CrossMatchTable()
    xmatch.load(args.xmatch_file)
    xmatch.get_star_skycoords()

    # For each target, search the CrossMatchTable to find objects matching
    # the target's coordinates.  This may result in zero to multiple hits.
    # Note the expected search radius is in arcsec.
    targetlist = list(targets.keys())
    search_results = {}
    for targetname in targetlist:
        target_data = targets[targetname]

        results = search_crossmatch_table.search_on_coordinates(xmatch,
                                        target_data['ra'], target_data['dec'],
                                        args.radius)
        if len(results) > 1:
            print(targetname+': found '+str(len(results))+' matches')
        elif len(results) == 1:
            print(targetname+': found '+str(len(results))+' match')
        else:
            print(targetname+': no matches')

        if len(results) > 0:
            rome_stars = []
            for j in range(0,len(results),1):
                field_idx = results['field_id'][j] - 1
                qid = int(xmatch.field_index[field_idx]['quadrant'])
                rome_stars.append({'field_id': int(results['field_id'][j]),
                                    'ra': results['ra'][j],
                                    'dec': results['dec'][j],
                                    'quadrant': qid,
                                    'separation_deg': results['separation'][j]})
            search_results[targetname] = {'target_data': target_data,
                                          'search_radius_deg': args.radius/3600.0,
                                            'rome_stars': rome_stars}

    # Output the search results as a JSON file:
    output_search_results(args.output_file, search_results)

def load_object_list(input_file):

    if not path.isfile(input_file):
        raise IOError('Cannot find input list of target coordinates at '
                        + input_file)

    with open(input_file, "r") as read_file:
        data = json.load(read_file)

    return data

def output_search_results(output_file, search_results):

    json_data = json.dumps(search_results, indent=4)
    with open(output_file, 'w') as write_file:
        write_file.write(json_data)
        write_file.close()

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
