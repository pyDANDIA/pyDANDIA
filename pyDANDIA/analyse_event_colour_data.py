from sys import argv
from os import getcwd, path, remove, environ
import numpy as np
from astropy import table
from pyDANDIA import  photometry_classes
from pyDANDIA import  logs
from pyDANDIA import  event_colour_analysis
from pyDANDIA import  spectral_type_data
from pyDANDIA import  red_clump_utilities
from pyDANDIA import  config_utils

def run_event_colour_analysis():

    (input_file, output_dir) = get_args()

    source = photometry_classes.load_star_from_json()
    log = logs.start_stage_log( config['output_dir'], 'analyse_event_colours' )

    (source, blend) = match_source_blend_isochrones(params,source,blend,log)

    (source, blend) = calc_source_blend_ang_radii(source, blend, log)

    (source, blend) = calc_source_blend_physical_radii(source, blend, log)

    (source,blend) = calc_source_blend_distance(source, blend, RC, log)

    lens = calc_lens_parameters(params, source, RC, log)

    output_red_clump_data_latex(params,RC,log)

    output_source_blend_data_latex(params,source,blend,log)

    output_lens_parameters_latex(params,source,lens,log)


    logs.close_log(log)

def get_args():

    if len(argv) == 1:

        input_file = input('Please enter the path to the dereddened event photometry results file: ')
        output_dir = input('Please enter the path to the output directory: ')

    else:

        input_file = argv[1]
        output_dir = argv[2]

    return input_file, output_dir
