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

    config = get_args()

    log = logs.start_stage_log( config['output_dir'], 'analyse_event_colours' )

    (source, blend, RC) = load_event_data(config,log)

    (source,blend) = calc_phot_properties(source, blend, RC, log)

    (source, blend) = event_colour_analysis.match_source_blend_isochrones(config,source,blend,log)

    (source, blend) = event_colour_analysis.calc_source_blend_ang_radii(source, blend, log)

    (source, blend) = event_colour_analysis.calc_source_blend_physical_radii(source, blend, log)

    (source,blend) = event_colour_analysis.calc_source_blend_distance(source, blend, RC, log)

    lens = event_colour_analysis.calc_lens_parameters(config, source, RC, log)

    event_colour_analysis.output_source_blend_data_latex(config,source,blend,log)

    event_colour_analysis.output_lens_parameters_latex(config,source,lens,log)

    logs.close_log(log)

def get_args():

    if len(argv) == 1:

        config_file = input('Please enter the path to the configuration file: ')

    else:

        config_file = argv[1]

    config = config_utils.build_config_from_json(config_file)
    print(config)

    return config

def load_event_data(config,log):
    """Function to load the parameters of the source, blend and Red Clump"""

    source = photometry_classes.Star(file_path=config['source_parameters_file'])
    log.info('Loaded source data:')
    log.info(source.summary(show_mags=True))
    log.info(source.summary(show_mags=False,show_colours=True))
    log.info(source.summary(show_mags=False,johnsons=True))

    blend = photometry_classes.Star(file_path=config['blend_parameters_file'])
    log.info('Loaded blend data:')
    log.info(blend.summary(show_mags=True))
    log.info(blend.summary(show_mags=False,show_colours=True))
    log.info(blend.summary(show_mags=False,johnsons=True))

    RC = photometry_classes.Star(file_path=config['red_clump_parameters_file'])
    log.info('Loaded Red Clump data:')
    log.info(RC.summary(show_mags=True))
    log.info(RC.summary(show_mags=False,show_colours=True))
    log.info(RC.summary(show_mags=False,johnsons=True))

    log.info('\n')
    log.info('Extinction, d(g) = '+str(RC.A_g)+' +/- '+str(RC.sig_A_g)+'mag')
    log.info('Extinction, d(r) = '+str(RC.A_r)+' +/- '+str(RC.sig_A_r)+'mag')
    log.info('Extinction, d(i) = '+str(RC.A_i)+' +/- '+str(RC.sig_A_i)+'mag')
    log.info('Reddening, E(g-r) = '+str(RC.Egr)+' +/- '+str(RC.sig_Egr)+'mag')
    log.info('Reddening, E(g-i) = '+str(RC.Egi)+' +/- '+str(RC.sig_Egi)+'mag')
    log.info('Reddening, E(r-i) = '+str(RC.Eri)+' +/- '+str(RC.sig_Eri)+'mag')

    log.info('\n')
    log.info('Extinction, d(V) = '+str(RC.A_V)+' +/- '+str(RC.sig_A_V)+'mag')
    log.info('Extinction, d(I) = '+str(RC.A_I)+' +/- '+str(RC.sig_A_I)+'mag')
    log.info('Reddening, E(V-I) = '+str(RC.EVI)+' +/- '+str(RC.sig_EVI)+'mag')

    return source, blend, RC

def calc_phot_properties(source, blend, RC, log):
    """Function to calculate the de-reddened and extinction-corrected
    photometric properties of the target
    """

    source.calibrate_phot_properties(RC,log=log)
    blend.calibrate_phot_properties(RC,log=log)

    log.info('\nSource star extinction-corrected magnitudes and de-reddened colours:\n')
    log.info(source.summary(show_mags=True))
    log.info(source.summary(show_mags=False,show_colours=True))
    log.info(source.summary(show_mags=False,show_cal=True))
    log.info(source.summary(show_mags=False,show_cal=True,show_colours=True))
    log.info(source.summary(show_mags=False,johnsons=True,show_cal=True))

    log.info('\nBlend extinction-corrected magnitudes and de-reddened colours:\n')
    log.info(blend.summary(show_mags=True))
    log.info(blend.summary(show_mags=False,show_colours=True))
    log.info(blend.summary(show_mags=False,show_cal=True))
    log.info(blend.summary(show_mags=False,show_cal=True,show_colours=True))
    log.info(blend.summary(show_mags=False,johnsons=True,show_cal=True))

    return source,blend

if __name__ == '__main__':
    run_event_colour_analysis()
