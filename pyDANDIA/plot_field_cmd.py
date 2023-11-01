from sys import argv
from os import path
import numpy as np
from pyDANDIA import hd5_utils
from pyDANDIA import logs
from pyDANDIA import metadata
from pyDANDIA import pipeline_setup
from pyDANDIA import normalize_photometry
from pyDANDIA import crossmatch
from pyDANDIA import field_photometry
from pyDANDIA import plot_rms
from pyDANDIA import plotly_lightcurves
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def calc_field_cmds():

    params = get_args()

    log = logs.start_stage_log( params['red_dir'], 'field_cmd' )

    # Crossmatch table provides information on the filter used for each image
    xmatch = crossmatch.CrossMatchTable()
    xmatch.load(params['crossmatch_file'],log=log)
    log.info('Loaded crossmatch table for the field')

    # List of CMDs to plot.  Each tuple contains the magnitude passband
    # for the X-axis in the first entry and the two passbands to subtract
    # to create the colour.
    # Therefore (g, g, i) -> g vs. (g-i)
    plot_list = [ ('g', 'g', 'i'), ('g', 'g', 'r'), ('i', 'g', 'i'), ('r', 'r', 'i') ]
    log.info('Identified list of plots to create')

    # Identify stars with valid measurements.  Since the photometric calibration
    # is transformed relative to the primary reference, we select the data for
    # those stars only.
    primary_ref = 'lsc_doma'

    # Plot the CMD diagram for each combination of passbands
    for (f1, f2, f3) in plot_list:
        col1 = 'cal_'+f1+'_mag_'+primary_ref
        col4 = 'cal_'+f1+'_magerr_'+primary_ref
        col2 = 'cal_'+f2+'_mag_'+primary_ref
        col5 = 'cal_'+f2+'_magerr_'+primary_ref
        col3 = 'cal_'+f3+'_mag_'+primary_ref
        col6 = 'cal_'+f3+'_magerr_'+primary_ref

        jdx = np.logical_and(xmatch.stars[col1] > 0.0, xmatch.stars[col2] > 0.0)
        jdx = np.logical_and(jdx, xmatch.stars[col3] > 0.0)
        jdx = np.logical_and(jdx, xmatch.stars[col4] <= 0.1)
        jdx = np.logical_and(jdx, xmatch.stars[col5] <= 0.1)
        jdx = np.logical_and(jdx, xmatch.stars[col6] <= 0.1)
        select_stars = np.where(jdx)[0]
        data = np.zeros((len(select_stars),3))
        data[:,0] = xmatch.stars['field_id'][select_stars]
        data[:,1] = xmatch.stars[col2][select_stars] - xmatch.stars[col3][select_stars]
        data[:,2] = xmatch.stars[col1][select_stars]

        if params['target_field_id']:
            field_idx = params['target_field_id'] - 1
            target_mag = xmatch.stars[col1][field_idx]
            target_colour = xmatch.stars[col2][field_idx] - xmatch.stars[col3][field_idx]
            target_params = [params['target_field_id'], target_mag, target_colour]
        else:
            target_params = [None, None, None]

        log.info(str(len(select_stars))+' stars have valid measurements in '
                    +f1+', '+f2+' and '+f3+' bands')

        # Plot interactive RMS diagram
        plot_file = path.join(params['red_dir'], params['plot_file_root']+'_'+f1+'_'+f2+f3+'.html')
        axis_labels = ['('+f2+'-'+f3+') [mag]', f1+' [mag]']
        plotly_lightcurves.plot_interactive(data, plot_file, axis_labels,
                    target_params,
                    title=params['field_name']+' CMD', yreverse=True)

    logs.close_log(log)

def get_args():

    params = {}

    if len(argv) == 1:

        params['red_dir'] = input('Please enter the path to the top-level data directory: ')
        params['field_name'] = input('Please enter the name of the field: ')
        params['target_field_id'] = input('To highlight a specific object, enter its field ID or None: ')

    else:

        params['red_dir'] = argv[1]
        params['field_name'] = argv[2]
        if len(argv) == 4:
            params['target_field_id'] = argv[3]

    params['crossmatch_file'] = path.join(params['red_dir'],
                            params['field_name']+'_field_crossmatch.fits')
    params['plot_file_root'] = params['field_name']+'_cmd'
    if 'none' in str(params['target_field_id']).lower():
        params['target_field_id'] = None
    else:
        params['target_field_id'] = int(params['target_field_id'])

    return params


if __name__ == '__main__':
    calc_field_cmds()
