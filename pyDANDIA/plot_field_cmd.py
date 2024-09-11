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
    plot_list = get_plot_list(params)
    log.info('Identified list of plots to create')

    # Identify stars with valid measurements.  Since the photometric calibration
    # is transformed relative to the primary reference, we select the data for
    # those stars only.

    # Plot the CMD diagram for each combination of passbands
    for (fy, f1, f2) in plot_list:
        y_col = 'cal_'+fy+'_mag_'+params['primary_ref']
        yerr_col = 'cal_'+fy+'_magerr_'+params['primary_ref']
        mag_col_blue = 'cal_'+f1+'_mag_'+params['primary_ref']
        merr_col_blue = 'cal_'+f1+'_magerr_'+params['primary_ref']
        mag_col_red = 'cal_'+f2+'_mag_'+params['primary_ref']
        merr_col_red = 'cal_'+f2+'_magerr_'+params['primary_ref']

        jdx = np.logical_and(xmatch.stars[mag_col_blue] > 0.0, xmatch.stars[mag_col_red] > 0.0)
        jdx = np.logical_and(jdx, xmatch.stars[merr_col_blue] <= 0.5)
        jdx = np.logical_and(jdx, xmatch.stars[merr_col_red] <= 0.5)
        select_stars = np.where(jdx)[0]
        data = np.zeros((len(select_stars),3))
        cmd_data = np.zeros((len(select_stars),7))
        data[:,0] = xmatch.stars['field_id'][select_stars]
        data[:,1] = xmatch.stars[mag_col_blue][select_stars] - xmatch.stars[mag_col_red][select_stars]
        data[:,2] = xmatch.stars[y_col][select_stars]
        cmd_data[:,0] = data[:,0]
        cmd_data[:,1] = xmatch.stars[mag_col_blue][select_stars]
        cmd_data[:,2] = xmatch.stars[merr_col_blue][select_stars]
        cmd_data[:,3] = xmatch.stars[mag_col_red][select_stars]
        cmd_data[:,4] = xmatch.stars[merr_col_red][select_stars]
        cmd_data[:,5] = data[:,1]
        cmd_data[:,6] = calc_colour_uncertainty(xmatch.stars[merr_col_blue][select_stars],
                                                xmatch.stars[merr_col_red][select_stars])

        if params['target_field_id']:
            field_idx = params['target_field_id'] - 1
            target_mag = xmatch.stars[y_col][field_idx]
            target_colour = xmatch.stars[mag_col_blue][field_idx] - xmatch.stars[mag_col_red][field_idx]
            target_colour_err = calc_colour_uncertainty(xmatch.stars[merr_col_blue][field_idx],
                                                xmatch.stars[merr_col_red][field_idx])
            target_params = [params['target_field_id'], target_colour, target_mag]
            log.info('Identified target star ' + str(params['target_field_id']) + ' with photometry: \n'
                     + f1 + ' = ' + str(xmatch.stars[mag_col_blue][field_idx])
                     + ' +/- ' + str(xmatch.stars[merr_col_blue][field_idx])
                     + ', ' + f2 + ' = ' + str(xmatch.stars[mag_col_red][field_idx])
                     + ' +/- ' + str(xmatch.stars[merr_col_red][field_idx])
                     + ', ' + f1+'-'+f2 + ' = ' + str(target_colour) + ' +/- ' + str(target_colour_err)
                     )
        else:
            target_params = [None, None, None]

        if params['f3']:
            log.info(str(len(select_stars))+' stars have valid measurements in '
                    +f1+', '+f2+' and '+f3+' bands')
        else:
            log.info(str(len(select_stars)) + ' stars have valid measurements in '
                     + f1 + ' and ' + f2 + ' bands')

        # Plot interactive RMS diagram
        plot_file = path.join(params['red_dir'], params['plot_file_root']+'_'+fy+'_'+f1+f2+'.html')
        axis_labels = ['('+f1+'-'+f2+') [mag]', fy+' [mag]']
        plotly_lightcurves.plot_interactive(data, plot_file, axis_labels,
                    target_params,
                    title=params['field_name']+' CMD', yreverse=True)

        # Output CMD data file
        output_cmd_data(params, cmd_data, f1, f2)

    # Plot colour-colour diagram (g-i) .vs. (r-i)
    if params['f3']:
        col1 = 'cal_g_mag_' + params['primary_ref']
        col2 = 'cal_r_mag_' + params['primary_ref']
        col3 = 'cal_i_mag_' + params['primary_ref']
        col4 = 'cal_g_magerr_' + params['primary_ref']
        col5 = 'cal_r_magerr_' + params['primary_ref']
        col6 = 'cal_i_magerr_' + params['primary_ref']

        jdx = np.logical_and(xmatch.stars[col1] > 0.0, xmatch.stars[col2] > 0.0)
        jdx = np.logical_and(jdx, xmatch.stars[col3] > 0.0)
        jdx = np.logical_and(jdx, xmatch.stars[col4] <= 0.1)
        jdx = np.logical_and(jdx, xmatch.stars[col5] <= 0.1)
        jdx = np.logical_and(jdx, xmatch.stars[col6] <= 0.1)
        select_stars = np.where(jdx)[0]
        data = np.zeros((len(select_stars), 3))
        data[:, 0] = xmatch.stars['field_id'][select_stars]
        data[:, 1] = xmatch.stars[col1][select_stars] - xmatch.stars[col2][select_stars] # Watch array indexing
        data[:, 2] = xmatch.stars[col2][select_stars] - xmatch.stars[col3][select_stars]

        if params['target_field_id']:
            field_idx = params['target_field_id'] - 1
            target_x = xmatch.stars[col1][field_idx] - xmatch.stars[col2][field_idx]
            target_y = xmatch.stars[col2][field_idx] - xmatch.stars[col3][field_idx]
            target_params = [params['target_field_id'], target_y, target_x]
        else:
            target_params = [None, None, None]

        log.info(str(len(select_stars)) + ' stars have valid measurements in '
                 + f1 + ', ' + f2 + ' and ' + f3 + ' bands')

        # Plot interactive colour-colour diagram
        plot_file = path.join(params['red_dir'], params['plot_file_root'] + '_colour_colour.html')
        axis_labels = ['SDSS (g-r) [mag]', 'SDSS (r-i) [mag]']
        plotly_lightcurves.plot_interactive(data, plot_file, axis_labels,
                                            target_params,
                                            title=params['field_name'])

    logs.close_log(log)

def output_cmd_data(params, cmd_data, f1, f2):
    """
    Output a datafile containing the data for a CMD
    """

    output_file = path.join(params['red_dir'], 'cmd_data_' + f1+f2 + '.csv')
    fmt = '%i,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f'

    header = 'Star Field ID, mag (' + f1 + '), mag_error (' + f1 + '), mag (' + f2 + '), mag_error (' + f2 + '), ' \
            + f1+'-'+f2 + ' [mag], ('+f1+'-'+f2+') error [mag]'

    np.savetxt(output_file, cmd_data, delimiter=',', header=header, fmt=fmt)

def calc_colour_uncertainty(blue_band_merr, red_band_merr):

    errs = np.sqrt((blue_band_merr * blue_band_merr) + \
            (red_band_merr * red_band_merr))

    return errs

def get_plot_list(params):
    """
    Identify the list of CMDs to plot.  Each tuple contains the magnitude passband
    for the X-axis in the first entry and the two passbands to subtract
    to create the colour.
    Therefore (g, g, i) -> g vs. (g-i)
    """
    if not params['f3']:
        plot_list = [
            (params['f1'], params['f1'], params['f2'])
        ]
    else:
        plot_list = [('g', 'g', 'i'), ('g', 'g', 'r'), ('i', 'g', 'i'), ('r', 'r', 'i')]

    return plot_list

def get_args():

    params = {}

    if len(argv) == 1:

        params['red_dir'] = input('Please enter the path to the top-level data directory: ')
        params['field_name'] = input('Please enter the name of the field: ')
        params['primary_ref'] = input('Site-tel identifier to use as the primary reference dataset, e.g. lsc_doma: ')
        params['f1'] = input('First (bluest) filter in the set (one of {g, r, i}):')
        params['f2'] = input('Second (redward) filter in the set (one of {g, r, i}):')
        params['f3'] = input('Third (reddest) filter in the set (one of {g, r, i} or None):')
        params['target_field_id'] = input('To highlight a specific object, enter its field ID or None: ')

    else:

        params['red_dir'] = argv[1]
        params['field_name'] = argv[2]
        params['primary_ref'] = argv[3]
        params['f1'] = argv[4]
        params['f2'] = argv[5]
        params['f3'] = argv[6]
        if len(argv) == 8:
            params['target_field_id'] = argv[7]
        else:
            params['target_field_id'] = None

    if 'none' in str(params['f3']).lower():
        params['f3'] = None
    if 'none' in str(params['target_field_id']).lower():
        params['target_field_id'] = None

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
