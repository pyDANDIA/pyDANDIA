from os import path
from sys import argv
from pyDANDIA import analyse_cmd
from pyDANDIA import config_utils
from pyDANDIA import  logs
from pyDANDIA import  crossmatch
from astropy.coordinates import SkyCoord, Angle
from astropy import units as u
from astropy.table import Column, Table
import numpy as np
import copy
import matplotlib as mpl
import matplotlib.pyplot as plt

def field_colour_analysis():

    config = get_args()

    log = logs.start_stage_log( config['output_dir'], 'field_cmd' )

    xmatch = crossmatch.CrossMatchTable()
    xmatch.load(config['field_xmatch_file'],log=log)

    xmatch = calc_colour_photometry(config, xmatch, log)

    (valid_stars, selected_stars) = apply_star_selection(config, xmatch, log)

    plot_field_colour_mag_diagram(config, xmatch, valid_stars, selected_stars, '(g-i)', 'g', log)
    plot_field_colour_mag_diagram(config, xmatch, valid_stars, selected_stars, '(r-i)', 'i', log)
    plot_field_colour_mag_diagram(config, xmatch, valid_stars, selected_stars, '(r-i)', 'r', log)
    plot_field_colour_mag_diagram(config, xmatch, valid_stars, selected_stars, '(g-r)', 'g', log)
    plot_field_colour_mag_diagram(config, xmatch, valid_stars, selected_stars, '(g-i)', 'i', log)

    plot_field_colour_colour_diagram(config, xmatch, valid_stars, selected_stars, log)

    output_photometry(config, xmatch, selected_stars, log)

    logs.close_log(log)

def plot_field_colour_mag_diagram(config, xmatch, valid_stars, selected_stars,
                                    colour, magnitude, log):
    """Function to plot a colour-magnitude diagram from the field cross-match
    table, for the selected colour index {(g-r), (r-i), (g-r)} and magnitude
    where the magnitude parameter is one of {g, r, i}"""

    fig = plt.figure(1,(10,10))

    ax = plt.subplot(111)

    plt.rcParams.update({'font.size': 25})

    mag_column = 'cal_'+magnitude+'_mag_'+config['reference_dataset_code']
    (default_marker_colour, field_marker_colour, marker_colour) = plot_data_colours()
    if len(selected_stars) < len(valid_stars):
        marker_colour = field_marker_colour

    # Plot selected field stars
    if not config['plot_selected_stars_only']:
        plt.scatter(xmatch.stars[colour][valid_stars],xmatch.stars[mag_column][valid_stars],
                 c=marker_colour, marker='.', s=1,
                 label='Stars within field of view')

    plt.scatter(xmatch.stars[colour][selected_stars],xmatch.stars[mag_column][selected_stars],
              c=default_marker_colour, marker='*', s=1,
              label='Stars meeting selection criteria')


    plt.xlabel('SDSS '+colour+' [mag]')

    plt.ylabel('SDSS-'+magnitude+' [mag]')

    [xmin,xmax,ymin,ymax] = plt.axis()
    col_key = colour.replace('(','').replace(')','').replace('-','')
    xmin = config['plot_'+col_key+'_range'][0]
    xmax = config['plot_'+col_key+'_range'][1]
    ymin = config['plot_'+magnitude+'_range'][0]
    ymax = config['plot_'+magnitude+'_range'][1]
    plt.axis([xmin,xmax,ymax,ymin])

    xticks = np.arange(xmin,xmax,0.1)
    yticks = np.arange(ymin,ymax,0.2)

    #ax.set_xticks(xticks,minor=True)
    ax.set_xticklabels(xticks,minor=True, fontdict={'size': 25})
    #ax.set_yticks(yticks,minor=True)
    ax.set_yticklabels(yticks,minor=True,fontdict={'size': 25})
    ax.title.set_size(25)
    ax.xaxis.label.set_fontsize(25)
    ax.yaxis.label.set_fontsize(25)
    ax.tick_params(axis="x", labelsize=25)
    ax.tick_params(axis="y", labelsize=25)

    plot_file = path.join(config['output_dir'],'colour_magnitude_diagram_'+\
                                            magnitude+'_vs_'+colour.replace('(','').replace(')','')\
                                            +'.pdf')
    plt.grid()

    if config['legend']:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * -0.025,
                 box.width, box.height * 0.95])

        l = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=2)

        l.legendHandles[0]._sizes = [50]
        if len(l.legendHandles) > 1:
            l.legendHandles[1]._sizes = [50]

        plt.rcParams.update({'legend.fontsize':25})
    plt.rcParams.update({'font.size':25})
    plt.rcParams.update({'axes.titlesize': 25})
    plt.rcParams.update({'font.size': 25})

    if config['interactive']:
        plt.show()
    else:
        plt.savefig(plot_file,bbox_inches='tight')

    plt.close(1)

    log.info('Colour-magnitude diagram output to '+plot_file)

def plot_field_colour_colour_diagram(config, xmatch, valid_stars, selected_stars, log):
    """Function to plot a colour-colour diagram from the field cross-match
    table, for colour indices (g-i) .vs. (r-i)"""

    fig = plt.figure(1,(10,10))

    ax = plt.subplot(111)

    plt.rcParams.update({'font.size': 25})

    colour1 = '(g-i)'
    colour2 = '(r-i)'
    (default_marker_colour, field_marker_colour, marker_colour) = plot_data_colours()
    if len(selected_stars) < len(valid_stars):
        marker_colour = field_marker_colour

    # Plot selected field stars
    if not config['plot_selected_stars_only']:
        plt.scatter(xmatch.stars[colour1][valid_stars],xmatch.stars[colour2][valid_stars],
                 c=marker_colour, marker='.', s=1,
                 label='Stars within field of view')

    plt.scatter(xmatch.stars[colour1][selected_stars],xmatch.stars[colour2][selected_stars],
              c=default_marker_colour, marker='*', s=1,
              label='Stars meeting selection criteria')


    plt.xlabel('SDSS '+colour1+' [mag]')

    plt.ylabel('SDSS-'+colour2+' [mag]')

    [xmin,xmax,ymin,ymax] = plt.axis()
    col_key1 = colour1.replace('(','').replace(')','').replace('-','')
    col_key2 = colour2.replace('(','').replace(')','').replace('-','')
    xmin = config['plot_'+col_key1+'_range'][0]
    xmax = config['plot_'+col_key1+'_range'][1]
    ymin = config['plot_'+col_key2+'_range'][0]
    ymax = config['plot_'+col_key2+'_range'][1]
    plt.axis([xmin,xmax,ymin,ymax])

    xticks = np.arange(xmin,xmax,0.1)
    yticks = np.arange(ymin,ymax,0.2)

    #ax.set_xticks(xticks,minor=True)
    ax.set_xticklabels(xticks,minor=True, fontdict={'size': 25})
    #ax.set_yticks(yticks,minor=True)
    ax.set_yticklabels(yticks,minor=True,fontdict={'size': 25})
    ax.title.set_size(25)
    ax.xaxis.label.set_fontsize(25)
    ax.yaxis.label.set_fontsize(25)
    ax.tick_params(axis="x", labelsize=25)
    ax.tick_params(axis="y", labelsize=25)

    plot_file = path.join(config['output_dir'],'colour_colour_diagram_'\
                                            +colour1.replace('(','').replace(')','')\
                                            +'_vs_'+colour2.replace('(','').replace(')','')\
                                            +'.pdf')
    plt.grid()

    if config['legend']:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * -0.025,
                 box.width, box.height * 0.95])

        l = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=2)

        l.legendHandles[0]._sizes = [50]
        if len(l.legendHandles) > 1:
            l.legendHandles[1]._sizes = [50]

        plt.rcParams.update({'legend.fontsize':25})
    plt.rcParams.update({'font.size':25})
    plt.rcParams.update({'axes.titlesize': 25})
    plt.rcParams.update({'font.size': 25})

    if config['interactive']:
        plt.show()
    else:
        plt.savefig(plot_file,bbox_inches='tight')

    plt.close(1)

    log.info('Colour-colour diagram output to '+plot_file)

def plot_data_colours():
    #default_marker_colour = '#8c6931'
    #field_marker_colour = '#E1AE13'
    default_marker_colour = '#000000'
    field_marker_colour = '#E1AE13'
    marker_colour = default_marker_colour
    return default_marker_colour, field_marker_colour, marker_colour

def calc_colour_data(blue_phot, blue_phot_err, red_phot, red_phot_err):

    col_data = np.zeros(len(red_phot))
    col_data.fill(-99.999)
    col_data_err = np.zeros(len(red_phot))
    col_data_err.fill(-99.999)

    col_index = np.where(np.logical_and(np.greater(blue_phot,0.0),np.greater(red_phot,0.0)))

    col_data[col_index] = blue_phot[col_index] - red_phot[col_index]

    col_data_err[col_index] = np.sqrt( (blue_phot_err[col_index]*blue_phot_err[col_index])  + \
                                        (red_phot_err[col_index]*red_phot_err[col_index]) )

    return col_data, col_data_err

def calc_colour_photometry(config, xmatch, log):

    (gimag, gimerr) = calc_colour_data(xmatch.stars['cal_g_mag_'+config['reference_dataset_code']],
                                        xmatch.stars['cal_g_magerr_'+config['reference_dataset_code']],
                                        xmatch.stars['cal_i_mag_'+config['reference_dataset_code']],
                                        xmatch.stars['cal_i_magerr_'+config['reference_dataset_code']])
    xmatch.stars.add_column(Column(name='(g-i)', data=gimag))
    xmatch.stars.add_column(Column(name='(g-i)_error', data=gimerr))

    (rimag, rimerr) = calc_colour_data(xmatch.stars['cal_r_mag_'+config['reference_dataset_code']],
                                        xmatch.stars['cal_r_magerr_'+config['reference_dataset_code']],
                                        xmatch.stars['cal_i_mag_'+config['reference_dataset_code']],
                                        xmatch.stars['cal_i_magerr_'+config['reference_dataset_code']])
    xmatch.stars.add_column(Column(name='(r-i)', data=rimag))
    xmatch.stars.add_column(Column(name='(r-i)_error', data=rimerr))

    (grmag, grmerr) = calc_colour_data(xmatch.stars['cal_g_mag_'+config['reference_dataset_code']],
                                        xmatch.stars['cal_g_magerr_'+config['reference_dataset_code']],
                                        xmatch.stars['cal_r_mag_'+config['reference_dataset_code']],
                                        xmatch.stars['cal_r_magerr_'+config['reference_dataset_code']])
    xmatch.stars.add_column(Column(name='(g-r)', data=grmag))
    xmatch.stars.add_column(Column(name='(g-r)_error', data=grmerr))

    log.info('Computed colour information for all stars with valid measurements for reference dataset '+config['reference_dataset_code'])

    return xmatch

def output_photometry(config, xmatch, selected_stars, log):

    if str(config['photometry_data_file']).lower() != 'none':

        log.info('Outputting multiband photometry to file')

        f = open(path.join(config['output_dir'],config['photometry_data_file']), 'w')
        f.write('# All measured floating point quantities in units of magnitude\n')
        f.write('# Selected indicates whether a star lies within the selection radius of a given location, if any.  1=true, 0=false\n')
        f.write('# Field_ID   ra_deg   dec_deg   g  sigma_g    r  sigma_r    i  sigma_i   (g-i)  sigma(g-i) (g-r)  sigma(g-r)  (r-i) sigma(r-i)  Selected  Gaia_ID parallax parallax_error proper_motion\n')

        for j,star in enumerate(xmatch.stars):
            if j in selected_stars:
                selected = 1
            else:
                selected = 0
            f.write( str(star['field_id'])+' '+\
                        str(star['ra'])+' '+str(star['dec'])+' '+\
                        str(star['cal_g_mag_'+config['reference_dataset_code']])+' '+str(star['cal_g_magerr_'+config['reference_dataset_code']])+' '+\
                        str(star['cal_r_mag_'+config['reference_dataset_code']])+' '+str(star['cal_r_magerr_'+config['reference_dataset_code']])+' '+\
                        str(star['cal_i_mag_'+config['reference_dataset_code']])+' '+str(star['cal_i_magerr_'+config['reference_dataset_code']])+' '+\
                        str(star['(g-i)'])+' '+str(star['(g-i)_error'])+' '+\
                        str(star['(g-r)'])+' '+str(star['(g-r)_error'])+' '+\
                        str(star['(r-i)'])+' '+str(star['(r-i)_error'])+' '+\
                        str(selected)+' '+str(star['gaia_source_id'])+' '+\
                        str(star['parallax'])+' '+str(star['parallax_error'])+' '+\
                        str(star['pm'])+'\n' )

        f.close()

        log.info('Completed output of multiband photometry')


def apply_star_selection(config, xmatch, log):

    log.info('Applying star selection criteria:')

    # Initialize selection array of star array indices to include all stars:
    selected_stars = np.arange(0,len(xmatch.stars),1, dtype='int')

    # Select stars by quality criteria:
    qc_idx = select_by_photometry_quality(xmatch, config, log)

    # Initialize the combined selection index based on those stars which
    # meet the quality criteria
    idx = copy.copy(qc_idx)

    # Select stars by spacial cut:
    if config['selection_radius'] > 0.0:
        spacial_idx = select_by_position(xmatch, config, log)
        idx = list(set(idx).intersection(set(spacial_idx)))

    # Select stars by parallax:
    if config['parallax_min'] > -99.0 and config['parallax_max'] < 99.0:
        parallax_idx = select_by_parallax(xmatch, config, log)
        idx = list(set(idx).intersection(set(parallax_idx)))

    # Select stars by proper motion:
    if config['pm_min'] > -99.0 and config['pm_max'] < 99.0:
        pm_idx = select_by_parallax(xmatch, config, log)
        idx = list(set(idx).intersection(set(pm_idx)))

    if len(idx) == 0:
        raise ValueError('All stars excluded by combined selection criteria')

    selected_stars = selected_stars[idx]

    log.info('Total number of stars selected: '+str(len(selected_stars)))

    return qc_idx, selected_stars

def select_by_photometry_quality(xmatch, config, log):
    gcol = 'cal_g_mag_'+config['reference_dataset_code']
    gerrcol = 'cal_g_magerr_'+config['reference_dataset_code']
    rcol = 'cal_r_mag_'+config['reference_dataset_code']
    rerrcol = 'cal_r_magerr_'+config['reference_dataset_code']
    icol = 'cal_i_mag_'+config['reference_dataset_code']
    ierrcol = 'cal_i_magerr_'+config['reference_dataset_code']

    qc_idx1 = np.where(np.logical_and(xmatch.stars[gcol] > 0.0,
                                      xmatch.stars[gerrcol] <= config['g_sigma_max']))[0]
    print(xmatch.stars[gcol][qc_idx1], xmatch.stars[gerrcol][qc_idx1])
    qc_idx2 = np.where(np.logical_and(xmatch.stars[rcol] > 0.0,
                                      xmatch.stars[rerrcol] <= config['r_sigma_max']))[0]
    qc_idx3 = np.where(np.logical_and(xmatch.stars[icol] >  0.0,
                                      xmatch.stars[ierrcol] <= config['i_sigma_max']))[0]

    # qc_idx1 = np.where( np.logical_and( np.less_equal(xmatch.stars['cal_g_magerr_'+config['reference_dataset_code']], config['g_sigma_max']),
    #                     np.less_equal(xmatch.stars['cal_r_magerr_'+config['reference_dataset_code']], config['r_sigma_max']) ) )[0]
    # qc_idx2 = np.where( np.logical_and( np.less_equal(xmatch.stars['cal_i_magerr_'+config['reference_dataset_code']], config['i_sigma_max']),
    #                     np.less_equal(xmatch.stars['(g-i)_error'], config['gi_sigma_max']) ) )[0]
    # qc_idx3 = np.where( np.logical_and( np.less_equal(xmatch.stars['(r-i)_error'], config['ri_sigma_max']),
    #                     np.less_equal(xmatch.stars['(g-r)_error'], config['gr_sigma_max']) ) )[0]
    qc_idx = set(qc_idx1).intersection(set(qc_idx2))
    qc_idx = np.array(list(qc_idx.intersection(set(qc_idx3))))

    log.info(' -> '+str(len(qc_idx))+' stars meet the quality selection criteria:')
    log.info('    Max phot uncertainty, g = '+str(config['g_sigma_max']))
    log.info('    Max phot uncertainty, r = '+str(config['r_sigma_max']))
    log.info('    Max phot uncertainty, i = '+str(config['i_sigma_max']))
    log.info('    Max phot uncertainty, (g-i) = '+str(config['gi_sigma_max']))
    log.info('    Max phot uncertainty, (r-i) = '+str(config['ri_sigma_max']))
    log.info('    Max phot uncertainty, (g-r) = '+str(config['gr_sigma_max']))

    if len(qc_idx) == 0:
        raise ValueError('All stars excluded by quality control criteria')

    return qc_idx

def select_by_position(xmatch, config, log):
    stars = SkyCoord(xmatch.stars['ra'], xmatch.stars['dec'],
                    frame='icrs', unit=(u.deg, u.deg))
    target = SkyCoord(config['target_ra'], config['target_dec'],
                    frame='icrs', unit=(u.hourangle, u.deg))
    separations = target.separation(stars)

    spacial_idx = np.where(separations < Angle((config['selection_radius']/60.0), unit=u.deg))[0]
    log.info(' -> '+str(len(spacial_idx))+' stars meet the spacial selection critera:')
    log.info('    Within '+str(config['selection_radius'])+'arcmin of '+config['target_ra']+', '+config['target_dec'])
    log.info('    i.e. within '+str(config['selection_radius'])+'arcmin of '+str(target.ra.deg)+' deg, '+str(target.dec.deg))

    if len(spacial_idx) == 0:
        raise ValueError('All stars excluded by spacial selection criteria')

    return spacial_idx

def select_by_parallax(xmatch, config, log):

    idx1 = np.where(xmatch.stars['gaia_source_id'] != '0.0')[0]
    idx2 = np.where(np.logical_and( np.greater_equal(xmatch.stars['parallax'], config['parallax_min']),
                        np.less_equal(xmatch.stars['parallax'], config['parallax_max']) ) )[0]
    parallax_idx = np.array(list(set(idx1).intersection(set(idx2))))

    log.info(' -> '+str(len(parallax_idx))+' stars have meet the parallax selection critera:')
    log.info('    '+str(len(idx1))+' stars have Gaia measurements')
    log.info('    Between '+str(config['parallax_min'])+' and '+str(config['parallax_max']))

    if len(parallax_idx) == 0:
        raise ValueError('All stars excluded by parallax selection criteria')

    return parallax_idx

def select_by_proper_motion(xmatch, config, log):

    idx1 = np.where(xmatch.stars['gaia_source_id'] != '0.0')[0]
    idx2 = np.where(np.logical_and( np.greater_equal(xmatch.stars['proper_motion'], config['pm_min']),
                        np.less_equal(xmatch.stars['proper_motion'], config['pm_max']) ) )[0]
    pm_idx = np.array(list(set(idx1).intersection(set(idx2))))

    log.info(' -> '+str(len(pm_idx))+' stars have meet the proper motion selection critera:')
    log.info('    '+str(len(idx1))+' stars have Gaia measurements')
    log.info('    Between '+str(config['pm_min'])+' and '+str(config['pm_max']))

    if len(pm_idx) == 0:
        raise ValueError('All stars excluded by spacial selection criteria')

    return pm_idx

def get_args():

    if len(argv) == 1:
        config_file = input('Please enter the path to the configuration file: ')
    else:
        config_file = argv[1]

    config = config_utils.build_config_from_json(config_file)

    for key in ['plot_selected_stars_only', 'interactive', 'legend']:
        if 'false' in str(config[key]).lower():
            config[key] = False
        else:
            config[key] = True

    for key in ['selection_radius', 'parallax_min', 'parallax_max', 'pm_min', 'pm_max']:
        config[key] = float(config[key])

    return config


if __name__ == '__main__':
    field_colour_analysis()
