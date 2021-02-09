from sys import path
from os import argv
from pyDANDIA import analyse_cmd
from pyDANDIA import config_utils
from pyDANDIA import  logs
from pyDANDIA import  crossmatch
from astropy.coordinates import SkyCoord
from astropy import units as u

def plot_field_colour_mag_diagram():

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
    if not params['plot_selected_radius_only']:
        plt.scatter(xmatch.stars[colour][valid_stars],xmatch.stars[mag_column][valid_stars],
                 c=marker_colour, marker='.', s=1,
                 label='Stars within field of view')

    plt.scatter(xmatch.stars[col_key][selected_stars],xmatch.stars[mag_column][selected_stars],
              c=default_marker_colour, marker='*', s=5,
              label='Stars meeting selection criteria')


    plt.xlabel('SDSS '+colour+' [mag]')

    plt.ylabel('SDSS-'+magnitude+' [mag]')

    [xmin,xmax,ymin,ymax] = plt.axis()
    col_key = colour.replace('(','').replace(')','').replace('-','')
    xmin = params['plot_'+col_key+'_range'][0]
    xmax = params['plot_'+col_key+'_range'][1]
    ymin = params['plot_'+magnitude+'_range'][0]
    ymax = params['plot_'+magnitude+'_range'][1]
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
                                            magnitude+'_vs_'+colour\
                                            +'.pdf')

    plt.grid()

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

    if params['interactive']:
        plt.show()

    plt.savefig(plot_file,bbox_inches='tight')

    plt.close(1)

    log.info('Colour-magnitude diagram output to '+plot_file)

def plot_data_colours():
    default_marker_colour = '#8c6931'
    field_marker_colour = '#E1AE13'
    marker_colour = default_marker_colour
    return default_marker_colour, field_marker_colour, marker_colour

def calc_colour_data(blue_phot, blue_phot_err, red_phot, red_phot_err):

    col_data = np.zeros(len(red_phot))
    col_data.fill(-99.999)
    col_data_err = np.zeros(len(red_phot))
    col_data_err.fill(-99.999)

    col_index = (blue_phot > 0.0 && red_phot > 0.0)

    col_data[col_index] = blue_phot[col_index] - red_phot[col_index]

    col_data_err[col_index] = np.sqrt( (blue_phot_err[col_index]*blue_phot_err[col_index])  + \
                                        (red_phot_err[col_index]*red_phot_err[col_index]) )

    return col_data, col_data_err

def calc_colour_photometry(config, xmatch, log):

    (gimag, gimerr) = calc_colour_data(xmatch.stars['cal_g_mag_'+config['reference_dataset_code']],
                                        xmatch.stars['cal_g_magerr_'+config['reference_dataset_code']],
                                        xmatch.stars['cal_i_mag_'+config['reference_dataset_code']],
                                        xmatch.stars['cal_i_magerr_'+config['reference_dataset_code']])

    xmatch.stars.add_column(gimag, name='(g-i)')
    xmatch.stars.add_column(gimerr, name='(g-i)_error')

    (rimag, rimerr) = calc_colour_data(xmatch.stars['cal_r_mag_'+config['reference_dataset_code']],
                                        xmatch.stars['cal_r_magerr_'+config['reference_dataset_code']],
                                        xmatch.stars['cal_i_mag_'+config['reference_dataset_code']],
                                        xmatch.stars['cal_i_magerr_'+config['reference_dataset_code']])

    xmatch.stars.add_column(rimag, name='(r-i)')
    xmatch.stars.add_column(rimerr, name='(r-i)_error')

    (grmag, grmerr) = calc_colour_data(xmatch.stars['cal_g_mag_'+config['reference_dataset_code']],
                                        xmatch.stars['cal_g_magerr_'+config['reference_dataset_code']],
                                        xmatch.stars['cal_r_mag_'+config['reference_dataset_code']],
                                        xmatch.stars['cal_r_magerr_'+config['reference_dataset_code']])

    xmatch.stars.add_column(rimag, name='(g-r)')
    xmatch.stars.add_column(rimerr, name='(g-r)_error')

    log.info('Computed colour information for all stars with valid measurements for reference dataset '+config['reference_dataset_code'])

    return xmatch

def output_photometry(config, xmatch, selected_stars, log):

    if str(config['photometry_data_file']).lower() != 'none':

        log.info('Outputting multiband photometry to file')

        f = open(path.join(config['output_dir'],config['photometry_data_file']), 'w')
        f.write('# All measured floating point quantities in units of magnitude\n')
        f.write('# Selected indicates whether a star lies within the selection radius of a given location, if any.  1=true, 0=false\n')
        f.write('# Star   x_pix    y_pix   ra_deg   dec_deg   g  sigma_g    r  sigma_r    i  sigma_i   (g-i)  sigma(g-i) (g-r)  sigma(g-r)  (r-i) sigma(r-i)  Selected  Gaia_ID\n')

        for star in xmatch.stars:
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
                        str(selected)+' '+str(int(star['gaia_source_id']))+'\n' )

        f.close()

        log.info('Completed output of multiband photometry')


def apply_star_selection(config, xmatch, log):

    log.info('Applying star selection criteria:')

    # Initialize selection array of star array indices to include all stars:
    selected_stars = np.arange(0,len(xmatch.stars),1, dtype='int')

    # Select stars by quality criteria:
    qc_idx = np.where(xmatch.stars['cal_g_magerr_'+config['reference_dataset_code']] <= float(config['g_sigma_max']) && \
                        xmatch.stars['cal_r_magerr_'+config['reference_dataset_code']] <= float(config['r_sigma_max']) && \
                        xmatch.stars['cal_i_magerr_'+config['reference_dataset_code']] <= float(config['i_sigma_max']) && \
                        xmatch.stars['(g-i)_error'] <= float(config['gi_sigma_max']) && \
                        xmatch.stars['(r-i)_error'] <= float(config['ri_sigma_max']) && \
                        xmatch.stars['(g-r)_error'] <= float(config['gr_sigma_max']))[0]
    log.info(' -> '+str(len(qc_idx))+' stars meet the quality selection criteria:')
    log.info('    Max phot uncertainty, g = '+config['g_sigma_max'])
    log.info('    Max phot uncertainty, r = '+config['r_sigma_max'])
    log.info('    Max phot uncertainty, i = '+config['i_sigma_max'])
    log.info('    Max phot uncertainty, (g-i) = '+config['gi_sigma_max'])
    log.info('    Max phot uncertainty, (r-i) = '+config['ri_sigma_max'])
    log.info('    Max phot uncertainty, (g-r) = '+config['gr_sigma_max'])

    # Select stars by spacial cut:
    if float(config['selection_radius']) == 0.0:
        stars = SkyCoord(xmatch.stars['ra'], xmatch.stars['dec'],
                        frame='icrs', unit=(u.deg, u.deg))
        target = SkyCoord(config['target_ra'], config['target_dec'],
                        frame='icrs', unit(u.hourangle, u.deg))
        separations = target.separation(stars)

        spacial_idx = np.where(separations < float(config['selection_radius'])/60.0)[0]
        log.info(' -> '+str(len(spacial_idx))+' stars meet the spacial selection critera:')
        log.info('    Within '+config['selection_radius']+'arcmin of '+config['target_ra']+', '+config['target_dec'])

    # Combine selection criteria to return the selected stars index:
    idx = list(set(qc_idx).intersection(set(spacial_idx)))
    selected_stars = selected_stars[idx]

    log.info('Total number of stars selected: '+str(len(selected_stars)))

    return qc_idx, selected_stars

def get_args():

    if len(argv) == 1:
        config_file = input('Please enter the path to the configuration file: ')
    else:
        config_file = argv[1]

    config = config_utils.build_config_from_json(config_file)


if __name__ == '__main__':
