# -*- coding: utf-8 -*-
"""
Created on Tue May 21 20:49:49 2019

@author: rstreet
"""

from sys import argv
import sqlite3
from os import getcwd, path, remove, environ
import numpy as np
from astropy import table
from astropy.coordinates import SkyCoord
from astropy import units
import matplotlib.pyplot as plt
from pyDANDIA import  photometry_classes
from pyDANDIA import  phot_db
from pyDANDIA import  logs
from pyDANDIA import  event_colour_analysis
from pyDANDIA import  spectral_type_data
from pyDANDIA import  red_clump_utilities

def plot_cmd(params):
    """Function to plot a colour-magnitude diagram from a field phot_db"""

    log = logs.start_stage_log( params['red_dir'], 'plot_cmd' )

    conn = phot_db.get_connection(dsn=params['db_file_path'])

    (photometry, stars) = extract_reference_instrument_calibrated_photometry(conn,log)

    photometry = calculate_colours(photometry,stars,log)

    RC = localize_red_clump_db(photometry,stars,log)

    RC = event_colour_analysis.measure_RC_offset(params,RC,log)

    plot_colour_mag_diagram(params, photometry, stars, RC, 'r', 'i', 'i', log)
    plot_colour_mag_diagram(params, photometry, stars, RC, 'r', 'i', 'r', log)
    plot_colour_mag_diagram(params, photometry, stars, RC, 'g', 'r', 'g', log)
    plot_colour_mag_diagram(params, photometry, stars, RC, 'g', 'i', 'g', log)

    plot_colour_colour_diagram(params, photometry, RC, log)

    conn.close()

    logs.close_log(log)

def get_args():

    params = {}

    if len(argv) < 5:

        params['db_file_path'] = input('Please enter the path to the photometry database for the field: ')
        params['red_dir'] = input('Please enter the directory path for output: ')
        params['target_ra'] = input('Please enter the RA of the field centre [sexigesimal]: ')
        params['target_dec'] = input('Please enter the Dec of the field centre [sexigesimal]: ')

    else:

        params['db_file_path'] = argv[1]
        params['red_dir'] = argv[2]
        params['target_ra'] = argv[3]
        params['target_dec'] = argv[4]

    return params

def extract_reference_instrument_calibrated_photometry(conn,log):
    """Function to extract from the phot_db the calibrated photometry for stars
    in the field from the data from the photometric reference instrument.
    By default this is defined to be lsc.doma.1m0a.fa15.
    """
    def fetch_star_phot(star_id,phot_table):
        jdx = np.where(phot_table['star_id'] == star_id)[0]
        if len(jdx) == 0:
            return 0.0,0.0
        else:
            return phot_table['calibrated_mag'][jdx],phot_table['calibrated_mag_err'][jdx]

    facility_code = phot_db.get_facility_code({'site': 'lsc',
                                               'enclosure': 'doma',
                                               'telescope': '1m0a',
                                               'instrument': 'fa15'})

    query = 'SELECT facility_id, facility_code FROM facilities WHERE facility_code="'+facility_code+'"'
    t = phot_db.query_to_astropy_table(conn, query, args=())
    if len(t) == 0:
        raise IOError('No photometry for primary reference facility '+facility_code+' found in phot_db')
        
    facility_id = t['facility_id'][0]

    stars = phot_db.fetch_stars_table(conn)

    filters = {'g': None, 'r': None, 'i': None}
    refimgs = {'g': None, 'r': None, 'i': None}
    for f in filters.keys():

        filter_name = f+'p'

        query = 'SELECT filter_id, filter_name FROM filters WHERE filter_name="'+filter_name+'"'
        t = phot_db.query_to_astropy_table(conn, query, args=())
        filters[f] = t['filter_id'][0]

        query = 'SELECT refimg_id FROM reference_images WHERE facility="'+str(facility_id)+'" AND filter="'+str(filters[f])+'"'
        t = phot_db.query_to_astropy_table(conn, query, args=())
        if len(t) > 0:
            refimgs[f] = t['refimg_id'][0]
        else:
            raise IOError('No reference image data in DB for facility '+str(facility_code)+'.')

    photometry = {'phot_table_g': [], 'phot_table_r': [], 'phot_table_i': []}

    log.info('-> Extracting photometry for stars')


    for j,star in enumerate(stars):
        for f,fid in filters.items():
            query = 'SELECT star_id, calibrated_mag, calibrated_mag_err FROM phot WHERE star_id="'+str(star['star_id'])+\
                            '" AND facility="'+str(facility_id)+\
                            '" AND filter="'+str(fid)+\
                            '" AND software="1'+\
                            '" AND reference_image="'+str(refimgs[f])+\
                            '" AND phot_type="PSF_FITTING"'

            data = phot_db.query_to_astropy_table(conn, query, args=())

            if len(data) > 0:
                photometry['phot_table_'+f].append([data['star_id'][0], data['calibrated_mag'][0], data['calibrated_mag_err'][0]])
            else:
                photometry['phot_table_'+f].append([star['star_id'],0.0,0.0])

        if j%1000.0 == 0.0:
            print('--> Completed '+str(j)+' stars out of '+str(len(stars)))

    for f,fid in filters.items():
        table_data = np.array(photometry['phot_table_'+f])

        data = [table.Column(name='star_id',
                        data=table_data[:,0]),
                table.Column(name='calibrated_mag',
                        data=table_data[:,1]),
                table.Column(name='calibrated_mag_err',
                        data=table_data[:,2])]
        photometry['phot_table_'+f] = table.Table(data=data)

    log.info('Extracted photometry for '+str(len(stars))+' stars')

    photometry['g'] = np.zeros(len(stars))
    photometry['gerr'] = np.zeros(len(stars))
    photometry['r'] = np.zeros(len(stars))
    photometry['rerr'] = np.zeros(len(stars))
    photometry['i'] = np.zeros(len(stars))
    photometry['ierr'] = np.zeros(len(stars))

    for s in range(0,len(stars),1):

        sid = stars['star_id'][s]

        (photometry['g'][s],photometry['gerr'][s]) = fetch_star_phot(sid,photometry['phot_table_g'])
        (photometry['r'][s],photometry['rerr'][s]) = fetch_star_phot(sid,photometry['phot_table_r'])
        (photometry['i'][s],photometry['ierr'][s]) = fetch_star_phot(sid,photometry['phot_table_i'])

    return photometry, stars

def calculate_colours(photometry,stars,log):

    def calc_colour_data(blue_index, red_index, blue_phot, blue_phot_err,
                                                red_phot, red_phot_err):

        col_index = list(set(blue_index).intersection(set(red_index)))

        col_data = np.zeros(len(red_phot))
        col_data.fill(-99.999)
        col_data_err = np.zeros(len(red_phot))
        col_data_err.fill(-99.999)

        col_data[col_index] = blue_phot[col_index] - red_phot[col_index]

        col_data_err[col_index] = np.sqrt( (blue_phot_err[col_index]*blue_phot_err[col_index])  + \
                                            (red_phot_err[col_index]*red_phot_err[col_index]) )

        return col_data, col_data_err

    gdx = np.where(photometry['g'] != 0.0)[0]
    rdx = np.where(photometry['r'] != 0.0)[0]
    idx = np.where(photometry['i'] != 0.0)[0]

    (photometry['gr'],photometry['gr_err']) = calc_colour_data(gdx, rdx,
                                         photometry['g'], photometry['gerr'],
                                         photometry['r'], photometry['rerr'])
    (photometry['gi'],photometry['gi_err']) = calc_colour_data(gdx, idx,
                                         photometry['g'], photometry['gerr'],
                                         photometry['i'], photometry['ierr'])
    (photometry['ri'],photometry['ri_err']) = calc_colour_data(rdx, idx,
                                         photometry['r'], photometry['rerr'],
                                         photometry['i'], photometry['ierr'])

    log.info('Calculated colour data for stars detected in ROME data')

    gdx = np.where(stars['vphas_gmag'] != 0.0)[0]
    rdx = np.where(stars['vphas_rmag'] != 0.0)[0]
    idx = np.where(stars['vphas_imag'] != 0.0)[0]

    photometry['gr_cat'] = calc_colour_data(gdx, rdx,
                                         stars['vphas_gmag'], stars['vphas_gmag_error'],
                                         stars['vphas_rmag'], stars['vphas_rmag_error'])
    photometry['gi_cat'] = calc_colour_data(gdx, idx,
                                         stars['vphas_gmag'], stars['vphas_gmag_error'],
                                         stars['vphas_imag'], stars['vphas_imag_error'])
    photometry['ri_cat'] = calc_colour_data(rdx, idx,
                                         stars['vphas_rmag'], stars['vphas_rmag_error'],
                                         stars['vphas_imag'], stars['vphas_imag_error'])

    log.info('Calculated VPHAS catalog colours for all stars identified within field')

    return photometry

def plot_colour_mag_diagram(params, photometry, stars, RC, blue_filter, red_filter,
                            yaxis_filter, log):
    """Function to plot a colour-magnitude diagram, highlighting the data for
    local stars close to the target in a different colour from the rest,
    and indicating the position of both the target and the Red Clump centroid.
    """

    add_rc_centroid = True

    col_key = blue_filter+red_filter

    fig = plt.figure(1,(10,10))

    ax = plt.subplot(111)

    plt.rcParams.update({'font.size': 18})

    cdx = np.where(photometry[col_key] != -99.999)[0]
    mdx = np.where(photometry[yaxis_filter] != 0.0)[0]
    jdx = list(set(cdx).intersection(set(mdx)))

    plt.scatter(photometry[col_key][jdx],photometry[yaxis_filter][jdx],
                 c='#8c6931', marker='.', s=1,
                 label='Stars within ROME field')


    if add_rc_centroid:
        plt.errorbar(getattr(RC,col_key), getattr(RC,yaxis_filter),
                 yerr=getattr(RC,'sig_'+yaxis_filter),
                 xerr=getattr(RC,'sig_'+col_key),
                 color='g', marker='s',markersize=10, label='Red Clump centroid')

    plt.xlabel('SDSS ('+blue_filter+'-'+red_filter+') [mag]')

    plt.ylabel('SDSS-'+yaxis_filter+' [mag]')

    [xmin,xmax,ymin,ymax] = plt.axis()

    plt.axis([xmin,xmax,ymax,ymin])

    xticks = np.arange(xmin,xmax,0.1)
    yticks = np.arange(ymin,ymax,0.2)

    ax.set_xticks(xticks,minor=True)
    ax.set_yticks(yticks,minor=True)

    plot_file = path.join(params['red_dir'],'colour_magnitude_diagram_'+\
                                            yaxis_filter+'_vs_'+blue_filter+red_filter\
                                            +'.pdf')

    plt.grid()

    if red_filter == 'i' and blue_filter == 'r' and yaxis_filter == 'i':
        plt.axis([0.5,2.0,20.2,13.5])

    if red_filter == 'i' and blue_filter == 'r' and yaxis_filter == 'r':
        plt.axis([0.0,1.5,21.0,13.5])

    if red_filter == 'r' and blue_filter == 'g':
        plt.axis([0.5,3.0,22.0,14.0])

    if red_filter == 'i' and blue_filter == 'g':
        plt.axis([0.5,4.4,22.0,14.0])

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * -0.025,
                 box.width, box.height * 0.95])

    l = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2)

    l.legendHandles[0]._sizes = [50]
    l.legendHandles[1]._sizes = [50]

    plt.rcParams.update({'legend.fontsize':18})
    plt.rcParams.update({'font.size':18})
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)

    plt.savefig(plot_file,bbox_inches='tight')

    plt.close(1)

    log.info('Colour-magnitude diagram output to '+plot_file)

def plot_colour_colour_diagram(params,photometry,RC,log):
    """Function to plot a colour-colour diagram, if sufficient data are
    available within the given star catalog"""

    filters = { 'i': 'SDSS-i', 'r': 'SDSS-r', 'g': 'SDSS-g' }

    fig = plt.figure(1,(10,10))

    ax = plt.axes()

    grx = np.where(photometry['gr'] != -99.999)[0]
    rix = np.where(photometry['ri'] != -99.999)[0]
    jdx = list(set(grx).intersection(set(rix)))

    inst_gr = photometry['gr'][jdx] - RC.Egr
    inst_ri = photometry['ri'][jdx] - RC.Eri

    ax.scatter(inst_gr, inst_ri,
               c='#8c6931', marker='.', s=1,
             label='Stars within ROME field')

    (spectral_type, luminosity_class, gr_colour, ri_colour) = spectral_type_data.get_spectral_class_data()

    plot_dwarfs = False
    plot_giants = True
    for i in range(0,len(spectral_type),1):

        spt = spectral_type[i]+luminosity_class[i]

        if luminosity_class[i] == 'V':
            c = '#8d929b'
        else:
            c = '#8d929b'

        if luminosity_class[i] == 'III' and plot_giants:

            plt.plot(gr_colour[i], ri_colour[i], marker='s', color=c,
                     markeredgecolor='k', alpha=0.5)

            plt.annotate(spt, (gr_colour[i], ri_colour[i]-0.1),
                            color='k', size=10, rotation=-30.0, alpha=1.0)

        if luminosity_class[i] == 'V' and plot_dwarfs:

            plt.plot(gr_colour[i], ri_colour[i], marker='s', color=c,
                     markeredgecolor='k', alpha=0.5)

            plt.annotate(spt, (gr_colour[i],
                           ri_colour[i]+0.1),
                             color='k', size=10,
                             rotation=-30.0, alpha=1.0)

    plt.xlabel('SDSS (g-r) [mag]')

    plt.ylabel('SDSS (r-i) [mag]')

    plot_file = path.join(params['red_dir'],'colour_colour_diagram.pdf')

    plt.axis([-1.0,2.0,-1.0,1.0])

    plt.grid()

    xticks = np.arange(-1.0,2.0,0.1)
    yticks = np.arange(-1.0,1.0,0.1)

    ax.set_xticks(xticks, minor=True)
    ax.set_yticks(yticks, minor=True)


    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * -0.025,
                 box.width, box.height * 0.95])

    l = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2)

    try:
        l.legendHandles[2]._sizes = [50]
        l.legendHandles[3]._sizes = [50]
    except IndexError:
        pass

    plt.rcParams.update({'legend.fontsize':18})
    plt.rcParams.update({'font.size':18})
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)

    plt.savefig(plot_file,bbox_inches='tight')

    plt.close(1)

    log.info('Colour-colour diagram output to '+plot_file)

def localize_red_clump_db(photometry,stars,log):
    """Function to calculate the centroid of the Red Clump stars in a
    colour-magnitude diagram"""

    def select_within_range(mags, colours, mag_min, mag_max, col_min, col_max):
        """Function to identify the set of array indices with values
        between the range indicated"""

        idx1 = np.where(colours >= col_min)[0]
        idx2 = np.where(colours <= col_max)[0]
        idx3 = np.where(mags >= mag_min)[0]
        idx4 = np.where(mags <= mag_max)[0]
        idx = set(idx1).intersection(set(idx2))
        idx = idx.intersection(set(idx3))
        idx = list(idx.intersection(set(idx4)))

        return idx

    RC = photometry_classes.Star()

    log.info('Localizing the Red Clump')

    ri_min = 0.8
    ri_max = 1.2
    i_min = 15.5
    i_max = 16.5

    r_min = 16.2
    r_max = 17.5

    gi_min = 2.5
    gi_max = 3.5

    gr_min = 1.5
    gr_max = 2.2
    g_min = 17.8
    g_max = 19.5

    log.info('Selected Red Clump giants between:')
    log.info('i = '+str(i_min)+' to '+str(i_max))
    log.info('r = '+str(r_min)+' to '+str(r_max))
    log.info('(r-i) = '+str(ri_min)+' to '+str(ri_max))
    log.info('g = '+str(g_min)+' to '+str(g_max))
    log.info('(g-r) = '+str(gr_min)+' to '+str(gr_max))
    log.info('(g-i) = '+str(gi_min)+' to '+str(gi_max))

    idx = select_within_range(photometry['i'], photometry['ri'], i_min, i_max, ri_min, ri_max)

    (RC.ri, RC.sig_ri, RC.i, RC.sig_i) = event_colour_analysis.calc_distribution_centroid_and_spread_2d(photometry['ri'][idx], photometry['i'][idx], use_iqr=True)

    idx = select_within_range(photometry['r'], photometry['ri'], r_min, r_max, ri_min, ri_max)

    (RC.r, RC.sig_r) = event_colour_analysis.calc_distribution_centre_and_spread(photometry['r'][idx], use_iqr=True)

    idx = select_within_range(photometry['g'], photometry['gr'], g_min, g_max, gr_min, gr_max)

    (RC.gr, RC.sig_gr, RC.g, RC.sig_g) = event_colour_analysis.calc_distribution_centroid_and_spread_2d(photometry['gr'][idx], photometry['g'][idx], use_iqr=True)

    idx = select_within_range(photometry['g'], photometry['gi'], g_min, g_max, gi_min, gi_max)

    (RC.gi, RC.sig_gi, RC.g, RC.sig_g) = event_colour_analysis.calc_distribution_centroid_and_spread_2d(photometry['gi'][idx], photometry['g'][idx], use_iqr=True)

    log.info('\n')
    log.info('Centroid of Red Clump Stars at:')
    log.info(RC.summary(show_mags=True))
    log.info(RC.summary(show_mags=False,show_colours=True))

    RC.transform_to_JohnsonCousins()

    log.info(RC.summary(show_mags=False,johnsons=True))

    return RC

if __name__ == '__main__':

    params = get_args()
    plot_cmd(params)
