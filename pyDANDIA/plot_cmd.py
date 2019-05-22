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

def plot_cmd(params):
    """Function to plot a colour-magnitude diagram from a field phot_db"""
    
    log = logs.start_stage_log( params['red_dir'], 'plot_cmd' )
    
    conn = phot_db.get_connection(dsn=params['db_file_path'])

    (photometry, stars) = extract_reference_instrument_calibrated_photometry(conn,log)
    
    photometry = calculate_colours(photometry,stars,log)

    RC = localize_red_clump_db(photometry,stars,log)
    
    plot_colour_mag_diagram(params, photometry, stars, RC, 'r', 'i', 'i', log)
    plot_colour_mag_diagram(params, photometry, stars, RC, 'r', 'i', 'r', log)
    plot_colour_mag_diagram(params, photometry, stars, RC, 'g', 'r', 'g', log)
    plot_colour_mag_diagram(params, photometry, stars, RC, 'g', 'i', 'g', log)
    
    conn.close()
    
    logs.close_log(log)
    
def get_args():
    
    params = {}
    
    if len(argv) == 2:
        
        params['db_file_path'] = input('Please enter the path to the photometry database for the field: ')
        params['red_dir'] = input('Please enter the directory path for output: ')
        
    else:
        
        params['db_file_path'] = argv[1]
        params['red_dir'] = argv[2]
    
    return params
    
def extract_reference_instrument_calibrated_photometry(conn,log):
    """Function to extract from the phot_db the calibrated photometry for stars
    in the field from the data from the photometric reference instrument.
    By default this is defined to be lsc.doma.1m0a.fa15.
    """
    def fetch_star_phot(star_id,phot_table):
        jdx = np.where(phot_table['star_id'] == star_id)[0]
        if len(jdx) == 0:
            return 0.0
        else:
            return phot_table['calibrated_mag'][jdx]
        
    facility_code = phot_db.get_facility_code({'site': 'lsc', 
                                               'enclosure': 'doma', 
                                               'telescope': '1m0a', 
                                               'instrument': 'fa15'})
                                                   
    query = 'SELECT facility_id, facility_code FROM facilities WHERE facility_code="'+facility_code+'"'
    t = phot_db.query_to_astropy_table(conn, query, args=())
    facility_id = t['facility_id'][0]
    
    filters = {'g': None, 'r': None, 'i': None}
    refimgs = {'g': None, 'r': None, 'i': None}
    photometry = {'phot_table_g': [], 'phot_table_r': [], 'phot_table_i': []}
    
    for f in filters.keys():

        filter_name = f+'p'
        
        query = 'SELECT filter_id, filter_name FROM filters WHERE filter_name="'+filter_name+'"'
        t = phot_db.query_to_astropy_table(conn, query, args=())
        filters[f] = t['filter_id'][0]
        
        query = 'SELECT refimg_id FROM reference_images WHERE facility="'+str(facility_id)+'" AND filter="'+str(filters[f])+'"'
        t = phot_db.query_to_astropy_table(conn, query, args=())
        refimgs[f] = t['refimg_id'][0]
        
        query = 'SELECT star_id, calibrated_mag, calibrated_mag_err FROM phot WHERE reference_image="'+str(refimgs[f])+'"'
        photometry['phot_table_'+f] = phot_db.query_to_astropy_table(conn, query, args=())
        
        query = 'SELECT star_id, vphas_gmag, vphas_rmag, vphas_imag FROM stars'
        stars = phot_db.query_to_astropy_table(conn, query, args=())
    
    log.info('Exracted photometry for '+str(len(stars))+' stars')
    
    photometry['g'] = np.zeros(len(stars))
    photometry['r'] = np.zeros(len(stars))
    photometry['i'] = np.zeros(len(stars))
    
    for s in range(0,len(stars),1):
        
        sid = stars['star_id'][s]
                
        photometry['g'][s] = fetch_star_phot(sid,photometry['phot_table_g'])
        photometry['r'][s] = fetch_star_phot(sid,photometry['phot_table_r'])
        photometry['i'][s] = fetch_star_phot(sid,photometry['phot_table_i'])
    
    return photometry, stars

def calculate_colours(photometry,stars,log):
    
    def calc_colour_data(blue_index, red_index, blue_phot, red_phot):
        
        col_index = list(set(blue_index).intersection(set(red_index)))
        
        col_data = np.zeros(len(red_phot))
        
        col_data[col_index] = blue_phot[col_index] - red_phot[col_index]
        
        return col_data
        
    gdx = np.where(photometry['g'] != 0.0)[0]
    rdx = np.where(photometry['r'] != 0.0)[0]
    idx = np.where(photometry['i'] != 0.0)[0]
    
    photometry['gr'] = calc_colour_data(gdx, rdx, 
                                         photometry['g'], photometry['r'])
    photometry['gi'] = calc_colour_data(gdx, idx, 
                                         photometry['g'], photometry['i'])
    photometry['ri'] = calc_colour_data(rdx, idx, 
                                         photometry['r'], photometry['i'])
    
    log.info('Calculated colour data for stars detected in ROME data')
    
    gdx = np.where(stars['vphas_gmag'] != 0.0)[0]
    rdx = np.where(stars['vphas_rmag'] != 0.0)[0]
    idx = np.where(stars['vphas_imag'] != 0.0)[0]
    
    photometry['gr_cat'] = calc_colour_data(gdx, rdx, 
                                         stars['vphas_gmag'], stars['vphas_rmag'])
    photometry['gi_cat'] = calc_colour_data(gdx, idx, 
                                         stars['vphas_gmag'], stars['vphas_imag'])
    photometry['ri_cat'] = calc_colour_data(rdx, idx, 
                                         stars['vphas_rmag'], stars['vphas_imag'])
    
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
    
    plt.scatter(photometry[col_key],photometry[yaxis_filter],
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
    