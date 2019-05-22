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
from pyDANDIA import  photometry_classes

def plot_cmd(params):
    """Function to plot a colour-magnitude diagram from a field phot_db"""
    
    conn = phot_db.get_connection(dsn=params['db_file_path'])

    (photometry, filters, refimgs) = extract_reference_instrument_calibrated_photometry()
    
    # Calculate colour data
    
    # plot CMDs
    
    conn.close()
    
def get_args():
    
    params = {}
    
    if len(argv) == 1:
        
        params['db_file_path'] = input('Please enter the path to the photometry database for the field: ')
        params['output_dir'] = input('Please enter the directory path for output: ')
        
    else:
        
        params['db_file_path'] = argv[1]
        params['output_dir'] = argv[2]
    
    return params
    
def extract_reference_instrument_calibrated_photometry():
    """Function to extract from the phot_db the calibrated photometry for stars
    in the field from the data from the photometric reference instrument.
    By default this is defined to be lsc.doma.1m0a.fa15.
    """
    
    ref_facility_code = phot_db.get_facility_code({'site': 'lsc', 
                                                   'enclosure': 'doma', 
                                                   'telescope': '1m0a', 
                                                   'instrument': 'fa15'})
                                                   
    query = 'SELECT facility_id, facility_code FROM facilities WHERE facility_code="'+facility_code+'"'
    
    facility = phot_db.query_to_astropy_table(conn, query, args=())

    filters = {'gp': None, 'rp': None, 'ip': None}
    refimgs = {'gp': None, 'rp': None, 'ip': None}
    photometry = {'gp': np.zeros(1), 'rp': np.zeros(1), 'ip': np.zeros(1)}
    
    for f in filters.keys():

        query = 'SELECT filter_id, filter_name FROM filters WHERE filter_name="'+f+'"'
        filters[f] = phot_db.query_to_astropy_table(conn, query, args=())
    
        query = 'SELECT refimg_id FROM reference_images WHERE facility="'+str(facility_id)+'" AND filter="'+str(filters[f]['filter_id'][0])+'"'
        refimgs[f] = query_to_astropy_table(conn, query, args=())
    
        query = 'SELECT calibrated_mag, calibrated_mag_err FROM phot WHERE reference_image="'+str(refimgs[f]['refimg_id'][0])+'"'
        photometry[f] = query_to_astropy_table(conn, query, args=())
    
    return photometry, filters, refimgs
    
def plot_colour_mag_diagram(params, mags, colours, local_mags, local_colours, 
                            target, blue_filter, red_filter, 
                            yaxis_filter, tol, log):
    """Function to plot a colour-magnitude diagram, highlighting the data for 
    local stars close to the target in a different colour from the rest, 
    and indicating the position of both the target and the Red Clump centroid.
    """
    
    def calc_colour_lightcurve(blue_lc, red_lc, y_lc):
        
        idx1 = np.where( red_lc['mag_err'] > 0.0 )[0]
        idx2 = np.where( blue_lc['mag_err'] > 0.0 )[0]
        idx3 = np.where( y_lc['mag_err'] > 0.0 )[0]
        idx = set(idx1).intersection(set(idx2))
        idx = list(idx.intersection(set(idx3)))
        
        mags = y_lc['mag'][idx]
        magerr = y_lc['mag_err'][idx]
        cols = blue_lc['mag'][idx] - red_lc['mag'][idx]
        colerr = np.sqrt(blue_lc['mag_err'][idx]**2 + red_lc['mag_err'][idx]**2)
        
        return mags, magerr, cols, colerr
    
    
    add_source_trail = False
    add_target_trail = True
    add_crosshairs = True
    add_source = True
    add_blend = True
    add_rc_centroid = True
    add_extinction_vector = True
    
    fig = plt.figure(1,(10,10))
    
    ax = plt.subplot(111)
    
    plt.rcParams.update({'font.size': 18})
        
    plt.scatter(colours,mags,
                 c='#E1AE13', marker='.', s=1, 
                 label='Stars within ROME field')
    
    plt.scatter(local_colours,local_mags,
                 c='#8c6931', marker='*', s=4, 
                 label='Stars < '+str(round(tol,1))+'arcmin of target')
    
    col_key = blue_filter+red_filter
    
    if getattr(source,blue_filter) != None and getattr(source,red_filter) != None\
        and add_source:
        
        plt.errorbar(getattr(source,col_key), getattr(source,yaxis_filter), 
                 yerr = getattr(source,'sig_'+yaxis_filter),
                 xerr = getattr(source,'sig_'+col_key), color='m',
                 marker='d',markersize=10, label='Source crosshairs')
        
        if add_crosshairs:
            plot_crosshairs(fig,getattr(source,col_key),getattr(source,yaxis_filter),'m')
        
        if add_source_trail:
            red_lc = source.lightcurves[red_filter]
            blue_lc = source.lightcurves[blue_filter]
            y_lc = source.lightcurves[yaxis_filter]
            
            (smags, smagerr, scols, scolerr) = calc_colour_lightcurve(blue_lc, red_lc, y_lc)
            
            plt.errorbar(scols, smags, yerr = smagerr, xerr = scolerr, 
                         color='m', marker='d',markersize=10, label='Source')
                 
    if getattr(blend,blue_filter) != None and getattr(blend,red_filter) != None \
        and add_blend:
        
        plt.errorbar(getattr(blend,col_key), getattr(blend,yaxis_filter), 
                 yerr = getattr(blend,'sig_'+yaxis_filter),
                 xerr = getattr(blend,'sig_'+col_key), color='b',
                 marker='v',markersize=10, label='Blend')
                
    if getattr(target,blue_filter) != None and getattr(target,red_filter) != None \
        and add_target_trail:
        
        plt.errorbar(getattr(target,col_key), getattr(target,yaxis_filter), 
                 yerr = getattr(target,'sig_'+yaxis_filter),
                 xerr = getattr(target,'sig_'+col_key), color='k',
                 marker='x',markersize=10)
        
        red_lc = target.lightcurves[red_filter]
        blue_lc = target.lightcurves[blue_filter]
        y_lc = target.lightcurves[yaxis_filter]
        
        (tmags, tmagerr, tcols, tcolerr) = calc_colour_lightcurve(blue_lc, red_lc, y_lc)
        
        plt.errorbar(tcols, tmags, yerr = tmagerr,xerr = tcolerr, 
                     color='k', marker='+',markersize=10, alpha=0.4,
                     label='Blended target')
    
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
    
    if add_extinction_vector:
        plot_extinction_vector(fig,params,yaxis_filter)
        
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

if __name__ == '__main__':
    
    params = get_args()
    plot_cmd(params)
    