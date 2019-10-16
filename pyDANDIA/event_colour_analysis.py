# -*- coding: utf-8 -*-
"""
Created on Tue May 15 20:48:12 2018

@author: rstreet
"""

from os import path, remove
from sys import argv
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.coordinates import matching
from astropy.table import Table
import astropy.units as u
from astropy import constants
from astropy.time import Time
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
import star_colour_data
import spectral_type_data
import jester_phot_transforms
import bilir_phot_transforms
import red_clump_utilities
import interp_Bessell_Brett
import isochrone_utilities
import photometry_classes
import logging
import stellar_radius_relations
import lens_properties
import pyslalib
import phot_source_colour

def perform_event_analysis():
    """Function to plot colour magnitude and colour-colour plots"""
    
    tol = 2.0       # Arcmin
    calib_on_colours = False
    
    params = get_args()
    
    log = start_log(params)
    
    (star_catalog,image_trios,catalog_header) = read_combined_star_catalog(params,log)
    
    lightcurves = read_lightcurves(params,log)
    
    target = find_target_data(params,star_catalog,lightcurves,image_trios,log)

    (source, blend) = calc_source_blend_params(params,log)
    
    source = calc_source_lightcurve(source, target, log)
    
    measure_photometric_source_colours(params,target,log)
    
    (det_idx, cat_idx, close_cat_idx) = index_valid_star_entries(star_catalog,
                                                                target,tol,log,
                                                                valid_cat=True)
    
    deltas = calibrate_instrumental_colour_colour_diagram(params,star_catalog,
                                                 catalog_header,target,
                                                 det_idx,cat_idx,close_cat_idx,
                                                 log,
                                                 calib=calib_on_colours)

    RC = localize_red_clump(star_catalog,close_cat_idx,log)
    
    analyse_colour_mag_diagrams(params,star_catalog,catalog_header,
                                target, source,blend,RC,
                                det_idx,cat_idx,close_cat_idx,log)
                                                 
    RC = measure_RC_offset(params,RC,target,log)
    
    (target,source,blend) = calc_phot_properties(target, source, blend, RC, log)
    
    plot_colour_colour_diagram(params,star_catalog,catalog_header,
                               target, source, blend, RC,
                               det_idx,cat_idx,close_cat_idx, log)
    
    (source, blend) = match_source_blend_isochrones(params,source,blend,log)
    
    (source, blend) = calc_source_blend_ang_radii(source, blend, log)
    
    (source, blend) = calc_source_blend_physical_radii(source, blend, log)
    
    (source,blend) = calc_source_blend_distance(source, blend, RC, log)
    
    lens = calc_lens_parameters(params, source, RC, log)
    
    output_red_clump_data_latex(params,RC,log)
    
    output_source_blend_data_latex(params,source,blend,log)
    
    output_lens_parameters_latex(params,source,lens,log)
    
def start_log(params, console=False):
    """Function to initialise a log file"""
    
    log_file = path.join(params['red_dir'],'colour_analysis.log')
    
    if path.isfile(log_file):
        remove(log_file)
        
    log = logging.getLogger( 'colour_analysis' )
    
    if len(log.handlers) == 0:
        log.setLevel( logging.INFO )
        file_handler = logging.FileHandler( log_file )
        file_handler.setLevel( logging.INFO )
        
        if console == True:
            console_handler = logging.StreamHandler()
            console_handler.setLevel( logging.INFO )
    
        formatter = logging.Formatter( fmt='%(asctime)s %(message)s', \
                                    datefmt='%Y-%m-%dT%H:%M:%S' )
        file_handler.setFormatter( formatter )

        if console == True:        
            console_handler.setFormatter( formatter )
    
        log.addHandler( file_handler )
        if console == True:            
            log.addHandler( console_handler )
    
    log.info('Analyzing event colour data')
    log.info('Initial parameters:')
    for key, value in params.items():
        log.info(key+': '+str(value))
        
    return log
    
def get_args():
    """Function to gather the necessary commandline arguments"""

    params = {}
    
    if len(argv) == 1:
        
        input_file = input('Please enter the path to the parameter file: ')

    else:

        input_file = argv[1]
    
    if path.isfile(input_file) == False:
        
        print('ERROR: Cannot find input parameter file')
        exit()
        
    flines = open(input_file,'r').readlines()
    
    str_keys = ['catalog_file', 'red_dir', 
                'target_ra', 'target_dec', 
                'star_class', 'isochrone_file',
                'target_lc_file_g', 'target_lc_file_r', 'target_lc_file_i']
                
    for line in flines:
        
        (key, value) = line.replace('\n','').split()
        
        if key in str_keys:
            
            params[key] = value
            
        else:
            
            if 'none' not in str(value).lower():
                params[key] = float(value)
            else:
                params[key] = None
            
    return params
    
def read_combined_star_catalog(params,log):
    """Function to read the photometric and star catalog data from a metadata file"""
    
    if path.isfile(params['catalog_file']) == False:
        
        return np.zeros(1)
    
    hdulist = fits.open(params['catalog_file'])
    
    data = hdulist[1].data
    
    header = hdulist[0].header
    
    star_catalog = Table(data)
    
    data = hdulist[2].data
    
    image_trios = Table(data)
    
    log.info('Read data from combined colour star catalog')
    
    return star_catalog, image_trios, header

def read_lightcurves(params,log):
    """Function to read in data from DanDIA format lightcurves for the target
    star in 3 colours"""
    
    lightcurves = { 'g':None, 'r':None , 'i': None}
    
    for f in lightcurves.keys():
        
        lc_file = params['target_lc_file_'+f]
        
        if path.isfile(lc_file):
            
            lightcurves[f] = read_rbn_lightcurve(lc_file,log)

    return lightcurves
    
def read_rbn_lightcurve(lc_file,log):
    """Function to read a lightcurve file in RoboNet format.
    Note: Reads the calibrated lightcurve data.    
    """
    
    if path.isfile(lc_file):
        
        lines = open(lc_file, 'r').readlines()
        
        imnames = []
        hjd = []
        cal_mag = []
        cal_mag_err = []
        
        for l in lines:
            
            if l[0:1] != '#':
                
                entries = l.replace('\n','').split()
                
                imnames.append( str(entries[0]).replace('.fits','').replace('_crop','') )
                hjd.append( float(entries[1]) )
                cal_mag.append( float(entries[8]) )
                cal_mag_err.append( float(entries[9]) )
        
        lc = Table()
        lc['images'] = imnames
        lc['hjd'] = hjd
        lc['mag'] = cal_mag
        lc['mag_err'] = cal_mag_err
    
        log.info('Read data for lightcurve '+lc_file)
        
    else:
        log.info('ERROR: Cannot access lightcurve file '+lc_file)
        
        lc = Table()
    
    return lc
    
def calc_source_blend_params(params,log):
    """Function to construct a dictionary of needed parameters for the 
    source and blend"""
    
    source = photometry_classes.Star()
    
    source.fs_g = params['f_s_g']
    source.sig_fs_g = params['sig_f_s_g']
    (source.g, source.sig_g) = flux_to_mag_pylima(source.fs_g,source.sig_fs_g)
    
    source.fs_r = params['f_s_r']
    source.sig_fs_r = params['sig_f_s_r']
    (source.r, source.sig_r) = flux_to_mag_pylima(source.fs_r,source.sig_fs_r)
    
    source.fs_i = params['f_s_i']
    source.sig_fs_i = params['sig_f_s_i']
    (source.i, source.sig_i) = flux_to_mag_pylima(source.fs_i,source.sig_fs_i)
    
    source.compute_colours(use_inst=True)
    source.transform_to_JohnsonCousins()
    
    log.info('\n')
    log.info('Source measured photometry:')
    log.info(source.summary(show_mags=True))
    log.info(source.summary(show_mags=False,show_colours=True))
    log.info(source.summary(show_mags=False,johnsons=True))
    
    blend = photometry_classes.Star()
    
    blend.fs_g = params['f_b_g']
    blend.sig_fs_g = params['sig_f_b_g']
    (blend.g, blend.sig_g) = flux_to_mag_pylima(blend.fs_g,blend.sig_fs_g)
    
    blend.fs_r = params['f_b_r']
    blend.sig_fs_r = params['sig_f_b_r']
    (blend.r, blend.sig_r) = flux_to_mag_pylima(blend.fs_r,blend.sig_fs_r)
    
    blend.fs_i = params['f_b_i']
    blend.sig_fs_i = params['sig_f_b_i']
    (blend.i, blend.sig_i) = flux_to_mag_pylima(blend.fs_i,blend.sig_fs_i)
    
    blend.compute_colours(use_inst=True)
    blend.transform_to_JohnsonCousins()
    
    log.info('\n')
    log.info('Blend measured photometry:')
    log.info(blend.summary(show_mags=True))
    log.info(blend.summary(show_mags=False,show_colours=True))
    log.info(blend.summary(show_mags=False,johnsons=True))
    
    return source, blend
    
def flux_to_mag_pylima(flux, flux_err):
    """Function to convert the flux and flux uncertainty measured by 
    modeling in pyLIMA to magnitudes

    Uses default pyLIMA zeropoint = 27.4 mag
    """
    
    def flux2mag(ZP, flux):
        
        return ZP - 2.5 * np.log10(flux)
    
    ZP = 27.40
    
    if flux < 0.0 or flux_err < 0.0:
        
        mag = 0.0
        mag_err = 0.0

    else:        

        mag = flux2mag(ZP, flux)
        
        mag_err = (2.5/np.log(10.0))*flux_err/flux
        
    return mag, mag_err

def mag_to_flux_pylima(mag, mag_err):
    """Function to convert magnitudes into flux units.
    Magnitude zeropoint is that used by pyLIMA.    
    """
    
    ZP = 27.40
    
    flux = 10**( (mag - ZP) / -2.5 )
    
    ferr = mag_err/(2.5*np.log10(np.e)) * flux
    
    return flux, ferr

    

def find_target_data(params,star_catalog,lightcurves,image_trios,log):
    """Function to identify the photometry for a given target, if present
    in the star catalogue"""
    
    target = photometry_classes.Star()
    
    if params['target_ra'] != None:
        
        target_location = SkyCoord([params['target_ra']], [params['target_dec']], unit=(u.hourangle, u.deg))
                
        stars = SkyCoord(star_catalog['RA'], star_catalog['DEC'], unit="deg")
        
        tolerance = 2.0 * u.arcsec
        
        match_data = matching.search_around_sky(target_location, stars, 
                                                seplimit=tolerance)    
                                                
        idx = np.argsort(match_data[2].value)
    
        if len(match_data[0]) > 0:
            target.star_index = star_catalog['star_index'][match_data[1][idx[0]]]
            target.ra = star_catalog['RA'][match_data[1][idx[0]]]
            target.dec = star_catalog['DEC'][match_data[1][idx[0]]]
            target.i = star_catalog['cal_ref_mag_ip'][match_data[1][idx[0]]]
            target.sig_i = star_catalog['cal_ref_mag_err_ip'][match_data[1][idx[0]]]
            target.r = star_catalog['cal_ref_mag_rp'][match_data[1][idx[0]]]
            target.sig_r = star_catalog['cal_ref_mag_err_rp'][match_data[1][idx[0]]]
            target.i_inst = star_catalog['ref_mag_ip'][match_data[1][idx[0]]]
            target.sig_i_inst = star_catalog['ref_mag_err_ip'][match_data[1][idx[0]]]
            target.r_inst = star_catalog['ref_mag_rp'][match_data[1][idx[0]]]
            target.sig_r_inst = star_catalog['ref_mag_err_rp'][match_data[1][idx[0]]]
            target.separation = match_data[2][idx[0]].to_string(unit=u.arcsec)
            try:
                target.g = star_catalog['cal_ref_mag_gp'][match_data[1][idx[0]]]
                target.sig_g = star_catalog['cal_ref_mag_err_gp'][match_data[1][idx[0]]]
                target.g_inst = star_catalog['ref_mag_gp'][match_data[1][idx[0]]]
                target.sig_g_inst = star_catalog['ref_mag_err_gp'][match_data[1][idx[0]]]
            except AttributeError:
                pass
            
            log.info('\n')
            log.info('Target identified as star '+str(target.star_index)+\
                        ' in the combined ROME catalog, with parameters:')
            log.info('RA = '+str(target.ra)+' Dec = '+str(target.dec))
            log.info('Measured ROME photometry, instrumental:')
            log.info(target.summary(show_mags=False, show_instrumental=True))
            log.info('Measured ROME photometry, calibrated to the VPHAS+ scale:')
            log.info(target.summary(show_mags=True))
            
            target.set_delta_mag(params)
            
            log.info('Assigned delta mag offsets between DanDIA lightcurve and pyDANDIA reference frame analysis:')
            for f in ['g', 'r', 'i']:
                log.info('Delta m('+f+') = '+str(getattr(target, 'delta_m_'+f))+' +/- '+str(getattr(target, 'sig_delta_m_'+f)))
            
        if target.i != None and target.r != None:
            
            target.compute_colours(use_inst=True)
            
            log.info(target.summary(show_mags=False,show_colours=True))
            
        target.transform_to_JohnsonCousins()
        
        log.info(target.summary(show_mags=False,johnsons=True))
        
    for f in ['i', 'r', 'g']:
        
        if f in lightcurves.keys():
            
            images = []
            hjds = []
            mags = []
            magerrs = []
            fluxes = []
            fluxerrs = []
            
            for i in image_trios[f+'_images']:
                name = str(i).replace('\n','').replace('.fits','')

                idx = np.where(lightcurves[f]['images'] == name)[0]
                
                if len(idx) > 0:
                    images.append(lightcurves[f]['images'][idx][0])
                    hjds.append(lightcurves[f]['hjd'][idx][0])
                    mags.append(lightcurves[f]['mag'][idx][0])
                    magerrs.append(lightcurves[f]['mag_err'][idx][0])
                    (flux,ferr) = mag_to_flux_pylima(lightcurves[f]['mag'][idx][0],
                                                     lightcurves[f]['mag_err'][idx][0])
                    fluxes.append(flux)
                    fluxerrs.append(ferr)
                    
                else:
                    images.append(name)
                    hjds.append(9999999.999)
                    mags.append(99.999)
                    magerrs.append(-9.999)
                    fluxes.append(9999999.999)
                    fluxerrs.append(-9999999.999)
                    
            lc = Table()
            lc['images'] = images
            lc['hjd'] = hjds
            lc['mag'] = mags
            lc['mag_err'] = magerrs
            lc['flux'] = fluxes
            lc['flux_err'] = fluxerrs
            
            target.lightcurves[f] = lc
            
    return target

def measure_photometric_source_colours(params,target,log):
    """Function to measure the source colours directly from multi-passband
    photometry"""
    
    log.info('\n')
    log.info('Attempting to estimate the source colours directly from the photometry')

    for f1,f2 in [ ('g','r'), ('g','i'), ('r','i')]:
        
        (source_colour,sig_source_colour,blend_flux, sig_blend_flux,fit) = phot_source_colour.measure_source_colour_odr(target.lightcurves[f1],
                                                                    target.lightcurves[f2])
        
        log.info('Fit to '+f2+' vs. '+f1+' flux:')
        log.info('Source colour ('+f1+'-'+f2+') = '+str(source_colour)+' +/- '+str(sig_source_colour))
        log.info('Blend flux in '+f2+': '+str(blend_flux)+' +/- '+str(sig_blend_flux))
    
        setattr(target,'fb_'+f2, blend_flux)
        setattr(target,'sig_fb_'+f2, sig_blend_flux)
        
        plot_file = path.join(params['red_dir'], 'flux_curve_'+f1+'_'+f2+'.eps')
    
        phot_source_colour.plot_bicolour_flux_curves(target.lightcurves[f1],
                                                     target.lightcurves[f2],
                                                     fit,f1,f2,
                                                     plot_file)
    
    try:
        
        #gr = target.fb_g / target.fb_r
        #sig_gr = np.sqrt( (target.sig_fb_g/target.fb_g)**2 + (target.sig_fb_r/target.fb_r)**2 )
        
        #setattr(target,'blend_gr',gr)
        #setattr(target,'blend_sig_gr',sig_gr)
        
        ri = target.fb_r / target.fb_i
        sig_ri = np.sqrt( (target.sig_fb_r/target.fb_r)**2 + (target.sig_fb_i/target.fb_i)**2 )
        
        setattr(target,'blend_ri',ri)
        setattr(target,'blend_sig_ri',sig_ri)
        
        log.info('Blend colours:')
        log.info('(g-r)_b = '+str(target.blend_gr)+' +/- '+str(target.blend_sig_gr))
        log.info('(r-i)_b = '+str(target.blend_ri)+' +/- '+str(target.blend_sig_ri))
        
    except AttributeError:
        
        pass
    
def calc_source_lightcurve(source, target, log):
    """Function to calculate the lightcurve of the source, based on the
    model source flux and the change in magnitude from the lightcurve"""
    
    log.info('\n')
    
    for f in ['i', 'r', 'g']:
        
        idx = np.where(target.lightcurves[f]['mag_err'] > 0)[0]
        
        dmag = np.zeros(len(target.lightcurves[f]['mag']))
        dmag.fill(99.99999)
        dmerr = np.zeros(len(target.lightcurves[f]['mag']))
        dmerr.fill(-9.9999)
        
        dmag[idx] = target.lightcurves[f]['mag'][idx] - getattr(target,f)
        dmerr[idx] = np.sqrt( (target.lightcurves[f]['mag_err'][idx])**2 + getattr(target,'sig_'+f)**2 )
        
        lc = Table()
        lc['images'] = target.lightcurves[f]['images']
        lc['hjd'] = target.lightcurves[f]['hjd']
        lc['mag'] = getattr(source,f) + dmag
        lc['mag_err'] = np.zeros(len(lc['mag']))
        lc['mag_err'] = dmerr
        
        lc['mag_err'][idx] = np.sqrt( dmerr[idx]*dmerr[idx] + (getattr(source,'sig_'+f))**2 )
    
        log.info('Calculated the source flux lightcurve in '+f)
        
        source.lightcurves[f] = lc
        
    return source
    
def index_valid_star_entries(star_catalog,target,tol,log,valid_cat=False):
    """Function to return an index of all stars with both full instrumental and
    catalogue entries"""
    
    idx1 = np.where(star_catalog['cal_ref_mag_ip'] > 0.0)[0]
    idx2 = np.where(star_catalog['cal_ref_mag_ip'] <= 22.0)[0]
    idx3 = np.where(star_catalog['cal_ref_mag_rp'] > 0.0)[0]
    idx4 = np.where(star_catalog['cal_ref_mag_rp'] <= 22.0)[0]
    idx5 = np.where(star_catalog['cal_ref_mag_gp'] > 0.0)[0]
    idx6 = np.where(star_catalog['cal_ref_mag_gp'] <= 22.0)[0]
    
    det_idx = set(idx1).intersection(set(idx2))
    det_idx = det_idx.intersection(set(idx3))
    det_idx = det_idx.intersection(set(idx4))
    det_idx = det_idx.intersection(set(idx5))
    det_idx = det_idx.intersection(set(idx6))
    
    log.info('Identified '+str(len(det_idx))+\
            ' detected stars with valid measurements in gri')
    
    if valid_cat == False:
        return list(det_idx), None, None
        
    idx4 = np.where(star_catalog['imag'] > 0.0)[0]
    idx5 = np.where(star_catalog['rmag'] > 0.0)[0]
    idx6 = np.where(star_catalog['gmag'] > 0.0)[0]
    
    cat_idx = det_idx.intersection(set(idx4))
    cat_idx = cat_idx.intersection(set(idx5))
    cat_idx = list(cat_idx.intersection(set(idx6)))
    det_idx = list(det_idx)
    
    log.info('Identified '+str(len(cat_idx))+\
            ' detected stars with valid catalogue entries in gri')
    
    close_idx = find_stars_close_to_target(star_catalog, target, tol, log)
    
    close_cat_idx = list(set(cat_idx).intersection(set(close_idx)))
    
    log.info('Identified '+str(len(close_cat_idx))+\
            ' stars close to the target with valid catalogue entries in gri')
            
    return det_idx, cat_idx, close_cat_idx

def find_stars_close_to_target(star_catalog, target, tol, log):
    """Function to identify stars close to the target"""
    
    tol = tol / 60.0        # Select stars within 2 arcmin of target
    det_stars = SkyCoord(star_catalog['RA'], star_catalog['DEC'], unit="deg")
    
    t = SkyCoord(target.ra, target.dec, unit="deg")
    
    seps = det_stars.separation(t)
    
    jdx = np.where(seps.deg < tol)[0]
    
    log.info('Identified '+str(len(jdx))+' stars within '+str(round(tol*60.0,1))+\
            'arcmin of the target')
    
    return jdx
    
def analyse_colour_mag_diagrams(params,star_catalog,catalog_header,
                                target,source,blend,RC,
                                det_idx,cat_idx,close_cat_idx,log):
    """Function to plot a colour-magnitude diagram"""
    
    tol = 2.0
    
    filters = { 'ip': 'SDSS-i', 'rp': 'SDSS-r', 'gp': 'SDSS-g' }
    
    inst_i = star_catalog['cal_ref_mag_ip'][det_idx]
    inst_r = star_catalog['cal_ref_mag_rp'][det_idx]
    inst_g = star_catalog['cal_ref_mag_gp'][det_idx]
    cal_i = star_catalog['imag'][cat_idx]
    cal_r = star_catalog['rmag'][cat_idx]
    cal_g = star_catalog['gmag'][cat_idx]
    inst_ri = inst_r - inst_i    # Catalogue column order is red -> blue
    inst_gr = inst_g - inst_r    
    inst_gi = inst_g - inst_i    
    cal_ri = cal_r - cal_i 
    cal_gr = cal_g - cal_r
    cal_gi = cal_g - cal_i
    
    linst_i = star_catalog['cal_ref_mag_ip'][close_cat_idx]
    linst_r = star_catalog['cal_ref_mag_rp'][close_cat_idx]
    linst_g = star_catalog['cal_ref_mag_gp'][close_cat_idx]
    lcal_i = star_catalog['imag'][close_cat_idx]
    lcal_r = star_catalog['rmag'][close_cat_idx]
    lcal_g = star_catalog['gmag'][close_cat_idx]
    linst_ri = linst_r - linst_i    # Catalogue column order is red -> blue
    linst_gr = linst_g - linst_r
    linst_gi = linst_g - linst_i
    lcal_ri = lcal_r - lcal_i
    lcal_gr = lcal_g - lcal_r
    lcal_gi = lcal_g - lcal_i
    
    plot_colour_mag_diagram(params,inst_i, inst_ri, linst_i, linst_ri, target, 
                            source, blend, RC, 'r', 'i', 'i', tol, log)
                            
    plot_colour_mag_diagram(params,inst_r, inst_ri, linst_r, linst_ri, target, 
                            source, blend, RC, 'r', 'i', 'r', tol, log)
                            
    plot_colour_mag_diagram(params,inst_g, inst_gr, linst_g, linst_gr, target, 
                            source, blend, RC, 'g', 'r', 'g', tol, log)
                            
    plot_colour_mag_diagram(params,inst_g, inst_gi, linst_g, linst_gi, target, 
                            source, blend, RC, 'g', 'i', 'g', tol, log)
    
def plot_crosshairs(fig,xvalue,yvalue,linecolour):
    
    ([xmin,xmax,ymin,ymax]) = plt.axis()
    
    xdata = np.linspace(xmin,xmax,10.0)
    ydata = np.zeros(len(xdata))
    ydata.fill(yvalue)
    
    plt.plot(xdata, ydata, linecolour+'-', alpha=0.5)
    
    ydata = np.linspace(ymin,ymax,10.0)
    xdata = np.zeros(len(ydata))
    xdata.fill(xvalue)
    
    plt.plot(xdata, ydata, linecolour+'-', alpha=0.5)
    
def plot_extinction_vector(fig,params,yaxis_filter):
    """Function to add a extinction vector to a colour-magnitude diagram."""
    
    [xmin,xmax,ymin,ymax] = plt.axis()
    
    deltay = (ymin-ymax) * 0.1
    deltax = (xmax-xmin) * 0.1
    
    ystart = ymin-deltay-params['A_'+yaxis_filter]
    
    plt.arrow(xmin+deltax, ystart, 0.0, params['A_'+yaxis_filter], 
              color='k', head_width=deltax*0.1, head_length=deltay/4.0)
    
    plt.text((xmin+deltax*0.9),ystart,'$A_'+yaxis_filter+'$')
    
def plot_colour_mag_diagram(params,mags, colours, local_mags, local_colours, 
                            target, source, blend, RC, blue_filter, red_filter, 
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
    
def plot_colour_colour_diagram(params,star_catalog,catalog_header,
                               target, source, blend, RC,
                               det_idx,cat_idx,close_cat_idx,log):
    """Function to plot a colour-colour diagram, if sufficient data are
    available within the given star catalog"""
    
    def calc_colours(g_lc,r_lc,i_lc):
        
        idx1 = np.where( g_lc['mag_err'] > 0.0 )[0]
        idx2 = np.where( r_lc['mag_err'] > 0.0 )[0]
        idx3 = np.where( i_lc['mag_err'] > 0.0 )[0]
        idx = set(idx1).intersection(set(idx2))
        idx = list(idx.intersection(set(idx3)))
        
        gr = g_lc['mag'][idx] - r_lc['mag'][idx] - RC.Egr
        gr_err = np.sqrt(g_lc['mag_err'][idx]**2 + r_lc['mag_err'][idx]**2)
        ri = r_lc['mag'][idx] - i_lc['mag'][idx] - RC.Eri
        ri_err = np.sqrt(r_lc['mag_err'][idx]**2 + i_lc['mag_err'][idx]**2)
        
        return gr, gr_err, ri, ri_err
    
    
    add_source_trail = False
    add_target_trail = True
    add_blend = True
    add_source = True
    add_crosshairs = True
    add_blend = True
    add_rc_centroid = True
    
    tol = 2.0
    
    filters = { 'ip': 'SDSS-i', 'rp': 'SDSS-r', 'gp': 'SDSS-g' }
    
    try:
    
        inst_i = star_catalog['cal_ref_mag_ip'][det_idx]
        inst_r = star_catalog['cal_ref_mag_rp'][det_idx]
        inst_g = star_catalog['cal_ref_mag_gp'][det_idx]
        inst_gr = inst_g - inst_r - RC.Egr
        inst_ri = inst_r - inst_i - RC.Eri
        
        linst_i = star_catalog['cal_ref_mag_ip'][close_cat_idx]
        linst_r = star_catalog['cal_ref_mag_rp'][close_cat_idx]
        linst_g = star_catalog['cal_ref_mag_gp'][close_cat_idx]
        lcal_i = star_catalog['imag'][close_cat_idx]
        lcal_r = star_catalog['rmag'][close_cat_idx]
        lcal_g = star_catalog['gmag'][close_cat_idx]
        linst_gr = linst_g - linst_r - RC.Egr
        linst_ri = linst_r - linst_i - RC.Eri
        
        fig = plt.figure(1,(10,10))
        
        ax = plt.axes()
        
        ax.scatter(inst_gr, inst_ri, 
                   c='#E1AE13', marker='.', s=1, 
                 label='Stars within ROME field')
        
        ax.scatter(linst_gr, linst_ri, marker='*', s=4, c='#8c6931',
                   label='Stars < '+str(round(tol,1))+'arcmin of target')
                     
        if source.gr_0 != None and source.ri_0 != None and add_source:
            
            plt.plot(source.gr_0, source.ri_0,'md',markersize=10, label='Source')
            
            if add_crosshairs:
                plot_crosshairs(fig,source.gr_0, source.ri_0,'m')
        
            if add_source_trail:
                g_lc = source.lightcurves['g']
                r_lc = source.lightcurves['r']
                i_lc = source.lightcurves['i']
                
                (sgr, sgr_err, sri, sri_err) = calc_colours(g_lc,r_lc,i_lc)
                
                plt.errorbar(sgr, sri, yerr = sri_err, xerr = sgr_err, 
                             color='m',marker='d',markersize=10, label='Source')
        
        if blend.gr_0 != None and blend.ri_0 != None and add_blend:
            #plt.plot(blend.gr_0, blend.ri_0,'bv',markersize=10, label='Blend')
            plt.errorbar(blend.gr_0, blend.ri_0, 
                         yerr = blend.sig_gr_0, xerr = blend.sig_ri_0, 
                             color='b',marker='v',markersize=10, label='Blend')

        if target.lightcurves['g'] != None and target.lightcurves['r'] != None\
            and target.lightcurves['i'] != None and add_target_trail:
            
            g_lc = target.lightcurves['g']
            r_lc = target.lightcurves['r']
            i_lc = target.lightcurves['i']
            
            (tgr, tgr_err, tri, tri_err) = calc_colours(g_lc,r_lc,i_lc)
            
            plt.errorbar(tgr, tri, yerr = tri_err, xerr = tgr_err, 
                         color='k',marker='+',markersize=10, alpha=0.4,
                         label='Blended target')
        
            plt.errorbar(target.gr_0, target.ri_0, 
                         yerr = target.sig_ri_0, xerr = target.sig_gr_0, 
                         color='k',marker='x',markersize=10)
                         
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
        
    except AttributeError:
            
        log.info('Warning: Insufficient data for colour-colour diagram')
        
def calibrate_instrumental_colour_colour_diagram(params,star_catalog,
                                                 catalog_header,target,
                                                 det_idx,cat_idx,close_cat_idx,
                                                 log,
                                                 calib=True):
    """Function to plot a colour-colour diagram, if sufficient data are
    available within the given star catalog"""
    
    tol = 2.0
    
    filters = { 'ip': 'SDSS-i', 'rp': 'SDSS-r', 'gp': 'SDSS-g' }
    
    deltas = {}
    deltas['di'] = 0.0
    deltas['dr'] = 0.0
    deltas['dg'] = 0.0
    deltas['dgr'] = 0.0
    deltas['dri'] = 0.0
    
    if calib:
        
        try:
        
            inst_i = star_catalog['cal_ref_mag_ip'][close_cat_idx]
            inst_r = star_catalog['cal_ref_mag_rp'][close_cat_idx]
            inst_g = star_catalog['cal_ref_mag_gp'][close_cat_idx]
            inst_gr = inst_g - inst_r    # Catalogue column order is red -> blue
            inst_ri = inst_r - inst_i   
            
            cal_i = star_catalog['imag'][close_cat_idx]
            cal_r = star_catalog['rmag'][close_cat_idx]
            cal_g = star_catalog['gmag'][close_cat_idx]
            cal_gr = cal_g - cal_r    # Catalogue column order is red -> blue
            cal_ri = cal_r - cal_i    
            
            plot_phot_transform(params, inst_i, cal_i, 'SDSS-i')
            plot_phot_transform(params, inst_r, cal_r, 'SDSS-r')
            plot_phot_transform(params, inst_g, cal_g, 'SDSS-g')
            
            deltas['di'] = np.median(cal_i - inst_i)
            deltas['dr'] = np.median(cal_r - inst_r)
            deltas['dg'] = np.median(cal_g - inst_g)
            deltas['dgr'] = np.median(cal_gr - inst_gr)
            deltas['dri'] = np.median(cal_ri - inst_ri)
            
            if len(target) > 0 and target['cal_ref_mag_ip'] > 0.0 and \
                target['cal_ref_mag_rp'] > 0.0 and target['cal_ref_mag_gp']:
                target_gr = target['cal_ref_mag_gp'] - target['cal_ref_mag_rp']
                target_ri = target['cal_ref_mag_rp'] - target['cal_ref_mag_ip']
        
            fig = plt.figure(1, (10,10))
            
            plt.rcParams.update({'font.size': 18})
            
            plt.plot(inst_gr, inst_ri,'+',
                     color='#2b8c85',markersize=4,alpha=0.4,
                     label='Instrumental')
            plt.plot(cal_gr, cal_ri,'.',
                     color='#8c6931',markersize=4,alpha=0.4,
                     label='Catalogue')
            
    #        plt.plot(inst_gr+deltas['dgr'], inst_ri+deltas['dri'],'.',
    #                 color='m',markersize=2,alpha=0.4,
    #                 label='Calibrated')
                     
    #        if len(target) > 0 and target['mag1'] > 0.0 and \
    #            target['mag2'] > 0.0 and target['mag2']:
    #            plt.plot(target_colour2, target_colour1,'md',markersize=6)
                
            plt.xlabel('SDSS (g-r) [mag]')
        
            plt.ylabel('SDSS (r-i) [mag]')
            
            plot_file = path.join(params['red_dir'],'calib_colour_colour_diagram.eps')
            
            plt.grid()
            
            plt.axis([-1.0,5.0,0.0,2.0])
            plt.legend()
            
            plt.savefig(plot_file)
        
            plt.close(1)
            
            log.info('Calibration colour-colour diagram output to '+plot_file)
            
            log.info('Measured offsets in photometry:')
            
            for key, value in deltas.items():
                log.info(key+' = '+str(value))
            
        except AttributeError:
            
            deltas = {}
            log.info('Warning: Insufficient data for colour-colour diagram')
        
    return deltas

def plot_phot_transform(params, inst_mag, cal_mag, bandpass):
    """Function to plot the relationship between the catalogue and instrumental
    magnitudes of a single passband"""
    
    fig = plt.figure(2)

    plt.plot(cal_mag, inst_mag,'k.')

    plt.xlabel('Catalog magnitude')

    plt.ylabel('Instrumental magnitude')
    
    plt.title('Relation between instrumental and catalogue magnitudes in '+\
                bandpass)
                
    [xmin,xmax,ymin,ymax] = plt.axis()
    
    plt.axis([xmax,xmin,ymax,ymin])
    
    plt.savefig(path.join(params['red_dir'],
                'phot_transform_'+bandpass+'.eps'))

    plt.close(2)

def localize_red_clump(star_catalog,close_cat_idx,log):
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
    
    inst_i = star_catalog['cal_ref_mag_ip'][close_cat_idx]
    inst_r = star_catalog['cal_ref_mag_rp'][close_cat_idx]
    inst_g = star_catalog['cal_ref_mag_gp'][close_cat_idx]
    cal_i = star_catalog['imag'][close_cat_idx]
    cal_r = star_catalog['rmag'][close_cat_idx]
    cal_g = star_catalog['gmag'][close_cat_idx]
    inst_ri = inst_r - inst_i    # Catalogue column order is red -> blue
    inst_gi = inst_g - inst_i 
    inst_gr = inst_g - inst_r  
    cal_ri = cal_r - cal_i 
    cal_gi = cal_g - cal_i
    cal_gr = cal_g - cal_r
    
    log.info('\n')
    log.info('Localizing the Red Clump')
    log.info('Median (r-i), i: '+str(np.median(inst_ri))+', '+str(np.median(inst_i)))
    log.info('Median (g-i), i: '+str(np.median(inst_gi))+', '+str(np.median(inst_i)))
    log.info('Median (g-r), g: '+str(np.median(inst_gr))+', '+str(np.median(inst_g)))
    
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
    
    idx = select_within_range(inst_i, inst_ri, i_min, i_max, ri_min, ri_max)
    
    (RC.ri, RC.sig_ri, RC.i, RC.sig_i) = calc_distribution_centroid_and_spread_2d(inst_ri[idx], inst_i[idx], use_iqr=True)
    
    idx = select_within_range(inst_r, inst_ri, r_min, r_max, ri_min, ri_max)
    
    (RC.r, RC.sig_r) = calc_distribution_centre_and_spread(inst_r[idx], use_iqr=True)
    
    idx = select_within_range(inst_g, inst_gr, g_min, g_max, gr_min, gr_max)
    
    (RC.gr, RC.sig_gr, RC.g, RC.sig_g) = calc_distribution_centroid_and_spread_2d(inst_gr[idx], inst_g[idx], use_iqr=True)
    
    idx = select_within_range(inst_g, inst_gi, g_min, g_max, gi_min, gi_max)
    
    (RC.gi, RC.sig_gi, RC.g, RC.sig_g) = calc_distribution_centroid_and_spread_2d(inst_gi[idx], inst_g[idx], use_iqr=True)
    
    log.info('\n')
    log.info('Centroid of Red Clump Stars at:')
    log.info(RC.summary(show_mags=True))
    log.info(RC.summary(show_mags=False,show_colours=True))
    
    RC.transform_to_JohnsonCousins()
    
    log.info(RC.summary(show_mags=False,johnsons=True))
    
    return RC

def calc_distribution_centroid_and_spread_2d(xdata, ydata, use_iqr=False):
    """Function to calculate the centroid of a 2D distribution and
    estimate the uncertainty on those values by different statistics"""
    
    (xcentre, sig_x) = calc_distribution_centre_and_spread(xdata, use_iqr=use_iqr)
    (ycentre, sig_y) = calc_distribution_centre_and_spread(ydata, use_iqr=use_iqr)
    
    return xcentre, sig_x, ycentre, sig_y

def calc_distribution_centre_and_spread(xdata, use_iqr=False):
    """Function to calculate the centroid of a 1D distribution and
    estimate the uncertainty on those values by different statistics"""
    
    xcentre = np.median(xdata)
    
    xmad = np.median(abs(xdata - xcentre))
    
    xiq_min = np.percentile(xdata,25.0)
    xiq_max = np.percentile(xdata,75.0)
    xiqr = (xiq_max - xiq_min)/2.0
    
    if use_iqr:
        sig_x = xiqr
    else:
        sig_x = xmad
    
    return xcentre, sig_x
    
def measure_RC_offset(params,RC,log):
    """Function to calculate the offset of the Red Clump from its expected 
    values, taken from Bensby et al. (2017), 2017, A&A, 605A, 89 for V, I bands and
    Hawkins et al. (2017) MNRAS, 471, 722 for 2MASS J,H,Ks.
    """
    
    in_use = False
    use_2mass = False
    
    RC = red_clump_utilities.get_essential_parameters(RC=RC)
    
    log.info('\n Red Clump colours and absolute SDSS magnitudes:')
    log.info('Mg_RC,0 = '+str(RC.M_g_0)+' +/- '+str(RC.sig_Mg_0)+'mag')
    log.info('Mr_RC,0 = '+str(RC.M_r_0)+' +/- '+str(RC.sig_Mr_0)+'mag')
    log.info('Mi_RC,0 = '+str(RC.M_i_0)+' +/- '+str(RC.sig_Mi_0)+'mag')
    log.info('MI_RC,0 = '+str(RC.M_I_0)+' +/- '+str(RC.sig_MI_0)+'mag')
    log.info('MV_RC,0 = '+str(RC.M_V_0)+' +/- '+str(RC.sig_MV_0)+'mag')
    log.info('(g-r)_RC,0 = '+str(RC.gr_0)+' +/- '+str(RC.sig_gr_0)+'mag')
    log.info('(g-i)_RC,0 = '+str(RC.gi_0)+' +/- '+str(RC.sig_gi_0)+'mag')
    log.info('(r-i)_RC,0 = '+str(RC.ri_0)+' +/- '+str(RC.sig_ri_0)+'mag')
    log.info('(V-I)_RC,0 = '+str(RC.VI_0)+' +/- '+str(RC.sig_VI_0)+'mag')
    
    if use_2mass:
        RC.transform_2MASS_to_SDSS()
        
        log.info('\n Red Clump NIR colours and magnitudes:')
        log.info('J_RC,0 = '+str(RC.M_J_0)+' +/- '+str(RC.sig_MJ_0)+'mag')
        log.info('H_RC,0 = '+str(RC.M_H_0)+' +/- '+str(RC.sig_MH_0)+'mag')
        log.info('Ks_RC,0 = '+str(RC.M_Ks_0)+' +/- '+str(RC.sig_MKs_0)+'mag')
        log.info('(J-H)_RC,0 = '+str(RC.JH_0)+' +/- '+str(RC.sig_JH_0)+'mag')
        log.info('(H-Ks)_RC,0 = '+str(RC.HK_0)+' +/- '+str(RC.sig_HK_0)+'mag')
    
    RC.D = red_clump_utilities.calc_red_clump_distance(params['target_ra'],params['target_dec'],log=log)
    RC = red_clump_utilities.calc_apparent_magnitudes(RC)
    
    log.info('\n Red Clump apparent SDSS magnitudes at distance '+str(RC.D)+'Kpc')
    log.info('g_RC,app = '+str(RC.m_g_0)+' +/- '+str(RC.sig_mg_0)+'mag')
    log.info('r_RC,app = '+str(RC.m_r_0)+' +/- '+str(RC.sig_mr_0)+'mag')
    log.info('i_RC,app = '+str(RC.m_i_0)+' +/- '+str(RC.sig_mi_0)+'mag')
    log.info('V_RC,app = '+str(RC.m_V_0)+' +/- '+str(RC.sig_mV_0)+'mag')
    log.info('I_RC,app = '+str(RC.m_I_0)+' +/- '+str(RC.sig_mI_0)+'mag')
    
    if in_use:
        RC.transform_to_JohnsonCousins()
        
        log.info('\n Derived Red Clump instrumental colours and magnitudes:')
        log.info(RC.summary(show_mags=False,johnsons=True))
    
    RC.A_g = RC.g - RC.m_g_0
    RC.sig_A_g = np.sqrt(RC.sig_mg_0*RC.sig_mg_0)
    RC.A_r = RC.r - RC.m_r_0
    RC.sig_A_r = np.sqrt(RC.sig_mr_0*RC.sig_mr_0)
    RC.A_i = RC.i - RC.m_i_0
    RC.sig_A_i = np.sqrt(RC.sig_mi_0*RC.sig_mi_0)
    
    RC.A_I = RC.I - RC.m_I_0
    RC.sig_A_I = RC.sig_mI_0
    RC.A_V = RC.V - RC.m_V_0
    RC.sig_A_V = RC.sig_mV_0
        
    RC.Egr = RC.gr - RC.gr_0
    RC.sig_Egr = np.sqrt( (RC.sig_gr_0*RC.sig_gr_0) )
    RC.Egi = RC.gi - RC.gi_0
    RC.sig_Egi = np.sqrt( (RC.sig_gi_0*RC.sig_gi_0) )
    RC.Eri = RC.ri - RC.ri_0
    RC.sig_Eri = np.sqrt( (RC.sig_ri_0*RC.sig_ri_0) )

    RC.EVI = RC.VI - RC.VI_0
    RC.sig_EVI = RC.sig_VI_0
    
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
    
    return RC
    
def calc_phot_properties(target, source, blend, RC, log):
    """Function to calculate the de-reddened and extinction-corrected 
    photometric properties of the target
    """
    in_use = False
    
    target.calibrate_phot_properties(RC,log=log)
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
    
    return target,source,blend

def calc_source_blend_ang_radii(source, blend, log):
    """Function to calculate the angular radius of the source star"""
    
    log.info('\n')
    log.info('Calculating the angular radius of the source star:')
    source.calc_stellar_ang_radius(log)
    log.info('Source angular radius (from SDSS (g-i), Boyajian+ 2014 relations) = '+str(round(source.ang_radius,4))+' +/- '+str(round(source.sig_ang_radius,4)))
    
    log.info('\n')
    log.info('Calculating the angular radius of the blend:')
    blend.calc_stellar_ang_radius(log)
    log.info('Blend angular radius (from SDSS (g-i), Boyajian+ 2014 relations) = '+str(round(blend.ang_radius,4))+' +/- '+str(round(blend.sig_ang_radius,4)))
    
    return source, blend
    
def calc_source_blend_physical_radii(source, blend, log):
    """Function to infer the physical radius of the source star from the 
    Torres mass-radius relation based on Teff, logg, and Fe/H
    
    Assumes a solar metallicity of Zsol = 0.0152.    
    """
    
    source.calc_physical_radius(log)
                    
    blend.calc_physical_radius(log)
    
    log.info('\n')
    log.info('Stellar radii derived from Torres relation (applies to main sequence and giants):')
    log.info('Source radius: '+\
                    str(round(source.radius,2))+' +/- '+str(round(source.sig_radius,2))+' Rsol')
    log.info('Blend radius: '+\
                    str(round(blend.radius,2))+' +/- '+str(round(blend.sig_radius,2))+' Rsol')
    
    source.starR_large_giant = 14.2
    source.sig_starR_large_giant = 0.2
    source.starR_small_giant = 8.1
    source.sig_starR_small_giant = 0.1
    
    log.info('\n')
    log.info('If the source were a giant, assigning possible physical radii values, based on data from Gaulme, P. et al. (2016), ApJ, 832, 121.')
    log.info('Small giant radius = '+str(source.starR_small_giant)+' +/- '+str(source.sig_starR_small_giant)+' Rsol')
    log.info('Large giant radius = '+str(source.starR_large_giant)+' +/- '+str(source.sig_starR_large_giant)+' Rsol')
    
    return source, blend
    
def convert_ndp(value,ndp):
    """Function to convert a given floating point value to a string, 
    rounded to the given number of decimal places, and suffix with zero
    if the value rounds to fewer decimal places than expected"""
    
    value = str(round(value,ndp))
    
    dp = value.split('.')[-1]
    
    while len(dp) < ndp:
        
        value = value + '0'
        
        dp = value.split('.')[-1]
    
    return value

def calc_source_blend_distance(source,blend,RC,log):
    """Function to calculate the distance to the source star, given the
    angular and physical radius estimates"""
    
    
    log.info('\n')
    
    source.calc_distance(log)
    
    log.info('Inferred source distance: '+str(source.D)+' +/- '+str(source.sig_D)+' pc')
    
    try:
        log.info('Inferred source distance if its a small red giant: '+\
        str(source.D_small_giant)+' +/- '+str(source.sig_D_small_giant)+' pc')
        
        log.info('Inferred source distance if its a large red giant: '+\
        str(source.D_large_giant)+' +/- '+str(source.sig_D_large_giant)+' pc')
        
    except AttributeError:
        pass
    
    blend.calc_distance(log)
    
    log.info('Inferred blend distance: '+str(blend.D)+' +/- '+str(blend.sig_D)+' pc')
    
    (Rstar, sig_Rstar) = stellar_radius_relations.scale_source_distance(source.ang_radius, source.sig_ang_radius, RC.D*1000.0 ,log)
    
    source.radius = Rstar
    source.sig_radius = sig_Rstar
    
    return source, blend

def match_source_blend_isochrones(params,source,blend,log):
    """Function to find the closest matching isochrone for both source
    and blend parameters"""
    
    log.info('\n')
    log.info('Analysing isochrones for source star\n')
    star_data = isochrone_utilities.analyze_isochrones(source.gr_0,source.ri_0, 
                                                       params['isochrone_file'],
                                                       log=log)
    source.mass = star_data[0]
    source.sig_mass = star_data[1]
    source.teff = star_data[2]
    source.sig_teff = star_data[3]
    source.logg = star_data[4]
    source.sig_logg = star_data[5]
    source.estimate_luminosity_class(log=log)
    
    log.info('\n')
    log.info('Analysing isochrones for blend\n')
    
    star_data = isochrone_utilities.analyze_isochrones(blend.gr_0,blend.ri_0, 
                                                       params['isochrone_file'],
                                                       log=log)
    blend.mass = star_data[0]
    blend.sig_mass = star_data[1]
    blend.teff = star_data[2]
    blend.sig_teff = star_data[3]
    blend.logg = star_data[4]
    blend.sig_logg = star_data[5]
    blend.estimate_luminosity_class(log=log)

    return source, blend

def calc_lens_parameters(params, source, RC, log):
    """Function to compute the physical parameters of the lens"""

    earth_position = pyslalib.slalib.sla_epv(params['t0']-2400000.0)
    
    v_earth = earth_position[1]     # Earth's heliocentric velocity vector
    
    pi_E = [ params['pi_E_N'], params['pi_E_E'] ]
    sig_pi_E = [ params['sig_pi_E_N'], params['sig_pi_E_E'] ]
    
    lens = lens_properties.Lens()
    lens.ra = params['target_ra']
    lens.dec = params['target_dec']
    lens.tE = params['tE']
    lens.sig_tE = params['sig_tE']
    lens.t0 = params['t0']
    lens.sig_t0 = params['sig_t0']
    lens.rho = params['rho']
    lens.sig_rho = params['sig_rho']
    lens.pi_E = np.array([ params['pi_E_N'], params['pi_E_E'] ])
    lens.sig_pi_E = np.array([ params['sig_pi_E_N'], params['sig_pi_E_E'] ])
    
    lens.calc_angular_einstein_radius(source.ang_radius,source.sig_ang_radius,log=log)
    
    lens.calc_distance(RC.D,0.0,log)
    lens.calc_distance_modulus(log)
    lens.calc_einstein_radius(log)
    
    lens.q = 10**(params['logq'])
    lens.sig_q = (params['sig_logq']/params['logq']) * lens.q
    lens.s = 10**(params['logs'])
    lens.sig_s = (params['sig_logs']/params['logs']) * lens.s
    
    lens.calc_masses(log)
    
    lens.calc_projected_separation(log)
    
    if params['dsdt'] != None and params['dalphadt'] != None:
        lens.dsdt = params['dsdt']
        lens.sig_dsdt = params['sig_dsdt']
        lens.dalphadt = params['dalphadt']
        lens.sig_dalphadt = params['sig_dalphadt']
    
        lens.calc_orbital_energies(log)
    
    lens.calc_rel_proper_motion(log)
    
    return lens
    
def output_red_clump_data_latex(params,RC,log):
    """Function to output a LaTeX format table with the data for the Red Clump"""
    
    file_path = path.join(params['red_dir'],'red_clump_data_table.tex')
    
    t = open(file_path, 'w')
    
    t.write('\\begin{table}[h!]\n')
    t.write('\\centering\n')
    t.write('\\caption{Photometric properties of the Red Clump, with absolute magnitudes ($M_{\\lambda}$) taken from \cite{Ruiz-Dern2018}, and the measured properties from ROME data.} \label{tab:RCproperties}\n')
    t.write('\\begin{tabular}{ll}\n')
    t.write('\\hline\n')
    t.write('\\hline\n')
    t.write('$M_{g,RC,0}$ & '+convert_ndp(RC.M_g_0,3)+' $\pm$ '+convert_ndp(RC.sig_Mg_0,3)+'\,mag\\\\\n')
    t.write('$M_{r,RC,0}$ & '+convert_ndp(RC.M_r_0,3)+' $\pm$ '+convert_ndp(RC.sig_Mr_0,3)+'\,mag\\\\\n')
    t.write('$M_{i,RC,0}$ & '+convert_ndp(RC.M_i_0,3)+' $\pm$ '+convert_ndp(RC.sig_Mi_0,3)+'\,mag\\\\\n')
    t.write('$(g-r)_{RC,0}$ & '+convert_ndp(RC.gr_0,3)+' $\pm$ '+convert_ndp(RC.sig_gr_0,3)+'\,mag\\\\\n')
    t.write('$(g-i)_{RC,0}$ & '+convert_ndp(RC.gi_0,3)+' $\pm$ '+convert_ndp(RC.sig_gi_0,3)+'\,mag\\\\\n')
    t.write('$(r-i)_{RC,0}$ & '+convert_ndp(RC.ri_0,3)+' $\pm$ '+convert_ndp(RC.sig_ri_0,3)+'\,mag\\\\\n')
    t.write('$m_{g,RC,0}$ & '+convert_ndp(RC.m_g_0,3)+' $\pm$ '+convert_ndp(RC.sig_mg_0,3)+'\,mag\\\\\n')
    t.write('$m_{r,RC,0}$ & '+convert_ndp(RC.m_r_0,3)+' $\pm$ '+convert_ndp(RC.sig_mr_0,3)+'\,mag\\\\\n')
    t.write('$m_{i,RC,0}$ & '+convert_ndp(RC.m_i_0,3)+' $\pm$ '+convert_ndp(RC.sig_mi_0,3)+'\,mag\\\\\n')
    t.write('$m_{g,RC,\\rm{centroid}}$  & '+convert_ndp(RC.g,2)+' $\pm$ '+convert_ndp(RC.sig_g,2)+'\,mag\\\\\n')
    t.write('$m_{r,RC,\\rm{centroid}}$  & '+convert_ndp(RC.r,2)+' $\pm$ '+convert_ndp(RC.sig_r,2)+'\,mag\\\\\n')
    t.write('$m_{i,RC,\\rm{centroid}}$  & '+convert_ndp(RC.i,2)+' $\pm$ '+convert_ndp(RC.sig_i,2)+'\,mag\\\\\n')
    t.write('$(g-r)_{RC,\\rm{centroid}}$ & '+convert_ndp(RC.gr,2)+'  $\pm$ '+convert_ndp(RC.sig_gr,2)+'\,mag\\\\\n')
    t.write('$(r-i)_{RC,\\rm{centroid}}$ & '+convert_ndp(RC.ri,2)+' $\pm$ '+convert_ndp(RC.sig_ri,2)+'\,mag\\\\\n')
    t.write('$A_{g}$ & '+convert_ndp(RC.A_g,3)+' $\pm$ '+convert_ndp(RC.sig_A_g,3)+'\,mag\\\\\n')
    t.write('$A_{r}$ & '+convert_ndp(RC.A_r,3)+' $\pm$ '+convert_ndp(RC.sig_A_r,3)+'\,mag\\\\\n')
    t.write('$A_{i}$ & '+convert_ndp(RC.A_i,3)+' $\pm$ '+convert_ndp(RC.sig_A_i,3)+'\,mag\\\\\n')
    t.write('$E(g-r)$ & '+convert_ndp(RC.Egr,3)+' $\pm$ '+convert_ndp(RC.sig_Egr,3)+'\,mag\\\\\n')
    t.write('$E(r-i)$ & '+convert_ndp(RC.Eri,3)+' $\pm$ '+convert_ndp(RC.sig_Eri,3)+'\,mag\\\\\n')
    t.write('\\hline\n')
    t.write('\\end{tabular}\n')
    t.write('\\end{table}\n')

    t.close()
    
    log.info('\n')
    log.info('Output red clump data in laTex table to '+file_path)
    
def output_source_blend_data_latex(params,source,blend,log):
    """Function to output a LaTex format table with the source and blend data"""

    file_path = path.join(params['red_dir'],'source_blend_data_table.tex')
    
    t = open(file_path, 'w')

    t.write('\\begin{table}[h!]\n')
    t.write('\\centering\n')
    t.write('\\caption{Photometric properties of the source star (S) and blend (b).} \label{tab:targetphot}\n')
    t.write('\\begin{tabular}{llll}\n')
    t.write('\\hline\n')
    t.write('\\hline\n')
    t.write('$m_{g,\\rm S}$ & '+convert_ndp(source.g,3)+' $\pm$ '+convert_ndp(source.sig_g,3)+'\,mag & $m_{g,b}$ & '+convert_ndp(blend.g,3)+' $\pm$ '+convert_ndp(blend.sig_g,3)+'\,mag\\\\\n')
    t.write('$m_{r,\\rm S}$ & '+convert_ndp(source.r,3)+' $\pm$ '+convert_ndp(source.sig_r,3)+'\,mag & $m_{r,b}$ & '+convert_ndp(blend.r,3)+' $\pm$ '+convert_ndp(blend.sig_r,3)+'\,mag\\\\\n')
    t.write('$m_{i,\\rm S}$ & '+convert_ndp(source.i,3)+' $\pm$ '+convert_ndp(source.sig_i,3)+'\,mag & $m_{i,b}$ & '+convert_ndp(blend.i,3)+' $\pm$ '+convert_ndp(blend.sig_i,3)+'\,mag\\\\\n')
    t.write('$(g-r)_{\\rm S}$ & '+convert_ndp(source.gr,3)+' $\pm$ '+convert_ndp(source.sig_gr,3)+'\,mag & $(g-r)_{b}$ & '+convert_ndp(blend.gr,3)+' $\pm$ '+convert_ndp(blend.sig_gr,3)+'\,mag\\\\\n')
    t.write('$(g-i)_{\\rm S}$ & '+convert_ndp(source.gi,3)+' $\pm$ '+convert_ndp(source.sig_gi,3)+'\,mag & $(g-i)_{b}$ & '+convert_ndp(blend.gi,3)+' $\pm$ '+convert_ndp(blend.sig_gi,3)+'\,mag\\\\\n')
    t.write('$(r-i)_{\\rm S}$ & '+convert_ndp(source.ri,3)+' $\pm$ '+convert_ndp(source.sig_ri,3)+'\,mag & $(r-i)_{b}$ & '+convert_ndp(blend.ri,3)+' $\pm$ '+convert_ndp(blend.sig_ri,3)+'\,mag\\\\\n')
#    t.write('$m_{g,s,0}$ & '+convert_ndp(source.g_0,3)+' $\pm$ '+convert_ndp(source.sig_g_0,3)+'\,mag & $m_{g,b,0}$ & '+convert_ndp(blend.g_0,3)+' $\pm$ '+convert_ndp(blend.sig_g_0,3)+'\,mag\\\\\n')
#    t.write('$m_{r,s,0}$ & '+convert_ndp(source.r_0,3)+' $\pm$ '+convert_ndp(source.sig_r_0,3)+'\,mag & $m_{r,b,0}$ & '+convert_ndp(blend.r_0,3)+' $\pm$ '+convert_ndp(blend.sig_r_0,3)+'\,mag\\\\\n')
#    t.write('$m_{i,s,0}$ & '+convert_ndp(source.i_0,3)+' $\pm$ '+convert_ndp(source.sig_i_0,3)+'\,mag & $m_{i,b,0}$ & '+convert_ndp(blend.i_0,3)+' $\pm$ '+convert_ndp(blend.sig_i_0,3)+'\,mag\\\\\n')
#    t.write('$(g-r)_{s,0}$ & '+convert_ndp(source.gr_0,3)+' $\pm$ '+convert_ndp(source.sig_gr_0,3)+'\,mag & $(g-r)_{b,0}$ & '+convert_ndp(blend.gr_0,3)+' $\pm$ '+convert_ndp(blend.sig_gr_0,3)+'\,mag\\\\\n')
#    t.write('$(r-i)_{s,0}$ & '+convert_ndp(source.ri_0,3)+' $\pm$ '+convert_ndp(source.sig_ri_0,3)+'\,mag & $(r-i)_{b,0}$ & '+convert_ndp(blend.ri_0,3)+' $\pm$ '+convert_ndp(blend.sig_ri_0,3)+'\,mag\\\\\n')
    t.write('$m_{g,\\rm S,0}$ & '+convert_ndp(source.g_0,3)+' $\pm$ '+convert_ndp(source.sig_g_0,3)+'\,mag &  & \\\\\n')
    t.write('$m_{r,\\rm S,0}$ & '+convert_ndp(source.r_0,3)+' $\pm$ '+convert_ndp(source.sig_r_0,3)+'\,mag &  & \\\\\n')
    t.write('$m_{i,\\rm S,0}$ & '+convert_ndp(source.i_0,3)+' $\pm$ '+convert_ndp(source.sig_i_0,3)+'\,mag &  & \\\\\n')
    t.write('$(g-r)_{\\rm S,0}$ & '+convert_ndp(source.gr_0,3)+' $\pm$ '+convert_ndp(source.sig_gr_0,3)+'\,mag &  & \\\\\n')
    t.write('$(g-i)_{\\rm S,0}$ & '+convert_ndp(source.gi_0,3)+' $\pm$ '+convert_ndp(source.sig_gi_0,3)+'\,mag &  & \\\\\n')
    t.write('$(r-i)_{\\rm S,0}$ & '+convert_ndp(source.ri_0,3)+' $\pm$ '+convert_ndp(source.sig_ri_0,3)+'\,mag &  & \\\\\n')
    t.write('\\hline\n')
    t.write('\\end{tabular}\n')
    t.write('\\end{table}\n')

    t.close()

    log.info('Output source and blend data in laTex table to '+file_path)

def output_lens_parameters_latex(params,source,lens,log):
    """Function to output a LaTex format table with the lens parameters"""
    
    file_path = path.join(params['red_dir'],'lens_data_table.tex')
    
    t = open(file_path, 'w')

    t.write('\\begin{table}[h!]\n')
    t.write('\\centering\n')
    t.write('\\caption{Physical properties of the source and lens system} \\label{tab:lensproperties}\n')
    t.write('\\begin{tabular}{lll}\n')
    t.write('\\hline\n')
    t.write('\\hline\n')
    t.write('Parameter   &   Units    &   Value \\\\\n')
    t.write('$\\theta_{\\rm{S}}$  & $\\mu$as     & '+convert_ndp(source.ang_radius,3)+'$\pm$'+convert_ndp(source.sig_ang_radius,3)+'\\\\\n')
    t.write('$\\theta_{\\rm{E}}$  & $\\mu$as     & '+convert_ndp(lens.thetaE,3)+'$\pm$'+convert_ndp(lens.sig_thetaE,3)+'\\\\\n')
    t.write('$R_{\\rm{S}}$       & $R_{\\odot}$ & '+convert_ndp(source.radius,3)+'$\pm$'+convert_ndp(source.sig_radius,3)+'\\\\\n')
    t.write('$M_{L,tot}$        & $M_{\\odot}$ & '+convert_ndp(lens.ML,3)+'$\pm$'+convert_ndp(lens.sig_ML,3)+'\\\\\n')
    t.write('$M_{L,1}$          & $M_{\\odot}$ & '+convert_ndp(lens.M1,3)+'$\pm$'+convert_ndp(lens.sig_M1,3)+'\\\\\n')
    t.write('$M_{L,2}$          & $M_{\\odot}$ & '+convert_ndp(lens.M2,3)+'$\pm$'+convert_ndp(lens.sig_M2,3)+'\\\\\n')
    t.write('$D_{L}$            & Kpc         & '+convert_ndp(lens.D,3)+'$\pm$'+convert_ndp(lens.sig_D,3)+'\\\\\n')
    t.write('$a_{\\perp}$       & AU          & '+convert_ndp(lens.a_proj,3)+'$\pm$'+convert_ndp(lens.sig_a_proj,3)+'\\\\\n')
#    t.write('KE/PE              &             & '+convert_ndp(lens.kepe,3)+'$\pm$'+convert_ndp(lens.sig_kepe,3)+'\\\\\n')
    t.write('$\mu$              & mas yr$^{-1}$ & '+convert_ndp(lens.mu_rel,2)+'$\pm$'+convert_ndp(lens.sig_mu_rel,2)+'\\\\\n')
    t.write('\\hline\n')
    t.write('\\end{tabular}\n')
    t.write('\\end{table}\n')

    t.close()

    log.info('Output lens parameters in laTex table to '+file_path)
  
if __name__ == '__main__':
    
    perform_event_analysis()
    