# -*- coding: utf-8 -*-
"""
Created on Wed May  9 16:33:56 2018

@author: rstreet
"""
import os
import sys
from pyDANDIA import pipeline_setup
from pyDANDIA import logs
from pyDANDIA import metadata
from pyDANDIA import catalog_utils
from astropy.coordinates import SkyCoord
from astropy.coordinates import matching
import astropy.units as u
from astropy.table import Table
import vizier_tools
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

VERSION = 'calibrate_photometry_0.0.1'

def calibrate_photometry():
    """Function to calculate the photometric transform between the instrumental
    magnitudes produced by the pyDANDIA pipeline and catalog data."""
    
    params = get_args()

    setup = pipeline_setup.pipeline_setup(params)
    
    log = logs.start_stage_log( setup.red_dir, 'phot_calib', version=VERSION )
    
    (reduction_metadata, params, star_catalog) = fetch_metadata(setup,params,log)
    
    star_catalog = select_good_detected_stars(star_catalog,params,log)
    
    vphas_cat = fetch_catalog_sources_within_image(params,log)
    
    vphas_cat = select_calibration_stars(vphas_cat,params,log)
    
    catalog_file = os.path.join(params['red_dir'],'vphas_catalog.fits')
        
    match_index = match_stars_by_position(star_catalog,vphas_cat,log)
    
#    catalog_utils.output_vphas_catalog_file(catalog_file,vphas_cat,match_index=match_index)
    
    fit = calc_phot_calib(params,star_catalog,vphas_cat,match_index,log)
    
    star_catalog = apply_phot_calib(star_catalog,fit,log)
    
    output_to_metadata(star_catalog, reduction_metadata,vphas_cat,match_index,
                       setup,params,log)
    logs.close_log(log)
    
def get_args():
    """Function to gather the necessary commandline arguments"""
    
    params = { 'red_dir': '',
               'metadata': '',
               'log_dir': '',
               'catalog': '',
               'pipeline_config_dir': '',
               'software_dir': '', 
               'verbosity': '',
            }

    if len(sys.argv) > 1:
        params['red_dir'] = sys.argv[1]
        params['metadata'] = sys.argv[2]
        params['log_dir'] = sys.argv[3]
        params['ra'] = sys.argv[4]
        params['dec'] = sys.argv[5]
        
        if len(sys.argv) > 6:
            
            for a in sys.argv[6:]:
                (key,value) = a.split('=')
                
                params[key] = value
                
    else:
        params['red_dir'] = input('Please enter the path to the reduction directory: ')
        params['metadata'] = input('Please enter the name of the metadata file: ')
        params['log_dir'] = input('Please enter the path to the log directory: ')
        params['ra'] = input('Please enter the target RA [sexigesimal format]: ')
        params['dec'] = input('Please enter the target Dec [sexigesimal format]: ')
        params['det_mags_max'] = input('Please enter the faintest instrumental magnitude bin to use in calibration (or none to accept the defaults): ')
        params['det_mags_min'] = input('Please enter the brightest instrumental magnitude bin to use in calibration (or none to accept the defaults): ')
        params['cat_merr_max'] = input('Please enter the maximum allowed photometric uncertainty for a catalog measurement (or none to accept the defaults): ')
    
    params['target'] = SkyCoord([params['ra']], [params['dec']], 
                        unit=(u.hourangle, u.deg))
    
    for key in ['det_mags_max', 'det_mags_min', 'cat_merr_max']:
        if key in params.keys():
            params[key] = float(params[key])
        else:
            params[key] = None
            
    return params

def fetch_metadata(setup,params,log):
    """Function to extract the information necessary for the photometric
    calibration from a metadata file, adding information to the params 
    dictionary"""
    
    reduction_metadata = metadata.MetaData()
    reduction_metadata.load_a_layer_from_file( setup.red_dir, 
                                              params['metadata'],
                                              'data_architecture' )
    reduction_metadata.load_a_layer_from_file( setup.red_dir, 
                                               params['metadata'],
                                              'reduction_parameters' )
    reduction_metadata.load_a_layer_from_file( setup.red_dir, 
                                              params['metadata'], 
                                              'headers_summary' )
    reduction_metadata.load_a_layer_from_file( setup.red_dir, 
                                              params['metadata'],
                                              'star_catalog' )
    
    try:
        reduction_metadata.load_a_layer_from_file( setup.red_dir, 
                                              params['metadata'],
                                              'phot_calib' )
    except KeyError:
        pass
    
    params['fov'] = reduction_metadata.reduction_parameters[1]['FOV'][0]
    params['refimage'] = reduction_metadata.data_architecture[1]['REF_IMAGE'][0]
    iref = reduction_metadata.headers_summary[1]['IMAGES'].tolist().index(params['refimage'])
    params['ra'] = reduction_metadata.headers_summary[1]['RAKEY'][iref]
    params['dec'] = reduction_metadata.headers_summary[1]['DECKEY'][iref]
    params['filter'] = reduction_metadata.headers_summary[1]['FILTKEY'][iref]
    params['cat_mag_col'] = params['filter'].replace('p','') + 'mag'
    params['cat_err_col'] = 'e_'+params['filter'].replace('p','') + 'mag'
    
    log.info('Gathered information from metadata file '+params['metadata']+':')
    log.info('Image field of view: '+str(params['fov'])+'sq. deg')
    log.info('Reference image: '+params['refimage']+\
                ', index '+str(iref)+' in dataset')
    log.info('Filter used for dataset: '+params['filter'])
    log.info('Pointing center coordinates: '+params['ra']+' '+params['dec'])
    
    star_catalog = Table()
    star_catalog['star_index'] = reduction_metadata.star_catalog[1]['star_index']
    star_catalog['RA'] = reduction_metadata.star_catalog[1]['RA_J2000']
    star_catalog['DEC'] = reduction_metadata.star_catalog[1]['DEC_J2000']
    star_catalog['mag'] = reduction_metadata.star_catalog[1]['ref_mag']
    star_catalog['mag_err'] = reduction_metadata.star_catalog[1]['ref_mag_err']
    star_catalog['clean'] = np.zeros(len(reduction_metadata.star_catalog[1]['ref_mag']))
    star_catalog['cal_ref_mag'] = np.zeros(len(reduction_metadata.star_catalog[1]['ref_mag']))
    star_catalog['cal_ref_mag_err'] = np.zeros(len(reduction_metadata.star_catalog[1]['ref_mag']))
    
    log.info('Extracted star catalog')
    
    return reduction_metadata, params, star_catalog
    
def fetch_catalog_sources_within_image(params,log):
    """Function to extract the objects from the VPHAS+ catalogue within the
    field of view of the reference image, based on the metadata information."""
    
    params['radius'] = (np.sqrt(params['fov'])/2.0)*60.0
    
    log.info('Search radius: '+str(params['radius'])+' arcmin')
    
    vphas_cat = vizier_tools.search_vizier_for_sources(params['ra'], 
                                                       params['dec'], 
                                                        params['radius'], 
                                                        'VPHAS+')
        
    log.info('VPHAS+ search returned '+str(len(vphas_cat))+' entries')
        
    return vphas_cat    

def select_good_detected_stars(star_catalog,params,log,
                               select_around_target=False):
    """Function to identify and flag stars detected in the reference image
    for which the photometry is poor."""
    
    idx1 = np.where(star_catalog['mag'] > 10.0)[0].tolist()
    idx2 = np.where(star_catalog['mag_err'] > 0.0)[0].tolist()
    
    idx = set(idx1).intersection(set(idx2))

    log.info('Found '+str(len(list(idx)))+' detected stars with good photometry')
    
    stars = SkyCoord(star_catalog['RA'], star_catalog['DEC'], unit="deg")
    
    separations = params['target'].separation(stars)
    
    jdx = np.where(separations < 3.0*u.arcmin)
    
    log.info('Found '+str(len(list(jdx[0])))+' detected stars around target location')
    
    kdx = list(idx.intersection(set(jdx[0])))
    
    if len(jdx[0]) < 100:
        kdx = list(idx)
        
        log.info('WARNING: Could not selected on position; too few detected stars with good photometry around target location')
        
    star_catalog['clean'][kdx] = 1.0

    log.info('Identified '+str(len(kdx))+' detected stars with good photometry')
    
    return star_catalog
    
def select_calibration_stars(vphas_cat,params,log):
    """Function to remove the entries from the VPHAS catalog with poor 
    photometry. 
    Based on code by Y. Tsapras.
    
    Returns the same vphas_cat Table, but uses the clean Column to indicate
    selected stars with 1 and deselected stars with 0. 
    """
    
    cerr = params['cat_err_col']
    cmag = params['cat_mag_col']
    
    limit_mag = 17.5
    if params['filter'] == 'gp': limit_mag = 22.0
    if params['filter'] == 'rp': limit_mag = 18.0
        
    log.info('Using limiting mag '+str(limit_mag)+\
                    ' for catalog selection for filter '+params['filter'])
            
    vphas_cat['clean'] = 0
    
    idx = []
    for f in ['g','r','i']:
        
        col = 'e_'+f+'mag'

        med = np.median(vphas_cat[col][np.where(vphas_cat[col]>0)])

        max_err = 2.0 * med

        log.info('Median photometric uncertainty of catalog stars: '+str(med))
        log.info('Excluding catalog stars with uncertainty > '+str(max_err))
                    
        idx1 = np.where(vphas_cat[col] <= max_err)
        idx2 = np.where(vphas_cat[col] > 0)
        idx3 = np.where(vphas_cat[cmag] < limit_mag)
        
        jdx = (set(idx1[0]).intersection(set(idx2[0]))).intersection(set(idx3[0]))
        
        if len(idx) == 0:
            idx = list(idx) + list(jdx)
        else:
            idx = list(set(idx).intersection(jdx))
    
    vphas_cat['clean'][list(idx)] = 1
    
    log.info('Selected '+str(len(idx))+\
            ' stars suitable for use in photometric calibration')
    
    return vphas_cat

def match_stars_by_position(star_catalog,vphas_cat,log):
    """Function to cross-match stars by position.
    
    Returns:
        :param dict match_index: { Index in vphas_cat: index in star_cat }
    """
    
    ddx = np.where(star_catalog['clean'] == 1.0)[0]
    
    det_stars = SkyCoord(star_catalog['RA'][ddx], star_catalog['DEC'][ddx], unit="deg")
    
    vdx = np.where(vphas_cat['clean'] == 1.0)[0]

    cat_stars = SkyCoord(vphas_cat['_RAJ2000'][vdx], vphas_cat['_DEJ2000'][vdx], unit="deg")
    
    tolerance = 2.0 * u.arcsec
    
    match_data = matching.search_around_sky(det_stars, cat_stars, 
                                             seplimit=tolerance)    
    
    idx = np.argsort(match_data[2].value)
    
    det_index = match_data[0][idx]
    cat_index = match_data[1][idx]
    
    match_index = np.array(list(zip(ddx[det_index],vdx[cat_index])))
    
    log.info('Matched '+str(len(match_index)))
    
    return match_index

def calc_phot_calib(params,star_catalog,vphas_cat,match_index,log):
    """Function to plot the photometric calibration"""

    cmag = params['cat_mag_col']
    cerr = params['cat_err_col']
    
    fit = [0.0, 0.0]
    
    fit = model_phot_transform2(params,star_catalog,vphas_cat,
                                   match_index,fit,log)
                                   
    for i in range(0,1,1):

        fit = model_phot_transform2(params,star_catalog,vphas_cat,
                                   match_index,fit,log, diagnostics=True)
        
        log.info('Fit result ['+str(i)+']: '+repr(fit))

        match_index = exclude_outliers(vphas_cat,star_catalog,params,
                                        match_index,fit,log)
    
    log.info('Final fitted photometric calibration: '+repr(fit))

    return fit
    
def model_phot_transform(params,star_catalog,vphas_cat,match_index,fit,
                         log, diagnostics=True):
    """Function to make an initial guess at the fit parameters"""
    
    
    log.info('Fit initial parameters: '+repr(fit))
    
    cmag = params['cat_mag_col']
    cerr = params['cat_err_col']
    
    cat_mags = vphas_cat[cmag][match_index[:,1]]
    cat_merrs = vphas_cat[cerr][match_index[:,1]]
    det_mags = star_catalog['mag'][match_index[:,0]]
 
    config = set_calibration_limits(params,log)
    
    xibin = 0.5
    xbin1 = config['det_mags_max']
    xbin2 = xbin1 - xibin
    
    binned_data = []
    peak_bin = []
    xbins = []
    ybins = []
    
    while xbin2 > config['det_mags_min']:
        
        idx1 = np.where(det_mags <= xbin1)
        idx2 = np.where(det_mags > xbin2)
        idx = list(set(idx1[0].tolist()).intersection(set(idx2[0].tolist())))
        
#        print 'X: ',xbin1,xbin2, len(idx)
        
        if len(idx) > 0:
            
            ybin1 = cat_mags.max()
            yibin = 0.5
            ybin2 = ybin1 - yibin
            
            xbin_max = 0.0
            ybin_max = 0.0
            row_max = 0
            
            while ybin2 > cat_mags.min():
                
                jdx1 = np.where(cat_mags[idx] <= ybin1)
                jdx2 = np.where(cat_mags[idx] > ybin2)
                jdx3 = np.where(cat_merrs[idx] <= config['cat_merr_max'])
                jdx = set(jdx1[0].tolist()).intersection(set(jdx2[0].tolist()))
                jdx = list(set(jdx).intersection(set(jdx3[0].tolist())))
                
                if len(jdx) > 0:
                    
                    kdx = (np.array(idx)[jdx]).tolist()
                    
                    row = []
                    row.append( (xbin1 - (xibin/2.0)) )
                    row.append( (ybin1 - (yibin/2.0)) )
                    row.append(np.median(cat_mags[kdx]))
                    row.append(len(kdx))
                    
                    binned_data.append(row)
#                    print ' -> Y: ',ybin1,ybin2,row
                    
                    if len(kdx) > row_max:
                        xbin_max = (xbin1 - (xibin/2.0))
                        ybin_max = np.median(cat_mags[kdx])
                        row_max = len(kdx)
                        
                ybin1 -= yibin
                ybin2 = ybin1 - yibin
            
            if row_max > 5:
                xbins.append(xbin_max)
                ybins.append(ybin_max)
                peak_bin.append(row_max)
#                print 'Local maximum: ',xbins[-1], ybins[-1], peak_bin[-1], row_max
            
        xbin1 -= xibin
        xbin2 = xbin1 - xibin
    
    xbins = np.array(xbins)
    ybins = np.array(ybins)
    
    fit = calc_transform(fit, xbins, ybins)
    
    if diagnostics:
        
        f = open(os.path.join(params['red_dir'],'binned_phot.dat'),'w')
        for i in range(0,len(xbins),1):
            f.write(str(xbins[i])+' '+str(ybins[i])+'\n')
        f.close()
        
        fig = plt.figure(2)
        
        plt.errorbar(star_catalog['mag'][match_index[:,0]],
                     vphas_cat[cmag][match_index[:,1]], 
                     xerr=star_catalog['mag_err'][match_index[:,0]],
                     yerr=vphas_cat[cerr][match_index[:,1]],
                     color='m', fmt='none')
        
        plt.plot(xbins,ybins,'g+',markersize=4)

        xplot = np.linspace(xbins.min(),xbins.max(),50)
        yplot = phot_func(fit,xplot)
    
        plt.plot(xplot, yplot,'k-')

        plt.xlabel('Instrumental magnitude')
        
        plt.ylabel('VPHAS+ catalog magnitude')
            
        [xmin,xmax,ymin,ymax] = plt.axis()
        
        plt.axis([xmax,xmin,ymax,ymin])
        
        plt.savefig(os.path.join(params['red_dir'],
                    'phot_model_transform_'+params['filter']+'.png'))
    
        plt.close(2)

    log.info('Fitted parameters: '+repr(fit))
    
    return fit

def set_calibration_limits(params,log):
    """Function to use the parameters given or set defaults"""
    
    defaults = {'gp': {'det_mags_max': 15.0,
                       'det_mags_min': 10.0,
                       'cat_merr_max': 0.04},
                'rp': {'det_mags_max': 13.5,
                       'det_mags_min': 10.0,
                       'cat_merr_max': 0.04},
                'ip': {'det_mags_max': 13.5,
                       'det_mags_min': 10.0,
                       'cat_merr_max': 0.04}}
    
    def_params = defaults[params['filter']]
    
    set_params = {}
    
    log.info('Set calibration limits: ')
    for key in ['det_mags_max', 'det_mags_min', 'cat_merr_max']:
        
        if params[key] != None:
            
            set_params[key] = params[key]
            
        else:
            
            set_params[key] = def_params[key]
            
        log.info(key+' = '+str(set_params[key]))
    
    return set_params
    
def model_phot_transform2(params,star_catalog,vphas_cat,match_index,fit,
                         log, diagnostics=True):
    """Function to make an initial guess at the fit parameters"""
    
    
    log.info('Fit initial parameters: '+repr(fit))
    
    cmag = params['cat_mag_col']
    cerr = params['cat_err_col']
    
    cat_mags = vphas_cat[cmag][match_index[:,1]]
    cat_merrs = vphas_cat[cerr][match_index[:,1]]
    det_mags = star_catalog['mag'][match_index[:,0]]
    det_mag_errs = star_catalog['mag_err'][match_index[:,0]]

    config = set_calibration_limits(params,log)
    
    xibin = 0.5
    xbin1 = config['det_mags_max']
    xbin2 = xbin1 - xibin
    
    binned_data = []
    peak_bin = []
    xbins = []
    ybins = []
    
    (hist_data,xedges,yedges) = np.histogram2d(det_mags,cat_mags,bins=24)
    hist_data.T
    
    idx = np.where(hist_data < (hist_data.max()*0.05))
    hist_data[idx] = 0
    
    idx = np.where(hist_data > (hist_data.max()*0.05))
    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2
    
    k = np.where(xcenters[idx[0]] < config['det_mags_max'])
    xbins = xcenters[idx[0][k]]
    ybins = []
    for x in idx[0][k]:
        
        k = np.where(hist_data[x,:] == (hist_data[x,:].max()))
        ybins.append(ycenters[k][0])
    
    if len(xbins) <= 1 or len(ybins) <= 1:
        raise ValueError('Insufficient datapoint selected by calibration magnitude limits')
        exit()
        
    fit = calc_transform(fit, xbins, ybins)
    
    if diagnostics:
        
        f = open(os.path.join(params['red_dir'],'binned_phot.dat'),'w')
        for i in range(0,len(xbins),1):
            f.write(str(xbins[i])+' '+str(ybins[i])+'\n')
        f.close()
        
        fig = plt.figure(3)
        
        plt_errs = False
        if plt_errs:
            plt.errorbar(star_catalog['mag'][match_index[:,0]],
                     vphas_cat[cmag][match_index[:,1]], 
                     xerr=star_catalog['mag_err'][match_index[:,0]],
                     yerr=vphas_cat[cerr][match_index[:,1]],
                     color='m', fmt='none')
        else:
            plt.plot(star_catalog['mag'][match_index[:,0]],
                     vphas_cat[cmag][match_index[:,1]],'m.', markersize=1)
        
        plt.plot(xbins,ybins,'g+',markersize=4)

        xplot = np.linspace(xbins.min(),xbins.max(),50)
        yplot = phot_func(fit,xplot)
    
        plt.plot(xplot, yplot,'k-')

        plt.xlabel('Instrumental magnitude')
        
        plt.ylabel('VPHAS+ catalog magnitude')
            
        [xmin,xmax,ymin,ymax] = plt.axis()
        
        plt.axis([xmax,xmin,ymax,ymin])
        
        plt.savefig(os.path.join(params['red_dir'],
                    'phot_model_transform_'+params['filter']+'.png'))
    
        plt.close(3)

    log.info('Fitted parameters: '+repr(fit))
    
    return fit

def phot_weighted_mean(data,sigma):
    """Function to calculate the mean of a set of magnitude measurements, 
    weighted by their photometric errors"""
    
    sig2 = sigma * sigma
    wmean = (data/sig2).sum() / (1.0/sig2).sum()
    sig_wmean = 1.0 / (1.0/sig2).sum()
    
    return wmean, sig_wmean

def phot_func(p,mags):
    """Photometric transform function"""
    
    return p[0] + p[1]*mags

def errfunc(p,x,y):
    """Function to calculate the residuals on the photometric transform"""
    
    return y - phot_func(p,x)
    
def calc_transform(pinit, x, y):
    """Function to calculate the photometric transformation between a set
    of catalogue magnitudes and the instrumental magnitudes for the same stars
    """
        
    (pfit,iexec) = optimize.leastsq(errfunc,pinit,args=(x,y))
    
    return pfit

def exclude_outliers(vphas_cat,star_catalog,params,match_index,fit,log):
    
    cmag = params['cat_mag_col']
    
    pred_mags = phot_func(fit, (vphas_cat[cmag][match_index[:,1]]) )
    
    residuals = star_catalog['mag'][match_index[:,0]] - pred_mags
    
    (median,MAD) = calc_MAD(residuals)
    
    log.info('Median, MAD of photometric residuals: '+str(median)+' '+str(MAD))
    
    jdx = np.where(residuals >= (median - 3.0*MAD))[0]
    kdx = np.where(residuals <= (median + 3.0*MAD))[0]
    dx = list(set(jdx).intersection(set(kdx)))

    log.info('Excluded '+str(len(match_index)-len(dx))+', '+\
            str(len(match_index))+' stars remaining')
    
    match_index = np.array(list(zip(match_index[dx,0],match_index[dx,1])))
    
    return match_index

def calc_MAD(x):
    """Function to calculate the Median Absolute Deviation from a single-column
    array of floats"""
    
    median = np.median(x)
    MAD = np.median(abs(x - np.median(x)))
    
    return median, MAD

def apply_phot_calib(star_catalog,fit_params,log):
    """Function to apply the computed photometric calibration to calculate 
    calibrated magnitudes for all detected stars"""
    
    mags = star_catalog['mag']
    
    cal_mags = phot_func(fit_params,mags)
    
    idx = np.where(star_catalog['mag'] < 7.0)
    cal_mags[idx] = 0.0
    
    star_catalog['cal_ref_mag'] = cal_mags
    star_catalog['cal_ref_mag_err'] = star_catalog['mag_err']

    log.info('Calculated calibrated reference magnitudes for all detected stars')
    
    return star_catalog

def output_to_metadata(star_catalog, reduction_metadata,vphas_cat,match_index,
                       setup,params,log):
    """Function to output the star catalog to the reduction metadata. 
    Creates a phot_catalog extension if none exists, or overwrites an 
    existing one"""
    
    cmag = params['cat_mag_col']
    
    try:
        
        reduction_metadata.phot_calib[1]['star_index'] = star_catalog['star_index'][:]
        reduction_metadata.phot_calib[1]['cal_ref_mag'] = star_catalog['cal_ref_mag'][:]
        reduction_metadata.phot_calib[1]['cal_ref_mag_err'] = star_catalog['cal_ref_mag_err'][:]
        reduction_metadata.phot_calib[1]['_RAJ2000'][match_index[:,0]] = vphas_cat['_RAJ2000'][match_index[:,1]]
        reduction_metadata.phot_calib[1]['_DEJ2000'][match_index[:,0]] = vphas_cat['_DEJ2000'][match_index[:,1]]
        reduction_metadata.phot_calib[1]['imag'][match_index[:,0]] = vphas_cat['imag'][match_index[:,1]]
        reduction_metadata.phot_calib[1]['e_imag'][match_index[:,0]] = vphas_cat['e_imag'][match_index[:,1]]
        reduction_metadata.phot_calib[1]['rmag'][match_index[:,0]] = vphas_cat['rmag'][match_index[:,1]]
        reduction_metadata.phot_calib[1]['e_rmag'][match_index[:,0]] = vphas_cat['e_rmag'][match_index[:,1]]
        reduction_metadata.phot_calib[1]['imag'][match_index[:,0]] = vphas_cat['imag'][match_index[:,1]]
        reduction_metadata.phot_calib[1]['e_imag'][match_index[:,0]] = vphas_cat['e_imag'][match_index[:,1]]
        reduction_metadata.phot_calib[1]['clean'][match_index[:,0]] = vphas_cat['clean'][match_index[:,1]]

        log.info('Updating existing phot_calib table')
        
    except AttributeError:
        
        phot_catalog = np.zeros([len(star_catalog),12])
        phot_catalog[:,0] = star_catalog['star_index'][:]
        phot_catalog[:,1] = star_catalog['cal_ref_mag'][:]
        phot_catalog[:,2] = star_catalog['cal_ref_mag_err'][:]
        phot_catalog[match_index[:,0],3] = vphas_cat['_RAJ2000'][match_index[:,1]]
        phot_catalog[match_index[:,0],4] = vphas_cat['_DEJ2000'][match_index[:,1]]
        phot_catalog[match_index[:,0],5] = vphas_cat['imag'][match_index[:,1]]
        phot_catalog[match_index[:,0],6] = vphas_cat['e_imag'][match_index[:,1]]
        phot_catalog[match_index[:,0],7] = vphas_cat['rmag'][match_index[:,1]]
        phot_catalog[match_index[:,0],8] = vphas_cat['e_rmag'][match_index[:,1]]
        phot_catalog[match_index[:,0],9] = vphas_cat['gmag'][match_index[:,1]]
        phot_catalog[match_index[:,0],10] = vphas_cat['e_gmag'][match_index[:,1]]
        phot_catalog[match_index[:,0],11] = vphas_cat['clean'][match_index[:,1]]
    
        reduction_metadata.create_phot_calibration_layer(phot_catalog,log=log)
        
        log.info('Creating phot_calib table')
        
    reduction_metadata.save_a_layer_to_file(setup.red_dir, 
                                            params['metadata'],
                                            'phot_calib', log=log)
        
if __name__ == '__main__':
    
    calibrate_photometry()