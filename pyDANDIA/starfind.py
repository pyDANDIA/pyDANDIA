###############################################################################
#
# starfind.py - identify the stars in a given image.
#
# dependencies:
#      numpy 1.8+
#      astropy 1.0+
#      scipy 0.15+
#      scikit-image 0.11+
#      scikit-learn 0.18+
#      matplotlib 1.3+
#      photutils 0.3.2+
#
# Developed by Yiannis Tsapras
# as part of the ROME/REA LCO Key Project.
#
# version 0.1a (development)
#
# Last update: 19 Oct 2017
###############################################################################

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.visualization import SqrtStretch, AsymmetricPercentileInterval
from astropy.visualization import ZScaleInterval
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.nddata import Cutout2D
from astropy import units as u
from astropy import table
from photutils import background, detection, DAOStarFinder
from photutils import CircularAperture
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from pyDANDIA import  metadata
from pyDANDIA import  pipeline_setup
from pyDANDIA import image_handling
import time
from datetime import datetime
import sys
import os

from pyDANDIA import  logs
from pyDANDIA import  config_utils
from pyDANDIA import  psf
from pyDANDIA import empirical_psf_simple


###############################################################################
def starfind(setup, path_to_image, reduction_metadata, plot_it=False,
                                                           thumbsize = 250,
                                                           log=None):
    """
    The routine will quickly identify stars in a given image and return
    a star list and image quality parameters. The output is to be used
    to select suitable candidate images for constructing a template
    reference image.

    :param object setup: this is an instance of the ReductionSetup class. See
                         reduction_control.py

    :param string path_to_image: The full path to the image to be processed.

    :param object reduction_metadata: The reduction metadata object from
                         which to extract the saturation value.

    :param boolean plot_it: Do you want to plot the selected stars?

    :param int thumbsize: Size of the pixel subregions to split a large image
                          into. Default 250x250.

    :param string log: Full The full path to the log file.

    :return status, report, params: the first two are strings reporting whether
                          the stage was completed successfully. params is
                          a dictionary with the image quality parameters.

    :rtype string, string, dictionary
    """

    imname = path_to_image.split('/')[-1]

    if log != None:
        log.info('Starting starfind for image '+imname)


    params = { 'sky': 0.0, 'sigma_y': 0.0, 'sigma_x': 0.0, 'corr_xy':0.0, 'nstars':0, 'sat_frac':0.0, 'symmetry' : 1. }

    image_structure = image_handling.determine_image_struture(path_to_image, log=log)

    t0 = time.time()
    im = fits.open(path_to_image)

    header = im[image_structure['sci']].header
    scidata = im[image_structure['sci']].data

    # Get size of image
    ymax, xmax = scidata.shape

    # If it is a large image, consider thumbnail sized (deftault 250x250) pixel
    # subregions and choose the one with the fewest saturated pixels to evaluate
    # stats
    try:

        saturation = reduction_metadata.reduction_parameters[1]['MAXVAL']

    except:

        if log != None:

            status = 'ERROR'
            report = ('Could not extract the saturation parameter '
                      'from the configuration file.')

            log.info(report)

            return status, report, params




    nr_sat_pix = 1000000
    bestx1 = -1
    bestx2 = -1
    besty1 = -1
    besty2 = -1

    # Attempt to consider subregrions from a large image
    # provided the thumbsize is smaller than half the size of the image.
    # Alternatively use the whole image.
    if (thumbsize <= xmax/2.0) and (thumbsize <= ymax/2.0):
        regionsx = np.arange(0, xmax, thumbsize)
        regionsy = np.arange(0, ymax, thumbsize)

        for i in regionsx[0:-1]:

            x1 = i
            x2 = i + thumbsize

            for j in regionsy[0:-1]:

                y1 = j
                y2 = j + thumbsize
                nr_pix = len(scidata[y1:y2,x1:x2][np.where(scidata[y1:y2,x1:x2] > saturation)])
                #print x1, x2, y1, y2, nr_pix

                if nr_pix < nr_sat_pix:
                    nr_sat_pix = nr_pix
                    bestx1 = x1
                    bestx2 = x2
                    besty1 = y1
                    besty2 = y2

    else:
        nr_pix = len(scidata[:,:][np.where(scidata[:,:] > saturation)])
        nr_sat_pix = nr_pix
        bestx1 = 1
        bestx2 = xmax
        besty1 = 1
        besty2 = ymax

    #mean, median, std = sigma_clipped_stats(scidata[1:ymax, 1:xmax], sigma=3.0, maxiters=5)
    # Evaluate mean, median and standard deviation for the selected subregion
    mean, median, std = sigma_clipped_stats(scidata[besty1:besty2, bestx1:bestx2],
                                            sigma=3.0, maxiters=5)

    if log is not None:
        log.info(' -> Measured mean, median std for image: '+str(mean)+', '+\
                    str(median)+', '+str(std)+' counts')

    # Identify stars
    daofind = DAOStarFinder(fwhm=3.0, threshold=5.*std)
    #import pdb; pdb.set_trace()

    sources = daofind(scidata[besty1:besty2,bestx1:bestx2] - median)

    # Write steps to a log file
    if log != None:
        log.info("Identifying sources on image %s ...\n" % path_to_image.split('/')[-1])
        if  (sources is None)  :

            status = 'ERROR'
            report = 'Insufficient number of sources found. Stopping execution.'

            log.info(report)

            return status, report, params

        elif (len(sources) > 0 and len(sources) <= 5):

            log.info('WARNING: Too few sources detected on image.')

        else:

            log.info("Found %s sources.\n" % str(len(sources)))

            log.info('Using best sources to determine FWHM (up to 30).')

    # Discount saturated stars
    sources = sources[np.where(sources['peak'] < saturation)]
    sources.sort('peak')
    sources.reverse()

    # Store the number of identified sources and fraction of saturated pixels
    # Discount stars too close to the edges of the image
    nstars = len(sources)
    if (thumbsize <= xmax/2.0) and (thumbsize <= ymax/2.0):
        sat_frac = nr_sat_pix/(thumbsize*thumbsize)
        sources = sources[np.where((sources['xcentroid'] > 10) &
                          (sources['xcentroid'] < thumbsize-10) &
                          (sources['ycentroid'] < thumbsize-10) &
                          (sources['ycentroid'] > 30))]
    else:
        sat_frac = nr_sat_pix/(xmax*ymax)
        sources = sources[np.where((sources['xcentroid'] > 10) &
                          (sources['xcentroid'] < xmax-10) &
                          (sources['ycentroid'] < ymax-10) &
                          (sources['ycentroid'] > 30))]

    # Keep only up to 100 stars
    sources = sources[0:100]

    sources_with_close_stars_ids = []
    # Discount stars with close neighbours (within r=10 pix)

    for i in np.arange(len(sources)):

        source_i = sources[i]

        for other_source in sources[i+1:]:

            if (np.sqrt((source_i['xcentroid']-other_source['xcentroid'])**2 +
                     (source_i['ycentroid']-other_source['ycentroid'])**2) <= 10 ):

                sources_with_close_stars_ids.append(i)
                    #print source_i, other_source

            continue

    # Keep up to 30 isolated sources only (may be fewer)
    sources.remove_rows(sources_with_close_stars_ids)
    sources = sources[0:30]

    if log != None:
        log.info("Kept %s sources.\n" % str(len(sources)))

    #return sources
    # Uncomment the following line to display source list in browser window:
    #sources.show_in_browser()

    # Fit a model to identified sources and estimate PSF shape parameters
    sky_arr = []
    sigma_x_arr = []
    sigma_y_arr = []
    corr_xy_arr = []

    i = 0
    while (i <= len(sources)-1):

        try:
            i_peak = sources[i]['peak']
            position = [sources[i]['xcentroid'], sources[i]['ycentroid']]
            #print position
            stamp_size = (20,20)
            cutout = Cutout2D(scidata[besty1:besty2,bestx1:bestx2],
                                 position, stamp_size) # in pixels
            yc, xc = cutout.position_cutout
            yy, xx = np.indices(cutout.data.shape)
            fit = psf.fit_star(cutout.data, yy, xx, psf_model='BivariateNormal')
            fit_params = fit[0]
            fit_errors = fit[1].diagonal()**0.5
            biv = psf.BivariateNormal()
            background = psf.ConstantBackground()
            fit_residuals = psf.error_star_fit_function(fit_params,
                                cutout.data, biv, background, yy, xx)

            fit_residuals = fit_residuals.reshape(cutout.data.shape)
            cov = fit[1]*np.sum(fit_residuals**2)/((stamp_size[0])**2-6)
            fit_errors = cov.diagonal()**0.5
            #print fit_params
            model = biv.psf_model(yy, xx, fit_params)
            if np.isnan(fit_params[3]) == True or np.isnan(fit_params[4]) == True:
                sigma_x_arr.append(0.0)
                sigma_y_arr.append(0.0)
            else:
                sigma_y_arr.append(fit_params[3])
                sigma_x_arr.append(fit_params[4])
            corr_xy_arr.append(fit_params[5])
            sky_arr.append(fit_params[6])

            if plot_it == True:
                plt.figure(figsize=(3,8))
                plt.subplot(3,1,1)
                plt.imshow(cutout.data, cmap='gray', origin='lower')
                plt.colorbar()
                plt.title("Data")
                plt.subplot(3,1,2)
                plt.imshow(model, cmap='gray', origin='lower')
                plt.colorbar()
                plt.title("Model")
                plt.subplot(3,1,3)
                plt.imshow(fit_residuals, cmap='gray', origin='lower')
                plt.title("Residual")
                plt.colorbar()
                plt.savefig(path.join(setup.red_dir,'starfind_model.png'))
        except:

            for key, value in params.items():

                if np.isnan(value):
                    params[key] = 0.0

            if log != None:
                log.info("Could not fit source: %s." %str(i))

        i = i + 1

    # Estimate the median values for the parameters over the stars identified
    if len(sigma_x_arr) > 0:
        params['sky'] = np.median(sky_arr)
        params['sigma_y'] = np.median(sigma_y_arr)
        params['sigma_x'] = np.median(sigma_x_arr)
        params['corr_xy'] = np.median(corr_xy_arr)
        params['nstars'] = nstars
        params['sat_frac'] = sat_frac

        try:
            if xmax>200 and ymax>200:
                psf_emp, psf_error_emp = empirical_psf_simple.empirical_psf_median(np.copy(scidata)[:200,:200], 20, saturation)
            else:
                psf_emp, psf_error_emp = empirical_psf_simple.empirical_psf_median(np.copy(scidata), 20, saturation)
            #imgname = os.path.basename(path_to_image)
            #hduout=fits.PrimaryHDU(psf_emp)
            #hduout.writeto('psf_'+imgname,overwrite = True)
            symmetry_metric = empirical_psf_simple.symmetry_check(psf_emp)
            params['symmetry'] = symmetry_metric
        except Exception as e:

            if log != None:
                report = ('Could not extract the symmetry based on the PSF ')
                log.info(report)

    if log != None:
        log.info('Measured median values:')
        log.info('Sky background = '+str(params['sky']))
        log.info('FWHM X = '+str(params['sigma_x']))
        log.info('FWHM Y = '+str(params['sigma_y']))
        log.info('Corr XY = '+str(params['corr_xy']))
        log.info('Nstars = '+str(params['nstars']))
        log.info('Saturation fraction = '+str(params['sat_frac']))
        log.info('symmetry = '+str(params['symmetry']))
    # If plot_it is True, plot the sources found
    if plot_it == True:

        temp = np.copy(scidata[besty1:besty2,bestx1:bestx2])
        temp[np.where(temp < median)] = median
        temp[np.where(temp > 25000)] = 25000
        positions = (sources['xcentroid'], sources['ycentroid'])
        apertures = CircularAperture(positions, r=4.)
        norm = ImageNormalize(interval=AsymmetricPercentileInterval(10,40),
                              stretch=SqrtStretch())
        plt.imshow(temp, cmap='gray', origin='lower', norm=norm)
        plt.title("Data")
        #plt.colorbar()
        apertures.plot(color='red', lw=1.2, alpha=0.5)
        plt.savefig(path.join(setup.red_dir,'stars250x250.png'))
        im.close()

    if log != None:
        log.info("Finished processing image %s in %.3f seconds.\n" % (str(imname), (time.time()-t0)))

    status = 'OK'
    report = 'Completed successfully'
    return status, report, params


###############################################################################
def build_star_finder(reduction_metadata, image_path, log):
    """Function to construct a DAOStarFinder object to detect point sources
    in an image"""

    log.info('Building star finder for '+os.path.basename(image_path))

    image_idx = reduction_metadata.images_stats[1]['IM_NAME'].tolist().index(os.path.basename(image_path))

    fwhm = reduction_metadata.images_stats[1]['FWHM'][image_idx]
    sky_bkgd = reduction_metadata.images_stats[1]['SKY'][image_idx]
    satvalue = reduction_metadata.reduction_parameters[1]['MAXVAL'][0]

    sky_bkgd_sig = np.sqrt(sky_bkgd)

    log.info('FWHM = '+str(fwhm))
    log.info('Sky background = '+str(sky_bkgd))

    det_threshold = 3.0 * sky_bkgd_sig

    log.info('Sky background sigma = '+str(sky_bkgd_sig))

    daofind = DAOStarFinder(fwhm=fwhm, threshold=det_threshold, peakmax=satvalue)

    log.info('Completed star finder object')

    return daofind

def detect_sources(setup, reduction_metadata, image_path, scidata, log,
                   diagnostics=False,table_format='table'):
    """Function to detect all sources in the given image

    :param MetaData reduction_metadata: pipeline metadata for this dataset
    :param str image_path: path to image file to be analyzed
    :param array scidata: image pixel data
    :param logging log: Open reduction log object
    :param diagnostics Bool: Switch for additional diagnostic plots
    :param str table_format: Output table format {'table', 'array'}

    Returns:

    :param array detected_sources: position information on all objects in the image
    """

    col_names = { 'id': 'index', 'xcentroid': 'x', 'ycentroid': 'y',
                  'sharpness': 'sharpness', 'roundness1': 'roundness1',
                  'roundness2': 'roundness2', 'npix': 'npix', 'sky': 'sky',
                  'peak': 'peak', 'flux': 'ref_flux', 'mag': 'mag' }

    daofind = build_star_finder(reduction_metadata, image_path, log)

    sources = daofind(scidata)

    if table_format == 'table':
        detected_sources = [ table.Column(name='index', data=sources['id'].data),
                             table.Column(name='x', data=sources['xcentroid'].data),
                             table.Column(name='y', data=sources['ycentroid'].data),
                             table.Column(name='ra', data=np.zeros(len(sources))),
                             table.Column(name='dec', data=np.zeros(len(sources))),
                             table.Column(name='ref_flux', data=sources['flux'].data),
                             table.Column(name='ref_flux_err', data=np.zeros(len(sources))),
                             table.Column(name='ref_mag', data=np.zeros(len(sources))),
                             table.Column(name='ref_mag_err', data=np.zeros(len(sources))) ]

        detected_sources = table.Table(detected_sources)

    else:
        for i, col in enumerate(sources.colnames):

            detected_sources[:,i] = sources[col].data

    log.info('Detected '+str(len(sources)))

    return detected_sources

def plot_detected_sources(image_path,file_path,sources):
    """Function to output a PNG image of an image overplotted with
    the x,y positions of detected objects"""

    image = fits.getdata(image_path)

    fig = plt.figure(1)

    norm = ImageNormalize(image, \
                interval=ZScaleInterval())

    plt.imshow(image, origin='lower', cmap=plt.cm.viridis,
               norm=norm)

    plt.plot(sources[:,1],sources[:,2],'r+')

    plt.xlabel('X pixel')

    plt.ylabel('Y pixel')

    plt.axis('equal')

    plt.savefig(file_path)

    plt.close(1)


###############################################################################
def run_starfind(setup, reduction_metadata):
    """
    Function to enable starfind to be run from the commandline

    :param object setup: this is an instance of the ReductionSetup class. See
                         reduction_control.py
    :param object reduction_metadata: the metadata object

    :return status, report: two strings reporting whether the stage was
                            completed successfully
    :rtype string, string
    """

    params = {}

    params['ref_image'] = raw_input('Please enter the path to the image to be analyzed: ')
    opt = raw_input('Do you want diagnosic plots output? T or F [default: F]: ')
    if 'T' in str(opt).upper():
        params['plot'] = True
    else:
        params['plot'] = False

    log = logs.start_pipeline_log(params['red_dir'], 'starfind')

    setup = pipeline_setup(params)

    (status, report, params) = starfind(setup, params['ref_image'],
                                        reduction_metadata,
                                        plot_it=params['plot'],
                                        log=log)
    logs.close_log(log)

    return status, report
