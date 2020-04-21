# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 16:41:42 2017

@author: rstreet
"""

from os import path
from sys import exit
import numpy as np
from scipy import optimize
from pyDANDIA import  logs
from pyDANDIA import  metadata
from pyDANDIA import  psf
from pyDANDIA import  stage0
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import visualization

def model_sky_background(setup,reduction_metadata,log,ref_star_catalog,
                        image_path=None, bandpass=None,
                        diagnostics=True):
    """Function to model the sky background of a real image by masking out
    the positions of the known stars in order to fit a better model to the
    background

    Returns the output of scipy.optimize.leastsq fit for the sky
    background model.
    """

    star_masked_image = load_masked_image(setup, reduction_metadata, log, image_path=image_path)

    psf_diameter = reduction_metadata.psf_dimensions[1]['psf_radius'][0]*2.0
    sat_value = reduction_metadata.reduction_parameters[1]['MAXVAL'][0]
    sat_value = 120000.0

    log.info('Statistics of masked reference image: ')
    log.info('Minimum = '+str(star_masked_image.min()))
    log.info('Maximum = '+str(star_masked_image.max()))
    log.info('Mean = '+str(star_masked_image.mean()))
    log.info('Std. dev = '+str(star_masked_image.std()))
    log.info('Median = '+str(np.median(star_masked_image)))

    (bin_counts, bins) = np.histogram(star_masked_image.flatten(),1000,range=(-10000.0, 20000.0))
    idx = np.where(bin_counts == bin_counts.max())
    most_freq_value = bins[idx[0]]
    log.info('Most frequent pixel value '+str(most_freq_value))

    dbin_min = 500.0
    dbin_max = 5000.0
    if bandpass != None and 'g' in bandpass:
        dbin_min=250.0
    log.info('Applying sky count bin selection limits: '+str(dbin_min)+' to '+str(dbin_max))

    delta_bins = ((bins[1:]+bins[0:-1])/2.0)[0:-1]
    delta_bin_counts = bin_counts[1:]-bin_counts[0:-1]
    idx1 = np.where(delta_bins > dbin_min)[0]
    idx2 = np.where(delta_bins < dbin_max)[0]
    idx = list(set(idx1).intersection(set(idx2)))
    jdx = np.where(delta_bin_counts[idx] == delta_bin_counts[idx].max())
    floor_value = delta_bins[idx][jdx[0]]
    if len(floor_value) > 1:
        floor_value = [floor_value[0]]
    log.info('Floor of most frequent pixel value curve '+str(floor_value))

    if floor_value > 3.0*most_freq_value:
        bkgd_value = most_freq_value
        log.info('Floor estimator seems to have over-estimated the sky background')
        log.info('Reverting to most frequent pixel value '+str(bkgd_value))
    else:
        bkgd_value = floor_value
        log.info('Using the floor estimator for the sky background = '+str(bkgd_value))

    if diagnostics:
        fig = plt.figure(1)
        plt.hist(star_masked_image.flatten(),1000,range=(0.0, 5000.0), log=True)
        (xmin,xmax,ymin,ymax) = plt.axis()
        plt.axis([0.0, 5000, ymin, ymax])
        plt.plot([floor_value, floor_value], [ymin,ymax], 'g-.', label='Floor estimator')
        plt.plot([most_freq_value, most_freq_value], [ymin,ymax], 'k-.', label='Most-frequent-value estimator')
        plt.plot([dbin_min, dbin_min], [ymin,ymax], 'm-.', label='Minimum threshold')
        plt.plot([dbin_max, dbin_max], [ymin,ymax], 'm-.', label='Maximum threshold')
        plt.xlabel('Pixel value [counts]')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig(path.join(setup.red_dir,'ref','ref_bkgd_hist.png'))
        plt.close(1)

        fig = plt.figure(2)
        plt.plot(delta_bins, delta_bin_counts, 'b.', label='Rate of change')
        (xmin,xmax,ymin,ymax) = plt.axis()
        plt.axis([0.0, 5000, ymin, ymax])
        plt.plot([floor_value, floor_value], [ymin,ymax], 'g-.', label='Floor estimator')
        plt.plot([most_freq_value, most_freq_value], [ymin,ymax], 'k-.', label='Most-frequent-value estimator')
        plt.plot([dbin_min, dbin_min], [ymin,ymax], 'm-.', label='Minimum threshold')
        plt.plot([dbin_max, dbin_max], [ymin,ymax], 'm-.', label='Maximum threshold')
        plt.xlabel('Pixel value [counts]')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig(path.join(setup.red_dir,'ref','ref_bkgd_hist_rate_of_change.png'))
        plt.close(2)

    background_type = reduction_metadata.reduction_parameters[1]['BACK_VAR'].data[0]

    if background_type == 'constant':

        sky_params = { 'background_type':background_type,
          'nx': star_masked_image.shape[1], 'ny': star_masked_image.shape[0],
          'constant': 0.0 }

    elif background_type == 'gradient':

        sky_params = { 'background_type':background_type,
          'nx': star_masked_image.shape[1], 'ny': star_masked_image.shape[0],
          'a0': 0.0, 'a1': 0.0, 'a2': 0.0 }

    sky_model = generate_sky_model(sky_params)

    if background_type == 'constant':

        sky_params['constant'] = bkgd_value
        log.info('Fitted parameters: '+repr(sky_params))

    elif background_type == 'gradient':

        sky_fit = fit_sky_background(star_masked_image,sky_model,'constant',
                        log=log, guess=bkgd_value)

        sky_params['a0'] = sky_fit[0][0]
        sky_params['a1'] = sky_fit[0][1]
        sky_params['a2'] = sky_fit[0][2]


    sky_model = generate_sky_model(sky_params)
    Y,X = np.indices(star_masked_image.shape)
    sky_model_image = sky_model.background_model(Y,X)

    sky_sigma = (star_masked_image - sky_model_image).std()

    sky_model.varience = sky_sigma * sky_sigma

    return sky_model

def load_masked_image(setup, reduction_metadata, log, image_path=None):
    """Function to load the image data and apply the bad pixel mask"""

    if image_path == None:
        image_path = str(reduction_metadata.data_architecture[1]['REF_PATH'][0]) +'/'+ str(reduction_metadata.data_architecture[1]['REF_IMAGE'][0])
    scidata = fits.getdata(image_path)

    image_bpm = stage0.open_an_image(setup, path.dirname(image_path),
                               path.basename(image_path), log,  image_index=2)
    if image_bpm == None:
        image_bpm = stage0.open_an_image(setup, path.dirname(image_path),
                                path.basename(image_path),log,  image_index=1)

    scidata = scidata.data
    idx = np.where(image_bpm.data != 0)
    image_bpm.data[idx] = 0.0
    scidata = scidata + image_bpm.data

    return scidata

def build_psf_mask(setup,psf_diameter,diagnostics=False):
    """Function to construct a mask for the PSF of a single star, which
    is a 2D image array with 1.0 at all pixel locations within the PSF and
    zero everywhere outside it.

    DEPRECIATED but still functional
    Previously called with the sequence:
    psf_mask = build_psf_mask(setup,psf_diameter,diagnostics=True)

    star_masked_image = mask_stars(setup,ref_image,ref_star_catalog, psf_mask,
                           diagnostics=False)

    sat_masked_image = mask_saturated_pixels(setup,ref_image,sat_value,log)

    star_masked_image.mask = star_masked_image.mask + sat_masked_image.mask
    """

    half_psf = int(psf_diameter/2.0)
    half_psf2 = half_psf*half_psf

    pxmax = 2*half_psf + 1
    pymax = pxmax

    psf_mask = np.ones([pymax,pxmax])

    pxmin = -half_psf
    pxmax = half_psf + 1
    pymin = -half_psf
    pymax = half_psf + 1

    for dx in range(pxmin,pxmax,1):

        dx2 = dx*dx

        for dy in range(pymin,pymax,1):

            if (dx2 + dy*dy) > half_psf2:
                psf_mask[dx + half_psf, dy + half_psf] = 0.0

    if diagnostics == True:
        fig = plt.figure(1)

        norm = visualization.ImageNormalize(psf_mask, \
                            interval=visualization.ZScaleInterval())

        plt.imshow(psf_mask, origin='lower', cmap=plt.cm.viridis, norm=norm)

        plt.xlabel('X pixel')

        plt.ylabel('Y pixel')

        plt.savefig(path.join(setup.red_dir,'ref','psf_mask.png'))

        plt.close(1)

    return psf_mask

def mask_stars(setup,ref_image,ref_star_catalog, psf_mask, diagnostics=False):
    """Function to create a mask for an image, returning a 2D image that has
    a value of 0.0 in every pixel that falls within the PSF of a star, and
    a value of 1.0 everywhere else."""

    psf_diameter = psf_mask.shape[0]

    half_psf = int(psf_diameter/2.0)

    pxmin = 0
    pxmax = psf_mask.shape[0]
    pymin = 0
    pymax = psf_mask.shape[1]

    star_mask = np.ones(ref_image.shape)

    for j in range(0,len(ref_star_catalog),1):

        xstar = int(float(ref_star_catalog[j,1]))
        ystar = int(float(ref_star_catalog[j,2]))

        xmin = xstar-half_psf
        xmax = xstar+half_psf+1
        ymin = ystar-half_psf
        ymax = ystar+half_psf+1

        px1 = pxmin
        px2 = pxmax + 1
        py1 = pymin
        py2 = pymax + 1

        if xmin < 0:

            px1 = abs(xmin)
            xmin = 0

        if ymin < 0:

            py1 = abs(ymin)
            ymin = 0

        if xmax > ref_image.shape[0]:

            px2 = px2 - (xmax - ref_image.shape[0] + 1)
            xmax = ref_image.shape[0] + 1

        if ymax > ref_image.shape[1]:

            py2 = py2 - (ymax - ref_image.shape[1] + 1)
            ymax = ref_image.shape[1] + 1

        star_mask[ymin:ymax,xmin:xmax] = star_mask[ymin:ymax,xmin:xmax] + \
                                            psf_mask[py1:py2,px1:px2]

    idx_mask = np.where(star_mask > 1.0)
    star_mask[idx_mask] = np.nan

    idx = np.isnan(star_mask)
    idx = np.where(idx == False)

    star_mask[idx] = 0.0
    star_mask[idx_mask] = 1.0

    masked_ref_image = np.ma.masked_array(ref_image,mask=star_mask)

    if diagnostics == True:

        fig = plt.figure(2)

        norm = visualization.ImageNormalize(star_mask, \
                            interval=visualization.ZScaleInterval())

        plt.imshow(star_mask, origin='lower', cmap=plt.cm.viridis, norm=norm)

        plt.plot(ref_star_catalog[:,1],ref_star_catalog[:,2],'r+')

        plt.xlabel('X pixel')

        plt.ylabel('Y pixel')

        plt.colorbar()

        plt.savefig(path.join(setup.red_dir,'ref_star_mask.png'))

        plt.close(2)


        fig = plt.figure(3)

        norm = visualization.ImageNormalize(masked_ref_image, \
                            interval=visualization.ZScaleInterval())

        plt.imshow(masked_ref_image, origin='lower',
                   cmap=plt.cm.viridis, norm=norm)

        plt.plot(ref_star_catalog[:,1],ref_star_catalog[:,2],'r+')

        plt.xlabel('X pixel')

        plt.ylabel('Y pixel')

        plt.colorbar()

        plt.savefig(path.join(setup.red_dir,'ref','masked_ref_image.png'))

        plt.close(3)

    return masked_ref_image

def mask_saturated_pixels(setup,image,saturation_value,log):
    """Function to mask saturated pixels in an image"""

    idx = np.where(image >= saturation_value)

    sat_mask = np.zeros(image.shape)

    sat_mask[idx] = 1.0

    masked_image = np.ma.masked_array(image, mask=sat_mask)

    return masked_image

def mask_saturated_pixels_quick(setup,image,saturation_value,log, min_value = None):
    """Function to mask saturated pixels in an image"""

    idx = np.where(image >= saturation_value)
    if min_value != None:
        idx2 = np.where(image < min_value)

    sat_mask = np.zeros(image.shape)

    sat_mask[idx] = 1.0
    if min_value != None:
        sat_mask[idx2] = 1.0

    sat_mask[::,::2] = 1.

    sat_mask[::2] = 1.

    masked_image = np.ma.masked_array(image, mask=sat_mask)

    return masked_image

def fit_sky_background(image,sky_model,background_type,log=None, guess=None):
    """Function to perform a Least Squares Fit of a given background
    model to a 2D image where the stars have been masked out"""

    init_pars = sky_model.background_guess(guess=guess)

    if log != None:

        log.info('Fitting '+background_type+\
        ' sky background model with initial parameters '+repr(init_pars))

    params = get_model_param_dict(init_pars,background_type,
                                  image.shape[1],image.shape[0])

    fit = optimize.leastsq(error_sky_fit_function, init_pars, args=(
        image, background_type), full_output=1)

    if log != None:

        log.info('Fitted parameters: ')

        for p in enumerate(fit[0]):

            log.info(str(p))

    return fit

def generate_sky_model(params):
    """Function to calculate the sky background model based on the provided
    parameters derived from a Background model"""

    p = get_model_param_list(params)

    if params['background_type'] == 'constant':

        model = psf.ConstantBackground()

    elif params['background_type'] == 'gradient':

        model = psf.GradientBackground()

    model.update_background_parameters(p)

    return model

def generate_sky_model_image(params):
    """Function to calculate the sky background model based on the provided
    parameters derived from a Background model"""

    p = get_model_param_list(params)

    sky_model = generate_sky_model(params)

    Y_data, X_data = np.indices((params['ny'],params['nx']))

    sky_bkgd = sky_model.background_model(Y_data,X_data, parameters=p)

    return sky_bkgd

def get_model_param_list(params):
    """Function to extract the correct parameters for a sky model
    from a dictionary, depending on the type of model used."""

    if params['background_type'] == 'constant':

        p = tuple([params['constant']])

    elif params['background_type'] == 'gradient':

        p = { 'a0': params['a0'], 'a1': params['a1'], 'a2': params['a2'] }

    return p

def get_model_param_dict(par_list,background_type,nx,ny):
    """Function to compose the correct parameters into dictionary format,
    depending on the type of model used."""

    params = { 'background_type': background_type, 'nx': nx, 'ny': ny }

    if background_type == 'constant':

        params['constant'] = par_list[0]

    elif background_type == 'gradient':

        params['a0'] = par_list[0]
        params['a1'] = par_list[1]
        params['a2'] = par_list[2]

    return params

def error_sky_fit_function(par_list, sky_data, background_type):
    """Function to calculate the residuals, or error function, of a 2D
    sky background image from a model of that image."""

    p = get_model_param_dict(par_list,background_type,
                             sky_data.shape[1],sky_data.shape[0])

    sky_model = generate_sky_model_image(p)

    return np.ravel(sky_data - sky_model)
