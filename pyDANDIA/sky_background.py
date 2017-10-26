# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 16:41:42 2017

@author: rstreet
"""

from os import path
from sys import exit
import numpy as np
from scipy import optimize
import logs
import metadata
import psf
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import visualization

def model_sky_background(setup,reduction_metadata,log,ref_star_catalog):
    """Function to model the sky background of a real image by masking out
    the positions of the known stars in order to fit a better model to the 
    background
    
    Returns the output of scipy.optimize.leastsq fit for the sky
    background model.
    """
        
    ref_image = fits.getdata(reduction_metadata.reference_image_path)
    
    psf_size = reduction_metadata.reduction_parameters[1]['PSF_SIZE'][0]
    
    psf_mask = build_psf_mask(setup,psf_size,diagnostics=True)
    
    star_masked_image = mask_stars(setup,ref_image,ref_star_catalog, psf_mask, 
                           diagnostics=True)
    
    if reduction_metadata.background_type == 'constant':
        
        sky_params = { 'background_type': reduction_metadata.background_type, 
          'nx': ref_image.shape[1], 'ny': ref_image.shape[0],
          'constant': 0.0 }
          
    elif reduction_metadata.background_type == 'gradient':
        
        sky_params = { 'background_type': reduction_metadata.background_type, 
          'nx': ref_image.shape[1], 'ny': ref_image.shape[0],
          'a0': 0.0, 'a1': 0.0, 'a2': 0.0 }
          
    sky_model = generate_sky_model(sky_params)
    
    sky_fit = fit_sky_background(star_masked_image,
                                                sky_model,
                                                'constant')
    return sky_fit
    
def build_psf_mask(setup,psf_size,diagnostics=False):
    """Function to construct a mask for the PSF of a single star, which
    is a 2D image array with 1.0 at all pixel locations within the PSF and 
    zero everywhere outside it."""
    
    pxmin = 0
    pxmax = int(psf_size)
    pymin = 0
    pymax = int(psf_size)
    
    half_psf = int(psf_size/2.0)
    
    psf_mask = np.zeros([int(psf_size),int(psf_size)])
    
    for x in range(pxmin,pxmax,1):
        
        for y in range(pymin,pymax,1):
            
            dx = x - half_psf
            dy = y - half_psf
            r = np.sqrt(dx*dx + dy*dy)

            if r <= half_psf:
                
                psf_mask[x,y] = 1.0
                
    if diagnostics == True:
        fig = plt.figure(1)
    
        norm = visualization.ImageNormalize(psf_mask, \
                            interval=visualization.ZScaleInterval())
        
        plt.imshow(psf_mask, origin='lower', cmap=plt.cm.viridis, norm=norm)
    
        plt.xlabel('X pixel')
        
        plt.ylabel('Y pixel')

        plt.savefig(path.join(setup.red_dir,'psf_mask.png'))

        plt.close(1)
    
    return psf_mask
    
def mask_stars(setup,ref_image,ref_star_catalog, psf_mask, diagnostics=False):
    """Function to create a mask for an image, returning a 2D image that has 
    a value of 0.0 in every pixel that falls within the PSF of a star, and
    a value of 1.0 everywhere else."""

    psf_size = psf_mask.shape[0]
    
    half_psf = int(psf_size/2.0)
    
    pxmin = 0
    pxmax = psf_mask.shape[0]
    pymin = 0
    pymax = psf_mask.shape[1]
    
    star_mask = np.ones(ref_image.shape)
    
    for j in range(0,len(ref_star_catalog),1):
        
        xstar = int(ref_star_catalog[j,1])
        ystar = int(ref_star_catalog[j,2])
        
        xmin = xstar-half_psf
        xmax = xstar+half_psf
        ymin = ystar-half_psf
        ymax = ystar+half_psf
        
        px1 = pxmin
        px2 = pxmax
        py1 = pymin
        py2 = pymax
        
        if xmin < 0:
            
            px1 = abs(xmin)
            xmin = 0
        
        if ymin < 0:
            
            py1 = abs(ymin)
            ymin = 0
        
        if xmax > ref_image.shape[0]:
            
            px2 = px2 - (xmax - ref_image.shape[0])
            xmax = ref_image.shape[0]
            
        if ymax > ref_image.shape[1]:
            
            py2 = py2 - (ymax - ref_image.shape[1])
            ymax = ref_image.shape[1]
            
        star_mask[ymin:ymax,xmin:xmax] = star_mask[ymin:ymax,xmin:xmax] + \
                                            psf_mask[py1:py2,px1:px2]
        
    idx = np.where(star_mask > 1.0)
    star_mask[idx] = 0.0
    
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

    ### THIS NEEDS TO OUTPUT real image without the stars

    return star_mask

def fit_sky_background(image,sky_model,background_type):
    """Function to perform a Least Squares Fit of a given background 
    model to a 2D image where the stars have been masked out"""
    
    init_pars = sky_model.background_guess()
    
    params = get_model_param_dict(init_pars,background_type,
                                  image.shape[1],image.shape[0])
    
    fit = optimize.leastsq(error_sky_fit_function, init_pars, args=(
        image, background_type), full_output=1)
    
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

    sky_bkgd = sky_model.background_model(Y_data, X_data, p)
    
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
    