# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 16:41:42 2017

@author: rstreet
"""

from os import path
from sys import exit
import numpy as np
import logs
import metadata
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import visualization

def model_sky_background(setup,reduction_metadata,log,ref_star_catalog):
    """Function to model the sky background of an image"""
    
    # XXX Need to get the path to the reference image from the metadata
    
    ref_image = fits.getdata(reduction_metadata.ref_image_path)
    
    psf_size = reduction_metadata.reduction_parameters[1]['PSF_SIZE'][0]
    
    psf_mask = build_psf_mask(psf_size,diagnostics=True)
    
    star_mask = mask_stars(ref_image,ref_star_catalog, psf_mask, diagnostics=True)
 

def build_psf_mask(psf_size,diagnostics=False):
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
    
def mask_stars(ref_image,ref_star_catalog, psf_mask, diagnostics=False):
    """Function to create a mask of an image, returning a 2D image that has 
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

    return star_mask
    