# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 11:41:38 2018

@author: rstreet
"""

from sys import argv
from os import path
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from astropy.visualization import ZScaleInterval, ImageNormalize
matplotlib.use('TkAgg')
from skimage.measure import ransac
from skimage.transform import AffineTransform

def calc_coord_offset():
    """Function to calculate an image's offset in delta RA, delta Dec"""
    
    (star_true, star_meas) = get_star_positions()
        
    dra, ddec = star_meas.spherical_offsets_to(star_true)
    
    print('Offset from the stars measured -> true position:')
    print('Delta RA = '+str(dra.to(u.arcsec))+' Delta Dec = '+str(ddec.to(u.arcsec)))

def get_star_positions():
    """Function to gather user input for the reference star's true RA, Dec
    coordinates plus its apparent RA, Dec based on the existing image WCS"""
    
    if len(argv) == 1:
        true_ra = input('Please enter the reference stars TRUE RA [sexigesimal format]: ')
        true_dec = input('Please enter the reference stars TRUE Dec [sexigesimal format]: ')
        meas_ra = input('Please enter the reference stars MEASURED RA [sexigesimal format or degrees]: ')
        meas_dec = input('Please enter the reference stars MEASURED Dec [sexigesimal format or degrees]: ')
    else:
        true_ra = argv[1]
        true_dec = argv[2]
        meas_ra = argv[3]
        meas_dec = argv[4]
    
    star_true = SkyCoord(true_ra, true_dec, unit=(u.hourangle, u.deg))
    
    if ':' in meas_ra and ':' in meas_dec:
        
        star_meas = SkyCoord(meas_ra, meas_dec, unit=(u.hourangle, u.deg))

    elif '.' in meas_ra and '.' in meas_dec:

        star_meas = SkyCoord(meas_ra, meas_dec, unit=(u.deg, u.deg))

    else:

        print('ERROR: Measured RA, Dec given in inconsistent format.')
        
    return star_true, star_meas

def calc_offset_hist2d(setup,detected_stars, catalog_stars,
                       field_centre_ra, field_centre_dec,log,
                       diagnostics=False):
    """Function to apply the 2D histogram technique of calculating the 
    offset between two sets of world coordinates.
    Re-implementation of the the approach used in RoboCut, written by E. Bachelet.
    """
    
    nstars = len(detected_stars)
    
    (raXX,raYY) = np.meshgrid(detected_stars['ra'],catalog_stars['ra'])
    (decXX,decYY) = np.meshgrid(detected_stars['dec'],catalog_stars['dec'])

    deltaRA = (raXX - raYY).ravel()
    deltaDec = (decXX - decYY).ravel()
    
    if diagnostics:
        fig = plt.figure(1,(10,10))
        plt.subplot(2, 1, 1)
        plt.hist(deltaRA, 100)
        plt.xlabel('$\\Delta$RA')
        plt.subplot(2, 1, 2)
        plt.hist(deltaDec, 100)
        plt.xlabel('$\\Delta$Dec')
        plt.grid()
        plt.savefig(path.join(setup.red_dir,'ref','detected_catalog_sources_offset.png'))
        plt.close(1)
        
    best_offset = [[0,0],[0,0]]

    resolution_factor = 20
    
    while len(best_offset[0]) != 1:
        
        H = np.histogram2d(deltaRA, deltaDec, nstars*resolution_factor)
        
        best_offset = np.where(H[0] == np.max(H[0]))
        
        resolution_factor = resolution_factor/2.0
    
    RAbin = H[1][1]-H[1][0]
    DECbin = H[2][1]-H[2][0]
    RA_offset = (H[1][best_offset[0]] + RAbin/2.0)[0]
    Dec_offset = (H[2][best_offset[1]] + DECbin/2.0)[0]
    
    log.info('Measured transform: dRA='+str(RA_offset)+', dDec='+str(Dec_offset)+' deg')
    
    return RA_offset, Dec_offset

def calc_offset_pixels(setup, detected_stars, catalog_stars,log,
                       diagnostics=False):
    """Function to apply the 2D histogram technique of calculating the 
    offset between two sets of world coordinates.
    Re-implementation of the the approach used in RoboCut, written by E. Bachelet.
    """
    
    nstars = len(detected_stars)
    nxbins = int(detected_stars['x'].max() - detected_stars['x'].min())
    nybins = int(detected_stars['y'].max() - detected_stars['y'].min())
    resolution_factor = 2
    nbins = nxbins*resolution_factor
    
    (xXX,xYY) = np.meshgrid(detected_stars['x'],catalog_stars['x'])
    (yXX,yYY) = np.meshgrid(detected_stars['y'],catalog_stars['y'])
    
    #close = calc_separations(setup, detected_stars, catalog_stars,log)
    
    deltaX = (xXX - xYY).ravel()
    deltaY = (yXX - yYY).ravel()
    
    (xH,xedges) = np.histogram(deltaX,bins=nxbins)
    (yH,yedges) = np.histogram(deltaY,bins=nybins)
    xdx = np.where(xH == xH.max())
    ydx = np.where(yH == yH.max())
    
    best_offset = [[0,0],[0,0]]

    while len(best_offset[0]) != 1:
        nbins = nxbins*resolution_factor
        
        (H,xedges,yedges) = np.histogram2d(deltaX, deltaY, nbins)
        best_offset = np.where(H == np.max(H))
        resolution_factor = resolution_factor/2.0
    
    X_offset = (xedges[best_offset[0][0]] + xedges[best_offset[0][0]+1])/2.0
    Y_offset = (yedges[best_offset[1][0]] + xedges[best_offset[1][0]+1])/2.0
    
    log.info('Measured transform: dX='+str(X_offset)+', dY='+str(Y_offset)+' pixels')
    
    if diagnostics:
        fig = plt.figure(1,(10,10))
        plt.subplot(2, 1, 1)
        plt.hist(deltaX, nbins)
        [xmin,xmax,ymin,ymax] = plt.axis()
        plt.axis([-50.0,50.0,ymax*0.7,ymax])
        plt.plot([X_offset,X_offset],[ymin,ymax],'r-')
        plt.xlabel('$\\Delta$X')
        plt.grid()
        plt.subplot(2, 1, 2)
        plt.hist(deltaY, nbins)
        [xmin,xmax,ymin,ymax] = plt.axis()
        plt.axis([-50.0,50.0,ymax*0.7,ymax])
        plt.plot([Y_offset,Y_offset],[ymin,ymax],'r-')
        plt.xlabel('$\\Delta$Y')
        plt.grid()
        plt.savefig(path.join(setup.red_dir,'ref','detected_catalog_sources_offset.png'))
        plt.close(1)
        
        fig = plt.figure(1,(10,10))
        norm = ImageNormalize(H, interval=ZScaleInterval())
        plt.imshow(H, norm=norm)
        plt.colorbar()
        plt.savefig(path.join(setup.red_dir,'ref','detected_catalog_sources_offset_2D.png'))
        plt.close(1)
        #plt.show()
        
    return X_offset, Y_offset

def calc_separations(setup, detected_stars, catalog_stars,log):
    
    (xXX,xYY) = np.meshgrid(detected_stars['x'],catalog_stars['x'])
    (yXX,yYY) = np.meshgrid(detected_stars['y'],catalog_stars['y'])
    
    separations = np.sqrt( (xXX-xYY)**2 + (yXX-yYY)**2 )
    
    close = np.where(separations < 10.0)
    
    print(close)
    print(separations[close])
    
    for k in range(0,10,1):
        i = close[0][k]
        j = close[1][k]
        print(xXX[i],yXX[i],xYY[j],yYY[j])
    
    return close
    
def extract_nearby_stars(catalog,ra,dec,radius):
    
    centre = SkyCoord(ra,dec,frame='icrs', unit=(u.deg, u.deg))
    
    stars = SkyCoord(catalog['ra'], catalog['dec'], 
                     frame='icrs', unit=(u.deg, u.deg))
    
    separations = centre.separation(stars)
    
    idx = np.where(abs(separations.deg) <= radius)
    
    sub_catalog = catalog[idx]
    
    return sub_catalog

def detect_correspondances(setup, detected_stars, catalog_stars,log):
    
    det_array = np.zeros((len(detected_stars),2))
    det_array[:,0] = detected_stars['x'].data
    det_array[:,1] = detected_stars['y'].data
    
    cat_array = np.zeros((len(catalog_stars),2))
    cat_array[:,0] = catalog_stars['x'].data
    cat_array[:,1] = catalog_stars['y'].data
    
    (model, inliers) = ransac((det_array, cat_array), AffineTransform, min_samples=3,
                               residual_threshold=2, max_trials=100)
    
    log.info('RANSAC identified '+str(len(inliers))+' inlying objects in the matched set')
    log.info('Pixel offsets, dx='+str(model.translation[0])+', dy='+str(model.translation[1])+' pixels')
    log.info('Pixel scale factor '+repr(model.scale))
    log.info('Pixel rotation '+repr(model.rotation))
    
    dx = np.median(det_array[inliers,0] - cat_array[inliers,0])
    dy = np.median(det_array[inliers,1] - cat_array[inliers,1])
    log.info('Median offsets = '+str(dx)+', '+str(dy))
    
    det_array = np.zeros((len(detected_stars),2))
    det_array[:,0] = detected_stars['ra'].data
    det_array[:,1] = detected_stars['dec'].data
    
    cat_array = np.zeros((len(catalog_stars),2))
    cat_array[:,0] = catalog_stars['ra'].data
    cat_array[:,1] = catalog_stars['dec'].data
    
    (model2, inliers2) = ransac((det_array, cat_array), AffineTransform, min_samples=3,
                               residual_threshold=2, max_trials=100)
    
    log.info('RANSAC identified '+str(len(inliers2))+' inlying objects in the matched set in world coordinates')
    log.info('Pixel offsets, dx='+str(model2.translation[0])+', dy='+str(model2.translation[1])+' deg')
    log.info('Pixel scale factor '+repr(model2.scale))
    log.info('Pixel rotation '+repr(model2.rotation))
    
    return [ dx, dy ]

def transform_world_coordinates(setup, detected_stars, transform):
    
    ra = detected_stars['ra'].data
    dec = detected_stars['dec'].data
    
    ra1 = transform.scale[0] * ra * np.cos(transform.rotation) \
                - transform.scale[1] * dec * np.sin(transform.rotation + transform.shear) \
                + transform.translation[0]
                
    dec1 = transform.scale[0] * ra * np.cos(transform.rotation) \
                + transform.scale[1] * dec * np.sin(transform.rotation + transform.shear) \
                + transform.translation[1]
    
    detected_stars['ra'] = ra1
    detected_stars['dec'] = dec1
    
    return detected_stars
    
if __name__ == '__main__':
    
    calc_coord_offset()
    
