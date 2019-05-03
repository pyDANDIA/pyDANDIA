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
matplotlib.use('TkAgg')

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
    
    (xXX,xYY) = np.meshgrid(detected_stars['x'],catalog_stars['x'])
    (yXX,yYY) = np.meshgrid(detected_stars['y'],catalog_stars['y'])

    deltaX = (xXX - xYY).ravel()
    deltaY = (yXX - yYY).ravel()
    
    if diagnostics:
        fig = plt.figure(1,(10,10))
        plt.subplot(2, 1, 1)
        plt.hist(deltaX, 100)
        plt.xlabel('$\\Delta$X')
        plt.grid()
        plt.subplot(2, 1, 2)
        plt.hist(deltaY, 100)
        plt.xlabel('$\\Delta$Y')
        plt.grid()
        plt.savefig(path.join(setup.red_dir,'ref','detected_catalog_sources_offset.png'))
        plt.close(1)
    
    nxbins = int(detected_stars['x'].max() - detected_stars['x'].min())
    nybins = int(detected_stars['y'].max() - detected_stars['y'].min())
    (xH,xedges) = np.histogram(deltaX,bins=nxbins)
    (yH,yedges) = np.histogram(deltaY,bins=nybins)
    xdx = np.where(xH == xH.max())
    ydx = np.where(yH == yH.max())
    
    best_offset = [[0,0],[0,0]]

    resolution_factor = 2
    
    while len(best_offset[0]) != 1:
        
        (H,xedges,yedges) = np.histogram2d(deltaX, deltaY, nxbins*resolution_factor)
        
        best_offset = np.where(H == np.max(H))
        resolution_factor = resolution_factor/2.0
    
    X_offset = (xedges[best_offset[0][0]] + xedges[best_offset[0][0]+1])/2.0
    Y_offset = (yedges[best_offset[1][0]] + xedges[best_offset[1][0]+1])/2.0
    
    log.info('Measured transform: dRA='+str(X_offset)+', dDec='+str(Y_offset)+' pixels')
    
    return X_offset, Y_offset
    
def extract_nearby_stars(catalog,ra,dec,radius):
    
    centre = SkyCoord(ra,dec,frame='icrs', unit=(u.deg, u.deg))
    
    stars = SkyCoord(catalog['ra'], catalog['dec'], 
                     frame='icrs', unit=(u.deg, u.deg))
    
    separations = centre.separation(stars)
    
    idx = np.where(abs(separations.deg) <= radius)
    
    sub_catalog = catalog[idx]
    
    return sub_catalog
    
if __name__ == '__main__':
    
    calc_coord_offset()
    
