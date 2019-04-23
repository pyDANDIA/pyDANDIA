# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 11:41:38 2018

@author: rstreet
"""

from sys import argv
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

def calc_offset_hist2d(detected_sources_world, catalog_sources_world,
                       field_centre_ra, field_centre_dec):
    """Function to apply the 2D histogram technique of calculating the 
    offset between two sets of world coordinates.
    Re-implementation of the the approach used in RoboCut, written by E. Bachelet.
    """
    
    detected_stars = extract_nearby_stars(detected_sources_world,
                                          field_centre_ra, field_centre_dec,
                                          5.0)
    
    catalog_stars = extract_nearby_stars(catalog_sources_world,
                                          field_centre_ra, field_centre_dec,
                                          5.0)
    
    nstars = len(detected_stars)
    
    (raXX,raYY) = np.meshgrid(detected_stars['ra'],catalog_stars['ra'])
    (decXX,decYY) = np.meshgrid(detected_stars['dec'],catalog_stars['dec'])

    deltaRA = (raXX - raYY).ravel()
    deltaDec = (decXX - decYY).ravel()
    
    fig = plt.figure(1,(10,10))
    plt.subplot(2, 1, 1)
    plt.hist(deltaRA, 100)
    plt.xlabel('$\\Delta$RA')
    plt.subplot(2, 1, 2)
    plt.hist(deltaDec, 100)
    plt.xlabel('$\\Delta$Dec')
    plt.show()
    
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
    
    return RA_offset, Dec_offset

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
    
