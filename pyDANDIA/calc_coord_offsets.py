# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 11:41:38 2018

@author: rstreet
"""

from sys import argv
from astropy.coordinates import SkyCoord
import astropy.units as u

def calc_coord_offset():
    """Function to calculate an image's offset in delta RA, delta Dec"""
    
    (star_true, star_meas) = get_star_positions()
        
    dra, ddec = star_meas.spherical_offsets_to(star_true)
    
    print 'Offset from the stars measured -> true position:'
    print 'Delta RA = ',dra.to(u.arcsec),' Delta Dec = ',ddec.to(u.arcsec)

def get_star_positions():
    """Function to gather user input for the reference star's true RA, Dec
    coordinates plus its apparent RA, Dec based on the existing image WCS"""
    
    if len(argv) == 1:
        true_ra = raw_input('Please enter the reference stars TRUE RA [sexigesimal format]: ')
        true_dec = raw_input('Please enter the reference stars TRUE Dec [sexigesimal format]: ')
        meas_ra = raw_input('Please enter the reference stars MEASURED RA [sexigesimal format or degrees]: ')
        meas_dec = raw_input('Please enter the reference stars MEASURED Dec [sexigesimal format or degrees]: ')
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


if __name__ == '__main__':
    
    calc_coord_offset()
    