# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 15:16:11 2018

@author: rstreet
"""

import numpy as np

def d2r( angle_deg ):
    """Function to convert an angle in degrees to radians"""
    
    angle_rad = ( np.pi * angle_deg ) / 180.0
    return angle_rad

def r2d( angle_rad ):
    """Function to convert an angle in radians to degrees"""
    
    angle_deg = ( 180.0 * angle_rad ) / np.pi
    return angle_deg
    
def separation_two_points(pointA,pointB):
    """Function to calculate the separation between two points on the sky, A and B. 
    Input are tuples of (RA, Dec) for each point in decimal degrees.
    Output is the arclength between them in decimal degrees.
    This function uses the full formula for angular separation, and should be applicable
    at arbitrarily large distances."""
    
    # Convert to radians because numpy requires them:
    pA = ( d2r(pointA[0]), d2r(pointA[1]) )
    pB = ( d2r(pointB[0]), d2r(pointB[1]) )
    
    half_pi = np.pi/2.0
    
    d1 = half_pi - pA[1]
    d2 = half_pi - pB[1]
    dra = pA[0] - pB[0]
    
    cos_gamma = ( np.cos(d1) * np.cos(d2) ) + \
                        ( np.sin(d1) * np.sin(d2) * np.cos(dra) )
                        
    gamma = np.arccos(cos_gamma)
    
    gamma = r2d( gamma )
    
    return gamma