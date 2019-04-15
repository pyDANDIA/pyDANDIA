# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 15:49:31 2019

@author: rstreet
"""

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from pyDANDIA import  logs
from pyDANDIA import  match_utils


def find_xy_offset(catalog1, catalog2, log=None, diagnostics=False):
    """Function to measure the x,y pixel offset between two catalogs using
    the shortest-string method
    The input catalogs may be from a subframe.
    
    Inputs:
    :param np.array catalog1: [nstars,2], columns x,y
    :param np.array catalog2: [nstars,2], columns x,y
    
    Outputs:
    :param float x_offset: Offset in x-axis
    :param float y_offset: Offset in y-axis
    """
    
    if log!=None:
        log.info('Calculating offsets via shortest-string method')
        
    x_offset = None
    y_offset = None
    
    for it in range(0,2,1):
        
        string_lengths = []
        n_match = []
    
        if it == 0:
            sub_pixel = False
        else:
            sub_pixel = True
            
        (dxrange,dyrange) = calc_xy_ranges(catalog1,sub_pixel,
                                            x_offset,y_offset,log=log)
        
        for dy in dyrange:
            
            lengths = []
            n = []
            
            for dx in dxrange:
                
                catalog_prime = offset_catalog2_coords(catalog2,dx,dy)
                
                matched_stars = cross_match_catalogs(catalog1,catalog_prime)
                
                lengths.append(np.median(matched_stars.separation[:]))
                
                n.append(matched_stars.n_match)
                
            string_lengths.append( lengths )
            n_match.append(n)
            
            log.info(' -> Completed trial dx='+str(dx)+', dy='+str(dy))
            
        string_lengths = np.array(string_lengths)
        n_match = np.array(n_match)
        
        (x_offset, y_offset, shortest) = find_shortest_string(string_lengths,
                                                    n_match,dxrange,dyrange,log)
        
        if diagnostics:
            plot_offsets(dxrange,dyrange,string_lengths,shortest,sub_pixel)
    
    return x_offset, y_offset
    
def calc_xy_ranges(catalog1,sub_pixel,x_offset,y_offset,log=None):
    """Function to determine an appropriate range of offsets in x and y
    over which to determine the shortest strings
    
    On the first iteration, where sub_pixel=False, this function examines
    the range of pixel positions represented in the first catalog.
    The offset is assumed to be less than 1/4 of the range (image width and 
    height of the catalog1 positions.  
    
    On the second iteration, the range returned is a +/-5 pixels around
    the estimated offset from the first iteration.
    
    Returns:
        :param np.range dxrange: Range of deltax values
        :param np.range dyrange: Range of deltay values
    """
    
    if x_offset == None and y_offset == None:
        xmin = catalog1[:,0].min()
        xmax = catalog1[:,0].max()
        ymin = catalog1[:,1].min()
        ymax = catalog1[:,1].max()
        
        dxmin = max( (-(xmax-xmin)/2.0), -100 )
        dymin = max( (-(ymax-ymin)/2.0), -100 )
        dxmax = min( (xmax-xmin)/2.0, 100 )
        dymax = min( (ymax-ymin)/2.0, 100 )
    
    else:
        # Need to correct for output sign convention
        dxmin = -x_offset - 5.0
        dxmax = -x_offset + 5.0
        dymin = -y_offset - 5.0
        dymax = -y_offset + 5.0
        
    if sub_pixel:
        dxincr = 0.1
        dyincr = 0.1
    else:
        dxincr = 2.0
        dyincr = 2.0
    
    dxrange = np.arange(dxmin,dxmax,dxincr)
    dyrange = np.arange(dymin,dymax,dyincr)
    
    if log!=None:
        log.info('Examining deltax range: '+str(dxrange.min())+\
                ' to '+str(dxrange.max())+', '+str(len(dxrange))+\
                ' intervals of '+str(dxincr))
        log.info('Examining deltay range: '+str(dyrange.min())+\
                ' to '+str(dyrange.max())+', '+str(len(dyrange))+\
                ' intervals of '+str(dyincr))
    
    return dxrange, dyrange
    
def offset_catalog2_coords(catalog2,dx,dy):
    """Function to calculate the predicted star positions from catalog 2, 
    applying the current offset"""
    
    catalog_prime = np.zeros(catalog2.shape)
    catalog_prime[:,0] = catalog2[:,0] + dx
    catalog_prime[:,1] = catalog2[:,1] + dy
    
    return catalog_prime

def cross_match_catalogs(catalog1,catalog2):
    """Function to match stars between the objects detected in an image
    and those extracted from a catalog, using image pixel postions."""
    
    tol = 4.0   # pixels, ~1 arcsec
    
    matched_stars = match_utils.StarMatchIndex()
    
    for i in range(0,len(catalog1),1):
        
        dx = catalog2[:,0]-catalog1[i,0]
        dy = catalog2[:,1]-catalog1[i,1]
        
        sep = np.sqrt(dx*dx + dy*dy)
        
        idx = sep.argsort()
        
        if len(idx) > 0 and sep[idx[0]] <= tol:
            
            p = {'cat1_index': i,
                 'cat1_ra': None,
                 'cat1_dec': None,
                 'cat1_x': catalog1[i,0],
                 'cat1_y': catalog1[i,1],
                 'cat2_index': idx[0], 
                 'cat2_ra': None, 
                 'cat2_dec': None, 
                 'cat2_x': catalog2[idx[0],0],
                 'cat2_y': catalog2[idx[0],1],
                 'separation': sep[idx[0]]}
                 
            matched_stars.add_match(p)
            
            #print(matched_stars.summarize_last(type='pixels'))
            
    return matched_stars

def find_shortest_string(string_lengths,n_match,dxrange,dyrange,log=None):
    """Function to find the x,y offsets which produce the median shortest string"""
    
    idx = np.where(string_lengths == string_lengths.min())
    
    # Sign inversion here so that the offsets returned are oriented such that
    # catalog1 + offset -> catalog2
    x_offset = -dxrange[idx[1][0]]
    y_offset = -dyrange[idx[0][0]]
    
    if log!=None:
        log.info('Minimum string length '+str(string_lengths.min())+' at '+repr(idx))
        log.info('String length x,y offsets: '+str(x_offset)+', '+str(y_offset))
    
    return x_offset, y_offset, idx

def fetch_min_string_xy(dxrange,dyrange,string_lengths,shortest):
    
    xdata = string_lengths.min(axis=1)
    ydata = string_lengths.min(axis=0)
    
    xdata = np.median(string_lengths,axis=1)
    ydata = np.median(string_lengths,axis=0)
    
    xdata = string_lengths[shortest[1][0],:]
    ydata = string_lengths[:,shortest[0][0]]
    
    return xdata, ydata
    
def plot_offsets(dxrange,dyrange,string_lengths,shortest,sub_pixel):
    
    fig = plt.figure(1,(10,10))
    
    plt.subplot(211)
    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.95,
                wspace=0.1, hspace=0.3)
    
    (xdata,ydata) = fetch_min_string_xy(dxrange,dyrange,string_lengths,shortest)
    
    plt.plot(dxrange,xdata,'k.')
    
    plt.xlabel('X pixel')
    plt.ylabel('Minimum string length')
    
    plt.subplot(212)
    plt.plot(dyrange,ydata,'k.')
    
    plt.xlabel('Y pixel')
    plt.ylabel('Minimum string length')
    
    if sub_pixel:
        plt.savefig('offset_string_lengths_subpixel.png')
    else:
        plt.savefig('offset_string_lengths.png')
        
    plt.close(1)
    
    fig = plt.figure(2,(10,10))
    
    plt.imshow(string_lengths)
    
    plt.colorbar()
    
    if sub_pixel:
        plt.savefig('string_lengths_subpixel.png')
    else:
        plt.savefig('string_lengths.png')
    
    plt.close(2)
    