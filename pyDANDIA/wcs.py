# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 17:05:56 2017

@author: rstreet
"""
from os import path, getcwd
from sys import exit
import logs
import metadata
from astropy.io import fits
from astropy import wcs, coordinates, units
import subprocess
from astroquery.vizier import Vizier

def fit_wcs(meta,log,image_path,ref_catalog_path,source_catalog_path):
    """Function to calculate the World Coordinate System (WCS) for an image"""
    
    hdu = fits.open(image_path)
    
    image_wcs = wcs.WCS(hdu[0].header)
    
    
    # Obtain a list of objects within the current field of view
    
    # Run IMWCS
    
    
    
def wcs_summary(image_wcs):
    """Function to output a summary of the relevant WCS parameters from an 
    image header"""
    
    print('CRPIX: '+str(image_wcs.wcs.crpix))
    print('CD: '+str(image_wcs.wcs.cd))
    print('CRVAL: '+str(image_wcs.wcs.crval))
    print('CTYPE: '+str(image_wcs.wcs.ctype))
    

def run_imwcs(detected_sources,catalog_sources,input_image_path,output_image_path):
    """Function to run wcstools.imwcs to match detected to catalog objects and
    re-compute the WCS for an image"""
    
    n_wcs_stars = 200
    n_param = 6
    
    command = 'imwcs'
    args = ['-wvd '+detected_sources+\
                ' -c tab '+catalog_source+\
                ' -h '+str(n_wcs_stars)+\
                ' -n '+str(n_param)+\
                ' -q irs '+input_image_path+\
                ' -o '+output_image_path]
    
    p = subprocess.Popen([command,args], stdout=subprocess.PIPE)
    p.wait()
    
def get_catalog_objects_in_fov(ra, dec, radius):
    """Function to perform online query of the 2MASS catalogue and return
    a catalogue of known objects within the field of view"""
    
    v = Vizier(columns=['_RAJ2000', '_DEJ2000', 'Jmag', 'Hmag', 'Kmag'],column_filters={'Jmag':'<20'})
    c = coordinates.SkyCoord(ra+' '+dec, frame='icrs', unit=(units.hourangle, units.deg))
    r = radius * units.arcminute
    result=v.query_region(c,radius=r,catalog=['2MASS'])
    
    return result[0]
    
    