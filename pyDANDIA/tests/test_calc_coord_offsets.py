# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 09:29:51 2019

@author: rstreet
"""

from os import getcwd, path, remove
from sys import path as systempath
cwd = getcwd()
systempath.append(path.join(cwd,'../'))
import calc_coord_offsets
from astropy.table import Table, Column
import numpy as np

def test_calc_offset_hist2d():
    
    ra_cen = 275.0
    dec_cen = -27.5
    
    ras = np.linspace(270.0,280.0,100)
    decs = np.linspace(-30.0,-25.0,100)
    
    coord_data = [ Column(name='ra', data=ras), Column(name='dec', data=decs) ]
    detected_sources_world = Table(data=coord_data)
    
    ra_offset = 0.2 # deg
    dec_offset = 0.05 # deg
    
    ras = ras + ra_offset
    decs = decs + dec_offset
    
    coord_data = [ Column(name='ra', data=ras), Column(name='dec', data=decs) ]
    catalog_sources_world = Table(data=coord_data)
    
    (dra,ddec) = calc_coord_offsets.calc_offset_hist2d(detected_sources_world, 
                                                        catalog_sources_world,
                                                        ra_cen,dec_cen)
    print(dra,ddec)
    assert round(dra,1) == -ra_offset
    assert round(ddec,1) == -dec_offset

def test_extract_nearby_stars():
    
    ra_cen = 275.0
    dec_cen = -27.5
    radius = 1.0
    
    ras = np.linspace(270.0,280.0,100)
    decs = np.linspace(-30.0,-25.0,100)
    
    coord_data = [ Column(name='ra', data=ras), Column(name='dec', data=decs) ]
    catalog = Table(data=coord_data)
    
    sub_catalog = calc_coord_offsets.extract_nearby_stars(catalog,ra_cen,dec_cen,radius)
    
    assert type(sub_catalog) == type(catalog)
    assert sub_catalog['ra'].max() <= ra_cen+radius
    assert sub_catalog['ra'].min() >= ra_cen-radius
    assert sub_catalog['dec'].max() <= dec_cen+radius
    assert sub_catalog['dec'].min() >= dec_cen-radius
    assert len(sub_catalog) <= len(catalog)
    
if __name__ == '__main__':
    
    test_calc_offset_hist2d()
 #   test_extract_nearby_stars()
    