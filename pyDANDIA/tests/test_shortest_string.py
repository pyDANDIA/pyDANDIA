# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 16:27:34 2019

@author: rstreet
"""

from os import getcwd, path, remove
from sys import path as systempath
cwd = getcwd()
systempath.append(path.join(cwd,'../'))
import shortest_string
import numpy as np
import matplotlib.pyplot as plt
import logs
from astropy.table import Table, Column

def generate_test_catalogs_pixels():
    """Function to produce some test catalogs of pixel positions 
    for cross-matching"""
    
    n_cat1 = 100
    n_cat2 = 200
    
    x_offset_sim = 17.6
    y_offset_sim = -25.2
    
    catalog1 = np.zeros([n_cat1,2])
    catalog2 = np.zeros([n_cat2,2])
    
    catalog1[:,0] = np.random.normal(loc=50.0, scale=50.0,size=n_cat1)
    catalog1[:,1] = np.random.normal(loc=50.0, scale=50.0,size=n_cat1)
    catalog1 = abs(catalog1)
    
    catalog2 = np.zeros(catalog1.shape)
    catalog2[:,0] = catalog1[:,0] + x_offset_sim
    catalog2[:,1] = catalog1[:,1] + y_offset_sim

    fig = plt.figure(1,(10,10))
    
    plt.plot(catalog1[:,0],catalog1[:,1],'k.')
    plt.plot(catalog2[:,0],catalog2[:,1],'r.')
    
    plt.xlabel('X pixel')
    plt.ylabel('Y pixel')
    
    plt.savefig('sim_catalog_image.png')
    
    plt.close(1)
    
    return catalog1, catalog2, x_offset_sim, y_offset_sim

def generate_test_catalog_on_sky():
    """Function to produce some test catalogs of pixel positions 
    for cross-matching"""
    
    n_cat1 = 100
    n_cat2 = 200
    
    x_offset_sim = 17.6/60.0
    y_offset_sim = -25.2/60.0
    
    coord_data = [ Column(name='ra', data=np.random.normal(loc=260.0, scale=0.5,size=n_cat1), 
                   Column(name='dec', data=np.random.normal(loc=-27.0, scale=0.5,size=n_cat1) ]
    catalog1 = Table(data=coord_data)
    
    coord_data = [ Column(name='ra', data=(catalog1['ra'].data + x_offset_sim), 
                   Column(name='dec', data=(catalog1['dec'].data + y_offset_sim) ]
    catalog2 = Table(data=coord_data)
    
    fig = plt.figure(1,(10,10))
    
    plt.plot(catalog1['ra'].data,catalog1['dec'].data,'k.')
    plt.plot(catalog2['ra'].data,catalog2['dec'].data,'r.')
    
    plt.xlabel('RA [deg]')
    plt.ylabel('Dec [deg]')
    
    plt.savefig('sim_catalog_image.png')
    
    plt.close(1)
    
    return catalog1, catalog2, x_offset_sim, y_offset_sim
    
def test_find_xy_offset():
    
    log = logs.start_stage_log( cwd, 'test_shortest_string' )
    
    (catalog1, catalog2, x_offset_sim, y_offset_sim) = generate_test_catalogs_pixels()
    
    (x_offset, y_offset) = shortest_string.find_xy_offset(catalog1, catalog2,
                                                          log=log,
                                                          diagnostics=True)
    
    assert round(x_offset,1) == round(x_offset_sim,1)
    assert round(y_offset,1) == round(y_offset_sim,1)

    logs.close_log(log)

def test_find_on_sky_offset():
    
    log = logs.start_stage_log( cwd, 'test_shortest_string' )
    
    (catalog1, catalog2, x_offset_sim, y_offset_sim) = generate_test_catalogs_pixels()
    
    (x_offset, y_offset) = shortest_string.find_sky_offset(catalog1, catalog2,
                                                          log=log,
                                                          diagnostics=True)
    
    assert round(x_offset,1) == round(x_offset_sim,1)
    assert round(y_offset,1) == round(y_offset_sim,1)

    logs.close_log(log)


if __name__ == '__main__':
    
    test_find_xy_offset()
    