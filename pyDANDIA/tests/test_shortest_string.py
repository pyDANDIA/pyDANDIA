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

def generate_test_catalogs():
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
    
def test_find_xy_offset():
    
    log = logs.start_stage_log( cwd, 'test_wcs' )
    
    (catalog1, catalog2, x_offset_sim, y_offset_sim) = generate_test_catalogs()
    
    (x_offset, y_offset) = shortest_string.find_xy_offset(catalog1, catalog2,
                                                          diagnostics=True)
    
    assert round(x_offset,1) == round(x_offset_sim,1)
    assert round(y_offset,1) == round(y_offset_sim,1)

    logs.close_log(log)

if __name__ == '__main__':
    
    test_find_xy_offset()
    