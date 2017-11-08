# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 10:28:35 2017

@author: rstreet
"""

import os
import sys
cwd = os.getcwd()
sys.path.append(os.path.join(cwd,'../'))
import catalog_utils
import numpy as np

def add_column():
    
    ref_star_catalog_file = os.path.join(cwd,'data','star_catalog.fits')
    
    ref_star_catalog = catalog_utils.read_ref_star_catalog_file(ref_star_catalog_file)

    psf_idx = np.zeros(len(ref_star_catalog))
    
    ref_star_catalog = np.insert(ref_star_catalog,[13],psf_idx,axis=1)
    
    ref_star_catalog_file = ref_star_catalog_file.replace('.fits','_new.fits')
    
    catalog_utils.output_ref_catalog_file(ref_star_catalog_file,ref_star_catalog)
    

if __name__ == '__main__':
    
    add_column()