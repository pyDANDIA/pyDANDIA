# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 20:44:24 2017

@author: rstreet
"""

from os import getcwd, path, remove
from sys import exit
from sys import path as systempath
cwd = getcwd()
systempath.append(path.join(cwd,'../'))
import numpy as np
import logs
import psf_selection
import random
import pipeline_setup
import metadata


cwd = getcwd()
TEST_DIR = path.join(cwd,'data','proc','ROME-FIELD-0002_lsc-doma-1m0-05-fl15_ip')


def test_id_mid_range_stars():
    """Function to test the selection of stars in the reference image, excluding
    the brightest and faintest N% of those detected"""
    
    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DIR})
    
    log = logs.start_stage_log( cwd, 'test_psf_selection' )
    
    reduction_metadata = metadata.MetaData()
    reduction_metadata.load_a_layer_from_file(setup.red_dir, 
                                              'pyDANDIA_metadata.fits', 
                                              'reduction_parameters')
    log.info('Read metadata')
    
    # Generating test catalog with columns:
    # idx x  y  ra  dec  inst_mag inst_mag_err J  Jerr  H Herr   K   Kerr
    nstars = 10
    bright = 16.0
    faint = 23.0
    
    ref_star_catalog = np.zeros([nstars,13])
    ref_star_catalog[:,0] = range(0,nstars,1)
    ref_star_catalog[:,1] = random.normalvariate(100.0,100.0)
    ref_star_catalog[:,2] = random.normalvariate(100.0,100.0)
    ref_star_catalog[:,3] = random.normalvariate(17.0*15.0,20.0)
    ref_star_catalog[:,4] = random.normalvariate(-27.0,10.0)
    ref_star_catalog[:,5] = np.arange(bright,faint,(faint-bright)/float(nstars))
    ref_star_catalog[:,6] = 0.005 + ref_star_catalog[:,4]*0.05
    
    psf_stars_idx = np.array([1]*nstars)
    
    psf_range_thresh = reduction_metadata.reduction_parameters[1]['PSF_RANGE_THRESH'][0]
    
    log.info('Read psf range threshold = '+str(psf_range_thresh))
    
    nstar_cut = int(float(nstars) * (psf_range_thresh/100.0))
    istart = nstar_cut
    iend = len(ref_star_catalog) - nstar_cut
    
    test_psf_stars_idx = np.ones(nstars)
    test_psf_stars_idx[0:nstar_cut] = 0
    test_psf_stars_idx[(-1*nstar_cut):] = 0
    
    psf_stars_idx = psf_selection.id_mid_range_stars(setup,reduction_metadata,
                                                     log,
                                                     ref_star_catalog,
                                                     psf_stars_idx)
    
    assert psf_stars_idx.all() == test_psf_stars_idx.all()
    
    logs.close_log(log)


def test_id_crowded_stars():
    """Function to test the exclusion of crowded stars from the PSF star 
    selection process"""
    
    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DIR})
    
    log = logs.start_stage_log( cwd, 'test_psf_selection' )
    
    reduction_metadata = metadata.MetaData()
    reduction_metadata.load_a_layer_from_file(setup.red_dir, 
                                              'pyDANDIA_metadata.fits', 
                                              'reduction_parameters')
    log.info('Read metadata')
    
    nstars = 10
    bright = 16.0
    faint = 23.0
    
    ref_star_catalog = np.zeros([nstars,13])
    ref_star_catalog[:,0] = range(0,nstars,1)
    
    istar = -1
    for j in range(0,7,1):
        istar += 1
        ref_star_catalog[istar,1] = abs(random.normalvariate(100.0,100.0))
        ref_star_catalog[istar,2] = abs(random.normalvariate(100.0,100.0))
        ref_star_catalog[istar,5] = abs(random.normalvariate(16.0,0.5))
    
    exclude = []
    
    for j in range(istar,nstars,1):
        i = random.randint(0,2)
        xstar = ref_star_catalog[i,1]
        ystar = ref_star_catalog[i,2]
        ref_star_catalog[j,1] = random.normalvariate(xstar,1.0)
        ref_star_catalog[j,2] = random.normalvariate(ystar,1.0)
        ref_star_catalog[j,5] = abs(random.normalvariate(17.0,0.5))
        
        exclude.append(i)
    
    ref_star_catalog[:,3] = random.normalvariate(17.0*15.0,20.0)
    ref_star_catalog[:,4] = random.normalvariate(-27.0,10.0)
    ref_star_catalog[:,6] = 0.005 + ref_star_catalog[:,4]*0.05
    
    psf_stars_idx = np.array([1]*nstars)
    
    psf_stars_idx = psf_selection.id_crowded_stars(setup,reduction_metadata,log,
                                    ref_star_catalog,psf_stars_idx)

    psf_selection.plot_ref_star_catalog_positions(setup,reduction_metadata,log,
                                    ref_star_catalog, psf_stars_idx)
    
    star_index = np.where(psf_stars_idx == 1)[0]
    
    for j in exclude:
        
        assert j not in star_index
    
    logs.close_log(log)

def test_apply_psf_star_limit():
    
    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DIR})
    
    log = logs.start_stage_log( cwd, 'test_psf_selection' )
    
    reduction_metadata = metadata.MetaData()
    reduction_metadata.load_a_layer_from_file(setup.red_dir, 
                                              'pyDANDIA_metadata.fits', 
                                              'reduction_parameters')
    max_psf_stars = reduction_metadata.reduction_parameters[1]['MAX_PSF_STARS'][0]
    
    nstars = 4000
    bright = 16.0
    faint = 23.0
    
    ref_star_catalog = np.zeros([nstars,13])
    ref_star_catalog[:,0] = range(0,nstars,1)
    
    istar = -1
    for j in range(0,nstars,1):
        istar += 1
        ref_star_catalog[istar,1] = abs(random.normalvariate(100.0,100.0))
        ref_star_catalog[istar,2] = abs(random.normalvariate(100.0,100.0))
        ref_star_catalog[istar,7] = abs(random.normalvariate(16.0,0.5))
    
    psf_stars_idx = np.array([1]*nstars)
    
    psf_stars_idx = psf_selection.apply_psf_star_limit(reduction_metadata,ref_star_catalog,
                                         psf_stars_idx,log)
    
    idx = np.where(psf_stars_idx)[0]
    
    assert len(idx) == max_psf_stars
    
    logs.close_log(log)
    
if __name__ == '__main__':
    
    #test_id_crowded_stars()
    test_apply_psf_star_limit()