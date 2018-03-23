# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 21:09:00 2018

@author: rstreet
"""

import os
import sys
cwd = os.getcwd()
sys.path.append(os.path.join(cwd,'../'))
import survey_catalog
import logs
from astropy.table import Table
import numpy as np
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
import astropy.units as u

TEST_DIR = os.path.join(cwd,'data','proc',
                        'ROME-FIELD-0002_lsc-doma-1m0-05-fl15_ip')
LOG_DIR = os.path.join(cwd,'data','proc','logs')

def test_read_star_catalog():
    """Function to test the read of a star catalog from a reduction's 
    metadata file"""
    
    log = logs.start_stage_log(LOG_DIR, 'test_survey_catalog')
    
    catalog = survey_catalog.read_star_catalog(TEST_DIR,log)
        
    t = Table(np.zeros(5))
    
    assert type(catalog) == type(t)
    
    logs.close_log(log)
    
def test_list_reducted_datasets():
    """Function to test the gathering of datasets to be combined into
    a single survey catalogue."""
    
    log = logs.start_stage_log(LOG_DIR, 'test_survey_catalog')
    
    params = {'data_dir': os.path.join(TEST_DIR,'..')}
    
    params = survey_catalog.list_reduced_datasets(params,log)    
    
    t_list = [ os.path.join(TEST_DIR,'..',os.path.basename(TEST_DIR)) ]
    
    assert 'datasets' in params.keys()
    assert params['datasets'] == t_list
    
    logs.close_log(log)

def test_xmatch_catalog():
    """Function to test the catalogue cross-matching function for the survey
    catalog builder.
    Based on code by Y. Tsapras.
    """
    
    log = logs.start_stage_log(LOG_DIR, 'test_survey_catalog')
    
    catalog1 = survey_catalog.read_star_catalog(TEST_DIR,log)
    
    catalog2 = survey_catalog.read_star_catalog(TEST_DIR,log)
    
    (matched_table,blends) = survey_catalog.xmatch_catalogs(catalog1, catalog2, log)

    assert len(matched_table[0]) > int(len(catalog1)*0.9)
    assert len(blends) > 0
    
    logs.close_log(log)

def test_merge_catalog():
    """Function to test the catalogue cross-matching function for the survey
    catalog builder.
    Based on code by Y. Tsapras.
    """
    
    log = logs.start_stage_log(LOG_DIR, 'test_survey_catalog')
    
    catalog1 = survey_catalog.read_star_catalog(TEST_DIR,log)
    
    nrows = len(catalog1)
    
    row = [999, 999.99999, 999.999999, 99.99999, 9.99999, 99.9999, 99.999, 
           99.9999, 99.9999, 99.9999, 99.999, 99.9999, 99.9999, 99.9999, 
           99.999, 99.9999]

    catalog1.add_row(row)
    
    catalog2 = survey_catalog.read_star_catalog(TEST_DIR,log)
        
    new_stars = range(0,len(catalog2),1)
    
    star_catalog = survey_catalog.add_new_stars_to_catalog(new_stars,catalog2,None,log)
    
    star_catalog = survey_catalog.merge_catalogs(catalog1, star_catalog, log)
    
    assert len(star_catalog) == nrows + 1
    
    logs.close_log(log)
 
def test_create_survey_catalog():
    """Function to test the creation of a survey catalog Table object"""
     
    log = logs.start_stage_log(LOG_DIR, 'test_survey_catalog')
    
    star_catalog = survey_catalog.create_survey_catalog(log,10)
                 
    test_table = Table()
    
    assert type(star_catalog) == type(test_table)
    assert 'ID' in star_catalog.colnames
    assert 'RA_J2000' in star_catalog.colnames
    assert 'DEC_J2000' in star_catalog.colnames
    assert 'Blend' in star_catalog.colnames

    logs.close_log(log)
    
def test_read_existing_survey_catalog():
    """Function to test the reading of a previously-existing star catalog"""

    log = logs.start_stage_log(LOG_DIR, 'test_survey_catalog')
    
    params = { 'old_star_catalog': 'NONE' }
    
    star_catalog = survey_catalog.read_existing_survey_catalog(params,log)

    assert star_catalog == None
    
    params = { 'old_star_catalog': os.path.join(TEST_DIR,'..','survey_star_catalog.fits') }
    
    star_catalog = survey_catalog.read_existing_survey_catalog(params,log)
    
    test_table = Table()
    
    assert type(star_catalog) == type(test_table)
    assert len(star_catalog) > 0
    
    logs.close_log(log)

def test_add_new_stars_to_catalog():
    """Function to test the addition of new stars to an existing master catalogue."""

    log = logs.start_stage_log(LOG_DIR, 'test_survey_catalog')
    
    params = { 'old_star_catalog': os.path.join(TEST_DIR,'..','survey_star_catalog.fits') }
    
    star_catalog = survey_catalog.read_existing_survey_catalog(params,log)

    nrows = len(star_catalog)
    
    new_catalog = survey_catalog.read_star_catalog(TEST_DIR,log)
    
    new_stars = set([1,2,3])
    
    star_catalog = survey_catalog.add_new_stars_to_catalog(new_stars,new_catalog,star_catalog,log)
    
    assert len(star_catalog) == nrows+3
    
    logs.close_log(log)
    
if __name__ == '__main__':
    
    test_read_star_catalog()
    test_list_reducted_datasets()
    test_xmatch_catalog()
    test_merge_catalog()
    test_create_survey_catalog()
    test_read_existing_survey_catalog()
    