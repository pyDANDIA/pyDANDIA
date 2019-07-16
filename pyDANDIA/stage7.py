# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 10:53:23 2019

@author: rstreet
"""
import numpy as np
import os
import sys
from astropy.io import fits
from astropy.table import Table
from astropy.table import Column
from pyDANDIA import metadata
from pyDANDIA import logs
from pyDANDIA import phot_db


VERSION = 'pyDANDIA_stage7_v0.1'

def run_stage7(setup):
    """Driver function for pyDANDIA Stage 7: 
    Merging datasets from multiple telescopes and sites.
    """
    
    log = logs.start_stage_log(setup.red_dir, 'stage7', version=VERSION)
    log.info('Setup:\n' + setup.summary() + '\n')
    
    # Setup the DB connection and record dataset and software parameters
    conn = phot_db.get_connection(dsn=setup.phot_db_path)

    primary_ref = identify_primary_reference_datasets(conn, log)
    
    for f in ['ip', 'rp', 'gp']:
        
        if 'refimg_id_'+f in primary_ref.keys():
            primary_phot = extract_photometry_for_reference_image(conn,primary_ref,f,
                                                              primary_ref['refimg_id_'+f],
                                                              log)
        
            datasets = list_datasets_in_filter(conn,primary_ref,f,log)
        
        else:
            
            log.info('No primary photometry available in filter '+f)
            
    conn.close()
    logs.close_log(log)

    status = 'OK'
    report = 'Completed successfully'

    return status, report

def identify_primary_reference_datasets(conn, log):
    """Function to extract the parameters of the primary reference dataset and
    instrument for all wavelengths for the current field."""
    
    primary_ref = {}
    
    primary_ref['refimg_id_ip'] = phot_db.find_primary_reference_image_for_field(conn)
    
    query = 'SELECT facility, filter, software FROM reference_images WHERE refimg_id="'+str(primary_ref['refimg_id_ip'])+'"'
    t = phot_db.query_to_astropy_table(conn, query, args=())
    
    primary_ref['facility_id'] = t['facility'][0]
    primary_ref['software_id'] = t['software'][0]
    
    query = 'SELECT filter_id, filter_name FROM filters WHERE filter_name="ip"'
    t = phot_db.query_to_astropy_table(conn, query, args=())
    primary_ref['ip'] = t['filter_id'][0]
    
    for f in ['rp', 'gp']:
        query = 'SELECT filter_id, filter_name FROM filters WHERE filter_name="'+f+'"'
        t = phot_db.query_to_astropy_table(conn, query, args=())
        primary_ref[f] = t['filter_id'][0]
        
        query = 'SELECT refimg_id FROM reference_images WHERE facility="'+str(primary_ref['facility_id'])+\
                    '" AND software="'+str(primary_ref['software_id'])+\
                    '" AND filter="'+str(t['filter_id'][0])+'"'
        qs = phot_db.query_to_astropy_table(conn, query, args=())

        if len(qs) > 0:
            primary_ref['refimg_id_'+f] = qs['refimg_id'][0]
        else:
            log.info('WARNING: Database contains no primary reference image data in filter '+f)
    
    log.info('Identified the primary reference datasets for this field as:')
    for key, value in primary_ref.items():
        log.info(str(key)+' = '+str(value))
        
    return primary_ref

def extract_photometry_for_reference_image(conn,primary_ref,f,refimg_id,log):
    """Function to extract from the DB the photometry for the primary 
    reference dataset in the given filter"""
    
    query = 'SELECT phot_id, hjd, calibrated_mag, calibrated_mag_err, calibrated_flux, calibrated_flux_err FROM phot WHERE reference_image="'+str(refimg_id)+\
                '" AND software="'+str(primary_ref['software_id'])+\
                '" AND filter="'+str(primary_ref[f])+\
                '" AND facility="'+str(primary_ref['facility_id'])+'"'
    phot = phot_db.query_to_astropy_table(conn, query, args=())
    
    log.info(' -> Extracted photometry for '+str(len(phot))+' stars for reference image '+str(refimg_id)+' and filter '+f)
    
    return primary_phot

def list_datasets_in_filter(conn,primary_ref,f,log):
    """Function to identify all available datasets in the given 
    filter for this field, not including the primary reference dataset.
    
    Returns:
        QuerySet of matching reference_image entries
    """
    
    query = 'SELECT * FROM reference_images WHERE filter="'+str(primary_ref[f])+\
            '" AND software="'+str(primary_ref['software_id'])+\
            '" AND facility!="'+str(primary_ref['facility_id'])+'"'
    
    datasets = phot_db.query_to_astropy_table(conn, query, args=())
    print(datasets)
    
    return datasets
    