# -*- coding: utf-8 -*-
"""
Created on Thu May 30 14:30:19 2019

@author: rstreet
"""

from sys import argv
import sqlite3
from os import getcwd, path, remove, environ
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units
import matplotlib.pyplot as plt
from pyDANDIA import  phot_db

def extract_star_lightcurves_on_position(params):
    """Function to extract a lightcurve from a phot_db"""
    
    conn = phot_db.get_connection(dsn=params['db_file_path'])
    
    facilities = phot_db.fetch_facilities(conn)
    filters = phot_db.fetch_filters(conn)
    code_id = phot_db.get_stage_software_id(conn,'stage6')
    
    c = SkyCoord(params['ra'], params['dec'], frame='icrs', unit=(units.hourangle, units.deg))
    
    if 'radius' in params.keys():
        radius = float(params['radius'])
    else:
        radius = 2.0 / 3600.0
    
    results = phot_db.box_search_on_position(conn, c.ra.deg, c.dec.deg, radius, radius)
    
    if len(results) > 0:
        
        star_idx = np.where(results['separation'] == results['separation'].min())
    
        query = 'SELECT filter, facility, hjd, calibrated_mag, calibrated_mag_err FROM phot WHERE star_id="'+str(results['star_id'][star_idx][0])+\
                    '" AND software="'+str(code_id)+'"'
                    
        phot_table = phot_db.query_to_astropy_table(conn, query, args=())
        
        
        datasets = identify_unique_datasets(phot_table,facilities,filters)
        
        for setname,setlist in datasets.items():
            
            datafile = open(path.join(params['output_dir'],'star_'+str(results['star_id'][star_idx][0])+'_'+setname+'.dat'),'w')
            
            for j in setlist[2]:
                
                datafile.write(str(phot_table['hjd'][j])+'  '+str(phot_table['calibrated_mag'][j])+'  '+str(phot_table['calibrated_mag_err'][j])+'\n')
            
            datafile.close()
            
        message = 'OK'
        
    else:
        message = 'No stars within search region'
        
    conn.close()

    return message

def identify_unique_datasets(phot_table,facilities,filters):
    """Function to extract a list of the unique datasets from a table of 
    photometry, i.e. the list of unique combinations of facility and filter
    """
    
    datasets = {}
    
    for j in range(0,len(phot_table),1):
        
        i = np.where(facilities['facility_id'] == phot_table['facility'][j])
        facility_code = facilities['facility_code'][i][0]
        
        i = np.where(filters['filter_id'] == phot_table['filter'][j])
        fcode = filters['filter_name'][i][0]
        
        dataset_code = str(facility_code)+'_'+str(fcode)
        
        if dataset_code in datasets.keys():
            
            setlist = datasets[dataset_code]
            
        else:
            
            setlist = [ phot_table['facility'][j], 
                        phot_table['filter'][j],
                        [] ]
        
        setlist[2].append(j)
        
        datasets[dataset_code] =  setlist
    
    return datasets
    
if __name__ == '__main__':
    
    params = {}
    
    if len(argv) == 1:
        
        params['db_file_path'] = input('Please enter the path to the field photometric DB: ')
        params['ra'] = input('Please enter the RA [sexigesimal]: ')
        params['dec'] = input('Please enter the Dec [sexigesimal]: ')
        params['output_dir'] = input('Please enter the path to the output directory: ')
        
    else:
        
        params['db_file_path'] = argv[1]
        params['ra'] = argv[2]
        params['dec'] = argv[3]
        params['output_dir'] = argv[4]
    
    message = extract_star_lightcurves_on_position(params)
    