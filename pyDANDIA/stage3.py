# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 12:35:17 2017

@author: rstreet
"""
import os
import sys
from astropy.io import fits
from astropy import table
import numpy as np
from pyDANDIA import  logs
from pyDANDIA import  metadata
from pyDANDIA import  starfind
from pyDANDIA import  pipeline_setup
from pyDANDIA import  sky_background
from pyDANDIA import  wcs
from pyDANDIA import  psf
from pyDANDIA import  psf_selection
from pyDANDIA import  photometry
from pyDANDIA import  phot_db
from pyDANDIA import  utilities

VERSION = 'pyDANDIA_stage3_v0.3'

def run_stage3(setup):
    """Driver function for pyDANDIA Stage 3: 
    Detailed star find and PSF modeling
    """
        
    log = logs.start_stage_log( setup.red_dir, 'stage3', version=VERSION )
    
    reduction_metadata = metadata.MetaData()
    reduction_metadata.load_a_layer_from_file( setup.red_dir, 
                                              'pyDANDIA_metadata.fits', 
                                              'reduction_parameters' )
    reduction_metadata.load_a_layer_from_file( setup.red_dir, 
                                              'pyDANDIA_metadata.fits', 
                                              'headers_summary' )
    reduction_metadata.load_a_layer_from_file( setup.red_dir, 
                                              'pyDANDIA_metadata.fits', 
                                              'images_stats' )
    reduction_metadata.load_a_layer_from_file( setup.red_dir, 
                                              'pyDANDIA_metadata.fits', 
                                              'data_architecture' )
    reduction_metadata.load_a_layer_from_file( setup.red_dir, 
                                              'pyDANDIA_metadata.fits', 
                                              'headers_summary' )
    
    sane = check_metadata(reduction_metadata,log)
    
    if sane: 
    
        meta_pars = extract_parameters_stage3(reduction_metadata,log)
    
        sane = sanity_checks(reduction_metadata,log,meta_pars)

    if sane:
        
        scidata = fits.getdata(meta_pars['ref_image_path'])
        
        detected_sources = starfind.detect_sources(setup, reduction_metadata,
                                        meta_pars['ref_image_path'],
                                        (scidata-meta_pars['ref_sky_bkgd']),
                                        log,
                                        diagnostics=False)
        
        ref_flux = find_reference_flux(detected_sources,log)
        
        ref_star_catalog = wcs.reference_astrometry(setup,log,
                                        meta_pars['ref_image_path'],
                                        detected_sources,
                                        diagnostics=False)        
                                                    
        sky_model = sky_background.model_sky_background(setup,
                                        reduction_metadata,log,ref_star_catalog)
                
        ref_star_catalog = psf_selection.psf_star_selection(setup,
                                        reduction_metadata,
                                        log,ref_star_catalog,
                                        diagnostics=False)
                                                     
        reduction_metadata.create_star_catalog_layer(ref_star_catalog,log=log)
        
        
        psf_size = round( (meta_pars['ref_fwhm'] * 2.0), 0 )
        log.info('Calculated size of PSF = '+str(psf_size)+'pix')
        
        (psf_model,psf_status) = psf.build_psf(setup, reduction_metadata, 
                                            log, scidata, 
                                            ref_star_catalog, sky_model,
                                            psf_size,
                                            diagnostics=False)
        
        ref_star_catalog = photometry.run_psf_photometry(setup, 
                                             reduction_metadata, 
                                             log, 
                                             ref_star_catalog,
                                             meta_pars['ref_image_path'],
                                             psf_model,
                                             sky_model,
                                             ref_flux,
                                             psf_size=psf_size,
                                             centroiding=False)
                                             
        reduction_metadata.create_star_catalog_layer(ref_star_catalog,log=log)
        
        reduction_metadata.save_a_layer_to_file(setup.red_dir, 
                                                'pyDANDIA_metadata.fits',
                                                'star_catalog', log=log)
        
        #ref_db_id = add_reference_image_to_db(setup, reduction_metadata, log=log)
        
        #star_ids = ingest_stars_to_db(setup, ref_star_catalog, 
        #                              meta_pars['ref_image_name'], log=log)

        #ingest_star_catalog_to_db(setup, ref_star_catalog, ref_db_id, star_ids,
        #                          log=log)
        status = 'OK'
        report = 'Completed successfully'
        
    else:
        status = 'ERROR'
        report = 'Failed sanity checks'
        
    log.info('Stage 3: '+report)
    logs.close_log(log)
    
    return status, report

def check_metadata(reduction_metadata,log):
    """Function to verify sufficient information has been extracted from 
    the metadata
    
    :param MetaData reduction_metadata: pipeline metadata for this dataset
    
    Returns:
    
    :param boolean status: Status parameter indicating if conditions are OK 
                            to continue the stage.
    """

    if 'REF_PATH' not in reduction_metadata.data_architecture[1].keys():
        
        log.info('ERROR: Stage 3 cannot find path to reference image in metadata')
        
        return False
        
    else:
        
        return True
        
def sanity_checks(reduction_metadata,log,meta_pars):
    """Function to check that stage 3 has all the information that it needs 
    from the reduction metadata and reduction products from earlier stages 
    before continuing.
    
    :param MetaData reduction_metadata: pipeline metadata for this dataset
    :param logging log: Open reduction log object
    :param dict meta_pars: Essential parameters from the metadata
    
    Returns:
    
    :param boolean status: Status parameter indicating if conditions are OK 
                            to continue the stage.
    """

    ref_path =  str(reduction_metadata.data_architecture[1]['REF_PATH'][0]) +'/'+ str(reduction_metadata.data_architecture[1]['REF_IMAGE'][0])
    if not os.path.isfile(ref_path):
        # Message reduction_control?  Return error code?
        log.info('ERROR: Stage 3 cannot access reference image at '+\
                        ref_path)
        return False

    for key, value in meta_pars.items():
        
        if value == None:
            
            log.info('ERROR: Stage 3 cannot access essential metadata parameter '+key)
            
            return False
    
    log.info('Passed stage 3 sanity checks')
    
    return True

def extract_parameters_stage3(reduction_metadata,log):
    """Function to extract the metadata parameters necessary for this stage.
    
    :param MetaData reduction_metadata: pipeline metadata for this dataset

    Returns:

    :param dict meta_params: Dictionary of parameters and their values    
    """
    
    meta_pars = {}
    
    try:
        
        meta_pars['ref_image_path'] = str(reduction_metadata.data_architecture[1]['REF_PATH'][0]) +'/'+ str(reduction_metadata.data_architecture[1]['REF_IMAGE'][0])
        meta_pars['ref_image_name'] = str(reduction_metadata.data_architecture[1]['REF_IMAGE'][0])
        
    except AttributeError:
        
        meta_pars['ref_image_path'] = None
        meta_pars['ref_image_name'] = None
    
    idx = np.where(reduction_metadata.images_stats[1]['IM_NAME'].data ==  reduction_metadata.data_architecture[1]['REF_IMAGE'][0])

    if len(idx[0]) > 0:
    
        meta_pars['ref_sky_bkgd'] = reduction_metadata.images_stats[1]['SKY'].data[idx[0][0]]
        
        fwhmx = reduction_metadata.images_stats[1]['FWHM_X'].data[idx[0][0]]
        fwhmy = reduction_metadata.images_stats[1]['FWHM_Y'].data[idx[0][0]]
        pixscale = reduction_metadata.reduction_parameters[1]['PIX_SCALE'][0]
        
        meta_pars['ref_fwhm'] = np.sqrt( fwhmx*fwhmx + fwhmy*fwhmy ) / pixscale
        
        meta_pars['bandpass'] = str(reduction_metadata.headers_summary[1]['FILTKEY'].data[idx[0]][0]).replace('p','')
        
    else:
        
        meta_pars['ref_sky_bkgd'] = None

	
        
    try:
        
        meta_pars['sky_model_type'] = reduction_metadata.reduction_parameters[1]['BACK_VAR'][0]

    except AttributeError:
        
        meta_pars['sky_model_type'] = 'constant'
    
    if len(meta_pars) > 0:
        log.info('Extracted metadata parameters:')
        for key, value in meta_pars.items():
            log.info(key+' = '+str(value))
    
    return meta_pars
    
def find_reference_flux(detected_sources,log):
    """Function to identify the faintest star in the detected sources
    catalogue.  This will be used as the reference flux in the calculation
    of optimized photometry, following the method of Naylor (1997)
    
    Input:
    :param array detected_sources: Array of sources in the frame produced by
                                    DAOfind
    
    Output:
    :param float ref_flux: Reference flux
    """
    
    idx = np.where(detected_sources[:,9] > 0.0)
    
    idx = np.where( detected_sources[idx,9] == detected_sources[idx,9].min() )
    
    ref_flux = detected_sources[idx[0][0],9]
    
    log.info('Measured reference flux of faintest star (used for normalization) = '+\
                    str(ref_flux))
    
    return ref_flux

def add_reference_image_to_db(setup, reduction_metadata, log=None):
    """Function to ingest the reference image and corresponding star catalogue 
    to the photometry DB"""
    
    conn = phot_db.get_connection(dsn=setup.phot_db_path)
    
    ref_image_name = reduction_metadata.data_architecture[1]['REF_IMAGE'].data[0]
    ref_image_dir = reduction_metadata.data_architecture[1]['REF_PATH'].data[0]
    
    idx = np.where(ref_image_name == reduction_metadata.headers_summary[1]['IMAGES'].data)[0][0]
    ref_header = reduction_metadata.headers_summary[1][idx]
    
    query = 'SELECT refimg_name FROM reference_images'
    t = phot_db.query_to_astropy_table(conn, query, args=())
    
    if len(t) == 0 or ref_image_name not in t[0]['refimg_name']:
        
        phot_db.ingest_reference_in_db(conn, setup, ref_header, 
                                   ref_image_dir, ref_image_name,
                                   ref_header['OBJKEY'], VERSION)
        
        query = 'SELECT refimg_id,refimg_name,current_best FROM reference_images WHERE refimg_name="'+ref_image_name+'"'
        t = phot_db.query_to_astropy_table(conn, query, args=())
        
        ref_db_id = t[0]['refimg_id']
        
        if log!=None:
            log.info('Added reference image to phot_db as entry '+str(ref_db_id))
        
    else:
        
        query = 'SELECT refimg_id,refimg_name FROM reference_images WHERE refimg_name="'+ref_image_name+'"'
        t = phot_db.query_to_astropy_table(conn, query, args=())
        
        ref_db_id = t[0]['refimg_id']

        if log!=None:
            log.info('Reference image already in phot_db')

    return ref_db_id
    
def ingest_stars_to_db(setup, ref_star_catalog, ref_image_name, log=None):
    """Function to ingest the detected stars to the photometry database"""
    
    conn = phot_db.get_connection(dsn=setup.phot_db_path)
    
    query = 'SELECT refimg_id FROM reference_images WHERE refimg_name="'+\
                ref_image_name+'"'
    t = phot_db.query_to_astropy_table(conn, query, args=())

    refimg_id = t['refimg_id'].data[0]
    
    if log!=None:
        log.info('Ingesting stars detected in reference image '+ref_image_name+', refimg_id='+str(refimg_id))
        log.info('Length of ref_star_catalog = '+str(len(ref_star_catalog)))
    
    star_ids = []
    
    for j in range(0,len(ref_star_catalog),1):
        
        star_ids.append( phot_db.feed_to_table( conn, 'Stars', ['ra','dec'], 
                         [ref_star_catalog[j,3], ref_star_catalog[j,4]] ) )
    
        phot_db.update_table_entry(conn,'stars','reference_images',
                                   'star_id',star_ids[-1],refimg_id)
    
    conn.close()
    
    return star_ids
    
def ingest_star_catalog_to_db(setup, ref_star_catalog, ref_db_id, star_ids,
                              log=None):
    """Function to ingest the reference image photometry data for all stars
    detected in this image.
    
    Important: This function only ingests the photometry for a single bandpass,
                and only for instrumental rather than calibrated data, since
                the latter is produced by a different stage of the pipeline.
    
    :param Setup setup: Pipeline setup configuration
    :param np.array ref_star_catalog: Reference image photometry data
    :param int ref_db_id: Primary key of the reference image in the phot_db
    :param list star_ids: List of the DB primary keys of stars in the catalog
    :param Log log: Logger object [optional]
    """
    
    conn = phot_db.get_connection(dsn=setup.phot_db_path)
   
    ref_ids = np.zeros(len(ref_star_catalog), dtype='int')
    ref_ids.fill(ref_db_id)
    
    keys = ['reference_mag', 'reference_mag_err', 
            'reference_flux', 'reference_flux_err', 
            'reference_x', 'reference_y']
    
    ref_phot_ids = []
    
    for j in range(0,len(ref_star_catalog),1):
        
        values = [ ref_star_catalog[j,7], ref_star_catalog[j,8],
                   ref_star_catalog[j,5], ref_star_catalog[j,6],
                   ref_star_catalog[j,1], ref_star_catalog[j,2] ]
                   
        ref_phot_ids.append( phot_db.feed_to_table( conn, 'ref_phot', 
                                                keys, values ) )
    
        phot_db.update_table_entry(conn,'ref_phot','reference_images',
                                   'ref_phot_id',ref_phot_ids[-1],ref_db_id)
        
        phot_db.update_table_entry(conn,'ref_phot','star_id',
                                   'ref_phot_id',ref_phot_ids[-1],star_ids[j])
    
    if log!=None:
        log.info('Ingested '+str(len(ref_phot_ids))+\
                 ' reference image photometry entries into phot_db for reference image '+\
                 str(ref_db_id))
                 
    return ref_phot_ids
