# -*- coding: utf-8 -*-
"""
Created on Wed May  1 16:44:46 2019

@author: rstreet
"""
import os
import sys
from astropy.io import fits
from astropy import table
from astropy.wcs import WCS as aWCS
import numpy as np
from pyDANDIA import  logs
from pyDANDIA import  metadata
from pyDANDIA import  starfind
from pyDANDIA import  pipeline_setup
from pyDANDIA import  wcs
from pyDANDIA import  stage3
from pyDANDIA import  catalog_utils
from pyDANDIA import  calc_coord_offsets
from pyDANDIA import  calibrate_photometry
from pyDANDIA import  vizier_tools

VERSION = 'pyDANDIA_reference_astrometry_v0.1'

def run_reference_astrometry(setup):
    """Driver function to perform the object detection and astrometric analysis
    of the reference frame from a given dataset"""
    
    log = logs.start_stage_log( setup.red_dir, 'reference_astrometry', version=VERSION )
    
    
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
    
    sane = stage3.check_metadata(reduction_metadata,log)
    
    if sane: 
        meta_pars = stage3.extract_parameters_stage3(reduction_metadata,log)
    
        sane = stage3.sanity_checks(reduction_metadata,log,meta_pars)

    if sane:
                
        header = fits.getheader(meta_pars['ref_image_path'])
    
        image_wcs = aWCS(header)
        
        field = header['OBJECT']
        fov = reduction_metadata.reduction_parameters[1]['FOV'][0]
        
        detected_sources = detect_objects_in_reference_image(setup, 
                                                             reduction_metadata, 
                                                             meta_pars, 
                                                             image_wcs, log)
        
        gaia_sources = catalog_objects_in_reference_image(setup, header, 
                                                          image_wcs, log)
        
        vphas_sources = phot_catalog_objects_in_reference_image(setup, header, fov,
                                                                image_wcs, log)
        
        (bright_central_detected_stars, bright_central_gaia_stars) = wcs.extract_bright_central_stars(setup,detected_sources, 
                        gaia_sources, image_wcs, log, radius=0.05)
        
        transform = calc_coord_offsets.calc_offset_pixels(setup,bright_central_detected_stars, 
                                                      bright_central_gaia_stars,
                                                      log,
                                                      diagnostics=True)
        
        image_wcs = wcs.update_wcs(image_wcs,transform,header['PIXSCALE'],log,
                                                       transform_type='pixels')
    
        (detected_sources, gaia_sources) = update_computed_coordinates(setup, image_wcs, detected_sources, 
                                                        gaia_sources, log)
                                                        
        matched_stars_gaia = wcs.match_stars_world_coords(detected_sources,gaia_sources,log,
                                                 verbose=True)
        
        matched_stars_vphas = wcs.match_stars_world_coords(detected_sources,vphas_sources,log,
                                                 verbose=True)
                                                 
        ref_source_catalog = wcs.build_ref_source_catalog(detected_sources,\
                                                        gaia_sources, vphas_sources,\
                                                        matched_stars_gaia,
                                                        matched_stars_vphas,
                                                        image_wcs)
        
        log.info('Built reference image source catalogue of '+\
                 str(len(ref_source_catalog))+' objects')
        
        catalog_file = os.path.join(setup.red_dir,'ref', 'star_catalog.fits')
        catalog_utils.output_ref_catalog_file(catalog_file, ref_source_catalog)
        
        reduction_metadata.create_a_new_layer_from_table('star_catalog',ref_source_catalog)
        reduction_metadata.save_a_layer_to_file(setup.red_dir, 
                                                'pyDANDIA_metadata.fits',
                                                'star_catalog', log=log)
                                                
        log.info('-> Output reference source FITS catalogue')
        log.info('Completed astrometry of reference image')
    
    logs.close_log(log)
    
    return 0, 'OK'
    
def detect_objects_in_reference_image(setup, reduction_metadata, meta_pars, 
                                      image_wcs, log):
    
    scidata = fits.getdata(meta_pars['ref_image_path'])
    
    detected_objects = starfind.detect_sources(setup, reduction_metadata,
                                    meta_pars['ref_image_path'],
                                    (scidata-meta_pars['ref_sky_bkgd']),
                                    log,
                                    diagnostics=False)
    
    detected_sources = wcs.build_detect_source_catalog(detected_objects)

    detected_sources = wcs.calc_world_coordinates_astropy(setup,image_wcs,
                                          detected_sources,log)
                                          
    det_catalog_file = os.path.join(setup.red_dir,'ref', 'detected_stars_full.reg')
    catalog_utils.output_ds9_overlay_from_table(detected_sources,det_catalog_file,
                                                colour='green')
    
    return detected_sources

def catalog_objects_in_reference_image(setup, header, image_wcs, log):
    
    field = header['OBJECT']
        
    gaia_sources = wcs.fetch_catalog_sources_for_field(setup, field, header,
                                                      image_wcs,log,'Gaia')
    
    gaia_sources = wcs.calc_image_coordinates_astropy(setup, image_wcs, 
                                                      gaia_sources,log)
    
    cat_catalog_file = os.path.join(setup.red_dir,'ref', 'catalog_stars_full.reg')
    catalog_utils.output_ds9_overlay_from_table(gaia_sources,cat_catalog_file,
                                                colour='blue')
    
    return gaia_sources

def phot_catalog_objects_in_reference_image(setup, header, fov, image_wcs, log):
    """Function to extract the objects from the VPHAS+ catalogue within the
    field of view of the reference image, based on the metadata information."""
    
    ra = image_wcs.wcs.crval[0]
    dec = image_wcs.wcs.crval[1]
    radius = (np.sqrt(fov)/2.0)*60.0
    
    log.info('Search radius: '+str(radius)+' arcmin')
    
    vphas_sources = vizier_tools.search_vizier_for_sources(ra, dec, radius, 'VPHAS+',
                                                           coords='degrees')
    
    table_data = [ table.Column(name='source_id', data=vphas_sources['sourceID'].data),
                  table.Column(name='ra', data=vphas_sources['_RAJ2000'].data),
                  table.Column(name='dec', data=vphas_sources['_DEJ2000'].data),
                  table.Column(name='gmag', data=vphas_sources['gmag'].data),
                  table.Column(name='gmag_error', data=vphas_sources['e_gmag'].data),
                  table.Column(name='rmag', data=vphas_sources['rmag'].data),
                  table.Column(name='rmag_error', data=vphas_sources['e_rmag'].data),
                  table.Column(name='imag', data=vphas_sources['imag'].data),
                  table.Column(name='imag_error', data=vphas_sources['e_imag'].data),
                  table.Column(name='clean', data=vphas_sources['clean'].data),
                  ]
    vphas_sources = table.Table(data=table_data)
    
    log.info('VPHAS+ search returned '+str(len(vphas_sources))+' entries')
        
    return vphas_sources
    
def update_computed_coordinates(setup, image_wcs, detected_sources, 
                                gaia_sources, log):
    
    detected_sources = wcs.calc_world_coordinates_astropy(setup,image_wcs,
                                                  detected_sources,log)
    
    gaia_sources = wcs.calc_image_coordinates_astropy(setup, image_wcs, 
                                                      gaia_sources,log)
    
    cat_catalog_file = os.path.join(setup.red_dir,'ref', 'catalog_stars_full_updated.reg')
    catalog_utils.output_ds9_overlay_from_table(gaia_sources,cat_catalog_file,colour='red')

    wcs.plot_overlaid_sources(os.path.join(setup.red_dir,'ref'),
                      detected_sources, gaia_sources, interactive=False)
    
    return detected_sources, gaia_sources
    