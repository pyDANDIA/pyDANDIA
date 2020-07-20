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
from pyDANDIA import  psf
from pyDANDIA import  stage0
from pyDANDIA import  stage3
from pyDANDIA import  config_utils
from pyDANDIA import  catalog_utils
from pyDANDIA import  calc_coord_offsets
from pyDANDIA import  shortest_string
from pyDANDIA import  calibrate_photometry
from pyDANDIA import  vizier_tools
from pyDANDIA import  match_utils
from pyDANDIA import  utilities
from pyDANDIA import  image_handling
from skimage.transform import AffineTransform

VERSION = 'pyDANDIA_reference_astrometry_v0.2'

def run_reference_astrometry(setup, **kwargs):
    """Driver function to perform the object detection and astrometric analysis
    of the reference frame from a given dataset.

    The optional flag force_rotate_ref allows an override of the default
    pipeline configuration, in the event that the reference image for a specific
    dataset requires it.
    """

    log = logs.start_stage_log( setup.red_dir, 'reference_astrometry', version=VERSION )

    kwargs = get_default_config(kwargs, log)
    xmatch = True
    if 'catalog_xmatch' in kwargs.keys() and kwargs['catalog_xmatch'] == False:
        xmatch = False
        log.info('CATALOG XMATCH SWITCHED OFF')

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

        header = image_handling.get_science_header(meta_pars['ref_image_path'])

        image_wcs = aWCS(header)

        field = header['OBJECT']
        fov = reduction_metadata.reduction_parameters[1]['FOV'][0]
        stellar_density_threshold = reduction_metadata.reduction_parameters[1]['STAR_DENSITY_THRESH'][0]
        rotate_wcs = reduction_metadata.reduction_parameters[1]['ROTATE_WCS'][0]

        # Calculates initial RA,Dec from image WCS
        detected_sources = detect_objects_in_reference_image(setup,
                                                             reduction_metadata,
                                                             meta_pars,
                                                             image_wcs, log)

        stellar_density = utilities.stellar_density_wcs(detected_sources,
                                                        image_wcs)

        # Calculates initial x,y from image WCS, initializes (x,y) -> (x1,y1)
        gaia_sources = catalog_objects_in_reference_image(setup, header,
                                                          image_wcs, log,
                                                          stellar_density,
                                                          rotate_wcs,
                                                          kwargs['force_rotate_ref'],
                                                          stellar_density_threshold)

        vphas_sources = phot_catalog_objects_in_reference_image(setup, header, fov,
                                                                image_wcs, log, xmatch)

        selection_radius = 0.05 #degrees
        (bright_central_detected_stars, bright_central_gaia_stars, selection_radius) = \
            wcs.extract_bright_central_stars(setup,detected_sources, gaia_sources,
                                             image_wcs, log, radius=selection_radius)

        wcs.plot_overlaid_sources(os.path.join(setup.red_dir,'ref'),
                      bright_central_detected_stars, bright_central_gaia_stars, interactive=False)

        # Apply initial transform, if any
        transform = AffineTransform()
        it = 0
        max_it = 5
        iterate = True
        method = 'ransac'
        old_n_match = 0

        if kwargs['trust_wcs'] == True:
            log.info('Trusting original WCS solution, transformation will be calculated after catalog match to original pixel positions')
            transform = AffineTransform(translation=(0.0, 0.0))

            #stellar_density = utilities.stellar_density(bright_central_gaia_stars,
            #                                    selection_radius)

            matched_stars = match_utils.StarMatchIndex()
            matched_stars = wcs.match_stars_pixel_coords(bright_central_detected_stars,
                                                     bright_central_gaia_stars,log,
                                                     tol=2.0,verbose=False)

            if len(matched_stars.cat1_index) > 3:
                transform = calc_coord_offsets.calc_pixel_transform(setup,
                                            bright_central_gaia_stars[matched_stars.cat2_index],
                                            bright_central_detected_stars[matched_stars.cat1_index],
                                            log, coordinates='pixel')

        else:
            while iterate:
                it += 1
                log.info('STAR MATCHING ITERATION '+str(it))

                if it == 1 and (kwargs['dx'] != 0.0 or kwargs['dy'] != 0.0):
                    log.info('Applying initial transformation of catalog positions, dx='+str(kwargs['dx'])+', dy='+str(kwargs['dy'])+' pixels')
                    transform = AffineTransform(translation=(kwargs['dx'], kwargs['dy']))
                    stellar_density = utilities.stellar_density(bright_central_gaia_stars,
                                                        selection_radius)
                    matched_stars = match_utils.StarMatchIndex()

                elif it == 1 and method in ['histogram', 'ransac']:
                    log.info('Calculating transformation using the histogram method, iteration '+str(it))

                    stellar_density = utilities.stellar_density(bright_central_gaia_stars,
                                                        selection_radius)

                    if stellar_density < stellar_density_threshold:
                        log.info('Stellar density is low, '+str(round(stellar_density,2))+' sources/arcmin**2, so using whole catalog to calculate transformation')
                        transform = calc_coord_offsets.calc_offset_pixels(setup, detected_sources, gaia_sources,
                                                                      log, diagnostics=True)
                    else:

                        log.info('Stellar density is high, '+str(round(stellar_density,2))+' sources/arcmin**2, so using bright central stars to calculate transformation')
                        transform = calc_coord_offsets.calc_offset_pixels(setup,bright_central_detected_stars,
                                                          bright_central_gaia_stars,
                                                          log,
                                                          diagnostics=True)

                    matched_stars = match_utils.StarMatchIndex()

                    # This method is not robust when the residual offsets are small
                    # Therefore, any proposed transform is ignored if it is smaller
                    # than a threshold value
                    if abs(transform.translation[0]) < 3.0 and abs(transform.translation[1]) < 3.0 and (matched_stars.n_match<20):

                        transform = AffineTransform()

                        log.info('Histogram method found a small transform, below the methods reliability threshold.  This transform will be ignored in favour of the RANSAC method')

                elif it > 1 and method in ['histogram', 'ransac']:
                    log.info('Calculating transformation using the ransac method, iteration '+str(it))
                    matched_stars = wcs.match_stars_pixel_coords(bright_central_detected_stars,
                                                             bright_central_gaia_stars,log,
                                                             tol=2.0,verbose=False)

                    if len(matched_stars.cat1_index) > 3:
                        transform = calc_coord_offsets.calc_pixel_transform(setup,
                                                    bright_central_gaia_stars[matched_stars.cat2_index],
                                                    bright_central_detected_stars[matched_stars.cat1_index],
                                                    log, coordinates='pixel')

                    else:
                        raise ValueError('No matched stars')

                if method == 'shortest_string':
                    det_array = np.zeros((len(bright_central_detected_stars),2))
                    det_array[:,0] = bright_central_detected_stars['x'].data
                    det_array[:,1] = bright_central_detected_stars['y'].data

                    cat_array = np.zeros((len(bright_central_gaia_stars),2))
                    cat_array[:,0] = bright_central_gaia_stars['x'].data
                    cat_array[:,1] = bright_central_gaia_stars['y'].data

                    (x_offset,y_offset) = shortest_string.find_xy_offset(det_array, cat_array,
                                                                  log=log,
                                                                  diagnostics=True)

                if old_n_match <= matched_stars.n_match:
                    bright_central_gaia_stars = update_catalog_image_coordinates(setup, image_wcs,
                                                                bright_central_gaia_stars, log,
                                                                'catalog_stars_bright_revised_'+str(it)+'.reg',
                                                                stellar_density, rotate_wcs, kwargs['force_rotate_ref'],
                                                                stellar_density_threshold,
                                                                transform=transform, radius=selection_radius)

                log.info('Testing to see if iterations should continue:')
                log.info('Iteration = '+str(it)+' maximum iterations='+str(max_it))
                log.info('Transformation parameters: '+repr(transform.translation))
                log.info('Previous number of stars matched='+str(old_n_match)+' current number of stars matched='+str(matched_stars.n_match))

                if it >= max_it or (abs(transform.translation[0]) < 0.5 and abs(transform.translation[1]) < 0.5\
                    and transform.translation[0] != 0.0 and transform.translation[1] != 0.0) or \
                        old_n_match > matched_stars.n_match:
                    iterate = False
                    log.info('Coordinate transform halting on iteration '+str(it)+' (of '+str(max_it)+'), latest transform '+repr(transform.translation)+\
                            ', Nstars matched '+str(matched_stars.n_match)+', compared with '+str(old_n_match)+' previously')

                else:
                    old_n_match = matched_stars.n_match
                    log.info(' -> Iterations continue, iterate='+repr(iterate))

        log.info('Transforming catalogue coordinates')

        gaia_sources = update_catalog_image_coordinates(setup, image_wcs,
                                                        gaia_sources, log, 'catalog_stars_full_revised_'+str(it)+'.reg',
                                                        stellar_density, rotate_wcs, kwargs['force_rotate_ref'],
                                                        stellar_density_threshold,
                                                        transform=transform, radius=None)

        transform = calc_coord_offsets.calc_world_transform(setup,
                                                bright_central_detected_stars[matched_stars.cat1_index],
                                                bright_central_gaia_stars[matched_stars.cat2_index],
                                                log)

        detected_sources = calc_coord_offsets.transform_coordinates(setup, detected_sources,
                                                                transform, coords='radec',
                                                                verbose=True)

        log.info('Proceeding to x-match of full catalogs')

        if xmatch:
            matched_stars_gaia = wcs.match_stars_world_coords(detected_sources,gaia_sources,log,'Gaia',
                                                          radius=0.5, ra_centre=image_wcs.wcs.crval[0],
                                                          dec_centre=image_wcs.wcs.crval[1],
                                                          verbose=False)

            matched_stars_vphas = wcs.match_stars_world_coords(detected_sources,vphas_sources,log,'VPHAS+',
                                                          radius=0.5, ra_centre=image_wcs.wcs.crval[0],
                                                          dec_centre=image_wcs.wcs.crval[1],
                                                          verbose=False)

        else:
            matched_stars_gaia = matched_stars
            matched_stars_vphas = match_utils.StarMatchIndex()

        ref_source_catalog = wcs.build_ref_source_catalog(detected_sources,\
                                                        gaia_sources, vphas_sources,\
                                                        matched_stars_gaia,
                                                        matched_stars_vphas,
                                                        image_wcs)

        log.info('Built reference image source catalogue of '+\
                 str(len(ref_source_catalog))+' objects')

        reduction_metadata.create_a_new_layer_from_table('star_catalog',ref_source_catalog)
        reduction_metadata.save_a_layer_to_file(setup.red_dir,
                                                'pyDANDIA_metadata.fits',
                                                'star_catalog', log=log)

        log.info('-> Output reference source FITS catalogue')
        log.info('Completed astrometry of reference image')

    logs.close_log(log)

    return 'OK', 'Reference astrometry complete'

def get_default_config(kwargs, log):

    default_config = {'force_rotate_ref': False,
                      'dx': 0.0, 'dy': 0.0,
                      'trust_wcs': False}

    kwargs = config_utils.set_default_config(default_config, kwargs, log)

    return kwargs

def detect_objects_in_reference_image(setup, reduction_metadata, meta_pars,
                                      image_wcs, log):

    ref_image_path = os.path.join(setup.red_dir,'ref',os.path.basename(meta_pars['ref_image_path']))

    image_structure = image_handling.determine_image_struture(ref_image_path, log=log)

    scidata = stage0.open_an_image(setup, os.path.join(setup.red_dir,'ref'),
                               os.path.basename(meta_pars['ref_image_path']),
                               log,  image_index=image_structure['sci'])

    if image_structure['bpm'] != None:
        image_bpm = stage0.open_an_image(setup, os.path.join(setup.red_dir,'ref'),
                               os.path.basename(meta_pars['ref_image_path']),
                               log,  image_index=image_structure['bpm'])
#    if image_bpm == None:
#        image_bpm = stage0.open_an_image(setup, os.path.join(setup.red_dir,'ref'),
#                                   os.path.basename(meta_pars['ref_image_path']),
#                                   log,  image_index=1)

    scidata = scidata.data - meta_pars['ref_sky_bkgd']
    idx = np.where(image_bpm.data != 0)
    image_bpm.data[idx] = reduction_metadata.reduction_parameters[1]['MAXVAL'][0]
    scidata = scidata + image_bpm.data

    maskref = os.path.join(setup.red_dir,'ref','masked_ref_image.fits')

    psf.output_fits(scidata, maskref)

    detected_objects = starfind.detect_sources(setup, reduction_metadata,
                                    meta_pars['ref_image_path'],
                                    scidata,
                                    log,
                                    diagnostics=False)

    detected_sources = wcs.build_detect_source_catalog(detected_objects)

    detected_sources = wcs.calc_world_coordinates_astropy(setup,image_wcs,
                                          detected_sources,log)

    det_catalog_file = os.path.join(setup.red_dir,'ref', 'detected_stars_full.reg')
    catalog_utils.output_ds9_overlay_from_table(detected_sources,det_catalog_file,
                                                colour='green')

    return detected_sources

def catalog_objects_in_reference_image(setup, header, image_wcs, log,
                                        stellar_density, rotate_wcs,
                                        force_rotate_ref,
                                        stellar_density_threshold):

    field = str(header['OBJECT']).replace(' ','-')

    gaia_sources = wcs.fetch_catalog_sources_for_field(setup, field, header,
                                                      image_wcs,log,'Gaia-DR2')

    gaia_sources = wcs.calc_image_coordinates_astropy(setup, image_wcs,
                                                      gaia_sources, log,
                                                      stellar_density,
                                                      rotate_wcs, force_rotate_ref,
                                                      stellar_density_threshold)

    gaia_sources.add_column( table.Column(name='x1', data=np.copy(gaia_sources['x'])) )
    gaia_sources.add_column( table.Column(name='y1', data=np.copy(gaia_sources['y'])) )

    cat_catalog_file = os.path.join(setup.red_dir,'ref', 'catalog_stars_full.reg')
    catalog_utils.output_ds9_overlay_from_table(gaia_sources,cat_catalog_file,
                                                colour='blue')

    return gaia_sources

def phot_catalog_objects_in_reference_image(setup, header, fov, image_wcs, log, xmatch):
    """Function to extract the objects from the VPHAS+ catalogue within the
    field of view of the reference image, based on the metadata information."""

    table_data = [table.Column(name='source_id', data=np.array([])),
                  table.Column(name='ra', data=np.array([])),
                  table.Column(name='dec', data=np.array([])),
                  table.Column(name='gmag', data=np.array([])),
                  table.Column(name='gmag_error', data=np.array([])),
                  table.Column(name='rmag', data=np.array([])),
                  table.Column(name='rmag_error', data=np.array([])),
                  table.Column(name='imag', data=np.array([])),
                  table.Column(name='imag_error', data=np.array([])),
                  table.Column(name='clean', data=np.array([])),
                  ]

    if xmatch:
        ra = image_wcs.wcs.crval[0]
        dec = image_wcs.wcs.crval[1]
        diagonal = np.sqrt(header['NAXIS1']*header['NAXIS1'] + header['NAXIS2']*header['NAXIS2'])
        radius = diagonal*header['PIXSCALE']/60.0/2.0 #arcminutes

        log.info('VPHAS+ catalog search parameters: ')
        log.info('RA = '+str(ra)+', Dec = '+str(dec))
        log.info('Radius: '+str(radius)+' arcmin')

        vphas_sources = vizier_tools.search_vizier_for_sources(ra, dec, radius, 'VPHAS+', coords='degrees')

        if len(vphas_sources)>0:

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
    log.info('VPHAS+ search returned ' + str(len(vphas_sources)) + ' entries')

    return vphas_sources




def update_catalog_image_coordinates(setup, image_wcs, gaia_sources,
                                     log, filename,
                                     stellar_density, rotate_wcs, force_rotate_ref,
                                     stellar_density_threshold,
                                     transform=None, radius=None):

    gaia_sources = wcs.calc_image_coordinates_astropy(setup, image_wcs,
                                                      gaia_sources,log,
                                                      stellar_density,
                                                      rotate_wcs, force_rotate_ref,
                                                      stellar_density_threshold,
                                                      radius=radius)

    if transform != None:
        gaia_sources = calc_coord_offsets.transform_coordinates(setup, gaia_sources,
                                                                transform, coords='pixel')
        log.info('-> Updated catalog image coordinates')

    cat_catalog_file = os.path.join(setup.red_dir,'ref', filename)
    catalog_utils.output_ds9_overlay_from_table(gaia_sources,cat_catalog_file,
                                                colour='red',
                                                transformed_coords=True)

    return gaia_sources
