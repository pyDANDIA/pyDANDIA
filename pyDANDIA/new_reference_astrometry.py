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
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.gaia import Gaia
import numpy as np
from pyDANDIA import  logs
from pyDANDIA import  metadata
from pyDANDIA import  starfind
from pyDANDIA import  pipeline_setup
from pyDANDIA import  wcs
from pyDANDIA import  psf
from pyDANDIA import  stage0
from pyDANDIA import  stage3
from pyDANDIA import  stage6
from pyDANDIA import psf
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
from skimage.measure import ransac

from skimage.transform import rotate
from skimage.registration import phase_cross_correlation
from skimage import transform as tf
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
        log.info('CATALOG FULL CROSS-MATCH SWITCHED OFF')

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

        rotate_wcs = 0 # no assumption on the WCS

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
                                                          kwargs,
                                                          stellar_density_threshold)

        vphas_sources = phot_catalog_objects_in_reference_image(setup, header, fov,
                                                                image_wcs, log, xmatch)

        selection_radius = 0.05 #degrees
        (bright_central_detected_stars, bright_central_gaia_stars, selection_radius) = \
             wcs.extract_bright_central_stars(setup,detected_sources, gaia_sources,
                                            image_wcs, log, radius=selection_radius)

        wcs.plot_overlaid_sources(os.path.join(setup.red_dir,'ref'),
                      bright_central_detected_stars, bright_central_gaia_stars, interactive=False)




        ### NEW IMPLEMENTATION ###




        reference_image_name = reduction_metadata.data_architecture[1]['REF_IMAGE'].data[0]
        reference_image_directory = reduction_metadata.data_architecture[1]['REF_PATH'].data[0]
        ref_structure = image_handling.determine_image_struture(os.path.join(reference_image_directory, reference_image_name), log=log)
        reference_image, date = stage6.open_an_image(setup, reference_image_directory, reference_image_name, log, image_index=ref_structure['sci'])


        ref_header = image_handling.get_science_header(meta_pars['ref_image_path'])

        wcs_ref = aWCS(ref_header)

        ra = wcs_ref.wcs.crval[0]
        dec = wcs_ref.wcs.crval[1]

        diagonal = np.sqrt(ref_header['NAXIS1']*ref_header['NAXIS1'] +ref_header['NAXIS2']*ref_header['NAXIS2'])
        radius = diagonal*ref_header['PIXSCALE']/3600.0/4.0 # ~ 5 arcminutes

        mask = (gaia_sources['ra']-ra)**2+(gaia_sources['dec']-dec)**2<radius**2

        sub_catalog = gaia_sources[mask]

        model,X,Y = generate_gaia_image_model(wcs_ref,reference_image.shape,sub_catalog)

        translation_rotation = find_initial_image_rotation_translation(model,reference_image)
        translation = translation_rotation[:2]
        rota_matrix = np.array([[np.cos(translation_rotation[-1]),-np.sin(translation_rotation[-1])],[np.sin(translation_rotation[-1]),np.cos(translation_rotation[-1])]])

        affine_matrix = np.c_[np.r_[rota_matrix.tolist(),[[0,0]]],[translation[0],translation[1],1]]
        center_matrix = np.array([[1,0,-int(reference_image.shape[1]/2)],
                                  [0,1,-int(reference_image.shape[0]/2)],
                                  [0,0,1]])

        tot_transform = np.dot(affine_matrix,center_matrix)
        tot_transform[0][2] += int(reference_image.shape[1]/2)
        tot_transform[1][2] += int(reference_image.shape[0]/2)

        transform = tf.SimilarityTransform(np.linalg.pinv(tot_transform))
        #transform = tf.SimilarityTransform(tot_transform)
        matched_stars = match_utils.StarMatchIndex()


        #gaia_sources = update_catalog_image_coordinates(setup, image_wcs,
        #                                                detected_sources, log,
        #                                                'catalog_stars_bright_revised_'+str(0)+'.reg',
        #                                                stellar_density, rotate_wcs, kwargs,
        #                                                stellar_density_threshold,
        #                                                transform=transform, radius=selection_radius)

        x = bright_central_gaia_stars['x']
        y = bright_central_gaia_stars['y']

        new_coords = transform(np.c_[x,y])
        bright_central_gaia_stars['x1'] = new_coords[:,0]
        bright_central_gaia_stars['y1'] = new_coords[:,1]

        filename = 'catalog_stars_bright_revised1.reg'
        cat_catalog_file = os.path.join(setup.red_dir,'ref', filename)
        catalog_utils.output_ds9_overlay_from_table(bright_central_gaia_stars,cat_catalog_file,
                                                    colour='red',
                                                    transformed_coords=True)

        # Matched indices refer to the array entries in the bright-central subcatalogs
        matched_stars = wcs.match_stars_pixel_coords(bright_central_detected_stars,
                                                     bright_central_gaia_stars,log,
                                                     tol=10.0,verbose=False)


        aa = bright_central_detected_stars[matched_stars.cat1_index]['x']
        bb = bright_central_detected_stars[matched_stars.cat1_index]['y']
        cc = bright_central_gaia_stars[matched_stars.cat2_index]['x']
        dd = bright_central_gaia_stars[matched_stars.cat2_index]['y']

        transform_robust, inliers = ransac((np.c_[cc,dd],np.c_[aa,bb]),tf.AffineTransform,residual_threshold = 0.5,min_samples=np.min([len(aa),20]))


        #gaia_sources = update_catalog_image_coordinates(setup, image_wcs, gaia_sources, log, 'catalog_stars_bright_revised_'+str(0)+'.reg',
        #                                            stellar_density, rotate_wcs, kwargs,
        #                                            stellar_density_threshold,
        #                                            transform=transform_robust, radius=selection_radius)
        x = bright_central_gaia_stars['x']
        y = bright_central_gaia_stars['y']

        new_coords = transform_robust(np.c_[x,y])


        bright_central_gaia_stars['x1'] = new_coords[:,0]
        bright_central_gaia_stars['y1'] = new_coords[:,1]

        # Matched indices refer to the array entries in the bright-central subcatalogs
        matched_stars = wcs.match_stars_pixel_coords(bright_central_detected_stars,
                                                     bright_central_gaia_stars,log,
                                                     tol=10.0,verbose=False)

        log.info(matched_stars.summary(units='pixels'))

        ### NEW IMPLEMENTATION ###


        # Apply initial transform, if any
        #transform = AffineTransform()
        it = 0
        max_it = 5
        iterate = True
        method = 'ransac'
        old_n_match = 0


        log.info('Transforming catalogue coordinates')


        center_ra = bright_central_gaia_stars['ra'].mean()
        center_dec = bright_central_gaia_stars['dec'].mean()

        x = bright_central_gaia_stars['ra'] - center_ra
        y = bright_central_gaia_stars['dec'] - center_dec

        xx = bright_central_detected_stars['ra'] - center_ra
        yy = bright_central_detected_stars['dec'] - center_dec

        transform_robust, inliers = ransac((np.c_[xx,yy][matched_stars.cat1_index],np.c_[x,y][matched_stars.cat2_index]),tf.AffineTransform,residual_threshold = 0.0006,min_samples=np.min([len(aa),20]))

        x = detected_sources['ra'] - center_ra
        y = detected_sources['dec'] - center_dec
        new_coords = transform_robust(np.c_[x,y])


        detected_sources['ra'] = new_coords[:,0]+center_ra
        detected_sources['dec'] = new_coords[:,1]+center_dec

        if xmatch:
            log.info('Proceeding to x-match of full catalogs')
            matched_stars_gaia = wcs.match_stars_world_coords(detected_sources,gaia_sources,log,'Gaia',
                                                          radius=0.5, ra_centre=image_wcs.wcs.crval[0],
                                                          dec_centre=image_wcs.wcs.crval[1],
                                                          verbose=False)

            matched_stars_vphas = wcs.match_stars_world_coords(detected_sources,vphas_sources,log,'VPHAS+',
                                                          radius=0.5, ra_centre=image_wcs.wcs.crval[0],
                                                          dec_centre=image_wcs.wcs.crval[1],
                                                          verbose=False)

        else:
            log.info('Proceeding with x-match of stars in the centre of the frame only')
            matched_stars_gaia = match_utils.transfer_main_catalog_indices(matched_stars,
                                            bright_central_detected_stars, bright_central_gaia_stars,
                                            detected_sources, gaia_sources, log)

            matched_stars_vphas = match_utils.StarMatchIndex()

            # Matched indices should now refer to the array entries in the full catalogs
            verbose=True
            if verbose:
                for j in range(0,len(matched_stars_gaia.cat1_index),1):

                    idx1 = int(matched_stars_gaia.cat1_index[j])
                    idx2 = int(matched_stars_gaia.cat2_index[j])

                    log.info('Detected source '+str(idx1)+' '+str(detected_sources['ra'][idx1])+' '+str(detected_sources['dec'][idx1])+\
                        ' matched to Gaia source '+str(idx2)+' '+gaia_sources['source_id'][idx2]+' '+str(gaia_sources['ra'][idx2])+' '+str(gaia_sources['dec'][idx2]))


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

    log.info('Received kwargs:')
    for key, value in kwargs.items():
        log.info(key+': '+repr(value))

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
                                        stellar_density, rotate_wcs, kwargs,
                                        stellar_density_threshold):

    field = str(header['OBJECT']).replace(' ','-')

    gaia_sources = wcs.fetch_catalog_sources_for_field(setup, field, header,
                                                      image_wcs,log,'Gaia-DR2')

    gaia_sources = wcs.calc_image_coordinates_astropy(setup, image_wcs,
                                                      gaia_sources, log,
                                                      stellar_density,
                                                      rotate_wcs, kwargs,
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

        vphas_sources = vizier_tools.search_vizier_for_sources(ra, dec, radius, 'VPHAS+', coords='degrees',log=log)

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
                                     stellar_density, rotate_wcs, kwargs,
                                     stellar_density_threshold,
                                     transform=None, radius=None):

    gaia_sources = wcs.calc_image_coordinates_astropy(setup, image_wcs,
                                                      gaia_sources,log,
                                                      stellar_density,
                                                      rotate_wcs, kwargs,
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



### New WCS method

def find_initial_image_rotation_translation(model_img,img,delta = 500):
    #import pdb; pdb.set_trace()
    solutions = []
    for i in range(4):

        sol = phase_cross_correlation(model_img[int(model_img.shape[0]/2-delta):int(model_img.shape[0]/2+delta),int(model_img.shape[1]/2-delta):int(model_img.shape[1]/2+delta)],          rotate(img.astype(float),90*i)[int(img.shape[0]/2-delta):int(img.shape[0]/2+delta),int(model_img.shape[1]/2-delta):int(model_img.shape[1]/2+delta)],
                                      upsample_factor=10)
        #import pdb; pdb.set_trace()
        solutions.append([sol[0][1],sol[0][0],i*np.pi/2])

    solutions = np.array(solutions)
    print(solutions)
    good_combination =  (solutions[:,0]**2+solutions[:,1]**2).argmin()

    return solutions[good_combination]


def generate_gaia_image_model(wcs_img,img_shape,catalog):

    #catalog = vizier_tools.search_vizier_for_sources(str(ra),str(dec),radius,'Gaia-DR2',coords='degrees')

    X,Y = wcs_img.wcs_world2pix(catalog['ra'],catalog['dec'],0)


    sigma_psf = 3.0
    sources = Table()
    sources['flux'] = catalog['phot_g_mean_flux']
    sources['x_mean'] = X
    sources['y_mean'] = Y
    sources['x_stddev'] = sigma_psf*np.ones(len(X))
    sources['y_stddev'] = sources['x_stddev']
    sources['theta'] = [0]*len(X)
    sources['id'] = np.arange(0,len(X)).tolist()
    tshape = img_shape

    #size of the psf stamp

    size = 21

    yy,xx = np.indices((size,size))
    aa = psf.Gaussian2D()

    model = np.zeros(img_shape)

    for ind in range(len(X)):

        posy = int(sources['y_mean'][ind])
        posx = int(sources['x_mean'][ind])

        momo = aa.psf_model(yy,xx,[sources['flux'][ind],sources['y_mean'][ind]-posy+int((size-1)/2),sources['x_mean'][ind]-posx+int((size-1)/2),sources['y_stddev'][ind],sources['x_stddev'][ind]])

        model[posy-int((size-1)/2):posy+int((size-1)/2)+1, posx-int((size-1)/2):posx+int((size-1)/2+1)] += momo

    #only return bright ones
    mask = sources['flux']>10000

    return model,X[mask],Y[mask]
