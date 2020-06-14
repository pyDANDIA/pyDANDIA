# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 09:26:13 2019

@author: rstreet
"""

from os import path
from sys import argv
from astropy.io import fits
from astropy import table
from astropy import time
from astropy import units
from astropy.coordinates import SkyCoord, Angle
from skimage.transform import AffineTransform
import numpy as np
import glob
from shutil import move
from pyDANDIA import  logs
from pyDANDIA import  metadata
from pyDANDIA import  phot_db
from pyDANDIA import  pipeline_setup
from pyDANDIA import  time_utils
from pyDANDIA import  match_utils
from pyDANDIA import  calc_coord_offsets
from pyDANDIA import  wcs

VERSION = 'stage3_ingest_v1.0'

def run_stage3_db_ingest(setup, primary_ref=False, add_matched_stars=False):
    """Function to commit the information on, and measurements from, the
    reference image(s) used to reduce the data for a given field.
    """

    (facility_keys, software_keys, image_keys) = define_table_keys()

    log = logs.start_stage_log( setup.red_dir, 'stage3_db_ingest',
                               version=VERSION )
    if primary_ref:
        log.info('Running in PRIMARY-REF mode.')

    if not add_matched_stars:
        archive_existing_db(setup,primary_ref,log)
    else:
        log.info('Running to add the matched stars table to the metadata only')

    conn = phot_db.get_connection(dsn=setup.phot_db_path)

    reduction_metadata = metadata.MetaData()
    reduction_metadata.load_all_metadata(setup.red_dir, 'pyDANDIA_metadata.fits')

    dataset_params = harvest_stage3_parameters(setup,reduction_metadata)

    phot_db.check_before_commit(conn, dataset_params, 'facilities', facility_keys, 'facility_code')
    phot_db.check_before_commit(conn, dataset_params, 'software', software_keys, 'version')

    query = 'SELECT facility_id FROM facilities WHERE facility_code ="'+dataset_params['facility_code']+'"'
    dataset_params['facility'] = phot_db.query_to_astropy_table(conn, query, args=())['facility_id'][0]

    query = 'SELECT code_id FROM software WHERE version ="'+dataset_params['version']+'"'
    dataset_params['software'] = phot_db.query_to_astropy_table(conn, query, args=())['code_id'][0]

    query = 'SELECT filter_id FROM filters WHERE filter_name ="'+dataset_params['filter_name']+'"'
    dataset_params['filter'] = phot_db.query_to_astropy_table(conn, query, args=())['filter_id'][0]

    phot_db.check_before_commit(conn, dataset_params, 'images', image_keys, 'filename')

    if not add_matched_stars:
        commit_stamps_to_db(conn, reduction_metadata)

    ref_id_list = phot_db.find_reference_image_for_dataset(conn,dataset_params)

    if ref_id_list != None and len(ref_id_list) > 0 and add_matched_stars == False:
        phot_db.cascade_delete_reference_images(conn, ref_id_list, log)

    if not add_matched_stars:
        commit_reference_image(conn, dataset_params, log)
        commit_reference_component(conn, dataset_params, log)

    if primary_ref:

        if not add_matched_stars:
            star_ids = commit_stars(conn, dataset_params, reduction_metadata, log)

            commit_photometry(conn, dataset_params, reduction_metadata, star_ids, log)

        (matched_stars, transform) = generate_primary_ref_match_table(reduction_metadata,log)

    else:

        starlist = fetch_field_starlist(conn,dataset_params,log)

        primary_refimg_id = phot_db.find_primary_reference_image_for_field(conn)

        matched_stars = match_catalog_entries_with_starlist(conn,dataset_params,
                                                            starlist,
                                                            reduction_metadata,
                                                            primary_refimg_id,log,
                                                            verbose=True)

        (transform_xy,transform_sky) = calc_transform_to_primary_ref(setup,matched_stars,log)

        matched_stars = match_all_entries_with_starlist(setup,conn,dataset_params,
                                                        starlist,reduction_metadata,
                                                        primary_refimg_id,transform_sky,log,
                                                        verbose=True)

        if not add_matched_stars:
            commit_photometry_matching(conn, dataset_params, reduction_metadata,
                                                        matched_stars, log,
                                                        verbose=False)

    reduction_metadata.create_matched_stars_layer(matched_stars)
    reduction_metadata.create_transform_layer(transform_sky)
    reduction_metadata.save_a_layer_to_file(setup.red_dir,
                                          'pyDANDIA_metadata.fits',
                                          'matched_stars', log)
    reduction_metadata.save_a_layer_to_file(setup.red_dir,
                                        'pyDANDIA_metadata.fits',
                                        'transformation', log)
    conn.close()

    status = 'OK'
    report = 'Completed successfully'

    log.info('Stage 3 DB ingest: '+report)
    logs.close_log(log)

    return status, report

def define_table_keys():

    facility_keys = ['facility_code', 'site', 'enclosure',
                     'telescope', 'instrument', 'diameter_m',
                     'altitude_m', 'gain_eadu', 'readnoise_e', 'saturation_e' ]
    software_keys = ['code_name', 'stage', 'version']
    image_keys =    ['facility', 'filter', 'field_id', 'filename',
                     'date_obs_utc','date_obs_jd','exposure_time',
                     'fwhm','fwhm_err',
                     'ellipticity','ellipticity_err',
                     'slope','slope_err','intercept','intercept_err',
                     'wcsfrcat','wcsimcat','wcsmatch','wcsnref','wcstol','wcsra',
                     'wcsdec','wequinox','wepoch','radecsys',
                     'ctype1','ctype2','crpix1', 'crpix2', 'crval1', 'crval2',
                     'cdelt1','cdelt2','crota1','crota2',
                     'cunit1', 'cunit2',
                     'secpix1','secpix2',
                     'wcssep','equinox',
                     'cd1_1','cd1_2','cd2_1','cd2_2','epoch',
                     'airmass','moon_phase','moon_separation',
                     'delta_x','delta_y']

    return facility_keys, software_keys, image_keys

def define_stamp_keys():

    stamp_keys = ['stamp_index','xmin','xmax','ymin','ymax']

    return stamp_keys

def commit_stamps_to_db(conn, reduction_metadata):

    list_of_stamps = reduction_metadata.stamps[1]['PIXEL_INDEX'].tolist()
    stamps_params = {}
    stamp_keys = define_stamp_keys()
    for stamp in list_of_stamps:
        stamp_row = np.where(reduction_metadata.stamps[1]['PIXEL_INDEX'] == stamp)[0][0]
        xmin = int(reduction_metadata.stamps[1][stamp_row]['X_MIN'])
        xmax = int(reduction_metadata.stamps[1][stamp_row]['X_MAX'])
        ymin = int(reduction_metadata.stamps[1][stamp_row]['Y_MIN'])
        ymax = int(reduction_metadata.stamps[1][stamp_row]['Y_MAX'])

        stamps_params['stamp_index'] = str(stamp)
        stamps_params['xmin'] = str(xmin)
        stamps_params['xmax'] = str(xmax)
        stamps_params['ymin'] = str(ymin)
        stamps_params['ymax'] = str(ymax)

        phot_db.check_before_commit(conn, stamps_params, 'stamps', stamp_keys, 'stamp_index')

def configure_setup(log=None):

    params = {'datasets': [('red_dir_gp', 1), ('red_dir_rp', 2), ('red_dir_ip',3)],
              'stage': 'stage3_db_ingest',
              'red_dir_gp': None,
              'red_dir_rp': None,
              'red_dir_ip': None}

    if len(argv) == 1:

        params['red_dir_gp'] = input('Please enter the path to the SDSS-g reduction directory (or None): ')
        params['red_dir_rp'] = input('Please enter the path to the SDSS-r reduction directory (or None): ')
        params['red_dir_ip'] = input('Please enter the path to the SDSS-i reduction directory (or None): ')
        params['combined_starcat'] = input('Please enter the path to the combined star catalogue file: ')
        params['database_file_path'] = input('Please enter the path to the database file: ')
        params['log_dir'] = input('Please enter the path to the logging directory: ')

        for f, i in params['datasets']:

            if 'none' in params[f].lower():

                params[f] = None

    else:

        params['red_dir_gp'] = argv[1]
        params['red_dir_rp'] = argv[2]
        params['red_dir_ip'] = argv[3]
        params['combined_starcat'] = argv[4]
        params['database_file_path'] = argv[5]
        params['log_dir'] = argv[6]

    if log!=None:

        log.info('Input dataset configurations:')

    for f, i in params['datasets']:

        if f != None:

            pars = {'stage': params['stage'], 'red_dir': params[f]}

            setup = pipeline_setup.pipeline_setup(pars)

            params[f.replace('red_dir','setup')] = setup

            if log!=None:
                log.info(setup.summary())

    return params

def archive_existing_db(setup,primary_ref,log):
    """Function to archive an existing photometric DB, if one exists, in the
    event that a new primary reference is adopted"""

    def identify_archive_index(db_list):

        if len(db_list) > 0:

            idx = 0
            for db_file in db_list:

                suffix = db_file.split('.')[-1]

                if 'db' not in suffix:
                    try:
                        i = int(suffix)
                    except TypeError:
                        pass
                    if i > idx:
                        idx = i

        else:

            idx = 0

        idx += 1

        return idx

    if primary_ref:

        db_list = glob.glob(setup.phot_db_path+'*')

        idx = identify_archive_index(db_list)

        if path.isfile(setup.phot_db_path):

            dest = setup.phot_db_path + '.' + str(idx)

            move(setup.phot_db_path, dest)

            log.info('-> Archived old PHOT_DB to '+dest)

def harvest_stage3_parameters(setup,reduction_metadata):
    """Function to harvest the parameters required for ingest of a single
    dataset into the photometric database."""

    ref_path = reduction_metadata.data_architecture[1]['REF_PATH'][0]
    ref_filename = reduction_metadata.data_architecture[1]['REF_IMAGE'][0]

    ref_image_path = path.join(ref_path, ref_filename)

    dataset_params = harvest_image_params(reduction_metadata, ref_image_path, ref_image_path)

    dataset_params['psf_radius'] = reduction_metadata.psf_dimensions[1]['psf_radius'][0]

    # Software
    dataset_params['version'] = reduction_metadata.software[1]['stage3_version'][0]
    dataset_params['stage'] = 'stage3'
    dataset_params['code_name'] = 'stage3.py'

    return dataset_params

def harvest_image_params(reduction_metadata, image_path, ref_image_path):

    image_header = fits.getheader(image_path)

    image_params = {}

    # Facility
    image_params['site'] = image_header['SITEID']
    image_params['enclosure'] = image_header['ENCID']
    image_params['telescope'] = image_header['TELID']
    image_params['instrument'] = image_header['INSTRUME']
    if 'fl' in image_params['instrument']:
        image_params['instrument'] = image_params['instrument'].replace('fl','fa')
    image_params['facility_code'] = phot_db.get_facility_code(image_params)

    # Image parameters for single-frame reference image
    # NOTE: Stacked reference images not yet supported
    image_params['filename'] = path.basename(image_path)
    image_params['ref_filename'] = path.basename(ref_image_path)

    idx = np.where(reduction_metadata.headers_summary[1]['IMAGES'] == image_params['filename'])
    hdr_meta = reduction_metadata.headers_summary[1][idx]
    idx = np.where(reduction_metadata.images_stats[1]['IM_NAME'] == image_params['filename'])
    image_stats = reduction_metadata.images_stats[1][idx]

    image_params['diameter_m'] = float(image_header['TELID'].replace('a','').replace('m','.'))
    image_params['altitude_m'] = image_header['HEIGHT']
    image_params['gain_eadu'] = image_header['GAIN']
    image_params['readnoise_e'] = image_header['RDNOISE'] * image_params['gain_eadu']
    image_params['saturation_e'] = image_header['SATURATE'] * image_params['gain_eadu']

    image_params['field_id'] = hdr_meta['OBJKEY'][0]
    image_params['date_obs_utc'] = hdr_meta['DATEKEY'][0]
    t = time.Time(image_params['date_obs_utc'],format='isot', scale='utc')
    image_params['date_obs_jd'] = t.jd
    image_params['exposure_time'] = float(hdr_meta['EXPKEY'][0])
    image_params['RA'] = image_header['RA']
    image_params['Dec'] = image_header['DEC']
    image_params['filter_name'] = image_header['FILTER']
    image_params['fwhm'] = image_stats['FWHM'][0]
    image_params['fwhm_err'] = None
    image_params['ellipticity'] = None
    image_params['ellipticity_err'] = None
    image_params['slope'] = None
    image_params['slope_err'] = None
    image_params['intercept'] = None
    image_params['intercept_err'] = None
    image_params['wcsfrcat'] = None
    image_params['wcsimcat'] = None
    image_params['wcsmatch'] = None
    image_params['wcsnref'] = None
    image_params['wcstol'] = None
    image_params['wcsra'] = None
    image_params['wcsdec'] = None
    image_params['wequinox'] = None
    image_params['wepoch'] = None
    image_params['radecsys'] = None
    image_params['ctype1'] = set_if_present(image_header,'CTYPE1')
    image_params['ctype2'] = set_if_present(image_header,'CTYPE2')
    image_params['cdelt1'] = set_if_present(image_header,'CDELT1')
    image_params['cdelt2'] = set_if_present(image_header,'CDELT2')
    image_params['crpix1'] = set_if_present(image_header,'CRPIX1')
    image_params['crpix2'] = set_if_present(image_header,'CRPIX2')
    image_params['crval1'] = set_if_present(image_header,'CRVAL1')
    image_params['crval2'] = set_if_present(image_header,'CRVAL2')
    image_params['crota1'] = set_if_present(image_header,'CROTA1')
    image_params['crota2'] = set_if_present(image_header,'CROTA2')
    image_params['cunit1'] = set_if_present(image_header,'CUNIT1')
    image_params['cunit2'] = set_if_present(image_header,'CUNIT2')
    image_params['secpix1'] = set_if_present(image_header,'PIXSCALE')
    image_params['secpix2'] = set_if_present(image_header,'PIXSCALE')
    image_params['wcssep'] = None
    image_params['equinox'] = None
    image_params['cd1_1'] = set_if_present(image_header,'CD1_1')
    image_params['cd1_2'] = set_if_present(image_header,'CD1_2')
    image_params['cd2_1'] = set_if_present(image_header,'CD2_1')
    image_params['cd2_2'] = set_if_present(image_header,'CD2_2')
    image_params['epoch'] = None
    image_params['airmass'] = set_if_present(image_header,'AIRMASS')
    image_params['moon_phase'] = set_if_present(image_header,'MOONFRAC')
    image_params['moon_separation'] = set_if_present(image_header,'MOONDIST')
    image_params['delta_x'] = None
    image_params['delta_y'] = None

    image_params['hjd'] = time_utils.calc_hjd(image_params['date_obs_utc'],
                                  image_params['RA'],image_params['Dec'],
                                  image_params['exposure_time'])

    return image_params

def set_if_present(header, key):

    if key in header.keys():
        return header[key]
    else:
        return None

def commit_reference_image(conn, params, log):

    query = 'SELECT refimg_id,filename FROM reference_images WHERE filename ="'+params['ref_filename']+'"'
    ref_image = phot_db.query_to_astropy_table(conn, query, args=())

    if len(ref_image) != 0:

        log.info('Reference image '+params['ref_filename']+\
                    ' is already in the phot_db as entry '+\
                        str(ref_image['refimg_id']))

    else:

        query = 'SELECT facility_id, facility_code FROM facilities WHERE facility_code="'+params['facility_code']+'"'
        facility = phot_db.query_to_astropy_table(conn, query, args=())
        error_wrong_number_entries(facility,params['facility_code'])

        query = 'SELECT filter_id, filter_name FROM filters WHERE filter_name="'+params['filter_name']+'"'
        f = phot_db.query_to_astropy_table(conn, query, args=())
        error_wrong_number_entries(f,params['filter_name'])

        query = 'SELECT code_id, version FROM software WHERE version="'+params['version']+'"'
        code = phot_db.query_to_astropy_table(conn, query, args=())
        error_wrong_number_entries(code,params['version'])

        command = 'INSERT OR REPLACE INTO reference_images (facility,filter,software,filename) VALUES ('+\
            str(facility['facility_id'][0])+','+str(f['filter_id'][0])+','+str(code['code_id'][0])+',"'+str(params['ref_filename'])+'")'

        cursor = conn.cursor()

        cursor.execute(command)

        conn.commit()

        log.info('Submitted reference_image '+params['ref_filename']+' to phot_db')

def error_wrong_number_entries(results_table,param_value):

    if len(results_table) == 0:
        raise IOError(param_value+' is unknown to the database')
    elif len(results_table) > 1:
        raise IOError('Mulitple entries named '+param_value+' are known to the database')

def commit_reference_component(conn, params, log):
    """Function to create the reference component entry for a SINGLE-IMAGE
    reference"""

    table_keys = [ 'image', 'reference_image' ]

    query = 'SELECT img_id, filename FROM images WHERE filename ="'+params['filename']+'"'
    image = phot_db.query_to_astropy_table(conn, query, args=())
    error_wrong_number_entries(image,params['filename'])

    query = 'SELECT refimg_id, filename FROM reference_images WHERE filename ="'+params['ref_filename']+'"'
    refimage = phot_db.query_to_astropy_table(conn, query, args=())
    error_wrong_number_entries(refimage,params['filename'])

    table_keys = [ 'image', 'reference_image' ]

    command = 'INSERT OR REPLACE INTO reference_components (image,reference_image) VALUES ('+\
                str(image['img_id'][0])+','+str(refimage['refimg_id'][0])+')'

    cursor = conn.cursor()

    cursor.execute(command)

    conn.commit()

    log.info('Submitted reference_component '+params['filename']+' to phot_db')

def read_combined_star_catalog(params):

    if path.isfile(params['combined_starcat']) == False:

        raise IOError('Cannot find combined star catalogue file '+\
                        params['combined_starcat'])

    star_catalog = fits.getdata(params['combined_starcat'])

    return star_catalog

def commit_stars(conn, params, reduction_metadata, log,
                 search_for_match=False):

    query = 'SELECT refimg_id, filename FROM reference_images WHERE filename ="'+params['ref_filename']+'"'
    refimage = phot_db.query_to_astropy_table(conn, query, args=())
    error_wrong_number_entries(refimage,params['ref_filename'])

    tol = 1.0/3600.0

    n_stars = len(reduction_metadata.star_catalog[1])

    star_keys = ['star_index', 'ra', 'dec', 'reference_image',
                 'gaia_source_id', 'gaia_ra', 'gaia_ra_error', 'gaia_dec', 'gaia_dec_error',
                 'gaia_phot_g_mean_flux', 'gaia_phot_g_mean_flux_error',
                 'gaia_phot_bp_mean_flux', 'gaia_phot_bp_mean_flux_error',
                 'gaia_phot_rp_mean_flux', 'gaia_phot_rp_mean_flux_error',
                 'vphas_source_id', 'vphas_ra', 'vphas_dec',
                 'vphas_gmag', 'vphas_gmag_error',
                 'vphas_rmag', 'vphas_rmag_error',
                 'vphas_imag', 'vphas_imag_error', 'vphas_clean']

    star_ids = np.zeros(n_stars)

    log.info('Starting commit of '+str(n_stars)+' stars')

    for j in range(0,n_stars,1):

        idx = reduction_metadata.star_catalog[1]['index'][j]
        ra = reduction_metadata.star_catalog[1]['ra'][j]
        dec = reduction_metadata.star_catalog[1]['dec'][j]

        star_pars = [ str(idx), str(ra), str(dec), str(refimage['refimg_id'][0]),
                    str(reduction_metadata.star_catalog[1]['gaia_source_id'][j]),
                    str(reduction_metadata.star_catalog[1]['gaia_ra'][j]),
                    str(reduction_metadata.star_catalog[1]['gaia_ra_error'][j]),
                    str(reduction_metadata.star_catalog[1]['gaia_dec'][j]),
                    str(reduction_metadata.star_catalog[1]['gaia_dec_error'][j]),
                    str(reduction_metadata.star_catalog[1]['phot_g_mean_flux'][j]),
                    str(reduction_metadata.star_catalog[1]['phot_g_mean_flux_error'][j]),
                    str(reduction_metadata.star_catalog[1]['phot_bp_mean_flux'][j]),
                    str(reduction_metadata.star_catalog[1]['phot_bp_mean_flux_error'][j]),
                    str(reduction_metadata.star_catalog[1]['phot_rp_mean_flux'][j]),
                    str(reduction_metadata.star_catalog[1]['phot_rp_mean_flux_error'][j]),
                    str(reduction_metadata.star_catalog[1]['vphas_source_id'][j]),
                    str(reduction_metadata.star_catalog[1]['vphas_ra'][j]),
                    str(reduction_metadata.star_catalog[1]['vphas_dec'][j]),
                    str(reduction_metadata.star_catalog[1]['gmag'][j]),
                    str(reduction_metadata.star_catalog[1]['gmag_error'][j]),
                    str(reduction_metadata.star_catalog[1]['rmag'][j]),
                    str(reduction_metadata.star_catalog[1]['rmag_error'][j]),
                    str(reduction_metadata.star_catalog[1]['imag'][j]),
                    str(reduction_metadata.star_catalog[1]['imag_error'][j]),
                    str((int(reduction_metadata.star_catalog[1]['clean'][j]))) ]

        if search_for_match:

            results = phot_db.box_search_on_position(conn, ra, dec, tol, tol)

            if len(results) > 0:

                log.info('Catalog star at RA, Dec '+str(ra)+','+str(dec)+\
                     ' matches a star already in the phot_db with RA, Dec '+\
                     str(results['ra'][0])+','+str(results['dec'][0]))

                star_ids[j] = int(results['star_id'][0])

                submit = False

            else:

                submit = True

        else:

            submit = True

        if submit:

            command = 'INSERT OR REPLACE INTO stars ('+','.join(star_keys)+') VALUES ("'
            command += '","'.join(star_pars) + '")'

            cursor = conn.cursor()

            cursor.execute(command)

            conn.commit()

            results = phot_db.box_search_on_position(conn, ra, dec, tol, tol)

            if len(results) > 1:
                idx = np.where(results['separation'] == results['separation'].min())[0][0]
            else:
                idx = 0

            #log.info('Commited catalog star at RA, Dec '+str(ra)+','+str(dec)+\
            #         ' to the phot_db as star_id='+str(results['star_id'][idx]))

            star_ids[j] = int(results['star_id'][idx])

    log.info('Completed the ingest of '+str(n_stars)+' to the photometric database')

    return star_ids

def commit_photometry(conn, params, reduction_metadata, star_ids, log):

    log.info('Extracting dataset descriptors for ingest of photometry')

    query = 'SELECT facility_id, facility_code FROM facilities WHERE facility_code="'+params['facility_code']+'"'
    facility = phot_db.query_to_astropy_table(conn, query, args=())
    error_wrong_number_entries(facility,params['facility_code'])

    query = 'SELECT filter_id, filter_name FROM filters WHERE filter_name="'+params['filter_name']+'"'
    f = phot_db.query_to_astropy_table(conn, query, args=())
    error_wrong_number_entries(f,params['filter_name'])

    query = 'SELECT code_id, version FROM software WHERE version="'+params['version']+'"'
    code = phot_db.query_to_astropy_table(conn, query, args=())
    error_wrong_number_entries(code,params['version'])

    query = 'SELECT refimg_id, filename FROM reference_images WHERE filename ="'+params['ref_filename']+'"'
    refimage = phot_db.query_to_astropy_table(conn, query, args=())
    error_wrong_number_entries(refimage,params['ref_filename'])

    query = 'SELECT img_id, filename FROM images WHERE filename ="'+params['filename']+'"'
    image = phot_db.query_to_astropy_table(conn, query, args=())
    error_wrong_number_entries(image,params['filename'])

    key_list = ['star_id', 'star_dataset_id', 'reference_image', 'image',
                'facility', 'filter', 'software',
                'x', 'y', 'hjd', 'radius', 'magnitude', 'magnitude_err',
                'calibrated_mag', 'calibrated_mag_err',
                'flux', 'flux_err', 'calibrated_flux', 'calibrated_flux_err',
                'phot_scale_factor', 'phot_scale_factor_err',
                'local_background', 'local_background_err',
                'phot_type']

    wildcards = ','.join(['?']*len(key_list))

    n_stars = len(reduction_metadata.star_catalog[1])

    log.info('Starting ingest of photometry data for '+str(n_stars)+' stars')

    values = []
    for j in range(0,n_stars,1):

        k = reduction_metadata.star_catalog[1]['index'][j]
        x = str(reduction_metadata.star_catalog[1]['x'][j])
        y = str(reduction_metadata.star_catalog[1]['y'][j])
        mag = str(reduction_metadata.star_catalog[1]['ref_mag'][j])
        mag_err = str(reduction_metadata.star_catalog[1]['ref_mag_error'][j])
        cal_mag = str(reduction_metadata.star_catalog[1]['cal_ref_mag'][j])
        cal_mag_err = str(reduction_metadata.star_catalog[1]['cal_ref_mag_error'][j])
        flux = str(reduction_metadata.star_catalog[1]['ref_flux'][j])
        flux_err = str(reduction_metadata.star_catalog[1]['ref_flux_error'][j])
        cal_flux = str(reduction_metadata.star_catalog[1]['cal_ref_flux'][j])
        cal_flux_err = str(reduction_metadata.star_catalog[1]['cal_ref_flux_error'][j])
        sky = str(reduction_metadata.star_catalog[1]['sky_background'][j])
        sky_err = str(reduction_metadata.star_catalog[1]['sky_background_error'][j])

        entry = (str(int(star_ids[j])), str(int(star_ids[j])), str(refimage['refimg_id'][0]), str(image['img_id'][0]),
                   str(facility['facility_id'][0]), str(f['filter_id'][0]), str(code['code_id'][0]),
                    x, y, str(params['hjd']),
                    params['psf_radius'],
                    mag, mag_err, cal_mag, cal_mag_err,
                    flux, flux_err, cal_flux, cal_flux_err,
                    '0.0', '0.0',   # No phot scale factor for PSF fitting photometry
                    sky, sky_err,   # No background measurements propageted
                    'PSF_FITTING' )

        values.append(entry)

    command = 'INSERT OR REPLACE INTO phot('+','.join(key_list)+\
                ') VALUES ('+wildcards+')'

    cursor = conn.cursor()

    cursor.executemany(command,values)

    conn.commit()

    log.info('Completed ingest of photometry for '+str(len(star_ids))+' stars')

def commit_photometry_matching(conn, params, reduction_metadata, matched_stars,
                                log, verbose=False):

    log.info('Extracting dataset descriptors for ingest of matching photometry')

    query = 'SELECT facility_id, facility_code FROM facilities WHERE facility_code="'+params['facility_code']+'"'
    facility = phot_db.query_to_astropy_table(conn, query, args=())
    error_wrong_number_entries(facility,params['facility_code'])

    query = 'SELECT filter_id, filter_name FROM filters WHERE filter_name="'+params['filter_name']+'"'
    f = phot_db.query_to_astropy_table(conn, query, args=())
    error_wrong_number_entries(f,params['filter_name'])

    query = 'SELECT code_id, version FROM software WHERE version="'+params['version']+'"'
    code = phot_db.query_to_astropy_table(conn, query, args=())
    error_wrong_number_entries(code,params['version'])

    query = 'SELECT refimg_id, filename FROM reference_images WHERE filename ="'+params['filename']+'"'
    refimage = phot_db.query_to_astropy_table(conn, query, args=())
    error_wrong_number_entries(refimage,params['filename'])

    query = 'SELECT img_id, filename FROM images WHERE filename ="'+params['filename']+'"'
    image = phot_db.query_to_astropy_table(conn, query, args=())
    error_wrong_number_entries(image,params['filename'])

    key_list = ['star_id', 'star_dataset_id', 'reference_image', 'image',
                'facility', 'filter', 'software',
                'x', 'y', 'hjd', 'magnitude', 'magnitude_err',
                'calibrated_mag', 'calibrated_mag_err',
                'flux', 'flux_err',
                'calibrated_flux', 'calibrated_flux_err',
                'phot_scale_factor', 'phot_scale_factor_err',
                'local_background', 'local_background_err',
                'phot_type']

    wildcards = ','.join(['?']*len(key_list))

    n_stars = len(reduction_metadata.star_catalog[1])

    log.info('Starting ingest of photometry data for '+str(n_stars)+' detected stars in this dataset')

    values = []
    for i in range(0,matched_stars.n_match,1):

        j_cat = matched_stars.cat1_index[i]     # Starlist index in DB -> array index
        j_new = matched_stars.cat2_index[i]     # Star detected in image catalog index
        jj = j_new - 1

        # j_new index refers to star number in catalogue
        # jj index refers to array index
        x = str(reduction_metadata.star_catalog[1]['x'][jj])
        y = str(reduction_metadata.star_catalog[1]['y'][jj])
        mag = str(reduction_metadata.star_catalog[1]['ref_mag'][jj])
        mag_err = str(reduction_metadata.star_catalog[1]['ref_mag_error'][jj])
        cal_mag = str(reduction_metadata.star_catalog[1]['cal_ref_mag'][jj])
        cal_mag_err = str(reduction_metadata.star_catalog[1]['cal_ref_mag_error'][jj])
        flux = str(reduction_metadata.star_catalog[1]['ref_flux'][jj])
        flux_err = str(reduction_metadata.star_catalog[1]['ref_flux_error'][jj])
        cal_flux = str(reduction_metadata.star_catalog[1]['cal_ref_flux'][jj])
        cal_flux_err = str(reduction_metadata.star_catalog[1]['cal_ref_flux_error'][jj])
        sky = str(reduction_metadata.star_catalog[1]['sky_background'][jj])
        sky_err = str(reduction_metadata.star_catalog[1]['sky_background_error'][jj])

        entry = (str(int(j_cat)), str(int(j_new)), str(refimage['refimg_id'][0]), str(image['img_id'][0]),
                   str(facility['facility_id'][0]), str(f['filter_id'][0]), str(code['code_id'][0]),
                    x, y, str(params['hjd']),
                    mag, mag_err, cal_mag, cal_mag_err,
                    flux, flux_err, cal_flux, cal_flux_err,
                    '0.0', '0.0',   # No phot scale factor for PSF fitting photometry
                    sky, sky_err,   # No background measurements propageted
                    'PSF_FITTING' )

        values.append(entry)

        if verbose:
            log.info(str(entry))

    command = 'INSERT OR REPLACE INTO phot('+','.join(key_list)+\
                ') VALUES ('+wildcards+')'

    cursor = conn.cursor()

    cursor.executemany(command,values)

    conn.commit()

    log.info('Completed ingest of photometry for '+str(len(matched_stars.cat1_index))+\
            ' stars from this dataset matched with the field reference catalog')

def fetch_field_starlist(conn,params,log):

    query = 'SELECT * FROM stars'
    starlist = phot_db.query_to_astropy_table(conn, query, args=())

    log.info('Selected '+str(len(starlist))+' stars known in this field')

    return starlist

def match_catalog_entries_with_starlist(conn,params,starlist,reduction_metadata,
                                        refimg_id,log,verbose=False):

    log.info('Matching all stars from the reference frame with catalog identifications with the DB')

    matched_stars = match_utils.StarMatchIndex()

    idx = np.where(starlist['gaia_source_id'] != 'None')[0]
    jdx = np.where(reduction_metadata.star_catalog[1]['gaia_source_id'] != 'None')[0]

    log.info(str(len(idx))+' stars with Gaia identifications selected from the field starlist')
    log.info(str(len(jdx))+' stars with Gaia identifications selected from new reference image catalog')

    for star in starlist[idx]:

        kdx = np.where(reduction_metadata.star_catalog[1]['gaia_source_id'][jdx] == star['gaia_source_id'])

        if len(kdx[0]) == 1:

            query = 'SELECT star_id,x,y FROM phot WHERE reference_image="'+str(refimg_id)+'" AND star_id="'+str(star['star_id'])+'"'
            phot_data = phot_db.query_to_astropy_table(conn, query, args=())

            dataset_star = SkyCoord( reduction_metadata.star_catalog[1]['ra'][jdx[kdx[0]]],
                          reduction_metadata.star_catalog[1]['dec'][jdx[kdx[0]]],
                          frame='icrs', unit=(units.deg, units.deg) )

            field_star = SkyCoord( star['ra'], star['dec'],
                                    frame='icrs', unit=(units.deg, units.deg) )

            separation = dataset_star.separation(field_star)

            jj = jdx[kdx[0]][0]

            p = {'cat1_index': star['star_id'],
                 'cat1_ra': star['ra'],
                 'cat1_dec': star['dec'],
                 'cat1_x': phot_data['x'][0],
                 'cat1_y': phot_data['y'][0],
                 'cat2_index': jj+1,
                 'cat2_ra': reduction_metadata.star_catalog[1]['ra'][jj],
                 'cat2_dec': reduction_metadata.star_catalog[1]['dec'][jj],
                 'cat2_x': reduction_metadata.star_catalog[1]['x'][jj],
                 'cat2_y': reduction_metadata.star_catalog[1]['y'][jj],
                 'separation': separation[0]}

            matched_stars.add_match(p)

            if verbose:
                log.info(matched_stars.summarize_last(units='pixels'))

    return matched_stars

def calc_transform_to_primary_ref(setup,matched_stars,log):

    primary_cat_cartesian = table.Table( [ table.Column(name='x', data=matched_stars.cat1_x),
                                 table.Column(name='y', data=matched_stars.cat1_y) ] )

    refframe_cat_cartesian = table.Table( [ table.Column(name='x', data=matched_stars.cat2_x),
                                  table.Column(name='y', data=matched_stars.cat2_y) ] )

    primary_cat_sky = table.Table( [ table.Column(name='ra', data=matched_stars.cat1_ra),
                                 table.Column(name='dec', data=matched_stars.cat1_dec) ] )

    refframe_cat_sky = table.Table( [ table.Column(name='ra', data=matched_stars.cat2_ra),
                                  table.Column(name='dec', data=matched_stars.cat2_dec) ] )

    transform_cartesian = calc_coord_offsets.calc_pixel_transform(setup,
                                        refframe_cat_cartesian, primary_cat_cartesian,
                                        log, coordinates='pixel', diagnostics=True)

    transform_sky = calc_coord_offsets.calc_pixel_transform(setup,
                                        refframe_cat_sky, primary_cat_sky,
                                        log, coordinates='sky', diagnostics=True,
                                        plot_path=path.join(setup.red_dir, 'dataset_field_sky_offsets.png'))

    return transform_cartesian, transform_sky

def match_all_entries_with_starlist(setup,conn,params,starlist,reduction_metadata,
                                    refimg_id,transform_sky,log, verbose=False):

    psf_radius = reduction_metadata.psf_dimensions[1]['psf_radius'][0]
    #tol = Angle( (((psf_radius) * params['secpix1'])/3600.0) * units.deg )
    tol = psf_radius * params['secpix1']    # arcsec
    dra = 2.0*tol                           # arcsec
    ddec = 2.0*tol                          # arcsec

    log.info('Matching all stars from starlist with the transformed coordinates of stars detected in the new reference image')
    log.info('Match tolerance: '+str(tol)+' deg')

    matched_stars = match_utils.StarMatchIndex()

    query = 'SELECT code_id FROM software WHERE stage="stage3"'
    software = phot_db.query_to_astropy_table(conn, query, args=())['code_id'][0]

    query = 'SELECT phot_id,star_id,x,y,image,filter,software FROM phot WHERE reference_image="'+str(refimg_id)+\
                        '" AND star_id IN '+str(tuple(starlist['star_id'].data))+\
                        ' AND software="'+str(software)+'"'
    phot_data = phot_db.query_to_astropy_table(conn, query, args=())

    field_stars = table.Table( [table.Column(name='star_id', data=starlist['star_id']),
                                table.Column(name='ra', data=starlist['ra']),
                                table.Column(name='dec', data=starlist['dec']),
                                table.Column(name='x', data=phot_data['x']),
                                table.Column(name='y', data=phot_data['y'])] )

    refframe_coords = table.Table( [ table.Column(name='star_id', data=reduction_metadata.star_catalog[1]['index']),
                                     table.Column(name='ra', data=reduction_metadata.star_catalog[1]['ra']),
                                     table.Column(name='dec', data=reduction_metadata.star_catalog[1]['dec']),
                                     table.Column(name='x', data=reduction_metadata.star_catalog[1]['x']),
                                     table.Column(name='y', data=reduction_metadata.star_catalog[1]['y']) ] )

    refframe_coords = calc_coord_offsets.transform_coordinates(setup, refframe_coords, transform_sky, coords='radec')

    #dataset_stars = SkyCoord( refframe_coords['ra'], refframe_coords['dec'],
    #                          frame='icrs', unit=(units.deg, units.deg) )

    log.info('Transformed star coordinates from the reference image')
    log.info('Matching all stars against field starlist of '+str(len(phot_data))+':')

    star_index = jdx = np.arange(0,len(refframe_coords),1)

    matched_stars = wcs.cross_match_star_catalogs(field_stars, refframe_coords, star_index, log,
                                    dra=dra, ddec=ddec, tol=tol)

#    for j in range(0,len(phot_data),1):

#        field_star = SkyCoord( starlist['ra'][j], starlist['dec'][j],
#                                frame='icrs', unit=(units.deg, units.deg) )

#        separation = field_star.separation(dataset_stars)

#        jdx = np.where(separation == separation.min())[0]

#        p = {'cat1_index': phot_data['star_id'][j],
#             'cat1_ra': starlist['ra'][j],
#             'cat1_dec': starlist['dec'][j],
#             'cat1_x': phot_data['x'][j],
#             'cat1_y': phot_data['y'][j],
#             'cat2_index': jdx[0]+1,
#             'cat2_ra': reduction_metadata.star_catalog[1]['ra'][jdx[0]],
#             'cat2_dec': reduction_metadata.star_catalog[1]['dec'][jdx[0]],
#             'cat2_x': reduction_metadata.star_catalog[1]['x'][jdx[0]],
#             'cat2_y': reduction_metadata.star_catalog[1]['y'][jdx[0]],
#             'separation': separation[jdx[0]].value}

#        if separation[jdx[0]] <= tol:
#            matched_stars.add_match(p)

#            if verbose:
#                log.info(matched_stars.summarize_last(units='deg'))
#        else:
#            log.info('No match found for '+repr(p))

#    log.info('Matched '+str(matched_stars.n_match)+' stars')

    return matched_stars

def generate_primary_ref_match_table(reduction_metadata,log):

    matched_stars = match_utils.StarMatchIndex()

    matched_stars.cat1_index = list(reduction_metadata.star_catalog[1]['index'].data)
    matched_stars.cat1_ra = list(reduction_metadata.star_catalog[1]['ra'].data)
    matched_stars.cat1_dec = list(reduction_metadata.star_catalog[1]['dec'].data)
    matched_stars.cat1_x = list(reduction_metadata.star_catalog[1]['x'].data)
    matched_stars.cat1_y = list(reduction_metadata.star_catalog[1]['y'].data)
    matched_stars.cat2_index = list(reduction_metadata.star_catalog[1]['index'].data)
    matched_stars.cat2_ra = list(reduction_metadata.star_catalog[1]['ra'].data)
    matched_stars.cat2_dec = list(reduction_metadata.star_catalog[1]['dec'].data)
    matched_stars.cat2_x = list(reduction_metadata.star_catalog[1]['x'].data)
    matched_stars.cat2_y = list(reduction_metadata.star_catalog[1]['y'].data)
    matched_stars.separation = [ 0.0 ] * len(reduction_metadata.star_catalog[1]['index'].data)
    matched_stars.n_match = len(reduction_metadata.star_catalog[1]['index'].data)

    transform = AffineTransform(matrix=np.zeros((3,3)))

    log.info('Generated matched_stars and transform for a primary reference dataset')

    return matched_stars, transform
