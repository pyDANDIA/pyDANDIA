# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 10:53:23 2019

@author: rstreet
"""
import numpy as np
import os
import sys
from astropy.io import fits
from astropy.table import Table, Column, vstack
from pyDANDIA import metadata
from pyDANDIA import logs
from pyDANDIA import phot_db
from pyDANDIA import photometry
import numpy as np


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

    stars_table = phot_db.fetch_stars_table(conn)
    log.info(' -> Extracted starlist of '+str(len(stars_table))+' objects')

    cal_stars = select_calibration_star_sample(stars_table, log,
                                               primary_ref['refimg_id_ip'],
                                                sample_size=20000)

    for f in ['ip', 'rp', 'gp']:

        if 'refimg_id_'+f in primary_ref.keys():
            primary_phot = extract_photometry_for_reference_image(conn,primary_ref,cal_stars,f,
                                                              primary_ref['refimg_id_'+f],
                                                              primary_ref['facility_id'],
                                                              log)

            ref_image_list = list_reference_images_in_filter(conn,primary_ref,f,log)

            for ref_image in ref_image_list:

                ref_phot = extract_photometry_for_reference_image(conn,primary_ref,cal_stars,f,
                                                              ref_image['refimg_id'],
                                                              ref_image['facility'],
                                                              log)

                (delta_mag, delta_mag_err) = calc_magnitude_offset(primary_phot, ref_phot, log)

                apply_magnitude_offset(conn, ref_phot, refimg_id, delta_mag, delta_mag_err, log)

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

def select_calibration_star_sample(stars_table, log, refimg_id, sample_size=20000):
    """Function to select a subset of the stars with catalogue photometry,
    since extracting all photometry from the DB is prohibitively slow.

    This function returns a dictionary of the index of stars in the stars_table
    and the stars row entries.
    """

    log.info('Selecting a sample of '+str(sample_size)+\
              ' randomly selected stars with catalogue photometry in reference image '+\
              str(refimg_id))

    cal_stars = {}

    trials = 0

    while trials < len(stars_table) and len(cal_stars) < sample_size:
        j = np.random.randint(0, len(stars_table))

        trials += 1

        s = stars_table[j]

        if s['vphas_source_id'] != None:

            cal_stars[j] = s

    return cal_stars

def extract_photometry_for_reference_image(conn,primary_ref,cal_stars,f,
                                           refimg_id,facility_id,log):
    """Function to extract from the DB the photometry for a specific
    reference image in the given filter"""

    log.info('Extracting photometry for '+str(len(cal_stars))+\
              ' randomly selected stars with photometry in reference image '+\
              str(refimg_id))

    ref_phot = Table()

    for j in cal_stars.keys():

        s = cal_stars[j]

        query = 'SELECT phot_id, star_id, hjd, calibrated_mag, calibrated_mag_err, calibrated_flux, calibrated_flux_err FROM phot WHERE reference_image="'+str(refimg_id)+\
                '" AND image="'+str(refimg_id)+\
                '" AND star_id="'+str(s['star_id'])+\
                '" AND software="'+str(primary_ref['software_id'])+\
                '" AND filter="'+str(primary_ref[f])+\
                '" AND facility="'+str(facility_id)+'"'

        star_phot = phot_db.query_to_astropy_table(conn, query, args=())

        if len(ref_phot) == 0 and len(star_phot) > 0 and star_phot['calibrated_mag'] != 0.0:
            ref_phot = star_phot
        elif len(ref_phot) > 0 and len(star_phot) > 0:
            ref_phot = vstack([ref_phot, star_phot])

        if len(ref_phot)%1000 == 0:
            log.info(' -> Extracted photometry for '+str(len(ref_phot))+' stars out of '+str(len(cal_stars)))

    if len(ref_phot) > 0:
        log.info(' -> Extracted photometry for '+str(len(ref_phot))+' stars for reference image '+str(refimg_id)+' and filter '+f)
    else:
        log.info(' -> No photometry available for reference image '+str(refimg_id)+' and filter '+f)

    return ref_phot

def list_reference_images_in_filter(conn,primary_ref,f,log):
    """Function to identify all available datasets in the given
    filter for this field, not including the primary reference dataset.

    Returns:
        QuerySet of matching reference_image entries
    """

    log.info('Identifying all current reference image in filter '+str(f))

    query = 'SELECT * FROM reference_images WHERE filter="'+str(primary_ref[f])+\
            '" AND software="'+str(primary_ref['software_id'])+\
            '" AND facility!="'+str(primary_ref['facility_id'])+'"'

    ref_image_list = phot_db.query_to_astropy_table(conn, query, args=())

    log.info(repr(ref_image_list))

    return ref_image_list

def calc_magnitude_offset(primary_phot, ref_phot, log):
    """Function to calculate the weighted average magnitude offset between the
    measured magnitudes of stars in the reference image of a dataset relative
    to the primary reference dataset in that passband."""

    log.info('Computing the magnitude offset between the primary and reference photometry')

    numer = 0.0
    denom = 0.0
    for j,star in enumerate(primary_phot['star_id']):
        mag_pri_ref = primary_phot['calibrated_mag'][j]
        merr_pri_ref = primary_phot['calibrated_mag_err'][j]

        jj = np.where(ref_phot['star_id'] == star)

        if len(ref_phot[jj]) > 0 and merr_pri_ref != np.nan and merr_pri_ref > 0.0:
            mag_ref = ref_phot['calibrated_mag'][jj][0]
            merr_ref = ref_phot['calibrated_mag_err'][jj][0]

            if merr_ref != np.nan and merr_ref > 0.0:
                sigma = np.sqrt(merr_pri_ref*merr_pri_ref + merr_ref*merr_ref)

                numer += (mag_pri_ref - mag_ref) / (sigma*sigma)
                denom += (1/sigma*sigma)

                print('Sigma:',sigma)
                print((mag_pri_ref - mag_ref), (sigma*sigma))
                print(numer, denom)

                if np.isnan(numer):
                    print(mag_ref, merr_ref)
                    print(map_pri_ref, merr_pri_ref)
                    exit()

    delta_mag = numer / denom
    delta_mag_err = 1.0 / denom

    log.info(' -> Delta_mag = '+str(delta_mag)+', delta_mag_err = '+str(delta_mag_err))

    return delta_mag, delta_mag_err

def apply_magnitude_offset(conn, ref_phot, refimg_id, delta_mag, delta_mag_err, log):
    """Function to apply the calculated magnitude offset to ALL photometry
    calculated relative to the reference image in the database."""

    log.info('Applying the magnitude offset to all photometry calculated using reference image '+str(refimg_id))

    query = 'SELECT phot_id, star_id, hjd, calibrated_mag, calibrated_mag_err, calibrated_flux, calibrated_flux_err FROM phot WHERE reference_image="'+str(refimg_id)+'"'
    phot_data = phot_db.query_to_astropy_table(conn, query, args=())

    values = []
    for dp in phot_data:

        dp['calibrated_mag'] += delta_mag
        dp['calibrated_mag_err'] = np.sqrt(dp['calibrated_mag_err']*dp['calibrated_mag_err'] + delta_mag_err*delta_mag_err)

        (cal_flux, cal_flux_error) = photometry.convert_mag_to_flux(dp['calibrated_mag_err'],
                                                                    dp['calibrated_mag_err'])
        dp['calibrated_flux'] = cal_flux
        dp['calibrated_flux_err'] = cal_flux_error

        values.append( ( str(dp['phot_id']), str(dp['star_id']), str(dp['hjd']),
                        str(dp['calibrated_mag']), str(dp['calibrated_mag_err']),
                        str(dp['calibrated_flux']), str(dp['calibrated_flux_err']) ) )

    command = 'INSERT OR REPLACE INTO phot (phot_id, star_id, hjd, calibrated_mag, calibrated_mag_err, calibrated_flux, calibrated_flux_err) VALUES (?,?,?,?,?,?,?)'

    cursor = conn.cursor()

    cursor.executemany(command, values)

    conn.commit()
