# -*- coding: utf-8 -*-
"""
Created on Tue May 21 20:49:49 2019

@author: rstreet
"""

from sys import argv
import sqlite3
from os import getcwd, path, remove, environ
import numpy as np
from astropy import table
from astropy.coordinates import SkyCoord
from astropy import units as u
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from pyDANDIA import  photometry_classes
from pyDANDIA import  phot_db
from pyDANDIA import  logs
from pyDANDIA import  event_colour_analysis
from pyDANDIA import  spectral_type_data
from pyDANDIA import  red_clump_utilities
from pyDANDIA import  config_utils
from pyDANDIA import  lightcurves
from pyDANDIA import  metadata
from pyDANDIA import  crossmatch

def run_field_colour_analysis():
    """Function to analyse the colour information for a given field pointing"""

    config = get_args()

    log = logs.start_stage_log( config['output_dir'], 'analyse_cmd' )

    event_model = config_utils.load_event_model(config['event_model_parameters_file'], log)

    (source, blend) = calc_source_blend_params(config,event_model,log)

    (photometry, stars) = extract_reference_instrument_calibrated_photometry(config,log)

    target = load_target_timeseries_photometry(config,photometry,log)

    selected_stars = find_stars_close_to_target(config,stars,target,log)

    photometry = calculate_colours(photometry,stars,log)

    source = calc_source_lightcurve(source, target, log)

    # Add QC?
    selected_phot = extract_local_star_photometry(photometry,selected_stars,log)

    RC = localize_red_clump_db(config,photometry,stars,selected_phot,log)

    RC = event_colour_analysis.measure_RC_offset(config,RC,log)
    photometry_classes.output_red_clump_data_latex(config, RC, log)

    plot_colour_mag_diagram(config, photometry, stars, selected_stars, selected_phot, RC, source, blend, 'r', 'i', 'i', log)
    plot_colour_mag_diagram(config, photometry, stars, selected_stars, selected_phot, RC, source, blend, 'r', 'i', 'r', log)
    plot_colour_mag_diagram(config, photometry, stars, selected_stars, selected_phot, RC, source, blend, 'g', 'r', 'g', log)
    plot_colour_mag_diagram(config, photometry, stars, selected_stars, selected_phot, RC, source, blend, 'g', 'i', 'g', log)
    plot_colour_mag_diagram(config, photometry, stars, selected_stars, selected_phot, RC, source, blend, 'g', 'i', 'i', log)

    plot_colour_colour_diagram(config, photometry, selected_phot, RC, source, blend, log)

    output_photometry(config, stars, photometry, selected_stars, log)

    source.output_json(path.join(config['output_dir'],'source_parameters.json'))
    blend.output_json(path.join(config['output_dir'],'blend_parameters.json'))
    RC.output_json(path.join(config['output_dir'],'red_clump_parameters.json'))

    logs.close_log(log)

def get_args():

    if len(argv) == 1:

        config_file = input('Please enter the path to the JSON configuration file: ')

    else:

        config_file = argv[1]

    config = config_utils.build_config_from_json(config_file)

    # Handle those keywords which may have Boolean entries
    boolean_keys = ['interactive', 'add_rc_centroid', 'add_blend',
                    'add_source', 'add_source_trail', 'add_crosshairs',
                    'plot_selected_radius_only']
    for key in boolean_keys:
        if 'true' in str(config[key]).lower():
            config[key] = True
        else:
            config[key] = False

    # Handle those keywords which may have None values:
    none_allowed_keys = ['target_field_id','db_file_path','xmatch_file_path']
    for key in none_allowed_keys:
        if 'none' in str(config[key]).lower():
            config[key] = None

    # Handle dictionary keywords which may have None entries
    none_allowed_keys = ['target_lightcurve_files', 'red_dirs']
    for main_key in none_allowed_keys:
        orig_dict = config[main_key]
        new_dict = {}
        for key, value in orig_dict.items():
            if 'none' == str(value).lower():
                new_dict[key] = None
            else:
                new_dict[key] = value
        config[main_key] = new_dict

    # Sanity check configuration:
    if not config['db_file_path'] and not config['xmatch_file_path']:
        raise IOError('Error in configuration: need to specify either a photometry database or a cross-match table')
    if config['db_file_path'] and config['xmatch_file_path']:
        raise IOError('Error in configuration: need to specify either a photometry database OR a cross-match table, not both')

    if config['xmatch_file_path']:
        n = 0
        for key in ['g', 'r', 'i']:
            if config['red_dirs'][key] == None:
                n+=1
        if n > 1:
            raise IOError('Insufficient reduction directories specified for colour analysis')

    return config


def calc_source_blend_params(config,event_model,log):
    """Function to construct a dictionary of needed parameters for the
    source and blend"""

    source = photometry_classes.Star()
    blend = photometry_classes.Star()

    if len(event_model) > 0:
        filterset = ['g','r','i']
        log.info('Using the following datasets as the flux references for the source and blend fluxes:')
        for f in filterset:

            ref_dataset = config['flux_reference_datasets'][f]
            log.info(ref_dataset)

            if event_model['source_fluxes'][ref_dataset] != None and event_model['source_flux_errors'][ref_dataset] != None:
                setattr(source, 'fs_'+f, event_model['source_fluxes'][ref_dataset])
                setattr(source, 'sig_fs_'+f, event_model['source_flux_errors'][ref_dataset])
                source.convert_fluxes_pylima(f)

            log.info('Source: '+repr(event_model['source_fluxes'][ref_dataset])+' +/- '+repr(event_model['source_flux_errors'][ref_dataset]))

            if event_model['blend_fluxes'][ref_dataset] != None and event_model['blend_flux_errors'][ref_dataset] != None:
                setattr(blend, 'fs_'+f, event_model['blend_fluxes'][ref_dataset])
                setattr(blend, 'sig_fs_'+f, event_model['blend_flux_errors'][ref_dataset])
                blend.convert_fluxes_pylima(f)

            log.info('Blend: '+repr(event_model['blend_fluxes'][ref_dataset])+' +/- '+repr(event_model['blend_flux_errors'][ref_dataset]))

        source.compute_colours(use_inst=True)
        source.transform_to_JohnsonCousins()

        log.info('\n')
        log.info('Source measured photometry:')
        log.info(source.summary(show_mags=True))
        log.info(source.summary(show_mags=False,show_colours=True))
        log.info(source.summary(show_mags=False,johnsons=True))

        blend.compute_colours(use_inst=True)
        blend.transform_to_JohnsonCousins()

        log.info('\n')
        log.info('Blend measured photometry:')
        log.info(blend.summary(show_mags=True))
        log.info(blend.summary(show_mags=False,show_colours=True))
        log.info(blend.summary(show_mags=False,johnsons=True))

    else:
        log.info('No event model supplied, so no source and blend information available')

    return source, blend

def fetch_star_phot(star_id,phot_table):
        jdx = np.where(phot_table['star_id'] == star_id)[0]

        if len(jdx) == 0:
            return 0.0,0.0
        else:
            return phot_table['calibrated_mag'][jdx],phot_table['calibrated_mag_err'][jdx]

def extract_reference_instrument_calibrated_photometry(config,log):
    """Function to extract from the calibrated photometry for stars
    in the field from the data from the photometric reference instrument.
    """

    if config['db_file_path']:
        (photometry, stars) = get_reference_photometry_from_db(config, log)
    else:
        (photometry, stars) = get_reference_photometry_from_metadata(config, log)

    return photometry, stars

def get_reference_photometry_from_db(config, log):
    """Function to extract the calibrated photometry from the
    photometric database for the primary reference instrument, which is
    defined to be lsc.doma.1m0a.fa15 by default.
    """

    log.info('Extracting the calibrated reference instrument photometry from the photometric database')

    conn = phot_db.get_connection(dsn=config['db_file_path'])

    facility_code = phot_db.get_facility_code({'site': 'lsc',
                                               'enclosure': 'doma',
                                               'telescope': '1m0a',
                                               'instrument': 'fa15'})

    query = 'SELECT facility_id, facility_code FROM facilities WHERE facility_code="'+facility_code+'"'
    t = phot_db.query_to_astropy_table(conn, query, args=())
    if len(t) == 0:
        raise IOError('No photometry for primary reference facility '+facility_code+' found in phot_db')

    facility_id = t['facility_id'][0]

    stars = phot_db.fetch_stars_table(conn)

    filters = {'g': None, 'r': None, 'i': None}
    refimgs = {'g': None, 'r': None, 'i': None}
    for f in filters.keys():

        filter_name = f+'p'

        query = 'SELECT filter_id, filter_name FROM filters WHERE filter_name="'+filter_name+'"'
        t = phot_db.query_to_astropy_table(conn, query, args=())
        filters[f] = t['filter_id'][0]

        query = 'SELECT refimg_id FROM reference_images WHERE facility="'+str(facility_id)+'" AND filter="'+str(filters[f])+'"'
        t = phot_db.query_to_astropy_table(conn, query, args=())
        if len(t) > 0:
            refimgs[f] = t['refimg_id'][0]
        else:
            raise IOError('No reference image data in DB for facility '+str(facility_code)+'.')

    photometry = {'phot_table_g': [], 'phot_table_r': [], 'phot_table_i': []}

    log.info('-> Extracting photometry for stars')

    for j,star in enumerate(stars):
        for f,fid in filters.items():
            query = 'SELECT star_id, calibrated_mag, calibrated_mag_err FROM phot WHERE star_id="'+str(star['star_id'])+\
                            '" AND facility="'+str(facility_id)+\
                            '" AND filter="'+str(fid)+\
                            '" AND software="1'+\
                            '" AND reference_image="'+str(refimgs[f])+\
                            '" AND phot_type="PSF_FITTING"'

            data = phot_db.query_to_astropy_table(conn, query, args=())

            if len(data) > 0:
                photometry['phot_table_'+f].append([data['star_id'][0], 0.0,0.0, 0.0, 0.0, data['calibrated_mag'][0], data['calibrated_mag_err'][0], 'None'])
            else:
                photometry['phot_table_'+f].append([star['star_id'],0.0,0.0,0.0,0.0, 0.0, 0.0, 'None'])

        if j%1000.0 == 0.0:
            print('--> Completed '+str(j)+' stars out of '+str(len(stars)))

    conn.close()

    (photometry, stars) = repack_photometry(photometry, stars, log)

    return photometry, stars

def repack_photometry(photometry, stars, log):
    for f in ['g', 'r', 'i']:
        table_data = np.array(photometry['phot_table_'+f])
        if len(table_data) > 0:
            data = [table.Column(name='star_id',
                            data=table_data[:,0]),
                    table.Column(name='x',
                            data=table_data[:,1]),
                    table.Column(name='y',
                            data=table_data[:,2]),
                    table.Column(name='ra',
                            data=table_data[:,3]),
                    table.Column(name='dec',
                            data=table_data[:,4]),
                    table.Column(name='calibrated_mag',
                            data=table_data[:,5]),
                    table.Column(name='calibrated_mag_err',
                            data=table_data[:,6]),
                    table.Column(name='gaia_source_id',
                            data=table_data[:,7])]
        else:
            data = [table.Column(name='star_id',
                            data=[]),
                    table.Column(name='x',
                            data=[]),
                    table.Column(name='y',
                            data=[]),
                    table.Column(name='ra',
                            data=[]),
                    table.Column(name='dec',
                            data=[]),
                    table.Column(name='calibrated_mag',
                            data=[]),
                    table.Column(name='calibrated_mag_err',
                            data=[]),
                    table.Column(name='gaia_source_id',
                            data=[])]
        photometry['phot_table_'+f] = table.Table(data=data)

    log.info('Extracted photometry for '+str(len(stars))+' stars')

    photometry['g'] = np.zeros(len(stars))
    photometry['gerr'] = np.zeros(len(stars))
    photometry['r'] = np.zeros(len(stars))
    photometry['rerr'] = np.zeros(len(stars))
    photometry['i'] = np.zeros(len(stars))
    photometry['ierr'] = np.zeros(len(stars))
    photometry['x'] = np.zeros(len(stars))
    photometry['y'] = np.zeros(len(stars))
    photometry['ra'] = np.zeros(len(stars))
    photometry['dec'] = np.zeros(len(stars))
    photometry['gaia_source_id'] = np.empty(len(stars), dtype="S20")

    for s in range(0,len(stars),1):

        sid = stars['star_id'][s]
        photometry['x'][s] = stars['x'][s]
        photometry['y'][s] = stars['y'][s]
        photometry['ra'][s] = stars['ra'][s]
        photometry['dec'][s] = stars['dec'][s]
        photometry['gaia_source_id'][s] = stars['gaia_source_id'][s]

        (photometry['g'][s],photometry['gerr'][s]) = fetch_star_phot(sid,photometry['phot_table_g'])
        (photometry['r'][s],photometry['rerr'][s]) = fetch_star_phot(sid,photometry['phot_table_r'])
        (photometry['i'][s],photometry['ierr'][s]) = fetch_star_phot(sid,photometry['phot_table_i'])

    return photometry, stars

def get_reference_photometry_from_metadata(config, log):
    """Function to extract the calibrated photometry from the crossmatch table
    and HDF photometry files.
    """

    log.info('Extracting the calibrated reference instrument photometry from the cross-match table and HDF5 photometry files')

    xmatch = crossmatch.CrossMatchTable()
    xmatch.load(config['xmatch_file_path'])

    photometry = {'phot_table_g': [], 'phot_table_r': [], 'phot_table_i': []}

    log.info('-> Extracting photometry for primary reference i-band dataset')
    primary_metadata = metadata.MetaData()
    primary_metadata.load_all_metadata(config['red_dirs']['i'], 'pyDANDIA_metadata.fits')

    stars = convert_star_catalog_to_stars_table(primary_metadata)

    for j in range(0,len(stars),1):
        data = primary_metadata.star_catalog[1][j]
        photometry['phot_table_i'].append([data['index'], data['x'], data['y'], data['ra'], data['dec'], data['cal_ref_mag'], data['cal_ref_mag_error'], data['gaia_source_id']])

    for f in ['g', 'r']:
        if config['red_dirs'][f]:
            log.info('-> Extracting photometry for '+f+'-band data')

            dataset_metadata = metadata.MetaData()
            dataset_metadata.load_all_metadata(config['red_dirs'][f], 'pyDANDIA_metadata.fits')

            matched_stars = xmatch.fetch_match_table_for_reduction(config['red_dirs'][f])

            if matched_stars.n_match > 0:
                log.info('--> Extracting data for '+str(matched_stars.n_match)+' matched stars')

                for star in stars:
                    if star['star_id'] in matched_stars.cat1_index:
                        j = matched_stars.cat1_index.index(star['star_id'])
                        dataset_j = matched_stars.cat2_index[j] - 1

                        data = dataset_metadata.star_catalog[1][dataset_j]
                        photometry['phot_table_'+f].append([star['star_id'], star['x'], star['y'], star['ra'], star['dec'], data['cal_ref_mag'], data['cal_ref_mag_error'], star['gaia_source_id']])
                    else:
                        photometry['phot_table_'+f].append([star['star_id'],0.0,0.0,0.0,0.0,0.0,0.0,'None'])

            else:
                log.info('No stars matched for dataset '+config['red_dirs'][f])

    (photometry, stars) = repack_photometry(photometry, stars, log)

    return photometry, stars

def convert_star_catalog_to_stars_table(reduction_metadata):

    star_catalog = reduction_metadata.star_catalog[1]

    table_data = [table.Column(name='star_id', data=star_catalog['index']),
                  table.Column(name='star_index', data=star_catalog['index']),
                  table.Column(name='x', data=star_catalog['x']),
                  table.Column(name='y', data=star_catalog['y']),
                  table.Column(name='ra', data=star_catalog['ra']),
                  table.Column(name='dec', data=star_catalog['dec']),
                  table.Column(name='reference_image', data=[-1]*len(star_catalog)),
                  table.Column(name='gaia_source_id', data=star_catalog['gaia_source_id']),
                  table.Column(name='gaia_ra', data=star_catalog['gaia_ra']),
                  table.Column(name='gaia_ra_error', data=star_catalog['gaia_ra_error']),
                  table.Column(name='gaia_dec', data=star_catalog['gaia_dec']),
                  table.Column(name='gaia_dec_error', data=star_catalog['gaia_dec_error']),
                  table.Column(name='phot_g_mean_flux', data=star_catalog['phot_g_mean_flux']),
                  table.Column(name='phot_g_mean_flux_error', data=star_catalog['phot_g_mean_flux_error']),
                  table.Column(name='phot_bp_mean_flux', data=star_catalog['phot_bp_mean_flux']),
                  table.Column(name='phot_bp_mean_flux_error', data=star_catalog['phot_bp_mean_flux_error']),
                  table.Column(name='phot_rp_mean_flux', data=star_catalog['phot_rp_mean_flux']),
                  table.Column(name='phot_rp_mean_flux_error', data=star_catalog['phot_rp_mean_flux_error']),
                  table.Column(name='vphas_source_id', data=star_catalog['vphas_source_id']),
                  table.Column(name='vphas_ra', data=star_catalog['vphas_ra']),
                  table.Column(name='vphas_dec', data=star_catalog['vphas_dec']),
                  table.Column(name='vphas_gmag', data=star_catalog['gmag']),
                  table.Column(name='vphas_gmag_error', data=star_catalog['gmag_error']),
                  table.Column(name='vphas_rmag', data=star_catalog['rmag']),
                  table.Column(name='vphas_rmag_error', data=star_catalog['rmag_error']),
                  table.Column(name='vphas_imag', data=star_catalog['imag']),
                  table.Column(name='vphas_imag_error', data=star_catalog['imag_error']),
                  table.Column(name='vphas_clean', data=star_catalog['clean'])]

    return table.Table(data=table_data)

def find_stars_close_to_target(config,stars,target,log):
    """Function to identify those stars which are within the search radius of
    the target_ra, dec given"""

    if config['selection_radius'] > 0.0:
        tol = config['selection_radius'] / 60.0

        det_stars = SkyCoord(stars['ra'], stars['dec'], unit="deg")

        t = SkyCoord(target.ra, target.dec, unit="deg")
        seps = det_stars.separation(t)

        jdx = np.where(seps.deg < tol)[0]

        log.info('Identified '+str(len(jdx))+' stars within '+str(round(tol*60.0,1))+\
                'arcmin of the target')

        if len(jdx) == 0:
            raise ValueError('No stars identified within the given selection radius')

    else:
        jdx = np.arange(0,len(stars),1)

        log.info('No selection radius given, so using all stars within field of view')

    return jdx

def extract_local_star_photometry(photometry,selected_stars,log,extract_errors=True):
    """Function to extract the photometry for stars local to the given
    target coordinates"""

    g = table.Column(data=photometry['g'][selected_stars], name='g')
    r = table.Column(data=photometry['r'][selected_stars], name='r')
    i = table.Column(data=photometry['i'][selected_stars], name='i')
    gr = table.Column(data=photometry['gr'][selected_stars], name='gr')
    ri = table.Column(data=photometry['ri'][selected_stars], name='ri')
    gi = table.Column(data=photometry['gi'][selected_stars], name='gi')

    if extract_errors:
        gerr = table.Column(data=photometry['gerr'][selected_stars], name='gerr')
        rerr = table.Column(data=photometry['rerr'][selected_stars], name='rerr')
        ierr = table.Column(data=photometry['ierr'][selected_stars], name='ierr')
        gr_err = table.Column(data=photometry['gr_err'][selected_stars], name='gr_err')
        ri_err = table.Column(data=photometry['ri_err'][selected_stars], name='ri_err')
        gi_err = table.Column(data=photometry['gi_err'][selected_stars], name='gi_err')

        selected_phot = table.Table([g,gerr,r,rerr,i,ierr,gr,gr_err,gi,gi_err,ri, ri_err])

    else:
        selected_phot = table.Table([g,r,i,gr,gi,ri])

    log.info('Extracted the photometry for '+str(len(selected_phot))+\
             ' stars close to the target coordinates')

    return selected_phot

def calculate_colours(photometry,stars,log):

    def calc_colour_data(blue_index, red_index, blue_phot, blue_phot_err,
                                                red_phot, red_phot_err):

        col_index = list(set(blue_index).intersection(set(red_index)))

        col_data = np.zeros(len(red_phot))
        col_data.fill(-99.999)
        col_data_err = np.zeros(len(red_phot))
        col_data_err.fill(-99.999)

        col_data[col_index] = blue_phot[col_index] - red_phot[col_index]

        col_data_err[col_index] = np.sqrt( (blue_phot_err[col_index]*blue_phot_err[col_index])  + \
                                            (red_phot_err[col_index]*red_phot_err[col_index]) )

        return col_data, col_data_err

    gdx = np.where(photometry['g'] != 0.0)[0]
    rdx = np.where(photometry['r'] != 0.0)[0]
    idx = np.where(photometry['i'] != 0.0)[0]

    (photometry['gr'],photometry['gr_err']) = calc_colour_data(gdx, rdx,
                                         photometry['g'], photometry['gerr'],
                                         photometry['r'], photometry['rerr'])
    (photometry['gi'],photometry['gi_err']) = calc_colour_data(gdx, idx,
                                         photometry['g'], photometry['gerr'],
                                         photometry['i'], photometry['ierr'])
    (photometry['ri'],photometry['ri_err']) = calc_colour_data(rdx, idx,
                                         photometry['r'], photometry['rerr'],
                                         photometry['i'], photometry['ierr'])

    log.info('Calculated colour data for stars detected in ROME data')

    gdx = np.where(stars['vphas_gmag'] != 0.0)[0]
    rdx = np.where(stars['vphas_rmag'] != 0.0)[0]
    idx = np.where(stars['vphas_imag'] != 0.0)[0]

    photometry['gr_cat'] = calc_colour_data(gdx, rdx,
                                         stars['vphas_gmag'], stars['vphas_gmag_error'],
                                         stars['vphas_rmag'], stars['vphas_rmag_error'])
    photometry['gi_cat'] = calc_colour_data(gdx, idx,
                                         stars['vphas_gmag'], stars['vphas_gmag_error'],
                                         stars['vphas_imag'], stars['vphas_imag_error'])
    photometry['ri_cat'] = calc_colour_data(rdx, idx,
                                         stars['vphas_rmag'], stars['vphas_rmag_error'],
                                         stars['vphas_imag'], stars['vphas_imag_error'])

    log.info('Calculated VPHAS catalog colours for all stars identified within field')

    return photometry

def load_target_timeseries_photometry(config,photometry,log):
    """Function to read in timeseries photometry extracted for a single star"""

    target = photometry_classes.Star()

    if 'None' not in config['target_ra'] and 'None' not in config['target_dec']:
        t = SkyCoord(config['target_ra']+' '+config['target_dec'],
                        unit=(u.hourangle,u.degree), frame='icrs')
        target.ra = t.ra.value
        target.dec = t.dec.value

    if config['target_field_id'] != None:

        target.star_index = config['target_field_id']

        (target.g,target.sig_g) = fetch_star_phot(target.star_index,photometry['phot_table_g'])
        (target.r,target.sig_r) = fetch_star_phot(target.star_index,photometry['phot_table_r'])
        (target.i,target.sig_i) = fetch_star_phot(target.star_index,photometry['phot_table_i'])

        log.info('\n')
        log.info('Target identified as star '+str(target.star_index)+\
                    ' in the combined ROME catalog, with parameters:')
        log.info('RA = '+str(target.ra)+' Dec = '+str(target.dec))
        log.info('Measured ROME photometry, calibrated to the VPHAS+ scale:')
        log.info(target.summary(show_mags=True))

        if target.i != None and target.r != None:

            target.compute_colours(use_inst=True)

            log.info(target.summary(show_mags=False,show_colours=True))

        target.transform_to_JohnsonCousins()

        log.info(target.summary(show_mags=False,johnsons=True))

    for f in ['i', 'r', 'g']:
        file_path = config['target_lightcurve_files'][f]

        if file_path != None:

            data = lightcurves.read_pydandia_lightcurve(file_path, skip_zero_entries=True)

            lc = table.Table()
            lc['hjd'] = data['hjd']
            lc['mag'] = data['calibrated_mag']
            lc['mag_err'] = data['calibrated_mag_err']
            (fluxes,fluxerrs) = photometry_classes.mag_to_flux_pylima(lc['mag'],lc['mag_err'])
            lc['flux'] = fluxes
            lc['flux_err'] = fluxerrs

            target.lightcurves[f] = lc

            log.info('Read '+str(len(lc))+' datapoints from the '+f\
                        +'-band lightcurve for the target')

        else:

            log.info('No lightcurve file specified for the target in '+f+'-band')

    return target

def calc_source_lightcurve(source, target, log):
    """Function to calculate the lightcurve of the source, based on the
    model source flux and the change in magnitude from the lightcurve"""

    log.info('\n')

    for f in ['i', 'r', 'g']:
        if target.lightcurves[f] != None:
            idx = np.where(target.lightcurves[f]['mag_err'] > 0)[0]

            dmag = np.zeros(len(target.lightcurves[f]['mag']))
            dmag.fill(99.99999)
            dmerr = np.zeros(len(target.lightcurves[f]['mag']))
            dmerr.fill(-9.9999)

            dmag[idx] = target.lightcurves[f]['mag'][idx] - getattr(target,f)
            dmerr[idx] = np.sqrt( (target.lightcurves[f]['mag_err'][idx])**2 + getattr(target,'sig_'+f)**2 )

            lc = table.Table()
            lc['hjd'] = target.lightcurves[f]['hjd']
            if getattr(source,f) != None:
                lc['mag'] = getattr(source,f) + dmag
            else:
                lc['mag'] = dmag
            lc['mag_err'] = np.zeros(len(lc['mag']))
            lc['mag_err'] = dmerr

            if getattr(source,'sig_'+f) != None:
                lc['mag_err'][idx] = np.sqrt( dmerr[idx]*dmerr[idx] + (getattr(source,'sig_'+f))**2 )
            else:
                lc['mag_err'][idx] = dmerr[idx]

            log.info('Calculated the source flux lightcurve in '+f)

            source.lightcurves[f] = lc

    return source

def plot_data_colours():
    default_marker_colour = '#8c6931'
    field_marker_colour = '#E1AE13'
    marker_colour = default_marker_colour
    return default_marker_colour, field_marker_colour, marker_colour

def plot_colour_mag_diagram(params, photometry, stars, selected_stars, selected_phot,
                            RC, source, blend, blue_filter, red_filter,
                            yaxis_filter, log):
    """Function to plot a colour-magnitude diagram, highlighting the data for
    local stars close to the target in a different colour from the rest,
    and indicating the position of both the target and the Red Clump centroid.
    """

    def calc_colour_lightcurve(blue_lc, red_lc, y_lc):

        idx1 = np.where( red_lc['mag_err'] > 0.0 )[0]
        idx2 = np.where( blue_lc['mag_err'] > 0.0 )[0]
        idx3 = np.where( y_lc['mag_err'] > 0.0 )[0]
        idx = set(idx1).intersection(set(idx2))
        idx = list(idx.intersection(set(idx3)))

        mags = y_lc['mag'][idx]
        magerr = y_lc['mag_err'][idx]
        cols = blue_lc['mag'][idx] - red_lc['mag'][idx]
        colerr = np.sqrt(blue_lc['mag_err'][idx]**2 + red_lc['mag_err'][idx]**2)

        return mags, magerr, cols, colerr

    def select_valid_data(phot_array, params, col_key, col_err_key, yaxis_filter, y_err_key):
        cdx = np.where(phot_array[col_key] != -99.999)[0]
        cdx2 = np.where(phot_array[col_err_key] <= params[col_key+'_sigma_max'])[0]
        #cdx2 = np.where(phot_array[col_err_key] < phot_array[col_key])[0]
        mdx = np.where(phot_array[yaxis_filter] != 0.0)[0]
        mdx2 = np.where(phot_array[y_err_key] <= params[yaxis_filter+'_sigma_max'])[0]
        #mdx2 = np.where(phot_array[y_err_key] < phot_array[yaxis_filter])[0]
        jdx = list(set(cdx).intersection(set(cdx2)))
        jdx = list(set(jdx).intersection(set(mdx)))
        jdx = list(set(jdx).intersection(set(mdx2)))

        return jdx

    col_key = blue_filter+red_filter
    col_err_key = blue_filter+red_filter+'_err'
    y_err_key = yaxis_filter+'err'

    fig = plt.figure(1,(10,10))

    ax = plt.subplot(111)

    plt.rcParams.update({'font.size': 25})

    # Selection for full starlist
    jdx = select_valid_data(photometry, params, col_key, col_err_key, yaxis_filter, y_err_key)

    # Selection for sub-region only
    if params['selection_radius'] > 0.0:
        jdx_region = select_valid_data(selected_phot, params, col_key, col_err_key, yaxis_filter, y_err_key)

    (default_marker_colour, field_marker_colour, marker_colour) = plot_data_colours()
    if len(selected_stars) < len(photometry['i']):
        marker_colour = field_marker_colour

    if not params['plot_selected_radius_only']:
        plt.scatter(photometry[col_key][jdx],photometry[yaxis_filter][jdx],
                 c=marker_colour, marker='.', s=1,
                 label='Stars within field of view')

    if params['selection_radius'] > 0.0:
        plt.scatter(selected_phot[col_key][jdx_region],selected_phot[yaxis_filter][jdx_region],
                  c=default_marker_colour, marker='*', s=5,
                  label='Stars < '+str(round(params['selection_radius'],1))+'arcmin of target')

    if params['plot_selected_radius_only'] and params['selection_radius'] <= 0.0:
        raise IOError('Configuration indicates only stars within a selected radius should be plotted but selection radius <= 0.0arcmin')

    if params['add_rc_centroid']:
        plt.errorbar(getattr(RC,col_key), getattr(RC,yaxis_filter),
                 yerr=getattr(RC,'sig_'+yaxis_filter),
                 xerr=getattr(RC,'sig_'+col_key),
                 color='g', marker='s',markersize=10, label='Red Clump centroid')

    if getattr(blend,blue_filter) != None and getattr(blend,red_filter) != None \
        and params['add_blend']:

        plt.errorbar(getattr(blend,col_key), getattr(blend,yaxis_filter),
                 yerr = getattr(blend,'sig_'+yaxis_filter),
                 xerr = getattr(blend,'sig_'+col_key), color='b',
                 marker='v',markersize=10, label='Blend')

    if getattr(source,blue_filter) != None and getattr(source,red_filter) != None\
        and params['add_source']:

        plt.errorbar(getattr(source,col_key), getattr(source,yaxis_filter),
                 yerr = getattr(source,'sig_'+yaxis_filter),
                 xerr = getattr(source,'sig_'+col_key), color='m',
                 marker='d',markersize=10, label='Source crosshairs')

        if params['add_crosshairs']:
            plot_crosshairs(fig,getattr(source,col_key),getattr(source,yaxis_filter),'m')

        if params['add_source_trail']:
            red_lc = source.lightcurves[red_filter]
            blue_lc = source.lightcurves[blue_filter]
            y_lc = source.lightcurves[yaxis_filter]

            (smags, smagerr, scols, scolerr) = calc_colour_lightcurve(blue_lc, red_lc, y_lc)

            plt.errorbar(scols, smags, yerr = smagerr, xerr = scolerr,
                         color='m', marker='d',markersize=5, fmt='none', label='Source')

    plt.xlabel('SDSS ('+blue_filter+'-'+red_filter+') [mag]')

    plt.ylabel('SDSS-'+yaxis_filter+' [mag]')

    [xmin,xmax,ymin,ymax] = plt.axis()
    xmin = params['plot_'+col_key+'_range'][0]
    xmax = params['plot_'+col_key+'_range'][1]
    ymin = params['plot_'+yaxis_filter+'_range'][0]
    ymax = params['plot_'+yaxis_filter+'_range'][1]
    plt.axis([xmin,xmax,ymax,ymin])

    xticks = np.arange(xmin,xmax,0.1)
    yticks = np.arange(ymin,ymax,0.2)

    #ax.set_xticks(xticks,minor=True)
    ax.set_xticklabels(xticks,minor=True, fontdict={'size': 25})
    #ax.set_yticks(yticks,minor=True)
    ax.set_yticklabels(yticks,minor=True,fontdict={'size': 25})
    ax.title.set_size(25)
    ax.xaxis.label.set_fontsize(25)
    ax.yaxis.label.set_fontsize(25)
    ax.tick_params(axis="x", labelsize=25)
    ax.tick_params(axis="y", labelsize=25)

    plot_file = path.join(params['output_dir'],'colour_magnitude_diagram_'+\
                                            yaxis_filter+'_vs_'+blue_filter+red_filter\
                                            +'.pdf')

    plt.grid()

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * -0.025,
             box.width, box.height * 0.95])

    l = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=2)

    l.legendHandles[0]._sizes = [50]
    if len(l.legendHandles) > 1:
        l.legendHandles[1]._sizes = [50]

    plt.rcParams.update({'legend.fontsize':25})
    plt.rcParams.update({'font.size':25})
    plt.rcParams.update({'axes.titlesize': 25})
    plt.rcParams.update({'font.size': 25})

    if params['interactive']:
        plt.show()

    plt.savefig(plot_file,bbox_inches='tight')

    plt.close(1)

    log.info('Colour-magnitude diagram output to '+plot_file)

def plot_colour_colour_diagram(params,photometry,selected_phot,RC,source,blend,log):
    """Function to plot a colour-colour diagram, if sufficient data are
    available within the given star catalog"""

    filters = { 'i': 'SDSS-i', 'r': 'SDSS-r', 'g': 'SDSS-g' }

    (default_marker_colour, field_marker_colour, marker_colour) = plot_data_colours()
    if len(selected_phot) < len(photometry['i']):
        marker_colour = field_marker_colour

    fig = plt.figure(1,(10,10))

    ax = plt.axes()

    def select_plot_data(params,photometry):
        grx = np.where(photometry['gr'] != -99.999)[0]
        grx2 = np.where(photometry['gr_err'] <= params['gr_sigma_max'])[0]
        rix = np.where(photometry['ri'] != -99.999)[0]
        rix2 = np.where(photometry['ri_err'] <= params['ri_sigma_max'])[0]
        jdx = list(set(grx).intersection(set(grx2)))
        jdx = list(set(jdx).intersection(set(rix)))
        jdx = list(set(jdx).intersection(set(rix2)))
        return jdx

    jdx = select_plot_data(params,photometry)
    inst_gr = photometry['gr'][jdx] - RC.Egr
    inst_ri = photometry['ri'][jdx] - RC.Eri

    if params['selection_radius'] > 0.0:
        jdx_region = select_plot_data(params, selected_phot)
        region_inst_gr = selected_phot['gr'][jdx_region] - RC.Egr
        region_inst_ri = selected_phot['ri'][jdx_region] - RC.Eri

    if not params['plot_selected_radius_only']:
        ax.scatter(inst_gr, inst_ri,
                   c=marker_colour, marker='.', s=1,
                 label='Stars within field of view')

    if params['selection_radius'] > 0.0:
        plt.scatter(region_inst_gr,region_inst_ri,
                  c=default_marker_colour, marker='*', s=5,
                  label='Stars < '+str(round(params['selection_radius'],1))+'arcmin of target')

    if params['plot_selected_radius_only'] and params['selection_radius'] <= 0.0:
        raise IOError('Configuration indicates only stars within a selected radius should be plotted but selection radius <= 0.0arcmin')

    (spectral_type, luminosity_class, gr_colour, ri_colour) = spectral_type_data.get_spectral_class_data()

    plot_dwarfs = False
    plot_giants = True
    for i in range(0,len(spectral_type),1):

        spt = spectral_type[i]+luminosity_class[i]

        if luminosity_class[i] == 'V':
            c = '#8d929b'
        else:
            c = '#8d929b'

        if luminosity_class[i] == 'III' and plot_giants:

            plt.plot(gr_colour[i], ri_colour[i], marker='s', color=c,
                     markeredgecolor='k', alpha=0.5)

            plt.annotate(spt, (gr_colour[i], ri_colour[i]-0.1),
                            color='k', size=18, rotation=-30.0, alpha=1.0)

        if luminosity_class[i] == 'V' and plot_dwarfs:

            plt.plot(gr_colour[i], ri_colour[i], marker='s', color=c,
                     markeredgecolor='k', alpha=0.5)

            plt.annotate(spt, (gr_colour[i],
                           ri_colour[i]+0.1),
                             color='k', size=18,
                             rotation=-30.0, alpha=1.0)

    if params['add_source']:
        plt.errorbar(source.gr-RC.Egr, source.ri-RC.Eri,
             yerr = source.sig_ri,
             xerr = source.sig_gr, color='m',
             marker='d',markersize=10, label='Source')

    if params['add_blend']:
        plt.errorbar(blend.gr-RC.Egr, blend.ri-RC.Eri,
             yerr = blend.sig_ri,
             xerr = blend.sig_gr, color='b',
             marker='v',markersize=10, label='Blend')

    plt.xlabel('SDSS (g-r) [mag]')

    plt.ylabel('SDSS (r-i) [mag]')

    plot_file = path.join(params['output_dir'],'colour_colour_diagram.pdf')

    scale_axes = False
    if scale_axes:
        plt.axis([-2.0,2.0,-1.0,2.0])

        xticks = np.arange(-1.0,2.0,0.1)
        yticks = np.arange(-1.0,1.0,0.1)

        ax.set_xticks(xticks, minor=True)
        ax.set_yticks(yticks, minor=True)

    else:
        [xmin,xmax,ymin,ymax] = plt.axis()
        xmin = params['plot_gr_range'][0]-RC.Egr
        xmax = params['plot_gr_range'][1]-RC.Egr
        ymin = params['plot_ri_range'][0]-RC.Eri
        ymax = params['plot_ri_range'][1]-RC.Eri
        plt.axis([xmin,xmax,ymin,ymax])

        xticks = np.arange(xmin,xmax,0.1)
        yticks = np.arange(ymin,ymax,0.2)

        ax.set_xticklabels(xticks,minor=True, fontdict={'size': 25})
        ax.set_yticklabels(yticks,minor=True, fontdict={'size': 25})

    plt.grid()

    add_legend = True
    if add_legend:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * -0.025,
                 box.width, box.height * 0.95])

        l = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2)

        try:
            l.legendHandles[2]._sizes = [50]
            l.legendHandles[3]._sizes = [50]
        except IndexError:
            pass

    plt.rcParams.update({'legend.fontsize':25})
    plt.rcParams.update({'font.size':25})
    plt.rc('xtick', labelsize=25)
    plt.rc('ytick', labelsize=25)
    ax.title.set_size(25)
    ax.xaxis.label.set_fontsize(25)
    ax.yaxis.label.set_fontsize(25)
    ax.tick_params(axis="x", labelsize=25)
    ax.tick_params(axis="y", labelsize=25)

    if params['interactive']:
        plt.show()

    plt.savefig(plot_file,bbox_inches='tight')

    plt.close(1)

    log.info('Colour-colour diagram output to '+plot_file)

def plot_crosshairs(fig,xvalue,yvalue,linecolour):

    ([xmin,xmax,ymin,ymax]) = plt.axis()

    xdata = np.linspace(int(xmin),int(xmax),10.0)
    ydata = np.zeros(len(xdata))
    ydata.fill(yvalue)

    plt.plot(xdata, ydata, linecolour+'-', alpha=0.5)

    ydata = np.linspace(int(ymin),int(ymax),10.0)
    xdata = np.zeros(len(ydata))
    xdata.fill(xvalue)

    plt.plot(xdata, ydata, linecolour+'-', alpha=0.5)

def localize_red_clump_db(config,photometry,stars,selected_phot,log):
    """Function to calculate the centroid of the Red Clump stars in a
    colour-magnitude diagram"""

    def select_within_range(mags, colours, mag_min, mag_max, col_min, col_max):
        """Function to identify the set of array indices with values
        between the range indicated"""

        idx1 = np.where(colours >= col_min)[0]
        idx2 = np.where(colours <= col_max)[0]
        idx3 = np.where(mags >= mag_min)[0]
        idx4 = np.where(mags <= mag_max)[0]
        idx = set(idx1).intersection(set(idx2))
        idx = idx.intersection(set(idx3))
        idx = list(idx.intersection(set(idx4)))

        return idx

    RC = photometry_classes.Star()

    log.info('Localizing the Red Clump')

    log.info('Selected Red Clump giants between:')
    log.info('i = '+str(config['RC_i_range'][0])+' to '+str(config['RC_i_range'][1]))
    log.info('r = '+str(config['RC_r_range'][0])+' to '+str(config['RC_r_range'][1]))
    log.info('g = '+str(config['RC_g_range'][0])+' to '+str(config['RC_g_range'][1]))
    log.info('(r-i) = '+str(config['RC_ri_range'][0])+' to '+str(config['RC_ri_range'][1]))
    log.info('(g-r) = '+str(config['RC_gr_range'][0])+' to '+str(config['RC_gr_range'][1]))
    log.info('(g-i) = '+str(config['RC_gi_range'][0])+' to '+str(config['RC_gi_range'][1]))

    idx = select_within_range(selected_phot['i'], selected_phot['ri'],
                              config['RC_i_range'][0], config['RC_i_range'][1],
                              config['RC_ri_range'][0], config['RC_ri_range'][1])

    (RC.ri, RC.sig_ri, RC.i, RC.sig_i) = event_colour_analysis.calc_distribution_centroid_and_spread_2d(selected_phot['ri'][idx],
                                                                                        selected_phot['i'][idx], use_iqr=True)

    idx = select_within_range(selected_phot['r'], selected_phot['ri'],
                              config['RC_r_range'][0], config['RC_r_range'][1],
                              config['RC_ri_range'][0], config['RC_ri_range'][1])

    (RC.r, RC.sig_r) = event_colour_analysis.calc_distribution_centre_and_spread(selected_phot['r'][idx], use_iqr=True)

    idx = select_within_range(selected_phot['g'], selected_phot['gr'],
                              config['RC_g_range'][0], config['RC_g_range'][1],
                              config['RC_gr_range'][0], config['RC_gr_range'][1])

    (RC.gr, RC.sig_gr, RC.g, RC.sig_g) = event_colour_analysis.calc_distribution_centroid_and_spread_2d(selected_phot['gr'][idx],
                                                                                        selected_phot['g'][idx], use_iqr=True)

    idx = select_within_range(selected_phot['g'], selected_phot['gi'],
                              config['RC_g_range'][0], config['RC_g_range'][1],
                              config['RC_gi_range'][0], config['RC_gi_range'][1])

    (RC.gi, RC.sig_gi, RC.g, RC.sig_g) = event_colour_analysis.calc_distribution_centroid_and_spread_2d(selected_phot['gi'][idx],
                                                                                        selected_phot['g'][idx], use_iqr=True)

    log.info('\n')
    log.info('Centroid of Red Clump Stars at:')
    log.info(RC.summary(show_mags=True))
    log.info(RC.summary(show_mags=False,show_colours=True))

    RC.transform_to_JohnsonCousins()

    log.info(RC.summary(show_mags=False,johnsons=True))

    return RC

def output_photometry(config, stars, photometry, selected_stars, log):

    if str(config['photometry_data_file']).lower() != 'none':

        log.info('Outputting multiband photometry to file')

        f = open(path.join(config['output_dir'],config['photometry_data_file']), 'w')
        f.write('# All measured floating point quantities in units of magnitude\n')
        f.write('# Selected indicates whether a star lies within the selection radius of a given location, if any.  1=true, 0=false\n')
        f.write('# Star   x_pix    y_pix   ra_deg   dec_deg   g  sigma_g    r  sigma_r    i  sigma_i   (g-i)  sigma(g-i) (g-r)  sigma(g-r)  (r-i) sigma(r-i)  Selected  Gaia_ID\n')

        for j in range(0,len(photometry['i']),1):
            sid = stars['star_id'][j]
            if j in selected_stars:
                selected = 1
            else:
                selected = 0
            f.write( str(sid)+' '+\
                        str(photometry['x'][j])+' '+str(photometry['y'][j])+' '+\
                        str(photometry['ra'][j])+' '+str(photometry['dec'][j])+' '+\
                        str(photometry['g'][j])+' '+str(photometry['gerr'][j])+' '+\
                        str(photometry['r'][j])+' '+str(photometry['rerr'][j])+' '+\
                        str(photometry['i'][j])+' '+str(photometry['ierr'][j])+' '+\
                        str(photometry['gi'][j])+' '+str(photometry['gi_err'][j])+' '+\
                        str(photometry['gr'][j])+' '+str(photometry['gr_err'][j])+' '+\
                        str(photometry['ri'][j])+' '+str(photometry['ri_err'][j])+' '+\
                        str(selected)+' '+str(photometry['gaia_source_id'])+'\n' )

        f.close()

        log.info('Completed output of multiband photometry')

if __name__ == '__main__':

    run_field_colour_analysis()
