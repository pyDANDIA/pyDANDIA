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
import matplotlib.pyplot as plt
from pyDANDIA import  photometry_classes
from pyDANDIA import  phot_db
from pyDANDIA import  logs
from pyDANDIA import  event_colour_analysis
from pyDANDIA import  spectral_type_data
from pyDANDIA import  red_clump_utilities
from pyDANDIA import  config_utils
from pyDANDIA import  lightcurves

def run_field_colour_analysis():
    """Function to analyse the colour information for a given field pointing"""

    config = get_args()

    log = logs.start_stage_log( config['output_dir'], 'analyse_cmd' )

    conn = phot_db.get_connection(dsn=config['db_file_path'])

    event_model = config_utils.load_event_model(config['event_model_parameters_file'], log)

    (source, blend) = calc_source_blend_params(config,event_model,log)

    (photometry, stars) = extract_reference_instrument_calibrated_photometry(conn,log)

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

    plot_colour_colour_diagram(config, photometry, RC, log)

    source.output_json(path.join(config['output_dir'],'source_parameters.json'))
    blend.output_json(path.join(config['output_dir'],'blend_parameters.json'))
    RC.output_json(path.join(config['output_dir'],'red_clump_parameters.json'))

    conn.close()

    logs.close_log(log)

def get_args():

    if len(argv) == 1:

        config_file = input('Please enter the path to the JSON configuration file: ')

    else:

        config_file = argv[1]

    config = config_utils.build_config_from_json(config_file)

    # Handle those keywords which may have Boolean entries
    boolean_keys = ['interactive', 'add_rc_centroid', 'add_blend',
                    'add_source', 'add_source_trail', 'add_crosshairs']
    for key in boolean_keys:
        if 'true' in str(config[key]).lower():
            config[key] = True
        else:
            config[key] = False

    # Handle those keywords which may have None values:
    none_allowed_keys = ['target_field_id']
    for key in none_allowed_keys:
        if 'none' in str(config[key]).lower():
            config[key] = None

    # Handle dictionary keywords which may have None entries
    none_allowed_keys = ['target_lightcurve_files']
    for key in none_allowed_keys:
        orig_dict = config[key]
        new_dict = {}
        for key, value in orig_dict.items():
            if 'none' == str(value).lower():
                new_dict[key] = None
            else:
                new_dict[key] = value
        config[key] = new_dict

    return config


def calc_source_blend_params(config,event_model,log):
    """Function to construct a dictionary of needed parameters for the
    source and blend"""

    source = photometry_classes.Star()
    blend = photometry_classes.Star()

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

    return source, blend

def fetch_star_phot(star_id,phot_table):
        jdx = np.where(phot_table['star_id'] == star_id)[0]
        if len(jdx) == 0:
            return 0.0,0.0
        else:
            return phot_table['calibrated_mag'][jdx],phot_table['calibrated_mag_err'][jdx]

def extract_reference_instrument_calibrated_photometry(conn,log):
    """Function to extract from the phot_db the calibrated photometry for stars
    in the field from the data from the photometric reference instrument.
    By default this is defined to be lsc.doma.1m0a.fa15.
    """

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
                photometry['phot_table_'+f].append([data['star_id'][0], data['calibrated_mag'][0], data['calibrated_mag_err'][0]])
            else:
                photometry['phot_table_'+f].append([star['star_id'],0.0,0.0])

        if j%1000.0 == 0.0:
            print('--> Completed '+str(j)+' stars out of '+str(len(stars)))

    for f,fid in filters.items():
        table_data = np.array(photometry['phot_table_'+f])

        data = [table.Column(name='star_id',
                        data=table_data[:,0]),
                table.Column(name='calibrated_mag',
                        data=table_data[:,1]),
                table.Column(name='calibrated_mag_err',
                        data=table_data[:,2])]
        photometry['phot_table_'+f] = table.Table(data=data)

    log.info('Extracted photometry for '+str(len(stars))+' stars')

    photometry['g'] = np.zeros(len(stars))
    photometry['gerr'] = np.zeros(len(stars))
    photometry['r'] = np.zeros(len(stars))
    photometry['rerr'] = np.zeros(len(stars))
    photometry['i'] = np.zeros(len(stars))
    photometry['ierr'] = np.zeros(len(stars))

    for s in range(0,len(stars),1):

        sid = stars['star_id'][s]

        (photometry['g'][s],photometry['gerr'][s]) = fetch_star_phot(sid,photometry['phot_table_g'])
        (photometry['r'][s],photometry['rerr'][s]) = fetch_star_phot(sid,photometry['phot_table_r'])
        (photometry['i'][s],photometry['ierr'][s]) = fetch_star_phot(sid,photometry['phot_table_i'])

    return photometry, stars

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

def extract_local_star_photometry(photometry,selected_stars,log):
    """Function to extract the photometry for stars local to the given
    target coordinates"""

    g = table.Column(data=photometry['g'][selected_stars], name='g')
    r = table.Column(data=photometry['r'][selected_stars], name='r')
    i = table.Column(data=photometry['i'][selected_stars], name='i')
    gr = table.Column(data=photometry['gr'][selected_stars], name='gr')
    ri = table.Column(data=photometry['ri'][selected_stars], name='ri')
    gi = table.Column(data=photometry['gi'][selected_stars], name='gi')
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

    if config['target_field_id'] != None:

        target.star_index = config['target_field_id']
        t = SkyCoord(config['target_ra']+' '+config['target_dec'],
                        unit=(u.hourangle,u.degree), frame='icrs')
        target.ra = t.ra.value
        target.dec = t.dec.value
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

    col_key = blue_filter+red_filter

    fig = plt.figure(1,(10,10))

    ax = plt.subplot(111)

    plt.rcParams.update({'font.size': 18})

    cdx = np.where(photometry[col_key] != -99.999)[0]
    mdx = np.where(photometry[yaxis_filter] != 0.0)[0]
    jdx = list(set(cdx).intersection(set(mdx)))

    default_marker_colour = '#8c6931'
    field_marker_colour = '#E1AE13'
    marker_colour = default_marker_colour
    if len(selected_stars) < len(photometry['i']):
        marker_colour = field_marker_colour

    plt.scatter(photometry[col_key][jdx],photometry[yaxis_filter][jdx],
                 c=marker_colour, marker='.', s=1,
                 label='Stars within field of view')

    plt.scatter(selected_phot[col_key],selected_phot[yaxis_filter],
                  c=default_marker_colour, marker='*', s=4,
                  label='Stars < '+str(round(params['selection_radius'],1))+'arcmin of target')

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

    ax.set_xticks(xticks,minor=True)
    ax.set_yticks(yticks,minor=True)

    plot_file = path.join(params['output_dir'],'colour_magnitude_diagram_'+\
                                            yaxis_filter+'_vs_'+blue_filter+red_filter\
                                            +'.pdf')

    plt.grid()

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * -0.025,
             box.width, box.height * 0.95])

    l = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2)

    l.legendHandles[0]._sizes = [50]
    l.legendHandles[1]._sizes = [50]

    plt.rcParams.update({'legend.fontsize':18})
    plt.rcParams.update({'font.size':18})
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)

    if params['interactive']:
        plt.show()

    plt.savefig(plot_file,bbox_inches='tight')

    plt.close(1)

    log.info('Colour-magnitude diagram output to '+plot_file)

def plot_colour_colour_diagram(params,photometry,RC,log):
    """Function to plot a colour-colour diagram, if sufficient data are
    available within the given star catalog"""

    filters = { 'i': 'SDSS-i', 'r': 'SDSS-r', 'g': 'SDSS-g' }

    fig = plt.figure(1,(10,10))

    ax = plt.axes()

    grx = np.where(photometry['gr'] != -99.999)[0]
    rix = np.where(photometry['ri'] != -99.999)[0]
    jdx = list(set(grx).intersection(set(rix)))

    inst_gr = photometry['gr'][jdx] - RC.Egr
    inst_ri = photometry['ri'][jdx] - RC.Eri

    ax.scatter(inst_gr, inst_ri,
               c='#8c6931', marker='.', s=1,
             label='Stars within ROME field')

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
                            color='k', size=10, rotation=-30.0, alpha=1.0)

        if luminosity_class[i] == 'V' and plot_dwarfs:

            plt.plot(gr_colour[i], ri_colour[i], marker='s', color=c,
                     markeredgecolor='k', alpha=0.5)

            plt.annotate(spt, (gr_colour[i],
                           ri_colour[i]+0.1),
                             color='k', size=10,
                             rotation=-30.0, alpha=1.0)

    plt.xlabel('SDSS (g-r) [mag]')

    plt.ylabel('SDSS (r-i) [mag]')

    plot_file = path.join(params['output_dir'],'colour_colour_diagram.pdf')

    scale_axes = False
    if scale_axes:
        plt.axis([-1.0,2.0,-1.0,1.0])

    plt.grid()

    xticks = np.arange(-1.0,2.0,0.1)
    yticks = np.arange(-1.0,1.0,0.1)

    ax.set_xticks(xticks, minor=True)
    ax.set_yticks(yticks, minor=True)

    if scale_axes:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * -0.025,
                 box.width, box.height * 0.95])

        l = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2)

        try:
            l.legendHandles[2]._sizes = [50]
            l.legendHandles[3]._sizes = [50]
        except IndexError:
            pass

    plt.rcParams.update({'legend.fontsize':18})
    plt.rcParams.update({'font.size':18})
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)

    if params['interactive']:
        plt.show()

    plt.savefig(plot_file,bbox_inches='tight')

    plt.close(1)

    log.info('Colour-colour diagram output to '+plot_file)

def plot_crosshairs(fig,xvalue,yvalue,linecolour):

    ([xmin,xmax,ymin,ymax]) = plt.axis()

    xdata = np.linspace(xmin,xmax,10.0)
    ydata = np.zeros(len(xdata))
    ydata.fill(yvalue)

    plt.plot(xdata, ydata, linecolour+'-', alpha=0.5)

    ydata = np.linspace(ymin,ymax,10.0)
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

if __name__ == '__main__':

    run_field_colour_analysis()