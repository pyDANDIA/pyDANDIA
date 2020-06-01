from sys import argv
from os import getcwd, path, remove, environ
import numpy as np
from pyDANDIA import  phot_db
from pyDANDIA import  hd5_utils
from pyDANDIA import  pipeline_setup
from pyDANDIA import  metadata
from pyDANDIA import  logs
from pyDANDIA import pipeline_control
from pyDANDIA import stage3_db_ingest
from pyDANDIA import match_utils
from scipy import optimize
import matplotlib.pyplot as plt
from pyDANDIA import  hd5_utils
from astropy import table

VERSION = 'cross_calibrate_field_phot_v0.1.0'

def run_cross_calibration(setup):
    """Function to perform a cross-calibration of the photometric calibration
    between multiple datasets of the same pointing and filter, using the
    primary reference dataset in each filter as the 'gold standard' of
    reference."""

    log = logs.start_stage_log( setup.log_dir, 'cross_calibration', version=VERSION )

    conn = phot_db.get_connection(dsn=setup.phot_db_path)

    datasets = pipeline_control.get_datasets_for_reduction(setup,log)

    primary_ref_facility = phot_db.find_primary_reference_facility(conn,log=log)
    filters = phot_db.fetch_filters(conn)
    facilities = phot_db.fetch_facilities(conn)

    primary_ref_phot = load_primary_reference_photometry(conn,log)

    for red_dir,dataset_status in datasets.items():
        reduction_metadata = metadata.MetaData()
        reduction_metadata.load_all_metadata(path.join(setup.base_dir,red_dir),
                                                'pyDANDIA_metadata.fits')
        matched_stars = reduction_metadata.load_matched_stars()

        dataset_params = stage3_db_ingest.harvest_stage3_parameters(setup,reduction_metadata)
        dataset_params['filter_key'] = str(dataset_params['filter_name']).replace('p','')

        facility_code = phot_db.get_facility_code({'site': dataset_params['site'],
                                                    'enclosure': dataset_params['enclosure'],
                                                    'telescope': dataset_params['telescope'],
                                                    'instrument': dataset_params['instrument']})

        if facility_code in primary_ref_facility['facility_code']:
            log.info('No cross-calibration required for dataset '+facility_code+\
                        ', since this is the primary reference for filter '+\
                        dataset_params['filter_name'])

        else:
            filter_id = phot_db.get_filter_id(filters, dataset_params['filter_name'])
            facility_id = phot_db.get_facility_id(facilities, facility_code)
            dataset_setup = pipeline_setup.pipeline_setup({'red_dir': path.join(setup.base_dir,red_dir)})

            log.info('-> Extracting photometry for facility '+facility_code+'='+str(facility_id)+\
                            ', filter '+dataset_params['filter_name']+'='+str(filter_id))

            dataset_phot = phot_db.load_reference_image_photometry(conn,facility_id,filter_id)

            log.info(' -> Retrieved photometry for '+str(len(dataset_phot))+' stars')

            matched_stars = match_phot_tables(primary_ref_phot[dataset_params['filter_key']],
                                                dataset_phot,log)

            matched_phot = extract_matched_stars_phot(matched_stars,
                                primary_ref_phot[dataset_params['filter_key']],
                                dataset_phot,log)

            phot_model = calc_cross_calibration(matched_phot,
                                                facility_code,
                                                path.join(setup.base_dir,red_dir),
                                                log, diagnostics=True)

            reduction_metadata.create_phot_calibration_layer(phot_model,'cross_phot_calib')
            reduction_metadata.save_updated_metadata(
                path.join(setup.base_dir,red_dir), 'pyDANDIA_metadata.fits',
                log=log)

            log.info('Reading timeseries photometry for '+facility_code)

            dataset_photometry = hd5_utils.load_dataset_timeseries_photometry(dataset_setup,log,25)

            dataset_photometry = apply_photometric_transform(dataset_photometry,phot_model,log)

            hd5_utils.write_phot_hd5(dataset_setup, dataset_photometry,log=log)


    status = 'OK'
    report = 'Completed photometric cross-calibration'

    log.info('Field photometric cross-calibration: '+report)
    logs.close_log(log)

    return status, report

def get_args():

    params = {}

    if len(argv) != 3:
        params['base_dir'] = input('Please enter the path to the base directory: ')
        params['phot_db_path'] = input('Please enter the path to the database file: ')
    else:
        params['base_dir'] = argv[1]
        params['phot_db_path'] = argv[2]

    params['red_mode'] = 'field_cross_calib'
    params['log_dir'] = path.join(params['base_dir'],'logs')
    params['pipeline_config_dir'] = path.join(params['base_dir'],'config')
    params['verbosity'] = 0

    setup = pipeline_setup.pipeline_setup(params)
    setup.phot_db_path = params['phot_db_path']

    return setup

def load_primary_reference_photometry(conn,log):
    """Function to extract from the photometric database the reference image
    photometry for the primary reference dataset for the indicated filter"""

    log.info('Loading photometry from the field primary reference datasets for all filters')

    primary_ref_facility = phot_db.find_primary_reference_facility(conn,log=log)
    primary_ref_id = primary_ref_facility['facility_id'][0]

    log.info('ID of primary reference facility '+primary_ref_facility['facility_code'][0]+' = '+str(primary_ref_id))

    primary_ref_phot = {'g': None, 'r': None, 'i': None}

    filters = phot_db.fetch_filters(conn)
    facilities = phot_db.fetch_facilities(conn)

    for f in primary_ref_phot.keys():
        filter_id = phot_db.get_filter_id(filters, f+'p')
        log.info('Extracting photometry for filter '+f+', ID='+str(filter_id))

        results = phot_db.load_reference_image_photometry(conn,primary_ref_id,filter_id)

        primary_ref_phot[f] = results

        log.info(' -> Retrieved photometry for '+str(len(results))+' stars')

    return primary_ref_phot

def phot_func(p,mags):
    """Photometric transform function"""

    return p[0] + p[1]*mags

def errfunc(p,x,y):
    """Function to calculate the residuals on the photometric transform"""

    return y - phot_func(p,x)

def calc_transform(pinit, x, y):
    """Function to calculate the photometric transformation between a set
    of catalogue magnitudes and the instrumental magnitudes for the same stars
    """

    (pfit,iexec) = optimize.leastsq(errfunc,pinit,args=(x,y))

    return pfit

def calc_cross_calibration(matched_phot,dataset_label,output_dir,log,
                            diagnostics=True):
    """Function to calculate a transformation function between two
    photometric datasets"""

    pinit = [0.0, 0.0]
    model_mags = np.zeros(len(matched_phot))
    residuals = np.zeros(len(matched_phot))
    idx = np.where(matched_phot['dataset_calibrated_mag'] > 0.0)[0]
    log.info('Calculating photometric cross-transform, starting with '+str(len(idx))+' stars')

    for it in range(0,4,1):
        model = calc_transform(pinit, matched_phot['dataset_calibrated_mag'][idx],
                                matched_phot['primary_ref_calibrated_mag'][idx])

        log.info('-> Iteration '+str(it)+' model coefficients '+\
                    str(model[0])+', '+str(model[1]))

        model_mags[idx] = phot_func(model, matched_phot['dataset_calibrated_mag'][idx])
        residuals[idx] = abs(matched_phot['primary_ref_calibrated_mag'][idx] - model_mags[idx])
        sigma_threshold = np.median(residuals) + 3.0*residuals.std()
        max_res = residuals.max()
        thresholds = np.percentile(residuals,[75.0,80.0,90.0])
        idx = residuals < thresholds[2]

        selected = np.where(idx == True)[0]
        log.info('-> Refined star selection after iteration '+str(it)+' using '+str(len(selected))+' stars')

    fig = plt.figure(1)

    plt.plot(matched_phot['dataset_calibrated_mag'],
            matched_phot['primary_ref_calibrated_mag'],
            marker='.',markersize=1, markerfacecolor='#969799',linestyle='',
            label='All stars')
    plt.plot(matched_phot['dataset_calibrated_mag'][idx],
            matched_phot['primary_ref_calibrated_mag'][idx],'k.',markersize=1,
            label='Selected stars')
    x = np.arange(matched_phot['dataset_calibrated_mag'][idx].min(),
                    matched_phot['dataset_calibrated_mag'][idx].max(),0.2)
    plt.plot(x, phot_func(model, x), 'r-')
    plt.xlabel('Dataset calibrated mag')
    plt.ylabel('Primary reference calibrated mag')
    plt.title(dataset_label)
    plt.grid()
    plt.legend()
    plot_file = path.join(output_dir,'phot_cross_calibration.png')
    plt.savefig(plot_file)
    plt.close(1)

    return model

def match_phot_tables(phot_table1, phot_table2, log):
    """Function to compile a matched index of stars which
    appear in both photometry tables"""

    matched_stars = match_utils.StarMatchIndex()

    for j,star_id in enumerate(phot_table1['star_id']):
        idx = np.where(phot_table2['star_id'] == star_id)[0]
        if len(idx) > 0:
            matched_stars.cat1_index.append(j)
            matched_stars.cat2_index.append(idx[0])
            matched_stars.n_match += 1

    log.info('Identified '+str(matched_stars.n_match)+' matching stars in both photometry tables')

    return matched_stars

def extract_matched_stars_phot(matched_stars, primary_ref_phot_table1,
                                dataset_phot_table2, log):
    """Function to extract arrays of photometry for stars with measurements
    in both datasets"""

    phot = []
    for j in range(0,matched_stars.n_match,1):
        if primary_ref_phot_table1['calibrated_mag'][matched_stars.cat1_index[j]] > 0.0 and \
            dataset_phot_table2['calibrated_mag'][matched_stars.cat2_index[j]] > 0.0:
            phot.append( [primary_ref_phot_table1['calibrated_mag'][matched_stars.cat1_index[j]],
                            dataset_phot_table2['calibrated_mag'][matched_stars.cat2_index[j]]] )
    phot = np.array(phot)

    matched_phot = table.Table( [ table.Column(data=phot[:,0], name='primary_ref_calibrated_mag'),
                                  table.Column(data=phot[:,1], name='dataset_calibrated_mag') ] )

    log.info('Extracted photometry for matching stars')

    return matched_phot

def apply_photometric_transform(dataset_photometry,model,log):
    """Function to apply the photometric model to the timeseries photometry of
    a dataset"""

    mask = dataset_photometry[:,:,13] > 0.0
    dataset_photometry[mask,23] = phot_func(model, dataset_photometry[mask,13])
    dataset_photometry[mask,24] = dataset_photometry[mask,14]

    log.info('Applied photometric transformation')

    return dataset_photometry

if __name__ == '__main__':

    setup = get_args()

    (status, report) = run_cross_calibration(setup)
