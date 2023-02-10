from os import path, rename
from sys import argv
import numpy as np
from pyDANDIA import crossmatch
from pyDANDIA import hd5_utils
from pyDANDIA import logs
from pyDANDIA import pipeline_setup
from pyDANDIA import plot_rms
from pyDANDIA import calibrate_photometry
from pyDANDIA import normalize_photometry_quad1
from pyDANDIA import photometry
from pyDANDIA import field_photometry
from astropy.table import Table, Column
import matplotlib.pyplot as plt

def run_phot_normalization_quads(setup, **params):
    """Function to normalize the photometry between different datasets taken
    of the same field in the same filter but with different instruments.

    Since different reference images are used for different datasets, there
    remain small offsets in the calibrated magnitudes of the lightcurves.
    This function compares the mean magnitudes of constant stars in the field
    to calculate these small magnitude offsets and correct for them.

    The normalization parameters are determined from quadrant 1, by
    normalize_photometry_quad1.py.  This code applies that normalization to the
    """
    log = logs.start_stage_log( setup.red_dir, 'postproc_phot_norm_quads' )

    xmatch = crossmatch.CrossMatchTable()
    xmatch.load(params['crossmatch_file'],log=log)

    # Identify the datasets to be used as the primary reference in each
    # filter:
    xmatch.id_primary_datasets_per_filter()
    log.info('Identified datasets to be used as the primary references in each filter: '\
                +repr(xmatch.reference_datasets))

    # Extract list of filters from xmatch.images['filter'] column
    filter_list = np.unique(xmatch.images['filter'].data)
    log.info('Identified list of filters to process: '+repr(filter_list))

    # Now the same photometric normalization derived for quadrant 1 can be
    # applied to the timeseries photometry for the other 3 quadrants
    for qid in range(1,5,1):
        log.info('\nApplying photometric normalization to quadrant '+str(qid)+'\n')

        # Read in the timeseries photometry:
        log.info('Loading the timeseries photometry from quadrant 1')
        file_path = path.join(setup.red_dir, params['field_name']
                                +'_quad'+str(qid)+'_photometry.hdf5')
        phot_data = hd5_utils.read_phot_from_hd5_file(file_path,
                                                        return_type='array')
        log.info('-> Completed photometry load')

        for filter in filter_list:
            ref_datacode = xmatch.reference_datasets[filter]
            sitecode = get_site_code(ref_datacode)
            log.info('Reference dataset in '+filter+' is '+ref_datacode+', sitecode='+sitecode)

            dataset_index = np.where(xmatch.datasets['dataset_filter'] == filter)[0]

            for idset in dataset_index:
                dset_datacode = xmatch.datasets['dataset_code'][idset]
                dset_sitecode = get_site_code(dset_datacode)

                # Normalize any dataset that isn't the same as the reference dataset
                if dset_datacode != ref_datacode:
                    log.info('Normalizing dataset '+dset_datacode+', sitecode='+dset_sitecode)
                    image_index = np.where(xmatch.images['dataset_code'] == dset_datacode)[0]
                    (fit, covar_fit) = fetch_dataset_phot_normalization(idset, xmatch, log)

                    # Apply the photometric normalization
                    (mag_col, mag_err_col) = field_photometry.get_field_photometry_columns('corrected')
                    (norm_mag_col, norm_mag_err_col) = field_photometry.get_field_photometry_columns('normalized')
                    phot_data = normalize_timeseries_photometry(phot_data, image_index,
                                                                fit, covar_fit,
                                                                mag_col, mag_err_col,
                                                                norm_mag_col, norm_mag_err_col,
                                                                log)

                    # Output the resulting photometry
                    output_quadrant_photometry(params, setup, qid, xmatch,
                                                phot_data, log)

    logs.close_log(log)

    status = 'OK'
    report = 'Completed successfully'
    return status, report


if __name__ == '__main__':
    (setup, params) = normalize_photometry_quad1.get_args()
    (status, report) = run_phot_normalization_quads(setup, **params)
