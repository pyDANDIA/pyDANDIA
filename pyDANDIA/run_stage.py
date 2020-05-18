# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 17:04:53 2017

@author: rstreet
"""
import numpy as np
np.random.seed(1234567) #define seed for reproducible results.
import matplotlib
matplotlib.use('Agg')
from os import getcwd, path, remove
from sys import argv, exit
from sys import path as systempath
#cwd = getcwd()
#systempath.append(path.join(cwd,'../'))
from pyDANDIA import pipeline_setup
from pyDANDIA import stage0
from pyDANDIA import stage1
from pyDANDIA import stage2
from pyDANDIA import stage3
from pyDANDIA import calibrate_photometry
from pyDANDIA import reference_astrometry
from pyDANDIA import stage3_db_ingest
from pyDANDIA import stage4
from pyDANDIA import stage5
#from pyDANDIA.db import astropy_interface
from pyDANDIA import stage6
from pyDANDIA import starfind
from pyDANDIA import logs
from pyDANDIA import image_coadd

def run_stage_stand_alone():
    """Function to run a stage or section of pyDANDIA in stand alone mode."""

    params = get_args()

    setup = pipeline_setup.pipeline_setup(params)

    if params['stage'] == 'stage0':

        (status, report, reduction_metadata) = stage0.run_stage0(setup)

    elif params['stage'] == 'stage1':

        (status, report) = stage1.run_stage1(setup)

    elif params['stage'] == 'starfind':
        reduction_metadata = metadata.MetaData()
        try:
            reduction_metadata.load_all_metadata(metadata_directory=setup.red_dir,
                                                 metadata_name='pyDANDIA_metadata.fits')

        except:
            status = 'ERROR'
            report = 'Could not load the metadata file.'
            return status, report

        (status, report) = starfind.run_starfind(setup, reduction_metadata)

    elif params['stage'] == 'stage2':

        (status, report) = stage2.run_stage2(setup, empirical_ranking=params['empirical_ref'], n_stack = int(params['stack_ref']))

    elif params['stage'] == 'reference_astrometry':

        (status, report) = reference_astrometry.run_reference_astrometry(setup,
                                                    force_rotate_ref=params['rotate_ref'],
                                                    dx=params['dx'],
                                                    dy=params['dy'],
                                                    trust_wcs=params['trust_wcs'])

    elif params['stage'] == 'stage3':

        (status, report) = stage3.run_stage3(setup, cl_params=params)

    elif params['stage'] == 'calibrate_photometry':

        (status, report) = calibrate_photometry.calibrate_photometry_catalog(setup)

    elif params['stage'] == 'stage3_db_ingest':

        (status, report) = stage3_db_ingest.run_stage3_db_ingest(setup,
                                            primary_ref=params['primary_ref'],
                                            add_matched_stars=params['add_matched_stars'])

    elif params['stage'] == 'stage4':

        (status, report) = stage4.run_stage4(setup)


    elif params['stage'] == 'stage5':

        (status, report) = stage5.run_stage5(setup)

    elif params['stage'] == 'stage6':

        (status, report) = stage6.run_stage6(setup)

    elif params['stage'] == 'image_coadd':

        (status, report) = image_coadd.run_coadd(setup)

    else:

        print('ERROR: Unsupported stage name given')
        exit()

    print('Completed '+params['stage']+' with status:')
    print(repr(status)+': '+str(report))


def get_args():
    """Function to acquire the commandline arguments needed to run a stage
    of pyDANDIA in stand alone mode."""

    helptext = """              RUN STAGE STAND ALONE

            Call sequence is:
            > python run_stage.py [stage] [path to reduction directory] [path to phot_db] [-v]

            where stage is the name of the stage or code to be run, one of:
                stage0, stage1, stage2, stage3, stage4, stage5, stage6,
                starfind, stage3_db_ingest, reference_astrometry

            and the path to the reduction directory is given to the dataset
            to be processed

            and field indicates the name of the field being process
            (e.g. ROME-FIELD-0001, etc)

            The optional -v flag controls the verbosity of the pipeline
            logging output.  Values
            N can be:
            -v 0 [Default] Essential logging output only, written to log file.
            -v 1           Detailed logging output, written to log file.
            -v 2           Detailed logging output, written to screen and to log file.
            """

    params = {}

    if len(argv) == 1 or '-help' in argv:

        print(helptext)
        exit()

    if len(argv) < 4:

        params['stage'] = input('Please enter the name of the stage or code you wish to run: ')
        params['red_dir'] = input('Please enter the path to the reduction directory: ')
        params['db_file_path'] = input('Please enter the path to the photometric database: ')

    else:

        params['stage'] = argv[1]
        params['red_dir'] = argv[2]
        params['db_file_path'] = argv[3]


    if '-primary-ref' in argv or '-primary_ref' in argv:
        params['primary_ref'] = True
    else:
        params['primary_ref'] = False

    if '-empirical-ref' in argv or '-empirical_ref' in argv:
        params['empirical_ref'] = True
    else:
        params['empirical_ref'] = False

    if '-stack_ref' in argv or '-stack-ref' in argv:
        params['stack_ref'] = 10
    else:
        params['stack_ref'] = 1

    if '-v' in argv:
        idx = argv.index('-v')
        if len(argv) >= idx + 1:
            params['verbosity'] = int(argv[idx+1])

    if '-force-ref-rotation' in argv:
        params['rotate_ref'] = True
    else:
        params['rotate_ref'] = False

    if '-add-matched-stars' in argv:
        params['add_matched_stars'] = True
    else:
        params['add_matched_stars'] = False

    if '-trust-wcs' in argv:
        params['trust_wcs'] = True
    else:
        params['trust_wcs'] = False

    params['dx'] = 0.0
    params['dy'] = 0.0
    params['n_sky_bins'] = -1
    for a in argv:
        if '-dx' in a:
            params['dx'] = float(str(a).split('=')[-1])
        if '-dy' in a:
            params['dy'] = float(str(a).split('=')[-1])
        if '-n_sky_bins' in a:
            params['n_sky_bins'] = float(str(a).split('=')[-1])

    if str(params['db_file_path']).split('.')[-1] != 'db':
        raise ValueError(params['db_file_path']+' does not end in .db.  Is this a database file path?')

    return params


if __name__ == '__main__':

    run_stage_stand_alone()
