from os import getcwd, path, remove
from sys import exit, argv
from sys import path as systempath
cwd = getcwd()
systempath.append(path.join(cwd,'../'))
from pyDANDIA import cross_calibrate_field_phot
from pyDANDIA import pipeline_setup
from pyDANDIA import phot_db
from pyDANDIA import logs
from pyDANDIA import match_utils
from astropy import table
import numpy as np

TEST_DIR = path.join(cwd,'data','proc',
                        'ROME-FIELD-0002_lsc-doma-1m0-05-fl15_ip')

def test_get_args():

    setup = cross_calibrate_field_phot.get_args()
    test_setup = pipeline_setup.PipelineSetup()

    assert type(setup) == type(test_setup)

def test_load_primary_reference_photometry():

    log = logs.start_stage_log( TEST_DIR, 'test_cross_calib' )

    if len(argv) > 0:
        phot_db_path = argv[1]
    else:
        phot_db_path = input('Please enter the path to the test photometry database: ')

    conn = phot_db.get_connection(dsn=phot_db_path)

    results = cross_calibrate_field_phot.load_primary_reference_photometry(conn,log)

    assert type(results) == type({})
    for f in ['g', 'r', 'i']:
        assert f in results.keys()
        assert type(results[f]) == type(table.Table())

    conn.close()

    logs.close_log(log)

def test_calc_cross_calibration():

    log = logs.start_stage_log( TEST_DIR, 'test_cross_calib' )

    test_plot_file = path.join(TEST_DIR,'phot_cross_calibration.png')
    if path.isfile(test_plot_file):
        remove(test_plot_file)

    a0 = 1.5
    a1 = 2.5
    data1 = np.arange(14.0, 21.0, 0.5)
    data2 = a0*data1 + a1
    matched_phot = table.Table([ table.Column(data=data1,name='primary_ref_calibrated_mag'),
                                 table.Column(data=data2,name='dataset_calibrated_mag') ])

    model = cross_calibrate_field_phot.calc_cross_calibration(matched_phot,
                                                'Test dataset',TEST_DIR,log,
                                                diagnostics=True)

    assert path.isfile(test_plot_file)
    assert type(model) == type(np.zeros(1))

    logs.close_log(log)

def test_match_phot_tables():

    log = logs.start_stage_log( TEST_DIR, 'test_cross_calib' )

    t1_min = 1
    t1_max = 100
    t2_min = 20
    t2_max = t1_max

    table1 = table.Table([ table.Column(data=np.arange(t1_min, t1_max, 1),name='star_id')])
    table2 = table.Table([ table.Column(data=np.arange(t2_min, t2_max, 1),name='star_id')])

    matched_stars = cross_calibrate_field_phot.match_phot_tables(table1, table2, log)

    assert type(matched_stars) == type(match_utils.StarMatchIndex())
    assert len(matched_stars.cat1_index) == len(matched_stars.cat2_index)

    logs.close_log(log)

if __name__ == '__main__':

    #test_get_args()
    #test_load_primary_reference_photometry()
    #test_calc_cross_calibration()
    test_match_phot_tables()
