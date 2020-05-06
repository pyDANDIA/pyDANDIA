import numpy as np
import os
import sys
cwd = os.getcwd()
sys.path.append(os.path.join(cwd,'../'))
from pyDANDIA import photometry_classes
from pyDANDIA import analyse_cmd
from pyDANDIA import logs
from astropy import table

def test_load_target_timeseries_photometry():

    if len(sys.argv) < 4:
        g_filec = input('Please enter path to the g-band lightcurve file: ')
        r_file = input('Please enter path to the g-band lightcurve file: ')
        i_file = input('Please enter path to the g-band lightcurve file: ')
    else:
        g_file = sys.argv[1]
        r_file = sys.argv[2]
        i_file = sys.argv[3]

    cwd = os.getcwd()
    log = logs.start_stage_log( cwd, 'test_analyse_cmd' )

    config = { 'target_field_id': 10000,
                'target_ra': '18:00:00.0', 'target_dec': '-29:00:00.0',
                'target_lightcurve_files': { 'g': g_file, 'r': r_file, 'i': i_file } }
    n_stars = 10
    phot_table = table.Table( [table.Column(name='star_id', data=np.zeros(n_stars)),
                               table.Column(name='calibrated_mag', data=np.ones(n_stars)),
                               table.Column(name='calibrated_mag_err', data=np.ones(n_stars))] )
    phot_table['star_id'][0] = config['target_field_id']

    photometry = { 'phot_table_g': phot_table, 'phot_table_r': phot_table, 'phot_table_i': phot_table }

    target = analyse_cmd.load_target_timeseries_photometry(config,photometry,log)

    assert type(target) == type(photometry_classes.Star())
    assert target.star_index == config['target_field_id']
    for f in ['g', 'r', 'i']:
        assert type(target.lightcurves[f] == type(table.Table()))

    logs.close_log(log)

if __name__ == '__main__':

    test_load_target_timeseries_photometry()
