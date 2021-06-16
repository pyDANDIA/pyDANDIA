from pyDANDIA import normalize_photometry
from pyDANDIA import logs
import test_field_photometry

def test_find_constant_stars():

    params = test_field_photometry.test_params()
    log = logs.start_stage_log( params['log_dir'], 'test_norm_phot' )

    phot_data = test_field_photometry.test_photometry_array()

    normalize_photometry.find_constant_stars(params, phot_data, 'ip', 1, log, diagnostics=True)

    logs.close_log(log)


if __name__ == '__main__':
    test_find_constant_stars()
