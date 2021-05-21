from pyDANDIA import lightcurves
from os import path

def test_setname():

    test_params = [{'red_dir': '/test/path/MOA-2021-BLG-123_cpt-doma-1m0-05-fa06_ip'},
                    {'red_dir': '/test/path/Gaia21azb_ip'},
                    {'red_dir': '/test/path/Gaia21azb_ep02_rp'}]
    test_setnames = ['cpt-doma-1m0-05-fa06_ip', 'Gaia21azb_ip', 'Gaia21azb_ep02_rp']

    for i, params in enumerate(test_params):
        setname = lightcurves.get_setname(params)
        print(setname, test_setnames[i])
        assert setname == test_setnames[i]
