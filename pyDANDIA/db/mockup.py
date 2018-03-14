"""
Produce some mockup data to fill the database with test data.
"""

import random

from . import common

def create_stars_if_necessary(conn, n):
    """stuffs n random stars into the stars table if there's less rows in 
    there.

    (warning: minimal star id is 1!)

    Also, don't use this, read an actual catalog instead.
    """
    if list(conn.execute("SELECT COUNT(*) FROM stars"))[0][0]<n:
        for i in range(n):
            common.feed_to_table(conn, "stars", [
                    "ra", 
                    "dec",
                ], [
                    random.normalvariate(266.4, 15), 
                    random.normalvariate(-29, 15),
            ])


def get_test_data():
    """returns fake data for test and demo purposes.

    It's an exposure row and a sequence of photometry records.
    """
    return {
            'jd': 2540000+random.random()*500,
            'telescope_id': ['fred', 'joe'][random.randint(0, 1)],
            'instrument_id': ['cam1','cam2'][random.randint(0, 1)],
            'filter_id': ['filt1','filt2'][random.randint(0, 1)],
            'airmass': random.random()+0.1,
            'name': '/foo/bar/baz%0d.fz'%random.randint(0, 100000000),
            'fwhm' : random.random()+0.1,
            'fwhm_err' : random.random()+0.1,
            'ellipticity' : random.random()+0.1,
            'ellipticity_err' : random.random()+0.1,
            'exposure_time' : random.randint(0, 1),
            'moon_phase' : random.random()+0.1,
            'moon_separation' : random.random()*60+0.1
        }, [{
                'star_id': i+1,
                'diff_flux': random.random()*10,
                'diff_flux_err': random.random()*0.1,
                'phot_scale_factor': 0.14+random.random()*0.1,
                'phot_scale_factor_error': 0.014+random.random()*0.01,
                'magnitude': random.uniform(14,19),
                'magnitude_err': 0.014+random.random()*0.01,
                'local_background': random.random(),
                'local_background_error': random.random()/100,
                'delta_x': 0.1*random.random(),
                'delta_y': 0.1*random.random(),
                'residual_x': 0.01*random.random(),
                'residual_y': 0.01*random.random(),
                } for i in range(random.randint(5, 15))]
