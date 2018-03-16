"""
Produce some mockup data to fill the database with test data.
"""

import random
import os
from os import getcwd, path
from sys import path as systempath
cwd = getcwd()
systempath.append(path.join(cwd, '../'))

from phot_db import *

def fill_with_fake_entries():
    conn = get_connection(database_file_path)
    ensure_tables(conn)
    create_stars_if_necessary(conn, n=5)
    create_reference_images_if_necessary(conn, n=2)
    create_exposures_if_necessary(conn, n=10)
    create_phot_if_necessary(conn, n=10)
    conn.commit()
    conn.close()

def create_stars_if_necessary(conn, n):
    """stuffs n random stars into the stars table if there are fewer than n rows in 
    there.
    
    (warning: minimum star id is 1!)
    
    !!!WARNING!!! Only use this for testing purposes. 
    In normal operations you want to populate the star list from the star catalog 
    returned from starfinder.
    """
    if list(conn.execute("SELECT COUNT(*) FROM stars"))[0][0]<n:
        for i in range(n):
            feed_to_table(conn, "stars", [
                    "ra", 
                    "dec",
                ], [
                    random.normalvariate(266.4, 15), 
                    random.normalvariate(-29, 15),
            ])

def create_reference_images_if_necessary(conn, n):
    """stuffs n random reference images into the reference_images table if there are fewer than n rows in 
    there.
    
    (warning: minimum exposure id is 1!)
    
    !!!WARNING!!! Only use this for testing purposes. 
    In normal operations the pipeline will enter the reference_images.
    """
    if list(conn.execute("SELECT COUNT(*) FROM reference_images"))[0][0]<n:
        for i in range(n):
            feed_to_table(conn, "reference_images", [
                    "telescope_id",
                    "instrument_id",
                    "filter_id",
                    "refimg_ellipticity",
                    "refimg_ellipticity_err",
                    "refimg_fwhm",
                    "refimg_fwhm_err",
                    "slope",
                    "slope_err",
                    "intercept",
                    "intercept_err",
                    "refimg_name",
                ], [
                    random.choice(telescopes.keys()),
                    random.choice(instruments.keys()),
                    random.choice(filters.keys()),
                    random.uniform(0.0,0.2),
                    random.uniform(0.005,0.01),
                    random.uniform(0.8,1.1),
                    random.uniform(0.005,0.01),
                    random.uniform(0.1,1.0),
                    random.uniform(0.01,0.09),
                    random.uniform(1,10),
                    random.uniform(0.01,0.8),
                    "ref_image_"+str(i),
            ])

def create_exposures_if_necessary(conn, n):
    """stuffs n random exposures into the exposures table if there are fewer than n rows in 
    there.
    
    (warning: minimum exposure id is 1!)
    
    !!!WARNING!!! Only use this for testing purposes. 
    In normal operations the pipeline will enter the exposures.
    """
    if list(conn.execute("SELECT COUNT(*) FROM exposures"))[0][0]<n:
        jd_start = 2458100.5
        for i in range(n):
            feed_to_table(conn, "exposures", [
                    "reference_image",
                    "jd",
                    "exposure_ellipticity",
                    "exposure_ellipticity_err",
                    "exposure_fwhm",
                    "exposure_fwhm_err",
                    "airmass",
                    "exposure_time",
                    "moon_phase",
                    "moon_separation",
                    "delta_x",
                    "delta_y",
                    "exposure_name",
                ], [
                    random.randint(1,2),
                    jd_start+1.0+random.uniform(0.1,0.6),
                    random.uniform(0.0,0.5),
                    random.uniform(0.01,0.05),
                    random.uniform(0.8,5.0),
                    random.uniform(0.01,0.05),
                    random.uniform(1.0,4.0),
                    random.randint(100,300),
                    random.uniform(0.0,1.0),
                    random.uniform(1,60),
                    random.uniform(0.01,1.0),
                    random.uniform(0.01,1.0),
                    "my_image_"+str(i),
            ])

def create_phot_if_necessary(conn, n):
    """stuffs n random photometry points into the phot table if there are fewer than n rows in 
    there.
    
    (warning: minimum phot id is 1!)
    
    !!!WARNING!!! Only use this for testing purposes. 
    In normal operations the pipeline will enter these points.
    """
    if list(conn.execute("SELECT COUNT(*) FROM phot"))[0][0]<n:
        for i in range(n):
            feed_to_table(conn, "phot", [
                    "exposure_id", 
                    "star_id",
                    "reference_flux",
                    "reference_flux_err",
                    "reference_mag",
                    "reference_mag_err",
                    "diff_flux",
                    "diff_flux_err",
                    "magnitude",
                    "magnitude_err",
                    "phot_scale_factor",
                    "phot_scale_factor_err",
                    "local_background",
                    "local_background_err",
                    "residual_x",
                    "residual_y",
                ], [
                    random.randint(1,10),
                    random.randint(1,5),
                    random.uniform(5000,60000),
                    random.uniform(200,8000),
                    random.uniform(14,19),
                    random.uniform(0.01,0.05),
                    random.uniform(-100,100),
                    random.uniform(0.01,0.05),
                    random.uniform(14,19),
                    random.uniform(0.01,0.05),
                    random.uniform(0.1,1.0),
                    random.uniform(0.01,0.05),
                    random.uniform(500,2000),
                    random.uniform(100,500),
                    random.uniform(0.01,0.05),
                    random.uniform(0.01,0.05),
            ])


#def get_test_data():
#    """returns fake data for test and demo purposes.
#    
#    It's an exposure row and a sequence of photometry records.
#    """
#    return {
#            'jd': 2540000+random.random()*500,
#            'telescope_id': ['fred', 'joe'][random.randint(0, 1)],
#            'instrument_id': ['cam1','cam2'][random.randint(0, 1)],
#            'filter_id': ['filt1','filt2'][random.randint(0, 1)],
#            'airmass': random.random()+0.1,
#            'name': '/foo/bar/baz%0d.fz'%random.randint(0, 100000000),
#            'fwhm' : random.random()+0.1,
#            'fwhm_err' : random.random()+0.1,
#            'ellipticity' : random.random()+0.1,
#            'ellipticity_err' : random.random()+0.1,
#            'exposure_time' : random.randint(0, 1),
#            'moon_phase' : random.random()+0.1,
#            'moon_separation' : random.random()*60+0.1
#        }, [{
#                'star_id': i+1,
#                'diff_flux': random.random()*10,
#                'diff_flux_err': random.random()*0.1,
#                'phot_scale_factor': 0.14+random.random()*0.1,
#                'phot_scale_factor_error': 0.014+random.random()*0.01,
#                'magnitude': random.uniform(14,19),
#                'magnitude_err': 0.014+random.random()*0.01,
#                'local_background': random.random(),
#                'local_background_error': random.random()/100,
#                'delta_x': 0.1*random.random(),
#                'delta_y': 0.1*random.random(),
#                'residual_x': 0.01*random.random(),
#                'residual_y': 0.01*random.random(),
#                } for i in range(random.randint(5, 15))]
