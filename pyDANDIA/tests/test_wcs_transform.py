"""
@author: rstreet
"""

from os import getcwd, path, remove
from sys import path as systempath
cwd = getcwd()
systempath.append(path.join(cwd,'../'))
import copy
import logs
import wcs
import stage3
import pipeline_setup
import metadata
import match_utils
import photometry
from astropy.io import fits
from astropy.table import Table, Column
from astropy.wcs import WCS as aWCS
from astropy.coordinates import SkyCoord
from astropy import units
import calc_coord_offsets
import reference_astrometry
import numpy as np
from skimage.transform import AffineTransform

## Test configuration
cwd = getcwd()
TEST_DATA = path.join(cwd,'data/proc/ROME-FIELD-0002_lsc-doma-1m0-05-fl15_ip')

def establish_test_env():
    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DATA})
    meta = metadata.MetaData()
    meta.load_a_layer_from_file( setup.red_dir,
                                'pyDANDIA_metadata.fits',
                                'data_architecture' )
    meta.load_a_layer_from_file( setup.red_dir,
                                'pyDANDIA_metadata.fits',
                                'images_stats' )
    meta.load_a_layer_from_file( setup.red_dir,
                                'pyDANDIA_metadata.fits',
                                'reduction_parameters' )
    meta.load_a_layer_from_file( setup.red_dir,
                                'pyDANDIA_metadata.fits',
                                'headers_summary' )

    return setup, meta

def simulate_detected_sources(nstars=100):
    xmin = 1.0
    xmax = 4000.0
    ymin = 1.0
    ymax = 4000.0
    ramin = 223.180
    decmin = -63.2
    pixscale = 0.389
    median_flux = 10000.0

    detected_objects = np.zeros((nstars,9))
    detected_objects[:,0] = np.arange(0,nstars,1)
    detected_objects[:,1] = np.linspace(xmin,xmax,nstars)
    detected_objects[:,2] = abs(np.random.standard_normal(nstars))*ymax/2.0 + ymin
    detected_objects[:,3] = ramin + detected_objects[:,1]*pixscale/3600.0
    detected_objects[:,4] = decmin + detected_objects[:,2]*pixscale/3600.0
    detected_objects[:,5] = abs(np.random.standard_normal(nstars))*median_flux
    detected_objects[:,6] = 0.001 + 0.1*detected_objects[:,5]
    for j in range(0,nstars,1):
        (detected_objects[j,7], detected_objects[j,8],flux,flux_err) = photometry.convert_flux_to_mag(detected_objects[j,5],
                                                                                        detected_objects[j,6])

    detected_data = [ Column(name='index', data=detected_objects[:,0]),
                      Column(name='x', data=detected_objects[:,1]),
                      Column(name='y', data=detected_objects[:,2]),
                      Column(name='ra', data=detected_objects[:,3]),
                      Column(name='dec', data=detected_objects[:,4]),
                      Column(name='ref_flux', data=detected_objects[:,5]),
                      Column(name='ref_flux_err', data=detected_objects[:,6]),
                      Column(name='ref_mag', data=detected_objects[:,7]),
                      Column(name='ref_mag_err', data=detected_objects[:,8]) ]

    return Table(data=detected_data)

def test_offset_transform():

    (setup, meta) = establish_test_env()
    log = logs.start_stage_log( cwd, 'test_wcs' )

    # Configure pixel transform:
    dx = 130.0
    dy = 20.0
    transform_pix = AffineTransform(translation=(dx, dy))

    # Test catalog of sources:
    detected_sources = simulate_detected_sources()
    updated_sources = copy.deepcopy(detected_sources)

    transform_deg = calc_coord_offsets.convert_pixel_to_world_transform(meta,transform_pix,detected_sources,log)

    updated_sources = calc_coord_offsets.transform_coordinates(setup, updated_sources,
                                                            transform_deg, coords='offsetradec',
                                                            verbose=True)
    for j in range(0,len(detected_sources),1):
        assert(detected_sources['ra'][j]+transform_deg.params[0,2] == updated_sources['ra'][j])
        assert(detected_sources['dec'][j]+transform_deg.params[1,2] == updated_sources['dec'][j])

    logs.close_log(log)


if __name__ == '__main__':
    test_offset_transform()
