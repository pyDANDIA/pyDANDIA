# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 17:14:17 2017

@author: rstreet
"""
from os import getcwd, path, remove
from sys import path as systempath
cwd = getcwd()
systempath.append(path.join(cwd,'../'))
import logs
import wcs
import stage3
import pipeline_setup
import match_utils
import calc_coord_offsets
from astropy.io import fits
from astropy.table import Table, Column
from astropy import wcs as aWCS
from astropy.coordinates import SkyCoord
from astropy import units
import catalog_utils
import numpy as np

test_full_frame = True
cwd = getcwd()
TEST_DATA = path.join(cwd,'data')

if test_full_frame:

    TEST_DIR = path.join(cwd,'data','proc',
                        'ROME-FIELD-0002_lsc-doma-1m0-05-fl15_ip')
else:

    TEST_DIR = path.join(cwd,'data','proc',
                        'ROME-FIELD-0002_lsc-doma-1m0-05-fl15_ip')

def test_reference_astrometry():

    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DIR})

    log = logs.start_stage_log( cwd, 'test_wcs' )

    image_path = path.join(TEST_DATA,'lsc1m005-fl15-20170701-0144-e91_cropped.fits')
    detected_sources_file = path.join(TEST_DATA,'lsc1m005-fl15-20170701-0144-e91_cropped_sources.txt')

    outputs = ['reference_detected_sources_pixels.png',
               'reference_detected_sources_world.png',
               'astrometry_separations.png',
               'star_catalog.fits']

    for item in outputs:
        if path.isfile(path.join(TEST_DATA,item)):
            remove(path.join(TEST_DATA,item))

    detected_sources = catalog_utils.read_source_catalog(detected_sources_file,
                                                         table_format=True)

    ref_source_catalog = wcs.reference_astrometry(setup,log,image_path,
                                                  detected_sources,
                                                  find_transform=True,
                                                  diagnostics=True)

    assert path.isfile(path.join(TEST_DATA,'ref','reference_detected_sources_pixels.png')) == True
    assert path.isfile(path.join(TEST_DATA,'ref','reference_detected_sources_world.png')) == True
    assert path.isfile(path.join(TEST_DATA,'ref','astrometry_separations.png')) == True
    assert path.isfile(path.join(TEST_DATA,'ref','star_catalog.fits')) == True
    assert path.isfile(path.join(TEST_DATA,'ref','ref_image_wcs.fits')) == True

    logs.close_log(log)


def test_search_vizier_for_objects_in_fov():
    """Function to test the online extraction of a catalogue of known
    sources within a given field of view"""

    image = path.join(TEST_DATA,'lsc1m005-fl15-20170701-0144-e91_cropped.fits')
    header = fits.getheader(image)
    radius = ( float(header['pixscale'])*float(header['NAXIS1']) ) / 60.0
    catalog = wcs.search_vizier_for_objects_in_fov(header['RA'], header['Dec'], radius)

    assert len(catalog) == 50

def test_search_vizier_for_2mass_sources():
    """Function to test the online extraction of a catalogue of known
    sources within a given field of view"""

    ra = '270.8210141'
    dec = '-26.909889'
    radius = 2027*0.467*60.0/2.0
    catalog = wcs.search_vizier_for_2mass_sources(ra, dec, radius)

    assert len(catalog) > 0

def test_search_vizier_for_gaia_sources():
    """Function to test the online extraction of a catalogue of known
    sources within a given field of view"""

    ra = '270.8210141'
    dec = '-26.909889'
    radius = 2027*0.467*60.0/2.0
    catalog = wcs.search_vizier_for_gaia_sources(ra, dec, radius)

    assert len(catalog) > 0

def test_fetch_catalog_sources_for_field():

    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DIR})

    log = logs.start_stage_log( cwd, 'test_wcs' )

    field = 'ROME-FIELD-01'
    header = {'CRPIX1': 199.8948336988849,
                'CRPIX2': 199.7880003717081,
                'CRVAL1': 269.5375,
                'CRVAL2': -28.01805555555556,
                'CUNIT1': 'deg     ',
                'CUNIT2': 'deg     ',
                'CD1_1': 0.0001081,
                'CD1_2': 0.0000000,
                'CD2_1': 0.0000000,
                'CD2_2': -0.0001081,
                'NAXIS1': 4096,
                'PIXSCALE': 0.26}

    image_wcs = aWCS(header)

    catalog_sources = wcs.fetch_catalog_sources_for_field(setup,field,header,
                                                          image_wcs,log,'2MASS')

    assert type(catalog_sources) == type(Table())
    assert len(catalog_sources) > 0

    catalog_sources = wcs.fetch_catalog_sources_for_field(setup,field,header,
                                                          image_wcs,log,'Gaia')

    assert type(catalog_sources) == type(Table())
    assert len(catalog_sources) > 0

    logs.close_log(log)

def test_match_stars_world_coords():

    log = logs.start_stage_log( cwd, 'test_wcs' )

    detected_coords = np.array(  [  [269.55310076, -28.04655246, 1866.259876089409, 1800.9376518684423],
                                    [269.52451024, -28.04663018, 1891.7929031808592, 1801.4127610075063],
                                    [269.53827323, -28.04670613, 1907.0905410137123, 1800.9376518684423],
                                    [269.53893954, -28.04667918, 1935.2019124607257, 1802.3499854421846],
                                    [269.5575608 , -28.04728878, 2148.2925744150702, 1800.8690300095063]  ] )

    detected_data = [ Column(name='ra', data=detected_coords[:,0]),
                      Column(name='dec', data=detected_coords[:,1]),
                      Column(name='x', data=detected_coords[:,2]),
                      Column(name='y', data=detected_coords[:,3]) ]

    detected_sources = Table(data=detected_data)

    catalog_coords = np.array( [ [269.52935399, -28.04729399, 1866.259876089409, 1800.9376518684423],
                                 [269.54868567, -28.04728673, 1891.7929031808592, 1801.4127610075063],
                                 [269.5575608 , -28.04728878, 1907.0905410137123, 1800.9376518684423],
                                 [269.51156351, -28.04750404, 1935.2019124607257, 1802.3499854421846] ] )

    catalog_data = [ Column(name='ra', data=catalog_coords[:,0]),
                     Column(name='dec', data=catalog_coords[:,1]),
                     Column(name='x', data=catalog_coords[:,2]),
                     Column(name='y', data=catalog_coords[:,3]) ]

    catalog_sources = Table(data=catalog_data)

    test_match = match_utils.StarMatchIndex()

    matched_stars = wcs.match_stars_world_coords(detected_sources,
                                                 catalog_sources,log,'Gaia',
                                                 verbose=True)

    assert matched_stars.n_match == 1
    assert 'StarMatchIndex' in str(type(matched_stars)).split("'")[1].split('.')[-1]
    assert detected_coords[matched_stars.cat1_index[0],0] == catalog_coords[matched_stars.cat2_index[0],0]
    assert detected_coords[matched_stars.cat1_index[0],1] == catalog_coords[matched_stars.cat2_index[0],1]
    assert matched_stars.cat1_ra[0] == matched_stars.cat2_ra[0]
    assert matched_stars.cat1_dec[0] == matched_stars.cat2_dec[0]

    logs.close_log(log)

def test_match_stars_pixel_coords():

    log = logs.start_stage_log( cwd, 'test_wcs' )

    detected_data = [ Column(name='x', data=np.linspace(10.0,500.0,20)),
                      Column(name='y', data=np.linspace(25.0,550.0,20)),
                      Column(name='ra', data=np.linspace(269.2,269.8,20)),
                      Column(name='dec', data=np.linspace(-28.4,-28.0,20)) ]

    detected_sources = Table(data=detected_data)

    catalog_data = [ Column(name='x', data=np.linspace(10.0,500.0,20)),
                     Column(name='y', data=np.linspace(25.0,550.0,20)),
                      Column(name='ra', data=np.linspace(269.2,269.8,20)),
                      Column(name='dec', data=np.linspace(-28.4,-28.0,20)) ]

    catalog_sources = Table(data=catalog_data)

    test_match = match_utils.StarMatchIndex()

    matched_stars = wcs.match_stars_pixel_coords(detected_sources,
                                                 catalog_sources,log,
                                                 verbose=True)

    assert matched_stars.n_match == len(catalog_sources)
    assert detected_sources['x'][matched_stars.cat1_index[0]] == catalog_sources['x'][matched_stars.cat2_index[0]]
    assert detected_sources['y'][matched_stars.cat1_index[0]] == catalog_sources['y'][matched_stars.cat2_index[0]]
    assert matched_stars.cat1_ra[0] == matched_stars.cat2_ra[0]
    assert matched_stars.cat1_dec[0] == matched_stars.cat2_dec[0]

    logs.close_log(log)

def test_image_wcs():

    image = path.join(TEST_DATA,'lsc1m005-fl15-20170701-0144-e91_cropped.fits')
    header = fits.getheader(image)

    image_wcs = aWCS(header)

    catalog_coords = np.array( [ [269.52935399, -28.04729399],
                                 [269.54868567, -28.04728673],
                                 [269.5575608 , -28.04728878],
                                 [269.51156351, -28.04750404] ] )

    positions = image_wcs.wcs_world2pix(catalog_coords,1)

    print(image_wcs)
    print(positions)

    crota2 = np.pi/2.0
    image_wcs.wcs.cd[0,0] = image_wcs.wcs.cd[0,0] * np.cos(crota2)
    image_wcs.wcs.cd[0,1] = -image_wcs.wcs.cd[0,0] * np.sin(crota2)
    image_wcs.wcs.cd[1,0] = image_wcs.wcs.cd[1,0] * np.sin(crota2)
    image_wcs.wcs.cd[1,1] = image_wcs.wcs.cd[1,1] * np.cos(crota2)

    positions2 =  image_wcs.wcs_world2pix(catalog_coords,1)

    print(image_wcs)
    print(positions2)

    image_wcs.wcs.crpix[0] += 0.1
    image_wcs.wcs.crpix[1] += 0.1

    positions3 = image_wcs.wcs_world2pix(catalog_coords,1)

    print(image_wcs)
    print(positions3)

def test_calc_world_coordinates():

    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DIR})

    log = logs.start_stage_log( cwd, 'test_wcs' )

    image_path = path.join(TEST_DATA,'lsc1m005-fa15-20190416-0242-e91.fits')

    detected_sources = np.array(  [  [0, 317.0, 2716.65, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1234.0]  ] )
    star = SkyCoord( '17:59:27.04 -28:36:37.0', frame='icrs', unit=(units.hourangle, units.deg))


    coords_table = wcs.calc_world_coordinates(setup,image_path,detected_sources,log)

    logs.close_log(log)

    print(coords_table)
    assert type(coords_table) == type(Table())

def test_calc_world_coordinates_astropy():

    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DIR})

    log = logs.start_stage_log( cwd, 'test_wcs' )

    image_path = path.join(TEST_DATA,'lsc1m005-fa15-20190416-0242-e91.fits')
    header = fits.getheader(image_path)

    image_wcs = aWCS(header)

    coord_data = np.array(  [  [0, 317.0, 2716.65, 0.0, 0.0, 1234.0, 2.0, 0.0, 0.0]  ] )

    detected_sources = [ Column(name='index', data=coord_data[:,0]),
                        Column(name='x', data=coord_data[:,1]),
                        Column(name='y', data=coord_data[:,2]),
                        Column(name='ra', data=coord_data[:,2]),
                        Column(name='dec', data=coord_data[:,2]),
                        Column(name='ref_flux', data=coord_data[:,2]),
                        Column(name='ref_flux_err', data=coord_data[:,2]),
                        Column(name='ref_mag', data=coord_data[:,2]),
                        Column(name='ref_mag_err', data=coord_data[:,2]) ]

    detected_sources = Table(data=detected_sources)

    star = SkyCoord( '17:59:27.04 -28:36:37.0', frame='icrs', unit=(units.hourangle, units.deg))

    coords_table = wcs.calc_world_coordinates_astropy(setup,image_wcs,detected_sources,log)

    logs.close_log(log)

    print(coords_table['ra'][0], star.ra.value)
    print(coords_table['dec'][0], star.dec.value)

    assert type(coords_table) == type(Table())
    assert round(star.ra.value,2) == round(coords_table['ra'][0],2)
    assert round(star.dec.value,2) == round(coords_table['dec'][0],2)

def test_calc_image_coordinates():

    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DIR})

    log = logs.start_stage_log( cwd, 'test_wcs' )

    image_path = path.join(TEST_DATA,'lsc1m005-fl15-20170701-0144-e91_cropped.fits')

    catalog_sources = np.array( [ [269.52935399, -28.04729399],
                                 [269.54868567, -28.04728673],
                                 [269.5575608 , -28.04728878],
                                 [269.51156351, -28.04750404] ] )

    coord_data = [ Column(name='ra', data=catalog_sources[:,0]),
                   Column(name='dec', data=catalog_sources[:,1]) ]
    catalog_sources = Table(data=coord_data)

    catalog_sources = wcs.calc_image_coordinates(setup, image_path, catalog_sources,log)

    logs.close_log(log)

    print(catalog_sources)

    assert 'x' in catalog_sources.colnames
    assert 'y' in catalog_sources.colnames

def test_calc_image_coordinates_astropy():

    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DIR})

    log = logs.start_stage_log( cwd, 'test_wcs' )

    image_path = path.join(TEST_DATA,'lsc1m005-fa15-20190416-0242-e91.fits')

    hdu = fits.open(image_path)

    header = hdu[0].header

    image_wcs = aWCS(header)

    star = SkyCoord( '17:59:27.04 -28:36:37.0', frame='icrs', unit=(units.hourangle, units.deg))
    star_coords = [317.0, 2716.65]

    catalog_sources = np.array( [ [star.ra.deg, star.dec.deg] ] )

    coord_data = [ Column(name='ra', data=catalog_sources[:,0]),
                   Column(name='dec', data=catalog_sources[:,1]) ]

    catalog_sources = Table(data=coord_data)

    catalog_sources = wcs.calc_image_coordinates_astropy(setup, image_wcs, catalog_sources,log)

    print(catalog_sources)

    assert 'x' in catalog_sources.colnames
    assert 'y' in catalog_sources.colnames

    assert star_coords[0] == catalog_sources['x'][0]
    assert star_coords[1] == catalog_sources['y'][1]

    logs.close_log(log)


def test_calc_image_coordinates_astropy2():

    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DIR})

    log = logs.start_stage_log( cwd, 'test_wcs' )

    image_path = path.join(TEST_DATA,'lsc1m005-fl15-20170701-0144-e91_cropped.fits')

    hdu = fits.open(image_path)

    header = hdu[0].header

    image_wcs = aWCS(header)

    star = SkyCoord( 269.5498866502567, -28.0300687302989, frame='icrs', unit=(units.deg, units.deg))
    star_coords = [301.0, 311.0]

    catalog_sources = np.array( [ [star.ra.deg, star.dec.deg] ] )

    coord_data = [ Column(name='ra', data=catalog_sources[:,0]),
                   Column(name='dec', data=catalog_sources[:,1]) ]

    catalog_sources = Table(data=coord_data)

    catalog_sources = wcs.calc_image_coordinates_astropy(setup, image_wcs, catalog_sources,log)

    print(catalog_sources)

    assert 'x' in catalog_sources.colnames
    assert 'y' in catalog_sources.colnames

    assert round(star_coords[0],0) == round(catalog_sources['x'][0],0)
    assert round(star_coords[1],0) == round(catalog_sources['y'][0],0)

    logs.close_log(log)

def test_calc_image_coordinates_astropy3():

    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DIR})

    log = logs.start_stage_log( cwd, 'test_wcs' )

    star = SkyCoord( 269.5498866502567, -28.0300687302989, frame='icrs', unit=(units.deg, units.deg))
    star_coords = [301.0, 311.0]

    catalog_sources = np.array( [ [star.ra.deg, star.dec.deg] ] )
    coord_data = [ Column(name='ra', data=catalog_sources[:,0]),
                   Column(name='dec', data=catalog_sources[:,1]) ]
    catalog_sources = Table(data=coord_data)

    image_path = path.join(TEST_DATA,'lsc1m005-fl15-20170701-0144-e91_cropped.fits')

    hdu = fits.open(image_path)

    header = hdu[0].header

    image_wcs = aWCS(header)

    updated_catalog1 = wcs.calc_image_coordinates_astropy(setup, image_wcs, catalog_sources,log)

    print(updated_catalog1)

    assert round(star_coords[0],0) == round(updated_catalog1['x'][0],0)
    assert round(star_coords[1],0) == round(updated_catalog1['y'][0],0)



    star = SkyCoord( '17:59:27.04 -28:36:37.0', frame='icrs', unit=(units.hourangle, units.deg))
    star_coords = [317.0, 2716.65]
    star_coords = [324.0, 2722.0]

    catalog_sources = np.array( [ [star.ra.deg, star.dec.deg] ] )
    coord_data = [ Column(name='ra', data=catalog_sources[:,0]),
                   Column(name='dec', data=catalog_sources[:,1]) ]
    catalog_sources = Table(data=coord_data)

    image_path2 = path.join(TEST_DATA,'lsc1m005-fa15-20190416-0242-e91.fits')

    hdu2 = fits.open(image_path2)

    header2 = hdu2[0].header

    image_wcs2 = aWCS(header2)

    updated_catalog2 = wcs.calc_image_coordinates_astropy(setup, image_wcs2, catalog_sources,log)

    print(updated_catalog2)

    assert round(star_coords[0],0) == round(updated_catalog2['x'][0],0)
    assert round(star_coords[1],0) == round(updated_catalog2['y'][0],0)

    logs.close_log(log)

def test_extract_bright_central_stars():

    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DIR})

    log = logs.start_stage_log( cwd, 'test_wcs' )

    field = 'ROME-FIELD-02'

    radius = 0.02

    image_path = path.join(TEST_DATA,'lsc1m005-fl15-20170701-0144-e91_cropped.fits')

    header = fits.getheader(image_path)

    image_wcs = aWCS(header)

    detected_sources_file = path.join(TEST_DATA,'lsc1m005-fl15-20170701-0144-e91_cropped_sources.txt')

    detected_sources = catalog_utils.read_source_catalog(detected_sources_file,
                                                         table_format=True)

    detected_sources = wcs.calc_world_coordinates_astropy(setup,image_wcs,
                                                          detected_sources,log)

    brightest_detected = detected_sources['ref_flux'].max()
    faintest_detected = detected_sources['ref_flux'].min()

    catalog_sources = wcs.fetch_catalog_sources_for_field(setup,field,header,
                                                          image_wcs,log,'Gaia')

    catalog_sources = wcs.calc_image_coordinates_astropy(setup, image_wcs,
                                                         catalog_sources,log)

    jdx = []
    for i,flux in enumerate(catalog_sources['phot_rp_mean_flux']):
        if np.isfinite(flux):
            jdx.append(i)

    brightest_catalog = catalog_sources['phot_rp_mean_flux'][jdx].max()
    faintest_catalog = catalog_sources['phot_rp_mean_flux'][jdx].min()

    (bright_central_detected_stars, bright_central_catalog_stars) = wcs.extract_bright_central_stars(setup,detected_sources,
                                                                                                    catalog_sources,
                                                                                                    image_wcs, log,
                                                                                                    radius)

    assert len(bright_central_detected_stars) < len(detected_sources)
    assert len(bright_central_catalog_stars) < len(catalog_sources)

    centre = SkyCoord(image_wcs.wcs.crval[0], image_wcs.wcs.crval[1],
                      frame='icrs', unit=(units.deg, units.deg))

    det_stars = SkyCoord(bright_central_detected_stars['ra'], bright_central_detected_stars['dec'],
                     frame='icrs', unit=(units.deg, units.deg))

    separations = centre.separation(det_stars)

    brightest_remaining = bright_central_detected_stars['ref_flux'].max()
    faintest_remaining = bright_central_detected_stars['ref_flux'].min()

    print('Detected ',brightest_detected,faintest_detected)
    print('Detected, remaining ',brightest_remaining,faintest_remaining)

    assert separations.to(units.deg).value.max() <= radius
    assert brightest_remaining < brightest_detected
    assert faintest_remaining > faintest_detected

    cat_stars = SkyCoord(bright_central_catalog_stars['ra'], bright_central_catalog_stars['dec'],
                     frame='icrs', unit=(units.deg, units.deg))

    separations = centre.separation(cat_stars)

    jdx = []
    for i,flux in enumerate(bright_central_catalog_stars['phot_rp_mean_flux']):
        if np.isfinite(flux):
            jdx.append(i)

    brightest_remaining = bright_central_catalog_stars['phot_rp_mean_flux'][jdx].max()
    faintest_remaining = bright_central_catalog_stars['phot_rp_mean_flux'][jdx].min()

    print('Catalog ',brightest_catalog,faintest_catalog)
    print('Catalog, remaining ',brightest_remaining,faintest_remaining)

    assert separations.to(units.deg).value.max() <= radius
    assert brightest_remaining < brightest_catalog
    assert faintest_remaining > faintest_catalog

    logs.close_log(log)

class AffineTransform():

    def __init__(self):

        self.scale = [ 0.0, 0.0 ]
        self.translation = [ 0.0, 0.0 ]
        self.rotation = 0.0
        self.shear = 0.0
        self.matrix = np.zeros([3,3])

def test_image_wcs_object():

    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DIR})

    log = logs.start_stage_log( cwd, 'test_wcs' )

    header1 = {'CTYPE1':'RA---TAN',
              'CTYPE2':'DEC--TAN',
              'CRVAL1': 270.0749817,
              'CRVAL2': -28.5375586,
              'CRPIX1': 2048.0,
              'CRPIX2': 2048.0,
              'CD1_1' : -0.0001081,
              'CD1_2' : 0.0,
              'CD2_1' : 0.0,
              'CD2_2' : 0.0001081,
              'NAXIS1': 4096,
              'NAXIS2': 4096}

    header2 = {'CTYPE1':'RA---TAN',
              'CTYPE2':'DEC--TAN',
              'CRVAL1': 270.0749817,
              'CRVAL2': -28.5375586,
              'CRPIX1': 2040.5850854933833,
              'CRPIX2': 2047.4916249890805,
              'CD1_1' : -0.0001081,
              'CD1_2' : 0.0,
              'CD2_1' : 0.0,
              'CD2_2' : 0.0001081,
              'NAXIS1': 4096,
              'NAXIS2': 4096}

    image_wcs1 = aWCS(header1)
    image_wcs2 = aWCS(header2)

    test_xy = np.array( [[2059.161, 2000.866]] )
    test_xy_ref = np.array( [[header1['CRPIX1'], header1['CRPIX2']]] )
    test_radec = np.array([ [270.0772669215088, -28.532480828945243] ])
    test_radec_ref = np.array([ [270.0749817, -28.5375586] ])

    test_c = SkyCoord(test_radec, frame='icrs', unit=(units.deg, units.deg))

    table_data = [ Column(name='ra', data=test_radec[:,0]),
                   Column(name='dec', data=test_radec[:,1]) ]
    catalog_sources = Table(data=table_data)

    world_coords = image_wcs1.wcs_pix2world(test_xy, 1)
    world_coords_ref = image_wcs1.wcs_pix2world(test_xy_ref, 1)
    pixel_coords = image_wcs1.wcs_world2pix(test_radec, 1)
    pixel_coords_ref = image_wcs1.wcs_world2pix(test_radec_ref, 1)
    c = SkyCoord(world_coords, frame='icrs', unit=(units.deg, units.deg))

    catalog_sources = wcs.calc_image_coordinates_astropy(setup, image_wcs1, catalog_sources,log)

    sep = test_c.separation(c)

    print('Initial WCS:')
    print('-> Input RA, Dec: '+repr(test_radec))
    print('-> Computed x,y pixel position: '+repr(pixel_coords))
    print('-> Computed x,y pixel ref position: '+repr(pixel_coords_ref))
    print('-> Input x,y coordinates: '+repr(test_xy))
    print('-> Computed world coordinates: '+repr(world_coords))
    print('-> Computed world coordinates (wcs function): '+repr(catalog_sources))
    print('-> Computed world coordinates ref pixel: '+repr(world_coords_ref))
    print('-> Separation = '+str(sep[0]))

    world_coords = image_wcs2.wcs_pix2world(test_xy, 1)
    c = SkyCoord(world_coords, frame='icrs', unit=(units.deg, units.deg))

    sep = test_c.separation(c)

    test_xy_ref = np.array( [[header2['CRPIX1'], header2['CRPIX2']]] )
    world_coords_ref = image_wcs2.wcs_pix2world(test_xy_ref, 1)
    pixel_coords = image_wcs2.wcs_world2pix(test_radec, 1)

    table_data = [ Column(name='ra', data=test_radec[:,0]),
                   Column(name='dec', data=test_radec[:,1]) ]
    catalog_sources = Table(data=table_data)

    catalog_sources2 = wcs.calc_image_coordinates_astropy(setup, image_wcs2, catalog_sources,log)

    print('Refined WCS:')
    print('-> Input RA, Dec: '+repr(test_radec))
    print('-> Computed x,y pixel position: '+repr(pixel_coords))
    print('-> Input x,y coordinates: '+repr(test_xy))
    print('-> Computed world coordinates: '+repr(world_coords))
    print('-> Separation = '+str(sep[0]))
    print('-> Computed world coordinates ref pixel: '+repr(world_coords_ref))
    print('-> Computed world coordinates (wcs function): '+repr(catalog_sources))
    print('-> Computed world coordinates (wcs function): '+repr(catalog_sources2))

    table_data = [ Column(name='x', data=test_xy[:,0]),
                   Column(name='y', data=test_xy[:,1]) ]
    catalog_sources = Table(data=table_data)
    catalog_sources2 = wcs.calc_world_coordinates_astropy(setup, image_wcs2, catalog_sources,log)
    print('-> Ref star computed world coordinates (wcs function): '+repr(catalog_sources2))

    transform = AffineTransform()
    transform.translation = [-0.4098763052732579, -0.6609356377717006]
    transform.scale = [1.0011820643023979, 1.0012879895508038]
    transform.rotation = 0.002582311161541344
    transform.shear = 0.0
    transform.matrix = np.array( [[ 1.00117873, -0.00328719, -0.40987631],
                                  [ 0.00258536,  1.00128259, -0.66093564],
                                  [ 0.        ,  0.        ,  1.        ]] )

    catalog_sources2 = calc_coord_offsets.transform_coordinates(setup, catalog_sources2, transform, coords='radec')
    print('-> Ref star computed world coordinates (wcs function): '+repr(catalog_sources2))
    c = SkyCoord(catalog_sources2['ra'], catalog_sources2['dec'], frame='icrs', unit=(units.deg, units.deg))

    sep = test_c.separation(c)
    print('-> Separation = '+str(sep[0]))

    logs.close_log(log)

def test_select_nearest_stars_in_catalog():

    nstars = 10
    star_index = np.linspace(1,nstars,nstars)
    ra = np.linspace(267.1,267.9,nstars)
    dec = np.linspace(-29.1,-29.9,nstars)
    x = np.linspace(100.0,100.0+float(nstars),nstars)
    y = np.linspace(150.0,150.0+float(nstars),nstars)

    catalog_sources = Table( [Column(name='star_id', data=star_index),
                                Column(name='ra', data=ra),
                                Column(name='dec', data=dec),
                                Column(name='x', data=x),
                                Column(name='y', data=y)] )

    star_index2 = star_index + 1
    ra_offset = 0.00005
    dec_offset = 0.000025
    ra2 = ra + ra_offset
    dec2 = dec + dec_offset
    x2 = x
    y2 = y

    detected_sources = Table( [Column(name='star_id', data=star_index2),
                                Column(name='ra', data=ra2),
                                Column(name='dec', data=dec2),
                                Column(name='x', data=x2),
                                Column(name='y', data=y2)] )

    catalog_star = SkyCoord(catalog_sources['ra'][0],
                             catalog_sources['dec'][0],
                             frame='icrs', unit=(units.deg, units.deg))

    dra = (20.0 * 0.389) / 3600.0
    ddec = (20.0 * 0.389) / 3600.0
    tol = ra_offset*1.2

    nearest_stars_index = wcs.select_nearest_stars_in_catalog(catalog_sources, detected_sources,
                                        catalog_star,dra,ddec)

    assert type(nearest_stars_index) == type([])

    positions = SkyCoord(detected_sources['ra'][nearest_stars_index],
                        detected_sources['dec'][nearest_stars_index],
                        frame='icrs', unit=(units.deg, units.deg))
    separations = catalog_star.separation(positions)
    assert np.all(separations.value < tol)

def test_match_star_without_duplication():

    log = logs.start_stage_log( cwd, 'test_wcs' )

    nstars = 10
    star_index = np.linspace(1,nstars,nstars)
    ra = np.linspace(267.1,267.9,nstars)
    dec = np.linspace(-29.1,-29.9,nstars)
    x = np.linspace(100.0,100.0+float(nstars),nstars)
    y = np.linspace(150.0,150.0+float(nstars),nstars)

    catalog_sources = Table( [Column(name='star_id', data=star_index),
                                Column(name='ra', data=ra),
                                Column(name='dec', data=dec),
                                Column(name='x', data=x),
                                Column(name='y', data=y)] )

    star_index2 = star_index + 1
    ra_offset = 0.000025
    dec_offset = 0.000015
    ra2 = ra + ra_offset
    dec2 = dec + dec_offset
    x2 = x
    y2 = y

    detected_sources = Table( [Column(name='star_id', data=star_index2),
                                Column(name='ra', data=ra2),
                                Column(name='dec', data=dec2),
                                Column(name='x', data=x2),
                                Column(name='y', data=y2)] )

    cat_idx = 0
    catalog_star = SkyCoord(catalog_sources['ra'][cat_idx],
                             catalog_sources['dec'][cat_idx],
                             frame='icrs', unit=(units.deg, units.deg))

    dra = (20.0 * 0.389) / 3600.0
    ddec = (20.0 * 0.389) / 3600.0
    tol = ra_offset*2.0

    det_sources = SkyCoord(detected_sources['ra'],
                                       detected_sources['dec'],
                                       frame='icrs',
                                       unit=(units.deg, units.deg))

    matched_stars = match_utils.StarMatchIndex()

    nearest_stars_index = wcs.select_nearest_stars_in_catalog(catalog_sources, detected_sources,
                                        catalog_star,dra,ddec)

    matched_stars = wcs.match_star_without_duplication(catalog_star,cat_idx,det_sources,nearest_stars_index,
                                        detected_sources, catalog_sources,
                                        tol,matched_stars,log,verbose=True)

    assert type(matched_stars) == type(match_utils.StarMatchIndex())
    print(matched_stars.summary())
    assert matched_stars.cat1_index[0] == matched_stars.cat2_index[0]

    logs.close_log(log)

def test_cross_match_star_catalogs():

    log = logs.start_stage_log( cwd, 'test_wcs' )

    do_full_frame_test = False
    if do_full_frame_test:
        header = { 'NAXIS1': 4096, 'NAXIS2': 4096,
                   'CTYPE1': 'RA---TAN', 'CTYPE2': 'DEC--TAN',
                   'CRPIX1': 2048.0, 'CRPIX2': 2048.0,
                   'CRVAL1': 267.8358954, 'CRVAL2': -30.0608178,
                   'CUNIT1': 'deg', 'CUNIT2': 'deg',
                   'CD1_1': -0.0001081, 'CD1_2': 0.0,
                   'CD2_1': 0.0, 'CD2_2': 0.0001081 }
        image_wcs = aWCS.WCS(header)

        nstars = 100000
        star_index = list(range(0,nstars,1))
        pix_coords = np.zeros([len(star_index),2])
        for j in star_index:
            pix_coords[j-1,0] = np.random.uniform(1.0,4096.0)
            pix_coords[j-1,1] = np.random.uniform(1.0,4096.0)
        world_coords = image_wcs.wcs_pix2world(pix_coords,1)

        catalog_sources = Table( [Column(name='star_id', data=star_index),
                                    Column(name='ra', data=world_coords[:,0]),
                                    Column(name='dec', data=world_coords[:,1]),
                                    Column(name='x', data=pix_coords[:,0]),
                                    Column(name='y', data=pix_coords[:,1])] )

        star_index2 = list(np.array(star_index) + 1)
        ra_offset = 0.000025
        dec_offset = 0.0
        world_coords2 = np.zeros([len(star_index),2])
        world_coords2[:,0] = world_coords[:,0] + ra_offset
        world_coords2[:,1] = world_coords[:,1] + dec_offset

        detected_sources = Table( [Column(name='star_id', data=star_index2),
                                    Column(name='ra', data=world_coords2[:,0]),
                                    Column(name='dec', data=world_coords2[:,1]),
                                    Column(name='x', data=pix_coords[:,0]),
                                    Column(name='y', data=pix_coords[:,1])] )

        matched_stars = wcs.cross_match_star_catalogs(detected_sources,
                                                        catalog_sources,
                                                        star_index, log)

        assert type(matched_stars) == type(match_utils.StarMatchIndex())

    star_id = [84291, 84292]
    x = [3233.74923, 3237.4708]
    y = [1381.30456, 1381.39523]
    ra = [267.98273, 267.9832]
    dec = [-29.98989, -29.9899]
    catalog_sources = Table( [Column(name='star_id', data=star_id),
                                Column(name='ra', data=ra),
                                Column(name='dec', data=dec),
                                Column(name='x', data=x),
                                Column(name='y', data=y)] )
    star_index = [0,1]
    detected_sources = Table( [Column(name='star_id', data=[78284]),
                                Column(name='ra', data=[267.98297]),
                                Column(name='dec', data=[-29.98976]),
                                Column(name='x', data=[3259.79919]),
                                Column(name='y', data=[1489.70321])] )
    matched_stars = wcs.cross_match_star_catalogs(detected_sources,
                                                    catalog_sources,
                                                    star_index, log)
    print(matched_stars.summary())
    logs.close_log(log)

if __name__ == '__main__':

    #test_reference_astrometry()
    #test_search_vizier_for_2mass_sources()
    #test_fetch_catalog_sources_for_field()
    #test_search_vizier_for_gaia_sources()
    test_match_stars_world_coords()
    #test_image_wcs()
    #test_calc_world_coordinates()
    #test_calc_world_coordinates_astropy()
    #test_calc_image_coordinates()
    #test_calc_image_coordinates_astropy2()
    #test_calc_image_coordinates_astropy3()
    #test_extract_bright_central_stars()
    #test_match_stars_pixel_coords()
    #test_image_wcs_object()
    #test_select_nearest_stars_in_catalog()
    #test_match_star_without_duplication()
    #test_cross_match_star_catalogs()
