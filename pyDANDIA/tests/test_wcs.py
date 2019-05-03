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
from astropy.io import fits
from astropy.table import Table, Column
from astropy.wcs import WCS as aWCS
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
    
    detected_coords = np.array(  [  [269.55310076, -28.04655246], 
                                    [269.52451024, -28.04663018],
                                    [269.53827323, -28.04670613], 
                                    [269.53893954, -28.04667918],
                                    [269.5575608 , -28.04728878]  ] )
    
    detected_data = [ Column(name='ra', data=detected_coords[:,0]),
                      Column(name='dec', data=detected_coords[:,1]) ]
    
    detected_sources = Table(data=detected_data)
    
    catalog_coords = np.array( [ [269.52935399, -28.04729399], 
                                 [269.54868567, -28.04728673],
                                 [269.5575608 , -28.04728878], 
                                 [269.51156351, -28.04750404] ] )
    
    catalog_data = [ Column(name='ra', data=catalog_coords[:,0]),
                     Column(name='dec', data=catalog_coords[:,1]) ]
    
    catalog_sources = Table(data=catalog_data)
    
    test_match = match_utils.StarMatchIndex()
    
    matched_stars = wcs.match_stars_world_coords(detected_sources,
                                                 catalog_sources,log,
                                                 verbose=True)
    
    assert matched_stars.n_match == 1
    assert type(matched_stars) == type(test_match)
    assert detected_coords[matched_stars.cat1_index[0],0] == catalog_coords[matched_stars.cat2_index[0],0]
    assert detected_coords[matched_stars.cat1_index[0],1] == catalog_coords[matched_stars.cat2_index[0],1]
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
    
if __name__ == '__main__':

    test_reference_astrometry()
    #test_search_vizier_for_2mass_sources()
    #test_fetch_catalog_sources_for_field()
    #test_search_vizier_for_gaia_sources()
    #test_match_stars_world_coords()
    #test_image_wcs()
    #test_calc_world_coordinates()
    #test_calc_world_coordinates_astropy()
    #test_calc_image_coordinates()
    #test_calc_image_coordinates_astropy2()
    #test_calc_image_coordinates_astropy3()
    #test_extract_bright_central_stars()
    