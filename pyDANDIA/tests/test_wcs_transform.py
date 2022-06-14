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
import utilities
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

def simulate_detected_sources():
    nstars = 18
    xmin = 1.0
    xmax = 4000.0
    ymin = 1.0
    ymax = 4000.0
    ramin = 223.180
    decmin = -63.2
    pixscale = 0.389
    median_flux = 10000.0

    detected_objects = np.zeros((nstars,9))
    detected_objects[:,0] = [61595,61596,61597,61598,61599,61600,61634,61635,
                            61636,61637,61671,61672,61673,61674,61675,61702,
                            61705,61706]
    detected_objects[:,1] = [1866.259876089409,1891.7929031808592,1907.0905410137123,
                             1935.2019124607257,2148.2925744150702,2190.2529364275756,
                             1949.580327866272,1974.2498944762538,2127.59001619264,
                             2141.683057977994,1813.1582869695958,1821.3748461824944,
                             1911.4080152246452,2115.0533245264855,2132.57807039651,
                             1851.4796960070746,
                             1999.6118906403672,2189.1417642226297]
    detected_objects[:,2] = [1800.9986644016294,1801.4127610075063,1800.9376518684423,
                             1802.3499854421846,1800.8690300095063,1800.8420664908099,
                             1802.0695485480514,1801.9607906542526,1801.5581618458314,
                             1801.671344733282,1803.1659729966043,1803.3601722905473,
                             1802.7058701182052,1802.3518269480653,1803.258655935338,
                             1804.55459864179,
                             1804.0460251300985,1802.4371232391418]
    detected_objects[:,3] = [0.04533065812370925,0.039441372322136806,0.03591300787522929,
                             0.029428942594996132,-0.019720658320715612,-0.029398891415177642,
                             0.02611257615146829,0.020422528198685685,-0.014945532023227171,
                             -0.01819610017207023,0.057578202619573096,0.05568301961614626,
                             0.034916977252840445,-0.012053885445276902,-0.01609591473208465,
                             0.04873915641568434,
                             0.01457267890859899,-0.029142413014107516]
    detected_objects[:,4] = [-0.025292190637813843,-0.025249085347468718,-0.025301318248232445,
                             -0.025150015884186853,-0.02531061931530631,-0.025311577903082576,
                             -0.02518091483368367,-0.025193487933883756,-0.025236839975605545,
                             -0.025224135546864528,-0.025053652853700197,-0.025033388708081798,
                             -0.025110404212008046,-0.025151398159814146,-0.025052859219215844,
                             -0.02490671975642344,
                             -0.024968669536236103,-0.025139212860338926]
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

def simulate_catalog(setup, image_wcs,log,stellar_density,rotate_wcs, kwargs,
                    stellar_density_threshold):

    data = np.array([
            [223.242061399, -61.9983935237, 5874743064496619264, 7.3465,
                10.8444, 355.88, 1.3162, 512.97, 12.742, 108.09, 8.3288],
            [223.247903766, -61.9984166775, 5874742995777138304, 7.4695,
                11.8964, 266.52, 1.1721, 316.76, 14.313, 98.133, 7.9626],
            [223.251298944, -61.9983572828, 5874742995777137152, 9.4326,
                15.7704, 212.93, 1.1587, 271.24, 18.316, 94.534, 8.9558],
            [223.258011872, -61.9986626363, 5874742995780231936, 1.5482,
                1.7178, 106.83, 1.796, 261.57, 41.883, 150.93, 30.941],
            [223.307017903, -61.9981422323, 5874748145412556544, 2.0377,
                2.7718, 1757.5, 1.8132, 1626.2, 14.684, 764.54, 11.821],
            [223.316724201, -61.9980941483, 5874748149790023040, 5.9766,
                7.7822, 461.5, 1.5141, 678.87, 30.621, 247.82, 15.228],
            [223.261295109, -61.9984209183, 5874742995780232704, 18.5803,
                37.9282, 100.89, 0.88752, 103.31, 22.039, 34.597, 5.0799],
            [223.266921489, -61.9984304239, 5874742991451821312, 2.5111,
                3.4042, 1685.1, 2.0938, 2040.9, 14.245, 435.1, 12.465],
            [223.302311308, -61.9982577893, 5874742269897295872, 3.2925,
                4.7413, 794.92, 1.3553, 898.62, 14.362, 266.79, 14.369],
            [223.305527403, -61.9982527614, 5874742274273528960, 3.2611,
                4.6917, 759.93, 1.1985, 793.57, 21.635, 366.87, 18.355],
            [223.229811266, -61.9986643088, 5874742961420485504, 23.3793,
                48.4062, 97.253, 1.3575, 190.15, 48.785, 73.445, 25.249],
            [223.231754169, -61.9986652809, 5874742961417410304, 13.3221,
                16.0345, 158.33, 1.0581, 177.16, 11.072, 76.275, 7.7919],
            [223.128686534, -62.0160110162, 5874742789622460032, 3.3382,
                1.9457, 71.184, 1.3982, np.NaN, np.NaN, np.NaN, np.NaN],
            [223.299414868, -61.9983637717, 5874742274225733888, 35.9094,
                74.8406, 77.644, 1.1453, 64.153, 25.845, 15.749, 5.6494],
            [223.303503271, -61.9984220692, 5874742274222594560, 7.8912,
                12.7631, 270.88, 1.0314, 315.08, 16.312, 164.14, 37.204],
            [223.238681841, -61.9988062916, 5874743064499702656, 3.1049,
                2.4196, 67.791, 1.4177, np.NaN, np.NaN, np.NaN, np.NaN],
            [223.272819214, -61.9986086719, 5874743030187777664, 11.2452,
                22.349, 215.84, 1.4401, np.NaN, np.NaN, np.NaN, np.NaN],
            [223.316467112, -61.9984229245, 5874748149743289216, 9.0459,
                10.6429, 343.49, 1.3936, np.NaN, np.NaN, np.NaN, np.NaN],
            ])

    catalog_data = [ Column(name='ra', data=data[:,0]),
                      Column(name='dec', data=data[:,1]),
                      Column(name='source_id', data=data[:,2]),
                      Column(name='ra_error', data=data[:,3]),
                      Column(name='dec_error', data=data[:,4]),
                      Column(name='phot_g_mean_flux', data=data[:,5]),
                      Column(name='phot_g_mean_flux_error', data=data[:,6]),
                      Column(name='phot_rp_mean_flux', data=data[:,7]),
                      Column(name='phot_rp_mean_flux_error', data=data[:,8]),
                      Column(name='phot_bp_mean_flux', data=data[:,7]),
                      Column(name='phot_bp_mean_flux_error', data=data[:,8]) ]

    catalog_sources = Table(data=catalog_data)
    catalog_sources = wcs.calc_image_coordinates_astropy(setup, image_wcs,
                                                          catalog_sources, log,
                                                          stellar_density,
                                                          rotate_wcs, kwargs,
                                                          stellar_density_threshold)

    catalog_sources.add_column( Column(name='x1', data=np.copy(catalog_sources['x'])) )
    catalog_sources.add_column( Column(name='y1', data=np.copy(catalog_sources['y'])) )

    return catalog_sources

def simulate_image_wcs():
    header = fits.Header()
    header.set('ctype1','RA---TAN')
    header.set('ctype2','DEC--TAN')
    header.set('crpix1',2048.0)
    header.set('crpix2',2048.0)
    header.set('crval1',223.3142693)
    header.set('crval2',-62.0251698)
    header.set('cunit1','deg')
    header.set('cunit2','deg')
    header.set('cd1_1',-0.0001081)
    header.set('cd1_2',0.0000000)
    header.set('cd2_1',0.0000000)
    header.set('cd2_2',0.0001081)

    return aWCS(header)

def test_offset_transform():

    (setup, meta) = establish_test_env()
    log = logs.start_stage_log( cwd, 'test_wcs' )

    # Configure pixel transform:
    dx = 131.7
    dy = 1.3
    pixscale = 0.389/3600   # Converted to degrees
    transform_pixel = AffineTransform(translation=(dx, dy))

    # Generate catalog of detected sources:
    detected_sources = simulate_detected_sources()
    print('Detected source catalog: ')
    print(detected_sources)

    # Generate other required data input:
    image_wcs = simulate_image_wcs()
    rotate_wcs = 1
    stellar_density_threshold = 10.0
    selection_radius = 0.05 #degrees
    stellar_density = utilities.stellar_density_wcs(detected_sources,
                                                    image_wcs)
    it = 0
    kwargs = {'force_rotate_ref': False,
              'dx': dx, 'dy': dy,
              'trust_wcs': False,
              'max_iter_wcs': 5,
              'wcs_method': 'ransac'}

    # Generate test catalogs of reference sources:
    catalog_sources = simulate_catalog(setup, image_wcs, log, stellar_density,
                                        rotate_wcs, kwargs, stellar_density_threshold)

    print('Setup simulated dataset')

    dx = []
    dy = []
    for j in range(0,len(detected_sources),1):
        dx.append(detected_sources['x'][j] - catalog_sources['x1'][j])
        dy.append(detected_sources['y'][j] - catalog_sources['y1'][j])
    dx = np.array(dx)
    dy = np.array(dy)
    print('Estimated pixel offset between catalogs: ',np.median(dx),np.median(dy))

    # Apply initial-guess pixel offsets to reference catalog pixel positions:
    catalog_sources = reference_astrometry.update_catalog_image_coordinates(setup, image_wcs,
                                                catalog_sources, log,
                                                'catalog_sources_revised_'+str(it)+'.reg',
                                                stellar_density, rotate_wcs, kwargs,
                                                stellar_density_threshold,
                                                transform=transform_pixel,
                                                radius=selection_radius)
    it += 1
    print('Updated catalog pixel coordinates with initial offsets')
    print(catalog_sources)

    # Replecate matching algorithm
    matched_stars = match_utils.StarMatchIndex()
    matched_stars = wcs.match_stars_pixel_coords(detected_sources,
                                             catalog_sources,log,
                                             tol=2.0,verbose=True)
    print('Matched '+str(matched_stars.n_match)+' stars')

    transform_pixel = calc_coord_offsets.calc_pixel_transform(setup,
                                    catalog_sources[matched_stars.cat2_index],
                                    detected_sources[matched_stars.cat1_index],
                                    log, coordinates='pixel')
    print('Measured pixel transform: ',transform_pixel)

    catalog_sources = reference_astrometry.update_catalog_image_coordinates(setup, image_wcs,
                                                catalog_sources, log,
                                                'catalog_sources_revised_'+str(it)+'.reg',
                                                stellar_density, rotate_wcs, kwargs,
                                                stellar_density_threshold,
                                                transform=transform_pixel,
                                                radius=selection_radius)
    print('Revised catalog pixel coordinates with computed offsets')
    print(catalog_sources)

    (transform_world,field_centres) = calc_coord_offsets.calc_world_transform(setup,
                                            detected_sources[matched_stars.cat1_index],
                                            catalog_sources[matched_stars.cat2_index],
                                            log)
    print('Measured WCS transform: ',transform_world)

    detected_sources = calc_coord_offsets.transform_coordinates(setup, detected_sources,
                                                                    transform_world, field_centres,
                                                                    coords='radec',
                                                                    verbose=True)
    print(detected_sources)
    print(catalog_sources)

    for j in range(0,len(detected_sources),1):
        np.testing.assert_almost_equal(detected_sources['ra'][j], catalog_sources['ra'][j], 0.001)
        np.testing.assert_almost_equal(detected_sources['dec'][j], catalog_sources['dec'][j], 0.001)

    logs.close_log(log)


if __name__ == '__main__':
    test_offset_transform()
