from os import path, remove
from pyDANDIA import crossmatch
from pyDANDIA import crossmatch_datasets
from pyDANDIA import crossmatch_field_gaia
from pyDANDIA import match_utils
from pyDANDIA import metadata
from pyDANDIA import logs
import test_field_photometry
from astropy.table import Table, Column
from astropy import units as u
import numpy as np

def test_params():
    params = {'primary_ref': 'primary_ref_dataset',
              'datasets': { 'primary_ref_dataset': ['primary_ref', '/Users/rstreet1/OMEGA/test_data/non_ref_dataset_p/', 'ip'],
                            'dataset0' : [ 'non_ref', '/Users/rstreet1/OMEGA/test_data/non_ref_dataset0/', 'ip' ],
                            'dataset1' : [ 'non_ref', '/Users/rstreet1/OMEGA/test_data/non_ref_dataset1/', 'rp' ],
                            'dataset2' : [ 'non_ref', '/Users/rstreet1/OMEGA/test_data/non_ref_dataset1/', 'gp' ]},
              'file_path': 'crossmatch_table.fits',
              'log_dir': '.',
              'gaia_dr': 'Gaia_DR2',
              'separation_threshold': (2.0/3600.0)*u.deg}

    return params

def test_matched_stars():
    matched_stars = match_utils.StarMatchIndex()
    p = {'cat1_index': 1,
         'cat1_ra': 250.0,
         'cat1_dec': -27.5,
         'cat1_x': 1000.0,
         'cat1_y': 1000.0,
         'cat2_index': 1,
         'cat2_ra': 250.1,
         'cat2_dec': -27.5,
         'cat2_x': 1000.0,
         'cat2_y': 1000.0,
         'separation': 0.1}
    matched_stars.add_match(p)
    return matched_stars

def test_orphans():
    orphans = match_utils.StarMatchIndex()
    p = {'cat1_index': None,
         'cat1_ra': 0.0,
         'cat1_dec': 0.0,
         'cat1_x': 0.0,
         'cat1_y': 0.0,
         'cat2_index': 1,
         'cat2_ra': 252.0,
         'cat2_dec': -27.2,
         'cat2_x': 0.0,
         'cat2_y': 0.0,
         'separation': -1.0}
    orphans.add_match(p)
    return orphans

def test_field_index(xmatch):

    xmatch.field_index.add_row([1,267.61861696019145, -29.829605383706895, 4, 1, None, 1, 0, 0, 0])
    xmatch.field_index.add_row([2,267.70228408545813, -29.83032824102953, 4, 2, None, 2, 0, 0, 0])
    xmatch.field_index.add_row([3,267.9873108673885, -29.829734325692858, 3, 1, None, 3, 0, 0, 0])
    xmatch.field_index.add_row([4,267.9585073984874, -29.83002538112054, 3, 2, None, 4, 0, 0, 0])
    xmatch.field_index.add_row([5,267.9623466389135, -29.82994179424344, 3, 3, None, 5, 0, 0, 0])
    xmatch.field_index.add_row([6,267.943683356543, -29.830113202355186, 3, 4, None, 6, 0, 0, 0])
    xmatch.field_index.add_row([7,267.90449275089594, -29.830465810573223, 3, 5, None, 7, 0, 0, 0])
    xmatch.field_index.add_row([8,267.9504950018423, -29.830247462548577, 3, 6, None, 8, 0, 0, 0])
    xmatch.field_index.add_row([9,267.9778110411362, -29.83012645385565, 3, 7, None, 9, 0, 0, 0])
    xmatch.field_index.add_row([10,267.7950771349625, -29.830849947501875, 4, 3, None, 10, 0, 0, 0])
    xmatch.field_index.add_row([11,268.06583501505446, -29.83070761362742, 3, 84, 4056397427727492224, 11, 0, 0, 0])
    xmatch.field_index.add_row([12,268.0714302057775, -29.830599528895256, 3, 85, 4056403303242709888, 12, 0, 0, 0])
    xmatch.field_index.add_row([13,268.07569655803013, -29.83064274854432, 3, 86, 4056403307573006208, 13, 0, 0, 0])
    xmatch.field_index.add_row([14,268.07663104709775, -29.830575490772073, 3, 87, 4056403303204313344, 14, 0, 0, 0])
    xmatch.field_index.add_row([15,268.07816636284224, -29.830684523662065, 3, 88, 4056403307572525184, 15, 0, 0, 0])

    return xmatch

def test_gaia_catalog():
    nstars = 5
    table_data = [  Column(name='source_id', data = np.array([4056397427727492224, 4056403303242709888, 4056403307573006208, 4056403307618099840, 4056403307572525184])),
                    Column(name='ra', data = np.array([268.0657435285, 268.07133070742, 268.07560517009, 268.07658092476, 268.07819547689])),
                    Column(name='ra_error', data = np.array([1.023, 0.1089, 0.4107, 38.0801, 2.0125])),
                    Column(name='dec', data = np.array([-29.83088151459, -29.83073507254, -29.83081220358, -29.83107857205, -29.83092724519])),
                    Column(name='dec_error', data = np.array([0.9754, 0.0946, 0.3265, 17.3181, 1.3702])) ]

    colnames = ['phot_g_mean_flux', 'phot_g_mean_flux_error', 'phot_bp_mean_flux', 'phot_bp_mean_flux_error',
                'phot_rp_mean_flux', 'phot_rp_mean_flux_error', 'proper_motion', 'pm_ra', 'pm_dec',
                'parallax', 'parallax_error']

    for col in colnames:
        table_data.append( Column(name=col, data=np.zeros(nstars)) )

    gaia_star_field_ids = [ 11, 12, 13, 14, 15 ]

    return Table(table_data), gaia_star_field_ids

def test_stars_table(xmatch):

    xmatch.stars.add_row([1,267.61861696019145, -29.829605383706895]+[0.0]*36)
    xmatch.stars.add_row([2,267.70228408545813, -29.83032824102953]+[0.0]*36)
    xmatch.stars.add_row([3,267.9873108673885, -29.829734325692858]+[0.0]*36)
    xmatch.stars.add_row([4,267.9585073984874, -29.83002538112054]+[0.0]*36)
    xmatch.stars.add_row([5,267.9623466389135, -29.82994179424344]+[0.0]*36)
    xmatch.stars.add_row([6,267.943683356543, -29.830113202355186]+[0.0]*36)
    xmatch.stars.add_row([7,267.90449275089594, -29.830465810573223]+[0.0]*36)
    xmatch.stars.add_row([8,267.9504950018423, -29.830247462548577]+[0.0]*36)
    xmatch.stars.add_row([9,267.9778110411362, -29.83012645385565]+[0.0]*36)
    xmatch.stars.add_row([10,267.7950771349625, -29.830849947501875]+[0.0]*36)
    xmatch.stars.add_row([11,268.06583501505446, -29.83070761362742]+[0.0]*36)
    xmatch.stars.add_row([12,268.0714302057775, -29.830599528895256]+[0.0]*36)
    xmatch.stars.add_row([13,268.07569655803013, -29.83064274854432]+[0.0]*36)
    xmatch.stars.add_row([14,268.07663104709775, -29.830575490772073]+[0.0]*36)
    xmatch.stars.add_row([15,268.07816636284224, -29.830684523662065]+[0.0]*36)

    return xmatch

def test_metadata():
    dataset_metadata = metadata.MetaData()
    nstars = 10
    star_catalog = np.zeros((nstars,21))
    for j in range(1,len(star_catalog),1):
        star_catalog[j-1,0] = j
        star_catalog[j-1,1] = 250.0
        star_catalog[j-1,2] = -17.5
    star_catalog[1,13] = '4062470305390995584'
    dataset_metadata.create_star_catalog_layer(data=star_catalog)


    return dataset_metadata

def test_create():
    params = test_params()

    xmatch = crossmatch.CrossMatchTable()
    xmatch.create(params)
    xmatch.init_field_index(primary_metadata)
    for key in params['datasets'].keys():
        assert(key+'_index' in xmatch.field_index.colnames)
    assert(len(xmatch.datasets) == len(params['datasets']))

def test_add_dataset():

    params = {'primary_ref_dir': '/Users/rstreet1/OMEGA/test_data/primary_ref_dataset/',
              'primary_ref_filter': 'ip',
              'red_dir_list': [ '/Users/rstreet1/OMEGA/test_data/non_ref_dataset/' ],
              'red_dataset_filters': [ 'rp' ],
              'file_path': 'crossmatch_table.fits'}

    xmatch = crossmatch.CrossMatchTable()
    xmatch.create(params)

    test_keys = ['PRIMARY', 'PRIMFILT', 'DATASET0', 'FILTER0']
    assert(x in xmatch.header.keys() for x in test_keys)
    assert(len(xmatch.matched_stars) == 1)

    xmatch.add_dataset('/Users/rstreet1/OMEGA/test_data/non_ref_dataset2/', 'rp')
    assert(len(xmatch.matched_stars) == 2)

def test_dataset_index():

    params = {'primary_ref_dir': '/Users/rstreet1/OMEGA/test_data/primary_ref_dataset',
              'primary_ref_filter': 'ip',
              'red_dir_list': [ '/Users/rstreet1/OMEGA/test_data/non_ref_dataset' ],
              'red_dataset_filters': [ 'rp' ],
              'file_path': 'crossmatch_table.fits'}

    xmatch = crossmatch.CrossMatchTable()
    xmatch.create(params)

    dataset_idx = xmatch.dataset_index(params['red_dir_list'][0]+'/')
    assert(dataset_idx == 0)

def test_save():

    params = test_params()

    if path.isfile(params['file_path']):
      remove(params['file_path'])

    xmatch = crossmatch.CrossMatchTable()
    xmatch.create(params)

    row = [1, 250.0, -27.5, 1, 1, '4062470305390987584', 1, 0, 0, 0]
    xmatch.field_index.add_row(row)

    row = [1, 'cpt1m010-fl16-20170720-0104-e91.fits', 'i', 2456655.5000]
    xmatch.images.add_row(row)

    row = [1, 256.5, -27.2, \
            17.0, 0.02, 16.7, 0.02, 16.0, 0.02, \
            17.0, 0.02, 16.7, 0.02, 16.0, 0.02, \
            17.0, 0.02, 16.7, 0.02, 16.0, 0.02, \
            '4062470305390987584', 256.5, 0.001, -27.2, 0.001] + [0.0]*13
    xmatch.stars.add_row(row)

    xmatch.save(params['file_path'])

    assert(path.isfile(params['file_path']))

def test_load():
    params = test_params()
    test_save()

    xmatch = crossmatch.CrossMatchTable()
    xmatch.load(params['file_path'])

    assert(xmatch.datasets != None)
    assert(xmatch.field_index != None)
    assert(type(xmatch.datasets) == type(Table()))
    assert(type(xmatch.field_index) == type(Table()))
    assert(len(xmatch.datasets) > 0)
    assert(len(xmatch.field_index) > 0)

def test_init_field_index():
    params = test_params()
    xmatch = crossmatch.CrossMatchTable()
    xmatch.create(params)

    nstars = 10
    m = metadata.MetaData()
    star_catalog = np.zeros((nstars,21))
    for j in range(1,len(star_catalog),1):
        star_catalog[j-1,0] = j
        star_catalog[j-1,1] = 250.0
        star_catalog[j-1,2] = -17.5
    m.create_star_catalog_layer(data=star_catalog)

    xmatch.init_field_index(m, 'ip')

    assert len(xmatch.field_index) == nstars

def test_init_stars_table():
    params = test_params()
    xmatch = crossmatch.CrossMatchTable()
    xmatch.create(params)

    row = [1, 250.0, -27.5, 1, 1, '4062470305390987584', 1, 0, 0, 0]
    xmatch.field_index.add_row(row)

    xmatch.init_stars_table()

    assert(len(xmatch.field_index) == len(xmatch.stars))

def test_update_field_index():

    params = test_params()
    log = logs.start_stage_log( params['log_dir'], 'test_crossmatch' )
    xmatch = crossmatch.CrossMatchTable()
    xmatch.create(params)
    row = [1, 250.0, -27.5, 1, 1, '4062470305390987584', 1, 0, 0, 0]
    xmatch.field_index.add_row(row)
    dataset_code = 'dataset0'
    matched_stars = test_matched_stars()
    orphans = test_orphans()
    dataset_metadata = test_metadata()

    xmatch.update_field_index(dataset_code, matched_stars, orphans,
                             dataset_metadata, log)

    # Check that the index of matched star from dataset0 has been added to the
    # row entry for the corresponding star:
    assert(xmatch.field_index[dataset_code+'_index'][1] == 1)
    # Check that an additional entry has been added to the field index for
    # the orphan object:
    assert(len(xmatch.field_index) == 2)

    logs.close_log(log)

def test_assign_quadrants():
    params = test_params()
    log = logs.start_stage_log( params['log_dir'], 'test_crossmatch' )
    xmatch = crossmatch.CrossMatchTable()
    xmatch.create(params)
    row = [1, 250.0, -27.5, 0, 0, '4062470305390987584', 1, 0, 0, 0]
    xmatch.field_index.add_row(row)

    xmatch.assign_stars_to_quadrants()

    assert(xmatch.field_index['quadrant'][0] != 0)
    assert(xmatch.field_index['quadrant_id'][0] != 0)

    logs.close_log(log)

def test_cone_search():

    params = test_params()
    log = logs.start_stage_log( params['log_dir'], 'test_crossmatch' )

    xmatch = crossmatch.CrossMatchTable()
    xmatch.create(params)
    row = [1, 250.0, -27.5, 1, 1, '4062470305390987584', 1, 0, 0, 0]
    xmatch.field_index.add_row(row)

    idx = xmatch.cone_search(250.0, -27.5, 2.0, log=log)
    print(idx)
    assert(len(idx) == 1)
    logs.close_log(log)

def test_match_field_index_with_gaia_catalog():

    params = test_params()
    log = logs.start_stage_log( params['log_dir'], 'test_crossmatch' )

    xmatch = crossmatch.CrossMatchTable()
    xmatch.create(params)
    xmatch = test_field_index(xmatch)
    xmatch = test_stars_table(xmatch)

    (gaia_data, gaia_star_field_ids) = test_gaia_catalog()

    xmatch.match_field_index_with_gaia_catalog(gaia_data, params, log)

    for g, gaia_field_id in enumerate(gaia_star_field_ids):
        assert(int(xmatch.field_index['gaia_source_id'][gaia_field_id-1]) == int(gaia_data['source_id'][g]))
        assert(int(xmatch.stars['gaia_source_id'][gaia_field_id-1]) == int(gaia_data['source_id'][g]))
        assert(xmatch.stars['gaia_ra'][gaia_field_id-1] == gaia_data['ra'][g])
        assert(xmatch.stars['gaia_dec'][gaia_field_id-1] == gaia_data['dec'][g])

    logs.close_log(log)

def test_load_gaia_catalog_file():

    params = {}
    params['gaia_catalog_file'] = '/Users/rstreet1/ROMEREA/test_data/config/ROME-FIELD-01_Gaia_EDR3.fits'
    params['log_dir'] = '.'

    log = logs.start_stage_log( params['log_dir'], 'test_crossmatch' )

    gaia_data = crossmatch_field_gaia.load_gaia_catalog(params,log)

    assert(type(gaia_data) == type(Table()))

    logs.close_log(log)

def test_record_dataset_stamps():

    params = test_params()
    log = logs.start_stage_log( params['log_dir'], 'test_crossmatch' )

    xmatch = crossmatch.CrossMatchTable()
    xmatch.create(params)

    meta = metadata.MetaData()
    meta = test_field_photometry.test_star_catalog(meta)
    meta = test_field_photometry.test_headers_summary(meta)
    meta = test_field_photometry.test_images_stats(meta)
    meta = test_field_photometry.test_stamps_table(meta)

    xmatch.record_dataset_stamps('dataset0', meta, log)

    assert('stamps' in dir(xmatch))
    assert(type(xmatch.stamps) == type(Table()))
    columns = ['dataset_code', 'filename', 'stamp_id', 'xmin', 'xmax', 'ymin', 'ymax',\
                'warp_matrix_0', 'warp_matrix_1', 'warp_matrix_2', \
                'warp_matrix_3', 'warp_matrix_4', 'warp_matrix_5', \
                'warp_matrix_6', 'warp_matrix_7', 'warp_matrix_8']
    for column in columns:
        assert(column in xmatch.stamps.colnames)
    assert(len(xmatch.stamps) == len(meta.images_stats[1])*len(meta.stamps[1]))

def test_get_imagesets():

    params = test_params()
    xmatch = crossmatch.CrossMatchTable()
    xmatch.create(params)

    imagesets = xmatch.get_imagesets()
    print(imagesets)

def test_create_normalizations_tables():

    params = {'primary_ref': 'ROME-FIELD-01_lsc-doma-1m0-05-fa15_ip',
              'datasets': { 'ROME-FIELD-01_lsc-doma-1m0-05-fa15_ip': ['primary_ref', '/Users/rstreet1/OMEGA/test_data/non_ref_dataset_p/', 'ip'],
                            'ROME-FIELD-01_coj-doma-1m0-11-fa12_ip' : [ 'non_ref', '/Users/rstreet1/OMEGA/test_data/non_ref_dataset0/', 'ip' ]},
              'file_path': 'crossmatch_table.fits',
              'log_dir': '.',
              'gaia_dr': 'Gaia_DR2',
              'separation_threshold': (2.0/3600.0)*u.deg}
    xmatch = crossmatch.CrossMatchTable()
    xmatch.create(params)
    xmatch.field_index.add_row([1,267.61861696019145, -29.829605383706895, 4, 1, '4056436121079692032', 1, 0])
    xmatch.field_index.add_row([2,267.70228408545813, -29.83032824102953, 4, 2, '4056436121079692033', 2, 0])
    xmatch.field_index.add_row([3,267.9873108673885, -29.829734325692858, 3, 1, '4056436121079692034', 3, 0])
    xmatch.field_index.add_row([4,267.9585073984874, -29.83002538112054, 3, 2, '4056436121079692035', 4, 0])

    xmatch.create_normalizations_tables()
    
    # Test default creation of a set of empty tables:
    assert(hasattr(xmatch, 'normalizations'))
    assert(type(xmatch.normalizations) == type({}))
    for priref, data_table in xmatch.normalizations.items():
        assert(type(data_table) == type(Table([])))


if __name__ == '__main__':
    #test_create()
#    test_add_dataset()
#    test_dataset_index()
    #test_save()
    #test_load()
    #test_init_field_index()
    #test_update_field_index()
    #test_assign_quadrants()
    #test_cone_search()
    #test_init_stars_table()
    #test_match_field_index_with_gaia_catalog()
    #test_load_gaia_catalog_file()
    #test_record_dataset_stamps()
    #test_get_imagesets()
    test_create_normalizations_tables()
