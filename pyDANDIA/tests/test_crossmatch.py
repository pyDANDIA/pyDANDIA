from os import path, remove
from pyDANDIA import crossmatch
from pyDANDIA import crossmatch_datasets
from pyDANDIA import match_utils
from pyDANDIA import metadata
from pyDANDIA import logs
from astropy.table import Table
import numpy as np

def test_params():
    params = {'primary_ref': 'primary_ref_dataset',
              'datasets': { 'primary_ref_dataset': ['primary_ref', '/Users/rstreet1/OMEGA/test_data/non_ref_dataset_p/', 'none'],
                            'dataset0' : [ 'non_ref', '/Users/rstreet1/OMEGA/test_data/non_ref_dataset0/', 'none' ],
                            'dataset1' : [ 'non_ref', '/Users/rstreet1/OMEGA/test_data/non_ref_dataset1/', 'none' ],
                            'dataset2' : [ 'non_ref', '/Users/rstreet1/OMEGA/test_data/non_ref_dataset1/', 'none' ]},
              'file_path': 'crossmatch_table.fits',
              'log_dir': '.'}

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
    test_init_stars_table()
