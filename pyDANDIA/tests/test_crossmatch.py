from pyDANDIA import crossmatch
from pyDANDIA import crossmatch_datasets

def test_create():
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

if __name__ == '__main__':
    test_create()
    test_add_dataset()
    
