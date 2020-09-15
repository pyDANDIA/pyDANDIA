from os import path
from sys import argv
from pyDANDIA import crossmatch

def build_crossmatch_table(params):

    xmatch = crossmatch.CrossMatchTable()

    if path.isfile(params['file_path']):
        xmatch.load(params['file_path'])
    else:
        xmatch.create(params)

    
def get_args():
    params = {}

    if len(argv) < 5:
        params['primary_ref_dir'] = input('Please enter the path to the primary reference dataset: ')
        params['primary_ref_filter'] = input('Please enter the filter name used for the primary reference dataset: ')
        params['red_dir_list'] = [input('Please enter the path to the dataset to cross-match: ')]
        params['red_dataset_filters'] = [input('Please enter filter name used for the dataset: ')]
        params['file_path'] = input('Please enter the path to the crossmatch table: ')
    else:
        params['primary_ref_dir'] = argv[1]
        params['primary_ref_filter'] = argv[2]
        params['red_dir_list'] = [argv[2]]
        params['red_dataset_filters'] = [argv[4]]
        params['file_path'] = argv[5]

    return params


if __name__ == '__main__':
    params = get_args()
    build_crossmatch_table(params)
