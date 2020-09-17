from astropy.io import fits
from astropy.table import Table
from astropy.table import Column
from astropy.coordinates import SkyCoord
from pyDANDIA import match_utils
import numpy as np
from os import path

class CrossMatchTable():
    """Class describing the cross-identification of stars between multiple
    datasets"""

    def __init__(self):
        self.datasets = None
        self.matched_stars = []

    def create(self, params):
        xmatch = fits.HDUList()

        col_list = [ Column(name='primary_ref_dir', data=[params['primary_ref_dir']], dtype='str'),
                     Column(name='primary_ref_filter', data=[params['primary_ref_filter']], dtype='str')]

        for i,red_dir in enumerate(params['red_dir_list']):
            col_list.append( Column(name='dataset'+str(i), data=[red_dir], dtype='str') )
            col_list.append( Column(name='filter'+str(i), data=[params['red_dataset_filters'][i]], dtype='str') )

            self.init_matched_stars_table()

        self.datasets = Table(col_list)

    def add_dataset(self, red_dir, filter_name):
        idx = len(self.matched_stars)

        self.datasets.add_column(Column(name='dataset'+str(idx), data=[red_dir], dtype='str'))
        self.datasets.add_column(Column(name='filter'+str(idx), data=[filter_name], dtype='str'))

        self.init_matched_stars_table()

        return idx

    def init_matched_stars_table(self):
        headers = ['dataset_star_id', 'dataset_ra', 'dataset_dec', 'dataset_x', 'dataset_y',
                    'field_star_id', 'field_ra', 'field_dec', 'field_x', 'field_y',
                    'separation']

        columns = []
        for key in headers:
            if 'id' in key:
                columns.append( Column(name=key, data=[], dtype='int') )
            else:
                columns.append( Column(name=key, data=[], dtype='float') )

        self.matched_stars.append(Table(columns))

    def dataset_index(self, red_dir):
        """Method to search the header index of matched data directories and
        return it's index in the matched_stars table list, or -1 if not present"""

        idx = -1
        for col in self.datasets.colnames:
            if 'dataset' in col and self.datasets[col][0] == red_dir:
                idx = int(col.replace('dataset',''))

        return idx

    def save(self, file_path):
        """Output crossmatch table to file"""
        hdr = fits.Header()
        hdr['NAME'] = 'Crossmatch table'

        hdu_list = [fits.PrimaryHDU(header=hdr), fits.BinTableHDU(self.datasets, name='DATASETS')]
        for i,x in enumerate(self.matched_stars):
            try:
                hdu_list.append( fits.BinTableHDU(x.output_as_table(), name='match_table_'+str(i)) )
            except:
                import pdb; pdb.set_trace()

        hdu_list = fits.HDUList(hdu_list)
        hdu_list.writeto(file_path, overwrite=True)


    def load_matched_stars(self, binary_table):
        """Method to load the matched_stars list"""

        matched_stars = match_utils.StarMatchIndex()

        matched_stars.cat1_index = list(binary_table.data['field_star_id'])
        matched_stars.cat1_ra = list(binary_table.data['field_ra'])
        matched_stars.cat1_dec = list(binary_table.data['field_dec'])
        matched_stars.cat1_x = list(binary_table.data['field_x'])
        matched_stars.cat1_y = list(binary_table.data['field_y'])
        matched_stars.cat2_index = list(binary_table.data['dataset_star_id'])
        matched_stars.cat2_ra = list(binary_table.data['dataset_ra'])
        matched_stars.cat2_dec = list(binary_table.data['dataset_dec'])
        matched_stars.cat2_x = list(binary_table.data['dataset_x'])
        matched_stars.cat2_y = list(binary_table.data['dataset_y'])
        matched_stars.separation = list(binary_table.data['separation'])
        matched_stars.n_match = len(matched_stars.cat1_index)

        return matched_stars

    def load(self,file_path):
        """Method to load an existing crossmatch table"""

        if not path.isfile(file_path):
            raise IOError('Cannot find cross-match table at '+file_path)

        hdu_list = fits.open(file_path, mmap=True)
        table_data = []
        for col in hdu_list[1].columns.names:
            table_data.append( Column(hdu_list[1].data[col], name=col) )
        self.datasets = Table( table_data )

        for table_extn in hdu_list[2:]:
            self.matched_stars.append(self.load_matched_stars(table_extn))

    def fetch_match_table_for_reduction(red_dir):
        """Method to return the matched_stars object corresponding to the
        reduction given"""

        idx = self.dataset_index(red_dir)

        if idx > 0:
            matched_stars = self.matched_stars[idx]
        else:
            matched_stars = match_utils.StarMatchIndex()

        return matched_stars
