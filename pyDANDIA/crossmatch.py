from astropy.io import fits
from astropy.table import Table
from astropy.table import Column
from pyDANDIA import match_utils

class CrossMatchTable():
    """Class describing the cross-identification of stars between multiple
    datasets"""

    def __init__(self):
        self.header = None
        self.matched_stars = []

    def create(self, params):
        xmatch = fits.HDUList()

        header = fits.Header()
        header.update({'PRIMARY': params['primary_ref_dir']})
        header.update({'PRIMFILT': params['primary_ref_filter']})
        for i,red_dir in enumerate(params['red_dir_list']):
            header.update({'DATASET'+str(i): red_dir})
            header.update({'FILTER'+str(i): params['red_dataset_filters'][i]})
            self.init_matched_stars_table()

        self.header = header

    def add_dataset(self, red_dir, filter_name):
        idx = len(self.matched_stars)
        self.header.update({'DATASET'+str(idx): red_dir})
        self.header.update({'FILTER'+str(idx): filter_name})
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
        for key, value in self.header:
            if 'DATASET' in key and value == red_dir:
                idx = int(key.replace('DATASET',''))

        return idx

    def update_matched_stars_table(self, dataset_idx, matched_stars):
        self.matched_stars[dataset_idx] = Table( [ Column(name='dataset_star_id', data = np.array(matched_stars.cat2_index), dtype='int'),
                              Column(name='dataset_ra', data = np.array(matched_stars.cat2_ra), dtype='float'),
                              Column(name='dataset_dec', data = np.array(matched_stars.cat2_dec), dtype='float'),
                              Column(name='dataset_x', data = np.array(matched_stars.cat2_x), dtype='float'),
                              Column(name='dataset_y', data = np.array(matched_stars.cat2_y), dtype='float'),
                              Column(name='field_star_id', data = np.array(matched_stars.cat1_index), dtype='int'),
                              Column(name='field_ra', data = np.array(matched_stars.cat1_ra), dtype='float'),
                              Column(name='field_dec', data = np.array(matched_stars.cat1_dec), dtype='float'),
                              Column(name='field_x', data = np.array(matched_stars.cat1_x), dtype='float'),
                              Column(name='field_y', data = np.array(matched_stars.cat1_y), dtype='float'),
                              Column(name='separation', data = np.array(matched_stars.separation), dtype='float') ] )

    def save(self, file_path):
        """Output crossmatch table to file"""

        hdu_list = [fits.PrimaryHDU(self.header)]
        hdu_list.append(fits.BinTableHDU(x, name='MATCH_TABLE_'+str(i)) for i,x in enumerate(self.matched_stars))
        hdu_list = fits.HDUList(hdu_list)
        hdu_list.writeto(file_path, overwrite=True)


    def load_matched_stars(self, dataset_idx):
        """Method to load the matched_stars list"""

        matched_stars = match_utils.StarMatchIndex()

        matched_stars.cat1_index = list(self.matched_stars[dataset_idx]['field_star_id'])
        matched_stars.cat1_ra = list(self.matched_stars[dataset_idx]['field_ra'])
        matched_stars.cat1_dec = list(self.matched_stars[dataset_idx]['field_dec'])
        matched_stars.cat1_x = list(self.matched_stars[dataset_idx]['field_x'])
        matched_stars.cat1_y = list(self.matched_stars[dataset_idx]['field_y'])
        matched_stars.cat2_index = list(self.matched_stars[dataset_idx]['dataset_star_id'])
        matched_stars.cat2_ra = list(self.matched_stars[dataset_idx]['dataset_ra'])
        matched_stars.cat2_dec = list(self.matched_stars[dataset_idx]['dataset_dec'])
        matched_stars.cat2_x = list(self.matched_stars[dataset_idx]['dataset_x'])
        matched_stars.cat2_y = list(self.matched_stars[dataset_idx]['dataset_y'])
        matched_stars.separation = list(self.matched_stars[dataset_idx]['separation'])
        matched_stars.n_match = len(matched_stars.cat1_index)

        return matched_stars

    def load(self,file_path):
        """Method to load an existing crossmatch table"""

        if not path.isfile(file_path):
            raise IOError('Cannot find cross-match table at '+file_path)

        hdu_list = fits.open(file_path, mmap=True)
        self.header = hdu_list.header

        for extn in hdu_list[1:]:
            self.matched_stars.append(self.load_matched_stars(extn))
