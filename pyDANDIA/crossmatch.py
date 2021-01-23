from astropy.io import fits
from astropy.table import Table
from astropy.table import Column
from astropy.coordinates import SkyCoord
from astropy import units
from pyDANDIA import match_utils
import numpy as np
from os import path

class CrossMatchTable():
    """Class describing the cross-identification of stars between multiple
    datasets"""

    def __init__(self):
        self.field_index = Table()
        self.primary_ref_code = None
        self.primary_ref_dir = None
        self.primary_ref_filter = None
        self.datasets = Table()
        self.matched_stars = {}
        self.orphans = {}

    def create(self, params):
        xmatch = fits.HDUList()

        pref = params['primary_ref']
        self.primary_ref_code = pref
        self.primary_ref_dir = params['datasets'][pref][1]
        self.primary_ref_filter = None

        field_index_columns = [Column(name='field_id', data=[], dtype='int'),
                    Column(name='ra', data=[], dtype='float'),
                    Column(name='dec', data=[], dtype='float'),
                    Column(name='gaia_source_id', data=[], dtype='str'),
                    Column(name=pref+'_index', data=[], dtype='int')]

        # Format: dataset_id: [reference status, red_dir path, filter]
        codes = []
        reddirs = []
        filters = []
        pref_index = []
        for dataset_id, dataset_info in params['datasets'].items():
            if dataset_id == pref:
                pref_index.append(1)
            else:
                pref_index.append(0)
            codes.append(dataset_id)
            reddirs.append(dataset_info[1])
            filters.append(dataset_info[2])
            if dataset_id != pref:
                field_index_columns.append(Column(name=dataset_id+'_index', data=[], dtype='int'))

        dataset_columns = [ Column(name='dataset_code', data=codes, dtype='str'),
                     Column(name='dataset_red_dir', data=reddirs, dtype='str'),
                     Column(name='dataset_filter', data=filters, dtype='str'),
                     Column(name='primary_ref',data=pref_index, dtype='int')]

        self.datasets = Table(dataset_columns)
        self.field_index = Table(field_index_columns)

    def add_dataset_header(self, dataset_idx, dataset_code, dataset_info):

        if '/' in dataset_info[0][-1:]:
            red_dir = dataset_info[0][0:-1]
        else:
            red_dir = dataset_info[0]
        self.datasets.add_column(Column(name='dataset_code'+str(dataset_idx), data=[dataset_code], dtype='str'))
        self.datasets.add_column(Column(name='dataset_red_dir'+str(dataset_idx), data=[dataset_info[0]], dtype='str'))
        self.datasets.add_column(Column(name='dataset_filter'+str(dataset_idx), data=[dataset_info[1]], dtype='str'))

    def add_dataset(self, red_dir, filter_name):
        idx = self.count_datasets()
        self.add_dataset_header(idx, red_dir, filter_name)
        self.init_matched_stars_table()

        return idx

    def count_datasets(self):
        n_datasets = 0
        for col in self.datasets.colnames:
            if 'dataset' in col:
                n_datasets += 1
        return n_datasets

    def init_matched_stars_table(self):
        matched_stars = match_utils.StarMatchIndex()
        self.matched_stars.append(matched_stars)

    def init_field_index(self, metadata):

        ncol = len(self.field_index.colnames)

        for star in metadata.star_catalog[1][0:1000]:
            self.field_index.add_row( [star['index'], star['ra'], star['dec'], star['gaia_source_id'], star['index']]+([0]*(ncol-5)) )

    def match_dataset_with_field_index(self, dataset_metadata, params, log,
                                    verbose=True):
        """Method to perform a crossmatch of the star_catalog from a single
        dataset's reduction against the field catalog.
        Note that the separation_threshold parameter is expected to be in
        decimal degrees"""

        log.info('Matching dataset against field index')

        matched_stars = match_utils.StarMatchIndex()
        orphans = match_utils.StarMatchIndex()

        for dataset_star_row in dataset_metadata.star_catalog[1][0:1000]:
            dataset_star = SkyCoord( dataset_star_row['ra'], dataset_star_row['dec'],
                      frame='icrs', unit=(units.deg, units.deg) )
            check_against_full_index = True
            if verbose:
                log.info(' -> Dataset star '+str(dataset_star_row['index'])+', '+repr(dataset_star))

            # Use matching Gaia IDs to accelerate the match process, if available
            if dataset_star_row['gaia_source_id'] != 'none':
                jdx = np.where(self.field_index['gaia_source_id'] == dataset_star_row['gaia_source_id'])[0]

                if verbose:
                    log.info(' --> Gaia ID: '+str(dataset_star_row['gaia_source_id']))
                    log.info(' --> Found '+str(len(jdx))+' matching entries in field index')

                # If a match is found,
                if len(jdx) > 0:
                    field_star_row = self.field_index[jdx]
                    field_star = SkyCoord( field_star_row['ra'], field_star_row['dec'],
                                        frame='icrs', unit=(units.deg, units.deg) )

                    # Check separation is below match threshold
                    (star_matches, separation) = self.check_separation(dataset_star, field_star,
                                                params['separation_threshold'])

                    #print(star_matches, separation)

                    # If so, add to the match index
                    if star_matches:
                        matched_stars = self.add_to_match_index(field_star_row, dataset_star_row,
                                                separation, matched_stars, log,
                                                verbose=verbose)
                    check_against_full_index = False

                else:
                    check_against_full_index = True

            if verbose:
                log.info(' --> Check against full field index? '+str(check_against_full_index))

            # If no Gaia ID is available, or if no matching ID is found,
            # check the star against the rest of the field index:
            if check_against_full_index:
                for field_star_row in self.field_index:
                    #print(field_star_row)
                    field_star = SkyCoord( field_star_row['ra'], field_star_row['dec'],
                                        frame='icrs', unit=(units.deg, units.deg) )

                    (star_matches, separation) = self.check_separation(dataset_star, field_star,
                                                params['separation_threshold'])

                    if star_matches:
                        matched_stars = self.add_to_match_index(field_star_row, dataset_star_row,
                                                separation, matched_stars, log,
                                                verbose=verbose)

                        if verbose:
                            log.info(' ---> Matches field star: '+str(field_star_row['field_id'])+' '+str(field_star_row['ra'])+' '+str(field_star_row['dec']))
                            log.info(' ---> match output: '+repr(star_matches)+' '+repr(separation))

            # Orphans: If we get to here and still have no matching star in
            # the field_index, then this dataset has detected a star not found
            # in the field's primary reference or other datasets.
            orphans = self.add_to_orphans(dataset_star_row, orphans, log, verbose=verbose)

        return matched_stars, orphans

    def check_separation(self,dataset_star, field_star, separation_threshold):
        separation = dataset_star.separation(field_star)

        if separation <= separation_threshold:
            return True, separation
        else:
            return False, separation

    def add_to_match_index(self,field_star, star, separation, matched_stars, log,
                        verbose=False):
        #print('Field: ', field_star)
        #print('Dataset: ', star)
        #print('Separation: ', separation)

        p = {'cat1_index': field_star['field_id'],
             'cat1_ra': field_star['ra'],
             'cat1_dec': field_star['dec'],
             'cat1_x': 0.0,
             'cat1_y': 0.0,
             'cat2_index': star['index'],
             'cat2_ra': star['ra'],
             'cat2_dec': star['dec'],
             'cat2_x': 0.0,
             'cat2_y': 0.0,
             'separation': separation.value}

        matched_stars.add_match(p, log=log, verbose=True)

        if verbose:
            log.info('Linked to known star: '+matched_stars.summarize_last(units='deg'))

        return matched_stars

    def add_to_orphans(self,star, orphans, log, verbose=False):
        p = {'cat1_index': None,
             'cat1_ra': 0.0,
             'cat1_dec': 0.0,
             'cat1_x': 0.0,
             'cat1_y': 0.0,
             'cat2_index': star['index'],
             'cat2_ra': star['ra'],
             'cat2_dec': star['dec'],
             'cat2_x': 0.0,
             'cat2_y': 0.0,
             'separation': -1.0}

        orphans.add_match(p)

        if verbose:
            log.info('Added orphan: '+repr(star['index'])+' '+str(star['ra'])+' '+str(star['dec']))

        return orphans

    def update_field_index(self, dataset_code, matched_stars, orphans, dataset_metadata, log):

        # Update field index with matched stars
        for j in range(0,len(matched_stars.cat1_index),1):
            jfield = np.where(self.field_index['field_id'] == matched_stars.cat1_index[j])
            row = self.field_index[jfield]
            row[dataset_code+'_index'] = matched_stars.cat2_index[j]
            self.field_index[jfield] = row

        # Append orphans to the end of the field index
        ncol = len(self.field_index.colnames)
        dataset_col = self.field_index.colnames.index(dataset_code+'_index')
        for j in range(0,len(orphans.cat2_index),1):
            jfield = len(self.field_index) + 1
            gaia_id =  dataset_metadata.star_catalog[1][int(orphans.cat2_index[j])]['gaia_source_id']
            row = [jfield, orphans.cat2_ra[j], orphans.cat2_dec[j], gaia_id] + [0]*(ncol-4)
            row[dataset_col] = orphans.cat2_index[j]
            self.field_index.add_row(row)

    def dataset_index(self, red_dir):
        """Method to search the header index of matched data directories and
        return it's index in the matched_stars table list, or -1 if not present"""

        if '/' in red_dir[-1:]:
            red_dir = red_dir[0:-1]

        idx = -1
        for col in self.datasets.colnames:
            #import pdb; pdb.set_trace()
            if 'dataset' in col and self.datasets[col][0] == red_dir:
                idx = int(col.replace('dataset',''))

        return idx

    def save(self, file_path):
        """Output crossmatch table to file"""
        hdr = fits.Header()
        hdr['NAME'] = 'Crossmatch table'
        hdr['PRIREFID'] = self.primary_ref_code

        hdu_list = [fits.PrimaryHDU(header=hdr), fits.BinTableHDU(self.field_index, name='FIELD_INDEX'), fits.BinTableHDU(self.datasets, name='DATASETS')]
#        for i,x in enumerate(self.matched_stars):
#            try:
#                hdu_list.append( fits.BinTableHDU(x.output_as_table(), name='match_table_'+str(i)) )
#            except:
#                import pdb; pdb.set_trace()

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

    def fetch_match_table_for_reduction(self,red_dir):
        """Method to return the matched_stars object corresponding to the
        reduction given"""

        idx = self.dataset_index(red_dir)

        if idx >= 0:
            matched_stars = self.matched_stars[idx]
        else:
            matched_stars = match_utils.StarMatchIndex()

        return matched_stars
