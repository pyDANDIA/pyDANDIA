from astropy.io import fits
from astropy.table import Table
from astropy.table import Column
from astropy.coordinates import SkyCoord
from astropy import units
from pyDANDIA import match_utils
import numpy as np
from os import path
import copy

class CrossMatchTable():
    """Class describing the cross-identification of stars between multiple
    datasets"""

    def __init__(self):
        self.field_index = Table()
        self.primary_ref_code = None
        self.primary_ref_dir = None
        self.primary_ref_filter = None
        self.gaia_dr = None
        self.datasets = Table()

    def create(self, params):
        xmatch = fits.HDUList()

        pref = params['primary_ref']
        self.primary_ref_code = pref
        self.primary_ref_dir = params['datasets'][pref][1]
        self.primary_ref_filter = None

        field_index_columns = [Column(name='field_id', data=[], dtype='int'),
                    Column(name='ra', data=[], dtype='float'),
                    Column(name='dec', data=[], dtype='float'),
                    Column(name='gaia_source_id', data=[], dtype='S19'),
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
                     Column(name='dataset_filter', data=filters, dtype='S8'),
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

    def init_field_index(self, metadata, filter_id):

        self.datasets[0]['dataset_filter'] = filter_id

        ncol = len(self.field_index.colnames)

        for star in metadata.star_catalog[1]:
            self.field_index.add_row( [star['index'], star['ra'], star['dec'], star['gaia_source_id'], star['index']]+([0]*(ncol-5)) )

    def match_dataset_with_field_index(self, dataset_metadata, params, log,
                                    verbose=True):
        """Method to perform a crossmatch of the star_catalog from a single
        dataset's reduction against the field catalog.
        Note that the separation_threshold parameter is expected to be in
        decimal degrees

        During the matching process, new potential matches are compared
        against the list of existing known matches involving the same stars
        to check for duplications and to replace existing matches if a closer
        match is found.

        This process can result in a star that was previously matched to a
        catalogue object being removed from the matched_stars list, if a
        closer match is found.  For this reason, the match process is repeated,
        so that alternative matches can be sought for those stars.
        """

        log.info('Matching dataset against field index')

        matched_stars = match_utils.StarMatchIndex()
        orphans = match_utils.StarMatchIndex()

        # Cross-match all entries in the field catalog with those stars
        # detected in the current dataset:
        dataset_stars = SkyCoord( dataset_metadata.star_catalog[1]['ra'],
                                    dataset_metadata.star_catalog[1]['dec'],
                                    frame='icrs', unit=(units.deg, units.deg) )

        field_stars = SkyCoord(self.field_index['ra'], self.field_index['dec'],
                            frame='icrs', unit=(units.deg, units.deg) )

        (field_idx, separations2D, separations3D) = dataset_stars.match_to_catalog_3d(field_stars)

        # Add stars with matches within the separation_threshold to the
        # matched_stars index
        constraint = separations2D < params['separation_threshold']
        matching_dataset_index = np.arange(0,len(dataset_metadata.star_catalog[1]),1)[constraint]
        matching_field_index = field_idx[constraint]

        for j,jdataset in enumerate(matching_dataset_index):
            jfield = matching_field_index[j]
            field_star_row = self.field_index[jfield]
            dataset_star_row = dataset_metadata.star_catalog[1][jdataset]
            (star_added,matched_stars) = self.add_to_match_index(field_star_row, dataset_star_row,
                                    separations2D[jdataset], matched_stars, log,
                                    verbose=verbose)

        # Add all other stars to the orphan's list:
        unmatched_dataset_index = np.arange(0,len(dataset_metadata.star_catalog[1]),1)
        matched_star_indices = np.array(matched_stars.cat2_index) - 1
        unmatched_dataset_index = np.delete(unmatched_dataset_index,matched_star_indices)
        for jdataset in unmatched_dataset_index:
            dataset_star_row = dataset_metadata.star_catalog[1][jdataset]
            (star_added, orphans) = self.add_to_orphans(dataset_star_row, orphans,
                                                    log, verbose=verbose)

        return matched_stars, orphans

    def match_field_index_with_gaia_catalog(self, gaia_data, params, log):
        log.info('Matching field index against Gaia data')

        matched_stars = match_utils.StarMatchIndex()

        field_stars = SkyCoord(self.field_index['ra'], self.field_index['dec'],
                            frame='icrs', unit=(units.deg, units.deg) )
        gaia_stars = SkyCoord(gaia_data['ra'], gaia_data['dec'],
                            frame='icrs', unit=(units.deg, units.deg) )

        (field_idx, separations2D, separations3D) = dataset_stars.match_to_catalog_3d(field_stars)

        constraint = separations2D < params['separation_threshold']
        matching_gaia_index = np.arange(0,len(gaia_data),1)[constraint]
        matching_field_index = field_idx[constraint]

        for j,jgaia in enumerate(matching_gaia_index):
            jfield = matching_field_index[j]
            p = {'cat1_index': jfield,          # Index NOT target ID
                 'cat1_ra': field_stars['ra'][jfield],
                 'cat1_dec': field_stars['dec'][jfield],
                 'cat1_x': 0.0,
                 'cat1_y': 0.0,
                 'cat2_index': int(gaia_data['source_id'][jgaia]),  # Source ID
                 'cat2_ra': gaia_data['ra'][jgaia],
                 'cat2_dec': gaia_data['dec'][jgaia],
                 'cat2_x': 0.0,
                 'cat2_y': 0.0,
                 'separation': separation.value}

            star_added = matched_stars.add_match(p, log=log, verbose=True)

        for j in range(0,matched_stars.n_match,1):
            self.field_index['gaia_source_id'][matched_stars.cat1_index[j]] = str(matched_stars.cat2_index[j])

        log.info('Matched '+str(matched_stars.n_match)+' Gaia targets to stars in field index')

        self.gaia_dr = params['gaia_dr']

    def check_separation(self,dataset_star, field_star, separation_threshold,log):
        separation = dataset_star.separation(field_star)

        if separation <= separation_threshold:
            return True, separation
        else:
            return False, separation

    def add_to_match_index(self,field_star, star, separation, matched_stars, log,
                        verbose=False):
        """Expects field_star and star to be Table rows"""
        #print('Field: ', field_star['field_id'])
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

        star_added = matched_stars.add_match(p, log=log, verbose=True)

        return star_added, matched_stars

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

        # Append the orphan without checking for duplication, since no cat1
        # index is given
        star_added = orphans.add_match(p,log=log,verbose=True,replace_worse_matches=False)

        return star_added, orphans

    def update_field_index(self, dataset_code, matched_stars, orphans, dataset_metadata, log):

        # Update field index with matched stars
        log.info('Updating field index with matched stars:')
        for j in range(0,len(matched_stars.cat1_index),1):
            jfield = np.where(self.field_index['field_id'] == matched_stars.cat1_index[j])
            row = self.field_index[jfield]
            row[dataset_code+'_index'] = matched_stars.cat2_index[j]
            self.field_index[jfield] = row
            log.info(repr(row))

        # Append orphans to the end of the field index
        log.info('Updating field index with orphans:')
        log.info(repr(orphans.cat2_index))
        ncol = len(self.field_index.colnames)
        dataset_col = self.field_index.colnames.index(dataset_code+'_index')
        for j in range(0,len(orphans.cat2_index),1):
            jfield = len(self.field_index) + 1
            jdataset = orphans.cat2_index[j]
            gaia_id =  dataset_metadata.star_catalog[1][int(jdataset)]['gaia_source_id']
            row = [jfield, orphans.cat2_ra[j], orphans.cat2_dec[j], gaia_id] + [0]*(ncol-4)
            row[dataset_col] = orphans.cat2_index[j]
            self.field_index.add_row(row)
            log.info(repr(row)+' dataset star '+str(j))

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
        hdr['GAIA_DR'] = self.gaia_dr

        hdu_list = [fits.PrimaryHDU(header=hdr),
                    fits.BinTableHDU(self.field_index, name='FIELD_INDEX'),
                    fits.BinTableHDU(self.datasets, name='DATASETS')]
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

    def load(self,file_path, log=None):
        """Method to load an existing crossmatch table"""

        if not path.isfile(file_path):
            raise IOError('Cannot find cross-match table at '+file_path)

        hdu_list = fits.open(file_path, mmap=True)
        table_data = []
        for col in hdu_list[0].columns.names:
            table_data.append( Column(hdu_list[1].data[col], name=col) )
        self.field_index = Table( table_data )

        for col in hdu_list[1].columns.names:
            table_data.append( Column(hdu_list[1].data[col], name=col) )
        self.datasets = Table( table_data )

        if log:
            log.info('Loaded crossmatch table from '+file_path)
